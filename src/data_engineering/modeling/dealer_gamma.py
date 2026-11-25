"""Dealer Gamma (DG) — does option-dealer hedging leave a footprint in BTC/ETH on Deribit?

Pre-registration: discussions/Dealer_Gamma/001_claude_branch_preregistration.md

Data (all already on disk, nothing new is collected):
    history trades   <DERIBIT_DATA_ROOT>/history/trades/<CCY>/{option,future}/<YYYY>/<YYYY-MM-DD>.parquet
                     every print with the TAKER side ('direction'), iv, index_price, block/combo ids
    collector grid   <DERIBIT_DATA_ROOT>/daily/option_grid{,_1dte}/expiry=YYYY-MM-DD/option_grid.parquet
                     5-minute open interest and 24h volume (0DTE since 2026-09-19, 1DTE since 2026-10-06)
    DVOL             <DERIBIT_DATA_ROOT>/history/dvol/<CCY>_dvol_1h.csv

Positioning model
    customer = taker of each option print; net customer position per instrument = cumulative signed amount.
    dealer position = - net customer position. Dealer dollar gamma per 1% move:
        G(t) = sum_i dealer_i * gamma_BS(S, K_i, iv_i, T_i) * S^2 * 0.01        [USD per 1% move]
    G > 0: dealers long gamma (they sell rallies / buy dips). G < 0: dealers short gamma (they chase moves).
    gamma uses the instrument's last traded iv and the Deribit index (r = q = 0).

Stages (notebooks/Dealer_Gamma/):
    01  data gate      reconstruction vs the collector's open interest and 24h volume
    02  build          5-minute GEX grid, perpetual bars, expiry strike tables, settlement proxy (cached)
    03  regime         GEX at 08:05 UTC -> next-session realized vol and 5-minute autocorrelation
    04  hedge flow     predicted hedge (-100 * G * r) -> signed perpetual taker flow over the next 15 minutes
    05  pinning        settlement proxy vs the dealer-long-gamma strike, against a mirrored placebo strike
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "DG-0.1.0"

CURRENCIES = ("BTC", "ETH")
PERP = {"BTC": "BTC-PERPETUAL", "ETH": "ETH-PERPETUAL"}

DATA_START = "2024-01-02"
CLEAN_FROM = "2024-01-16"       # instruments first seen before this may carry an unseen opening position
DISCOVERY = ("2024-07-01", "2025-06-30")
HOLDOUT_START = "2025-07-01"

GRID_MS = 300_000
DAY_MS = 86_400_000
YEAR_MS = 365.25 * DAY_MS
BARS = DAY_MS // GRID_MS        # 288
EXPIRY_HOUR = 8
PIN_REF_HOUR = 5                # strike tables taken at 05:00 UTC, 3 h before the 08:00 expiry
PIN_BAND = 0.05                 # candidate pin strikes within +-5% of S at 05:00
SETTLE_FROM_MIN = 7 * 60 + 30   # Deribit delivery = 30-minute index average before 08:00
EXCLUDE_EXPIRY_WINDOW = (7 * 60 + 50, 8 * 60 + 15)   # minutes of day excluded in the hedge-flow test
ADV_SESSIONS = 30
FLOW_H = (1, 3)                 # bars ahead (primary = 3, i.e. 15 minutes)
PRIMARY_H = 3
MIN_SESSION_BARS = 250

VARIANTS = ("recon", "recon_noblock", "recon_clean", "short_all", "call_long_put_short")
PRIMARY = "recon"

OPT_RE = re.compile(r"^(?P<u>[A-Z]+)-(?P<d>\d{1,2})(?P<m>[A-Z]{3})(?P<y>\d{2})-(?P<k>[\d.d]+)-(?P<cp>[CP])$")
MONTHS = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV",
                                       "DEC"], start=1)}


@dataclass(frozen=True)
class Config:
    data_start: str = DATA_START
    clean_from: str = CLEAN_FROM
    discovery: tuple = DISCOVERY
    holdout_start: str = HOLDOUT_START
    pin_ref_hour: int = PIN_REF_HOUR
    pin_band: float = PIN_BAND
    adv_sessions: int = ADV_SESSIONS
    primary_h: int = PRIMARY_H
    primary_variant: str = PRIMARY
    hac_lags: int = 5
    gate_volume_band: tuple = (0.95, 1.05)
    gate_max_violation_share: float = 0.05
    alpha_discovery: float = 0.05
    alpha_holdout: float = 0.10


def registration(cfg: Config = Config()) -> dict:
    return {"version": VERSION, **asdict(cfg), "variants": list(VARIANTS)}


def ms(x) -> int:
    t = pd.Timestamp(x)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return int(t.value // 10**6)


def day_list(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D")]


def period_of(dates) -> np.ndarray:
    d = pd.to_datetime(pd.Series(dates)).dt.strftime("%Y-%m-%d").to_numpy()
    return np.where(d < DISCOVERY[0], "burn_in", np.where(d <= DISCOVERY[1], "discovery", "holdout"))


# ----------------------------------------------------------------------------- Black-Scholes gamma
def bs_gamma(S, K, sigma, T) -> np.ndarray:
    """Gamma per unit of underlying (r = q = 0). sigma annualised (decimal), T in years."""
    S = np.asarray(S, float); K = np.asarray(K, float); sigma = np.asarray(sigma, float); T = np.asarray(T, float)
    sq = np.maximum(sigma * np.sqrt(np.maximum(T, 0.0)), 1e-4)
    d1 = (np.log(S / K) + 0.5 * sq * sq) / sq
    return np.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) / (S * sq)


def bs_call_delta(S, K, sigma, T) -> np.ndarray:
    from scipy.stats import norm
    sq = np.maximum(sigma * np.sqrt(np.maximum(T, 0.0)), 1e-4)
    return norm.cdf((np.log(S / K) + 0.5 * sq * sq) / sq)


# ----------------------------------------------------------------------------- loaders
def _ts_ms(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s).to_numpy("int64")
    return pd.to_datetime(s, utc=True).dt.as_unit("ms").astype("int64").to_numpy()


def _read(path: Path, want: list[str]) -> pd.DataFrame | None:
    if not Path(path).is_file():
        return None
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    cols = [c for c in want if c in pf.schema.names]
    return pf.read(columns=cols).to_pandas()


def _flag(s: pd.Series | None, n: int) -> np.ndarray:
    if s is None:
        return np.zeros(n, bool)
    t = s.astype("string")
    return (t.notna() & t.ne("") & t.str.lower().ne("none") & t.str.lower().ne("nan")).fillna(False).to_numpy(bool)


def trades_path(history: Path, ccy: str, kind: str, day: str) -> Path:
    return Path(history) / "trades" / ccy / kind / day[:4] / f"{day}.parquet"


def load_options(history: Path, ccy: str, day: str) -> pd.DataFrame | None:
    """Option prints of one UTC day, sorted by time. sgn = +1 taker buy, -1 taker sell; amount in coin."""
    df = _read(trades_path(history, ccy, "option", day),
               ["timestamp", "trade_seq", "instrument_name", "direction", "amount", "iv", "index_price",
                "block_trade_id", "combo_id", "expiry", "strike", "option_type"])
    if df is None:
        return None
    n = len(df)
    out = pd.DataFrame({"ts": _ts_ms(df["timestamp"]), "name": df["instrument_name"].astype(str).to_numpy(),
                        "amt": pd.to_numeric(df["amount"], errors="coerce").to_numpy(float),
                        "sgn": np.where(df["direction"].astype(str).str.lower().eq("buy"), 1.0, -1.0),
                        "iv": pd.to_numeric(df.get("iv"), errors="coerce").to_numpy(float) if "iv" in df else np.nan,
                        "idx": pd.to_numeric(df.get("index_price"), errors="coerce").to_numpy(float)
                        if "index_price" in df else np.nan,
                        "blk": _flag(df.get("block_trade_id"), n) | _flag(df.get("combo_id"), n)})
    meta = parse_names(out["name"])
    out = pd.concat([out, meta], axis=1)
    out = out[np.isfinite(out.amt) & (out.amt > 0) & out.expiry_ms.notna()]
    order = ["ts"] + (["trade_seq"] if "trade_seq" in df else [])
    if "trade_seq" in df:
        out["trade_seq"] = pd.to_numeric(df.loc[out.index, "trade_seq"], errors="coerce")
    return out.sort_values(order, kind="stable").reset_index(drop=True)


def parse_names(names: pd.Series) -> pd.DataFrame:
    u = names.astype(str).unique()
    rec = {}
    for n in u:
        m = OPT_RE.match(n)
        if m:
            rec[n] = (int(pd.Timestamp(2000 + int(m["y"]), MONTHS[m["m"]], int(m["d"]), EXPIRY_HOUR, tz="UTC").value
                          // 10**6), float(m["k"].replace("d", ".")), m["cp"] == "C")
        else:
            rec[n] = (np.nan, np.nan, False)
    t = pd.DataFrame.from_dict(rec, orient="index", columns=["expiry_ms", "strike", "call"])
    return t.reindex(names.astype(str).to_numpy()).reset_index(drop=True)


def load_perp(history: Path, ccy: str, day: str) -> pd.DataFrame | None:
    """Perpetual prints of one UTC day: price, USD amount, taker sign, index price."""
    df = _read(trades_path(history, ccy, "future", day),
               ["timestamp", "trade_seq", "instrument_name", "direction", "price", "amount", "index_price"])
    if df is None:
        return None
    df = df[df["instrument_name"].astype(str).eq(PERP[ccy])]
    out = pd.DataFrame({"ts": _ts_ms(df["timestamp"]),
                        "price": pd.to_numeric(df["price"], errors="coerce").to_numpy(float),
                        "usd": pd.to_numeric(df["amount"], errors="coerce").to_numpy(float),
                        "sgn": np.where(df["direction"].astype(str).str.lower().eq("buy"), 1.0, -1.0),
                        "idx": pd.to_numeric(df["index_price"], errors="coerce").to_numpy(float)})
    return out.sort_values("ts", kind="stable").reset_index(drop=True)


# ----------------------------------------------------------------------------- position book
class Book:
    """Net customer position per option instrument, accumulated from signed prints."""

    FIELDS = (("strike", float), ("expiry", np.int64), ("call", bool), ("first", np.int64), ("net", float),
              ("net_nb", float), ("gross", float), ("iv", float))

    def __init__(self, clean_from_ms: int, cap: int = 4096):
        self.clean_from_ms = clean_from_ms
        self.ix: dict[str, int] = {}
        self.n = 0
        self.a = {f: np.zeros(cap, t) for f, t in self.FIELDS}
        self.a["iv"][:] = np.nan

    def _grow(self, need: int):
        cap = len(self.a["net"])
        if need <= cap:
            return
        new = max(need, 2 * cap)
        for f, t in self.FIELDS:
            b = np.zeros(new, t)
            if f == "iv":
                b[:] = np.nan
            b[:cap] = self.a[f]
            self.a[f] = b

    def index(self, trades: pd.DataFrame) -> np.ndarray:
        names = trades["name"].to_numpy()
        new = [n for n in pd.unique(names) if n not in self.ix]
        if new:
            first = trades.groupby("name", sort=False).agg(ts=("ts", "min"), k=("strike", "first"),
                                                           e=("expiry_ms", "first"), c=("call", "first"))
            self._grow(self.n + len(new))
            for n in new:
                i = self.n; self.ix[n] = i; self.n += 1
                r = first.loc[n]
                self.a["strike"][i] = r.k; self.a["expiry"][i] = int(r.e); self.a["call"][i] = bool(r.c)
                self.a["first"][i] = int(r.ts)
        return np.fromiter((self.ix[n] for n in names), np.int64, len(names))

    def apply(self, ii: np.ndarray, t: pd.DataFrame):
        if len(ii) == 0:
            return
        sa = t["sgn"].to_numpy() * t["amt"].to_numpy()
        np.add.at(self.a["net"], ii, sa)
        nb = ~t["blk"].to_numpy(bool)
        np.add.at(self.a["net_nb"], ii[nb], sa[nb])
        np.add.at(self.a["gross"], ii, t["amt"].to_numpy())
        iv = t["iv"].to_numpy(float)
        ok = np.isfinite(iv) & (iv > 0)
        if ok.any():
            r = ii[ok][::-1]; v = iv[ok][::-1]
            u, first = np.unique(r, return_index=True)
            self.a["iv"][u] = v[first] / 100.0

    def candidates(self, after_ms: int) -> np.ndarray:
        return np.flatnonzero(self.a["expiry"][: self.n] > after_ms)

    def gamma_usd(self, cand: np.ndarray, t_ms: int, S: float):
        """(live index subset, USD gamma per 1% per coin) for instruments alive at t_ms with a position."""
        a = self.a
        live = cand[(a["expiry"][cand] > t_ms) & np.isfinite(a["iv"][cand]) &
                    ((a["net"][cand] != 0) | (a["net_nb"][cand] != 0))]
        if len(live) == 0 or not np.isfinite(S):
            return live[:0], np.zeros(0)
        T = (a["expiry"][live] - t_ms) / YEAR_MS
        g = bs_gamma(S, a["strike"][live], a["iv"][live], T) * S * S * 0.01
        return live, g

    def gex(self, live: np.ndarray, g: np.ndarray, t_ms: int) -> dict:
        a = self.a
        if len(live) == 0:
            return {**{f"gex_{v}": 0.0 for v in VARIANTS}, "gex_recon_1d": 0.0, "n_live": 0}
        net = a["net"][live]; nb = a["net_nb"][live]; absn = np.abs(net)
        clean = a["first"][live] >= self.clean_from_ms
        near = (a["expiry"][live] - t_ms) <= DAY_MS
        return {"gex_recon": float(np.dot(-net, g)),
                "gex_recon_noblock": float(np.dot(-nb, g)),
                "gex_recon_clean": float(np.dot(-net * clean, g)),
                "gex_short_all": float(np.dot(-absn, g)),
                "gex_call_long_put_short": float(np.dot(np.where(a["call"][live], absn, -absn), g)),
                "gex_recon_1d": float(np.dot(-net * near, g)),
                "n_live": int(len(live))}


# ----------------------------------------------------------------------------- build (stage 02)
def _perp_bars(perp: pd.DataFrame | None, day_ms: int) -> pd.DataFrame:
    idx = pd.RangeIndex(1, BARS + 1, name="k")
    if perp is None or perp.empty:
        return pd.DataFrame(index=idx, columns=["perp_last", "buy_usd", "sell_usd", "n_perp", "idx_last"], dtype=float)
    k = ((perp.ts.to_numpy() - day_ms) // GRID_MS + 1).clip(1, BARS)
    z = perp.assign(k=k, buy=np.where(perp.sgn > 0, perp.usd, 0.0), sell=np.where(perp.sgn < 0, perp.usd, 0.0))
    g = z.groupby("k")
    out = pd.DataFrame({"perp_last": g.price.last(), "buy_usd": g.buy.sum(), "sell_usd": g.sell.sum(),
                        "n_perp": g.price.size().astype(float), "idx_last": g.idx.last()})
    out = out.reindex(idx)
    out[["buy_usd", "sell_usd", "n_perp"]] = out[["buy_usd", "sell_usd", "n_perp"]].fillna(0.0)
    return out


def _settle_proxy(perp: pd.DataFrame | None, opt: pd.DataFrame | None, day_ms: int) -> float:
    a, b = day_ms + SETTLE_FROM_MIN * 60_000, day_ms + EXPIRY_HOUR * 3_600_000
    for src in (perp, opt):
        if src is None or src.empty:
            continue
        w = src[(src.ts >= a) & (src.ts < b) & np.isfinite(src.idx)]
        if len(w):
            return float(w.groupby((w.ts - a) // 60_000).idx.mean().mean())
    return float("nan")


def build(history: Path, ccy: str, start: str, end: str, cfg: Config = Config(), log=None):
    """Sequential pass over every day. Returns (grid 5m, pins, settle, coverage) DataFrames."""
    book = Book(ms(cfg.clean_from))
    grid_parts, pins, settles, cov = [], [], [], []
    S_carry = float("nan")
    days = day_list(start, end)
    for di, day in enumerate(days):
        d0 = ms(day)
        opt = load_options(history, ccy, day)
        perp = load_perp(history, ccy, day)
        cov.append({"currency": ccy, "day": day, "option_file": opt is not None, "perp_file": perp is not None,
                    "option_prints": 0 if opt is None else len(opt), "perp_prints": 0 if perp is None else len(perp),
                    "option_block_share": float("nan") if opt is None or opt.empty else float(opt.blk.mean())})
        bars = _perp_bars(perp, d0)
        if opt is not None and len(opt):
            ii = book.index(opt)
            cuts = np.searchsorted(opt.ts.to_numpy(), d0 + GRID_MS * np.arange(1, BARS + 1), side="left")
        else:
            ii = np.zeros(0, np.int64); cuts = np.zeros(BARS, np.int64)
        cand = book.candidates(d0)
        opt_idx_last = None
        if opt is not None and len(opt):
            ob = opt.assign(k=((opt.ts - d0) // GRID_MS + 1).clip(1, BARS))
            opt_idx_last = ob[np.isfinite(ob.idx)].groupby("k").idx.last()
        rows = np.full((BARS, 8), np.nan); nlive = np.zeros(BARS, np.int64)
        lo = 0
        for k in range(1, BARS + 1):
            hi = int(cuts[k - 1])
            if hi > lo:
                book.apply(ii[lo:hi], opt.iloc[lo:hi])
            lo = hi
            s = bars.idx_last.iat[k - 1]
            if not np.isfinite(s) and opt_idx_last is not None and k in opt_idx_last.index:
                s = opt_idx_last.loc[k]
            if np.isfinite(s):
                S_carry = float(s)
            t = d0 + k * GRID_MS
            live, g = book.gamma_usd(cand, t, S_carry)
            r = book.gex(live, g, t)
            rows[k - 1, :6] = [r[f"gex_{v}"] for v in VARIANTS] + [r["gex_recon_1d"]]
            rows[k - 1, 6] = S_carry
            nlive[k - 1] = r["n_live"]
            if k == cfg.pin_ref_hour * 12 and len(live):
                exp = d0 + EXPIRY_HOUR * 3_600_000
                m = book.a["expiry"][live] == exp
                if m.any():
                    L = live[m]
                    pins.append(pd.DataFrame({"currency": ccy, "expiry_date": day, "S_ref": S_carry,
                                              "strike": book.a["strike"][L], "call": book.a["call"][L],
                                              "dg_recon": -book.a["net"][L] * g[m],
                                              "dg_noblock": -book.a["net_nb"][L] * g[m],
                                              "abs_pos": np.abs(book.a["net"][L]), "gross": book.a["gross"][L]}))
        part = pd.DataFrame(rows[:, :7], columns=[f"gex_{v}" for v in VARIANTS] + ["gex_recon_1d", "S"])
        part.insert(0, "ts", pd.to_datetime(d0 + GRID_MS * np.arange(1, BARS + 1), unit="ms", utc=True))
        part["n_live"] = nlive
        for c in ("perp_last", "buy_usd", "sell_usd", "n_perp"):
            part[c] = bars[c].to_numpy()
        part["option_file"] = opt is not None
        grid_parts.append(part)
        settles.append({"currency": ccy, "expiry_date": day, "settle_proxy": _settle_proxy(perp, opt, d0)})
        if log and (di % 30 == 0 or di == len(days) - 1):
            log(f"{ccy} {day}  instruments {book.n}  live {nlive[-1]}  gex_recon {rows[-1, 0]:.3g}")
    grid = pd.concat(grid_parts, ignore_index=True)
    grid.insert(1, "currency", ccy)
    pins = pd.concat(pins, ignore_index=True) if pins else pd.DataFrame()
    return grid, pins, pd.DataFrame(settles), pd.DataFrame(cov)


# ----------------------------------------------------------------------------- stage 01: data gate
def load_collector(data_root: Path, ccy: str) -> pd.DataFrame:
    parts = []
    for sub in ("option_grid", "option_grid_1dte"):
        for f in sorted((Path(data_root) / "daily" / sub).glob("expiry=*/option_grid.parquet")):
            df = _read(f, ["snapshot_time", "instrument_name", "api_currency", "underlying", "open_interest",
                           "volume_24h", "volume", "expiration_timestamp"])
            if df is None or df.empty:
                continue
            cur = df["api_currency"] if "api_currency" in df else df.get("underlying")
            df = df[cur.astype(str).eq(ccy) & df["instrument_name"].astype(str).str.match(OPT_RE.pattern)]
            if len(df):
                parts.append(df.assign(grid=sub))
    if not parts:
        return pd.DataFrame(columns=["t", "name", "oi", "vol24", "grid"])
    c = pd.concat(parts, ignore_index=True)
    vcol = "volume_24h" if "volume_24h" in c else ("volume" if "volume" in c else None)   # collector: stats.volume
    out = pd.DataFrame({"t": _ts_ms(c["snapshot_time"]), "name": c["instrument_name"].astype(str),
                        "oi": pd.to_numeric(c["open_interest"], errors="coerce"),
                        "vol24": pd.to_numeric(c[vcol], errors="coerce") if vcol else np.nan, "grid": c["grid"]})
    return out.dropna(subset=["oi"]).drop_duplicates(["t", "name"]).reset_index(drop=True)


def gate_compare(history: Path, coll: pd.DataFrame, ccy: str, lookback_days: int = 10) -> pd.DataFrame:
    """Per collector row: reconstructed net position and trailing-24h traded amount at the snapshot time."""
    if coll.empty:
        return pd.DataFrame()
    first = pd.to_datetime(coll.t.min(), unit="ms", utc=True).normalize() - pd.Timedelta(days=lookback_days)
    last = pd.to_datetime(coll.t.max(), unit="ms", utc=True).normalize()
    names = set(coll.name.unique())
    parts, missing = [], []
    for d in day_list(first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d")):
        o = load_options(history, ccy, d)
        if o is None:
            missing.append(d); continue
        parts.append(o[o.name.isin(names)][["ts", "name", "amt", "sgn"]])
    tr = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["ts", "name", "amt", "sgn"])
    res = []
    by = dict(tuple(tr.groupby("name"))) if len(tr) else {}
    for name, c in coll.groupby("name"):
        tt = c.t.to_numpy()
        z = by.get(name)
        if z is None or z.empty:
            net = np.zeros(len(tt)); v24 = np.zeros(len(tt))
        else:
            ts = z.ts.to_numpy(); cs = np.concatenate([[0.0], np.cumsum(z.sgn.to_numpy() * z.amt.to_numpy())])
            ca = np.concatenate([[0.0], np.cumsum(z.amt.to_numpy())])
            i1 = np.searchsorted(ts, tt, side="right"); i0 = np.searchsorted(ts, tt - DAY_MS, side="right")
            net = cs[i1]; v24 = ca[i1] - ca[i0]
        res.append(pd.DataFrame({"currency": ccy, "name": name, "t": tt, "grid": c.grid.to_numpy(),
                                 "oi": c.oi.to_numpy(), "vol24_coll": c.vol24.to_numpy(),
                                 "net_recon": net, "vol24_recon": v24}))
    out = pd.concat(res, ignore_index=True)
    out.attrs["missing_days"] = missing
    return out


def gate_summary(cmp: pd.DataFrame, cfg: Config = Config()) -> dict:
    if cmp.empty:
        return {"status": "NO_OVERLAP", "rows": 0}
    v = cmp[cmp.vol24_coll > 0]
    pooled = float(v.vol24_recon.sum() / v.vol24_coll.sum()) if len(v) else float("nan")
    ratio = (v.vol24_recon / v.vol24_coll)
    tol = 0.01 * cmp.oi + 0.1
    viol = np.abs(cmp.net_recon) > cmp.oi + tol
    from scipy.stats import spearmanr
    rho = float(spearmanr(np.abs(cmp.net_recon), cmp.oi).statistic) if len(cmp) > 10 else float("nan")
    ok = (cfg.gate_volume_band[0] <= pooled <= cfg.gate_volume_band[1]) and viol.mean() <= cfg.gate_max_violation_share
    return {"rows": int(len(cmp)), "instruments": int(cmp.name.nunique()),
            "first": str(pd.to_datetime(cmp.t.min(), unit="ms", utc=True)),
            "last": str(pd.to_datetime(cmp.t.max(), unit="ms", utc=True)),
            "volume_ratio_pooled": pooled, "volume_ratio_median": float(ratio.median()) if len(v) else float("nan"),
            "share_rows_volume_within_5pct": float((np.abs(ratio - 1) <= 0.05).mean()) if len(v) else float("nan"),
            "bound_violation_share": float(viol.mean()), "spearman_absnet_oi": rho,
            "median_absnet_over_oi": float((np.abs(cmp.net_recon) / cmp.oi.where(cmp.oi > 0)).median()),
            "missing_history_days": cmp.attrs.get("missing_days", []),
            "status": "PASS" if ok else "FAIL"}


# ----------------------------------------------------------------------------- shared helpers
def _session_date(ts: pd.Series) -> pd.Series:
    """Session d = (08:00 d, 08:00 d+1]; a bar is labelled by its END time."""
    return (pd.to_datetime(ts, utc=True) - pd.Timedelta(hours=EXPIRY_HOUR) - pd.Timedelta(milliseconds=1)).dt.floor("D")


def _ols(y, X, cov: str, groups=None, lags: int = 5):
    import statsmodels.api as sm
    X = sm.add_constant(X, has_constant="add")
    m = sm.OLS(y, X, missing="drop")
    if cov == "hac":
        return m.fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return m.fit(cov_type="cluster", cov_kwds={"groups": groups})


def _one_sided(t: float, sign: int) -> float:
    from scipy.stats import norm
    return float(norm.sf(sign * t))


def load_dvol(data_root: Path, ccy: str) -> pd.Series:
    p = Path(data_root) / "history" / "dvol" / f"{ccy}_dvol_1h.csv"
    if not p.is_file():
        return pd.Series(dtype=float)
    d = pd.read_csv(p)
    return pd.Series(d["close"].to_numpy(float), index=pd.to_datetime(d["t"], utc=True)).sort_index()


# ----------------------------------------------------------------------------- stage 03: regime
def daily_panel(grid: pd.DataFrame, dvol: pd.Series, cfg: Config = Config()) -> pd.DataFrame:
    g = grid.copy()
    g["sess"] = _session_date(g.ts)
    g["px"] = g.groupby("sess").perp_last.transform(lambda s: s.ffill(limit=3))
    g["r"] = np.log(g.px).groupby(g.sess).diff()
    g["usd"] = g.buy_usd + g.sell_usd
    g["rlag"] = g.groupby("sess").r.shift(1)
    agg = g.groupby("sess").agg(n_bars=("r", "count"), usd=("usd", "sum"), opt_ok=("option_file", "all"))
    agg["rv2"] = (g.r ** 2).groupby(g.sess).sum(min_count=1)
    agg["cross"] = (g.r * g.rlag).groupby(g.sess).sum(min_count=1)
    agg["log_rv"] = 0.5 * np.log(agg.rv2.where(agg.rv2 > 0))
    agg["ac1"] = agg.cross / agg.rv2.where(agg.rv2 > 0)
    agg["adv30"] = agg.usd.shift(1).rolling(cfg.adv_sessions, min_periods=cfg.adv_sessions // 2).mean()
    first = g[(g.ts.dt.hour == EXPIRY_HOUR) & (g.ts.dt.minute == 5)].set_index("sess")
    for v in VARIANTS + ("recon_1d",):
        agg[f"x_{v}"] = first[f"gex_{v}"].reindex(agg.index) / agg.adv30
    if len(dvol):
        ref = agg.index + pd.Timedelta(hours=EXPIRY_HOUR - 1)
        agg["log_dvol"] = np.log(dvol.reindex(ref, method="ffill").to_numpy())
    else:
        agg["log_dvol"] = np.nan
    agg["log_rv_lag"] = agg.log_rv.shift(1)
    agg["dow"] = agg.index.dayofweek
    agg["period"] = period_of(agg.index)
    agg = agg[(agg.n_bars >= MIN_SESSION_BARS) & agg.opt_ok]
    return agg.reset_index().rename(columns={"sess": "session"})


REGIME_TARGETS = {"log_rv": -1, "ac1": -1}    # hypothesised sign of the GEX coefficient


def regime_fit(panel: pd.DataFrame, period: str, variant: str = PRIMARY, cfg: Config = Config()) -> list[dict]:
    p = panel[panel.period == period].copy()
    x = f"x_{variant}"
    p = p.dropna(subset=[x, "log_dvol", "log_rv_lag", "log_rv", "ac1"])
    out = []
    if len(p) < 40:
        return [{"period": period, "variant": variant, "target": y, "n": len(p), "status": "TOO_FEW"}
                for y in REGIME_TARGETS]
    sd = p[x].std()
    X = pd.DataFrame({"gex_z": (p[x] - p[x].mean()) / sd, "log_dvol": p.log_dvol, "log_rv_lag": p.log_rv_lag})
    X = pd.concat([X, pd.get_dummies(p.dow, prefix="dow", drop_first=True, dtype=float)], axis=1)
    for y, sign in REGIME_TARGETS.items():
        f = _ols(p[y], X, "hac", lags=cfg.hac_lags)
        t = float(f.tvalues["gex_z"])
        out.append({"period": period, "variant": variant, "target": y, "n": int(f.nobs),
                    "beta_per_sd": float(f.params["gex_z"]), "t": t, "p_one_sided": _one_sided(t, sign),
                    "hyp_sign": sign, "r2": float(f.rsquared)})
    return out


def terciles(panel: pd.DataFrame, period: str, variant: str = PRIMARY) -> pd.DataFrame:
    p = panel[panel.period == period].dropna(subset=[f"x_{variant}"])
    if len(p) < 30:
        return pd.DataFrame()
    q = pd.qcut(p[f"x_{variant}"], 3, labels=["low (short gamma)", "mid", "high (long gamma)"])
    return p.groupby(q, observed=True).agg(n=("log_rv", "size"), gex_mean=(f"x_{variant}", "mean"),
                                           rv_day_pct=("log_rv", lambda s: 100 * np.exp(s).mean()),
                                           ac1=("ac1", "mean")).reset_index(names="tercile")


# ----------------------------------------------------------------------------- stage 04: hedge flow
def flow_panel(grid: pd.DataFrame, panel: pd.DataFrame, cfg: Config = Config()) -> pd.DataFrame:
    g = grid.copy()
    g["sess"] = _session_date(g.ts)
    g["px"] = g.groupby("sess").perp_last.transform(lambda s: s.ffill(limit=3))
    g["r"] = np.log(g.px).groupby(g.sess).diff()
    adv = panel.set_index("session").adv30 if len(panel) else pd.Series(dtype=float)
    g["adv5"] = g.sess.map(adv) / BARS
    flow = (g.buy_usd - g.sell_usd)
    for v in VARIANTS:
        G = g[f"gex_{v}"].groupby(g.sess).shift(1)
        g[f"h_{v}"] = -100.0 * G * g.r / g.adv5
        g[f"gl_{v}"] = G / g.adv5
    g["f_now"] = flow / g.adv5
    for h in FLOW_H:
        fwd = sum(flow.groupby(g.sess).shift(-j) for j in range(1, h + 1))
        g[f"f_fwd{h}"] = fwd / g.adv5
    mod = g.ts.dt.hour * 60 + g.ts.dt.minute
    g = g[~((mod >= EXCLUDE_EXPIRY_WINDOW[0]) & (mod <= EXCLUDE_EXPIRY_WINDOW[1])) & g.option_file]
    g["period"] = period_of(g.sess)
    keep = ["ts", "sess", "period", "r", "adv5", "f_now"] + [f"f_fwd{h}" for h in FLOW_H] + \
        [f"h_{v}" for v in VARIANTS] + [f"gl_{v}" for v in VARIANTS]
    return g[keep].replace([np.inf, -np.inf], np.nan).dropna(subset=["r", "adv5"]).reset_index(drop=True)


def flow_fit(fp: pd.DataFrame, period: str, variant: str = PRIMARY, target: str = "f_fwd3",
             winsor: float | None = None) -> dict:
    p = fp[fp.period == period].dropna(subset=["r", target, f"h_{variant}", f"gl_{variant}"])
    if len(p) < 500:
        return {"period": period, "variant": variant, "target": target, "n": len(p), "status": "TOO_FEW"}
    y = p[target]; H = p[f"h_{variant}"]
    if winsor:
        lo, hi = y.quantile([winsor, 1 - winsor]); y = y.clip(lo, hi)
        lo, hi = H.quantile([winsor, 1 - winsor]); H = H.clip(lo, hi)
    X = pd.DataFrame({"hedge": H, "r_bp": 1e4 * p.r, "gex": p[f"gl_{variant}"]})
    f = _ols(y, X, "cluster", groups=pd.factorize(p.sess)[0])
    t = float(f.tvalues["hedge"])
    return {"period": period, "variant": variant, "target": target, "winsor": winsor or 0.0, "n": int(f.nobs),
            "sessions": int(p.sess.nunique()), "b_hedge": float(f.params["hedge"]), "t": t,
            "p_one_sided": _one_sided(t, +1), "b_r_bp": float(f.params["r_bp"]), "r2": float(f.rsquared)}


# ----------------------------------------------------------------------------- stage 05: pinning
def expiry_type(day: str) -> str:
    d = pd.Timestamp(day)
    if d.dayofweek != 4:
        return "daily"
    last_friday = (d + pd.Timedelta(days=7)).month != d.month
    if last_friday and d.month in (3, 6, 9, 12):
        return "quarterly"
    return "monthly" if last_friday else "weekly"


def pin_events(pins: pd.DataFrame, settle: pd.DataFrame, cfg: Config = Config()) -> pd.DataFrame:
    """One row per (expiry, selection rule): pin strike K, mirrored placebo K', settlement-distance difference."""
    if pins.empty:
        return pd.DataFrame()
    st = settle.set_index(["currency", "expiry_date"]).settle_proxy
    agg = pins.groupby(["currency", "expiry_date", "strike"]).agg(S_ref=("S_ref", "first"), dg=("dg_recon", "sum"),
                                                                     dg_nb=("dg_noblock", "sum"),
                                                                     abs_pos=("abs_pos", "sum")).reset_index()
    agg = agg[np.abs(agg.strike / agg.S_ref - 1) <= cfg.pin_band]
    rows = []
    for (c, e), z in agg.groupby(["currency", "expiry_date"]):
        X = st.get((c, e), np.nan); S = float(z.S_ref.iat[0])
        if not np.isfinite(X) or not np.isfinite(S):
            continue
        picks = {}
        if (z.dg > 0).any():
            picks["long_gamma"] = z.loc[z.dg.idxmax(), "strike"]
        if (z.dg_nb > 0).any():
            picks["long_gamma_noblock"] = z.loc[z.dg_nb.idxmax(), "strike"]
        if (z.dg < 0).any():
            picks["short_gamma"] = z.loc[z.dg.idxmin(), "strike"]
        if (z.abs_pos > 0).any():
            picks["max_position"] = z.loc[z.abs_pos.idxmax(), "strike"]
        for rule, K in picks.items():
            Kp = 2 * S - K
            rows.append({"currency": c, "expiry_date": e, "type": expiry_type(e), "rule": rule, "S_ref": S,
                         "strike": K, "placebo": Kp, "settle": X, "d0_bp": 1e4 * abs(S - K) / S,
                         "dist_pin_bp": 1e4 * abs(X - K) / S, "dist_placebo_bp": 1e4 * abs(X - Kp) / S,
                         "delta_bp": 1e4 * (abs(X - K) - abs(X - Kp)) / S})
    ev = pd.DataFrame(rows)
    if len(ev):
        ev["period"] = period_of(ev.expiry_date)
    return ev


PIN_HYP = {"long_gamma": -1, "long_gamma_noblock": -1, "max_position": -1, "short_gamma": +1}


def pin_test(ev: pd.DataFrame, period: str, B: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    e = ev[ev.period == period]
    groups = [("all", e)] + [(t, e[e.type == t]) for t in ("daily", "weekly", "monthly", "quarterly")] + \
             [("friday", e[e.type != "daily"])]
    for name, z in groups:
        for (c, rule), w in z.groupby(["currency", "rule"]):
            x = w.delta_bp.to_numpy()
            if len(x) < 5:
                continue
            boots = rng.choice(x, (B, len(x))).mean(1)
            sign = PIN_HYP[rule]
            p = float((boots * sign <= 0).mean())
            out.append({"period": period, "group": name, "currency": c, "rule": rule, "n": len(x),
                        "mean_delta_bp": float(x.mean()), "median_delta_bp": float(np.median(x)),
                        "share_closer_to_pin": float((x < 0).mean()), "hyp_sign": sign, "p_one_sided": p})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- decisions
def decide(rows: pd.DataFrame, key_cols: list[str], cfg: Config = Config()) -> dict:
    """PASS needs the hypothesised sign with one-sided p < alpha in BOTH currencies in a period."""
    res = {}
    for per, alpha in (("discovery", cfg.alpha_discovery), ("holdout", cfg.alpha_holdout)):
        z = rows[rows.period == per]
        if z.empty:
            res[per] = "NOT_RUN"; continue
        verdicts = {}
        for k, w in z.groupby(key_cols):
            ok = set(w[w.p_one_sided < alpha].currency) >= set(CURRENCIES)
            verdicts[str(k)] = "PASS" if ok else "FAIL"
        res[per] = verdicts
    return res


def dump(obj, path: Path):
    Path(path).write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf8")


# ----------------------------------------------------------------------------- synthetic data (tests / smoke runs)
def synthetic_root(root: Path, days: list[str], seed: int = 0, pin_strength: float = 0.0) -> Path:
    """Writes history trades (options + perpetual), DVOL and a collector grid consistent with the trades."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    base = {"BTC": 60000.0, "ETH": 3000.0}
    step = {"BTC": 1000.0, "ETH": 50.0}
    for ccy in CURRENCIES:
        S = base[ccy]
        dvol_rows, coll_rows = [], []
        for di, day in enumerate(days):
            d0 = ms(day)
            # perp path (5-second steps)
            n = 17280
            r = rng.normal(0, 0.0004, n)
            px = S * np.exp(np.cumsum(r))
            ts = d0 + np.arange(n) * 5000 + rng.integers(0, 4000, n)
            perp = pd.DataFrame({"trade_id": np.arange(n).astype(str), "trade_seq": np.arange(n),
                                 "instrument_name": PERP[ccy], "timestamp": pd.to_datetime(ts, unit="ms", utc=True),
                                 "direction": np.where(rng.random(n) < 0.5, "buy", "sell"), "price": px,
                                 "amount": rng.uniform(100, 5000, n), "index_price": px * (1 + rng.normal(0, 1e-5, n))})
            fp = root / "history" / "trades" / ccy / "future" / day[:4]; fp.mkdir(parents=True, exist_ok=True)
            perp.to_parquet(fp / f"{day}.parquet", index=False)
            # options: daily expiries today+1..+3 and the next Friday
            exps = sorted({pd.Timestamp(day) + pd.Timedelta(days=j) for j in (0, 1, 2, 3)} |
                          {pd.Timestamp(day) + pd.Timedelta(days=(4 - pd.Timestamp(day).dayofweek) % 7 or 7)})
            m = 1500
            ets = np.sort(d0 + rng.integers(0, DAY_MS, m))
            pick = rng.integers(0, len(exps), m)
            ks = np.round(S / step[ccy]) * step[ccy] + step[ccy] * rng.integers(-5, 6, m)
            cps = np.where(rng.random(m) < 0.5, "C", "P")
            names, keep = [], []
            for j in range(m):
                e = exps[pick[j]]
                if ms(e) + EXPIRY_HOUR * 3_600_000 <= ets[j]:
                    keep.append(False); names.append(""); continue
                keep.append(True)
                names.append(f"{ccy}-{e.day}{e.strftime('%b').upper()}{e.strftime('%y')}-{int(ks[j])}-{cps[j]}")
            keep = np.array(keep)
            iv = rng.uniform(40, 70, m)
            opt = pd.DataFrame({"trade_id": np.arange(m).astype(str), "trade_seq": np.arange(m),
                                "instrument_name": np.array(names), "timestamp": pd.to_datetime(ets, unit="ms", utc=True),
                                "direction": np.where(rng.random(m) < 0.55, "buy", "sell"),
                                "amount": np.round(rng.uniform(0.1, 10, m), 1), "iv": iv,
                                "index_price": np.interp(ets, ts, px),
                                "block_trade_id": np.where(rng.random(m) < 0.05, "B1", None),
                                "combo_id": None})[keep]
            op = root / "history" / "trades" / ccy / "option" / day[:4]; op.mkdir(parents=True, exist_ok=True)
            opt.to_parquet(op / f"{day}.parquet", index=False)
            S = float(px[-1])
            for h in range(24):
                dvol_rows.append({"t": pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=h), "open": 55, "high": 56,
                                  "low": 54, "close": 55 + rng.normal(0, 2)})
        pd.DataFrame(dvol_rows).to_csv(_mk(root / "history" / "dvol") / f"{ccy}_dvol_1h.csv", index=False)
    # collector grid for the last 5 days, built from the trades themselves (open interest >= |net position|)
    hist = root / "history"
    per_day: dict[str, list] = {d: [] for d in days[-5:]}
    for ccy in CURRENCIES:
        tr = pd.concat([load_options(hist, ccy, d) for d in days], ignore_index=True)
        for day in days[-5:]:
            exp = ms(day) + EXPIRY_HOUR * 3_600_000
            snaps = exp - DAY_MS + GRID_MS * np.arange(0, BARS)
            for name, w in tr[tr.expiry_ms == exp].groupby("name"):
                ts = w.ts.to_numpy()
                cs = np.concatenate([[0.0], np.cumsum((w.sgn * w.amt).to_numpy())])
                ca = np.concatenate([[0.0], np.cumsum(w.amt.to_numpy())])
                i1 = np.searchsorted(ts, snaps, "right"); i0 = np.searchsorted(ts, snaps - DAY_MS, "right")
                per_day[day].append(pd.DataFrame({
                    "snapshot_time": pd.to_datetime(snaps, unit="ms", utc=True), "instrument_name": name,
                    "api_currency": ccy, "underlying": ccy, "open_interest": np.abs(cs[i1]) + 2.0,
                    "volume_24h": ca[i1] - ca[i0], "expiration_timestamp": exp}))
    for day, parts in per_day.items():
        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(
                _mk(root / "daily" / "option_grid" / f"expiry={day}") / "option_grid.parquet", index=False)
    return root


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------------------------------------------------------- notebook plumbing
SYNTHETIC_DAYS = ("2024-06-01", "2024-09-30")


def notebook_env(number: str, project_root: Path) -> dict:
    """Paths for a DG notebook. DG_SYNTHETIC=1 -> shared synthetic data under the temp folder (smoke runs only)."""
    import os
    import tempfile
    syn = os.getenv("DG_SYNTHETIC", "").strip() == "1"
    if syn:
        base = Path(tempfile.gettempdir()) / "dg_synthetic"
        root = base / "data"
        if not (root / "history").exists():
            synthetic_root(root, day_list(*SYNTHETIC_DAYS), seed=7)
        out = base / "out" / number
        cache = base / "cache"
    else:
        root = Path(os.environ["DERIBIT_DATA_ROOT"]).expanduser().resolve()
        out = Path(project_root) / "notebooks" / "Dealer_Gamma" / number
        cache = Path(os.getenv("DG_CACHE") or root / "derived" / "dealer_gamma").expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    return {"synthetic": syn, "data_root": root, "history": root / "history", "cache": cache, "out": out,
            "holdout": os.getenv("DG_HOLDOUT", "").strip() == "1", "force": os.getenv("DG_FORCE", "").strip() == "1"}


def available_days(history: Path, ccy: str) -> tuple[str, str]:
    """First and last UTC day with BOTH an option and a future trade file."""
    def days(kind):
        return {p.stem for p in (Path(history) / "trades" / ccy / kind).glob("*/*.parquet")}
    both = sorted(days("option") & days("future"))
    if not both:
        raise FileNotFoundError(f"no {ccy} option+future trade files under {history}")
    return max(both[0], DATA_START), both[-1]


def cache_paths(cache: Path, ccy: str) -> dict:
    return {k: Path(cache) / f"{k}_{ccy}.parquet" for k in ("grid", "pins", "settle", "coverage")} | \
        {"meta": Path(cache) / f"meta_{ccy}.json"}


def load_cache(cache: Path, ccy: str) -> dict:
    p = cache_paths(cache, ccy)
    if not p["meta"].is_file():
        raise FileNotFoundError(f"{p['meta']} missing: run notebook 02 (build) first")
    out = {k: pd.read_parquet(v) for k, v in p.items() if k != "meta" and v.is_file()}
    out["meta"] = json.loads(p["meta"].read_text())
    return out
