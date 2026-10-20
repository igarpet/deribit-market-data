"""
Liquidity Absorption A06: Deribit derivatives context around the frozen A04 entries, -240..+240 minutes.

Protocol: discussions/Liquidity_Absorption/023_claude_a06_deribit_entry_context_protocol.md

What it adds over A05a (Binance 1m price/flow): information that only exists on Deribit, built from the
downloaded public trade history (history.deribit.com):
    perpetual   aligned price path (bp), taker imbalance, LIQUIDATIONS by side, mark-index premium, block trades
    options     short-dated ATM implied vol, put-call skew, taker-signed option flow (call/put, delta proxy), blocks
    futures     nearest dated-future annualized basis vs index
Anchor: the A04 entry time (first post-signal Binance print, ms) floored to the minute; minute 0 = the entry minute.
Everything is signed to the trade side: positive = favourable to the trade (long +, short -).
Controls: 3 random non-signal anchors per event, same symbol and same UTC hour, July-September 2026, at least 240
minutes from any signal of that symbol, side copied from the event (so 'aligned' means the same thing).

Descriptive, outcome-conditioned, in-sample (the Jul-Sep 2026 quarter has been inspected since A02). Pre-entry
differences that survive Benjamini-Hochberg q <= 0.10 are CANDIDATES for the LA 022 holdout, nothing more.
Post-entry differences are partly mechanical (they define TARGET vs STOP) and are reported for anatomy only.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "LA-A06-0.1.0"
OUTCOMES = ("TARGET", "STOP", "TIMEOUT")
WINDOW_MIN = 240
PHASES = {"context": (-240, -61), "approach": (-60, -16), "formation": (-15, -1), "entry": (0, 5),
          "early": (6, 60), "late": (61, 240)}
PRE_PHASES = ("context", "approach", "formation")
N_CONTROLS = 3
SEED = 20261020
FDR_Q = 0.10
ATM_BAND = 0.03          # |ln K/S| for ATM IV
SKEW_BAND = (0.04, 0.12)  # OTM put ln K/S in [-0.12,-0.04], OTM call in [0.04, 0.12]
TENOR_DAYS = (1.0, 35.0)
MIN_FUT_DAYS = 3.0
OPT_RE = re.compile(r"^(?P<u>[A-Z]+)-(?P<d>\d{1,2})(?P<m>[A-Z]{3})(?P<y>\d{2})-(?P<k>[\d.d]+)-(?P<cp>[CP])$")
FUT_RE = re.compile(r"^(?P<u>[A-Z]+)-(?P<d>\d{1,2})(?P<m>[A-Z]{3})(?P<y>\d{2})$")
MONTHS = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV",
                                      "DEC"], start=1)}

# minute-feature definitions: name -> (source, description)
FEATURES = {
    "px_bp": "aligned perp price vs the last print before entry (bp)",
    "imb": "aligned perp taker imbalance (buy-sell)/(buy+sell)",
    "vol_rel": "perp traded amount / median per-minute amount of that day",
    "liq_with": "perp liquidation amount of positions ON the trade side (forced exits like ours), USD",
    "liq_against": "perp liquidation amount of positions AGAINST the trade side (forced exits of opponents), USD",
    "premium_bp": "aligned perp mark-index premium (bp)",
    "block_amt": "perp block-trade amount, USD",
    "fut_basis": "aligned nearest dated-future annualized basis (%), change vs entry",
    "atm_iv": "short-dated ATM option IV (vol points), change vs entry",
    "skew": "aligned put-minus-call IV (vol points), change vs entry; + = skew moving in favour of the trade",
    "opt_delta_flow": "aligned taker-signed option flow (call buys - call sells - put buys + put sells), coin units",
    "opt_amt": "option traded amount, coin units",
    "opt_block_amt": "option block-trade amount, coin units",
}
LEVEL_FEATURES = ("fut_basis", "atm_iv", "skew")       # expressed as change vs entry minute (carried forward)


@dataclass(frozen=True)
class Config:
    window_min: int = WINDOW_MIN
    n_controls: int = N_CONTROLS
    seed: int = SEED
    fdr_q: float = FDR_Q
    first_day: str = "2026-06-30"
    last_day: str = "2026-10-01"


def registration(cfg: Config = Config()) -> dict:
    return {"version": VERSION, **asdict(cfg), "phases": PHASES, "pre_phases": PRE_PHASES, "features": FEATURES,
            "atm_band": ATM_BAND, "skew_band": SKEW_BAND, "tenor_days": TENOR_DAYS, "min_future_days": MIN_FUT_DAYS,
            "status": "HISTORICAL_EXPLORATORY_OUTCOME_CONDITIONED_IN_SAMPLE"}


# ----------------------------------------------------------------------------- cohort and controls
def load_cohort(repo: Path) -> pd.DataFrame:
    o = pd.read_csv(Path(repo) / "notebooks" / "Liquidity_Absorption" / "04" / "04_primary_event_outcomes.csv",
                    low_memory=False)
    if len(o) != 452 or o.a02_candidate_row.duplicated().any():
        raise ValueError("Frozen A04 452-ID cohort absent or changed")
    keep = ["a02_candidate_row", "symbol", "side", "signal_time_utc", "reason", "level_type", "entry_time_ms",
            "exit_time_ms", "gross_return_bp", "target_distance_bp", "initial_risk_bp", "target_to_sigma60",
            "time_to_exit_min", "mfe_print_initial_r", "mae_print_initial_r"]
    c = o[[k for k in keep if k in o.columns]].copy()
    c = c[c.entry_time_ms.notna()].copy()
    c["entry_ms"] = c.entry_time_ms.astype("int64")
    c["group"] = np.where(c.reason.isin(OUTCOMES), c.reason, "OTHER")
    c["sign"] = np.where(c.side.eq("long"), 1.0, -1.0)
    c["currency"] = c.symbol.str.replace("USDT", "", regex=False)
    c["geometry"] = np.where(c.target_to_sigma60 <= 1.0, "near_target", "far_target")
    c["kind"] = "SIGNAL"
    c["anchor_id"] = c.a02_candidate_row.astype(str)
    return c.reset_index(drop=True)


def make_controls(cohort: pd.DataFrame, cfg: Config = Config()) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    lo = pd.Timestamp(cfg.first_day, tz="UTC") + pd.Timedelta(days=1)
    hi = pd.Timestamp(cfg.last_day, tz="UTC") - pd.Timedelta(days=1)
    days = pd.date_range(lo.normalize(), hi.normalize() - pd.Timedelta(days=1), freq="D")
    rows = []
    for _, e in cohort.iterrows():
        sig = cohort.loc[cohort.symbol == e.symbol, "entry_ms"].to_numpy()
        hour = pd.Timestamp(int(e.entry_ms), unit="ms", tz="UTC").hour
        got, tries = 0, 0
        while got < cfg.n_controls and tries < 500:
            tries += 1
            t = days[rng.integers(len(days))] + pd.Timedelta(hours=hour, minutes=int(rng.integers(60)))
            tm = int(t.value // 10**6)
            if np.min(np.abs(sig - tm)) < cfg.window_min * 60_000:
                continue
            r = e.copy()
            r["entry_ms"] = tm
            r["kind"] = "CONTROL"
            r["group"] = "CONTROL"
            r["anchor_id"] = f"C{int(e.a02_candidate_row)}_{got}"
            for col in ("reason", "gross_return_bp", "time_to_exit_min", "mfe_print_initial_r", "mae_print_initial_r"):
                if col in r:
                    r[col] = np.nan
            rows.append(r)
            got += 1
    return pd.DataFrame(rows).reset_index(drop=True)


# ----------------------------------------------------------------------------- Deribit minute bars
def _parse(names: pd.Series) -> pd.DataFrame:
    u = names.astype(str).unique()
    rec = {}
    for n in u:
        m = OPT_RE.match(n)
        if m:
            rec[n] = ("option", pd.Timestamp(2000 + int(m["y"]), MONTHS[m["m"]], int(m["d"]), 8, tz="UTC").value // 10**6,
                      float(m["k"].replace("d", ".")), m["cp"])
            continue
        if n.endswith("PERPETUAL"):
            rec[n] = ("perpetual", np.nan, np.nan, None)
            continue
        f = FUT_RE.match(n)
        if f:
            rec[n] = ("future", pd.Timestamp(2000 + int(f["y"]), MONTHS[f["m"]], int(f["d"]), 8, tz="UTC").value // 10**6,
                      np.nan, None)
        else:
            rec[n] = ("other", np.nan, np.nan, None)
    t = pd.DataFrame.from_dict(rec, orient="index", columns=["ikind", "expiry_ms", "strike", "cp"])
    return t.reindex(names.astype(str).to_numpy()).reset_index(drop=True)


def _ts_ms(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s).to_numpy("int64")
    return pd.to_datetime(s, utc=True).dt.as_unit("ms").astype("int64").to_numpy()


def _read(path: Path, want: list[str]) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    cols = [c for c in want if c in pf.schema.names]
    return pf.read(columns=cols).to_pandas()


def liquidation_sides(direction: pd.Series, liq: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Liquidated LONG / SHORT flags per print. 'T' = taker side liquidated, 'M' = maker side, 'MT' = both.
    Taker sell = taker was long; maker on a taker buy was selling = maker was long."""
    lq = liq.astype("string").fillna("")
    tak = lq.str.contains("T").to_numpy()
    mak = lq.str.contains("M").to_numpy()
    buy = direction.astype(str).eq("buy").to_numpy()
    long_liq = (tak & ~buy) | (mak & buy)
    short_liq = (tak & buy) | (mak & ~buy)
    return long_liq, short_liq


def perp_minutes(df: pd.DataFrame, perp: str) -> pd.DataFrame:
    z = df[df.instrument_name.astype(str).eq(perp)].copy()
    if z.empty:
        return pd.DataFrame()
    z["t"] = _ts_ms(z.timestamp)
    z["minute"] = z.t // 60_000
    z["price"] = pd.to_numeric(z.price, errors="coerce")
    z["amount"] = pd.to_numeric(z.amount, errors="coerce")
    buy = z.direction.astype(str).eq("buy")
    z["buy_amt"] = np.where(buy, z.amount, 0.0)
    z["sell_amt"] = np.where(~buy, z.amount, 0.0)
    ll, sl = liquidation_sides(z.direction, z.get("liquidation", pd.Series([None] * len(z), index=z.index)))
    z["liq_long"] = np.where(ll, z.amount, 0.0)
    z["liq_short"] = np.where(sl, z.amount, 0.0)
    if "mark_price" in z and "index_price" in z:
        z["prem"] = 1e4 * (pd.to_numeric(z.mark_price, errors="coerce") / pd.to_numeric(z.index_price, errors="coerce") - 1)
    else:
        z["prem"] = np.nan
    blk = z.get("block_trade_id", pd.Series([None] * len(z), index=z.index))
    z["block"] = np.where(blk.notna() & blk.astype(str).ne("") & blk.astype(str).ne("None"), z.amount, 0.0)
    z = z.sort_values("t")
    g = z.groupby("minute")
    out = pd.DataFrame({"close": g.price.last(), "buy_amt": g.buy_amt.sum(), "sell_amt": g.sell_amt.sum(),
                        "liq_long": g.liq_long.sum(), "liq_short": g.liq_short.sum(), "prem": g.prem.median(),
                        "block_amt": g.block.sum(), "n": g.size()})
    return out


def future_minutes(df: pd.DataFrame) -> pd.DataFrame:
    p = _parse(df.instrument_name)
    z = df.reset_index(drop=True).join(p)
    z = z[z.ikind.eq("future")].copy()
    if z.empty or "index_price" not in z:
        return pd.DataFrame()
    z["t"] = _ts_ms(z.timestamp)
    days = (z.expiry_ms - z.t) / 86_400_000
    z = z[days >= MIN_FUT_DAYS].copy()
    if z.empty:
        return pd.DataFrame()
    z["days"] = (z.expiry_ms - z.t) / 86_400_000
    nearest = z.groupby(z.t // 86_400_000).expiry_ms.transform("min")
    z = z[z.expiry_ms.eq(nearest)].copy()
    z["basis"] = 100 * (pd.to_numeric(z.price) / pd.to_numeric(z.index_price) - 1) * 365 / z.days
    z["minute"] = z.t // 60_000
    return pd.DataFrame({"fut_basis_lvl": z.groupby("minute").basis.median()})


def option_minutes(df: pd.DataFrame) -> pd.DataFrame:
    p = _parse(df.instrument_name)
    z = df.reset_index(drop=True).join(p)
    z = z[z.ikind.eq("option")].copy()
    if z.empty:
        return pd.DataFrame()
    z["t"] = _ts_ms(z.timestamp)
    z["minute"] = z.t // 60_000
    z["amount"] = pd.to_numeric(z.amount, errors="coerce")
    S = pd.to_numeric(z.get("index_price"), errors="coerce")
    z["lm"] = np.log(z.strike / S)
    z["tenor"] = (z.expiry_ms - z.t) / 86_400_000
    iv = pd.to_numeric(z.get("iv"), errors="coerce")
    ok_t = z.tenor.between(*TENOR_DAYS)
    buy = z.direction.astype(str).eq("buy")
    sgn = np.where(buy, 1.0, -1.0)
    z["delta_flow"] = np.where(z.cp.eq("C"), sgn, -sgn) * z.amount
    blk = z.get("block_trade_id", pd.Series([None] * len(z), index=z.index))
    z["block"] = np.where(blk.notna() & blk.astype(str).ne("") & blk.astype(str).ne("None"), z.amount, 0.0)
    z["atm_iv_x"] = np.where(ok_t & (z.lm.abs() <= ATM_BAND), iv, np.nan)
    z["put_iv_x"] = np.where(ok_t & z.cp.eq("P") & z.lm.between(-SKEW_BAND[1], -SKEW_BAND[0]), iv, np.nan)
    z["call_iv_x"] = np.where(ok_t & z.cp.eq("C") & z.lm.between(SKEW_BAND[0], SKEW_BAND[1]), iv, np.nan)
    g = z.groupby("minute")
    return pd.DataFrame({"atm_iv_lvl": g.atm_iv_x.median(), "put_iv": g.put_iv_x.median(), "call_iv": g.call_iv_x.median(),
                         "opt_delta_flow": g.delta_flow.sum(), "opt_amt": g.amount.sum(), "opt_block_amt": g.block.sum()})


def build_minutes(history: Path, currency: str, days: list[str], log=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dense per-minute Deribit table for one currency over the given UTC days, plus per-day coverage."""
    perp = f"{currency}-PERPETUAL"
    parts, cov = [], []
    fcols = ["timestamp", "instrument_name", "price", "amount", "direction", "mark_price", "index_price",
             "liquidation", "block_trade_id"]
    ocols = ["timestamp", "instrument_name", "price", "amount", "direction", "iv", "index_price", "block_trade_id"]
    for d in days:
        fp = Path(history) / "trades" / currency / "future" / d[:4] / f"{d}.parquet"
        op = Path(history) / "trades" / currency / "option" / d[:4] / f"{d}.parquet"
        f = _read(fp, fcols)
        o = _read(op, ocols)
        pm = perp_minutes(f, perp) if f is not None and len(f) else pd.DataFrame()
        fm = future_minutes(f) if f is not None and len(f) else pd.DataFrame()
        om = option_minutes(o) if o is not None and len(o) else pd.DataFrame()
        day = pd.concat([x for x in (pm, fm, om) if len(x)], axis=1) if (len(pm) or len(fm) or len(om)) else pd.DataFrame()
        if len(day):
            parts.append(day)
        cov.append({"currency": currency, "day": d, "future_file": fp.is_file(), "option_file": op.is_file(),
                    "perp_prints": int(pm.n.sum()) if len(pm) else 0, "option_prints_minutes": int(len(om)),
                    "has_liquidation_field": bool(f is not None and "liquidation" in f.columns)})
        if log:
            log(currency, d)
    if not parts:
        return pd.DataFrame(), pd.DataFrame(cov)
    m = pd.concat(parts).sort_index()
    m = m[~m.index.duplicated(keep="first")]
    full = np.arange(m.index.min(), m.index.max() + 1)
    m = m.reindex(full)
    for c in ("buy_amt", "sell_amt", "liq_long", "liq_short", "block_amt", "n", "opt_delta_flow", "opt_amt",
              "opt_block_amt"):
        if c in m:
            m[c] = m[c].fillna(0.0)
    for c in ("close", "prem", "fut_basis_lvl", "atm_iv_lvl", "put_iv", "call_iv"):
        if c in m:
            m[c] = m[c].ffill(limit=None if c in ("close", "prem") else 60)
    vol = m.buy_amt + m.sell_amt if "buy_amt" in m else pd.Series(0.0, index=m.index)
    day_med = vol.groupby(vol.index // 1440).transform(lambda x: max(float(np.median(x[x > 0])) if (x > 0).any() else 1.0, 1e-9))
    m["vol_rel"] = vol / day_med
    m.index.name = "minute"
    return m, pd.DataFrame(cov)


# ----------------------------------------------------------------------------- event panel
def event_panel(anchors: pd.DataFrame, minutes: dict, cfg: Config = Config()) -> pd.DataFrame:
    rel = np.arange(-cfg.window_min, cfg.window_min + 1)
    rows = []
    for _, e in anchors.iterrows():
        m = minutes.get(e.currency)
        if m is None or m.empty:
            continue
        m0 = int(e.entry_ms) // 60_000
        idx = m0 + rel
        if idx[0] < m.index[0] or idx[-1] > m.index[-1]:
            continue
        w = m.loc[idx]
        s = float(e.sign)
        prev = m.loc[:m0 - 1, "close"] if "close" in m else pd.Series(dtype=float)
        ref = float(prev.iloc[-1]) if len(prev) and np.isfinite(prev.iloc[-1]) else float(w["close"].iloc[cfg.window_min])
        tot = (w.buy_amt + w.sell_amt).to_numpy()
        imb = np.divide(w.buy_amt.to_numpy() - w.sell_amt.to_numpy(), tot, out=np.full(len(w), np.nan), where=tot > 0)
        liq_with = (w.liq_long if s > 0 else w.liq_short).to_numpy()
        liq_against = (w.liq_short if s > 0 else w.liq_long).to_numpy()
        d = {"anchor_id": e.anchor_id, "kind": e.kind, "group": e.group, "symbol": e.symbol, "side": e.side,
             "geometry": e.geometry, "rel_min": rel,
             "px_bp": s * 1e4 * (w.close.to_numpy() / ref - 1), "imb": s * imb, "vol_rel": w.vol_rel.to_numpy(),
             "liq_with": liq_with, "liq_against": liq_against, "premium_bp": s * w.prem.to_numpy(),
             "block_amt": w.block_amt.to_numpy()}
        for lvl, name, sg in (("fut_basis_lvl", "fut_basis", s), ("atm_iv_lvl", "atm_iv", 1.0)):
            if lvl in w:
                x = w[lvl].to_numpy(float)
                d[name] = sg * (x - x[cfg.window_min])
            else:
                d[name] = np.full(len(rel), np.nan)
        if "put_iv" in w:
            sk = (w.put_iv - w.call_iv).ffill(limit=60).bfill(limit=60).to_numpy(float)
            # skew moving in favour: for a long, put skew FALLING (less fear) is favourable -> -1 * change
            d["skew"] = -s * (sk - sk[cfg.window_min])
        else:
            d["skew"] = np.full(len(rel), np.nan)
        for c in ("opt_delta_flow", "opt_amt", "opt_block_amt"):
            d[c] = (s if c == "opt_delta_flow" else 1.0) * w[c].to_numpy(float) if c in w else np.full(len(rel), np.nan)
        rows.append(pd.DataFrame(d))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def phase_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Per anchor and phase: price change over the phase, sums of flow/liquidation/blocks, means of imbalance,
    activity and premium, and the level change (end minus start) for basis / ATM IV / skew."""
    out = []
    flows = ("liq_with", "liq_against", "block_amt", "opt_delta_flow", "opt_amt", "opt_block_amt")
    means = ("imb", "vol_rel", "premium_bp")
    for (aid, grp), g in panel.groupby(["anchor_id", "group"], sort=False):
        g = g.set_index("rel_min")
        base = {"anchor_id": aid, "group": grp, "kind": g.kind.iloc[0], "symbol": g.symbol.iloc[0],
                "geometry": g.geometry.iloc[0]}
        for ph, (a, b) in PHASES.items():
            w = g.loc[a:b]
            before = g.loc[a - 1] if (a - 1) in g.index else g.loc[a]
            r = {**base, "phase": ph, "px_bp": float(w.px_bp.iloc[-1] - before.px_bp)}
            for c in flows:
                r[c] = float(np.nansum(w[c]))
            for c in means:
                r[c] = float(np.nanmean(w[c])) if np.isfinite(w[c]).any() else np.nan
            for c in LEVEL_FEATURES:
                v = w[c].to_numpy(float)
                b0 = float(before[c]) if np.isfinite(before[c]) else (v[np.isfinite(v)][0] if np.isfinite(v).any() else np.nan)
                last = v[np.isfinite(v)][-1] if np.isfinite(v).any() else np.nan
                r[c] = float(last - b0) if np.isfinite(last) and np.isfinite(b0) else np.nan
            out.append(r)
    return pd.DataFrame(out)


def median_curves(panel: pd.DataFrame) -> pd.DataFrame:
    feats = list(FEATURES)
    q = panel.groupby(["group", "rel_min"])[feats]
    med = q.median().add_suffix("_med")
    lo = q.quantile(0.25).add_suffix("_q25")
    hi = q.quantile(0.75).add_suffix("_q75")
    n = panel.groupby(["group", "rel_min"]).anchor_id.nunique().rename("n")
    return pd.concat([n, med, lo, hi], axis=1).reset_index()


# ----------------------------------------------------------------------------- tests
def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    o = np.argsort(pv)
    r = pv[o] * len(pv) / (np.arange(len(pv)) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    qq = np.empty(len(pv))
    qq[o] = np.minimum(r, 1.0)
    q[ok] = qq
    return q


def compare(ph: pd.DataFrame, a: str, b: str, stratum: str = "all", phases=tuple(PHASES)) -> pd.DataFrame:
    from scipy.stats import mannwhitneyu
    feats = ["px_bp", "imb", "vol_rel", "liq_with", "liq_against", "premium_bp", "block_amt", "fut_basis", "atm_iv",
             "skew", "opt_delta_flow", "opt_amt", "opt_block_amt"]
    z = ph if stratum == "all" else ph[ph.geometry.eq(stratum)]
    rows = []
    for p_ in phases:
        w = z[z.phase.eq(p_)]
        for f in feats:
            x = w.loc[w.group.eq(a), f].dropna().to_numpy()
            y = w.loc[w.group.eq(b), f].dropna().to_numpy()
            if len(x) < 10 or len(y) < 10 or (np.ptp(np.r_[x, y]) == 0):
                rows.append({"stratum": stratum, "A": a, "B": b, "phase": p_, "feature": f, "nA": len(x), "nB": len(y),
                             "medA": np.nan, "medB": np.nan, "rank_biserial": np.nan, "p": np.nan})
                continue
            u, pv = mannwhitneyu(x, y, alternative="two-sided")
            rows.append({"stratum": stratum, "A": a, "B": b, "phase": p_, "feature": f, "nA": len(x), "nB": len(y),
                         "medA": float(np.median(x)), "medB": float(np.median(y)),
                         "rank_biserial": float(2 * u / (len(x) * len(y)) - 1), "p": float(pv)})
    t = pd.DataFrame(rows)
    t["pre_entry"] = t.phase.isin(PRE_PHASES)
    t["q_pre"] = np.nan
    pre = t.pre_entry.to_numpy()
    t.loc[pre, "q_pre"] = bh(t.loc[pre, "p"].to_numpy())
    t["q_all"] = bh(t.p.to_numpy())
    return t


def decision(tests: pd.DataFrame, panel: pd.DataFrame, cov: pd.DataFrame, cfg: Config = Config()) -> dict:
    tv = tests[(tests.A == "TARGET") & (tests.B == "STOP")]
    cand = tv[tv.pre_entry & (tv.q_pre <= cfg.fdr_q)]
    sig_ctl = tests[(tests.A == "SIGNAL") & (tests.B == "CONTROL") & tests.pre_entry & (tests.q_pre <= cfg.fdr_q)]
    n = panel.groupby("group").anchor_id.nunique().to_dict() if len(panel) else {}
    return {"version": VERSION, "status": registration(cfg)["status"], "anchors_by_group": n,
            "days_with_option_file": int(cov.option_file.sum()) if len(cov) else 0,
            "days_with_future_file": int(cov.future_file.sum()) if len(cov) else 0,
            "pre_entry_TARGET_vs_STOP_candidates_q10": cand[["stratum", "phase", "feature", "medA", "medB",
                                                            "rank_biserial", "p", "q_pre"]].to_dict("records"),
            "pre_entry_SIGNAL_vs_CONTROL_differences_q10": int(len(sig_ctl)),
            "reading": ("Candidates are hypotheses for the untouched LA 022 holdout; post-entry differences are partly "
                        "mechanical outcome definitions and are not usable as entry filters.")}


# ----------------------------------------------------------------------------- synthetic
def synthetic_history(root: Path, days: list[str], currency: str = "BTC", seed: int = 0,
                      planted: list[tuple[int, int, float]] | None = None, with_options: bool = True) -> None:
    """Writes fake future/option day files. `planted` = [(entry_ms, sign, usd)] adds liquidations of positions ON the
    trade side (longs forced to sell into a long entry's support) in the 10 minutes before each entry: the
    'absorbed liquidation' pattern, measured by liq_with."""
    rng = np.random.default_rng(seed)
    px = 60000.0
    for d in days:
        t0 = int(pd.Timestamp(d, tz="UTC").value // 10**6)
        n = 1440 * 4
        ts = np.sort(t0 + rng.integers(0, 86_400_000, n))
        steps = rng.normal(0, 3e-4, n)
        p = px * np.exp(np.cumsum(steps))
        px = float(p[-1])
        dirn = np.where(rng.random(n) < 0.5, "buy", "sell")
        liq = np.array([None] * n, dtype=object)
        amt = rng.uniform(100, 5000, n)
        if planted:
            for (em, sg, usd) in planted:
                sel = np.flatnonzero((ts >= em - 10 * 60_000) & (ts < em))
                for i in sel[:5]:
                    liq[i] = "T"
                    dirn[i] = "sell" if sg > 0 else "buy"      # long entry: longs liquidated (forced sells)
                    amt[i] = usd
        perp = pd.DataFrame({"timestamp": pd.to_datetime(ts, unit="ms", utc=True),
                             "instrument_name": f"{currency}-PERPETUAL", "price": p, "amount": amt, "direction": dirn,
                             "mark_price": p * 1.0001, "index_price": p, "liquidation": pd.array(liq, dtype="string"),
                             "block_trade_id": pd.array([None] * n, dtype="string")})
        exp = pd.Timestamp(d, tz="UTC") + pd.Timedelta(days=30)
        fname = f"{currency}-{exp.day}{list(MONTHS)[exp.month - 1]}{exp.year % 100:02d}"
        fut = perp.iloc[::10].copy()
        fut["instrument_name"] = fname
        fut["price"] = fut["price"] * 1.004
        fp = Path(root) / "trades" / currency / "future" / d[:4]
        fp.mkdir(parents=True, exist_ok=True)
        pd.concat([perp, fut]).to_parquet(fp / f"{d}.parquet", index=False)
        if with_options:
            k = np.round(p[::8] / 1000) * 1000
            m = len(k)
            cp = np.where(rng.random(m) < 0.5, "C", "P")
            names = [f"{currency}-{exp.day}{list(MONTHS)[exp.month - 1]}{exp.year % 100:02d}-{int(x)}-{c}" for x, c in zip(k, cp)]
            opt = pd.DataFrame({"timestamp": perp.timestamp.iloc[::8].to_numpy(), "instrument_name": names,
                                "price": 0.01, "amount": rng.uniform(0.1, 5, m),
                                "direction": np.where(rng.random(m) < 0.5, "buy", "sell"),
                                "iv": 50 + rng.normal(0, 2, m), "index_price": p[::8],
                                "block_trade_id": pd.array([None] * m, dtype="string")})
            op = Path(root) / "trades" / currency / "option" / d[:4]
            op.mkdir(parents=True, exist_ok=True)
            opt.to_parquet(op / f"{d}.parquet", index=False)
