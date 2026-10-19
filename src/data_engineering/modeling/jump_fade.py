"""
Jump Fade branch, JF 01: a passive ±kσ bracket around the price at each 5-minute boundary, with a passive
take-profit back at that price, a price stop and a time stop. Does the fade have positive expectancy on the
perpetual, before and after costs, against a sign-flip null?

Pre-registration: discussions/Jump_Fade/001_claude_branch_preregistration.md

Simulation on 1-MINUTE Binance USD-M perpetual OHLC (5-minute candles cannot order fill vs take-profit).
Per asset, one position at a time:
    decision at each 5m boundary t (minute END on a multiple of 5) while flat:
        P0 = close at t;  sigma_t = sqrt(EWMA_{hl=288 bars}(r5^2)) using 5m log returns ending <= t
        buy limit  L_b = P0 * exp(-K sigma_t),  sell limit L_s = P0 * exp(+K sigma_t), live for minutes (t, t+5]
        fill = strict trade-through (low < L_b / high > L_s); the other order is cancelled at once.
        both sides cross in the same minute -> both single-side outcomes are simulated, the WORSE is booked.
    position (side s, entry E = limit price):
        take-profit  TP = P0 (passive, maker): filled when price trades strictly through it (high > TP for a long)
        price stop   SL = P0 * exp(-s * M sigma_t), taker; gap through -> exit at the minute's open
        time stop    H minutes after the fill minute, taker at that minute's close
        conservative ordering: in the fill minute only the stop can trigger; in later minutes stop before TP.
    after an exit the next bracket is placed at the next 5m boundary at or after the exit minute.
Costs per side: maker / taker fee by venue + 1 bp slippage on taker exits.
Null: each 1-minute log return gets a random sign (shared across assets by timestamp); minute OHLC mirrored.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

VERSION = "jump_fade 1.0.0"

ASSETS = ("BTC", "ETH", "SOL", "XRP")
K = 1.5
M = 3.0
H = 60
HALFLIFE_5M = 288
VENUES = {"deribit": {"maker": 0.0, "taker": 5.0}, "binance": {"maker": 2.0, "taker": 5.0}}  # bp per side
PRIMARY_VENUE = "deribit"
SLIP_BP = 1.0
DEV_END = pd.Timestamp("2026-01-01", tz="UTC")
DATA_START = pd.Timestamp("2024-10-01", tz="UTC")
WARMUP_DAYS = 7
BURST_HL_FAST = 12           # 1 hour of 5m bars
BURST_Q = 0.90               # top decile of sigma_fast / sigma (thresholds from DEV, per asset)
JUMP_LOOKBACK_MIN = 15
JUMP_Z = 4.0                 # |r_1m| > 4 sigma_t / sqrt(5)
N_NULL = 199
BOOT_B = 2000
SEED = 20261019
ALPHA = 0.05
SPLITS = ("all", "burst", "recent_jump")
GRID_K = (1.0, 1.5, 2.0, 3.0)
GRID_M = (2.0, 3.0, 4.0)
GRID_H = (15, 30, 60, 120)


@dataclass(frozen=True)
class Params:
    k: float = K
    m: float = M
    h: int = H


def registration() -> dict:
    return {"version": VERSION, "assets": list(ASSETS), "K": K, "M": M, "H_min": H, "sigma_halflife_5m": HALFLIFE_5M,
            "venues_bp": VENUES, "primary_venue": PRIMARY_VENUE, "slip_bp_taker": SLIP_BP, "dev_end": str(DEV_END),
            "data_start": str(DATA_START), "burst": {"hl_fast_5m": BURST_HL_FAST, "quantile": BURST_Q},
            "recent_jump": {"lookback_min": JUMP_LOOKBACK_MIN, "z": JUMP_Z}, "n_null": N_NULL, "boot_B": BOOT_B,
            "grid": {"K": GRID_K, "M": GRID_M, "H": GRID_H}, "alpha": ALPHA,
            "source": "Binance USD-M perpetual 1m OHLC as the price path; costs at Deribit (primary) and Binance VIP0"}


# ----------------------------------------------------------------------------- data
def load_1m(path) -> pd.DataFrame:
    d = pd.read_parquet(path, columns=["open_time", "open", "high", "low", "close"])
    return dense_1m(d)


def dense_1m(d: pd.DataFrame) -> pd.DataFrame:
    """Dense minute grid indexed by minute END; missing minutes are NaN."""
    x = d.copy()
    x["t"] = pd.to_datetime(x["open_time"], utc=True) + pd.Timedelta(minutes=1)
    x = x.drop_duplicates("t", keep="last").set_index("t").sort_index()
    for c in ("open", "high", "low", "close"):
        x[c] = pd.to_numeric(x[c], errors="coerce").astype(float)
    x = x[(x["low"] > 0) & (x["high"] >= x["low"])]
    g = pd.date_range(x.index.min(), x.index.max(), freq="1min")
    return x[["open", "high", "low", "close"]].reindex(g)


def sign_flip(minutes: dict, seed: int) -> dict:
    """Random sign per minute timestamp, shared across assets; OHLC mirrored around the previous close."""
    allt = np.unique(np.concatenate([m.index.as_unit("ns").asi8 for m in minutes.values()]))
    rng = np.random.default_rng(seed)
    sg = np.where(rng.random(len(allt)) < 0.5, -1.0, 1.0)
    out = {}
    for a, m in minutes.items():
        s = sg[np.searchsorted(allt, m.index.as_unit("ns").asi8)]
        lo, lh, ll, lc = (np.log(m[c].to_numpy(float)) for c in ("open", "high", "low", "close"))
        prev = np.r_[lo[0], lc[:-1]]
        prev = np.where(np.isfinite(prev), prev, lo)
        r = np.where(np.isfinite(lc), lc - prev, 0.0)
        newc = lo[0] + np.cumsum(s * r) if np.isfinite(lo[0]) else np.cumsum(s * r)
        newprev = np.r_[newc[0] - s[0] * r[0], newc[:-1]]
        ro, rh, rl = lo - prev, lh - prev, ll - prev
        hi_rel = np.where(s > 0, rh, -rl)
        lo_rel = np.where(s > 0, rl, -rh)
        nanm = ~np.isfinite(lc)
        o2 = np.exp(newprev + s * ro)
        h2 = np.exp(newprev + hi_rel)
        l2 = np.exp(newprev + lo_rel)
        c2 = np.exp(newc)
        for arr in (o2, h2, l2, c2):
            arr[nanm] = np.nan
        out[a] = pd.DataFrame({"open": o2, "high": h2, "low": l2, "close": c2}, index=m.index)
    return out


def decision_inputs(m: pd.DataFrame, burst_thr: float | None = None) -> dict:
    """Per-minute arrays and the decision-time sigma, burst ratio and recent-jump flag (all causal)."""
    t = m.index
    is_dec = (t.minute % 5 == 0)
    c = m["close"].to_numpy(float)
    c5 = pd.Series(np.where(is_dec, c, np.nan), index=t)
    c5 = c5[is_dec]
    r5 = np.log(c5).diff()
    alpha = 1 - 0.5 ** (1 / HALFLIFE_5M)
    af = 1 - 0.5 ** (1 / BURST_HL_FAST)
    sq = (r5 ** 2)
    v = sq.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
    vf = sq.ewm(alpha=af, adjust=False, ignore_na=True).mean()
    sig5 = np.sqrt(v).where(r5.notna())
    ratio = np.sqrt(vf / v).where(r5.notna())
    warm = t[0] + pd.Timedelta(days=WARMUP_DAYS)
    sig5[sig5.index < warm] = np.nan
    sig = np.full(len(t), np.nan)
    rat = np.full(len(t), np.nan)
    pos = np.flatnonzero(is_dec)
    sig[pos] = sig5.to_numpy(float)
    rat[pos] = ratio.to_numpy(float)
    # recent jump: any 1m |log return| > JUMP_Z * sigma / sqrt(5) within the last JUMP_LOOKBACK_MIN minutes
    lc = np.log(c)
    r1 = np.r_[np.nan, np.diff(lc)]
    sig_ff = pd.Series(sig).ffill().to_numpy()
    big = np.abs(r1) > JUMP_Z * sig_ff / math.sqrt(5)
    cb = np.r_[0, np.cumsum(np.where(np.isfinite(r1), big, False))]
    i = np.arange(len(t))
    lo = np.maximum(i + 1 - JUMP_LOOKBACK_MIN, 0)
    recent = (cb[i + 1] - cb[lo]) > 0
    return {"t": t, "o": m["open"].to_numpy(float), "h": m["high"].to_numpy(float), "l": m["low"].to_numpy(float),
            "c": c, "sig": sig, "ratio": rat, "recent": recent}


# ----------------------------------------------------------------------------- simulator
try:
    from numba import njit as _njit
    _jit = _njit(cache=True)
except Exception:  # pragma: no cover
    def _jit(f):
        return f


@_jit
def _position(o, h, l, c, j, s, E, TP, SL, H):
    """Simulate a position opened in minute j. Returns (exit_index, exit_price, type) with
    type 0 = take-profit (maker), 1 = price stop (taker), 2 = time stop (taker)."""
    n = len(c)
    # fill minute: only the stop can trigger
    if s > 0:
        if l[j] <= SL:
            return j, SL, 1
    else:
        if h[j] >= SL:
            return j, SL, 1
    last = j
    for k in range(j + 1, min(j + H, n - 1) + 1):
        if not np.isfinite(c[k]):
            continue
        last = k
        if s > 0:
            if o[k] <= SL:
                return k, o[k], 1
            if l[k] <= SL:
                return k, SL, 1
            if h[k] > TP:
                return k, TP, 0
        else:
            if o[k] >= SL:
                return k, o[k], 1
            if h[k] >= SL:
                return k, SL, 1
            if l[k] < TP:
                return k, TP, 0
    return last, c[last], 2


@_jit
def simulate_kernel(o, h, l, c, sig, allow, K, M, H):
    """Returns arrays of trades: decision idx, fill idx, exit idx, side, entry, exit, type, ambiguous flag."""
    n = len(c)
    cap = n // 5 + 1
    dec = np.empty(cap, np.int64)
    fil = np.empty(cap, np.int64)
    ext = np.empty(cap, np.int64)
    side = np.empty(cap, np.int64)
    ent = np.empty(cap)
    exi = np.empty(cap)
    typ = np.empty(cap, np.int64)
    amb = np.empty(cap, np.int64)
    nt = 0
    i = 0
    while i < n - 6:
        if not (np.isfinite(sig[i]) and allow[i] and np.isfinite(c[i])):
            i += 1
            continue
        P0 = c[i]
        Lb = P0 * math.exp(-K * sig[i])
        Ls = P0 * math.exp(K * sig[i])
        filled = False
        for j in range(i + 1, i + 6):
            if not np.isfinite(c[j]):
                continue
            bf = l[j] < Lb
            sf = h[j] > Ls
            if not (bf or sf):
                continue
            a = 0
            if bf and sf:
                a = 1
                k1, x1, t1 = _position(o, h, l, c, j, 1, Lb, P0, P0 * math.exp(-M * sig[i]), H)
                k2, x2, t2 = _position(o, h, l, c, j, -1, Ls, P0, P0 * math.exp(M * sig[i]), H)
                g1 = x1 / Lb - 1.0
                g2 = -(x2 / Ls - 1.0)
                if g1 <= g2:
                    s, E, kx, xx, tx = 1, Lb, k1, x1, t1
                else:
                    s, E, kx, xx, tx = -1, Ls, k2, x2, t2
            elif bf:
                s, E = 1, Lb
                kx, xx, tx = _position(o, h, l, c, j, 1, Lb, P0, P0 * math.exp(-M * sig[i]), H)
            else:
                s, E = -1, Ls
                kx, xx, tx = _position(o, h, l, c, j, -1, Ls, P0, P0 * math.exp(M * sig[i]), H)
            dec[nt] = i
            fil[nt] = j
            ext[nt] = kx
            side[nt] = s
            ent[nt] = E
            exi[nt] = xx
            typ[nt] = tx
            amb[nt] = a
            nt += 1
            filled = True
            i = kx
            break
        if not filled:
            i += 5
        else:
            # next decision: first boundary (index with sig defined) strictly after... at or after exit
            i += 0
            while i < n and not np.isfinite(sig[i]):
                i += 1
    return dec[:nt], fil[:nt], ext[:nt], side[:nt], ent[:nt], exi[:nt], typ[:nt], amb[:nt]


def simulate(inp: dict, allow: np.ndarray, p: Params = Params()) -> pd.DataFrame:
    d, f, e, s, E, X, ty, am = simulate_kernel(inp["o"], inp["h"], inp["l"], inp["c"], inp["sig"],
                                               allow.astype(np.bool_), p.k, p.m, p.h)
    t = inp["t"]
    tr = pd.DataFrame({"decision": t[d], "fill": t[f], "exit": t[e], "side": s, "entry": E, "exit_px": X,
                       "type": ty, "ambiguous": am, "hold_min": (e - f).astype(int),
                       "sigma": inp["sig"][d], "ratio": inp["ratio"][d], "recent": inp["recent"][d]})
    tr["gross"] = tr["side"] * (tr["exit_px"] / tr["entry"] - 1.0)
    return tr


def add_costs(tr: pd.DataFrame, venue: str) -> pd.DataFrame:
    v = VENUES[venue]
    x = tr.copy()
    exit_bp = np.where(x["type"].to_numpy() == 0, v["maker"], v["taker"] + SLIP_BP)
    x["cost"] = (v["maker"] + exit_bp) / 1e4
    x["net"] = x["gross"] - x["cost"]
    return x


def allow_mask(inp: dict, split: str, burst_thr: float) -> np.ndarray:
    n = len(inp["c"])
    base = np.isfinite(inp["sig"])
    if split == "all":
        return base
    if split == "burst":
        return base & (np.nan_to_num(inp["ratio"]) >= burst_thr)
    if split == "recent_jump":
        return base & inp["recent"]
    raise ValueError(split)


def burst_threshold(inp: dict) -> float:
    r = pd.Series(inp["ratio"], index=inp["t"])
    r = r[(r.index < DEV_END) & r.notna()]
    return float(r.quantile(BURST_Q)) if len(r) else float("inf")


# ----------------------------------------------------------------------------- statistics
def pooled_gross_bp(trades: dict, period: str) -> float:
    g = []
    for tr in trades.values():
        x = tr[_period_mask(tr, period)]
        g.append(x["gross"].to_numpy())
    g = np.concatenate(g) if g else np.array([])
    return float(1e4 * g.mean()) if len(g) else float("nan")


def _period_mask(tr: pd.DataFrame, period: str) -> np.ndarray:
    if period == "dev":
        return (tr["decision"] < DEV_END).to_numpy()
    if period == "test":
        return (tr["decision"] >= DEV_END).to_numpy()
    return np.ones(len(tr), bool)


def daily_net(trades: dict, period: str, assets) -> pd.Series:
    """Equal-weight portfolio of assets: per day, mean over assets of summed net returns (1 unit per trade)."""
    cols = {}
    for a in assets:
        x = trades[a][_period_mask(trades[a], period)]
        day = pd.to_datetime(x["exit"]).dt.floor("D")
        cols[a] = x.groupby(day)["net"].sum()
    df = pd.DataFrame(cols)
    if df.empty:
        return pd.Series(dtype=float)
    idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    return df.reindex(idx).fillna(0.0).mean(axis=1)


def boot_mean(x: np.ndarray, B: int = BOOT_B, seed: int = SEED) -> tuple:
    x = np.asarray(x, float)
    if len(x) < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(B, len(x)), replace=True).mean(1)
    return float(np.quantile(m, ALPHA / 2)), float(np.quantile(m, 1 - ALPHA / 2))


def summary(trades: dict, split: str, period: str, venue: str) -> dict:
    rows = []
    for a, tr in trades.items():
        x = add_costs(tr[_period_mask(tr, period)], venue)
        rows.append(x.assign(asset=a))
    z = pd.concat(rows) if rows else pd.DataFrame()
    dn = daily_net({a: add_costs(t, venue) for a, t in trades.items()}, period, list(trades))
    lo, hi = boot_mean(dn.to_numpy())
    per_asset = z.groupby("asset").agg(trades=("gross", "size"), gross_bp=("gross", lambda s: 1e4 * s.mean()),
                                       net_bp=("net", lambda s: 1e4 * s.mean()), net_total=("net", "sum"))
    out = {"split": split, "period": period, "venue": venue, "trades": int(len(z)),
           "trades_per_day": float(len(z) / max(len(dn), 1)),
           "gross_bp": float(1e4 * z["gross"].mean()) if len(z) else np.nan,
           "net_bp": float(1e4 * z["net"].mean()) if len(z) else np.nan,
           "hit_tp": float((z["type"] == 0).mean()) if len(z) else np.nan,
           "share_stop": float((z["type"] == 1).mean()) if len(z) else np.nan,
           "share_time": float((z["type"] == 2).mean()) if len(z) else np.nan,
           "ambiguous": float(z["ambiguous"].mean()) if len(z) else np.nan,
           "mean_win_bp": float(1e4 * z.loc[z["gross"] > 0, "gross"].mean()) if (z["gross"] > 0).any() else np.nan,
           "mean_loss_bp": float(1e4 * z.loc[z["gross"] <= 0, "gross"].mean()) if (z["gross"] <= 0).any() else np.nan,
           "worst_trade_bp": float(1e4 * z["gross"].min()) if len(z) else np.nan,
           "net_bp_day": float(1e4 * dn.mean()) if len(dn) else np.nan, "net_bp_day_lo": 1e4 * lo,
           "net_bp_day_hi": 1e4 * hi,
           "max_dd_bp": float(1e4 * np.max(np.maximum.accumulate(np.r_[0, dn.cumsum()]) - np.r_[0, dn.cumsum()]))
           if len(dn) else np.nan,
           "assets_gross_pos": int((per_asset["gross_bp"] > 0).sum()),
           "assets_net_pos": int((per_asset["net_total"] > 0).sum())}
    return out, per_asset.reset_index().assign(split=split, period=period, venue=venue)


def run_split(inputs: dict, split: str, thr: dict, p: Params = Params()) -> dict:
    return {a: simulate(inp, allow_mask(inp, split, thr[a]), p) for a, inp in inputs.items()}


def _null_one(args):
    minutes, seed, splits, p = args
    fl = sign_flip(minutes, seed)
    inp = {a: decision_inputs(m) for a, m in fl.items()}
    thr = {a: burst_threshold(i) for a, i in inp.items()}
    out = {"seed": seed}
    for sp in splits:
        tr = run_split(inp, sp, thr, p)
        for per in ("dev", "test"):
            out[f"{sp}_{per}"] = pooled_gross_bp(tr, per)
    return out


_NULL_MIN = {}


def _init(m):
    global _NULL_MIN
    _NULL_MIN = m


def _null_job(args):
    seed, splits, p = args
    return _null_one((_NULL_MIN, seed, splits, p))


def run_null(minutes: dict, n: int, splits=SPLITS, p: Params = Params(), n_jobs: int = 1, progress=None) -> pd.DataFrame:
    args = [(SEED + 1000 + i, tuple(splits), p) for i in range(n)]
    out = []
    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init, initargs=(minutes,)) as ex:
            for i, r in enumerate(ex.map(_null_job, args)):
                out.append(r)
                if progress:
                    progress(i + 1, n)
    else:
        _init(minutes)
        for i, a in enumerate(args):
            out.append(_null_job(a))
            if progress:
                progress(i + 1, n)
    return pd.DataFrame(out)


def verdict(sumrows: pd.DataFrame, null: pd.DataFrame, real_gross: dict) -> dict:
    """Primary: split 'all', TEST. G1 gross reversion vs null; G2 net (primary venue)."""
    out = {}
    for sp in SPLITS:
        key = f"{sp}_test"
        g = real_gross[key]
        nd = null[key].dropna()
        pval = (1 + int((nd >= g).sum())) / (len(nd) + 1)
        r = sumrows[(sumrows.split == sp) & (sumrows.period == "test") & (sumrows.venue == PRIMARY_VENUE)].iloc[0]
        alpha = ALPHA if sp == "all" else ALPHA / 2
        g1 = bool(pval <= alpha and r["assets_gross_pos"] >= 3 and g > 0)
        g2 = bool(r["net_bp_day_lo"] > 0 and r["assets_net_pos"] >= 3)
        out[sp] = {"gross_bp": g, "null_median_bp": float(nd.median()), "null_p95_bp": float(nd.quantile(0.95)),
                   "p": pval, "alpha": alpha, "G1_reversion": g1, "G2_net_edge": g2,
                   "verdict": "FADE_EDGE" if (g1 and g2) else ("REVERSION_NOT_TRADEABLE" if g1 else "NO_REVERSION")}
    out["branch_verdict"] = out["all"]["verdict"] if not any(out[s]["verdict"] == "FADE_EDGE" for s in SPLITS) \
        else "FADE_EDGE:" + ",".join(s for s in SPLITS if out[s]["verdict"] == "FADE_EDGE")
    return out


# ----------------------------------------------------------------------------- synthetic
def synthetic_minutes(n_days: int = 500, assets=ASSETS, seed: int = 0, revert: float = 0.0,
                      start: str = "2024-10-01") -> dict:
    """1m OHLC random walk with Student-t shocks; `revert` > 0 makes each minute's return partially undo the
    previous 5 minutes (mean reversion), 0 = martingale."""
    rng = np.random.default_rng(seed)
    n = n_days * 1440
    t = pd.date_range(pd.Timestamp(start, tz="UTC") + pd.Timedelta(minutes=1), periods=n, freq="1min")
    common = rng.standard_t(4, n)
    out = {}
    for j, a in enumerate(assets):
        e = (0.7 * common + 0.7 * rng.standard_t(4, n)) * 4e-4 * (1 + 0.3 * j)
        r = e.copy()
        if revert:
            cs = np.cumsum(e)
            past5 = cs - np.r_[np.zeros(5), cs[:-5]]
            r[1:] -= revert * past5[:-1] / 5.0
        lc = np.log(100.0 * (j + 1)) + np.cumsum(r)
        prev = np.r_[lc[0], lc[:-1]]
        w = np.abs(rng.normal(0, 2e-4, n))
        out[a] = pd.DataFrame({"open": np.exp(prev), "high": np.exp(np.maximum(lc, prev) + w),
                               "low": np.exp(np.minimum(lc, prev) - w), "close": np.exp(lc)}, index=t)
    return out


def selftest() -> str:
    """Hand-built path: bracket buy fills, take-profit at P0 two minutes later."""
    t = pd.date_range("2025-01-01 00:01", periods=40, freq="1min", tz="UTC")
    c = np.full(40, 100.0)
    o, h, l = c.copy(), c + 0.01, c - 0.01
    sig = np.full(40, np.nan)
    sig[[4, 9, 14, 19, 24, 29, 34]] = 0.01               # boundaries (minute ends 00:05, 00:10, ...)
    l[5], c[5] = 98.0, 98.6                                # minute 00:06 trades through 100*exp(-0.015)=98.51
    o[6], h[6], l[6], c[6] = 98.6, 99.5, 98.55, 99.4
    o[7], h[7], l[7], c[7] = 99.4, 100.2, 99.3, 100.1     # trades through TP=100
    d, f, e, s, E, X, ty, am = simulate_kernel(o, h, l, c, sig, np.ones(40, np.bool_), 1.5, 3.0, 60)
    assert len(d) >= 1 and d[0] == 4 and f[0] == 5 and s[0] == 1 and e[0] == 7 and ty[0] == 0
    assert abs(E[0] - 100 * math.exp(-0.015)) < 1e-9 and abs(X[0] - 100.0) < 1e-12
    return f"{VERSION}: simulator selftest ok"
