"""Regime Structure 03 — what distinguishes big-move days from normal days?

Pre-registration: discussions/Regime_Structure/007_claude_rs03_preregistration.md.

Data: Binance USD-M perpetual 1m klines (BTC, ETH, SOL, XRP) -> 5-minute bars.
Session = the 24 h ending at 08:00 UTC (the Deribit daily expiry), labelled by its end date.

Big-move day (per asset)
    sigma_hat = sqrt(mean RV^2 of the previous 7 sessions)       (causal daily vol forecast)
    z         = |log P_end - log P_start| / sigma_hat             (what a straddle held to expiry is paid on)
    BIG       = z >= q85 of z on the development window (frozen per asset)

Feature groups
    PRE    known at 08:00 (session start): previous session and its last 4 h, trailing 7/30-session context.
    EARLY  first 2 h of the session (08:00-10:00); usable for entries after 10:00. Tested against BIG_REST:
           |move 10:00 -> 08:00| / (sigma_hat sqrt(22/24)) >= its dev q85 (the early move itself is excluded).
    DURING whole-session descriptors (consequences, descriptive only, no tests).

Statistic per feature: AUC = P(feature on a BIG day > feature on a normal day), pooled over assets,
95% CI from a bootstrap over sessions (all assets of a session resampled together).
Rule: a PRE or EARLY feature is a CANDIDATE if its development-window CI excludes 0.5; a candidate is
CONFIRMED if its test-window CI at level 1 - 0.05 / n_candidates (per group) excludes 0.5 on the same side.
Clustering: lift = P(BIG_{s+1} | BIG_s) / P(BIG) and cross-asset co-occurrence, session bootstrap.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

VERSION = "big_move_days v1.0"
TEST_START = pd.Timestamp("2026-04-01", tz="UTC")
ASSETS = ("BTC", "ETH", "SOL", "XRP")
BARS = 288                      # 5m bars per session


@dataclass(frozen=True)
class Params:
    forecast_sessions: int = 7
    context_sessions: int = 30
    big_q: float = 0.85
    last_bars: int = 48          # last 4 h of the previous session
    early_bars: int = 24         # first 2 h of the session
    shock_k: float = 3.0
    n_boot: int = 2000
    seed: int = 20261013


def registration(p: Params | None = None) -> dict:
    p = p or Params()
    return {**asdict(p), "version": VERSION, "test_start": str(TEST_START),
            "session": "24h ending 08:00 UTC, labelled by end date",
            "big": "z=|session log return|/sigma_hat(7 prior sessions) >= dev q85 per asset",
            "big_rest": "EARLY target: |10:00->08:00 return|/(sigma_hat sqrt(22/24)) >= dev q85 per asset",
            "rule": "candidate: dev 95% CI of AUC excludes 0.5; confirmed: test CI at 1-0.05/n_candidates excludes 0.5, same side"}


# ---------------------------------------------------------------- bars and sessions

def session_bars(df: pd.DataFrame) -> dict:
    """5m bars aligned to sessions starting at 08:00 UTC. Arrays shaped (n_sessions, 288)."""
    d = df.copy()
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d = d.drop_duplicates("open_time").set_index("open_time").sort_index()
    first = d.index[0].floor("1D") + pd.Timedelta(hours=8)
    if first < d.index[0]:
        first += pd.Timedelta(days=1)
    last = d.index[-1].floor("1D") + pd.Timedelta(hours=8)          # last complete session ends here
    if last > d.index[-1] + pd.Timedelta(minutes=1):
        last -= pd.Timedelta(days=1)
    grid = pd.date_range(first, last - pd.Timedelta(minutes=1), freq="1min")
    d = d.reindex(grid)
    logc = np.log(d["close"].astype(float).to_numpy())
    logc = pd.Series(logc).ffill().to_numpy()
    vol = d["volume"].astype(float).fillna(0.0).to_numpy()
    tb = d["taker_buy_volume"].astype(float).fillna(0.0).to_numpy()
    trd = d["trades"].astype(float).fillna(0.0).to_numpy() if "trades" in d else np.zeros(len(d))
    n = len(grid) // (BARS * 5)
    m = n * BARS * 5
    lc5 = logc[4:m:5].reshape(n, BARS)                          # close of each 5m bar
    start = np.r_[logc[0], lc5.ravel()[:-1]].reshape(n, BARS)   # previous close (first bar: first minute close)
    r5 = lc5 - start
    agg = lambda x: x[:m].reshape(n * BARS, 5).sum(1).reshape(n, BARS)
    v5, d5, t5 = agg(vol), agg(2 * tb - vol), agg(trd)
    sess = (first + pd.to_timedelta(np.arange(n) + 1, unit="D")).normalize()
    return {"session": sess, "r5": np.nan_to_num(r5), "v5": v5, "d5": d5, "t5": t5, "lc5": lc5,
            "missing": int(np.isnan(np.log(d["close"].astype(float).to_numpy()))[:m].sum())}


def _trail(x: np.ndarray, k: int, fn) -> np.ndarray:
    """fn over the previous k values (excluding the current), NaN until k are available."""
    s = pd.Series(x)
    return fn(s.shift(1).rolling(k, min_periods=k)).to_numpy()


def session_table(b: dict, p: Params) -> tuple:
    """Per-session table plus hourly return and volume matrices (n_sessions, 24)."""
    r5, v5, d5, t5 = b["r5"], b["v5"], b["d5"], b["t5"]
    n = len(r5)
    ret = r5.sum(1)
    rv = np.sqrt((r5 ** 2).sum(1))
    vol = v5.sum(1)
    trd = t5.sum(1)
    sig = np.sqrt(_trail(rv ** 2, p.forecast_sessions, lambda r: r.mean()))
    s5 = sig / np.sqrt(BARS)
    z = np.abs(ret) / sig
    vol_med = _trail(vol, p.context_sessions, lambda r: r.median())
    trd_med = _trail(trd, p.context_sessions, lambda r: r.median())
    rv30 = np.sqrt(_trail(rv ** 2, p.context_sessions, lambda r: r.mean()))
    L, E = p.last_bars, p.early_bars
    last_rv = np.sqrt((r5[:, -L:] ** 2).sum(1))
    last_vol = v5[:, -L:].sum(1)
    early_rv = np.sqrt((r5[:, :E] ** 2).sum(1))
    early_vol = v5[:, :E].sum(1)
    prev = lambda x: np.r_[np.nan, x[:-1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        path = np.abs(r5).sum(1)
        cum = np.cumsum(r5, 1)
        hr = r5.reshape(n, 24, 12).sum(2)
        t = pd.DataFrame({
            "session": b["session"], "ret": ret, "rv": rv, "sigma_hat": sig, "z": z, "zrv": rv / sig,
            "z_rest": np.abs(r5[:, E:].sum(1)) / (sig * np.sqrt((BARS - E) / BARS)),
            "up": ret > 0,
            # PRE (known at 08:00)
            "pre_rv_prev_vs_fcst": prev(rv) / sig,
            "pre_rv7_vs_rv30": sig / rv30,
            "pre_vol_prev_rel": prev(vol) / vol_med,
            "pre_trades_prev_rel": prev(trd) / trd_med,
            "pre_ret_prev_z": prev(ret) / sig,
            "pre_absret_prev_z": np.abs(prev(ret)) / sig,
            "pre_er_prev": prev(np.abs(ret) / path),
            "pre_flow_prev": prev(d5.sum(1) / vol),
            "pre_shocks_prev": prev((np.abs(r5) >= p.shock_k * s5[:, None]).sum(1).astype(float)),
            "pre_last4h_rv_rel": prev(last_rv) / _trail(last_rv, p.context_sessions, lambda r: r.median()),
            "pre_last4h_vol_rel": prev(last_vol) / _trail(last_vol, p.context_sessions, lambda r: r.median()),
            "pre_last4h_ret_z": prev(r5[:, -L:].sum(1)) / sig,
            "pre_dist_7d_high": (np.r_[np.nan, b["lc5"][:-1, -1]] - _trail(b["lc5"].max(1), 7, lambda r: r.max())) / sig,
            "pre_dist_7d_low": (np.r_[np.nan, b["lc5"][:-1, -1]] - _trail(b["lc5"].min(1), 7, lambda r: r.min())) / sig,
            "pre_weekend": b["session"].dayofweek.isin([5, 6]).astype(float),   # session ending Sat/Sun 08:00
            "pre_monday": (b["session"].dayofweek == 0).astype(float),
            # EARLY (08:00-10:00)
            "early_rv_rel": early_rv / _trail(early_rv, p.context_sessions, lambda r: r.median()),
            "early_vol_rel": early_vol / _trail(early_vol, p.context_sessions, lambda r: r.median()),
            "early_absret_z": np.abs(r5[:, :E].sum(1)) / sig,
            "early_flow": d5[:, :E].sum(1) / early_vol,
            # DURING (descriptive)
            "dur_er": np.abs(ret) / path,
            "dur_vol_rel": vol / vol_med,
            "dur_trades_rel": trd / trd_med,
            "dur_flow": d5.sum(1) / vol,
            "dur_flow_aligned": np.sign(ret) * d5.sum(1) / vol,
            "dur_shocks": (np.abs(r5) >= p.shock_k * s5[:, None]).sum(1).astype(float),
            "dur_max_excursion_z": np.abs(cum).max(1) / sig,
            "dur_giveback": (np.abs(cum).max(1) - np.abs(ret)) / sig,
            "dur_top_hour_share": np.abs(hr).max(1) / np.abs(hr).sum(1),
            "dur_top_hour": np.abs(hr).argmax(1).astype(float),     # hours after 08:00 UTC
        })
    t["window"] = np.where(t["session"] >= TEST_START.normalize(), "test", "dev")
    return t, hr, v5.reshape(n, 24, 12).sum(2)


def label_big(t: pd.DataFrame, p: Params) -> tuple[pd.DataFrame, float]:
    """BIG (whole session) for PRE/DURING; BIG_REST (10:00 -> 08:00) for EARLY features."""
    t = t.copy()
    thr = np.nan
    for col, out in (("z", "big"), ("z_rest", "big_rest")):
        ok = np.isfinite(t[col])
        q = float(t.loc[ok & (t["window"] == "dev"), col].quantile(p.big_q))
        t[out] = np.where(ok, t[col] >= q, np.nan)
        if col == "z":
            thr = q
    return t, thr


# ---------------------------------------------------------------- statistics

def auc(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    """P(x on y=1 > x on y=0) + 0.5 ties, with optional integer weights (bootstrap multiplicities)."""
    ok = np.isfinite(x)
    x, y = x[ok], y[ok].astype(bool)
    w = np.ones(len(x)) if w is None else w[ok]
    if w[y].sum() == 0 or w[~y].sum() == 0:
        return np.nan
    order = np.argsort(x, kind="mergesort")
    xs, ys, ws = x[order], y[order], w[order]
    # average-rank with weights, handling ties
    uniq, start = np.unique(xs, return_index=True)
    wsum = np.add.reduceat(ws, start)
    cum = np.cumsum(wsum) - wsum
    mid = cum + (wsum + 1) / 2.0                                  # weighted mid-rank per distinct value
    ranks = np.repeat(mid, np.diff(np.r_[start, len(xs)]))
    n1, n0 = ws[ys].sum(), ws[~ys].sum()
    r1 = (ranks[ys] * ws[ys]).sum()
    return float((r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def boot_auc(df: pd.DataFrame, col: str, p: Params, level: float = 0.95, target: str = "big") -> dict:
    d = df[np.isfinite(df[col]) & df[target].notna()]
    if d[target].sum() < 10:
        return {"n": int(len(d)), "auc": np.nan, "lo": np.nan, "hi": np.nan}
    x, y = d[col].to_numpy(float), d[target].to_numpy(bool)
    days = d["session"].to_numpy()
    u, inv = np.unique(days, return_inverse=True)
    rng = np.random.default_rng(p.seed + sum(map(ord, col)))
    est = auc(x, y)
    bs = np.empty(p.n_boot)
    for i in range(p.n_boot):
        cnt = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
        bs[i] = auc(x, y, cnt[inv].astype(float))
    a = (1 - level) / 2
    lo, hi = np.nanquantile(bs, [a, 1 - a])
    return {"n": int(len(d)), "n_big": int(y.sum()), "auc": est, "lo": float(lo), "hi": float(hi),
            "mean_big": float(np.nanmean(x[y])), "mean_normal": float(np.nanmean(x[~y])),
            "median_big": float(np.nanmedian(x[y])), "median_normal": float(np.nanmedian(x[~y]))}


def screen(t: pd.DataFrame, cols: list, p: Params, target: str = "big") -> pd.DataFrame:
    """Dev screen, then test confirmation at a Bonferroni level over the dev candidates."""
    dev = pd.DataFrame([{"feature": c, **boot_auc(t[t["window"] == "dev"], c, p, target=target)} for c in cols])
    dev["candidate"] = (dev["lo"] > 0.5) | (dev["hi"] < 0.5)
    k = int(dev["candidate"].sum())
    level = 1 - 0.05 / max(k, 1)
    tst = pd.DataFrame([{"feature": c, **boot_auc(t[t["window"] == "test"], c, p, level=level, target=target)} for c in cols])
    out = dev.merge(tst, on="feature", suffixes=("_dev", "_test"))
    side_dev = np.sign(out["auc_dev"] - 0.5)
    out["test_level"] = level
    out["target"] = target
    out["confirmed"] = out["candidate"] & (((out["lo_test"] > 0.5) & (side_dev > 0)) |
                                           ((out["hi_test"] < 0.5) & (side_dev < 0)))
    return out


def clustering(t: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Lift of BIG next session given BIG today, per window, pooled; and cross-asset co-occurrence."""
    rows = []
    rng = np.random.default_rng(p.seed)
    for w in ("dev", "test"):
        d = t[t["window"] == w].sort_values(["asset", "session"])
        d = d.assign(big_next=d.groupby("asset")["big"].shift(-1)).dropna(subset=["big", "big_next"])
        u, inv = np.unique(d["session"].to_numpy(), return_inverse=True)
        b, bn = d["big"].to_numpy(bool), d["big_next"].to_numpy(bool)

        def lift(wt):
            pb = (wt * bn).sum() / wt.sum()
            return ((wt * (b & bn)).sum() / (wt * b).sum()) / pb
        est = lift(np.ones(len(d)))
        bs = [lift(np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv].astype(float)) for _ in range(p.n_boot)]
        rows.append({"window": w, "stat": "P(big next | big) / P(big)", "value": est,
                     "lo": float(np.nanquantile(bs, 0.025)), "hi": float(np.nanquantile(bs, 0.975)), "n": int(len(d))})
        g = t[t["window"] == w].dropna(subset=["big"]).groupby("session")["big"].agg(["sum", "count"])
        g = g[g["count"] == g["count"].max()]
        pb = t[t["window"] == w]["big"].mean()
        k = int(g["count"].max()) if len(g) else 0
        exp_all = pb ** k
        rows.append({"window": w, "stat": f"share of sessions with all {k} assets big (independent: {exp_all:.4f})",
                     "value": float((g["sum"] == k).mean()), "lo": np.nan, "hi": np.nan, "n": int(len(g))})
        rows.append({"window": w, "stat": "share of big days that are big in >= 3 assets",
                     "value": float(t[t["window"] == w].merge(g.reset_index()[["session", "sum"]], on="session")
                                     .query("big == 1")["sum"].ge(3).mean()), "lo": np.nan, "hi": np.nan, "n": int(len(g))})
    return pd.DataFrame(rows)


def hourly_profile(H: np.ndarray, V: np.ndarray, t: pd.DataFrame) -> pd.DataFrame:
    """Mean |hourly return| / (sigma_hat / sqrt 24) and hourly volume / that hour's 30-session median,
    by hour after 08:00 UTC, BIG vs normal."""
    ok = t["big"].notna().to_numpy() & np.isfinite(t["sigma_hat"].to_numpy())
    s_h = t["sigma_hat"].to_numpy()[:, None] / np.sqrt(24)
    a = np.abs(H) / s_h
    vmed = pd.DataFrame(V).shift(1).rolling(30, min_periods=30).median().to_numpy()
    vr = V / vmed
    rows = []
    for name, m in (("big", ok & (t["big"].to_numpy() == 1)), ("normal", ok & (t["big"].to_numpy() == 0))):
        for h in range(24):
            rows.append({"set": name, "hour_after_0800": h, "utc_hour": (8 + h) % 24,
                         "abs_ret_rel": float(np.nanmean(a[m, h])), "volume_rel": float(np.nanmedian(vr[m, h]))})
    return pd.DataFrame(rows)


def run_asset(df: pd.DataFrame, asset: str, p: Params) -> dict:
    b = session_bars(df)
    t, H, V = session_table(b, p)
    t, thr = label_big(t, p)
    t["asset"] = asset
    return {"table": t, "H": H, "V": V, "threshold": thr, "missing_minutes": b["missing"]}


PRE = [c for c in ("pre_rv_prev_vs_fcst", "pre_rv7_vs_rv30", "pre_vol_prev_rel", "pre_trades_prev_rel", "pre_ret_prev_z",
                   "pre_absret_prev_z", "pre_er_prev", "pre_flow_prev", "pre_shocks_prev", "pre_last4h_rv_rel",
                   "pre_last4h_vol_rel", "pre_last4h_ret_z", "pre_dist_7d_high", "pre_dist_7d_low", "pre_weekend",
                   "pre_monday")]
EARLY = ["early_rv_rel", "early_vol_rel", "early_absret_z", "early_flow"]
DURING = ["dur_er", "dur_vol_rel", "dur_trades_rel", "dur_flow", "dur_flow_aligned", "dur_shocks",
          "dur_max_excursion_z", "dur_giveback", "dur_top_hour_share", "dur_top_hour"]


# ---------------------------------------------------------------- synthetic

def synthetic_klines(n_days: int = 400, seed: int = 0, start: str = "2025-03-01", planted: bool = True) -> pd.DataFrame:
    """1m klines; if planted, a big-move day is preceded by a quiet, low-volume last 4 h of the previous session."""
    rng = np.random.default_rng(seed)
    n = n_days * 1440
    t = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    day_scale = np.exp(rng.normal(0, 0.25, n_days + 2))
    big = rng.random(n_days + 2) < 0.15
    day_scale[big] *= 2.2
    sess = ((t - (t[0].floor("1D") + pd.Timedelta(hours=8))) // pd.Timedelta(days=1)).to_numpy() + 1
    sess = np.clip(sess, 0, n_days + 1)
    sig = 5e-4 * day_scale[sess]
    vol = rng.gamma(2.0, 50.0, n) * day_scale[sess]
    if planted:
        mins_to_end = ((t.hour.to_numpy() - 8) % 24) * 60 + t.minute.to_numpy()
        last4 = mins_to_end >= 20 * 60
        nxt_big = big[np.clip(sess + 1, 0, n_days + 1)]
        vol = np.where(last4 & nxt_big, vol * 0.4, vol)
    r = sig * rng.standard_normal(n)
    c = 100 * np.exp(np.cumsum(r))
    tb = vol * np.clip(0.5 + 0.3 * np.tanh(r / sig), 0.05, 0.95)
    return pd.DataFrame({"open_time": t, "close": c, "volume": vol, "taker_buy_volume": tb,
                         "trades": np.round(vol / 3)})
