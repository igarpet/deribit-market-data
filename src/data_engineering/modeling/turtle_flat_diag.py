"""Turtle Breakout 18: diagnostic - is BTC mean-reverting on the days the turtle is out of the market?

Protocol: discussions/Turtle_Breakout/020_claude_tb18_preregistration.md (fixed before any real-data output).

Data: Yahoo BTC-USD daily (TB 17 file), 2014-09-17 .. last bar; close-to-close log returns r_t.
Turtle state: the TB 17 FULL long-only run (TB 01 rules). Day t is IN if a long is held at its close
(entry date <= t < exit date), FLAT otherwise (after the 30-day warm-up).
Statistics, conditioned on the state of day t (the information a strategy has at t):
    VR(q)       variance ratio of the next q-day return: Var(r_{t+1}+...+r_{t+q}) / (q Var(r_{t+1})), q = 2, 5, 10, 20
    AC(k)       corr(r_t, r_{t+k}), k = 1..10
    REV(k,k)    mean next-k-day return of the top past-k-day-return quintile minus the bottom quintile, k = 1, 5
    Z20 table   next-5-day return by the bucket of z = (C - MA20) / (MA20 x sd20 of daily returns)
Nulls: the daily returns are permuted 2,000 times (the state mask kept), destroying serial dependence.
Decision on FLAT days: MR_PRESENT if
    (a) min over q of the permutation p of VR(q) < 1 is < 0.0125 (Bonferroni over 4), AND
    (b) REV(1,1) or REV(5,5) < 0 with permutation p < 0.025 (Bonferroni over 2; REV(5,5) alone has little power
        against lag-1 reversion, e.g. an AR(1) with phi = -0.25 gives REV(5,5) p ~ 0.2);
otherwise NO_MEAN_REVERSION. IN days and all days reported for contrast, FLAT days split by trend.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

VERSION = "TB18-0.1.0"
QS = (2, 5, 10, 20)


@dataclass(frozen=True)
class DParams:
    n_perm: int = 2000
    rev_k: int = 5
    rev_h: int = 5
    z_window: int = 20
    alpha_vr: float = 0.05 / 4
    alpha_rev: float = 0.05
    seed: int = 20261030
    status: str = "PREREGISTERED_TB020"


def registration(dp: DParams | None = None) -> dict:
    d = asdict(dp or DParams())
    d.update(version=VERSION, qs=list(QS), decision="FLAT days: min_q p(VR<1) < 0.0125 AND min_k p(REV(k,k)<0) < 0.025, k = 1, 5 -> "
             "MR_PRESENT; otherwise NO_MEAN_REVERSION")
    return d


def turtle_state(dates: pd.DatetimeIndex, trades: pd.DataFrame, start) -> pd.Series:
    """'in' / 'flat' per date from long trades (entry, exit dates); NaN before start (warm-up)."""
    s = pd.Series("flat", index=dates, dtype=object)
    for r in trades.itertuples():
        s[(dates >= r.entry) & (dates < r.exit)] = "in"
    s[dates < pd.Timestamp(start)] = np.nan
    return s


def trend_label(close: pd.Series) -> pd.Series:
    """Causal: 'down' if close < MA50 and the 20-day return < 0, 'up' if close > MA50 and the 20-day return > 0,
    else 'range'."""
    ma = close.rolling(50, min_periods=50).mean(); r20 = close / close.shift(20) - 1
    lab = pd.Series("range", index=close.index, dtype=object)
    lab[(close < ma) & (r20 < 0)] = "down"; lab[(close > ma) & (r20 > 0)] = "up"
    lab[ma.isna() | r20.isna()] = np.nan
    return lab


def _csum(r: np.ndarray):
    r = np.asarray(r, float)
    return np.r_[0.0, np.cumsum(np.nan_to_num(r))], np.r_[0, np.cumsum(~np.isfinite(r))]


def fwd_sum(r: np.ndarray, q: int) -> np.ndarray:
    """f[t] = r[t+1] + ... + r[t+q] (NaN where not available or a NaN inside the window)."""
    n = len(r); c, bad = _csum(r)
    out = np.full(n, np.nan)
    t = np.arange(max(n - q, 0))
    v = c[t + q + 1] - c[t + 1]; v[(bad[t + q + 1] - bad[t + 1]) > 0] = np.nan
    out[t] = v
    return out


def past_sum(r: np.ndarray, k: int) -> np.ndarray:
    n = len(r); c, bad = _csum(r)
    out = np.full(n, np.nan)
    t = np.arange(k - 1, n)
    v = c[t + 1] - c[t + 1 - k]; v[(bad[t + 1] - bad[t + 1 - k]) > 0] = np.nan
    out[t] = v
    return out


def vr(r: np.ndarray, mask: np.ndarray, q: int) -> float:
    f1 = fwd_sum(r, 1); fq = fwd_sum(r, q)
    m = mask & np.isfinite(fq) & np.isfinite(f1)
    v1 = np.var(f1[m], ddof=1)
    return float(np.var(fq[m], ddof=1) / (q * v1)) if v1 > 0 and m.sum() > 10 else np.nan


def ac(r: np.ndarray, mask: np.ndarray, k: int) -> float:
    x = r[:-k]; y = r[k:]; m = mask[:-k] & np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 10 else np.nan


def rev(r: np.ndarray, mask: np.ndarray, k: int, h: int) -> tuple[float, pd.DataFrame]:
    """Mean next-h return by quintile of the past-k return (quintiles within the mask); top - bottom."""
    p = past_sum(r, k); f = fwd_sum(r, h); m = mask & np.isfinite(p) & np.isfinite(f)
    if m.sum() < 25:
        return np.nan, pd.DataFrame()
    qcut = pd.qcut(p[m], 5, labels=False, duplicates="drop")
    T = pd.DataFrame({"q": qcut, "past": p[m], "fwd": f[m]}).groupby("q").agg(n=("fwd", "size"), past_mean=("past", "mean"), fwd_mean=("fwd", "mean"),
                                                                            fwd_up=("fwd", lambda s: (s > 0).mean()))
    return float(T.fwd_mean.iloc[-1] - T.fwd_mean.iloc[0]), T


def z_table(close: pd.Series, r: np.ndarray, mask: np.ndarray, w: int, h: int) -> pd.DataFrame:
    ma = close.rolling(w, min_periods=w).mean(); sd = pd.Series(r, index=close.index).rolling(w, min_periods=w).std()
    z = ((close - ma) / (ma * sd)).to_numpy()
    f = fwd_sum(r, h); m = mask & np.isfinite(z) & np.isfinite(f)
    bins = [-np.inf, -2, -1, 0, 1, 2, np.inf]
    b = pd.cut(z[m], bins)
    return pd.DataFrame({"z": b, "fwd": f[m]}).groupby("z", observed=False).fwd.agg(n="size", fwd_mean="mean", fwd_up=lambda s: (s > 0).mean())


def permutation(r: np.ndarray, mask: np.ndarray, dp: DParams) -> dict:
    """Null distributions with the returns permuted (mask kept)."""
    rng = np.random.default_rng(dp.seed)
    ok = np.isfinite(r)
    out = {f"VR{q}": np.empty(dp.n_perm) for q in QS}; out["REV1"] = np.empty(dp.n_perm); out["REV5"] = np.empty(dp.n_perm)
    for i in range(dp.n_perm):
        x = r.copy(); x[ok] = rng.permutation(r[ok])          # NaN positions (e.g. the first day) stay NaN
        for q in QS:
            out[f"VR{q}"][i] = vr(x, mask, q)
        out["REV1"][i] = rev(x, mask, 1, 1)[0]; out["REV5"][i] = rev(x, mask, 5, 5)[0]
    return out


def state_stats(r: np.ndarray, mask: np.ndarray, null: dict | None) -> dict:
    d = {"days": int(mask.sum()), "mean_next_day_pct": float(np.nanmean(fwd_sum(r, 1)[mask]) * 100),
         "ann_drift_pct": float(np.nanmean(fwd_sum(r, 1)[mask]) * 365 * 100), "ann_vol_pct": float(np.nanstd(fwd_sum(r, 1)[mask]) * np.sqrt(365) * 100)}
    for q in QS:
        v = vr(r, mask, q); d[f"VR{q}"] = v
        if null is not None:
            d[f"p_VR{q}_below"] = float((1 + np.sum(null[f"VR{q}"] <= v)) / (1 + len(null[f"VR{q}"]))) if np.isfinite(v) else np.nan
    for k in range(1, 11):
        d[f"AC{k}"] = ac(r, mask, k)
    for k in (1, 5):
        s, _ = rev(r, mask, k, k); d[f"REV{k}{k}"] = s
        if null is not None:
            d[f"p_REV{k}{k}_below"] = float((1 + np.sum(null[f"REV{k}"] <= s)) / (1 + len(null[f"REV{k}"]))) if np.isfinite(s) else np.nan
    return d


def decide(flat: dict, dp: DParams) -> dict:
    pv = np.nanmin([flat[f"p_VR{q}_below"] for q in QS]) if np.isfinite([flat[f"p_VR{q}_below"] for q in QS]).any() else np.nan
    a = bool(np.isfinite(pv) and pv < dp.alpha_vr)
    pr = [flat[f"p_REV{k}{k}_below"] for k in (1, 5)]
    prm = np.nanmin(pr) if np.isfinite(pr).any() else np.nan
    b = bool(np.isfinite(prm) and prm < dp.alpha_rev / 2)
    return {"min_p_VR_below": pv, "VR_pass": bool(a), "REV11": flat["REV11"], "REV55": flat["REV55"], "min_p_REV": prm, "REV_pass": b,
            "verdict": "MR_PRESENT" if (a and b) else "NO_MEAN_REVERSION"}
