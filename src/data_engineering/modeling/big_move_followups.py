"""Regime Structure 04 and 05 — follow-ups to RS 03 (RS 008 §7).

Pre-registration: discussions/Regime_Structure/009_claude_rs04_rs05_preregistration.md.

RS 04 — does 0DTE implied vol underreact to yesterday's realised vol?
    Predictor x = log(pre_rv_prev_vs_fcst) (RS 03's one confirmed feature, known at 08:00, Binance 1m).
    Entry: the ATM 0DTE straddle of the primary family at 12:00 UTC (20 h to the 08:00 expiry), VRP 02 code.
    Outcome y = log(move_over_implied) = log(|log(S_T/F)| / (IV_straddle sqrt(T) sqrt(2/pi))).
    beta = OLS slope of y on x over session-assets; 95% bootstrap CI over sessions.
        beta > 0 (CI) -> UNDERREACTION: implied rises less than the realised move after a high-RV day
        beta < 0 (CI) -> OVERREACTION
        else          -> NO_EVIDENCE
    Decomposition (reported): slope of log(realised |move| / sigma_hat) and of log(implied move / sigma_hat) on x.
    P&L (reported): R_net of the 12:00 straddle by predictor tercile (cutoffs from the RS 03 dev window).

RS 05 — is the big-day profile more than a big net move produces mechanically?
    Null: every 5-minute return multiplied by a random sign, one sign per (UTC session, bar) shared by all assets.
    It keeps |r| (so sigma_hat, RV, shocks and volume are unchanged) and destroys direction.
    BIG uses the frozen RS 03 thresholds in the real data and in every null draw.
    Statistics (test window, pooled over assets; 399 draws; two-sided p, Bonferroni over 5 -> 0.01):
        S1 VR_all      mean over all sessions of (net / RV)^2          (>1 trending, <1 mean-reverting intraday)
        S2 ER_big      mean efficiency ratio on BIG days
        S3 GIVE_big    mean giveback (max |excursion| - |net|) / sigma_hat on BIG days
        S4 TOP_big     mean share of the session's |hourly moves| in its largest hour on BIG days
        S5 LIFT        P(BIG tomorrow | BIG today) / P(BIG)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from data_engineering.modeling import big_move_days as bm
from data_engineering.modeling import vrp_long_straddle as ls

VERSION = "big_move_followups v1.0"
TEST_START = bm.TEST_START
NULL_EPOCH = pd.Timestamp("2024-01-01", tz="UTC")
STATS = ("S1_VR_all", "S2_ER_big", "S3_GIVE_big", "S4_TOP_big", "S5_LIFT")


@dataclass(frozen=True)
class Params:
    entry_utc_hour: int = 12
    n_boot: int = 4000
    n_null: int = 399
    family_alpha: float = 0.05
    seed: int = 20261013


def registration(p: Params | None = None) -> dict:
    p = p or Params()
    return {**asdict(p), "version": VERSION, "rs03": bm.VERSION, "vrp": ls.VERSION, "test_start": str(TEST_START),
            "rs04": "beta of log(move/implied) on log(pre_rv_prev_vs_fcst); session bootstrap 95% CI",
            "rs05": "real vs shared-sign 5m null; two-sided p <= 0.05/5; test window", "stats": list(STATS)}


# ================================================================ RS 04

def predictor_table(tables: dict) -> pd.DataFrame:
    """Per (session, asset): predictor and sigma_hat from RS 03 tables (session = expiry date string)."""
    rows = []
    for a, t in tables.items():
        rows.append(pd.DataFrame({"session": t["session"].dt.strftime("%Y-%m-%d"), "underlying": a,
                                  "x_raw": t["pre_rv_prev_vs_fcst"], "sigma_hat": t["sigma_hat"],
                                  "window_rs03": t["window"]}))
    out = pd.concat(rows, ignore_index=True)
    out["x"] = np.log(out["x_raw"])
    return out


def tercile_cutoffs(pred: pd.DataFrame) -> tuple:
    d = pred[(pred["window_rs03"] == "dev") & np.isfinite(pred["x"])]
    return tuple(float(v) for v in d["x"].quantile([1 / 3, 2 / 3]))


def entry_rows(x: pd.DataFrame, p: Params) -> pd.DataFrame:
    """The 12:00 UTC straddle per (session, underlying, family)."""
    e = x[(x["state_time"].dt.hour == p.entry_utc_hour) & (x["state_time"].dt.minute == 0)].copy()
    return e.drop_duplicates(["session", "underlying"])


def rs04_frame(entries: pd.DataFrame, pred: pd.DataFrame, cut: tuple) -> pd.DataFrame:
    d = entries.merge(pred, on=["session", "underlying"], how="inner")
    T = d["T"].to_numpy(float)
    implied_move = d["iv_straddle_mid"].to_numpy(float) * np.sqrt(T) * np.sqrt(2 / np.pi)
    real_move = np.abs(np.log(d["S_T"].to_numpy(float) / d["F"].to_numpy(float)))
    s_rem = d["sigma_hat"].to_numpy(float) * np.sqrt(T * 365.0)   # sigma_hat is per day; T in years
    with np.errstate(invalid="ignore", divide="ignore"):
        d["y"] = np.log(real_move / implied_move)
        d["y_real"] = np.log(real_move / s_rem)
        d["y_impl"] = np.log(implied_move / s_rem)
    d["tercile"] = np.select([d["x"] <= cut[0], d["x"] <= cut[1]], ["low", "mid"], "high")
    ok = np.isfinite(d["x"]) & np.isfinite(d["y_impl"])
    return d[ok].reset_index(drop=True)


def _slope(x, y, w):
    xm = np.sum(w * x) / w.sum()
    ym = np.sum(w * y) / w.sum()
    v = np.sum(w * (x - xm) ** 2)
    return np.sum(w * (x - xm) * (y - ym)) / v if v > 0 else np.nan


def session_boot_slope(d: pd.DataFrame, ycol: str, p: Params) -> dict:
    q = d[np.isfinite(d[ycol]) & np.isfinite(d["x"])]
    if q["session"].nunique() < 5:
        return {"n": int(len(q)), "beta": np.nan, "lo": np.nan, "hi": np.nan}
    x, y = q["x"].to_numpy(float), q[ycol].to_numpy(float)
    u, inv = np.unique(q["session"].to_numpy(), return_inverse=True)
    rng = np.random.default_rng(p.seed + sum(map(ord, ycol)))
    est = _slope(x, y, np.ones(len(x)))
    bs = np.array([_slope(x, y, np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv].astype(float))
                   for _ in range(p.n_boot)])
    lo, hi = np.nanquantile(bs, [0.025, 0.975])
    return {"n": int(len(q)), "sessions": int(len(u)), "beta": float(est), "lo": float(lo), "hi": float(hi)}


def rs04_verdict(s: dict) -> str:
    if not np.isfinite(s.get("lo", np.nan)):
        return "INSUFFICIENT"
    return "UNDERREACTION" if s["lo"] > 0 else ("OVERREACTION" if s["hi"] < 0 else "NO_EVIDENCE")


# ================================================================ RS 05

def null_signs(day_index: np.ndarray, draw: int, p: Params) -> np.ndarray:
    """(n_sessions, 288) signs for null draw `draw`; identical for every asset on the same session."""
    rng = np.random.default_rng([p.seed, draw])
    n_days = int(day_index.max()) + 1
    S = rng.choice(np.array([-1.0, 1.0]), size=(n_days, bm.BARS))
    return S[day_index]


def day_index(sessions: pd.DatetimeIndex) -> np.ndarray:
    d = ((sessions - NULL_EPOCH) / pd.Timedelta(days=1)).to_numpy()
    d = np.round(d).astype(int)
    if (d < 0).any():
        raise ValueError("session before NULL_EPOCH")
    return d


def path_stats(r5: np.ndarray, sig: np.ndarray, thr: float, window: np.ndarray) -> dict:
    """Per-window sums for S1-S5 for one asset (so assets can be pooled by adding)."""
    net = r5.sum(1)
    rv = np.sqrt((r5 ** 2).sum(1))
    path = np.abs(r5).sum(1)
    cum = np.cumsum(r5, 1)
    hr = np.abs(r5.reshape(len(r5), 24, 12).sum(2))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.abs(net) / sig
        vr = (net / rv) ** 2
        er = np.abs(net) / path
        give = (np.abs(cum).max(1) - np.abs(net)) / sig
        top = hr.max(1) / hr.sum(1)
    ok = np.isfinite(z)
    big = ok & (z >= thr)
    out = {}
    for w in ("dev", "test"):
        m = ok & (window == w)
        mb = big & (window == w)
        nxt_ok = m[:-1] & m[1:]
        b0, b1 = big[:-1] & nxt_ok, big[1:] & nxt_ok
        out[w] = np.array([np.nansum(vr[m]), m.sum(),
                           np.nansum(er[mb]), np.nansum(give[mb]), np.nansum(top[mb]), mb.sum(),
                           (b0 & b1).sum(), b0.sum(), b1.sum(), nxt_ok.sum()], float)
    return out


def finish(s: np.ndarray) -> dict:
    """Pooled sums -> the five statistics."""
    vr_s, n, er_s, gv_s, tp_s, nb, both, b0, b1, npair = s
    with np.errstate(invalid="ignore", divide="ignore"):
        lift = (both / b0) / (b1 / npair)
    return {"S1_VR_all": vr_s / n, "S2_ER_big": er_s / nb, "S3_GIVE_big": gv_s / nb, "S4_TOP_big": tp_s / nb,
            "S5_LIFT": lift}


def rs05_asset(b: dict, t: pd.DataFrame, thr: float, p: Params) -> dict:
    """Real sums and null sums (list over draws) for one asset."""
    sig = t["sigma_hat"].to_numpy(float)
    win = t["window"].to_numpy()
    real = path_stats(b["r5"], sig, thr, win)
    di = day_index(b["session"])
    null = [path_stats(b["r5"] * null_signs(di, j, p), sig, thr, win) for j in range(p.n_null)]
    return {"real": real, "null": null}


def rs05_evaluate(per_asset: list, p: Params) -> pd.DataFrame:
    rows = []
    alpha = p.family_alpha / len(STATS)
    for w in ("dev", "test"):
        real = finish(sum(a["real"][w] for a in per_asset))
        nulls = [finish(sum(a["null"][j][w] for a in per_asset)) for j in range(p.n_null)]
        for s in STATS:
            nv = np.array([d[s] for d in nulls], float)
            nv = nv[np.isfinite(nv)]
            r = real[s]
            med = float(np.median(nv))
            lo_tail = (1 + np.sum(nv <= r)) / (1 + len(nv))
            hi_tail = (1 + np.sum(nv >= r)) / (1 + len(nv))
            pval = min(1.0, 2 * min(lo_tail, hi_tail))
            rows.append({"window": w, "stat": s, "real": r, "null_median": med,
                         "null_q025": float(np.quantile(nv, 0.025)), "null_q975": float(np.quantile(nv, 0.975)),
                         "real_over_null": r / med if med else np.nan, "p_two_sided": pval, "alpha": alpha,
                         "significant": pval <= alpha})
    return pd.DataFrame(rows)
