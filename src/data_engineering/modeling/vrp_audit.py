"""
Volatility Risk Premium branch, step 1: pricing and variance-convention audit.

Discussion 018 §2.1: Module 12 reports realized-vs-implied gaps of opposite
sign at the terminal and the 30m horizon (e.g. BTC +0.33 vs -0.11 in median
log ratio). Before any conditional test, this module decomposes that gap
into its definitional ingredients:

    implied side   flat  IV^2 * dt          (the Black-76 pricing variance)
                   clock IV^2 * tau * clock share   (Module 12 allocation)
                   settlement weighting w_j^2 on or off
    realized side  sum r^2 to expiry, w^2-weighted or not; 30m RV
    sample         each definition on its own rows, or on common rows

and provides the contract-level checks (Black-76 repricing, time-basis offset
of the exchange IV, ATM IV feature vs straddle-implied IV).

Nothing here is fitted except the training-only clock profile reused from
vol_har / vol_targets. No result feeds a model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_engineering.modeling import vol_targets as vt


VRP_AUDIT_VERSION = "0.1.1"
MINUTES_PER_YEAR = vt.MINUTES_PER_YEAR
BAR_YEARS = vt.BAR_MIN / vt.MINUTES_PER_YEAR
TTE_EDGES_H = [0.5, 2.0, 4.0, 8.0, 16.0, 24.01]
TTE_LABELS = ["0.5-2h", "2-4h", "4-8h", "8-16h", "16-24h"]


def tte_bucket(tte_min: np.ndarray) -> pd.Categorical:
    return pd.cut(np.asarray(tte_min, float) / 60.0, TTE_EDGES_H, labels=TTE_LABELS, right=True)


# --------------------------------------------------
# Variance definitions per origin
# --------------------------------------------------

def realized_to_expiry(panel: vt.Panel, weighted: bool) -> np.ndarray:
    """sum_{j>t} (w_j^2) r_j^2 to the origin's expiry; NaN unless every bar
    to expiry is present with a finite return."""
    _, grid, pos, epos = vt._grid(panel.times, panel.expiry)
    rg = np.full(len(grid), np.nan)
    rg[pos] = panel.r
    good = np.isfinite(rg)
    return vt._weighted_suffix(np.where(good, rg, 0.0) ** 2, good, pos, epos, weighted=weighted)


def flat_time_to_expiry(panel: vt.Panel, weighted: bool) -> np.ndarray:
    """sum_{j>t} (w_j^2) * bar length in years: the flat-volatility pricing
    time to expiry (equals tau when unweighted)."""
    _, grid, pos, epos = vt._grid(panel.times, panel.expiry)
    ones = np.full(len(grid), BAR_YEARS)
    good = np.ones(len(grid), bool)
    return vt._weighted_suffix(ones, good, pos, epos, weighted=weighted)


def variance_definitions(panel: vt.Panel, implied: dict) -> pd.DataFrame:
    """All realized and implied variance definitions for every origin row.

    implied: output of vt.implied_variance (clock-allocated pieces)."""
    iv2 = panel.iv ** 2
    h30 = vt.HORIZONS["30m"] * BAR_YEARS
    h60 = vt.HORIZONS["60m"] * BAR_YEARS
    return pd.DataFrame({
        "session": panel.session_date,
        "tte_min": panel.tte_min,
        # realized
        "rv_exp": realized_to_expiry(panel, weighted=False),
        "rv_exp_w": realized_to_expiry(panel, weighted=True),
        "rv_30m": panel.fwd_rv["30m"],
        "rv_60m": panel.fwd_rv["60m"],
        # implied, flat in time (Black-76 pricing variance)
        "imp_exp_flat": iv2 * flat_time_to_expiry(panel, weighted=False),
        "imp_exp_flat_w": iv2 * flat_time_to_expiry(panel, weighted=True),
        "imp_30m_flat": iv2 * h30,
        "imp_60m_flat": iv2 * h60,
        # implied, clock-allocated (Module 12)
        "imp_exp_clock_w": implied["W_imp_settle"],
        "imp_30m_clock": implied["W_imp_alloc_30m"],
        "imp_60m_clock": implied["W_imp_alloc_60m"],
    })


COMPARISONS = {
    # name: (realized column, implied column)
    "exp_flat": ("rv_exp", "imp_exp_flat"),
    "exp_flat_w": ("rv_exp_w", "imp_exp_flat_w"),
    "exp_clock_w (M12 terminal)": ("rv_exp_w", "imp_exp_clock_w"),
    "30m_flat": ("rv_30m", "imp_30m_flat"),
    "30m_clock (M12 intraday)": ("rv_30m", "imp_30m_clock"),
    "60m_flat": ("rv_60m", "imp_60m_flat"),
    "60m_clock": ("rv_60m", "imp_60m_clock"),
}


def _stats(rv: np.ndarray, imp: np.ndarray, sessions: np.ndarray) -> dict:
    ok = np.isfinite(rv) & np.isfinite(imp) & (rv > 0) & (imp > 0)
    if ok.sum() < 5:
        return {"n": int(ok.sum())}
    lr = np.log(rv[ok] / imp[ok])
    ses = pd.Series(lr).groupby(sessions[ok]).median()
    return {
        "n": int(ok.sum()),
        "sessions": int(len(ses)),
        "median_log_ratio": float(np.median(lr)),
        "mean_log_ratio": float(np.mean(lr)),
        # aggregate ratio: what a variance seller's P&L scales with
        "log_ratio_of_sums": float(np.log(rv[ok].sum() / imp[ok].sum())),
        "share_rv_above_imp": float(np.mean(lr > 0)),
        "session_median_min": float(ses.min()),
        "session_median_max": float(ses.max()),
    }


def decomposition_table(defs: pd.DataFrame, asset: str, common_rows: bool = False) -> pd.DataFrame:
    """Realized-vs-implied statistics for every comparison, overall and by
    TTE bucket. common_rows=True restricts every comparison to rows where all
    definitions exist (removes sample-composition differences)."""
    sessions = defs["session"].to_numpy()
    base = np.ones(len(defs), bool)
    if common_rows:
        cols = sorted({c for pair in COMPARISONS.values() for c in pair})
        base = np.isfinite(defs[cols].to_numpy(float)).all(axis=1)
    bucket = tte_bucket(defs["tte_min"].to_numpy()).astype(str)
    rows = []
    for name, (rc, ic) in COMPARISONS.items():
        rv = np.where(base, defs[rc].to_numpy(float), np.nan)
        im = np.where(base, defs[ic].to_numpy(float), np.nan)
        rows.append({"underlying": asset, "comparison": name, "tte_bucket": "all", **_stats(rv, im, sessions)})
        for b in TTE_LABELS:
            m = bucket == b
            rows.append({"underlying": asset, "comparison": name, "tte_bucket": b,
                         **_stats(np.where(m, rv, np.nan), np.where(m, im, np.nan), sessions)})
    return pd.DataFrame(rows)


# --------------------------------------------------
# Index realized-variance noise check
# --------------------------------------------------

def sampling_ratio(panel: vt.Panel, step: int = 6) -> pd.DataFrame:
    """Per session: RV from 5-minute returns vs RV from non-overlapping
    step-bar returns (same span). A ratio well above 1 indicates
    microstructure noise / negative autocorrelation in 5-minute index returns
    (it would inflate realized variance against implied); below 1 indicates
    positive autocorrelation (trending)."""
    out = []
    for s in np.unique(panel.session_date):
        idx = np.flatnonzero(panel.session_date == s)
        r = panel.r[idx]
        if len(r) < 4 * step or np.isfinite(r).mean() < 0.95:
            continue
        rr = np.where(np.isfinite(r), r, 0.0)       # a missing bar contributes zero to both sums
        n = (len(rr) // step) * step
        if n < 4 * step:
            continue
        rv5 = float(np.sum(rr[:n] ** 2))
        agg = rr[:n].reshape(-1, step).sum(axis=1)
        rvk = float(np.sum(agg ** 2))
        out.append({"session": s, "rv_5m": rv5, f"rv_{step * 5}m": rvk,
                    "ratio_5m_over_coarse": rv5 / rvk if rvk > 0 else np.nan,
                    "lag1_autocorr": float(pd.Series(rr[:n]).autocorr(1))})
    return pd.DataFrame(out)


# --------------------------------------------------
# Exchange IV time basis
# --------------------------------------------------

def time_basis_offset(log_iv_ratio: np.ndarray, T_years: np.ndarray, weights: np.ndarray | None = None,
                      min_minutes: float = 30.0, max_abs_log_ratio: float = 0.5, iters: int = 30) -> dict:
    """If the exchange computes IV with T_ex = T + delta while we use T, the
    same price gives sigma_ours^2 T = sigma_ex^2 (T + delta), i.e.

        log(sigma_ours / sigma_ex) = 0.5 log(1 + delta / T) ~ 0.5 delta / T.

    Robust fit (0.1.1): rows with T below min_minutes or |log ratio| above
    max_abs_log_ratio (failed inversions near intrinsic) are dropped, and
    delta is estimated by Huber IRLS. The 0.1 weighted least squares was
    dominated by a few failed inversions and reported offsets of 50-160
    minutes that the repricing errors (Part A) rule out."""
    y = np.asarray(log_iv_ratio, float)
    T = np.asarray(T_years, float)
    w0 = np.ones_like(y) if weights is None else np.asarray(weights, float)
    ok = (np.isfinite(y) & np.isfinite(T) & (T * MINUTES_PER_YEAR >= min_minutes)
          & (np.abs(y) <= max_abs_log_ratio) & np.isfinite(w0) & (w0 > 0))
    if ok.sum() < 20:
        return {"n": int(ok.sum())}
    yy, x, w0 = y[ok], 0.5 / T[ok], w0[ok]
    X = np.column_stack([np.ones(len(yy)), x])
    beta = np.linalg.lstsq(X, yy, rcond=None)[0]
    for _ in range(iters):
        r = yy - X @ beta
        scale = 1.4826 * np.median(np.abs(r - np.median(r))) or 1e-6
        k = 1.345 * scale
        hub = np.where(np.abs(r) <= k, 1.0, k / np.maximum(np.abs(r), 1e-12))
        W = hub * np.minimum(w0 / np.median(w0), 10.0)
        beta = np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (W * yy))
    resid = yy - X @ beta
    return {"n": int(ok.sum()), "level_log_ratio": float(beta[0]),
            "delta_minutes": float(beta[1] * MINUTES_PER_YEAR),
            "resid_mad": float(1.4826 * np.median(np.abs(resid - np.median(resid))))}
