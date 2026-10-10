"""
Volatility Dynamics 03b: calibration of the Module 03 q99 precursor scan.

Module 03 scanned ~205 fields x 5 pre-event lags, kept each field's best lag,
and ranked by a discovery score with no reference distribution. 03b asks how
large that score gets by chance, using the same code path on placebo events.

Placebo ("day relabel"): every real q99 onset keeps its asset and exact UTC
five-minute slot, but its UTC day is mapped through a random permutation of
the valid days. Onsets on the same day move together, so the number of
events, their clock distribution, their cross-asset synchronicity and their
within-day clustering are preserved. Placebos landing on q99/non-eligible or
missing rows are dropped (counts reported).

What the null does and does NOT control
---------------------------------------
- It calibrates the *selection* inflation of the scan: best-of-5-lags,
  sparse features with few event timestamps, asset/lag "consistency" across
  synchronous assets, and exact-slot control noise.
- It does NOT remove volatility clustering: real q99 onsets occur in volatile
  periods, placebos do not. Volatility-type fields are therefore expected to
  beat this null. Whether anything is informative *beyond* the volatility
  environment is the question of 04b, not 03b.

Outputs per feature: observed Module 03 discovery score, null mean / q95,
per-feature permutation p, family-wise p (vs the max score over all
features in each placebo run), and the same for the pre-registered fixed
summary (mean percentile over the five lags, no best-lag selection).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from data_engineering.modeling import volatility_vd_b_core as vb


VERSION = "0.1.0"
RANK_BARS = (-12, -6, -3, -2, -1)     # Module 03 lags: -60,-30,-15,-10,-5 min
DEFAULT_N_PERM = 500
DEFAULT_SEED = 3030


def scan(
    grid: vb.Grid,
    X: np.ndarray,
    event_rows: np.ndarray,
    eligible: np.ndarray,
    *,
    max_controls: int = 8,
    min_controls: int = 4,
) -> pd.DataFrame:
    """Module 03 clock-only control scan restricted to the ranking lags."""
    cc = vb.clock_controls(grid, event_rows, eligible, max_controls=max_controls)
    P = vb.percentile_tensor(grid, X, event_rows, cc, RANK_BARS, min_controls=min_controls)
    return vb.module03_rankings(
        P, grid.time_ns[event_rows], grid.asset_code[event_rows], RANK_BARS, RANK_BARS,
    )


def placebo_null(
    grid: vb.Grid,
    X: np.ndarray,
    event_rows: np.ndarray,
    eligible: np.ndarray,
    *,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = DEFAULT_SEED,
    max_controls: int = 8,
    min_controls: int = 4,
    progress_every: int = 0,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """(n_perm, n_feat) arrays of placebo discovery scores and |window mean - 0.5|."""
    rng = np.random.default_rng(int(seed))
    days = vb.valid_days(grid)
    f = X.shape[1]
    S = np.full((int(n_perm), f), np.nan)
    W = np.full((int(n_perm), f), np.nan)
    counts = []
    for b in range(int(n_perm)):
        pl = vb.day_relabel_events(grid, event_rows, eligible, days, rng)
        r = scan(grid, X, pl, eligible, max_controls=max_controls, min_controls=min_controls)
        S[b] = r["discovery_score"].to_numpy(float)
        W[b] = np.abs(r["window_mean_percentile"].to_numpy(float) - 0.5)
        counts.append({
            "perm": b,
            "n_placebo_asset_events": int(len(pl)),
            "n_placebo_unique_times": int(len(np.unique(grid.time_ns[pl]))),
        })
        if progress_every and (b + 1) % int(progress_every) == 0:
            print(f"placebo {b + 1}/{n_perm}")
    return S, W, pd.DataFrame(counts)


def calibrate(
    observed: pd.DataFrame,
    features: Sequence[str],
    S: np.ndarray,
    W: np.ndarray,
) -> pd.DataFrame:
    """Per-feature and family-wise permutation p-values."""
    obs_s = observed["discovery_score"].to_numpy(float)
    obs_w = np.abs(observed["window_mean_percentile"].to_numpy(float) - 0.5)
    S0 = np.where(np.isfinite(S), S, 0.0)
    W0 = np.where(np.isfinite(W), W, 0.0)
    max_s = S0.max(axis=1)
    max_w = W0.max(axis=1)
    n = S.shape[0]
    rows = []
    for j, feat in enumerate(features):
        s = S[:, j]
        w = W[:, j]
        sf = s[np.isfinite(s)]
        wf = w[np.isfinite(w)]
        os_, ow = obs_s[j], obs_w[j]
        rows.append({
            "feature": feat,
            "observed_discovery_score": os_,
            "best_rel_min": observed["best_rel_bar"].iloc[j] * 5 if "best_rel_bar" in observed and pd.notna(observed["best_rel_bar"].iloc[j]) else np.nan,
            "n_unique_event_times_best": observed.get("best_n_unique_event_times", pd.Series(np.nan, index=observed.index)).iloc[j],
            "observed_window_mean_percentile": observed["window_mean_percentile"].iloc[j],
            "window_n_unique_event_times": observed["window_n_unique_event_times"].iloc[j],
            "null_n_valid": int(len(sf)),
            "null_mean_score": float(np.mean(sf)) if len(sf) else np.nan,
            "null_q95_score": float(np.quantile(sf, 0.95)) if len(sf) else np.nan,
            "p_feature_score": float((1 + np.sum(sf >= os_)) / (len(sf) + 1)) if np.isfinite(os_) and len(sf) else np.nan,
            "p_familywise_score": float((1 + np.sum(max_s >= os_)) / (n + 1)) if np.isfinite(os_) else np.nan,
            "null_q95_abs_window_dev": float(np.quantile(wf, 0.95)) if len(wf) else np.nan,
            "p_feature_window": float((1 + np.sum(wf >= ow)) / (len(wf) + 1)) if np.isfinite(ow) and len(wf) else np.nan,
            "p_familywise_window": float((1 + np.sum(max_w >= ow)) / (n + 1)) if np.isfinite(ow) else np.nan,
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["p_familywise_score", "observed_discovery_score"], ascending=[True, False]).reset_index(drop=True)


def familywise_thresholds(S: np.ndarray, W: np.ndarray) -> pd.DataFrame:
    S0 = np.where(np.isfinite(S), S, 0.0).max(axis=1)
    W0 = np.where(np.isfinite(W), W, 0.0).max(axis=1)
    return pd.DataFrame({
        "statistic": ["max_discovery_score", "max_abs_window_dev"],
        "null_q50": [np.quantile(S0, 0.5), np.quantile(W0, 0.5)],
        "null_q95": [np.quantile(S0, 0.95), np.quantile(W0, 0.95)],
        "null_q99": [np.quantile(S0, 0.99), np.quantile(W0, 0.99)],
    })


def selftest() -> dict:
    f = vb.synthetic_frame(n_days=20, seed=11)
    g = vb.build_grid(f)
    thr = vb.q99_thresholds(g)
    tail = vb.tail_flags(g, thr)
    ev = np.flatnonzero(vb.onset_flags(g, tail))
    elig = vb.eligible_origins(g, tail)
    feats = ["intrabar_range_bps", "volume_usd", "noise_feature"]
    X = g.feature_matrix(feats)
    obs = scan(g, X, ev, elig)
    S, W, cnt = placebo_null(g, X, ev, elig, n_perm=40, seed=1)
    cal = calibrate(obs, feats, S, W).set_index("feature")
    # vol proxies must beat the placebo (vol clustering is real in the synthetic data);
    # the pure-noise field must not.
    assert cal.loc["intrabar_range_bps", "p_feature_score"] < 0.1
    assert cal.loc["noise_feature", "p_feature_score"] > 0.05
    assert cnt["n_placebo_asset_events"].min() > 0
    return {"placebo_null": True, "vol_proxy_beats_placebo": True, "noise_not_flagged": True}
