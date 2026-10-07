"""
Session-level out-of-sample comparison (framework v3.2 §19.1).

The primary unit is the session. For two models scored on identical test
bars, the per-session difference is the sum of the per-bar log-score
differences within the session:

    d_s = sum_{t in session s} [log p_A(z_t | F_{t-1}) - log p_B(z_t | F_{t-1})]

Reported: mean and median d_s, the fraction of sessions won by A, a
whole-session bootstrap confidence interval for the mean, and the range of
the mean when each session is left out in turn. The per-bar Newey-West t is
secondary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_engineering.modeling.hsmm.baselines import hac_t


def paired_sessions(score_a: np.ndarray, score_b: np.ndarray, sessions: np.ndarray,
                    n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    diff = np.asarray(score_a, float) - np.asarray(score_b, float)
    d = pd.Series(diff).groupby(np.asarray(sessions)).sum()
    n = len(d)
    out = {"n_sessions": n, "mean_session_diff": float(d.mean()) if n else np.nan,
           "median_session_diff": float(d.median()) if n else np.nan,
           "sessions_won": float((d > 0).mean()) if n else np.nan,
           "per_bar_x1000": 1000 * float(diff.mean()), "per_bar_nw_t": hac_t(diff)}
    if n >= 2:
        rng = np.random.default_rng(seed)
        boot = d.to_numpy()[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
        a = (1 - level) / 2
        out["boot_lo"], out["boot_hi"] = float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))
        loo = [(d.sum() - v) / (n - 1) for v in d.to_numpy()]
        out["loso_min"], out["loso_max"] = float(min(loo)), float(max(loo))
    else:
        out.update(boot_lo=np.nan, boot_hi=np.nan, loso_min=np.nan, loso_max=np.nan)
    return out
