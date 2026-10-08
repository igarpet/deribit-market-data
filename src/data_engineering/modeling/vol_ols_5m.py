"""Five-minute NEXT-BAR volatility regression; research, not trading production.

Target: Parkinson VOLATILITY (not return or squared return) computed from the
NEXT 5m perpetual trade high/low. Bar t features predict the interval (t,t+5].
Trade-range microstructure bounce is a known limitation; compare alternative
target estimators before any deployment. No target spans an expiry session.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.linalg import qr

from data_engineering.modeling.full_information import (
    FEATURE_SPECS, raw_feature_frame, clock_core,
)

VOL_OLS_5M_VERSION = "0.1.0"
BAR = pd.Timedelta(minutes=5)
DEFAULT_DEVELOPMENT_END = "2026-10-07"


def _bool(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.map({"True": True, "False": False, "true": True,
                   "false": False, True: True, False: False})
    return s.fillna(False).astype(bool)


def make_panel(g: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (metadata and next-bar volatility, contemporaneous X, inventory).

    Input is ONE underlying of time_series_feature_state; state_time is the
    close/boundary of the current bar. The next bar must be exactly 5m later,
    in the same option session, and pass perp_integrity_ok. No imputation.
    """
    required = ("state_time", "underlying", "session_id", "high", "low",
                "perp_integrity_ok")
    missing = [c for c in required if c not in g]
    if missing:
        raise ValueError(f"Required feature columns missing: {missing}")
    if g["underlying"].nunique() != 1:
        raise ValueError("make_panel expects one underlying.")
    g = g.sort_values("state_time").reset_index(drop=True).copy()
    g["state_time"] = pd.to_datetime(g["state_time"], utc=True)
    if g["state_time"].duplicated().any():
        raise ValueError("Duplicate state_time in one asset; inspect upstream.")
    if not g["state_time"].is_monotonic_increasing:
        raise ValueError("state_time is not monotonic.")

    nxt_time = g["state_time"].shift(-1)
    contiguous = (nxt_time - g["state_time"]) == BAR
    same_session = g["session_id"].eq(g["session_id"].shift(-1))
    high = pd.to_numeric(g["high"].shift(-1), errors="coerce")
    low = pd.to_numeric(g["low"].shift(-1), errors="coerce")
    next_good = _bool(g["perp_integrity_ok"]).shift(-1).fillna(False).astype(bool)
    valid = contiguous & same_session & next_good & (low > 0) & (high >= low)
    sigma = np.log(high / low) / np.sqrt(4 * np.log(2))
    sigma = sigma.where(valid & np.isfinite(sigma))

    meta = g[["underlying", "state_time", "session_id"]].copy()
    if "expiry_date" in g:
        meta["expiry_date"] = g["expiry_date"]
    meta["sigma_next_5m"] = sigma
    meta["target_valid"] = sigma.notna()
    # Current-bar features only. Importing the established versioned catalog
    # avoids silently admitting future labels, IDs or readiness flags into X.
    X, used, omitted = raw_feature_frame(g, FEATURE_SPECS)
    for col in ("return_5m", "intrabar_range_bps"):
        if col in g:
            X[col] = pd.to_numeric(g[col], errors="coerce")
    core = clock_core(g)
    for col in core:
        X[col] = pd.to_numeric(core[col], errors="coerce")
    if "minutes_to_expiry" in g:
        tte = pd.to_numeric(g["minutes_to_expiry"], errors="coerce")
        X["log1p_tte_min"] = np.log1p(tte.where(tte >= 0))
    X = X.replace([np.inf, -np.inf], np.nan)
    inventory = pd.DataFrame(
        [{"name": s.name, "family": s.block, "source": s.column,
          "available_in_table": True} for s in used]
        + [{"name": s.name, "family": s.block, "source": s.column,
            "available_in_table": False} for s in omitted]
        + [{"name": c, "family": "clock_or_extra", "source": c,
            "available_in_table": True} for c in X.columns
           if c not in {s.name for s in FEATURE_SPECS}]
    )
    return meta, X, inventory


def session_split(meta: pd.DataFrame, development_end: str = DEFAULT_DEVELOPMENT_END
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronological sessions; do not touch HSMM 06 post-freeze confirmation."""
    t = pd.to_datetime(meta["state_time"], utc=True)
    date = t.dt.strftime("%Y-%m-%d").to_numpy()
    sessions = np.array(sorted(set(date[(date <= development_end)
                                        & meta["target_valid"].to_numpy()])))
    if len(sessions) < 8:
        raise ValueError(f"Need >=8 usable development dates; found {len(sessions)}")
    a = max(4, int(len(sessions) * 0.6))
    b = max(a + 2, int(len(sessions) * 0.8))
    if b >= len(sessions):
        raise ValueError("Insufficient chronological selection/test sessions")
    return tuple(np.isin(date, s) & (date <= development_end) for s in
                 (sessions[:a], sessions[a:b], sessions[b:])) + (sessions,)


def fit_log_floor(sigma: np.ndarray) -> float:
    """Training-only positive-scale epsilon for zero-range bars."""
    positive = sigma[np.isfinite(sigma) & (sigma > 0)]
    if len(positive) < 20:
        raise ValueError("Too few positive next-bar range volatilities.")
    return float(0.01 * np.median(positive))


@dataclass
class OLSFit:
    names: list[str]
    center: np.ndarray
    scale: np.ndarray
    beta: np.ndarray
    rank: int
    condition_number: float
    n: int

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        z = (X[self.names].to_numpy(float) - self.center) / self.scale
        return self.beta[0] + z @ self.beta[1:]


def _matrix(X: pd.DataFrame, y: np.ndarray, mask: np.ndarray,
            names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    ok = mask & np.isfinite(y)
    if names:
        ok &= np.isfinite(X[names].to_numpy(float)).all(axis=1)
    return np.flatnonzero(ok), y[ok]


def remove_aliases(X: pd.DataFrame, train: np.ndarray, names: list[str]
                   ) -> tuple[list[str], list[str]]:
    """Remove constants/exact linear aliases by pivoted QR, using train only."""
    vals = X.loc[train, names].to_numpy(float)
    if not np.isfinite(vals).all():
        raise ValueError("Alias detection requires complete training rows")
    sd = vals.std(axis=0)
    nonconstant = np.flatnonzero(sd > 1e-12)
    if len(nonconstant) == 0:
        raise ValueError("No nonconstant predictors")
    z = (vals[:, nonconstant] - vals[:, nonconstant].mean(axis=0)) / sd[nonconstant]
    _, R, piv = qr(z, pivoting=True, mode="economic")
    diag = np.abs(np.diag(R))
    rank = int((diag > (diag.max() if len(diag) else 0) * 1e-10).sum())
    kept_ids = set(nonconstant[piv[:rank]])
    keep = [n for i, n in enumerate(names) if i in kept_ids]
    drop = [n for n in names if n not in keep]
    return keep, drop


def fit_ols(X: pd.DataFrame, y: np.ndarray, train: np.ndarray,
            names: list[str]) -> OLSFit:
    idx, yy = _matrix(X, y, train, names)
    if len(idx) <= len(names) + 20:
        raise ValueError(f"n={len(idx)} insufficient for p={len(names)} OLS")
    vals = X.iloc[idx][names].to_numpy(float)
    mu, sd = vals.mean(axis=0), vals.std(axis=0)
    if not (sd > 1e-12).all():
        raise ValueError("Constant predictor; run remove_aliases first")
    Z = np.column_stack([np.ones(len(idx)), (vals - mu) / sd])
    beta, _, rank, _ = np.linalg.lstsq(Z, yy, rcond=None)
    if rank != len(names) + 1:
        raise ValueError(f"Rank deficient OLS ({rank}/{len(names)+1})")
    return OLSFit(names.copy(), mu, sd, beta, int(rank),
                  float(np.linalg.cond(Z)), len(idx))


def metrics(fit: OLSFit, X: pd.DataFrame, y: np.ndarray,
            mask: np.ndarray, label: str) -> dict:
    idx, _ = _matrix(X, y, mask, fit.names)
    if not len(idx):
        return {"split": label, "n": 0, "log_rmse": np.nan,
                "log_mae": np.nan, "r2": np.nan}
    truth = y[idx]
    pred = fit.predict(X.iloc[idx])
    resid = truth - pred
    sst = np.sum((truth - np.mean(truth)) ** 2)
    return {"split": label, "n": len(idx),
            "log_rmse": float(np.sqrt(np.mean(resid**2))),
            "log_mae": float(np.mean(np.abs(resid))),
            "r2": float(1 - np.sum(resid**2) / sst) if sst > 0 else np.nan}


def backward_validation(X: pd.DataFrame, y: np.ndarray, train: np.ndarray,
                        selection: np.ndarray, names: list[str],
                        max_drops: int = 15, min_improvement: float = 0.0
                        ) -> tuple[list[str], pd.DataFrame]:
    """Greedy backward deletion on FIXED complete-case rows.

    All candidates compare on the same train/selection population. Selection
    is exploratory and NOT the final untouched test. No t-test fishing.
    """
    keep = names.copy()
    history = []
    # Common mask fixed from the original full model.
    train_rows = train & np.isfinite(y) & np.isfinite(X[names]).all(axis=1).to_numpy()
    select_rows = selection & np.isfinite(y) & np.isfinite(X[names]).all(axis=1).to_numpy()
    if train_rows.sum() <= len(names) + 20 or select_rows.sum() < 20:
        raise ValueError("Insufficient full-model common rows for elimination")
    for step in range(max_drops + 1):
        base_fit = fit_ols(X, y, train_rows, keep)
        base = metrics(base_fit, X, y, select_rows, "selection")["log_rmse"]
        if step == max_drops or len(keep) <= 1:
            break
        choices = []
        for col in keep:
            subset = [n for n in keep if n != col]
            try:
                fit = fit_ols(X, y, train_rows, subset)
                score = metrics(fit, X, y, select_rows, "selection")["log_rmse"]
                choices.append((score, col))
            except ValueError:
                continue
        if not choices:
            break
        score, removed = min(choices)
        if not score < base - min_improvement:
            break
        history.append({"step": step + 1, "removed": removed, "p_before": len(keep),
                        "rmse_before": base, "rmse_after": score,
                        "improvement": base - score})
        keep.remove(removed)
    return keep, pd.DataFrame(history)


def prepare_asset(g: pd.DataFrame, coverage: float = 0.50,
                  development_end: str = DEFAULT_DEVELOPMENT_END):
    """Construct the initial all-eligible complete-case regression.

    Coverage filtering is recorded; it is NOT interpreted as proof of no
    predictive value. No missing data are imputed or deleted invisibly.
    """
    meta, X, inventory = make_panel(g)
    train, select, test, sessions = session_split(meta, development_end)
    positive = meta["sigma_next_5m"].to_numpy(float)
    epsilon = fit_log_floor(positive[train])
    y = np.log(positive + epsilon)
    y[~np.isfinite(positive)] = np.nan
    initial = list(X.columns)
    eligible = train & np.isfinite(y)
    cov = X.loc[eligible].notna().mean()
    names = cov.index[cov >= coverage].tolist()
    dropped_coverage = cov.index[cov < coverage].tolist()
    if not names:
        raise ValueError("No features meet train coverage gate")
    # Fixed complete-case panel for full OLS and all model comparisons.
    common = np.isfinite(X[names].to_numpy(float)).all(axis=1)
    tr_common = train & np.isfinite(y) & common
    names, dropped_alias = remove_aliases(X, tr_common, names)
    return {"meta": meta, "X": X, "inventory": inventory,
            "train": train, "selection": select, "test": test, "sessions": sessions,
            "y": y, "epsilon": epsilon, "features": names,
            "excluded_coverage": dropped_coverage,
            "excluded_aliases": dropped_alias, "coverage": cov,
            "original_feature_count": len(initial)}
