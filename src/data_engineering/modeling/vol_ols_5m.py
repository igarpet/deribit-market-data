"""Five-minute NEXT-BAR volatility OLS; research, not trading production.

Targets (both VOLATILITY, never returns or squared returns), one subsequent
5m bar, never spanning a gap or an option-session boundary:

  parkinson  sigma = log(H/L) / sqrt(4 log 2) from the NEXT perpetual trade bar.
             Known contamination: zero range on 0-1 trade bars, downward bias
             on sparse bars, upward bias from bid-ask bounce.
  index_abs  sigma = |r_index| / sqrt(2/pi) from the NEXT Deribit index 5m
             log return (E|r| = sigma sqrt(2/pi) under normality). Noisy, but
             free of trade sampling and bounce. Used to audit whether
             coefficients under `parkinson` reflect volatility or measurement.

Pure OLS. No HAR baseline: models are compared with two parameter-free
references, the training mean and persistence (current-bar value of the same
estimator).

v0.1.1 (Volatility Regression 002, approved by Ivan 2026-10-08):
  - removed the index-RV/clock "baseline" regression;
  - out-of-sample R^2 uses the TRAINING mean as benchmark;
  - second target `index_abs`; by default both targets use the same rows;
  - target-quality audit and clock-hour coverage tables;
  - dropped the raw `intrabar_range_bps` duplicate of `log_range_bps`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import qr

from data_engineering.modeling.full_information import (
    FEATURE_SPECS, raw_feature_frame, clock_core,
)

VOL_OLS_5M_VERSION = "0.1.1"
BAR = pd.Timedelta(minutes=5)
DEFAULT_DEVELOPMENT_END = "2026-10-07"
PARKINSON_C = float(np.sqrt(4.0 * np.log(2.0)))
ABS_RETURN_C = float(np.sqrt(2.0 / np.pi))
TARGETS = ("parkinson", "index_abs")
TARGET_COLUMNS = {"parkinson": "sigma_next_5m", "index_abs": "index_sigma_next_5m"}
CURRENT_COLUMNS = {"parkinson": "sigma_curr_5m", "index_abs": "index_sigma_curr_5m"}


def _bool(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.map({"True": True, "False": False, "true": True,
                   "false": False, True: True, False: False})
    return s.fillna(False).astype(bool)


def _num(g: pd.DataFrame, col: str) -> pd.Series:
    if col not in g:
        return pd.Series(np.nan, index=g.index, dtype=float)
    return pd.to_numeric(g[col], errors="coerce").astype(float)


def _parkinson(high: pd.Series, low: pd.Series, ok: pd.Series) -> pd.Series:
    valid = ok & (low > 0) & (high >= low)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.log(high / low) / PARKINSON_C
    return sigma.where(valid & np.isfinite(sigma))


def make_panel(g: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (metadata with targets, contemporaneous X, inventory).

    Input is ONE underlying of time_series_feature_state; state_time is the
    END of the current bar (trades in (t-5m, t]). Targets come from row t+1,
    which must be exactly 5m later and in the same option session. The
    Parkinson target bar must also pass perp_integrity_ok. No imputation.
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

    nxt_time = g["state_time"].shift(-1)
    link = ((nxt_time - g["state_time"]) == BAR) & \
        g["session_id"].eq(g["session_id"].shift(-1))

    perp_ok = _bool(g["perp_integrity_ok"])
    high, low = _num(g, "high"), _num(g, "low")
    sigma_curr = _parkinson(high, low, perp_ok)
    sigma_next = sigma_curr.shift(-1).where(link)

    r = _num(g, "return_5m")
    index_curr = (r.abs() / ABS_RETURN_C).where(np.isfinite(r))
    index_next = index_curr.shift(-1).where(link)

    meta = g[["underlying", "state_time", "session_id"]].copy()
    if "expiry_date" in g:
        meta["expiry_date"] = g["expiry_date"]
    meta["sigma_next_5m"] = sigma_next
    meta["index_sigma_next_5m"] = index_next
    meta["sigma_curr_5m"] = sigma_curr
    meta["index_sigma_curr_5m"] = index_curr
    # Target-bar diagnostics for the audit only; never enter X.
    meta["next_trade_count"] = _num(g, "trade_count").shift(-1).where(link)
    meta["next_range_bps"] = _num(g, "intrabar_range_bps").shift(-1).where(link)
    meta["next_spread_bps"] = _num(g, "book_twa_spread_bps").shift(-1).where(link)

    # Current-bar features only, from the established versioned catalog.
    X, used, omitted = raw_feature_frame(g, FEATURE_SPECS)
    if "return_5m" in g:
        X["return_5m"] = r
    core = clock_core(g)
    for col in core:
        X[col] = pd.to_numeric(core[col], errors="coerce")
    if "minutes_to_expiry" in g:
        tte = _num(g, "minutes_to_expiry")
        X["log1p_tte_min"] = np.log1p(tte.where(tte >= 0))
    X = X.replace([np.inf, -np.inf], np.nan)
    spec_names = {s.name for s in FEATURE_SPECS}
    inventory = pd.DataFrame(
        [{"name": s.name, "family": s.block, "source": s.column,
          "available_in_table": True} for s in used]
        + [{"name": s.name, "family": s.block, "source": s.column,
            "available_in_table": False} for s in omitted]
        + [{"name": c, "family": "clock_or_extra", "source": c,
            "available_in_table": True} for c in X.columns if c not in spec_names]
    )
    return meta, X, inventory


def session_labels(meta: pd.DataFrame) -> np.ndarray:
    """Option-expiry session label (YYYY-MM-DD), not the civil date."""
    src = meta["expiry_date"] if "expiry_date" in meta else meta["state_time"]
    return pd.to_datetime(src, utc=True).dt.strftime("%Y-%m-%d").to_numpy()


def session_split(meta: pd.DataFrame, valid: np.ndarray,
                  development_end: str = DEFAULT_DEVELOPMENT_END
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronological ~60/20/20 split by session; nothing after development_end."""
    date = session_labels(meta)
    sessions = np.array(sorted(set(date[(date <= development_end) & valid])))
    if len(sessions) < 8:
        raise ValueError(f"Need >=8 usable development sessions; found {len(sessions)}")
    a = max(4, int(len(sessions) * 0.6))
    b = max(a + 2, int(len(sessions) * 0.8))
    if b >= len(sessions):
        raise ValueError("Insufficient chronological selection/test sessions")
    return tuple(np.isin(date, s) & (date <= development_end) for s in
                 (sessions[:a], sessions[a:b], sessions[b:])) + (sessions,)


def fit_log_floor(sigma: np.ndarray) -> float:
    """Training-only positive-scale epsilon for zero-volatility bars."""
    positive = sigma[np.isfinite(sigma) & (sigma > 0)]
    if len(positive) < 20:
        raise ValueError("Too few positive next-bar volatilities.")
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
    rank = int((diag > diag.max() * 1e-10).sum())
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


def score(truth: np.ndarray, pred: np.ndarray, train_mean: float, label: str) -> dict:
    """Log-volatility RMSE/MAE and OOS R^2 against the TRAINING mean."""
    ok = np.isfinite(truth) & np.isfinite(pred)
    truth, pred = truth[ok], pred[ok]
    if not len(truth):
        return {"split": label, "n": 0, "log_rmse": np.nan,
                "log_mae": np.nan, "r2_vs_train_mean": np.nan}
    resid = truth - pred
    sst = np.sum((truth - train_mean) ** 2)
    return {"split": label, "n": int(len(truth)),
            "log_rmse": float(np.sqrt(np.mean(resid ** 2))),
            "log_mae": float(np.mean(np.abs(resid))),
            "r2_vs_train_mean": float(1 - np.sum(resid ** 2) / sst) if sst > 0 else np.nan}


def metrics(fit: OLSFit, X: pd.DataFrame, y: np.ndarray, mask: np.ndarray,
            label: str, train_mean: float) -> dict:
    idx, _ = _matrix(X, y, mask, fit.names)
    pred = fit.predict(X.iloc[idx]) if len(idx) else np.array([])
    return score(y[idx], pred, train_mean, label)


def backward_validation(X: pd.DataFrame, y: np.ndarray, train: np.ndarray,
                        selection: np.ndarray, names: list[str],
                        max_drops: int = 15, min_improvement: float = 0.0
                        ) -> tuple[list[str], pd.DataFrame]:
    """Greedy backward deletion on FIXED complete-case rows (selection RMSE).

    Selection is exploratory and NOT the final untouched test.
    """
    keep = names.copy()
    history = []
    complete = np.isfinite(X[names].to_numpy(float)).all(axis=1)
    train_rows = train & np.isfinite(y) & complete
    select_rows = selection & np.isfinite(y) & complete
    if train_rows.sum() <= len(names) + 20 or select_rows.sum() < 20:
        raise ValueError("Insufficient full-model common rows for elimination")
    mean_tr = float(np.mean(y[train_rows]))
    for step in range(max_drops + 1):
        base_fit = fit_ols(X, y, train_rows, keep)
        base = metrics(base_fit, X, y, select_rows, "selection", mean_tr)["log_rmse"]
        if step == max_drops or len(keep) <= 1:
            break
        choices = []
        for col in keep:
            subset = [n for n in keep if n != col]
            try:
                fit = fit_ols(X, y, train_rows, subset)
                s = metrics(fit, X, y, select_rows, "selection", mean_tr)["log_rmse"]
                choices.append((s, col))
            except ValueError:
                continue
        if not choices:
            break
        s, removed = min(choices)
        if not s < base - min_improvement:
            break
        history.append({"step": step + 1, "removed": removed, "p_before": len(keep),
                        "rmse_before": base, "rmse_after": s,
                        "improvement": base - s})
        keep.remove(removed)
    return keep, pd.DataFrame(history)


def prepare_asset(g: pd.DataFrame, target: str = "parkinson",
                  coverage: float = 0.50,
                  development_end: str = DEFAULT_DEVELOPMENT_END,
                  require_both_targets: bool = True):
    """Build the all-eligible complete-case regression for one target.

    require_both_targets=True keeps only rows where BOTH targets are valid,
    so coefficients under `parkinson` and `index_abs` come from the same
    observations. Coverage/alias exclusions are recorded, never read as
    "no predictive value". Nothing is imputed.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    meta, X, inventory = make_panel(g)
    sigma = meta[TARGET_COLUMNS[target]].to_numpy(float)
    valid = np.isfinite(sigma)
    if require_both_targets:
        for t in TARGETS:
            valid &= np.isfinite(meta[TARGET_COLUMNS[t]].to_numpy(float))
    train, select, test, sessions = session_split(meta, valid, development_end)
    train &= valid
    select &= valid
    test &= valid
    epsilon = fit_log_floor(sigma[train])
    y = np.where(valid, np.log(np.where(valid, sigma, 0.0) + epsilon), np.nan)
    current = meta[CURRENT_COLUMNS[target]].to_numpy(float)
    persistence = np.log(current + epsilon)  # NaN where current bar invalid

    cov = X.loc[train].notna().mean()
    names = cov.index[cov >= coverage].tolist()
    dropped_coverage = cov.index[cov < coverage].tolist()
    if not names:
        raise ValueError("No features meet train coverage gate")
    # Individually available features can have vanishing JOINT coverage.
    # Remove features for training support only; reported separately.
    dropped_joint = []
    while True:
        complete = np.isfinite(X[names].to_numpy(float)).all(axis=1)
        ntr = int((train & complete).sum())
        nsel = int((select & complete).sum())
        if ntr > max(100, 3 * (len(names) + 1)) and nsel >= 25:
            break
        if len(names) <= 2:
            raise ValueError(f"Insufficient joint support: train={ntr}, selection={nsel}")
        candidates = []
        for col in names:
            rest = [n for n in names if n != col]
            gain = int((train & np.isfinite(X[rest].to_numpy(float)).all(axis=1)).sum()) - ntr
            candidates.append((gain, -float(cov[col]), col))
        removed = max(candidates)[2]
        names.remove(removed)
        dropped_joint.append(removed)
    complete = np.isfinite(X[names].to_numpy(float)).all(axis=1)
    names, dropped_alias = remove_aliases(X, train & complete, names)
    complete = np.isfinite(X[names].to_numpy(float)).all(axis=1)
    # Common rows: every model and reference is scored on the same rows.
    common = complete & np.isfinite(y) & np.isfinite(persistence)
    return {"meta": meta, "X": X, "inventory": inventory, "target": target,
            "train": train, "selection": select, "test": test, "sessions": sessions,
            "valid": valid, "common": common,
            "y": y, "persistence": persistence, "epsilon": epsilon,
            "train_mean": float(np.mean(y[train & common])) if (train & common).any() else np.nan,
            "features": names,
            "excluded_coverage": dropped_coverage,
            "excluded_joint_coverage": dropped_joint,
            "excluded_aliases": dropped_alias, "coverage": cov}


def target_audit(meta: pd.DataFrame, rows: np.ndarray) -> dict:
    """Trade-range target quality on the given rows (development, both valid)."""
    m = meta.loc[rows]
    sp = m["sigma_next_5m"].to_numpy(float)
    si = m["index_sigma_next_5m"].to_numpy(float)
    tc = m["next_trade_count"].to_numpy(float)
    rng = m["next_range_bps"].to_numpy(float)
    spr = m["next_spread_bps"].to_numpy(float)
    out = {"n": int(len(m)),
           "share_zero_range": float(np.mean(sp == 0)) if len(sp) else np.nan,
           "share_index_zero": float(np.mean(si == 0)) if len(si) else np.nan}
    ok = np.isfinite(tc)
    out["share_trades_le_1"] = float(np.mean(tc[ok] <= 1)) if ok.any() else np.nan
    for q in (10, 50, 90):
        out[f"trade_count_p{q}"] = float(np.percentile(tc[ok], q)) if ok.any() else np.nan
    ok = np.isfinite(rng) & np.isfinite(spr)
    out["share_range_le_spread"] = float(np.mean(rng[ok] <= spr[ok])) if ok.any() else np.nan
    pos = (sp > 0) & (si > 0)
    lp, li = np.log(sp[pos]), np.log(si[pos])
    out["n_both_positive"] = int(pos.sum())
    out["corr_log_parkinson_log_index_abs"] = (
        float(np.corrcoef(lp, li)[0, 1]) if pos.sum() > 2 else np.nan)
    out["spearman_parkinson_index_abs"] = (
        float(pd.Series(sp[pos]).corr(pd.Series(si[pos]), method="spearman"))
        if pos.sum() > 2 else np.nan)
    ok = pos & np.isfinite(tc)
    out["corr_log_parkinson_log1p_trades"] = (
        float(np.corrcoef(np.log(sp[ok]), np.log1p(tc[ok]))[0, 1]) if ok.sum() > 2 else np.nan)
    ok = pos & np.isfinite(spr) & (spr > 0)
    out["corr_log_parkinson_log_spread"] = (
        float(np.corrcoef(np.log(sp[ok]), np.log(spr[ok]))[0, 1]) if ok.sum() > 2 else np.nan)
    ok = pos & np.isfinite(tc)
    out["corr_log_index_abs_log1p_trades"] = (
        float(np.corrcoef(np.log(si[ok]), np.log1p(tc[ok]))[0, 1]) if ok.sum() > 2 else np.nan)
    return out


def clock_coverage(d: dict) -> pd.DataFrame:
    """Per split and UTC hour of the predictor bar: target-valid vs scored rows."""
    hour = d["meta"]["state_time"].dt.hour.to_numpy()
    rows = []
    for split in ("train", "selection", "test"):
        mask = d[split]
        for h in range(24):
            at = hour == h
            rows.append({"split": split, "utc_hour": h,
                         "n_target_valid": int((mask & at).sum()),
                         "n_scored": int((mask & d["common"] & at).sum())})
    out = pd.DataFrame(rows)
    out["scored_share"] = out["n_scored"] / out["n_target_valid"].replace(0, np.nan)
    return out
