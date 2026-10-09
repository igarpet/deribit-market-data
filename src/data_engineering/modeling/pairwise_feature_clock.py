"""Same-UTC-clock five-minute feature relationship matrices (research v2).

For each underlying and every slot 00:00, 00:05, ..., 23:55 UTC, group
ONLY that exact five-minute clock slot from all available UTC dates. Within
each group, fit directed simple OLS using only pairwise-complete observations.

This descriptive study is NOT forward prediction. With ~19 days in the
existing dataset, individual fits are small-sample and may be unstable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_engineering.modeling.pairwise_feature_regression import pairwise_ols

PAIRWISE_CLOCK_VERSION = "2.0.0"
SLOTS_PER_DAY = 288
MIN_CLOCK_PAIR_ROWS = 6
CAUTION_N_BELOW = 12
ARRAY_KEYS = ("beta_raw", "error_rmse", "intercept", "correlation",
              "relative_rmse", "pair_counts")


def clock_slot_indices(state_time: pd.Series) -> np.ndarray:
    """Exact 5m UTC clock bucket, 0..287; reject off-grid timestamps.

    tz-aware timestamps are converted to UTC, not evaluated in local time.
    """
    t = pd.to_datetime(state_time, utc=True, errors="raise")
    if t.isna().any():
        raise ValueError("state_time contains NULL clock timestamps")
    if (t.dt.minute % 5 != 0).any() or (t.dt.second != 0).any() or \
            (t.dt.microsecond != 0).any() or (t.dt.nanosecond != 0).any():
        raise ValueError("state_time must align to exact five-minute UTC boundaries")
    return (t.dt.hour.to_numpy(dtype=np.int16) * 12
            + t.dt.minute.to_numpy(dtype=np.int16) // 5)


def slot_labels() -> np.ndarray:
    return np.array([f"{i//12:02d}:{(i%12)*5:02d}" for i in range(SLOTS_PER_DAY)])


@dataclass
class ClockMatrices:
    """Each array has shape [288 clock slots, response feature, predictor feature]."""
    features: tuple[str, ...]
    slots: np.ndarray
    beta_raw: np.ndarray
    error_rmse: np.ndarray
    intercept: np.ndarray
    correlation: np.ndarray
    relative_rmse: np.ndarray
    pair_counts: np.ndarray
    summary: pd.DataFrame

    def view(self, slot: str, field: str = "beta_raw") -> pd.DataFrame:
        """Convenient full 82x82 matrix for any exact UTC slot."""
        if field not in ARRAY_KEYS:
            raise ValueError(f"Unknown matrix: {field}")
        index = np.flatnonzero(self.slots == slot)
        if len(index) != 1:
            raise ValueError(f"Unknown UTC slot {slot!r}; use 'HH:MM' on a 5m grid")
        return pd.DataFrame(getattr(self, field)[int(index[0])],
                            index=self.features, columns=self.features)


def fit_clock_matrices(features: pd.DataFrame, state_time: pd.Series,
                       min_pair_rows: int = MIN_CLOCK_PAIR_ROWS,
                       caution_below: int = CAUTION_N_BELOW) -> ClockMatrices:
    """Compute all 288 clock-specific pairwise OLS matrix slices for one asset.

    No imputation, no time shifts, no pooling of 11:00 with 11:05.
    Each slot uses different DATES, at the same exact clock time.
    Missing feature values only disqualify their own pair.
    """
    if min_pair_rows < 3:
        raise ValueError("min_pair_rows must be >= 3")
    if caution_below <= min_pair_rows:
        raise ValueError("caution_below must exceed min_pair_rows")
    if len(features) != len(state_time):
        raise ValueError("features and state_time must have identical row counts")
    if not features.columns.is_unique:
        raise ValueError("Feature columns must be unique")
    n_features = len(features.columns)
    slots = slot_labels()
    indices = clock_slot_indices(pd.Series(state_time).reset_index(drop=True))
    day = pd.to_datetime(pd.Series(state_time).reset_index(drop=True),
                         utc=True).dt.normalize()
    # Underlying/state_time duplicates are a data-quality failure: never count
    # the same date and clock observation twice.
    if pd.DataFrame({"day": day, "slot": indices}).duplicated().any():
        raise ValueError("Duplicate UTC date / five-minute clock slot")
    shape = (SLOTS_PER_DAY, n_features, n_features)
    floats = {name: np.full(shape, np.nan, dtype=np.float64)
              for name in ARRAY_KEYS if name != "pair_counts"}
    counts = np.zeros(shape, dtype=np.uint32)
    summary = []
    for slot in range(SLOTS_PER_DAY):
        rows = np.flatnonzero(indices == slot)
        n_days = len(rows)
        n_fit = n_stable = 0
        if n_days:
            matched = features.iloc[rows].reset_index(drop=True)
            fit = pairwise_ols(matched, min_n=min_pair_rows)
            floats["beta_raw"][slot] = fit.beta.to_numpy()
            floats["error_rmse"][slot] = fit.rmse.to_numpy()
            floats["intercept"][slot] = fit.intercept.to_numpy()
            floats["correlation"][slot] = fit.standardized_beta.to_numpy()
            floats["relative_rmse"][slot] = fit.relative_rmse.to_numpy()
            counts[slot] = fit.n_pairs.to_numpy(dtype=np.uint32)
            valid = np.isfinite(floats["beta_raw"][slot])
            n_fit = int(valid.sum())
            n_stable = int((valid & (counts[slot] >= caution_below)).sum())
        summary.append({
            "clock_utc": str(slots[slot]),
            "clock_slot": slot,
            "n_dates": n_days,
            "max_pair_n": int(counts[slot].max(initial=0)),
            "n_fitted_directed_pairs": n_fit,
            "n_fitted_below_caution_n": n_fit - n_stable,
            "n_fitted_ge_caution_n": n_stable,
            "n_possible_directed_pairs": n_features * (n_features - 1),
        })
    return ClockMatrices(
        features=tuple(features.columns), slots=slots,
        pair_counts=counts, summary=pd.DataFrame(summary), **floats
    )


def save_clock_matrices(result: ClockMatrices, path: str | Path) -> None:
    """Compressed numeric 3D stacks; stores EVERY 5-minute clock slot."""
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Expected .npz destination")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, features=np.asarray(result.features, dtype="U100"),
        clock_utc=result.slots.astype("U5"),
        **{name: getattr(result, name) for name in ARRAY_KEYS},
    )


def load_clock_matrix(path: str | Path, slot: str,
                      field: str = "beta_raw") -> pd.DataFrame:
    """Read and display one clock matrix without materializing any CSV stack."""
    if field not in ARRAY_KEYS:
        raise ValueError(f"Unknown matrix: {field}")
    with np.load(path, allow_pickle=False) as f:
        slots = f["clock_utc"]
        idx = np.flatnonzero(slots == slot)
        if len(idx) != 1:
            raise ValueError("Unknown exact UTC five-minute slot: " + slot)
        names = f["features"].tolist()
        values = f[field][int(idx[0])]
    return pd.DataFrame(values, index=names, columns=names)


def pair_clock_trajectory(result: ClockMatrices, response: str,
                          predictor: str) -> pd.DataFrame:
    """Extract a single directed feature relationship for all 288 slots."""
    i = result.features.index(response)
    j = result.features.index(predictor)
    return pd.DataFrame({
        "clock_utc": result.slots,
        "response": response, "predictor": predictor,
        "beta": result.beta_raw[:, i, j],
        "rmse": result.error_rmse[:, i, j],
        "correlation": result.correlation[:, i, j],
        "relative_rmse": result.relative_rmse[:, i, j],
        "n_pairs": result.pair_counts[:, i, j],
    })
