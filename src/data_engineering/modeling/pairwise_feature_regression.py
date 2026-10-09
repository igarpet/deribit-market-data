"""Pairwise, contemporaneous, single-predictor OLS feature exploration.

For every ordered (response Y, predictor X) pair, on ONLY their jointly
observed five-minute bars, fit Y = intercept + beta * X + residual.
Results are descriptive, not causal, predictive, or inferential.

All matrices use ROW=response, COLUMN=predictor.
No global complete-case filtering; no imputation; no forward target or test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd

from data_engineering.modeling.full_information import (
    FEATURE_SPECS, clock_core, raw_feature_frame,
)

PAIRWISE_FEATURE_REGRESSION_VERSION = "1.0.0"
MIN_PAIR_OBSERVATIONS = 100
EXTRAS = (
    ("return_5m", "return_5m", "returns", "identity"),
    ("log1p_tte_min", "minutes_to_expiry", "clock", "log1p"),
)


def model_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All established FEATURE_SPECS plus explicit contemporaneous clock extras.

    The feature catalog's scope_mask() is applied as-is: no marks substituted
    for missing quotes, no bypass of perp/L2 readiness or settlement policy.
    This is a feature-catalog audit, NOT all numeric DuckDB metadata columns.
    """
    raw, used, missing = raw_feature_frame(df, FEATURE_SPECS)
    records = [
        {"feature": s.name, "source_column": s.column, "family": s.block,
         "transform": s.transform, "scope": s.scope or "unrestricted",
         "source_present": s in used}
        for s in FEATURE_SPECS
    ]
    for s in missing:
        raw[s.name] = np.nan
    if all(col in df for col in (
        "utc_minute_of_day", "session_progress_fraction",
        "settlement_locked_fraction", "is_weekend"
    )):
        extra_clock = clock_core(df)
        for feature in extra_clock.columns:
            raw[feature] = extra_clock[feature]
            records.append({"feature": feature,
                            "source_column": "derived clock",
                            "family": "clock", "transform": "derived",
                            "scope": "unrestricted", "source_present": True})
    else:
        for feature in (
            "session_progress", "settlement_locked", "weekend",
            "clock_sin1", "clock_cos1", "clock_sin2", "clock_cos2",
        ):
            raw[feature] = np.nan
            records.append({"feature": feature,
                            "source_column": "derived clock",
                            "family": "clock", "transform": "derived",
                            "scope": "unrestricted", "source_present": False})
    for name, source, family, transform in EXTRAS:
        if source in df:
            x = pd.to_numeric(df[source], errors="coerce").astype(float)
            if transform == "log1p":
                x = np.log1p(x.where(x >= 0))
            raw[name] = x.where(np.isfinite(x))
            present = True
        else:
            raw[name] = np.nan
            present = False
        records.append({"feature": name, "source_column": source, "family": family,
                        "transform": transform, "scope": "unrestricted",
                        "source_present": present})
    raw = raw.loc[:, ~raw.columns.duplicated()].astype(float)
    if raw.columns.empty:
        raise ValueError("No research features available")
    inventory = pd.DataFrame(records).drop_duplicates("feature")
    inventory["n_observed"] = raw.notna().sum().reindex(inventory.feature).to_numpy(int)
    inventory["n_total_states"] = len(raw)
    inventory["coverage"] = inventory["n_observed"] / max(1, len(raw))
    return raw, inventory


@dataclass
class PairwiseOLSResult:
    """Response features run down rows; predictor features run across columns."""
    beta: pd.DataFrame
    intercept: pd.DataFrame
    rmse: pd.DataFrame
    standardized_beta: pd.DataFrame
    relative_rmse: pd.DataFrame
    r_squared: pd.DataFrame
    n_pairs: pd.DataFrame

    def long_form(self) -> pd.DataFrame:
        names = list(self.beta.index)
        j, k = np.indices((len(names), len(names)))
        i, col = j.ravel(), k.ravel()
        result = pd.DataFrame({
            "response": np.asarray(names)[i],
            "predictor": np.asarray(names)[col],
        })
        for key in ("beta", "intercept", "rmse", "standardized_beta",
                    "relative_rmse", "r_squared", "n_pairs"):
            value = getattr(self, key).to_numpy()
            result[key] = value[i, col]
        return result.loc[result.response != result.predictor].reset_index(drop=True)


def pairwise_ols(features: pd.DataFrame, min_n: int = MIN_PAIR_OBSERVATIONS
                 ) -> PairwiseOLSResult:
    """All ordered OLS fits using vectorized pair-specific cross moments.

    For each response i and predictor j, the exact same paired observations
    calculate the slope, intercept, residual RMS and N. RMS uses denominator
    n (descriptive in-sample error), not n-2, and is NOT a forecast error.
    """
    if min_n < 3:
        raise ValueError("min_n must be at least 3")
    if not features.columns.is_unique:
        raise ValueError("Duplicate predictor names")
    if not len(features.columns):
        raise ValueError("No features")
    a = features.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    observed = np.isfinite(a)
    mask = observed.astype(float)
    values = np.where(observed, a, 0.0)
    squares = values * values
    count = mask.T @ mask
    sum_y = values.T @ mask
    sum_x = mask.T @ values
    sum_y2 = squares.T @ mask
    sum_x2 = mask.T @ squares
    sum_xy = values.T @ values
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inv_n = np.divide(1.0, count, out=np.zeros_like(count),
                          where=count > 0)
        sxx = sum_x2 - sum_x * sum_x * inv_n
        syy = sum_y2 - sum_y * sum_y * inv_n
        sxy = sum_xy - sum_y * sum_x * inv_n
        # Avoid fitting constant/near-constant variables and cancellation
        # when sums are large relative to their centered variation.
        x_precision = np.maximum(1e-20, 32 * np.finfo(float).eps * np.abs(sum_x2))
        y_precision = np.maximum(1e-20, 32 * np.finfo(float).eps * np.abs(sum_y2))
        valid = ((count >= min_n) & (sxx > x_precision)
                 & (syy > y_precision))
        np.fill_diagonal(valid, False)
        beta = np.divide(sxy, sxx, out=np.full_like(sxx, np.nan),
                         where=valid)
        intercept = np.where(valid, (sum_y - beta * sum_x) * inv_n, np.nan)
        sse = np.maximum(0.0, syy - beta * sxy)
        rmse = np.sqrt(np.where(valid, sse * inv_n, np.nan))
        corr = np.divide(sxy, np.sqrt(sxx * syy),
                         out=np.full_like(sxx, np.nan), where=valid)
        corr = np.clip(corr, -1, 1)
        r_squared = corr ** 2
        target_sd = np.sqrt(np.maximum(0, syy * inv_n))
        relative = np.divide(rmse, target_sd, out=np.full_like(sxx, np.nan),
                             where=valid)
    names = list(features.columns)
    def table(z): return pd.DataFrame(z, index=names, columns=names)
    return PairwiseOLSResult(
        beta=table(beta), intercept=table(intercept),
        rmse=table(rmse), standardized_beta=table(corr),
        relative_rmse=table(relative), r_squared=table(r_squared),
        n_pairs=table(count.astype(int)),
    )


def summarize_pairs(result: PairwiseOLSResult) -> dict:
    """Counts are directed off-diagonal cells (2 per undirected pair)."""
    good = np.isfinite(result.beta.to_numpy(float))
    n = len(result.beta)
    return {
        "n_features": n,
        "possible_directed_pairs": int(n * (n - 1)),
        "fitted_directed_pairs": int(good.sum()),
        "invalid_directed_pairs": int(n * (n - 1) - good.sum()),
        "max_joint_observations": int(result.n_pairs.to_numpy().max()),
    }
