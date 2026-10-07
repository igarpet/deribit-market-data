"""
Full-information supervised price benchmark.

Predict the next clock-scaled Deribit index return from causal market-state
measurements. No latent regimes are used here.

Missing values are never mean-filled: a model is fitted/scored only on rows
where every predictor in that specification is observed. Preprocessing
parameters are estimated on training rows only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FULL_INFORMATION_VERSION = "0.1"
CLIP_Z = 5.0


@dataclass(frozen=True)
class PredictorSpec:
    name: str
    column: str
    block: str
    transform: str = "identity"
    scope: str | None = None


FEATURE_SPECS: tuple[PredictorSpec, ...] = (
    PredictorSpec("ret_15m", "return_15m", "returns"),
    PredictorSpec("ret_30m", "return_30m", "returns"),
    PredictorSpec("ret_1h", "return_1h", "returns"),
    PredictorSpec("perp_ret_5m", "perp_mid_return_5m", "returns", scope="perp"),

    PredictorSpec("log_rv_30m", "realized_vol_30m_annualized", "volatility", "log"),
    PredictorSpec("log_rv_1h", "realized_vol_1h_annualized", "volatility", "log"),
    PredictorSpec("log_rv_4h", "realized_vol_4h_annualized", "volatility", "log"),
    PredictorSpec("log_range_bps", "intrabar_range_bps", "volatility", "log1p"),

    PredictorSpec("log_volume_usd", "volume_usd", "flow", "log1p"),
    PredictorSpec("log_trade_count", "trade_count", "flow", "log1p"),
    PredictorSpec("tfi", "trade_flow_imbalance", "flow"),
    PredictorSpec("buy_share", "buy_volume_share", "flow"),
    PredictorSpec("signed_volume", "signed_volume_usd", "flow", "signed_log1p"),
    PredictorSpec("log_avg_trade", "average_trade_size_usd", "flow", "log1p"),

    PredictorSpec("close_vs_vwap_bps", "close_vs_vwap_bps", "basis_oi"),
    PredictorSpec("perp_mid_basis_bps", "perpetual_mid_vs_index_bps", "basis_oi", scope="perp"),
    PredictorSpec("perp_mark_basis_bps", "perpetual_mark_vs_index_bps", "basis_oi", scope="perp"),
    PredictorSpec("microprice_mid_bps", "microprice_vs_mid_bps", "basis_oi", scope="perp"),
    PredictorSpec("oi_change", "open_interest_change_5m", "basis_oi", "signed_log1p"),
    PredictorSpec("oi_return", "open_interest_return_5m", "basis_oi"),

    PredictorSpec("log_book_updates", "book_update_count", "l2", "log1p", "micro"),
    PredictorSpec("log_spread_bps", "book_twa_spread_bps", "l2", "log", "micro"),
    PredictorSpec("obi_5", "book_twa_obi_5", "l2", scope="micro"),
    PredictorSpec("obi_10", "book_twa_obi_10", "l2", scope="micro"),
    PredictorSpec("obi_20", "book_twa_obi_20", "l2", scope="micro"),
    PredictorSpec("obi_20_change", "book_twa_obi_20_change_5m", "l2", scope="micro"),
    PredictorSpec("log_bid_depth20", "book_twa_bid_depth_20", "l2", "log", "micro"),
    PredictorSpec("log_ask_depth20", "book_twa_ask_depth_20", "l2", "log", "micro"),
    PredictorSpec("microprice_ref_bps", "book_twa_microprice_vs_reference_bps", "l2", scope="micro"),
    PredictorSpec("obi_20_min", "book_min_obi_20", "l2", scope="micro"),
    PredictorSpec("obi_20_max", "book_max_obi_20", "l2", scope="micro"),
    PredictorSpec("abs_obi_20", "book_twa_abs_obi_20", "l2", scope="instability"),
    PredictorSpec("obi_20_std", "book_obi_20_std", "l2", scope="instability"),
    PredictorSpec("obi_roughness", "book_obi_20_path_roughness_per_min", "l2", "log1p", "instability"),
    PredictorSpec("obi_range", "book_obi_20_range", "l2", scope="instability"),
    PredictorSpec("obi_coherence_noise", "book_obi_20_coherence_noise", "l2", scope="instability"),
    PredictorSpec("log_obi_transitions", "book_obi_20_transition_count", "l2", "log1p", "instability"),
    PredictorSpec("obi_zero_cross_rate", "book_obi_20_raw_zero_cross_rate", "l2", scope="instability"),
    PredictorSpec("obi_reversal_mean", "book_obi_20_reversal_strength_mean", "l2", scope="instability"),
    PredictorSpec("obi_reversal_max", "book_obi_20_reversal_strength_max", "l2", scope="instability"),

    PredictorSpec("log_atm_iv", "option_iv_atm_forward", "options", "log", "atm"),
    PredictorSpec("skew_025", "option_skew_025", "options", scope="shape"),
    PredictorSpec("skew_050", "option_skew_050", "options", scope="shape"),
    PredictorSpec("curvature_025", "option_curvature_025", "options", scope="shape"),
    PredictorSpec("curvature_050", "option_curvature_050", "options", scope="shape"),
    PredictorSpec("local_skew_025", "option_local_skew_slope_025", "options", scope="shape"),
    PredictorSpec("local_skew_050", "option_local_skew_slope_050", "options", scope="shape"),
    PredictorSpec("rr_25d", "rr_25d", "options", scope="option"),
    PredictorSpec("bf_25d", "bf_25d", "options", scope="option"),
    PredictorSpec("rr_10d", "rr_10d", "options", scope="option"),
    PredictorSpec("bf_10d", "bf_10d", "options", scope="option"),

    PredictorSpec("atm_iv_d5", "atm_iv_change_5m", "options", scope="atm"),
    PredictorSpec("atm_iv_d15", "atm_iv_change_15m", "options", scope="atm"),
    PredictorSpec("atm_iv_d30", "atm_iv_change_30m", "options", scope="atm"),
    PredictorSpec("atm_iv_d60", "atm_iv_change_1h", "options", scope="atm"),
    PredictorSpec("skew025_d5", "skew_025_change_5m", "options", scope="shape"),
    PredictorSpec("skew025_d15", "skew_025_change_15m", "options", scope="shape"),
    PredictorSpec("skew025_d30", "skew_025_change_30m", "options", scope="shape"),
    PredictorSpec("skew025_d60", "skew_025_change_1h", "options", scope="shape"),
    PredictorSpec("curv025_d5", "curvature_025_change_5m", "options", scope="shape"),
    PredictorSpec("curv025_d15", "curvature_025_change_15m", "options", scope="shape"),
    PredictorSpec("curv025_d30", "curvature_025_change_30m", "options", scope="shape"),
    PredictorSpec("curv025_d60", "curvature_025_change_1h", "options", scope="shape"),
    PredictorSpec("rr25_d5", "rr_25d_change_5m", "options", scope="option"),
    PredictorSpec("bf25_d5", "bf_25d_change_5m", "options", scope="option"),

    PredictorSpec("log_fit_atm_iv", "fit_atm_iv", "options", "log", "fit"),
    PredictorSpec("fit_skew_rel", "fit_skew_rel", "options", scope="fit"),
    PredictorSpec("fit_curvature_rel", "fit_curvature_rel", "options", scope="fit"),
    PredictorSpec("log_fit_wrmse", "fit_wrmse", "options", "log1p", "fit"),
    PredictorSpec("log_fit_quotes", "fit_n_quotes", "options", "log1p", "fit"),
    PredictorSpec("log_option_spread_med", "option_median_two_sided_spread_bps", "options", "log", "option"),
    PredictorSpec("log_option_spread_p90", "option_p90_two_sided_spread_bps", "options", "log", "option"),
    PredictorSpec("otm_bid_share_z2", "otm_bid_share_z2", "options", scope="option"),
)


CORE_NAMES = (
    "D", "z1", "z2", "z3", "vr_q6_prev",
    "session_progress", "settlement_locked", "weekend",
    "clock_sin1", "clock_cos1", "clock_sin2", "clock_cos2",
)


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.map({
            "True": True, "False": False,
            "true": True, "false": False,
            True: True, False: False,
        })
    return s.fillna(False).astype(bool)


def scope_mask(df: pd.DataFrame, scope: str | None) -> pd.Series:
    if scope is None:
        return pd.Series(True, index=df.index)

    settle = (
        _as_bool(df["in_settlement_window"])
        if "in_settlement_window" in df
        else pd.Series(False, index=df.index)
    )

    if scope == "perp":
        return (
            _as_bool(df["perp_integrity_ok"])
            if "perp_integrity_ok" in df
            else pd.Series(False, index=df.index)
        )
    if scope == "micro":
        return (
            _as_bool(df["microstructure_features_ready"])
            if "microstructure_features_ready" in df
            else pd.Series(False, index=df.index)
        )
    if scope == "instability":
        return (
            _as_bool(df["instability_features_ready"])
            if "instability_features_ready" in df
            else pd.Series(False, index=df.index)
        )
    if scope == "atm":
        ok = (
            _as_bool(df["option_atm_quote_ok"])
            if "option_atm_quote_ok" in df
            else pd.Series(False, index=df.index)
        )
        return ok & ~settle
    if scope == "shape":
        ok = (
            _as_bool(df["option_shape_quote_ok"])
            if "option_shape_quote_ok" in df
            else pd.Series(False, index=df.index)
        )
        return ok & ~settle
    if scope == "fit":
        ok = (
            _as_bool(df["fit_ok"])
            if "fit_ok" in df
            else pd.Series(False, index=df.index)
        )
        return ok & ~settle
    if scope == "option":
        ok = (
            _as_bool(df["option_snapshot_ready"])
            if "option_snapshot_ready" in df
            else pd.Series(False, index=df.index)
        )
        return ok & ~settle

    raise ValueError(f"Unknown scope {scope}")


def transformed_feature(df: pd.DataFrame, spec: PredictorSpec) -> pd.Series:
    x = pd.to_numeric(df[spec.column], errors="coerce").astype(float)

    if spec.transform == "identity":
        y = x
    elif spec.transform == "log":
        y = np.log(x.where(x > 0))
    elif spec.transform == "log1p":
        y = np.log1p(x.where(x >= 0))
    elif spec.transform == "signed_log1p":
        y = np.sign(x) * np.log1p(np.abs(x))
    else:
        raise ValueError(f"Unknown transform {spec.transform}")

    y = y.where(np.isfinite(y))
    return y.where(scope_mask(df, spec.scope))


def raw_feature_frame(
    df: pd.DataFrame,
    specs: tuple[PredictorSpec, ...] = FEATURE_SPECS,
):
    out = pd.DataFrame(index=df.index)
    used = []
    missing = []

    for spec in specs:
        if spec.column not in df:
            missing.append(spec)
            continue
        out[spec.name] = transformed_feature(df, spec)
        used.append(spec)

    return out, used, missing


def clock_core(df: pd.DataFrame) -> pd.DataFrame:
    minute = pd.to_numeric(
        df["utc_minute_of_day"],
        errors="coerce",
    ).to_numpy(float)

    u = 2.0 * np.pi * minute / 1440.0

    return pd.DataFrame(
        {
            "session_progress": pd.to_numeric(
                df["session_progress_fraction"],
                errors="coerce",
            ),
            "settlement_locked": pd.to_numeric(
                df["settlement_locked_fraction"],
                errors="coerce",
            ),
            "weekend": _as_bool(df["is_weekend"]).astype(float),
            "clock_sin1": np.sin(u),
            "clock_cos1": np.cos(u),
            "clock_sin2": np.sin(2.0 * u),
            "clock_cos2": np.cos(2.0 * u),
        },
        index=df.index,
    )


@dataclass
class Standardizer:
    names: list[str]
    mean: np.ndarray
    scale: np.ndarray
    clip: float = CLIP_Z

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        V = X[self.names].to_numpy(float)
        Z = (
            V
            - self.mean[None, :]
        ) / self.scale[None, :]
        return np.clip(
            Z,
            -self.clip,
            self.clip,
        )


def fit_standardizer(
    X: pd.DataFrame,
    names: list[str],
    rows: np.ndarray | None = None,
    clip: float = CLIP_Z,
) -> Standardizer:
    V = X[names].to_numpy(float)

    if rows is not None:
        V = V[np.asarray(rows, bool)]

    mean = np.nanmean(V, axis=0)
    scale = np.nanstd(V, axis=0)

    bad = (
        ~np.isfinite(mean)
        | ~np.isfinite(scale)
        | (scale <= 1e-12)
    )
    scale[bad] = 1.0
    mean[~np.isfinite(mean)] = 0.0

    return Standardizer(
        list(names),
        mean,
        scale,
        clip,
    )


@dataclass
class LinearModel:
    kind: str
    feature_names: list[str]
    intercept: float
    coef: np.ndarray
    sigma: float
    alpha: float = 0.0
    l1_ratio: float = 0.0
    rank: int | None = None
    condition_number: float | None = None
    n_train: int = 0

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return (
            self.intercept
            + Z @ self.coef
        )


def complete_rows(
    Z: np.ndarray,
    y: np.ndarray | None = None,
) -> np.ndarray:
    ok = np.all(
        np.isfinite(Z),
        axis=1,
    )

    if y is not None:
        ok &= np.isfinite(y)

    return ok


def _soft(
    x: float,
    lam: float,
) -> float:
    return (
        np.sign(x)
        * max(
            abs(x) - lam,
            0.0,
        )
    )


def fit_linear(
    Z: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    kind: str,
    alpha: float = 0.0,
    l1_ratio: float = 0.5,
    max_iter: int = 5000,
    tol: float = 1e-8,
) -> LinearModel:
    ok = complete_rows(
        Z,
        y,
    )

    X = np.asarray(
        Z[ok],
        float,
    )
    yy = np.asarray(
        y[ok],
        float,
    )

    if len(yy) <= X.shape[1] + 5:
        raise ValueError(
            "Too few complete training rows "
            f"({len(yy)}) for {X.shape[1]} predictors"
        )

    xm = X.mean(axis=0)
    ym = float(
        yy.mean()
    )

    Xc = (
        X
        - xm
    )
    yc = (
        yy
        - ym
    )

    rank = int(
        np.linalg.matrix_rank(
            Xc
        )
    )

    cond = (
        float(
            np.linalg.cond(
                Xc
            )
        )
        if Xc.shape[1]
        else 1.0
    )

    if kind == "ols":
        coef, *_ = np.linalg.lstsq(
            Xc,
            yc,
            rcond=None,
        )

    elif kind == "ridge":
        G = (
            Xc.T @ Xc
        ) / len(yy)

        G = (
            G
            + float(alpha)
            * np.eye(
                Xc.shape[1]
            )
        )

        h = (
            Xc.T @ yc
        ) / len(yy)

        coef = np.linalg.solve(
            G,
            h,
        )

    elif kind == "elasticnet":
        coef = np.zeros(
            Xc.shape[1]
        )

        col2 = np.mean(
            Xc * Xc,
            axis=0,
        )

        r = yc.copy()

        a1 = (
            float(alpha)
            * float(l1_ratio)
        )
        a2 = (
            float(alpha)
            * (
                1.0
                - float(l1_ratio)
            )
        )

        for _ in range(
            max_iter
        ):
            old = coef.copy()

            for j in range(
                Xc.shape[1]
            ):
                if col2[j] <= 1e-14:
                    coef[j] = 0.0
                    continue

                r += (
                    Xc[:, j]
                    * coef[j]
                )

                rho = float(
                    np.mean(
                        Xc[:, j]
                        * r
                    )
                )

                coef[j] = (
                    _soft(
                        rho,
                        a1,
                    )
                    / (
                        col2[j]
                        + a2
                    )
                )

                r -= (
                    Xc[:, j]
                    * coef[j]
                )

            if (
                np.max(
                    np.abs(
                        coef
                        - old
                    )
                )
                <= tol
                * max(
                    1.0,
                    np.max(
                        np.abs(
                            old
                        )
                    ),
                )
            ):
                break

    else:
        raise ValueError(
            f"Unknown model kind {kind}"
        )

    intercept = (
        ym
        - float(
            xm @ coef
        )
    )

    pred = (
        intercept
        + X @ coef
    )

    resid = (
        yy
        - pred
    )

    sigma = max(
        float(
            np.sqrt(
                np.mean(
                    resid
                    * resid
                )
            )
        ),
        1e-4,
    )

    return LinearModel(
        kind=kind,
        feature_names=list(feature_names),
        intercept=intercept,
        coef=coef,
        sigma=sigma,
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        rank=rank,
        condition_number=cond,
        n_train=int(
            len(yy)
        ),
    )


def gaussian_logscore(
    y: np.ndarray,
    pred: np.ndarray,
    sigma: float,
) -> np.ndarray:
    y = np.asarray(
        y,
        float,
    )

    pred = np.asarray(
        pred,
        float,
    )

    s = max(
        float(sigma),
        1e-8,
    )

    return (
        -0.5
        * np.log(
            2.0
            * np.pi
            * s
            * s
        )
        - 0.5
        * (
            (
                y
                - pred
            )
            / s
        ) ** 2
    )


def oos_metrics(
    y: np.ndarray,
    pred: np.ndarray,
    train_mean: float,
) -> dict:
    y = np.asarray(
        y,
        float,
    )

    pred = np.asarray(
        pred,
        float,
    )

    ok = (
        np.isfinite(y)
        & np.isfinite(pred)
    )

    y = y[ok]
    pred = pred[ok]

    if not len(y):
        return {
            "n": 0,
            "mse": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "corr": np.nan,
            "sign_accuracy": np.nan,
        }

    mse = float(
        np.mean(
            (
                y
                - pred
            ) ** 2
        )
    )

    den = float(
        np.mean(
            (
                y
                - float(train_mean)
            ) ** 2
        )
    )

    corr = (
        float(
            np.corrcoef(
                y,
                pred,
            )[0, 1]
        )
        if (
            len(y) > 2
            and np.std(pred) > 0
        )
        else np.nan
    )

    return {
        "n": int(
            len(y)
        ),
        "mse": mse,
        "rmse": float(
            np.sqrt(
                mse
            )
        ),
        "r2": (
            1.0
            - mse / den
            if den > 0
            else np.nan
        ),
        "corr": corr,
        "sign_accuracy": float(
            np.mean(
                np.sign(y)
                == np.sign(pred)
            )
        ),
    }


def block_map(
    specs: list[PredictorSpec],
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}

    for spec in specs:
        out.setdefault(
            spec.block,
            [],
        ).append(
            spec.name
        )

    return out
