"""
Deterministic clock baseline (framework v2.1, Step 2).

Every session runs 08:00 -> 08:00 UTC, so time of day and time to expiry
are one variable. The baseline removes the predictable part of a feature
that depends on the clock and on the calendar:

    x~_t = (x_t - m_x(clock_t, c_t)) / s_x(clock_t, c_t)

m_x and s_x are linear in a small set of deterministic regressors and are
estimated on a TRAINING window only. The fitted object is serialized so the
same baseline is reused out of sample, in walk-forward folds and to centre
episode durations in the memory study (Step 16).

Regressors (Notebook 06, 6 Oct 2026):

    location m_x:  1, cos/sin(2 pi k u / 1440) for k = 1..K   (K = 4)
                   weekend, weekend * cos/sin(2 pi k u / 1440), k = 1..2
    scale    s_x:  constant (default), or with scale="clock":
                   1, cos/sin(2 pi k u / 1440) for k = 1..2, weekend

u is the minute of day on the feature's clock:

  * "new_york": local minute of day in America/New_York. Used for returns,
    realized volatility and activity, whose intraday peak is the US
    session (13:30-14:30 UTC under US daylight saving). On 1 Nov 2026 the
    US open moves to 14:30 UTC; a New York clock keeps the peak aligned.
    Until a training window spans a US daylight-saving change, the New
    York and UTC clocks give identical fits (a constant shift of u is a
    rotation of each harmonic pair).
  * "expiry_utc": UTC minute of day, i.e. time to the 08:00 UTC expiry.
    Used for option features, whose availability and shape are tied to
    the expiry.

The weekend is the UTC calendar weekend (ISO day 6-7), as measured in
Notebook 06. Expiry-type terms are NOT included: one quarterly and one
weekly expiry in the sample did not improve out-of-sample fit.

Feature transforms:

  * "log_abs" (returns): the baseline is fitted on log|r| (zero returns
    are excluded from the fit). It defines a relative volatility factor

        f_t = exp(m_t - c),   c = 1/2 log( mean_train exp(2 m_t) )

    normalized so that the mean of f_t^2 over the training rows is 1.
    The adjusted return is r~_t = r_t / f_t.
  * "log", "log1p", "identity": the baseline is fitted on the transformed
    value; the adjusted feature is (x - m) / s.

Scale model: log((x - m)^2 + floor) is regressed on the scale regressors;
s_t = exp(0.5 * fitted + c_s) with c_s chosen so that the mean of
((x - m) / s)^2 over the training rows is 1 (this absorbs the log bias).

No lookahead: coefficients come from training rows only. Rows inside the
training window are marked in-sample by the caller.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd


CLOCK_BASELINE_VERSION = "0.2"

# v0.2 (first real run, 6 Oct 2026): the scale s_x is a CONSTANT by
# default. A clock-varying scale was never tested in Notebook 06, and on a
# 7-session training window it inflated out-of-sample adjusted values
# (mean x~^2 up to 10 for AVAX IV) by shrinking s at some hours.
# scale="clock" remains available. Degenerate-feature guard now accepts
# discrete shares (>= 3 distinct values, positive variance); skip reasons
# state the cause.

MINUTES_PER_DAY = 1440.0
NEW_YORK = "America/New_York"

DEFAULT_HARMONICS = 4
DEFAULT_WEEKEND_HARMONICS = 2
DEFAULT_SCALE_HARMONICS = 2

# Minimum evidence before the weekend terms are estimated. With fewer
# weekend sessions in the training window they are dropped (recorded in
# the fitted object), not extrapolated.
MIN_WEEKEND_STATES = 288
MIN_WEEKEND_DAYS = 2

MIN_TRAIN_ROWS = 500


@dataclass(frozen=True)
class FeatureSpec:
    """How one feature is clock-adjusted."""

    name: str
    column: str
    transform: str = "identity"     # identity | log | log1p | log_abs
    clock: str = "new_york"         # new_york | expiry_utc
    scope: str | None = None        # None | perp | fit | option
    scale: str = "constant"         # constant | clock | none
    harmonics: int = DEFAULT_HARMONICS
    weekend_harmonics: int = DEFAULT_WEEKEND_HARMONICS
    scale_harmonics: int = DEFAULT_SCALE_HARMONICS


# Features with an out-of-sample gain from a clock baseline in Notebook 06.
# Not included (no gain or worse than a constant out of sample): spread,
# volume, skew; curvature was mixed.
DEFAULT_FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec("return_5m", "return_5m", transform="log_abs", scale="none"),
    FeatureSpec("log_rv_30m", "realized_vol_30m_annualized", transform="log"),
    FeatureSpec("log_rv_1h", "realized_vol_1h_annualized", transform="log"),
    FeatureSpec("log_book_updates", "book_update_count", transform="log1p", scope="perp"),
    FeatureSpec("coherence_noise", "book_obi_20_coherence_noise", scope="perp"),
    FeatureSpec("log_fit_atm_iv", "fit_atm_iv", transform="log", clock="expiry_utc", scope="fit"),
    FeatureSpec("otm_bid_share_z2", "otm_bid_share_z2", clock="expiry_utc", scope="option"),
)

REQUIRED_COLUMNS = ("state_id", "state_time", "underlying", "is_weekend")


# --------------------------------------------------
# Clock coordinates and design matrices
# --------------------------------------------------

def add_clock_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Add clock coordinates used by the design matrices:

      clock_minute_utc        minute of day in UTC (0..1435)
      clock_minute_new_york   minute of day in America/New_York
      clock_ny_utc_offset_min New York offset from UTC in minutes
      clock_weekend           UTC weekend flag (float)
    """

    missing = [c for c in REQUIRED_COLUMNS if c not in frame]
    if missing:
        raise ValueError(f"Clock baseline input lacks columns: {missing}")

    out = frame.copy()
    t = pd.to_datetime(out["state_time"], utc=True)
    ny = t.dt.tz_convert(NEW_YORK)

    out["clock_minute_utc"] = (t.dt.hour * 60 + t.dt.minute).astype(float)
    out["clock_minute_new_york"] = (ny.dt.hour * 60 + ny.dt.minute).astype(float)
    out["clock_ny_utc_offset_min"] = (
        ny.dt.tz_localize(None) - t.dt.tz_localize(None)
    ).dt.total_seconds().div(60).round().astype(int)

    weekend = out["is_weekend"]
    if weekend.dtype == object:
        weekend = weekend.map({"True": True, "False": False, "true": True,
                               "false": False, True: True, False: False})
    if weekend.isna().any():
        raise ValueError("is_weekend has missing values.")
    out["clock_weekend"] = weekend.astype(bool).astype(float)

    iso_weekend = t.dt.dayofweek >= 5
    if (iso_weekend.astype(float) != out["clock_weekend"]).any():
        raise ValueError("is_weekend disagrees with the UTC calendar day.")

    return out


def _clock_minute(frame: pd.DataFrame, clock: str) -> np.ndarray:
    if clock == "new_york":
        return frame["clock_minute_new_york"].to_numpy(float)
    if clock == "expiry_utc":
        return frame["clock_minute_utc"].to_numpy(float)
    raise ValueError(f"Unknown clock: {clock}")


def design_matrix(
    frame: pd.DataFrame,
    clock: str,
    harmonics: int,
    weekend: bool,
    weekend_harmonics: int,
) -> tuple[np.ndarray, list[str]]:
    """Constant, Fourier terms and optional weekend terms."""

    u = _clock_minute(frame, clock)
    cols, names = [np.ones(len(frame))], ["const"]

    for k in range(1, harmonics + 1):
        angle = 2.0 * np.pi * k * u / MINUTES_PER_DAY
        cols += [np.cos(angle), np.sin(angle)]
        names += [f"cos{k}", f"sin{k}"]

    if weekend:
        w = frame["clock_weekend"].to_numpy(float)
        cols.append(w)
        names.append("weekend")
        for k in range(1, weekend_harmonics + 1):
            angle = 2.0 * np.pi * k * u / MINUTES_PER_DAY
            cols += [w * np.cos(angle), w * np.sin(angle)]
            names += [f"weekend_cos{k}", f"weekend_sin{k}"]

    return np.column_stack(cols), names


# --------------------------------------------------
# Feature values and scopes
# --------------------------------------------------

def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.map({"True": True, "False": False, "true": True,
                             "false": False, True: True, False: False})
    return series.fillna(False).astype(bool)


def scope_mask(frame: pd.DataFrame, scope: str | None) -> pd.Series:
    """Rows on which a feature is defined and may enter fit or transform."""

    if scope is None:
        return pd.Series(True, index=frame.index)

    settle = (
        _as_bool(frame["in_settlement_window"])
        if "in_settlement_window" in frame
        else pd.Series(False, index=frame.index)
    )

    if scope == "perp":
        if "perp_integrity_ok" not in frame:
            raise ValueError("Scope 'perp' requires perp_integrity_ok.")
        return _as_bool(frame["perp_integrity_ok"])
    if scope == "fit":
        if "fit_ok" not in frame:
            raise ValueError("Scope 'fit' requires fit_ok.")
        return _as_bool(frame["fit_ok"]) & ~settle
    if scope == "option":
        return ~settle

    raise ValueError(f"Unknown scope: {scope}")


def transformed_values(frame: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    """Feature on the scale the baseline is fitted on; NaN where undefined."""

    x = pd.to_numeric(frame[spec.column], errors="coerce").astype(float)

    if spec.transform == "identity":
        y = x
    elif spec.transform == "log":
        y = np.log(x.where(x > 0))
    elif spec.transform == "log1p":
        y = np.log1p(x.where(x >= 0))
    elif spec.transform == "log_abs":
        y = np.log(x.abs().where(x.abs() > 0))
    else:
        raise ValueError(f"Unknown transform: {spec.transform}")

    y = y.where(np.isfinite(y))
    return y.where(scope_mask(frame, spec.scope))


# --------------------------------------------------
# Fitted baselines
# --------------------------------------------------

@dataclass
class FittedFeatureBaseline:
    """One feature, one asset."""

    spec: dict
    underlying: str
    location_terms: list[str]
    location_coef: list[float]
    weekend_terms_used: bool
    scale_terms: list[str] = field(default_factory=list)
    scale_coef: list[float] = field(default_factory=list)
    scale_const: float = 0.0
    vol_factor_const: float | None = None
    n_train: int = 0
    train_r2_location: float | None = None
    train_ny_offsets: list[int] = field(default_factory=list)

    @property
    def feature_spec(self) -> FeatureSpec:
        return FeatureSpec(**self.spec)

    def _location(self, frame: pd.DataFrame) -> np.ndarray:
        s = self.feature_spec
        X, names = design_matrix(frame, s.clock, s.harmonics,
                                 self.weekend_terms_used, s.weekend_harmonics)
        if names != self.location_terms:
            raise RuntimeError("Location design does not match the fitted terms.")
        return X @ np.asarray(self.location_coef)

    def _scale(self, frame: pd.DataFrame) -> np.ndarray:
        s = self.feature_spec
        if s.scale == "none":
            return np.ones(len(frame))
        X, names = _scale_design(frame, s, self.weekend_terms_used)
        if names != self.scale_terms:
            raise RuntimeError("Scale design does not match the fitted terms.")
        return np.exp(0.5 * (X @ np.asarray(self.scale_coef)) + self.scale_const)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Columns <name>__clock_mean, __clock_scale, __clock_adj (and the
        volatility factor for log_abs features). `frame` must carry clock
        coordinates (add_clock_coordinates)."""

        s = self.feature_spec
        out = pd.DataFrame(index=frame.index)
        m = self._location(frame)
        defined = scope_mask(frame, s.scope).to_numpy()

        if s.transform == "log_abs":
            raw = pd.to_numeric(frame[s.column], errors="coerce").to_numpy(float)
            factor = np.exp(m - self.vol_factor_const)
            out[f"{s.name}__clock_vol_factor"] = factor
            out[f"{s.name}__clock_adj"] = np.where(defined, raw / factor, np.nan)
            return out

        y = transformed_values(frame, s).to_numpy(float)
        scale = self._scale(frame)
        out[f"{s.name}__clock_mean"] = m
        out[f"{s.name}__clock_scale"] = scale
        out[f"{s.name}__clock_adj"] = (y - m) / scale
        return out


def _ols(y: np.ndarray, X: np.ndarray, label: str) -> np.ndarray:
    rank = np.linalg.matrix_rank(X)
    if rank < X.shape[1]:
        raise RuntimeError(
            f"{label}: design is rank deficient ({rank} < {X.shape[1]}). "
            "Reduce harmonics or check the training window."
        )
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    if not np.all(np.isfinite(beta)):
        raise RuntimeError(f"{label}: non-finite coefficients.")
    return beta


def _scale_design(frame: pd.DataFrame, spec: FeatureSpec, use_weekend: bool) -> tuple[np.ndarray, list[str]]:
    if spec.scale == "constant":
        return design_matrix(frame, spec.clock, 0, False, 0)
    if spec.scale == "clock":
        return design_matrix(frame, spec.clock, spec.scale_harmonics, use_weekend, 0)
    raise ValueError(f"Unknown scale mode: {spec.scale}")


def training_shortfall(train: pd.DataFrame, spec: FeatureSpec) -> str | None:
    """Why a feature cannot be fitted on these training rows (None if it can)."""

    if spec.column not in train:
        return "column missing"
    y = transformed_values(train, spec).dropna()
    if len(y) < MIN_TRAIN_ROWS:
        raw = pd.to_numeric(train[spec.column], errors="coerce").notna().sum()
        return (f"{len(y)} usable training rows < {MIN_TRAIN_ROWS} "
                f"({int(raw)} non-null before scope/transform)")
    if np.unique(np.round(y.to_numpy(float), 12)).size < 3 or float(y.std()) <= 0:
        return "feature is (nearly) constant in the training window"
    return None


def weekend_identified(train: pd.DataFrame) -> bool:
    w = train["clock_weekend"] > 0
    days = pd.to_datetime(train.loc[w, "state_time"], utc=True).dt.floor("D").nunique()
    return bool(int(w.sum()) >= MIN_WEEKEND_STATES and days >= MIN_WEEKEND_DAYS and int((~w).sum()) > 0)


def fit_feature_baseline(
    train: pd.DataFrame,
    spec: FeatureSpec,
    underlying: str,
) -> FittedFeatureBaseline | None:
    """Fit one feature for one asset on training rows (with clock
    coordinates). Returns None when the training data are insufficient."""

    if training_shortfall(train, spec) is not None:
        return None

    y_all = transformed_values(train, spec)
    rows = train[y_all.notna()]
    y = y_all.dropna().to_numpy(float)

    use_weekend = weekend_identified(rows)
    label = f"{underlying}/{spec.name}"

    X, names = design_matrix(rows, spec.clock, spec.harmonics, use_weekend, spec.weekend_harmonics)
    beta = _ols(y, X, label)
    m = X @ beta
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - m) ** 2).sum()) / sst if sst > 0 else None

    fitted = FittedFeatureBaseline(
        spec=asdict(spec),
        underlying=underlying,
        location_terms=names,
        location_coef=beta.tolist(),
        weekend_terms_used=bool(use_weekend),
        n_train=int(len(y)),
        train_r2_location=None if r2 is None else float(r2),
        train_ny_offsets=sorted(int(v) for v in rows["clock_ny_utc_offset_min"].unique()),
    )

    if spec.transform == "log_abs":
        # Normalize over all training rows of the asset (zero returns
        # included), so mean f^2 = 1 where the feature is observed.
        defined = train[pd.to_numeric(train[spec.column], errors="coerce").notna()
                        & scope_mask(train, spec.scope)]
        Xd, _ = design_matrix(defined, spec.clock, spec.harmonics, use_weekend, spec.weekend_harmonics)
        fitted.vol_factor_const = float(0.5 * np.log(np.mean(np.exp(2.0 * (Xd @ beta)))))
        return fitted

    if spec.scale != "none":
        resid2 = (y - m) ** 2
        floor = max(1e-12, 1e-6 * float(resid2.mean()))
        Xs, snames = _scale_design(rows, spec, use_weekend)
        gamma = _ols(np.log(resid2 + floor), Xs, label + " (scale)")
        base = np.exp(0.5 * (Xs @ gamma))
        fitted.scale_terms = snames
        fitted.scale_coef = gamma.tolist()
        fitted.scale_const = float(0.5 * np.log(np.mean(resid2 / base ** 2)))

    return fitted


@dataclass
class ClockBaselineSet:
    """All fitted feature baselines for one training window."""

    baseline_id: str
    version: str
    train_start: str
    train_end: str
    created_at: str
    fitted: list[FittedFeatureBaseline]
    skipped: list[dict] = field(default_factory=list)

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        train_mask: pd.Series,
        specs: tuple[FeatureSpec, ...] = DEFAULT_FEATURE_SPECS,
        baseline_id: str | None = None,
    ) -> "ClockBaselineSet":
        """Fit every spec per asset on frame[train_mask]."""

        if frame["state_id"].duplicated().any():
            raise ValueError("Duplicate state_id in clock baseline input.")

        data = add_clock_coordinates(frame)
        train = data[train_mask.reindex(data.index).fillna(False).astype(bool)]
        if train.empty:
            raise ValueError("Empty training window.")

        t0 = pd.to_datetime(train["state_time"], utc=True).min()
        t1 = pd.to_datetime(train["state_time"], utc=True).max()
        fitted, skipped = [], []
        for asset, g in train.groupby("underlying"):
            for spec in specs:
                reason = training_shortfall(g, spec)
                if reason is not None:
                    skipped.append({"underlying": asset, "feature": spec.name, "reason": reason})
                    continue
                fitted.append(fit_feature_baseline(g, spec, str(asset)))

        spec_hash = hashlib.sha256(
            json.dumps([asdict(s) for s in specs], sort_keys=True).encode()
        ).hexdigest()[:8]
        bid = baseline_id or (
            f"clock_{CLOCK_BASELINE_VERSION}_{t0:%Y%m%dT%H%M}_{t1:%Y%m%dT%H%M}_{spec_hash}"
        )
        return cls(
            baseline_id=bid,
            version=CLOCK_BASELINE_VERSION,
            train_start=t0.isoformat(),
            train_end=t1.isoformat(),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            fitted=fitted,
            skipped=skipped,
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Adjusted features for every row of `frame` (any window). Adds:

          clock_baseline_id, clock_baseline_in_sample (state inside the
          training window), clock_dst_unseen (New York UTC offset not seen
          in training for that asset: the New York clock is extrapolated).
        """

        data = add_clock_coordinates(frame)
        t = pd.to_datetime(data["state_time"], utc=True)
        out = data[["state_id", "state_time", "underlying"]].copy()
        out["clock_baseline_id"] = self.baseline_id
        out["clock_baseline_in_sample"] = (
            (t >= pd.Timestamp(self.train_start)) & (t <= pd.Timestamp(self.train_end))
        )
        out["clock_dst_unseen"] = False

        parts = []
        for asset, g in data.groupby("underlying"):
            fbs = [fb for fb in self.fitted if fb.underlying == asset]
            if not fbs:
                continue
            seen = set(fbs[0].train_ny_offsets)
            out.loc[g.index, "clock_dst_unseen"] = ~g["clock_ny_utc_offset_min"].isin(seen)
            parts.append(pd.concat([fb.transform(g) for fb in fbs], axis=1))

        if parts:
            out = out.join(pd.concat(parts, axis=0))
        return out

    def to_dict(self) -> dict:
        return {
            "baseline_id": self.baseline_id,
            "version": self.version,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "created_at": self.created_at,
            "fitted": [asdict(fb) for fb in self.fitted],
            "skipped": self.skipped,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1)

    @classmethod
    def from_dict(cls, d: dict) -> "ClockBaselineSet":
        if d.get("version") != CLOCK_BASELINE_VERSION:
            raise ValueError(
                f"Clock baseline version {d.get('version')} != {CLOCK_BASELINE_VERSION}."
            )
        return cls(
            baseline_id=d["baseline_id"],
            version=d["version"],
            train_start=d["train_start"],
            train_end=d["train_end"],
            created_at=d["created_at"],
            fitted=[FittedFeatureBaseline(**fb) for fb in d["fitted"]],
            skipped=d.get("skipped", []),
        )

    @classmethod
    def from_json(cls, text: str) -> "ClockBaselineSet":
        return cls.from_dict(json.loads(text))
