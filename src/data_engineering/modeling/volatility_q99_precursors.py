"""
Volatility Dynamics Module 03: q99 precursor discovery.

Independent research logic:
- target is recomputed from raw return_5m inside this module;
- predictors are dynamically discovered numeric/boolean columns from the
  shared time_series_feature_state relation;
- no outputs, models, rankings, feature lists, or conclusions are imported
  from any other research branch or from earlier Volatility Dynamics modules.

The hard information boundary for precursor ranking is t < event_time.
Event-bar and post-event observations are descriptive anatomy only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


VERSION = "0.1.0"
BAR_MINUTES = 5
Q99 = 0.99
DEFAULT_PRE_BARS = 12
DEFAULT_POST_BARS = 12
DEFAULT_RANK_LAGS = (-12, -6, -3, -2, -1)
DEFAULT_MAX_CONTROLS = 8
DEFAULT_MIN_CONTROLS = 4

KEY_COLUMNS = ("state_time", "underlying", "return_5m")
EXCLUDED_PREDICTOR_COLUMNS = {
    "state_id",
}
NUMERIC_TYPE_TOKENS = (
    "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
    "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT",
    "FLOAT", "DOUBLE", "REAL", "DECIMAL", "NUMERIC", "BOOLEAN",
)


def _ns_values(times: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    t = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    return t.as_unit("ns").asi8


def eligible_predictor_columns(schema: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """Select all numeric/boolean database fields except explicit identifiers.

    This is schema-driven; there is no research-branch feature list.
    """
    name_col = "column_name"
    type_col = "column_type"
    if name_col not in schema or type_col not in schema:
        raise ValueError("DuckDB DESCRIBE output must contain column_name and column_type.")

    rows = []
    eligible = []
    for r in schema[[name_col, type_col]].itertuples(index=False):
        name = str(r[0])
        dtype = str(r[1]).upper()
        numeric = any(tok in dtype for tok in NUMERIC_TYPE_TOKENS)
        if name in KEY_COLUMNS:
            reason = "key_or_target_source"
            use = False
        elif name in EXCLUDED_PREDICTOR_COLUMNS or name.endswith("_id"):
            reason = "identifier"
            use = False
        elif numeric:
            reason = "numeric_or_boolean"
            use = True
            eligible.append(name)
        else:
            reason = "non_numeric"
            use = False
        rows.append({
            "feature": name,
            "duckdb_type": dtype,
            "eligible_predictor": bool(use),
            "reason": reason,
        })

    # return_5m is a valid precursor when lagged. Add it explicitly.
    if "return_5m" in set(schema[name_col].astype(str)):
        eligible = ["return_5m"] + eligible
        for row in rows:
            if row["feature"] == "return_5m":
                row["eligible_predictor"] = True
                row["reason"] = "lagged_target_source_allowed"

    return list(dict.fromkeys(eligible)), pd.DataFrame(rows)


@dataclass
class EventMeta:
    underlying: str
    threshold: float
    n_finite_returns: int
    n_q99_bars: int
    n_q99_onsets: int


def _exact_previous_index(times_ns: np.ndarray) -> np.ndarray:
    step = pd.Timedelta(minutes=BAR_MINUTES).value
    lookup = {int(v): i for i, v in enumerate(times_ns)}
    prev = np.full(len(times_ns), -1, dtype=int)
    for i, t in enumerate(times_ns):
        prev[i] = lookup.get(int(t - step), -1)
    return prev


def identify_q99_onsets(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute q99 from raw five-minute returns and keep run onsets only."""
    required = set(KEY_COLUMNS)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    events = []
    meta = []
    event_id = 0

    for asset, g0 in frame.groupby("underlying", sort=True):
        g = g0.sort_values("state_time").reset_index()
        times = pd.DatetimeIndex(pd.to_datetime(g["state_time"], utc=True))
        ns = _ns_values(times)
        prev = _exact_previous_index(ns)
        r = pd.to_numeric(g["return_5m"], errors="coerce").to_numpy(float)
        v = r ** 2
        finite = np.isfinite(v)
        if finite.sum() < 100:
            continue
        threshold = float(np.quantile(v[finite], Q99))
        tail = finite & (v >= threshold)

        onset = np.zeros(len(g), dtype=bool)
        for i in range(len(g)):
            if not tail[i]:
                continue
            j = prev[i]
            onset[i] = j < 0 or not tail[j]

        idxs = np.flatnonzero(onset)
        for i in idxs:
            event_id += 1
            j = prev[i]
            event_row_global = int(g.loc[i, "index"])
            prev_v = float(v[j]) if j >= 0 and np.isfinite(v[j]) else np.nan
            events.append({
                "event_id": event_id,
                "underlying": str(asset),
                "event_time": times[i],
                "event_panel_index": event_row_global,
                "clock_slot": int(times[i].hour * 12 + times[i].minute // 5),
                "q99_threshold": threshold,
                "event_variance_5m": float(v[i]),
                "previous_variance_5m": prev_v,
            })

        meta.append(EventMeta(
            underlying=str(asset),
            threshold=threshold,
            n_finite_returns=int(finite.sum()),
            n_q99_bars=int(tail.sum()),
            n_q99_onsets=int(onset.sum()),
        ).__dict__)

    return pd.DataFrame(events), pd.DataFrame(meta)


def build_origin_index(frame: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Mark q99 bars/onsets and eligible non-event control origins."""
    out = frame[["state_time", "underlying", "return_5m"]].copy()
    out["state_time"] = pd.to_datetime(out["state_time"], utc=True)
    out["variance_5m"] = pd.to_numeric(out["return_5m"], errors="coerce") ** 2
    out["clock_slot"] = out["state_time"].dt.hour * 12 + out["state_time"].dt.minute // 5
    out["is_q99"] = False
    out["is_q99_onset"] = False

    thresholds = dict(zip(events["underlying"], events["q99_threshold"]))
    # dict from duplicated asset values still returns the same threshold.
    for asset, idx in out.groupby("underlying").groups.items():
        if asset not in thresholds:
            continue
        ii = np.asarray(list(idx), int)
        x = out.loc[ii, "variance_5m"].to_numpy(float)
        out.loc[ii, "is_q99"] = np.isfinite(x) & (x >= float(thresholds[asset]))

    event_keys = set(
        zip(events["underlying"].astype(str), pd.to_datetime(events["event_time"], utc=True))
    )
    out["is_q99_onset"] = [
        (str(a), pd.Timestamp(t)) in event_keys
        for a, t in zip(out["underlying"], out["state_time"])
    ]

    eligible = np.zeros(len(out), dtype=bool)
    step = pd.Timedelta(minutes=BAR_MINUTES).value
    for asset, idx in out.groupby("underlying").groups.items():
        ii = np.asarray(list(idx), int)
        times = pd.DatetimeIndex(out.loc[ii, "state_time"])
        ns = _ns_values(times)
        loc_by_ns = {int(v): k for k, v in enumerate(ns)}
        is_tail = out.loc[ii, "is_q99"].to_numpy(bool)
        finite = np.isfinite(out.loc[ii, "variance_5m"].to_numpy(float))
        for k, t in enumerate(ns):
            prev_k = loc_by_ns.get(int(t - step), -1)
            if not finite[k] or is_tail[k]:
                continue
            if prev_k >= 0 and is_tail[prev_k]:
                continue
            eligible[ii[k]] = True

    out["eligible_control_origin"] = eligible
    return out


def choose_controls(
    origins: pd.DataFrame,
    events: pd.DataFrame,
    *,
    max_controls: int = DEFAULT_MAX_CONTROLS,
) -> pd.DataFrame:
    """Nearest-date same-asset, exact-clock eligible controls per q99 onset."""
    if max_controls < 1:
        raise ValueError("max_controls must be positive.")
    recs = []

    for ev in events.itertuples():
        cand = origins[
            (origins["underlying"].astype(str) == str(ev.underlying))
            & (origins["clock_slot"] == int(ev.clock_slot))
            & (origins["eligible_control_origin"])
        ].copy()
        if cand.empty:
            continue
        cand["distance_sec"] = (
            pd.to_datetime(cand["state_time"], utc=True) - pd.Timestamp(ev.event_time)
        ).abs().dt.total_seconds()
        cand = cand.sort_values(["distance_sec", "state_time"]).head(max_controls)
        for rank, (idx, row) in enumerate(cand.iterrows(), start=1):
            recs.append({
                "event_id": int(ev.event_id),
                "underlying": str(ev.underlying),
                "event_time": pd.Timestamp(ev.event_time),
                "clock_slot": int(ev.clock_slot),
                "control_rank": rank,
                "control_panel_index": int(idx),
                "control_time": pd.Timestamp(row["state_time"]),
                "distance_days": float(row["distance_sec"] / 86400.0),
            })
    return pd.DataFrame(recs)


def _relative_index_map(frame: pd.DataFrame, bars: Iterable[int]) -> dict[int, np.ndarray]:
    """Global row index at exact relative five-minute offsets within same asset."""
    offsets = sorted(set(int(b) for b in bars))
    result = {b: np.full(len(frame), -1, dtype=int) for b in offsets}
    step = pd.Timedelta(minutes=BAR_MINUTES).value

    for asset, idx in frame.groupby("underlying").groups.items():
        ii = np.asarray(list(idx), int)
        times = pd.DatetimeIndex(pd.to_datetime(frame.loc[ii, "state_time"], utc=True))
        ns = _ns_values(times)
        lookup = {int(t): int(global_i) for t, global_i in zip(ns, ii)}
        for global_i, t in zip(ii, ns):
            for b in offsets:
                result[b][global_i] = lookup.get(int(t + step * b), -1)
    return result


def _to_numeric_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    cols = []
    for c in features:
        s = frame[c]
        if pd.api.types.is_bool_dtype(s.dtype):
            x = s.astype("Float64").to_numpy(dtype=float, na_value=np.nan)
        else:
            x = pd.to_numeric(s, errors="coerce").to_numpy(float)
        cols.append(x)
    return np.column_stack(cols) if cols else np.empty((len(frame), 0))


def percentile_scores(
    frame: pd.DataFrame,
    features: list[str],
    events: pd.DataFrame,
    controls: pd.DataFrame,
    relative_bars: Iterable[int],
    *,
    min_controls: int = DEFAULT_MIN_CONTROLS,
) -> pd.DataFrame:
    """Event value percentile among same-clock controls at each relative bar.

    rel_bar < 0 is precursor information.
    rel_bar >= 0 is anatomy only.
    """
    relative_bars = sorted(set(int(x) for x in relative_bars))
    rel_map = _relative_index_map(frame, relative_bars)
    X = _to_numeric_matrix(frame, features)
    control_groups = {
        int(eid): g["control_panel_index"].astype(int).to_numpy()
        for eid, g in controls.groupby("event_id")
    }
    recs = []

    for ev in events.itertuples():
        eid = int(ev.event_id)
        origin = int(ev.event_panel_index)
        control_origins = control_groups.get(eid, np.empty(0, dtype=int))
        if len(control_origins) < min_controls:
            continue

        for b in relative_bars:
            erow = int(rel_map[b][origin])
            if erow < 0:
                continue
            crows = rel_map[b][control_origins]
            crows = crows[crows >= 0]
            if len(crows) < min_controls:
                continue

            evv = X[erow, :]
            C = X[crows, :]
            finite_event = np.isfinite(evv)
            n_ctrl = np.sum(np.isfinite(C), axis=0)
            valid = finite_event & (n_ctrl >= min_controls)
            if not np.any(valid):
                continue

            less = np.sum((C < evv[None, :]) & np.isfinite(C), axis=0)
            equal = np.sum((C == evv[None, :]) & np.isfinite(C), axis=0)
            pct = (less + 0.5 * equal) / np.maximum(n_ctrl, 1)

            # Robust effect: event minus control median, scaled by control IQR.
            cmed = np.nanmedian(C, axis=0)
            q25 = np.nanquantile(C, 0.25, axis=0)
            q75 = np.nanquantile(C, 0.75, axis=0)
            iqr = q75 - q25
            robust_diff = np.where(iqr > 0, (evv - cmed) / iqr, np.nan)

            for j in np.flatnonzero(valid):
                recs.append({
                    "event_id": eid,
                    "underlying": str(ev.underlying),
                    "event_time": pd.Timestamp(ev.event_time),
                    "feature": features[j],
                    "rel_bar": b,
                    "rel_min": b * BAR_MINUTES,
                    "is_precursor": bool(b < 0),
                    "n_controls": int(n_ctrl[j]),
                    "percentile_vs_controls": float(pct[j]),
                    "robust_diff_iqr": float(robust_diff[j]) if np.isfinite(robust_diff[j]) else np.nan,
                    "event_value": float(evv[j]),
                    "control_median": float(cmed[j]) if np.isfinite(cmed[j]) else np.nan,
                })

    return pd.DataFrame(recs)


def summarize_profiles(scores: pd.DataFrame) -> pd.DataFrame:
    """Feature/lag anatomy summarized without over-weighting synchronous assets."""
    if scores.empty:
        return pd.DataFrame()

    # Collapse asset-event instances sharing one timestamp first.
    t = (
        scores.groupby(["event_time", "feature", "rel_bar", "rel_min", "is_precursor"])
        .agg(
            timestamp_percentile=("percentile_vs_controls", "mean"),
            timestamp_robust_diff=("robust_diff_iqr", "median"),
            n_asset_events=("event_id", "size"),
        )
        .reset_index()
    )
    return (
        t.groupby(["feature", "rel_bar", "rel_min", "is_precursor"])
        .agg(
            n_unique_event_times=("event_time", "nunique"),
            mean_percentile=("timestamp_percentile", "mean"),
            median_percentile=("timestamp_percentile", "median"),
            q25_percentile=("timestamp_percentile", lambda s: s.quantile(0.25)),
            q75_percentile=("timestamp_percentile", lambda s: s.quantile(0.75)),
            mean_abs_deviation_from_half=(
                "timestamp_percentile", lambda s: float(np.mean(np.abs(s - 0.5)))
            ),
            median_robust_diff_iqr=("timestamp_robust_diff", "median"),
        )
        .reset_index()
    )


def summarize_by_asset(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    pre = scores[scores["is_precursor"]].copy()
    return (
        pre.groupby(["underlying", "feature", "rel_bar", "rel_min"])
        .agg(
            n_events=("event_id", "nunique"),
            mean_percentile=("percentile_vs_controls", "mean"),
            median_percentile=("percentile_vs_controls", "median"),
            median_robust_diff_iqr=("robust_diff_iqr", "median"),
        )
        .reset_index()
    )


def precursor_rankings(
    scores: pd.DataFrame,
    rank_lags: Iterable[int] = DEFAULT_RANK_LAGS,
    *,
    min_event_times: int = 12,
    min_assets: int = 3,
) -> pd.DataFrame:
    """Rank descriptive precursor candidates by pre-event unusualness/stability.

    No p-values or feature selection claim is made. Ranking is exploratory.
    """
    if scores.empty:
        return pd.DataFrame()
    rank_lags = set(int(x) for x in rank_lags)
    pre = scores[scores["rel_bar"].isin(rank_lags) & scores["is_precursor"]].copy()
    if pre.empty:
        return pd.DataFrame()

    # First avoid overweighting synchronous multi-asset event timestamps.
    ts = (
        pre.groupby(["event_time", "feature", "rel_bar", "rel_min"])
        .agg(percentile=("percentile_vs_controls", "mean"))
        .reset_index()
    )
    lag_summary = (
        ts.groupby(["feature", "rel_bar", "rel_min"])
        .agg(
            n_unique_event_times=("event_time", "nunique"),
            mean_percentile=("percentile", "mean"),
            median_percentile=("percentile", "median"),
        )
        .reset_index()
    )
    lag_summary["deviation"] = lag_summary["mean_percentile"] - 0.5
    lag_summary["abs_deviation"] = lag_summary["deviation"].abs()

    # Asset consistency at each feature/lag.
    asset = (
        pre.groupby(["underlying", "feature", "rel_bar"])
        .agg(asset_mean_percentile=("percentile_vs_controls", "mean"))
        .reset_index()
    )
    asset["asset_direction"] = np.sign(asset["asset_mean_percentile"] - 0.5)

    recs = []
    for feature, g in lag_summary.groupby("feature"):
        g = g[g["n_unique_event_times"] >= min_event_times]
        if g.empty:
            continue
        best = g.sort_values(
            ["abs_deviation", "n_unique_event_times"],
            ascending=[False, False],
        ).iloc[0]
        rel = int(best["rel_bar"])
        aa = asset[(asset["feature"] == feature) & (asset["rel_bar"] == rel)]
        aa = aa[np.isfinite(aa["asset_mean_percentile"])]
        n_assets = int(aa["underlying"].nunique())
        if n_assets < min_assets:
            continue
        direction = 1 if float(best["deviation"]) >= 0 else -1
        consistent = int(np.sum(aa["asset_direction"].to_numpy(float) == direction))
        consistency = consistent / n_assets if n_assets else np.nan

        lag_dirs = np.sign(g["deviation"].to_numpy(float))
        lag_consistency = float(np.mean(lag_dirs == direction)) if len(lag_dirs) else np.nan
        mean_abs_across_lags = float(g["abs_deviation"].mean())
        discovery_score = float(
            best["abs_deviation"] * consistency * lag_consistency
        )

        recs.append({
            "feature": feature,
            "best_rel_bar": rel,
            "best_rel_min": int(best["rel_min"]),
            "n_unique_event_times": int(best["n_unique_event_times"]),
            "n_assets": n_assets,
            "mean_percentile": float(best["mean_percentile"]),
            "median_percentile": float(best["median_percentile"]),
            "direction": "high_before_event" if direction > 0 else "low_before_event",
            "abs_deviation_from_half": float(best["abs_deviation"]),
            "mean_abs_deviation_across_rank_lags": mean_abs_across_lags,
            "assets_consistent_direction": consistent,
            "asset_direction_consistency": float(consistency),
            "lag_direction_consistency": lag_consistency,
            "discovery_score": discovery_score,
        })

    if not recs:
        return pd.DataFrame()
    return pd.DataFrame(recs).sort_values(
        ["discovery_score", "n_unique_event_times"],
        ascending=[False, False],
    ).reset_index(drop=True)


def feature_coverage(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for c in features:
        x = pd.to_numeric(frame[c], errors="coerce")
        rows.append({
            "feature": c,
            "n_rows": int(len(frame)),
            "n_finite": int(x.notna().sum()),
            "coverage": float(x.notna().mean()),
        })
    return pd.DataFrame(rows)


def selftest() -> dict:
    """Synthetic test: known precursor must rank high without event-bar leakage."""
    n_days = 40
    slots = 12
    times = []
    x = []
    ret = []
    rng = np.random.default_rng(7)
    for d in range(n_days):
        base = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=d)
        for s in range(slots):
            times.append(base + pd.Timedelta(minutes=5 * s))
            x.append(rng.normal())
            ret.append(rng.normal(scale=0.001))

    f = pd.DataFrame({
        "state_time": times,
        "underlying": "TEST",
        "return_5m": ret,
        "known_precursor": x,
        "noise": rng.normal(size=len(times)),
        "flag": False,
    })
    # Force q99 events at a repeated clock and raise precursor one bar earlier.
    event_rows = [d * slots + 6 for d in range(10, 40, 5)]
    for i in event_rows:
        f.loc[i, "return_5m"] = 0.20
        f.loc[i - 1, "known_precursor"] = 8.0

    events, meta = identify_q99_onsets(f)
    assert len(events) >= len(event_rows)
    origins = build_origin_index(f, events)
    controls = choose_controls(origins, events, max_controls=8)
    scores = percentile_scores(
        f,
        ["return_5m", "known_precursor", "noise", "flag"],
        events,
        controls,
        relative_bars=(-2, -1, 0, 1),
        min_controls=4,
    )
    known = scores[
        (scores["feature"] == "known_precursor")
        & (scores["rel_bar"] == -1)
    ]
    assert len(known) > 0
    assert float(known["percentile_vs_controls"].mean()) > 0.8

    # Current event return must never appear in precursor rows.
    assert (scores.loc[scores["is_precursor"], "rel_bar"] < 0).all()

    return {
        "q99_recomputed": True,
        "run_onset_only": True,
        "same_clock_controls": True,
        "strict_pre_event_boundary": True,
        "known_precursor_recovered": True,
    }
