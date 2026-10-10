"""
Volatility Dynamics Module 05: the -95 minute precursor motif.

This module was motivated by a visual pattern discovered after extending the
q99 event anatomy to +/-120 minutes. Because the pattern was discovered in the
same sample, Module 05 is explicitly exploratory characterization, not an
out-of-sample validation.

The object of study is a five-candle motif window:
    -105, -100, -95, -90, -85 minutes before q99 onset.

Key protections:
- q99 is recomputed from native return_5m**2;
- controls are same-asset / exact-clock and native-volatility matched, as in
  Module 04;
- events with ANY prior q99 bar in the preceding 120 minutes are separated
  from the clean sample, so the motif is not merely the decay/recovery of an
  earlier extreme-volatility event;
- simultaneous multi-asset event timestamps are collapsed before pooled
  summaries;
- no predictive or trading claim is made.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from data_engineering.modeling import volatility_q99_precursors as qp


VERSION = "0.1.0"
BAR_MINUTES = 5
PRIOR_Q99_LOOKBACK_MIN = 120

PROFILE_BARS = tuple(range(-24, 1))  # -120m through event candle.
MOTIF_BARS = (-21, -20, -19, -18, -17)  # -105,-100,-95,-90,-85
MOTIF_PRE_BARS = (-21, -20)
MOTIF_CENTER_BAR = -19
MOTIF_POST_BARS = (-18, -17)

ACTIVITY_FEATURES = (
    "intrabar_range_bps",
    "volume_usd",
    "trade_count",
)

IV_FEATURES = (
    "option_iv_atm_forward",
    "iv_put_25d",
    "iv_call_25d",
)

LIQUIDITY_FEATURES = (
    "book_twa_spread_bps",
    "book_update_count",
)

L2_FEATURES = (
    "book_obi_20_raw_zero_cross_count",
    "book_obi_20_transition_count",
    "book_obi_20_raw_zero_cross_rate",
    "book_obi_20_path_roughness_per_min",
)

ALL_MOTIF_FEATURES = tuple(dict.fromkeys(
    ACTIVITY_FEATURES + IV_FEATURES + LIQUIDITY_FEATURES + L2_FEATURES
))


@dataclass(frozen=True)
class MotifFeatureDef:
    feature: str
    family: str
    reason: str


FEATURE_DEFS = tuple(
    [MotifFeatureDef(f, "activity", "Module 03/04 broad activity or native-range precursor") for f in ACTIVITY_FEATURES]
    + [MotifFeatureDef(f, "iv_level", "Module 03/04 elevated option-IV level") for f in IV_FEATURES]
    + [MotifFeatureDef(f, "liquidity_book_activity", "Module 03/04 spread or book-activity precursor") for f in LIQUIDITY_FEATURES]
    + [MotifFeatureDef(f, "l2_instability", "Module 03/04 sparse L2 instability precursor") for f in L2_FEATURES]
)


def motif_contract(available_columns: Iterable[str]) -> pd.DataFrame:
    available = set(map(str, available_columns))
    return pd.DataFrame([
        {
            "feature": d.feature,
            "family": d.family,
            "reason": d.reason,
            "present": d.feature in available,
        }
        for d in FEATURE_DEFS
    ])


def _ns_values(times: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    t = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    return t.as_unit("ns").asi8


def prior_q99_history(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    *,
    lookback_min: int = PRIOR_Q99_LOOKBACK_MIN,
) -> pd.DataFrame:
    """For every row, identify whether a prior q99 bar occurred in lookback.

    Uses q99 thresholds carried by the independently recomputed event table.
    The current row itself is excluded from the lookback.
    """
    required = {"state_time", "underlying", "return_5m"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if events.empty:
        raise ValueError("events is empty")

    thresholds = (
        events.groupby("underlying")["q99_threshold"].first().astype(float).to_dict()
    )
    out = pd.DataFrame(index=frame.index)
    out["prior_q99_bar_lookback"] = False
    out["n_prior_q99_bars_lookback"] = 0
    out["minutes_since_latest_prior_q99"] = np.nan

    lookback_ns = pd.Timedelta(minutes=int(lookback_min)).value

    for asset, idx in frame.groupby("underlying").groups.items():
        if asset not in thresholds:
            continue
        ii = np.asarray(list(idx), dtype=int)
        g = frame.loc[ii]
        times = pd.DatetimeIndex(pd.to_datetime(g["state_time"], utc=True))
        ns = _ns_values(times)
        r = pd.to_numeric(g["return_5m"], errors="coerce").to_numpy(float)
        v = r ** 2
        tail = np.isfinite(v) & (v >= float(thresholds[asset]))

        tail_positions: list[int] = []
        left = 0
        for k, t in enumerate(ns):
            while left < len(tail_positions) and ns[tail_positions[left]] < t - lookback_ns:
                left += 1
            active = tail_positions[left:]
            if active:
                out.at[ii[k], "prior_q99_bar_lookback"] = True
                out.at[ii[k], "n_prior_q99_bars_lookback"] = int(len(active))
                latest = active[-1]
                out.at[ii[k], "minutes_since_latest_prior_q99"] = float(
                    (t - ns[latest]) / 60e9
                )
            if tail[k]:
                tail_positions.append(k)

    return out


def clean_event_and_origin_sets(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    origins: pd.DataFrame,
    *,
    lookback_min: int = PRIOR_Q99_LOOKBACK_MIN,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separate q99 events with/without an earlier q99 bar in lookback."""
    hist = prior_q99_history(frame, events, lookback_min=lookback_min)

    event_rows = []
    for ev in events.itertuples():
        i = int(ev.event_panel_index)
        event_rows.append({
            "event_id": int(ev.event_id),
            "underlying": str(ev.underlying),
            "event_time": pd.Timestamp(ev.event_time),
            "prior_q99_bar_lookback": bool(hist.at[i, "prior_q99_bar_lookback"]),
            "n_prior_q99_bars_lookback": int(hist.at[i, "n_prior_q99_bars_lookback"]),
            "minutes_since_latest_prior_q99": float(hist.at[i, "minutes_since_latest_prior_q99"])
                if pd.notna(hist.at[i, "minutes_since_latest_prior_q99"]) else np.nan,
        })
    audit = pd.DataFrame(event_rows)

    clean_ids = set(
        audit.loc[~audit["prior_q99_bar_lookback"], "event_id"].astype(int)
    )
    clean_events = events[events["event_id"].astype(int).isin(clean_ids)].copy()

    clean_origins = origins.copy()
    clean_origins["prior_q99_bar_lookback"] = hist["prior_q99_bar_lookback"].astype(bool)
    clean_origins["eligible_control_origin"] = (
        clean_origins["eligible_control_origin"].astype(bool)
        & ~clean_origins["prior_q99_bar_lookback"].astype(bool)
    )
    return clean_events, clean_origins, audit


def local_motif_event_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Unique-event-time motif contrasts for every feature.

    Simultaneous assets are averaged first. This avoids counting one common
    market timestamp multiple times in pooled motif summaries.
    """
    if scores.empty:
        return pd.DataFrame()

    use = scores[scores["rel_bar"].isin(MOTIF_BARS)].copy()
    if use.empty:
        return pd.DataFrame()

    ts = (
        use.groupby(["event_time", "feature", "rel_bar", "rel_min"])
        .agg(
            percentile=("percentile_vs_controls", "mean"),
            robust_diff_iqr=("robust_diff_iqr", "median"),
            n_asset_events=("event_id", "size"),
        )
        .reset_index()
    )

    rows = []
    for (event_time, feature), g in ts.groupby(["event_time", "feature"]):
        p = dict(zip(g["rel_bar"].astype(int), g["percentile"].astype(float)))
        r = dict(zip(g["rel_bar"].astype(int), g["robust_diff_iqr"].astype(float)))
        if not all(b in p for b in MOTIF_BARS):
            continue

        pre = float(np.mean([p[b] for b in MOTIF_PRE_BARS]))
        center = float(p[MOTIF_CENTER_BAR])
        post = float(np.mean([p[b] for b in MOTIF_POST_BARS]))
        window = float(np.mean([p[b] for b in MOTIF_BARS]))

        rows.append({
            "event_time": pd.Timestamp(event_time),
            "feature": feature,
            "n_asset_events": int(g["n_asset_events"].max()),
            "pre_mean_percentile": pre,
            "center_95m_percentile": center,
            "post_mean_percentile": post,
            "window_mean_percentile": window,
            "center_vs_neighbors": center - 0.5 * (pre + post),
            "pre_to_center": center - pre,
            "center_to_post_rebound": post - center,
            "pre_to_post": post - pre,
            "center_95m_robust_diff_iqr": (
                float(r[MOTIF_CENTER_BAR])
                if MOTIF_CENTER_BAR in r and np.isfinite(r[MOTIF_CENTER_BAR])
                else np.nan
            ),
        })

    return pd.DataFrame(rows)


def summarize_local_motif(local: pd.DataFrame) -> pd.DataFrame:
    if local.empty:
        return pd.DataFrame()

    rows = []
    for feature, g in local.groupby("feature"):
        rows.append({
            "feature": feature,
            "n_unique_event_times": int(g["event_time"].nunique()),
            "mean_pre_percentile": float(g["pre_mean_percentile"].mean()),
            "mean_center_95m_percentile": float(g["center_95m_percentile"].mean()),
            "mean_post_percentile": float(g["post_mean_percentile"].mean()),
            "mean_window_percentile": float(g["window_mean_percentile"].mean()),
            "median_window_percentile": float(g["window_mean_percentile"].median()),
            "mean_center_vs_neighbors": float(g["center_vs_neighbors"].mean()),
            "mean_rebound_95_to_post": float(g["center_to_post_rebound"].mean()),
            "median_rebound_95_to_post": float(g["center_to_post_rebound"].median()),
            "share_window_above_half": float(np.mean(g["window_mean_percentile"] > 0.5)),
            "share_center_below_half": float(np.mean(g["center_95m_percentile"] < 0.5)),
            "share_positive_rebound": float(np.mean(g["center_to_post_rebound"] > 0)),
        })
    return pd.DataFrame(rows)


def _event_asset_feature_grid(scores: pd.DataFrame) -> pd.DataFrame:
    """One row per asset-event/feature with complete five-bar motif."""
    use = scores[scores["rel_bar"].isin(MOTIF_BARS)].copy()
    rows = []
    for (event_id, underlying, event_time, feature), g in use.groupby(
        ["event_id", "underlying", "event_time", "feature"]
    ):
        p = dict(zip(g["rel_bar"].astype(int), g["percentile_vs_controls"].astype(float)))
        if not all(b in p for b in MOTIF_BARS):
            continue
        pre = float(np.mean([p[b] for b in MOTIF_PRE_BARS]))
        center = float(p[MOTIF_CENTER_BAR])
        post = float(np.mean([p[b] for b in MOTIF_POST_BARS]))
        rows.append({
            "event_id": int(event_id),
            "underlying": str(underlying),
            "event_time": pd.Timestamp(event_time),
            "feature": feature,
            "pre": pre,
            "center": center,
            "post": post,
            "window": float(np.mean([p[b] for b in MOTIF_BARS])),
            "notch": center - 0.5 * (pre + post),
            "rebound": post - center,
        })
    return pd.DataFrame(rows)


def motif_component_scores(
    scores: pd.DataFrame,
    contract: pd.DataFrame,
) -> pd.DataFrame:
    """Per asset-event family scores.

    Level components are mean percentile across features and all five motif
    candles. L2 rebound measures the post(-90/-85) minus -95 percentile.
    """
    grid = _event_asset_feature_grid(scores)
    if grid.empty:
        return pd.DataFrame()

    family_map = dict(zip(contract["feature"], contract["family"]))
    grid["family"] = grid["feature"].map(family_map)
    grid = grid[grid["family"].notna()].copy()

    rows = []
    for (event_id, asset, event_time), g in grid.groupby(
        ["event_id", "underlying", "event_time"]
    ):
        rec = {
            "event_id": int(event_id),
            "underlying": str(asset),
            "event_time": pd.Timestamp(event_time),
        }
        for fam in ("activity", "iv_level", "liquidity_book_activity", "l2_instability"):
            h = g[g["family"] == fam]
            rec[f"{fam}_n_features"] = int(h["feature"].nunique())
            rec[f"{fam}_level"] = float(h["window"].mean()) if len(h) else np.nan
            rec[f"{fam}_center_95m"] = float(h["center"].mean()) if len(h) else np.nan
        h = g[g["family"] == "l2_instability"]
        rec["l2_rebound_95_to_post"] = float(h["rebound"].mean()) if len(h) else np.nan
        rec["l2_notch_95"] = float(h["notch"].mean()) if len(h) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_components(components: pd.DataFrame) -> pd.DataFrame:
    """Collapse simultaneous assets by timestamp, then summarize components."""
    if components.empty:
        return pd.DataFrame()

    value_cols = [
        c for c in components.columns
        if c.endswith("_level")
        or c.endswith("_center_95m")
        or c in {"l2_rebound_95_to_post", "l2_notch_95"}
    ]
    ts = components.groupby("event_time")[value_cols].mean(numeric_only=True).reset_index()

    rows = []
    for c in value_cols:
        x = pd.to_numeric(ts[c], errors="coerce").dropna().to_numpy(float)
        if not len(x):
            continue
        neutral = 0.0 if c in {"l2_rebound_95_to_post", "l2_notch_95"} else 0.5
        rows.append({
            "component": c,
            "n_unique_event_times": int(len(x)),
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "q25": float(np.quantile(x, 0.25)),
            "q75": float(np.quantile(x, 0.75)),
            "neutral_reference": neutral,
            "mean_minus_neutral": float(np.mean(x) - neutral),
            "share_above_neutral": float(np.mean(x > neutral)),
        })
    return pd.DataFrame(rows)


def bootstrap_component_means(
    components: pd.DataFrame,
    *,
    n_boot: int = 4000,
    seed: int = 9505,
) -> pd.DataFrame:
    """Exploratory bootstrap over unique event timestamps.

    Because the -95 motif was selected after viewing these same data, these
    intervals are descriptive stability intervals, not confirmatory inference.
    """
    if components.empty:
        return pd.DataFrame()
    value_cols = [
        c for c in components.columns
        if c.endswith("_level")
        or c.endswith("_center_95m")
        or c in {"l2_rebound_95_to_post", "l2_notch_95"}
    ]
    ts = components.groupby("event_time")[value_cols].mean(numeric_only=True).reset_index()
    rng = np.random.default_rng(seed)
    rows = []

    for c in value_cols:
        x = pd.to_numeric(ts[c], errors="coerce").dropna().to_numpy(float)
        n = len(x)
        if n < 5:
            continue
        boot = np.empty(int(n_boot), float)
        for b in range(int(n_boot)):
            boot[b] = np.mean(rng.choice(x, size=n, replace=True))
        rows.append({
            "component": c,
            "n_unique_event_times": n,
            "observed_mean": float(np.mean(x)),
            "bootstrap_mean": float(np.mean(boot)),
            "bootstrap_q025": float(np.quantile(boot, 0.025)),
            "bootstrap_q975": float(np.quantile(boot, 0.975)),
            "bootstrap_seed": int(seed),
            "n_boot": int(n_boot),
        })
    return pd.DataFrame(rows)


def leave_one_asset_out(components: pd.DataFrame) -> pd.DataFrame:
    if components.empty:
        return pd.DataFrame()

    assets = sorted(components["underlying"].dropna().astype(str).unique())
    value_cols = [
        c for c in components.columns
        if c.endswith("_level")
        or c.endswith("_center_95m")
        or c in {"l2_rebound_95_to_post", "l2_notch_95"}
    ]
    rows = []
    for excluded in assets:
        h = components[components["underlying"].astype(str) != excluded]
        ts = h.groupby("event_time")[value_cols].mean(numeric_only=True)
        for c in value_cols:
            x = pd.to_numeric(ts[c], errors="coerce").dropna().to_numpy(float)
            if not len(x):
                continue
            neutral = 0.0 if c in {"l2_rebound_95_to_post", "l2_notch_95"} else 0.5
            rows.append({
                "excluded_asset": excluded,
                "component": c,
                "n_unique_event_times": int(len(x)),
                "mean": float(np.mean(x)),
                "mean_minus_neutral": float(np.mean(x) - neutral),
            })
    return pd.DataFrame(rows)


def selftest() -> dict:
    # Test the local motif arithmetic directly.
    rows = []
    for event_id, t in [(1, "2026-01-01 00:00:00+00:00"), (2, "2026-01-02 00:00:00+00:00")]:
        vals = {-21: 0.7, -20: 0.7, -19: 0.4, -18: 0.8, -17: 0.8}
        for b, p in vals.items():
            rows.append({
                "event_id": event_id,
                "underlying": "TEST",
                "event_time": pd.Timestamp(t),
                "feature": "book_obi_20_transition_count",
                "rel_bar": b,
                "rel_min": b * BAR_MINUTES,
                "percentile_vs_controls": p,
                "robust_diff_iqr": p - 0.5,
            })
    scores = pd.DataFrame(rows)
    local = local_motif_event_table(scores)
    assert len(local) == 2
    assert np.allclose(local["center_95m_percentile"], 0.4)
    assert np.allclose(local["center_to_post_rebound"], 0.4)
    assert np.all(local["center_vs_neighbors"] < 0)

    contract = pd.DataFrame([{
        "feature": "book_obi_20_transition_count",
        "family": "l2_instability",
    }])
    comp = motif_component_scores(scores, contract)
    assert np.allclose(comp["l2_rebound_95_to_post"], 0.4)

    return {
        "five_bar_motif_window": True,
        "center_is_minus_95m": True,
        "simultaneous_timestamp_collapse": True,
        "l2_rebound_metric": True,
    }
