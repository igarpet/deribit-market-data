"""
Volatility Dynamics Module 04: conditional q99 precursor discrimination.

Purpose
-------
Module 03 found several variables that were unusual before native 5-minute
q99 volatility onsets. Module 04 asks whether those candidate precursor
families remain unusual after controlling for the *pre-existing native
five-minute volatility environment*.

This module explicitly inherits only:
1. the q99 event definition and event/control utilities from Volatility
   Dynamics Module 03; and
2. a frozen candidate set selected from Module 03's own empirical results.

It imports no models, feature rankings, thresholds, or assumptions from any
other research branch.

Volatility matching uses only exact native five-minute squared-return
observations at fixed pre-event lags. No rolling 30m/60m realized-variance
object is used for matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from data_engineering.modeling import volatility_q99_precursors as qp


VERSION = "0.1.0"
BAR_MINUTES = 5

# Native single-bar volatility observations used only for matching.
# 24,12,6,3,2,1 bars = 120,60,30,15,10,5 minutes before the origin.
ENV_LAGS = (-24, -12, -6, -3, -2, -1)
RANK_LAGS = ENV_LAGS
ANATOMY_BARS = 24
DEFAULT_MAX_CONTROLS = 8
DEFAULT_MIN_ENV_DIMS = 4
DEFAULT_MIN_CONTROLS = 4

# Frozen from Module 03 evidence. Duplicated representations of the same
# economic quantity are deliberately avoided where possible.
PRIMARY_CANDIDATES = (
    "intrabar_range_bps",
    "volume_usd",
    "trade_count",
    "option_iv_atm_forward",
    "iv_put_25d",
    "iv_call_25d",
    "book_twa_spread_bps",
    "book_update_count",
    "book_obi_20_raw_zero_cross_count",
    "book_obi_20_transition_count",
    "book_obi_20_raw_zero_cross_rate",
    "book_obi_20_path_roughness_per_min",
)

# These are not candidates for incremental information. They are diagnostics
# that should attenuate if volatility matching is doing its job.
ENVIRONMENT_CHECKS = (
    "realized_vol_30m",
    "realized_vol_1h",
    "realized_vol_4h",
)

# Weak Module 03 channels retained as negative/contrast checks.
CONTRAST_CHECKS = (
    "trade_flow_imbalance",
    "signed_volume_usd",
    "open_interest_return_5m",
    "perpetual_mid_vs_index_bps",
)


@dataclass(frozen=True)
class CandidateDef:
    feature: str
    family: str
    module03_interpretation: str


CANDIDATE_DEFS = (
    CandidateDef("intrabar_range_bps", "native_price_activity", "broadly elevated before q99"),
    CandidateDef("volume_usd", "trade_activity", "broadly elevated before q99"),
    CandidateDef("trade_count", "trade_activity", "broadly elevated before q99"),
    CandidateDef("option_iv_atm_forward", "option_iv_level", "ATM IV level elevated before q99"),
    CandidateDef("iv_put_25d", "option_iv_level", "put-wing IV level elevated before q99"),
    CandidateDef("iv_call_25d", "option_iv_level", "call-wing IV level elevated before q99"),
    CandidateDef("book_twa_spread_bps", "liquidity", "spread widened into q99"),
    CandidateDef("book_update_count", "book_activity", "book update activity increased into q99"),
    CandidateDef("book_obi_20_raw_zero_cross_count", "l2_instability", "strong but sparse Module 03 candidate"),
    CandidateDef("book_obi_20_transition_count", "l2_instability", "strong but sparse Module 03 candidate"),
    CandidateDef("book_obi_20_raw_zero_cross_rate", "l2_instability", "strong but sparse Module 03 candidate"),
    CandidateDef("book_obi_20_path_roughness_per_min", "l2_instability", "strong but sparse Module 03 candidate"),
)


def candidate_contract(available_columns: Iterable[str]) -> pd.DataFrame:
    available = set(map(str, available_columns))
    rows = []
    for d in CANDIDATE_DEFS:
        rows.append({
            "feature": d.feature,
            "family": d.family,
            "module03_interpretation": d.module03_interpretation,
            "present": d.feature in available,
            "role": "primary_candidate",
        })
    for f in ENVIRONMENT_CHECKS:
        rows.append({
            "feature": f,
            "family": "volatility_environment_check",
            "module03_interpretation": "diagnostic only; should attenuate after native-vol matching",
            "present": f in available,
            "role": "environment_check",
        })
    for f in CONTRAST_CHECKS:
        rows.append({
            "feature": f,
            "family": "contrast_check",
            "module03_interpretation": "weak/inconsistent in Module 03",
            "present": f in available,
            "role": "contrast_check",
        })
    return pd.DataFrame(rows)


def _ns_values(times: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    t = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    return t.as_unit("ns").asi8


def build_native_vol_environment(
    frame: pd.DataFrame,
    lags: Iterable[int] = ENV_LAGS,
) -> pd.DataFrame:
    """Exact-lag native log-variance history for every origin.

    Each matching dimension is one single five-minute return squared at an
    exact timestamp before the origin. No rolling aggregation is created.
    """
    lags = tuple(sorted(set(int(x) for x in lags)))
    if any(x >= 0 for x in lags):
        raise ValueError("Environment lags must be strictly negative.")

    required = {"state_time", "underlying", "return_5m"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = frame[["state_time", "underlying", "return_5m"]].copy()
    out["state_time"] = pd.to_datetime(out["state_time"], utc=True)
    out["variance_5m"] = pd.to_numeric(out["return_5m"], errors="coerce") ** 2

    for lag in lags:
        out[f"env_logvar_{abs(lag) * BAR_MINUTES}m"] = np.nan

    step_ns = pd.Timedelta(minutes=BAR_MINUTES).value

    for asset, idx in out.groupby("underlying").groups.items():
        ii = np.asarray(list(idx), dtype=int)
        times = pd.DatetimeIndex(out.loc[ii, "state_time"])
        ns = _ns_values(times)
        lookup = {int(t): int(i) for t, i in zip(ns, ii)}
        variance = out.loc[ii, "variance_5m"].to_numpy(float)
        pos = variance[np.isfinite(variance) & (variance > 0)]
        eps = float(np.median(pos) * 1e-4) if len(pos) else np.nan

        for global_i, t in zip(ii, ns):
            for lag in lags:
                j = lookup.get(int(t + lag * step_ns), -1)
                if j < 0:
                    continue
                v = float(out.at[j, "variance_5m"])
                if np.isfinite(v) and np.isfinite(eps):
                    out.at[global_i, f"env_logvar_{abs(lag) * BAR_MINUTES}m"] = np.log(v + eps)

    env_cols = [f"env_logvar_{abs(lag) * BAR_MINUTES}m" for lag in lags]
    out["env_dims_available"] = out[env_cols].notna().sum(axis=1)
    return out


def _robust_asset_scales(env: pd.DataFrame, env_cols: list[str]) -> dict[str, np.ndarray]:
    scales: dict[str, np.ndarray] = {}
    for asset, g in env.groupby("underlying"):
        vals = g[env_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        q25 = np.nanquantile(vals, 0.25, axis=0)
        q75 = np.nanquantile(vals, 0.75, axis=0)
        scale = q75 - q25
        std = np.nanstd(vals, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, std)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        scales[str(asset)] = scale
    return scales


def choose_volatility_matched_controls(
    frame: pd.DataFrame,
    origins: pd.DataFrame,
    events: pd.DataFrame,
    *,
    max_controls: int = DEFAULT_MAX_CONTROLS,
    min_env_dims: int = DEFAULT_MIN_ENV_DIMS,
    lags: Iterable[int] = ENV_LAGS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same-asset/exact-clock controls nearest in native pre-vol history."""
    if max_controls < 1:
        raise ValueError("max_controls must be positive.")

    lags = tuple(sorted(set(int(x) for x in lags)))
    env = build_native_vol_environment(frame, lags=lags)
    env_cols = [f"env_logvar_{abs(lag) * BAR_MINUTES}m" for lag in lags]
    scales = _robust_asset_scales(env, env_cols)

    recs = []
    support = []

    for ev in events.itertuples():
        eid = int(ev.event_id)
        origin_i = int(ev.event_panel_index)
        asset = str(ev.underlying)
        clock_slot = int(ev.clock_slot)

        if origin_i not in env.index:
            continue
        evec = env.loc[origin_i, env_cols].to_numpy(float)
        escale = scales[asset]

        cand_idx = origins.index[
            (origins["underlying"].astype(str) == asset)
            & (origins["clock_slot"] == clock_slot)
            & (origins["eligible_control_origin"])
        ].to_numpy(dtype=int)

        cand_rows = []
        for ci in cand_idx:
            cvec = env.loc[ci, env_cols].to_numpy(float)
            common = np.isfinite(evec) & np.isfinite(cvec)
            n_common = int(common.sum())
            if n_common < min_env_dims:
                continue
            z = (evec[common] - cvec[common]) / escale[common]
            distance = float(np.sqrt(np.mean(z ** 2)))
            date_distance = abs(
                (pd.Timestamp(env.at[ci, "state_time"]) - pd.Timestamp(ev.event_time)).total_seconds()
            ) / 86400.0
            cand_rows.append((distance, date_distance, int(ci), n_common))

        cand_rows.sort(key=lambda x: (x[0], x[1], x[2]))
        chosen = cand_rows[:max_controls]

        for rank, (distance, date_distance, ci, n_common) in enumerate(chosen, start=1):
            recs.append({
                "event_id": eid,
                "underlying": asset,
                "event_time": pd.Timestamp(ev.event_time),
                "clock_slot": clock_slot,
                "control_rank": rank,
                "control_panel_index": ci,
                "control_time": pd.Timestamp(env.at[ci, "state_time"]),
                "env_distance": distance,
                "distance_days": date_distance,
                "n_env_dims": n_common,
            })

        support.append({
            "event_id": eid,
            "underlying": asset,
            "event_time": pd.Timestamp(ev.event_time),
            "n_eligible_same_clock": int(len(cand_idx)),
            "n_vol_comparable": int(len(cand_rows)),
            "n_controls_selected": int(len(chosen)),
            "median_selected_env_distance": (
                float(np.median([x[0] for x in chosen])) if chosen else np.nan
            ),
            "max_selected_env_distance": (
                float(np.max([x[0] for x in chosen])) if chosen else np.nan
            ),
        })

    return pd.DataFrame(recs), pd.DataFrame(support)


def environment_balance(
    env: pd.DataFrame,
    events: pd.DataFrame,
    controls: pd.DataFrame,
    lags: Iterable[int] = ENV_LAGS,
) -> pd.DataFrame:
    """Describe event-control balance on each native-vol matching dimension."""
    lags = tuple(sorted(set(int(x) for x in lags)))
    rows = []
    control_groups = {
        int(eid): g["control_panel_index"].astype(int).to_numpy()
        for eid, g in controls.groupby("event_id")
    }

    for lag in lags:
        col = f"env_logvar_{abs(lag) * BAR_MINUTES}m"
        event_vals = []
        control_medians = []
        diffs = []
        for ev in events.itertuples():
            eid = int(ev.event_id)
            ei = int(ev.event_panel_index)
            cis = control_groups.get(eid, np.empty(0, dtype=int))
            if ei not in env.index or len(cis) == 0:
                continue
            evv = float(env.at[ei, col])
            cv = pd.to_numeric(env.loc[cis, col], errors="coerce").to_numpy(float)
            cv = cv[np.isfinite(cv)]
            if not np.isfinite(evv) or len(cv) == 0:
                continue
            cm = float(np.median(cv))
            event_vals.append(evv)
            control_medians.append(cm)
            diffs.append(evv - cm)

        rows.append({
            "lag_bar": lag,
            "lag_min": lag * BAR_MINUTES,
            "n_events": int(len(diffs)),
            "event_median_logvar": float(np.median(event_vals)) if event_vals else np.nan,
            "control_median_logvar": float(np.median(control_medians)) if control_medians else np.nan,
            "median_event_minus_control_logvar": float(np.median(diffs)) if diffs else np.nan,
            "mean_abs_event_minus_control_logvar": float(np.mean(np.abs(diffs))) if diffs else np.nan,
        })
    return pd.DataFrame(rows)


def compare_rankings(
    clock_rankings: pd.DataFrame,
    conditional_rankings: pd.DataFrame,
    contract: pd.DataFrame,
) -> pd.DataFrame:
    """Side-by-side attenuation/survival of frozen Module 03 candidates."""
    base = contract[contract["role"] == "primary_candidate"][
        ["feature", "family", "module03_interpretation", "present"]
    ].copy()

    def slim(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame({"feature": []})
        keep = [
            "feature", "best_rel_min", "n_unique_event_times", "mean_percentile",
            "abs_deviation_from_half", "asset_direction_consistency",
            "lag_direction_consistency", "discovery_score",
        ]
        x = df[[c for c in keep if c in df.columns]].copy()
        return x.rename(columns={c: f"{prefix}_{c}" for c in x.columns if c != "feature"})

    out = base.merge(slim(clock_rankings, "clock"), on="feature", how="left")
    out = out.merge(slim(conditional_rankings, "conditional"), on="feature", how="left")

    c = pd.to_numeric(out.get("clock_abs_deviation_from_half"), errors="coerce")
    v = pd.to_numeric(out.get("conditional_abs_deviation_from_half"), errors="coerce")
    out["deviation_retention"] = np.where(c > 0, v / c, np.nan)

    cs = pd.to_numeric(out.get("clock_discovery_score"), errors="coerce")
    vs = pd.to_numeric(out.get("conditional_discovery_score"), errors="coerce")
    out["score_retention"] = np.where(cs > 0, vs / cs, np.nan)
    return out.sort_values(
        ["conditional_discovery_score", "conditional_n_unique_event_times"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def family_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()
    rows = []
    for fam, g in comparison.groupby("family"):
        vals = pd.to_numeric(g["conditional_abs_deviation_from_half"], errors="coerce")
        ret = pd.to_numeric(g["deviation_retention"], errors="coerce")
        score = pd.to_numeric(g["conditional_discovery_score"], errors="coerce")
        rows.append({
            "family": fam,
            "n_features": int(len(g)),
            "n_with_conditional_result": int(vals.notna().sum()),
            "median_conditional_abs_deviation": float(vals.median()) if vals.notna().any() else np.nan,
            "max_conditional_abs_deviation": float(vals.max()) if vals.notna().any() else np.nan,
            "median_deviation_retention": float(ret.median()) if ret.notna().any() else np.nan,
            "max_conditional_discovery_score": float(score.max()) if score.notna().any() else np.nan,
        })
    return pd.DataFrame(rows).sort_values(
        "max_conditional_discovery_score", ascending=False, na_position="last"
    ).reset_index(drop=True)


def selftest() -> dict:
    """Synthetic matching test with one incremental candidate."""
    rng = np.random.default_rng(12)
    n_days = 80
    slots = 30
    rows = []
    for d in range(n_days):
        date = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=d)
        latent = rng.normal()
        for s in range(slots):
            r = rng.normal(scale=0.001 * np.exp(0.2 * latent))
            rows.append({
                "state_time": date + pd.Timedelta(minutes=5 * s),
                "underlying": "TEST",
                "return_5m": r,
                "candidate": rng.normal(),
            })
    f = pd.DataFrame(rows)

    # Force repeated q99 events at the same clock, while keeping the preceding
    # native-vol history comparable to ordinary days and lifting the candidate.
    event_days = list(range(20, 80, 10))
    for d in event_days:
        i = d * slots + 20
        f.loc[i, "return_5m"] = 0.20
        f.loc[i - 1, "candidate"] = 6.0

    events, _ = qp.identify_q99_onsets(f)
    origins = qp.build_origin_index(f, events)
    controls, support = choose_volatility_matched_controls(
        f, origins, events, max_controls=8, min_env_dims=4
    )
    scores = qp.percentile_scores(
        f, ["candidate"], events, controls, relative_bars=(-1, 0), min_controls=4
    )
    x = scores[(scores["feature"] == "candidate") & (scores["rel_bar"] == -1)]
    assert len(x) > 0
    assert float(x["percentile_vs_controls"].mean()) > 0.75
    assert support["n_controls_selected"].max() <= 8

    return {
        "native_5m_environment_only": True,
        "same_asset_exact_clock": True,
        "volatility_distance_matching": True,
        "known_incremental_candidate_recovered": True,
    }
