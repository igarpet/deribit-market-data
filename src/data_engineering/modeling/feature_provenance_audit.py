"""Read-only lineage diagnostics for Volatility Regression 02.

All stages are aligned to development-period (underlying, state_time) keys.
Absent upstream columns, null values and readiness-gate exclusions are distinct.
No database mutations, no interpolation, no model fitting or target changes.
"""
from __future__ import annotations
from collections.abc import Mapping
import numpy as np
import pandas as pd

FEATURE_PROVENANCE_VERSION = "0.1.0"
KEYS = ("underlying", "state_time")
L2_COLUMNS = (
    "book_twa_abs_obi_20", "book_obi_20_std",
    "book_obi_20_path_roughness_per_min", "book_obi_20_range",
    "book_obi_20_coherence_noise", "book_obi_20_transition_count",
    "book_obi_20_raw_zero_cross_rate", "book_obi_20_reversal_strength_mean",
    "book_obi_20_reversal_strength_max",
)
EXTRA_RAW_COLUMNS = (
    "book_obi_20_raw_zero_cross_count", "book_stats_complete",
    "book_stats_coverage_ms", "collector_version", "collector_code_sha256",
    "mid_path_realized_variance", "mid_path_increment_count",
    "mid_path_valid_points",
)
GATES = (
    "perp_integrity_ok", "l2_intrawindow_ready",
    "microstructure_features_ready", "instability_features_ready",
    "l2_instability_scalar_available", "book_stats_complete",
    "in_settlement_window", "option_snapshot_ready",
    "option_atm_quote_ok", "option_shape_quote_ok", "fit_ok",
)
OPTION_CHANGES = (
    "atm_iv_change_5m", "atm_iv_change_15m", "atm_iv_change_30m",
    "atm_iv_change_1h", "skew_025_change_1h",
    "curvature_025_change_1h", "skew_025_change_30m",
)
OPTION_LEVELS = ("option_iv_atm_forward", "option_skew_025", "fit_atm_iv")


def bool_mask(frame: pd.DataFrame, col: str) -> np.ndarray:
    if col not in frame:
        return np.zeros(len(frame), dtype=bool)
    s = frame[col].replace({"true": True, "false": False,
                            "True": True, "False": False})
    return s.fillna(False).astype(bool).to_numpy()


def normalize_stage(frame: pd.DataFrame, time_col: str = "state_time") -> pd.DataFrame:
    if "underlying" not in frame or time_col not in frame:
        raise ValueError(f"Stage requires underlying and {time_col}")
    out = frame.copy()
    if time_col != "state_time":
        out = out.rename(columns={time_col: "state_time"})
    out["underlying"] = out["underlying"].astype(str)
    out["state_time"] = pd.to_datetime(out["state_time"], utc=True)
    return out.reset_index(drop=True)


def align_stages(final: pd.DataFrame,
                 stages: Mapping[str, pd.DataFrame]) -> tuple[dict, pd.DataFrame]:
    """Align source layers without expanding a duplicate-key join."""
    anchor = normalize_stage(final)
    if anchor.duplicated(list(KEYS)).any():
        raise ValueError("Duplicate model states; cannot form a 1:1 audit anchor")
    if "expiry_date" not in anchor:
        raise ValueError("Model state requires expiry_date")
    keys = anchor[list(KEYS)]
    result, diagnostics = {"final": anchor}, []
    for name, data in stages.items():
        source = normalize_stage(data)
        duplicates = int(source.duplicated(list(KEYS), keep=False).sum())
        source = source.drop_duplicates(list(KEYS), keep="first")
        joined = keys.merge(source, on=list(KEYS), how="left",
                            validate="one_to_one", indicator="_stage_match", sort=False)
        diagnostics.append({"stage": name, "source_rows": len(data),
                            "duplicate_key_rows": duplicates,
                            "matched_model_states": int((joined["_stage_match"] == "both").sum()),
                            "model_states": len(keys)})
        result[name] = joined.drop(columns="_stage_match")
    return result, pd.DataFrame(diagnostics)


def stage_coverage(stages: dict[str, pd.DataFrame],
                   fields: tuple[str, ...] = L2_COLUMNS) -> pd.DataFrame:
    """Counts distinguish native values from the model's L2 readiness gate."""
    anchor = stages["final"]
    rows = []
    for (asset, expiry), ids in anchor.groupby(
            ["underlying", "expiry_date"], dropna=False, sort=True).groups.items():
        positions = np.asarray(list(ids), dtype=int)
        for stage, frame in stages.items():
            part = frame.iloc[positions]
            gate = (bool_mask(part, "instability_features_ready")
                    if stage == "final" else np.ones(len(part), dtype=bool))
            for col in fields:
                exists = col in frame.columns
                observed = (part[col].notna().to_numpy() if exists
                            else np.zeros(len(part), dtype=bool))
                rows.append({"underlying": asset, "expiry_date": str(expiry),
                             "stage": stage, "field": col, "schema_present": bool(exists),
                             "states": len(part), "native_nonnull": int(observed.sum()),
                             "effective_nonnull": int((observed & gate).sum())})
    return pd.DataFrame(rows)


def l2_attribution(coverage: pd.DataFrame) -> pd.DataFrame:
    """Most upstream observable failure; not a claim that the collector failed."""
    rows = []
    for (asset, expiry, field), group in coverage.groupby(
            ["underlying", "expiry_date", "field"], sort=True):
        by = {r.stage: r for r in group.itertuples(index=False)}
        def v(stage, attr):
            row = by.get(stage)
            return getattr(row, attr) if row is not None else None
        raw, canon, market, final = (v(x, "native_nonnull")
                                     for x in ("raw", "canonical", "market", "final"))
        effective = v("final", "effective_nonnull")
        if v("raw", "schema_present") is False:
            diagnosis = "raw_field_absent_in_union_schema"
        elif raw == 0:
            diagnosis = "raw_field_null_or_unmatched"
        elif canon == 0:
            diagnosis = "raw_to_canonical_missing"
        elif market == 0:
            diagnosis = "canonical_to_market_missing"
        elif final == 0:
            diagnosis = "market_to_final_missing"
        elif effective == 0:
            diagnosis = "readiness_gate_excludes_all"
        elif effective < final:
            diagnosis = "partial_readiness_exclusion"
        else:
            diagnosis = "available"
        rows.append({"underlying": asset, "expiry_date": expiry, "field": field,
                     "raw_nonnull": raw, "canonical_nonnull": canon,
                     "market_nonnull": market, "final_nonnull": final,
                     "model_gate_nonnull": effective, "diagnosis": diagnosis})
    return pd.DataFrame(rows)


def gate_summary(final: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Overlapping quality-gate counts. Do not sum them as disjoint causes."""
    rows = []
    for (asset, expiry), group in final.groupby(
            ["underlying", "expiry_date"], dropna=False, sort=True):
        source = raw.iloc[group.index.to_numpy()]
        rec = {"underlying": asset, "expiry_date": str(expiry),
               "n_model_states": len(group),
               "raw_obi_std_nonnull": (int(source["book_obi_20_std"].notna().sum())
                                      if "book_obi_20_std" in source else 0)}
        for gate in GATES:
            if gate == "book_stats_complete":
                rec["raw_book_stats_complete"] = int(bool_mask(source, gate).sum())
            else:
                rec[gate] = int(bool_mask(group, gate).sum())
        for col in ("atm_iv_change_1h", "option_iv_atm_forward"):
            rec[col + "_native"] = int(group[col].notna().sum()) if col in group else 0
        rows.append(rec)
    return pd.DataFrame(rows)


def option_hour_coverage(final: pd.DataFrame) -> pd.DataFrame:
    f = final.copy()
    f["utc_hour"] = pd.to_datetime(f.state_time, utc=True).dt.hour
    rows = []
    for (asset, hour), group in f.groupby(["underlying", "utc_hour"], sort=True):
        rec = {"underlying": asset, "utc_hour": int(hour), "states": len(group)}
        for gate in ("in_settlement_window", "option_snapshot_ready",
                     "option_atm_quote_ok", "option_shape_quote_ok", "fit_ok"):
            rec[gate + "_true"] = int(bool_mask(group, gate).sum())
        for col in OPTION_CHANGES + OPTION_LEVELS:
            rec[col + "_native"] = (int(group[col].notna().sum())
                                    if col in group else 0)
        rows.append(rec)
    return pd.DataFrame(rows)


def model_coverage(g: pd.DataFrame, end: str = "2026-10-07"
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Availability only; uses the SAME frozen feature choice as notebook 01.

    Exclusion specifications are diagnostic masks, not newly fitted models.
    """
    from data_engineering.modeling.vol_ols_5m import prepare_asset
    from data_engineering.modeling.full_information import FEATURE_SPECS
    d = prepare_asset(g, target="parkinson", coverage=0.50,
                      development_end=end, require_both_targets=True)
    x, chosen = d["X"], d["features"]
    family = {s.name: s.block for s in FEATURE_SPECS}
    variants = {
        "full_01": chosen,
        "without_all_options": [n for n in chosen if family.get(n) != "options"],
        "without_option_changes": [n for n in chosen if not (
            family.get(n) == "options" and "_d" in n)],
        "without_all_l2": [n for n in chosen if family.get(n) != "l2"],
        "without_options_and_l2": [n for n in chosen
                                  if family.get(n) not in ("l2", "options")],
        "no_predictors": [],
    }
    summary, missing = [], []
    for split in ("train", "selection", "test"):
        eligible = np.asarray(d[split], dtype=bool)
        for label, names in variants.items():
            ok = (x[names].notna().all(axis=1).to_numpy()
                  if names else np.ones(len(x), dtype=bool))
            summary.append({"underlying": str(g.underlying.iloc[0]), "split": split,
                            "spec": label, "n_eligible": int(eligible.sum()),
                            "n_complete": int((eligible & ok).sum()),
                            "n_predictors": len(names)})
        for col in x.columns:
            observed = x[col].notna().to_numpy()
            missing.append({"underlying": str(g.underlying.iloc[0]),
                            "split": split, "feature": col,
                            "family": family.get(col, "clock_or_extra"),
                            "in_full_01": col in chosen,
                            "n_eligible": int(eligible.sum()),
                            "n_nonnull": int((eligible & observed).sum())})
    return pd.DataFrame(summary), pd.DataFrame(missing)
