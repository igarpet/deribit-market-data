"""Read-only lineage diagnostics for Volatility Regression 02.

v0.2.0 (VR 007, approved by Ivan 2026-10-09): no development cutoff for this
branch; adds unmatched-state attribution, simultaneity of NULL index returns
across assets, data-derived collector vintages, and feature distribution
shifts across vintage boundaries.

All stages are aligned to (underlying, state_time) keys.
Absent upstream columns, null values and readiness-gate exclusions are distinct.
No database mutations, no interpolation, no model fitting or target changes.
"""
from __future__ import annotations
from collections.abc import Mapping
import numpy as np
import pandas as pd

FEATURE_PROVENANCE_VERSION = "0.2.0"
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
    "mid_path_valid_points", "book_out_of_order_update_count",
    "book_event_time_monotonic",
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


# --------------------------------------------------------------------------
# v0.2.0 additions
# --------------------------------------------------------------------------

VINTAGES = ("pre_l2_instability", "l2_instability_pre_eventtime_fix",
            "eventtime_fix_unstamped", "futures_5m_l2_v0.3.0",
            "futures_5m_l2_v0.3.1")


def unmatched_states(final: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Model states with no raw futures row, by underlying, expiry and UTC hour.

    `raw` must already be normalized (state_time = raw bar_end).
    """
    f = normalize_stage(final)
    r = normalize_stage(raw)[list(KEYS)].drop_duplicates()
    j = f[[*KEYS, "expiry_date"]].merge(r, on=list(KEYS), how="left", indicator=True)
    j["matched"] = j["_merge"].eq("both")
    j["utc_hour"] = j["state_time"].dt.hour
    out = (j.groupby(["underlying", "expiry_date", "utc_hour"], dropna=False)
             .agg(states=("matched", "size"), matched=("matched", "sum"))
             .reset_index())
    out["unmatched"] = out["states"] - out["matched"]
    first_raw = r.groupby("underlying")["state_time"].min().rename("first_raw_bar_end")
    return out.merge(first_raw, on="underlying", how="left")


def null_return_simultaneity(final: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Are NULL 5m index returns asset-specific or common across assets?

    Returns (per-timestamp table, per-asset summary). A NULL that coincides
    with NULLs in every other asset points to a collector-wide gap, not to
    asset-specific index staleness.
    """
    f = normalize_stage(final)
    if "return_5m" not in f:
        raise ValueError("return_5m missing")
    f["ret_null"] = pd.to_numeric(f["return_5m"], errors="coerce").isna()
    f["own_bar_ineligible"] = ~bool_mask(f, "index_return_input_eligible") \
        if "index_return_input_eligible" in f else np.nan
    per_t = (f.groupby("state_time")
               .agg(assets=("underlying", "nunique"), null_assets=("ret_null", "sum"))
               .reset_index())
    per_t["all_assets_null"] = per_t["null_assets"].eq(per_t["assets"])
    per_t["share_assets_null"] = per_t["null_assets"] / per_t["assets"]
    f = f.merge(per_t[["state_time", "assets", "null_assets"]], on="state_time")
    rows = []
    for asset, g in f.groupby("underlying"):
        nulls = g[g["ret_null"]]
        n = len(nulls)
        rec = {"underlying": asset, "states": len(g), "null_returns": n,
               "null_share": n / len(g) if len(g) else np.nan,
               "null_isolated_share": float((nulls["null_assets"] == 1).mean()) if n else np.nan,
               "null_majority_share": float((nulls["null_assets"] > nulls["assets"] / 2).mean()) if n else np.nan,
               "null_all_assets_share": float((nulls["null_assets"] == nulls["assets"]).mean()) if n else np.nan}
        if "index_return_input_eligible" in g:
            rec["null_with_own_bar_ineligible_share"] = (
                float(nulls["own_bar_ineligible"].astype(bool).mean()) if n else np.nan)
        rows.append(rec)
    keep = per_t[per_t["null_assets"] > 0].copy()
    keep["expiry_hint_date"] = keep["state_time"].dt.strftime("%Y-%m-%d")
    return keep, pd.DataFrame(rows)


def collector_vintage(raw_aligned: pd.DataFrame) -> pd.Series:
    """Label each aligned raw row with a collector vintage derived from the data.

    Order of evidence: explicit `collector_version` stamp; else presence of the
    event-time audit field (`book_out_of_order_update_count`, added by the
    monotonic L2 event-time fix); else presence of L2 instability statistics;
    else the earliest vintage. Rows without a raw match are `<no_raw>`.
    """
    r = raw_aligned
    out = pd.Series("pre_l2_instability", index=r.index, dtype=object)
    if "book_obi_20_std" in r:
        out[r["book_obi_20_std"].notna()] = "l2_instability_pre_eventtime_fix"
    if "book_out_of_order_update_count" in r:
        out[r["book_out_of_order_update_count"].notna()] = "eventtime_fix_unstamped"
    if "collector_version" in r:
        v = r["collector_version"]
        out[v.notna()] = v[v.notna()].astype(str)
    present = [c for c in r.columns if c not in KEYS]
    if present:
        out[r[present].isna().all(axis=1)] = "<no_raw>"
    return out


def vintage_boundaries(vintage: pd.Series, state_time: pd.Series,
                       underlying: pd.Series) -> pd.DataFrame:
    """First and last state_time per (underlying, vintage)."""
    d = pd.DataFrame({"underlying": underlying.to_numpy(), "vintage": vintage.to_numpy(),
                      "state_time": pd.to_datetime(state_time, utc=True).to_numpy()})
    return (d.groupby(["underlying", "vintage"])["state_time"]
              .agg(first="min", last="max", states="size").reset_index()
              .sort_values(["underlying", "first"]))


def boundary_shift(X: pd.DataFrame, vintage: pd.Series, state_time: pd.Series,
                   window: pd.Timedelta = pd.Timedelta(hours=24),
                   families: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Standardized mean shift of each feature across each vintage boundary.

    Uses only `window` before and after each boundary so slow market changes
    contribute less. Compare L2 features with index/trade features: a shift
    confined to collector-derived features suggests a definitional break.
    shift = (mean_after - mean_before) / sd_before.
    """
    t = pd.to_datetime(state_time, utc=True).reset_index(drop=True)
    v = vintage.reset_index(drop=True)
    X = X.reset_index(drop=True)
    order = [x for x in VINTAGES if (v == x).any()]
    rows = []
    for before, after in zip(order[:-1], order[1:]):
        boundary = t[v == after].min()
        pre = (v == before) & (t > boundary - window) & (t <= boundary)
        post = (v == after) & (t > boundary) & (t <= boundary + window)
        for col in X.columns:
            a = pd.to_numeric(X.loc[pre, col], errors="coerce").dropna()
            b = pd.to_numeric(X.loc[post, col], errors="coerce").dropna()
            sd = a.std()
            rows.append({"boundary": f"{before} -> {after}", "boundary_time": boundary,
                         "feature": col,
                         "family": (families or {}).get(col, "clock_or_extra"),
                         "n_before": len(a), "n_after": len(b),
                         "mean_before": a.mean() if len(a) else np.nan,
                         "mean_after": b.mean() if len(b) else np.nan,
                         "sd_before": sd,
                         "sd_ratio": b.std() / sd if len(b) > 1 and sd > 0 else np.nan,
                         "std_mean_shift": (b.mean() - a.mean()) / sd
                         if len(a) > 1 and len(b) and sd > 0 else np.nan})
    return pd.DataFrame(rows)
