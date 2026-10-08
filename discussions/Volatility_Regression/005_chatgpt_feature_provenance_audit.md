# 005 — ChatGPT: Volatility Regression 02 feature-provenance audit implementation

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-09 |
| **Branch** | Volatility Regression |
| **Responds to** | VR 004 and the executed VR 01 v0.1.2 results (commit 8e52de1) |
| **Status** | Read-only audit implementation pushed; synthetic audit tests added; raw-local-DuckDB audit execution pending |

## 1. Question and evidence already available

The immediate scientific question is whether Volatility Regression 01's large loss of observations and nine absent L2 instability features arise from **collectors**, **historical schema vintages**, **data engineering**, **quality/readiness policy**, or **complete-case modeling**.

From the committed VR 01 outputs:

- Training complete-case retention: BTC 529/2461 (21.5%), ETH 506/2461 (20.6%), SOL 441/2459 (17.9%), AVAX 765/2298 (33.3%), HYPE 371/1520 (24.4%), XRP 521/2135 (24.4%).
- Nine L2 instability predictors have zero train coverage across all six fitted assets. This is **not**, on its own, proof of collector malfunction.
- No fitted asset has OLS complete-case observations in the 08:00–08:59 UTC hour. Option features with 60-minute within-session lags are an expected part of the loss.
- Multiple target-proxy artifacts make trade-based Parkinson volatility hard to interpret for less-liquid assets, independent of collector correctness.

From code inspection:

- `src/futures_state_cache.py` computes L2 instability statistics and the 10-second-grid summary `mid_path_realized_variance` as **five-minute aggregate fields**. Their historical availability remains to be verified.
- `src/data_engineering/features/market_time_series.py` uses `optional_column_expression()` to synthesize `CAST(NULL AS type)` for optional upstream L2 fields absent in the raw/canonical schema.
- `l2_instability_features_ready` requires intrawindow readiness, `perp_integrity_ok` and L2 scalar availability. Final `instability_features_ready` passes that flag downstream.
- `src/data_engineering/modeling/full_information.py` masks instability predictors using `instability_features_ready`; these are quality-scoped features, not plain numeric columns.
- `src/data_engineering/features/option_time_series.py` requires 12 previous five-minute observations, same session and compatible sources for 60-minute IV differences. The model also excludes some option scopes during settlement.

**No collector fault has yet been established.** All claims about which specific source stage caused missing L2 values await local execution.

## 2. Deliverables

- `notebooks/Volatility_Regression/02_feature_provenance_audit.ipynb`: read-only DuckDB/Parquet forensic notebook.
- `src/data_engineering/modeling/feature_provenance_audit.py`: versioned `FEATURE_PROVENANCE_VERSION=0.1.0` utilities.
- `tests/test_feature_provenance_audit.py`: synthetic tests for duplicate-key protection, schema/NULL distinction, upstream loss and readiness gating.
- Expected outputs: `02_raw_file_schema.csv`, `02_stage_schema.csv`, `02_alignment.csv`, `02_stage_coverage.csv`, `02_l2_attribution.csv`, `02_quality_gates.csv`, `02_collector_versions.csv`, `02_option_hour_coverage.csv`, `02_complete_case_waterfall.csv`, `02_feature_missingness.csv`, `02_errors.csv`, `02_run_metadata.json`.

The file-schema audit checks each daily futures Parquet file, while its row-value audit joins source states on `underlying` and `bar_end`. This prevents a newer union-by-name schema from making historical older file schemas look populated.

The stage comparison uses identical development-period model keys, and reports duplicates separately. Quality flags overlap and must never be added as disjoint causes. The OLS availability experiment uses Vol Reg 01's candidate feature-selection rules and compares masks after excluding specific feature families **without fitting or selecting a replacement model**.

The audit reads only data with expiry <= 2026-10-07. No prospective HSMM 06 confirmation outcomes are inspected, no raw records/quality gates are changed, and no tick data are requested.

## 3. Review requests for Claude

1. Check the exact lineage under realistic mixed collector-version Parquet schemas. Are null values in old raw files correctly separated from the quality masking that occurs in `raw_feature_frame`?
2. Check DuckDB read-only compatibility on the local Windows `deribit-data` environment, especially the canonical-market view, Parquet union_by_name, and timestamp-key alignment.
3. Inspect the raw daily-file schema and collector-version summaries **only through the preserved development cutoff**. Are absent L2 instability features absent raw, lost by engineering, or gated? Please quantify by asset and expiry, not guess.
4. Check whether the midprice-path aggregates have historical non-null coverage, including increment/valid-point counts; do not add a new target to 01 without Ivan's approval.
5. Examine `02_complete_case_waterfall.csv` and the options hour audit to identify the specific predictors responsible for losing most eligible 5-minute states.

## 4. Decisions and follow-up

This is an authorization to **audit**, not to rewrite collectors or relax quality policy. Proposed collector, schema, quality or modeling changes must be shown to Ivan with measured impact and require his approval.

**Local execution:** activate `deribit-data`, pull main, run

```bat
python -m unittest discover -s tests -p "test_feature_provenance_audit.py" -v
jupyter nbconvert --to notebook --execute --inplace notebooks/Volatility_Regression/02_feature_provenance_audit.ipynb
```

Commit the executed notebook and `02_*` results. On that evidence, we can give an attributable, ranked list of source failures versus expected missingness. Current outputs in 01 establish the downstream symptoms, not yet the raw-stage causes.
