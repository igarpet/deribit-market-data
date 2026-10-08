# 006 — ChatGPT: Volatility Regression 02 provenance audit results

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-09 |
| **Branch** | Volatility Regression |
| **Responds to** | 005 feature-provenance audit implementation; local execution commit `73767f52c16c25af33eb5dbf30c7de6864fb4e2b` |
| **Status** | Real-data audit complete and reviewed; recommendations require Ivan's approval before changes |

## 1. Evidence and run status

Audit `02_feature_provenance_audit.ipynb` executed all seven cells (execution counts 1–7); built-in subprocess unit-test invocation completed without error; `02_errors.csv` is empty. Metadata: 36,288 model-state rows, 32,025 raw-matched futures rows, 17 examined daily futures Parquet files; development expiry cutoff 2026-10-07; zero writes to DuckDB. Five synthetic provenance tests exist. This audit does not refit any volatility regression or evaluate future held-out outcomes.

## 2. L2 instability features: historical vintage, not observed present-day collector failure

`02_raw_file_schema.csv`: all nine L2 instability fields were absent from daily file **schemas** Sep 21–30; appeared Oct 1–7. The first available **non-null aligned values** appear in the Oct 2 expiry session. For each asset `book_obi_20_std` had 1,694 native raw observations across Oct 2–7. `02_quality_gates.csv` has 1,654–1,658 effective ready observations per asset (about 97.6–97.9% of raw-observed values).

Example BTC by expiry: Oct 2: raw/canonical/market/final 254, gated 247; Oct 3: 288/282; Oct 4: 288/281; Oct 5: 288/286; Oct 6: 288/277; Oct 7: 288/282. No raw→canonical→market→final value loss appears for the observed L2 scalar fields in these sessions, only a small quality-readiness exclusion.

The Vol Reg 01 training partition precedes this instrumented period, explaining the zero *training* coverage for nine L2 features. **Do not backfill unobserved earlier instability values or relabel quality-gated values as clean.** The `raw_field_null_or_unmatched` diagnosis aggregates multiple upstream causes; the separate per-file schema report resolves vintage absence.

Collector version metadata is absent in earlier files: `<unknown>` in the report means **no version tag**, not necessarily no L2 scalar measurement. Versioned files `futures_5m_l2_v0.3.0` and `v0.3.1` appear later.

## 3. Main cause of Vol Reg 01 regression row loss: feature-policy intersection

From `02_complete_case_waterfall.csv`, train eligible/common-row counts:

| Asset | Eligible | All features | Without options | Without L2 |
|---|---:|---:|---:|---:|
| BTC | 2461 | 529 | 1374 | 529 |
| ETH | 2461 | 506 | 1374 | 506 |
| SOL | 2459 | 441 | 1370 | 441 |
| AVAX | 2298 | 765 | 1264 | 765 |
| HYPE | 1520 | 371 | 746 | 371 |
| XRP | 2135 | 521 | 1160 | 521 |
| TRX | 523 | 175 | 249 | 175 |

Excluding all options from an **availability-only diagnostic** substantially increases complete rows. Excluding L2 alone does not change complete-row counts. This is not a proof that options lack predictive value or that L2 improves the forecast. These are nested masks on the fixed Vol Reg 01 candidate feature roster; no new regression has been fitted.

`02_feature_missingness.csv` identifies thin per-feature coverage in option skew/curvature changes, fitted surfaces and quote-scoped levels. `log_rv_4h` is also relatively sparse (~63–64% individually for large assets), explaining some of the remaining loss without options.

`02_option_hour_coverage.csv`: ATM IV change 1h is absent for all asset 08:00 UTC hours by construction (lag needs 12 same-session 5m states), while 07:30–08:00 settlement masks and quote-quality flags restrict option-scope predictors. Do not relax gates without a separate tested scientific rationale.

## 4. Existing raw mid-path realized-variance aggregate

`02_collector_versions.csv` records 5,348 non-null raw `mid_path_realized_variance` rows across version-stamped v0.3.0/v0.3.1 data. Both `mid_path_increment_count` and `mid_path_valid_points` are present in the raw files. However, `02_stage_schema.csv` shows `mid_path_realized_variance` (and provenance/version stamps) **are absent from the engineered market feature schema**. This suggests an engineering pass-through opportunity **without collecting tick data**. Assess missing/zero increments, coverage and stale books before adopting mid-path variance as a new target.

## 5. Residual raw-to-model state alignment caveat

`02_alignment.csv`: 32,025 of 36,288 model states matched one of the 17 raw Parquet files examined, with **4,263 unmatched model states**. Canonical and engineered market stages each have 36,288 aligned state rows. The examined futures file dates start September 21; the state calendar includes earlier rows. This is an important historical coverage boundary, not proof of an ongoing collector fault. Before final interpretation, audit unmatched states by expiry and UTC hour and correlate them with raw-file presence / price readiness.

No duplicate raw, canonical or market keys were reported.

## 6. Conclusions and outstanding decisions

1. **No current L2 collector failure established**; pre-Oct 1 schema vintage explains most historic L2 absence, later readiness gates remove relatively few observed values.
2. **Vol Reg 01 loses most model rows through complete-case intersections**, especially option scopes. Retained samples must be evaluated for clock and market selection bias.
3. **Do not alter collectors, option gates or frozen HSMM confirmation.** Maintain 5-minute collection; avoid tick storage.
4. Potential next research step: a nested feature-availability/forecasting design on chronological development sessions, with causally aligned and strictly paired rows. New predictive models must be separately approved; the 02 mask comparison does not imply they perform better.
5. Engineering candidate: expose existing raw mid-path aggregate with coverage diagnostics, but not as an automatic replacement for the Parkinson target.

### Requests for Claude

- Validate the attribution of the 4,263 missing raw matches by expiry (calendar warmup / missing early Parquet partitions / other problems).
- Check whether the 5,348 mid-path RV rows meet adequate increment count and valid-point coverage; review the 10-second-grid aggregation semantics.
- Critique the complete-case counterfactuals: what further breakdown identifies individual *marginal* bottlenecks when missingness is correlated?
- Review risks of training a new OLS on later L2-instrumented sessions without contaminating HSMM 06's reserved confirmation period.

**Decision for Ivan:** approve a focused follow-up audit/report for raw temporal alignment and mid-path quality before authorizing any data-pipeline adjustments or a new forecasting specification.