# 008 — Claude: 01 closed, 02 v0.2.0 additions, notebook 03 (nested blocks)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-09 |
| **Branch** | Volatility Regression |
| **Responds to** | 007; Ivan's approval of all four 007 decisions (2026-10-09) |
| **Status** | Code pushed. Synthetic tests: 01 11/11, 02 9/9, 03 6/6. 02 v0.2.0 and 03 executed end to end on a synthetic DuckDB plus synthetic raw Parquet only. No real-data results. |

## 1. Ivan's decisions (2026-10-09)

1. 01 is **closed** as a diagnostic. Its coefficients and deletion path are not interpretable (007 §2).
2. The three read-only additions to 02 are approved.
3. Next notebook:
   - option shape, fit and wing features are dropped by design;
   - \(n_{\text{train}} \geq 20p\);
   - collector version is controlled.
4. **Research branches are independent.** HSMM's holdout and frozen spec do not constrain this branch, so there is no development cutoff from 02 v0.2.0 onward.

## 2. 02 v0.2.0 (`FEATURE_PROVENANCE_VERSION = "0.2.0"`)

- **Scope:** all available expiries. The 01 complete-case reproduction keeps 01's own cutoff.
- **`02_unmatched_states.csv`:** unmatched model states by expiry and UTC hour, with each asset's first raw bar. This tests the calendar warm-up explanation (007 §3.4).
- **`02_null_return_summary.csv`, `02_null_return_by_time.csv`:** for each asset, the share of NULL `return_5m` that are isolated, majority-shared or all-asset-shared, and the share with the asset's own bar ineligible (007 §3.3).
- **`02_vintage_boundaries.csv`:** collector vintage per raw row, derived from the data in this order of evidence:
  1. the `collector_version` stamp;
  2. presence of the event-time audit field (`book_out_of_order_update_count`, added by `d2709fa`);
  3. presence of L2 instability fields;
  4. otherwise, the earliest vintage.

  This gives the real deploy time of the event-time fix if the field was written to Parquet. If it was not, that vintage simply does not appear.
- **`02_vintage_boundary_shift.csv`:** standardized mean shift and SD ratio of every model feature within ±24 h of each vintage boundary, by feature family. A definitional break shows as L2 shifting while index and trade features do not.

## 3. Notebook 03 (`vol_ols_5m_blocks.py`, v0.1.0)

| Element | Rule |
|---|---|
| Targets | Paired `parkinson` and `index_abs`, both current and next estimators required (01 v0.1.2 rule) |
| Excluded by design | Option scopes `shape`, `fit`, `option` (wings, spreads, OTM bid share); `atm_iv_d60`; L2 instability (absent before Oct 2); strict `log_rv_4h` |
| Blocks (nested) | B0 returns/vol/clock → B1 + flow/basis/OI → B2 + L2 time-weighted book → B3 + ATM IV level and 5/15/30m changes |
| `log_rv_4h_tol` | Branch-local: ≥ 44 of 48 index returns in (t−4h, t], rescaled. The shared column is untouched. |
| Admission | Feature admitted per asset if coverage ≥ 80% on development rows (never test rows); exclusions exported |
| Budget | Fit only if \(n_{\text{train}} \geq 20p\); otherwise recorded as not fitted |
| Evaluation | Last 25% of sessions = test, scored once. Forward-chained folds on the rest (train 1..k, validate k+1, k ≥ 6). Each block scored on its own complete rows and on rows common to all blocks |
| References | Train mean (\(R^2_{\text{OOS}}\) benchmark) and persistence, on each block's exact scored rows |
| Deletion | Inside B2 only. Remove a feature only if mean fold RMSE falls **and** it helps in a strict majority of folds. Rows fixed to B2's complete rows; test never used |
| Collector vintage | Test residual mean and RMSE by vintage per block, from `02_vintage_boundaries.csv` |
| Bottlenecks | Unique row loss per feature on development rows (007 §4.3) |

A vintage dummy is not used. With a chronological split, the newest vintage has no training rows, so its coefficient cannot be estimated. Residuals by vintage and the 02 boundary shifts are the control.

## 4. What to read first when results arrive

1. `02_vintage_boundary_shift.csv`: does L2 shift at the event-time fix while index and trade features do not?
2. `03_clock_coverage.csv`: B0–B2 should cover all 24 hours. B3 will thin before expiry, by construction.
3. `03_block_scores.csv`, common population, folds: is the B0→B1→B2→B3 increment consistent across folds? Is it the same under both targets?
4. `03_test_residual_by_vintage.csv`: are B2 residuals stable across vintages?

## 5. Questions for ChatGPT

1. Is the ±24 h window adequate for the boundary-shift test, given that clock seasonality is daily?
2. Any objection to fold-majority deletion over the alternative of no deletion at all?

## 6. Decisions needed from Ivan

1. Run 02 (v0.2.0), then 03, and commit the outputs (commands in `notebooks/Volatility_Regression/README.md`). Expect 03 to take on the order of 10–20 minutes.
