# 004 — ChatGPT: exact 288 five-minute UTC clock groups (Pairwise 02)

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-09 |
| **Branch** | Pairwise Feature Regression |
| **Responds to** | Ivan: group feature observations at the *same clock time across days* for **every** five-minute position, not just 11:00 |
| **Status** | Code and notebook v2.0.0 built; real-data run pending; no collector/model/frozen-spec changes |

## 1. Precisely what Ivan requested

Module 01 pairs each feature with other same-bar features, then pools **all clock hours and minutes**. For Module 02, fit **one separately measured feature relationship for every exact five-minute UTC slot**. There are 288 daily positions, **00:00, 00:05, ..., 11:00, 11:05, ..., 23:55**; every position is covered, not a selected example.

For each asset, feature pair (response i, predictor j), and clock position c:

\[
Y_{i,d,c}=\alpha_{ijc}+\beta_{ijc}X_{j,d,c}+\varepsilon_{ij,d,c},
\qquad d \in \{\text{UTC calendar dates with both values}\}.
\]

The beta and residual RMSE at each \((i,j,c)\) are based only on joint observations from that exact clock position across different UTC dates, **without any mixing with neighboring five-minute times or pooling by hour**. Each regression includes an intercept; forward returns and future targets remain outside scope.

## 2. Implementation

- `src/data_engineering/modeling/pairwise_feature_clock.py` v2.0.0 — exact UTC slot validation and grouping; pairwise-only missingness; slope, in-sample RMSE, normalized metrics and count; constant/missing input fits are NULL. Processes all 288 slots including empty positions.
- `tests/test_pairwise_feature_clock.py` — nine synthetic tests, covering 288 unique slots, UTC conversion, no 11:00↔11:05 leakage, pairwise missingness, low-N mask, duplicate-date rejection, off-grid timestamp rejection, compressed archive roundtrip and threshold checks. Run by the notebook inside the existing Jupyter kernel, avoiding earlier Windows DLL subprocess crash.
- `notebooks/Pairwise_Feature_Regression/02_exact_5m_clock_matrices.ipynb` — full read-only 5m feature-state extraction and exactly 288 slot-group OLS fits per asset. Processes one asset at a time to bound memory, writes all actual matrices and diagnostics.
- `notebooks/Pairwise_Feature_Regression/README.md` — execution details and full matrix retrieval.

The engine preserves Module 01's 82-column feature universe (catalog predictor specs and deterministic time features), transforms and source-scoped quality gates. One module-02 output per asset is `02/02_{ASSET}_clock_matrices.npz`, with **six** 288×82×82 matrices: `beta_raw`, `error_rmse`, `intercept`, `correlation`, `relative_rmse`, `pair_counts`. The feature names and 288 clock labels are also saved. This avoids committing thousands of overlapping huge CSV text files. Any matrix at a specified time is materialized as a DataFrame by `load_clock_matrix(..., "11:05", "beta_raw")`.

Exports also include 288-row clock coverage per asset, per-asset example trajectory curves across all 288 positions, a combined clock summary, the feature inventory and run metadata. The default `DISPLAY_CLOCK_UTC = "00:00"` is **only a preview**, and changing it does not affect the full fitting or archives.

## 3. Scientific cautions: deliberately transparent small n

The currently committed dataset has approximately **19 dates** per UTC clock position at most, many fewer for scoped option and newer L2 data. With exact 5m grouping, larger `MIN_PAIR_ROWS=100` from v1 would yield NO fitted regressions. For descriptive coverage, v2 lowers the minimum to **6 paired UTC dates** and marks **fewer than 12** pairs as low-sample exploratory in every slot report. This is **not** evidence of inferential reliability: even 12–19 days can be insufficient for stable beta estimates. The matrices should not be ranked as robust market laws; potential autocorrelation, collector changes, quote eligibility and daily market-regime mixture remain.

The gating, minimum n, and caution tier are explicit and reproducible. No p-values or forecasting assertions; no per-hour averaging. One asset and slot may contain many NULL regressions because its features were absent or lacked variation on those few dates.

## 4. Questions for Claude

1. Verify the 288-slot implementation, particularly UTC conversion and no inadvertent local-time/daylight-saving grouping.
2. Review whether `n≥6` **exploratory only**, separately flagged below 12, is the right policy or whether v3 should wait for more days rather than treating tiny fits as market relationships.
3. Evaluate compressed full-matrix storage versus giant per-slot CSVs for later reproducibility and GitHub size.
4. What next characterization of relationships across a daily cycle can be done without overstating precision or violating frozen study protocols?

## 5. Decisions needed from Ivan

1. Pull main and execute Pairwise 02 on local DuckDB, then push the full `02/` output directory and the executed notebook for review.
2. Any prospective forecasting test, hypothesis testing or changes to minimum sample standards belong to a separately versioned discussion; none are necessary for the requested v2.
