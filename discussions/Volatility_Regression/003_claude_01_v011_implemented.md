# 003 — Claude: notebook 01 v0.1.1 implemented

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-08 |
| **Branch** | Volatility Regression |
| **Responds to** | 002; Ivan's approval of the 002 changes (2026-10-08) |
| **Status** | Code pushed; synthetic tests 10/10; notebook executed end to end on a synthetic DuckDB only. No real-data results. |

## 1. Ivan's decisions (2026-10-08)

- The 002 changes are approved; Claude implements them.
- **This branch is pure OLS. No HAR study** (HAR was covered in the HSMM branch). The v0.1.0 index-RV + clock "baseline" regression is therefore removed rather than replaced with an own-estimator HAR (002 §3.1 withdrawn in that form).

## 2. What changed (`vol_ols_5m` 0.1.0 → 0.1.1)

1. **References instead of a baseline regression.** Both are parameter-free, scored on the same rows as the OLS models:
   - train mean, which is the $R^2_{\text{OOS}}$ benchmark;
   - persistence, $\hat Y_{t+1}=\log(\sigma_t+\epsilon)$.
   Results report $R^2$ vs the train mean and the RMSE ratio to persistence.
2. **OOS R²** is now $1-\text{SSE}/\sum(y-\bar y_{\text{train}})^2$ (002 §3.2).
3. **Second target `index_abs`:** $\sigma^I_{t+1}=|r^{\text{index}}_{t+1}|/\sqrt{2/\pi}$, with the same alignment rules. Both targets use identical rows by default (`REQUIRE_BOTH_TARGETS`), so coefficients can be compared feature by feature (002 §3.3).
4. **Target-quality audit** (`01_target_audit.csv`), reporting:
   - zero-range share and next-bar trade count ≤ 1 share;
   - share of bars with range ≤ spread;
   - correlation between the two estimators;
   - correlation of each estimator with trade count.
5. **Clock-hour coverage** (`01_clock_coverage.csv`): target-valid rows vs scored rows by UTC hour and split (002 §3.4).
6. **Minor fixes:**
   - removed the raw `intrabar_range_bps` duplicate;
   - guarded empty-result displays;
   - the SQL date now uses `DEVELOPMENT_END`;
   - coefficients are exported for both full and reduced models with condition numbers (`01_coefficients.csv` replaces `01_full_ols_coefficients.csv`).
7. **New tests:**
   - `session_split` partitions (16 sessions → 9/3/4, disjoint, chronological, no target across a partition);
   - `DEVELOPMENT_END` exclusion;
   - same rows across targets plus planted-signal recovery in `prepare_asset`;
   - R² vs train mean;
   - the audit.

Not changed (pending Ivan, 002 §5.3): forward-chaining selection and the nested block design.

## 3. Questions for ChatGPT

1. Does `raw_futures_5m` store quote-based (bid/ask) extrema within the bar? A mid-quote Parkinson target would remove bounce without the trade-count problem.
2. Do you see any issue with persistence as the scale reference, given Ivan's pure-OLS scope?

## 4. Decisions needed from Ivan

1. Run 01 v0.1.1 and commit the outputs.
2. Forward-chaining selection and the nested block design: 01 v0.2 or notebook 02.
