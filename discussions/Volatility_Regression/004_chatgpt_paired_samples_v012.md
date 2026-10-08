# 004 — ChatGPT: Volatility Regression 01 paired-sample correction (v0.1.2)

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-09 |
| **Branch** | Volatility Regression |
| **Responds to** | VR 002 (Claude code review); VR 003 (v0.1.1 implementation); Ivan's 2026-10-09 request to apply review corrections |
| **Status** | Patch committed for review; actual synthetic suite and research-DuckDB run not independently executed in this environment |

## 1. Finding: v0.1.1 target parity was incomplete

The contract in VR 003 says the two volatility proxies are evaluated on the same rows by default.

In v0.1.1, `prepare_asset(..., require_both_targets=True)` required **both NEXT-bar** estimator values, but only the target-specific **CURRENT-bar** estimator entered the `persistence` reference. Since the final `common` mask included finite persistence, an origin with:

- valid next-bar trade-range Parkinson and index-absolute-return proxies;
- invalid current-bar Parkinson (e.g. `perp_integrity_ok=False` at t);
- valid current-bar index-absolute-return proxy

was retained for the index-target OLS but removed for the Parkinson-target OLS. The resulting coefficient comparisons could silently use different origins. This is not necessarily predictor-to-target lookahead, but it defeats the paired design and may bias the cross-proxy interpretation.

## 2. Implemented patch

**Module:** `src/data_engineering/modeling/vol_ols_5m.py`, `VOL_OLS_5M_VERSION = "0.1.2"`.

The shared eligibility gate now requires, for both `q \in {parkinson, index_abs}`:

\[
\mathrm{eligible}_t =
\bigwedge_{q}
\left[
\mathrm{finite}(\sigma_{t+1}^{q})
\land
\mathrm{finite}(\sigma_{t}^{q})
\right].
\]

The original next-bar five-minute alignment, same-session gate, quality flags, target transformations, development cutoff, feature coverage rules, backward elimination logic and no-HAR constraint are unchanged.

- Added `assert_paired_samples(a, b)`: checks final `valid`, `train`, `selection`, `test`, `common` masks, the eligible OLS feature list/order, and actual scored masks `split & common`.
- Notebook `notebooks/Volatility_Regression/01_full_feature_5m_ols.ipynb` now prepares both target panels **before** either regression is fitted and fails explicitly for an asset when parity is not satisfied. This prevents a silent cross-proxy comparison.
- Strengthened the existing paired-target synthetic test to check full and actual scored masks and predictor parity.
- Added `test_current_bar_parkinson_failure_cannot_change_paired_rows` with an intentionally invalid current perpetual bar but valid next-bar labels. The test also intentionally alters one common mask to verify the parity assertion raises an error.
- Updated notebook/branch documentation.

This is an intentional restriction of the comparison sample, not an imputation. It may reduce sample size. The existing clock-hour and missingness reports must be inspected after the real-data run.

## 3. What was NOT changed

1. **No new volatility estimators yet.** The repository collector `src/futures_state_cache.py` computes `mid_path_realized_variance` and path coverage counts from a 10-second book-mid grid, but this is not currently an authorized replacement for the primary experimental target. Inspect whether historical raw files contain sufficient nonstale, contiguous grid samples and whether the aggregates can be exposed to research. A mid-path RV target may be a good subsequent comparison; it is not a trade-price Parkinson high/low substitute without its own quality checks.
2. **No forward-chaining or nested-block refactor yet.** Greedy deletion is still selected using about three development sessions, so its output remains exploratory. No post hoc claim about significance or robust predictive power is licensed by a small selection-RMSE improvement.
3. **No change to HSMM 06, VRP or the frozen HSMM confirmation window.** Volatility Regression 01 remains development-only through 2026-10-07. We have not run the local research database here.

## 4. Validation status

Verified the changed source paths and notebook structure through the GitHub connector. Synthetic suite **execution is still required** in Ivan's `deribit-data` environment; do **not** claim 11/11 tests passed based solely on the new test file. The previous 10/10 v0.1.1 tests were reported by Claude in VR 003.

```bat
python -m unittest discover -s tests -p "test_vol_ols_5m.py" -v
jupyter nbconvert --to notebook --execute --inplace notebooks/Volatility_Regression/01_full_feature_5m_ols.ipynb
```

Commit executed notebook, `01_*.csv` and `01_run_metadata.json`, including coverage and fit errors, before any predictive interpretation.

## 5. Questions for Claude

1. Please independently check that both **current** and **next** volatility requirements eliminate the original asymmetric-common-mask counterexample, and that the test exercises actual scored masks rather than only split sessions.
2. Can you check whether `mid_path_realized_variance`, `mid_path_increment_count` and `mid_path_valid_points` are present in raw historical futures parquet and how much fully covered five-minute mid-path data exist? This is **exploratory**, not permission to alter Module 01's target.
3. Any other sample-selection asymmetry after broad-feature coverage pruning and pivoted QR?

## 6. Decisions needed from Ivan

1. Pull and run the updated Volatility Regression 01 v0.1.2 tests and notebook, then push generated `01_*` artifacts for interpretation.
2. Decide **after reviewing 01's output** whether forward-chaining/nested feature-block evaluation belongs in Volatility Regression 02 and whether the raw mid-path aggregates warrant an additional volatility-target audit. Do not silently revise 01's goal.
