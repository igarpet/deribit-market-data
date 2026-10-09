# Pairwise Feature Regression — independent exploratory branch

**Research question:** What is the contemporaneous one-variable-at-a-time linear relationship between any two model-ready market features? How well does one feature explain another when each ordered pair is estimated in its own overlapping observed sample?

| Notebook | Description | Status |
|---|---|---|
| `01_pairwise_feature_ols_matrices.ipynb` | v1.0: independent simple OLS for every ordered feature pair, per underlying, beta and residual RMSE matrices, heatmaps and sample/scale diagnostics | Built; awaiting real-data execution |

**Reusable engine:** `src/data_engineering/modeling/pairwise_feature_regression.py` (`PAIRWISE_FEATURE_REGRESSION_VERSION = 1.0.0`).

**Tests:** `tests/test_pairwise_feature_regression.py` (nine synthetic checks; run locally).

**Discussion:** `../../discussions/Pairwise_Feature_Regression/`.

## The two primary matrices

Each cell fits `Y_row = alpha + beta * X_column + residual` on the overlap of valid observations **from the same five-minute bar**. All OLS fits include an intercept.

- `01/01_BTC_beta_raw.csv` (etc.): unstandardized beta, with row=response and column=predictor; units vary and the matrix is **not symmetric**.
- `01/01_BTC_error_rmse.csv` (etc.): root mean squared residual *in-sample*, in the response feature's transformed units; not out-of-sample error; also **not symmetric**.

Exports for each asset additionally include `intercept`, `beta_standardized`, `error_relative`, `r_squared`, `pair_counts`, `pairs_long`, and three PNG heatmaps. Branch-level outputs `01_summary.csv`, `01_feature_inventory.csv` and `01_run_metadata.json` document coverage and configuration.

**Why the extra matrices matter:** standardized beta from a one-predictor OLS equals Pearson's r, while relative RMSE is `sqrt(1-r²)` on the same pairwise sample. These normalized summaries are symmetric and expose nearly deterministic relationships; the raw beta/error matrices remain directional and depend on measurement units.

## Scope and interpretation

- All numeric predictor specs in `full_information.FEATURE_SPECS`, preserving their transform and feature-scoped integrity/quote-readiness gates; plus five-minute index return, deterministic clock features and time-to-expiry. Explicit identifiers, future outcomes, quality flags and raw collector telemetry do **not** become regression variables. The per-feature inventory exposes all catalog sources, including those absent or completely masked.
- All currently available expiries by default, with optional asset/date filters. Independent regressions **per asset** (never mix raw units across assets).
- `MIN_PAIR_ROWS = 100` by default. Missing/invalid/constant pairs are NaN in the matrices, and actual observation counts are exported.
- Pairwise complete observations **do not imply a common evaluation population** across different matrix cells. No global complete-case pruning or imputation.
- Exploratory **same-bar** relationships only. Serial dependence, deterministic feature aliases, shared source denominators and session/collector boundaries can dominate fitted values. No predictive, causal, independent-significance or trading claims.
- Does not alter any collection process, feature tables, HSMM models or Volatility Regression's frozen prospective spec.

## Running v1

From repository root, activate your `deribit-data` environment and pull latest main:

```bash
python -m unittest discover -s tests -p "test_pairwise_feature_regression.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/Pairwise_Feature_Regression/01_pairwise_feature_ols_matrices.ipynb
```

The notebook reads `DERIBIT_DATA_ROOT/database/research.duckdb` read-only. Results go to `notebooks/Pairwise_Feature_Regression/01/`. Commit the executed notebook and all `01/01_*` outputs (CSV, JSON, PNG). Heatmap generation can be disabled by setting `MAKE_HEATMAPS = False` before execution.

### Suggested follow-on v2 (not implemented)

Segment by expiry-hour or time-to-expiry, collector vintage and independent session; check common-clock partial effects, robust interval uncertainty or properly prospective lags. Any expansion should be a separately authorized iteration.
