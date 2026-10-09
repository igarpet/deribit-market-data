# Pairwise Feature Regression — independent exploratory branch

**Research question:** What is the contemporaneous one-variable-at-a-time linear relationship between any two model-ready market features? How well does one feature explain another when each ordered pair is estimated in its own overlapping observed sample?

| Notebook | Description | Status |
|---|---|---|
| `01_pairwise_feature_ols_matrices.ipynb` | v1.0: independent simple OLS for every ordered feature pair, per underlying, beta and residual RMSE matrices, heatmaps and sample/scale diagnostics | **Executed**, commit `860b86c`; 9/9 in-kernel tests pass; 46,032 fitted directed regressions; results reviewed in discussion 003 |
| `02_exact_5m_clock_matrices.ipynb` | v2.0: **288 exact five-minute UTC clock groups per asset**; independently regress all feature pairs at 00:00, 00:05, ..., 23:55 across available dates, producing beta/error matrices and sample diagnostics for each slot | Built; awaiting local execution |

**Reusable engines:** `src/data_engineering/modeling/pairwise_feature_regression.py` (v1.0.0), `src/data_engineering/modeling/pairwise_feature_clock.py` (v2.0.0 exact-UTC-clock matrix stacks).

**Tests:** `tests/test_pairwise_feature_regression.py` (nine v1 synthetic checks), `tests/test_pairwise_feature_clock.py` (nine exact-clock-group tests). Execute through the notebooks' in-kernel runners on Windows/Conda.

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

**Windows/Conda note:** the notebook performs its nine synthetic checks in the existing Jupyter kernel rather than spawning a second `python.exe`. This avoids subprocess DLL-search differences (such as Windows exit `0xC06D007F`) while still failing visibly on any actual unittest assertion. The standalone unittest command above remains optional as an independent environment diagnostic; if it crashes with a native Windows status code rather than a Python traceback, verify Conda/DLL resolution separately.

The notebook reads `DERIBIT_DATA_ROOT/database/research.duckdb` read-only. Results go to `notebooks/Pairwise_Feature_Regression/01/`. Commit the executed notebook and all `01/01_*` outputs (CSV, JSON, PNG). Heatmap generation can be disabled by setting `MAKE_HEATMAPS = False` before execution.


## 02 — every five-minute UTC position, separately across dates

**02 does not merely compare 11:00, or pool observations by hour.** For each asset it independently fits every ordered feature-pair regression at each of the **288** UTC clock positions: 00:00, 00:05, ..., 11:00, 11:05, ..., 23:55. An 11:05 beta uses only 11:05 timestamps from different UTC days and the matching 11:05 source features. All established feature transforms and masks remain intact. The *same time* always means the same 5m UTC slot, **not** the same local Madrid clock time during seasonal offset changes.

Primary outputs under `02/` for each asset:

- `02_{ASSET}_clock_matrices.npz`: full **288 × 82 × 82** arrays `beta_raw`, `error_rmse`, `intercept`, `correlation`, `relative_rmse`, `pair_counts`. Array axes are exact UTC clock slot, response feature (row), predictor feature (column). `features` and `clock_utc` arrays record labels. Compressed because thousands of separate large CSV matrices would be inefficient.
- `02_{ASSET}_clock_coverage.csv`: exactly 288 rows describing how many UTC calendar dates and usable directed regressions each position has.
- `02_{ASSET}_example_pair_traces.csv`: selected pair trajectories across all 288 positions, without restricting computation of any other pair.
- `02_all_clock_coverage.csv`, `02_feature_inventory.csv`, `02_run_metadata.json`: cross-asset coverage and reproducibility metadata.
- Preview PNGs for any chosen asset and clock slot; the preview selection does **not** change the 288 fitted slots.

To open a given matrix in Python or a Jupyter cell:

```python
from data_engineering.modeling.pairwise_feature_clock import load_clock_matrix

path = "notebooks/Pairwise_Feature_Regression/02/02_BTC_clock_matrices.npz"
beta_at_1105 = load_clock_matrix(path, "11:05", "beta_raw")
rmse_at_1105 = load_clock_matrix(path, "11:05", "error_rmse")
counts_at_1105 = load_clock_matrix(path, "11:05", "pair_counts")
```

### Short history caution

There are currently only **about 19 UTC calendar dates at each exact five-minute position** in the existing dataset. Fits require at least six jointly observed dates (`MIN_PAIR_ROWS = 6`), and any with **fewer than twelve** are explicitly counted as low-sample exploratory estimates. Even the 12–19-observation fits must not be treated as statistically reliable. Many newer L2-instability features are available for only a few days. An empty cell is not evidence of zero association.

**No prediction, causal claims, confidence tests, hour averaging, model selection or changes to frozen confirmation protocols.**

### Run v2 locally

From repository root under your `deribit-data` Jupyter kernel, after pulling GitHub:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/Pairwise_Feature_Regression/02_exact_5m_clock_matrices.ipynb
```

The notebook runs nine targeted tests within its current Python kernel to avoid the earlier Windows/Conda second-process DLL issue, then reads the research DuckDB in `read_only=True` mode. Commit the executed notebook and the full `02/` output directory for interpretation. Each asset is processed and released separately to limit memory consumption.

### Later research (requires separate approval)

Segment by expiry-hour or time-to-expiry, collector vintage and independent session; check common-clock partial effects, robust interval uncertainty or properly prospective lags. Any expansion should be a separately authorized iteration.
