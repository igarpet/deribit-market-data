# 001 — ChatGPT: Pairwise feature regression matrices, v1

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-09 |
| **Branch** | Pairwise Feature Regression |
| **Responds to** | User's request to create a separate notebook research folder with individually regressed features, beta matrix and error matrix |
| **Status** | New code, synthetic unit tests and notebook committed; local real-data execution pending |

## 1. Scientific object and precise matrix interpretation

For each underlying independently, and for every ordered model-feature pair \(i,j\), estimate the intercept and unstandardized slope

\[
Y_{i,t}=\alpha_{ij}+\beta_{ij} X_{j,t}+\varepsilon_{ij,t}
\]

on the joint availability mask \(S_{ij}=\{t: X_{j,t},Y_{i,t}\text{ finite}\}\) from the **same completed five-minute bar**. No pair is allowed to borrow or forward-fill missing observations; no future outcome appears in the matrix.

The two main exports are the **raw slope beta matrix** \(\beta_{ij}\) and **in-sample residual error matrix** \(e_{ij}=\sqrt{(\sum_{t\in S_{ij}}\varepsilon_{ij,t}^2)/|S_{ij}|}\). **Rows respond and columns predict**, in the algebraic sense only; the latter word does not imply future forecasting. The raw matrices are directional: \(\beta_{ij}\neq\beta_{ji}\) in general; \(e_{ij}\neq e_{ji}\) when the response scales differ.

Support columns export \(\alpha_{ij}\), pair size \(n_{ij}\), normalized beta \(r_{ij}\), \(R^2_{ij}=r_{ij}^2\), and response-normalized RMSE \(\sqrt{1-r_{ij}^2}\). For a single regressor with an intercept, *the standardized beta equals Pearson correlation*; a standardized beta-only matrix would therefore simply reproduce the correlation matrix rather than add a new statistic. We intentionally retain **raw directed slope and raw response-error** matrices, and use standardized ones for cross-feature visualization / QA.

## 2. Data and implementation

- Feature universe: `FEATURE_SPECS` from `src/data_engineering/modeling/full_information.py` plus current index return and core deterministic clock/expiry features. All existing transform (`log`, `log1p`, signed log) and quote/perp/microstructure/instability-scoped readiness masks are preserved.
- This feature catalog is broader than VR 03's reduced block selection. Old option shape/fit and L2 instability features are **included when observed**, never forced to appear by imputation. Missing inputs are explicitly shown in the inventory.
- Data source: read-only `time_series_feature_state`; all expiries by default; separate matrices per asset. No pooling across asset-price scales and no merging of frozen branch holdouts.
- Fitting: vectorized pair-specific cross moments. Complete cases only within each pair, not globally; `n \ge 100` and nonconstant response/predictor required; diagonal self-fits are masked as trivial.
- For each asset, CSV matrices, directed long form and beta/raw-RMSE/standardized-beta PNG heatmaps. Asset feature inventory + run metadata track coverage, catalog vintage, date range and library version.
- There is **no holdout optimization** and no hypothesis test. These slopes are **contemporaneous descriptive relationships**, not structural/causal or out-of-sample forecasts. Any proposed forecasting exercise belongs in a different preregistered protocol.

## 3. Code paths

- `src/data_engineering/modeling/pairwise_feature_regression.py` v1.0.0
- `tests/test_pairwise_feature_regression.py` — 9 synthetic cases: exact positive/negative slopes, directional errors, different paired missingness, min-n, constant features, long format and explicit missing catalog fields
- `notebooks/Pairwise_Feature_Regression/01_pairwise_feature_ols_matrices.ipynb`
- `notebooks/Pairwise_Feature_Regression/README.md`

The notebook runs its own targeted unit tests before database access and does not modify the research DB. Expected outputs are under `notebooks/Pairwise_Feature_Regression/01/`.

## 4. Interpretation pitfalls

- Raw slopes depend on the units and transformation of BOTH variables. A large absolute beta is not automatically an important relationship.
- Raw RMSE depends on the scale of the response. A low value is not evidence of prospective predictability.
- Regressions have potentially DIFFERENT n and timestamps. Pair-count coverage must accompany interpretation; comparisons across unrelated pairs are not controlled experiments.
- Trend, serial dependence, overlapping rolling vol estimates, deterministic aliases, session clock and collector-vintage transitions can mechanically inflate associations. Historical L2 features have limited availability.
- Pairwise relationships do not account for confounders or reveal independent feature contribution conditional on all other features.

## 5. Requests for Claude

1. Independently verify the vectorized pair-specific cross-moment algebra against explicit scalar intercept-OLS fits with structured missingness.
2. Check source-catalog completeness relative to the current time-series schema; if 'all features' should mean *every numeric raw state metric* instead of the modeling `FEATURE_SPECS` universe, document the proposed expansion before changing v1.
3. Verify scope-mask enforcement and interpret the impact of sparse historical L2/option columns.
4. Evaluate whether a by-time-to-expiry or by-session standardized visualization would provide distinct value for a **separate** v2, not by silently changing this experiment.

## 6. Decisions needed from Ivan

- Run the synthetic suite and v1 notebook locally, commit outputs and request a results review.
- Any proposed v2 stratification or change in feature universe is separate from this approved v1 build.
