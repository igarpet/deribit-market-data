# 001 — ChatGPT: native-frequency full-feature volatility OLS

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-08 |
| **Branch** | Volatility Regression |
| **Responds to** | Ivan's proposal for native 5m volatility OLS and backward feature elimination; HSMM 06 screened HAR/M4/M4_OLS |
| **Status** | Implementation delivered for review; synthetic test script committed, local research run not yet executed |

## 1. Research hypothesis

Predict the **volatility** of the next five-minute interval, not the mean or squared value of next-bar returns. For a trade-high/low five-minute bar, define the exploratory Parkinson volatility proxy:

\[
\sigma_{t+1,5m}^{P}=\frac{\log(H_{t+1}/L_{t+1})}{\sqrt{4\log 2}},
\qquad
Y_{t+1}=\log(\sigma_{t+1,5m}^{P}+\epsilon_{\mathrm{train}}).
\]

The explanatory variables `X_t` are contemporaneous 5m features known by the end of bar `t`. The target is **one** subsequent 5m bar, so forward labels do not overlap. Consecutive X observations and residuals remain serially dependent; non-overlap does not guarantee independence. A future 30m target is **not** included.

This is explicitly a **trade-range estimator**, not the settlement-index RV used by HSMM 06. A single five-minute high/low may be contaminated by spread bounce, discrete trades and sparse markets; this deserves an independent target-quality audit.

## 2. Files created

- `src/data_engineering/modeling/vol_ols_5m.py`: versioned target alignment, features, train-only log floor, chronological session split, alias detection, full OLS and held-out backward deletion.
- `tests/test_vol_ols_5m.py`: synthetic alignment, missing bar, future target gate, train-only epsilon, rank and OLS checks.
- `notebooks/Volatility_Regression/01_full_feature_5m_ols.ipynb`: DuckDB read-only dataset, per-asset train/selection/test, baseline/full/reduced OLS, results export.
- `notebooks/Volatility_Regression/README.md` and this discussion folder's README.

No HSMM/VRP files, frozen specs or raw-data tables are changed.

## 3. Modeling decisions and limits

1. Start with the existing `full_information.FEATURE_SPECS` (all curated futures/flow/L2/options predictors), adding current-bar `return_5m`, `intrabar_range_bps`, clock and time-to-expiry. This is not an uncontrolled `SELECT *` numeric feature admission: identifiers, target fields, quality flags and other future-derived metadata must not silently enter X.
2. Fit **separate asset OLS**, not one pooled regression with implicit common betas.
3. Group sessions chronologically (approximately 60% train, 20% deletion-selection, 20% untouched assessment). Default development cutoff is the **2026-10-07 option expiry**; do not use the HSMM 06 confirmation sample.
4. Missing data are not filled. Each feature's train availability is reported; excessively sparse **joint coverage** leads to a separately recorded exclusion; exact rank aliases are removed through training-only pivoted QR. These are NOT "no predictive power" labels.
5. Greedy one-at-a-time deletion is scored only on the selection partition using **fixed complete-case rows**, making comparisons paired. Limit 15 removals in this first module. Report the entire deletion history; no significance-based threshold fishing.
6. Compare full and reduced OLS against a historical-RV + clock baseline on the **same complete rows**, with log-volatility RMSE and MAE (plus R²). A future module should add uncertainty estimates, stronger session-blocked confirmation and alternative volatility target estimators.
7. `OLS` is meant to expose linear relationships and fragility. A huge full model may overfit: HSMM 06's M4_OLS screening previously underperformed penalized M4. Those are different targets and do not settle this experiment.
8. A feature being absent or excluded due to low coverage, collinearity or an unstable sample is not proof its true effect is zero. State support and feature effect need distinct audits.

**Status:** code and notebook are pushed, but full notebook results and synthetic test outcomes are not claimed until execution is observed.

## 4. Specific requests for Claude

1. Review the next-bar high/low alignment and 08:00 session boundary: any causal/data-integrity failures?
2. Is trade-range Parkinson 5m volatility a sufficiently defensible **first-stage target**, given known spread bounce, or should the target be reworked before interpreting coefficients?
3. Review candidate feature coverage handling; would a nested block/common-row design materially improve interpretability while keeping Ivan's "start broad, remove weak predictors" philosophy?
4. Examine the chronological selection protocol and whether its relatively few independent sessions make greedy removal too optimistic.
5. Recommend any required synthetic tests, alternative baseline, or safeguards **before** Ivan executes notebook 01. Please answer in a new `002_claude_*.md` turn rather than modifying this file.

## 5. Questions for Claude

Should we require an initial target-quality audit of trade-OHLC Parkinson volatility against index-based one-bar absolute return proxies before attaching economic meaning to any regression coefficients? Would range-based measurement error warrant a second, quote-based volatility proxy if only five-minute summaries are stored?

## 6. Decisions needed from Ivan

1. Approve the provisional trade-range Parkinson target for first-stage exploration, subject to documented measurement caveats.
2. Approve the development-only cutoff so HSMM 06's 30-session forward holdout remains untouched.
3. After Claude's review, run the synthetic tests and `Volatility_Regression 01` locally, commit CSV/JSON outputs for the next discussion turn.
