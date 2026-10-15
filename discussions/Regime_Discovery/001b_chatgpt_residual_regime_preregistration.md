# RD01-b — Residual regime discovery with causal confirmation

**Established October 15, 2026 before the RD01-b real-history run.** Historical exploratory only. Do not change RD01, CS04-A, collectors, HSMM branches or AM-T-HAR-Blend. Previous RD01 and CS04-A validations have already been viewed, so no split here is pristine prospective confirmation.

## Why RD01-b exists

RD01 found that a continuous Ridge conditional location model generally improved 5m and 15m signed-return density more than 27-cell rule states or a shallow raw-return CART. The research null is that continuous known features are enough. RD01-b tests whether a **short, horizon-specific residual-state correction** improves beyond both a direct model and a direct model with explicit pairwise interactions. A state's persistence is not evidence of a regime if it results from a smoothing rule.

## Frozen inputs and periods

Reuse the local CS04-A feature store `<DERIBIT_DATA_ROOT>/research/cs04a_feature_store/<ASSET>_features_and_forward_magnitude.parquet` for BTC/ETH primary, SOL/XRP secondary. Only use `state_time`, current-bar `return_5m`, `feature_ready`, `log_rv24h`, and RD01 `TREE_COLS` (past/current returns, futures executed flow, futures-spot basis, minute dynamics, trade count). Reconstruct next-5,15,30,60-minute signed net log returns strictly after each state_time; reject missing/intermediate gaps, midnight-crossing targets. No new downloads, no L2, no implied-volatility snapshots.

Exactly inherit RD01's three historical discovery folds: first 80% complete days only; expanding STRUCTURE period, later disjoint CALIBRATION period, one UTC-day embargo, subsequent 47-day VALIDATION. The final 20% of historical days is omitted from model search. All threshold fitting, feature scales, ridge parameters and residual-state means use historical training periods only. Fix all four horizons a priori. **Primary comparison horizon: 30 minutes**; 5, 15, 60 minutes are supporting diagnostics.

The volatility normalizer is the trailing 24-hour mean five-minute squared return multiplied by forecast horizon length, i.e. **RD01's causal proxy**. This is NOT the point-in-time variance prediction of AM-T-HAR-Blend. On identical validation targets, all five models share one TRAIN-only Student-t(6) standardized innovation scale; only location differs. No regime-specific volatility, skewness or Student-t df fitting.

## Five fixed predictors and tree inference

1. `baseline`: constant TRAIN-calibrated standardized return location.
2. `direct_ridge`: Ridge(alpha=100) on the nine continuous RD01 `TREE_COLS`, with TRAIN-only standardization and 0.5/99.5% winsorized standardized TRAIN outcomes.
3. `interaction_ridge`: same nine continuous variables plus **four** TRAIN-standardized feature products: momentum × flow, flow × basis, momentum × basis, and last-minute return × flow. Same Ridge alpha.
4. `residual_tree`: start with direct Ridge prediction, then add a shrinkage-adjusted state-specific residual correction. Train a CART structure with max 6 leaves, max depth 3, min 7% of tree-fit rows (at least 55), no hyperparameter search.
5. `residual_tree_sticky`: identical residual-tree leaf structure, but the online state changes only after **three consecutive five-minute leaf observations** agree on a new leaf. At a gap or each new UTC day, initialize state from currently observed leaf. No Viterbi, full smoothing or next-bar information. Residual corrections are calibrated for these confirmed-state identities independently on CALIBRATION.

The tree is trained separately at each horizon. Inside each STRUCTURE fold, fit its preliminary direct Ridge on the **first 65% of structure days**. For tree structure, use only the **remaining 35% of structure days** with nonoverlapping UTC-horizon anchor origins. CART targets are the normalized realized next-horizon returns **minus those earlier Ridge forecasts**, clipped at tree-fit 0.5/99.5% quantiles. Structure training and residual target construction are strictly chronological. Train the final direct Ridge and interaction Ridge on later, separate CALIBRATION days. For each regime, calibrate the final Ridge's standardized residual correction using later CALIBRATION observations: `n/(n+500) * mean_clipped_residual`. Unseen leaf identities default to zero correction, not a fitted validation value. No refit on VALIDATION.

**Interpretation:** this is a conditional location correction, not an HSMM, and not an assertion of structural state switching. Hysteresis guarantees artificial dwell; its apparent persistence must never count as independent evidence of market regimes. Each horizon's fitted tree can have different leaf definitions and cannot be compared as an identical state sequence.

## Prespecified evaluation and controls

- Raw signed-return proper **Student-t density NLL** on shared validation origins; primary 30m. CONDITIONAL UP/DOWN log loss is secondary, restricted to next nonzero returns; it is not alone an actionable direction forecast.
- Mandatory comparisons: `residual_tree - direct_ridge`, `residual_tree_sticky - direct_ridge`, both residual models against `interaction_ridge`, plus all candidates against baseline. Score paired complete UTC days, 300 bootstrap replicates (seed 2608). Report per-asset/per-fold before pooling equal asset-fold means. Bootstrap intervals are **exploratory unadjusted**, cannot establish selected-model confirmation.
- Report state occupancy, observed standardized future return, next-return positive/negative frequencies, observed residual after continuous Ridge, frozen TRAIN-predicted residual correction, heldout dates, and minimum support >=200 observations on >=20 days.
- Report raw versus confirmed-state five-minute persistence, daily-boundary-censored episodes and within-day transitions for the **30m-tree** states only. Even a large persistence increase after three-confirmation is mechanically induced.
- A **descriptive historical candidate** at 30m requires both negative average NLL relative to interaction Ridge and >=9 favorable of 12 asset-folds; with missing assets, it is not a primary cross-asset result. Not significance and not deployment approval. Emphasize BTC/ETH consistency and the persistence of results on future data.
- No overlay on the actual AM-T-HAR-Blend density is claimed; that requires matched ex-ante variance and full innovation density in RD03. No option trading or market returns are authorized.

## Saved artifacts

- New module `src/data_engineering/modeling/observable_regime_discovery_b.py`;
- Regression checks `tests/test_observable_regime_discovery_b.py`;
- Reproducible notebook `notebooks/Regime_Discovery/01b_residual_regime_validation.ipynb`;
- Compact CSV/JSON outputs only under `notebooks/Regime_Discovery/01b/`: scores, paired-day bootstrap, comparison summary, state forecasts, 30m-tree persistence and episode summaries, within-day transition matrix, tree definitions, coverage, manifest, decision.
- Local historical CS04-A Parquet remains entirely local. Do not publish origin-level rows, raw labels or dates of individual episodes.
- `RD01B_SYNTHETIC=1` runs a synthetic smoke test in a temporary folder; never commit or interpret its output as evidence. No parameter changes after reviewing RD01-b historical scores except through a new separately preregistered module.
