# RD01 — Observable regime discovery and signed-return behavior

**Date:** 2026-10-15. **Status:** HISTORICAL_DISCOVERY_NOT_CONFIRMATION. Frozen before the first real RD01 historical run. Models, partitions and tests are not a trading authorization. RD01 is **new** and distinct from the existing `Return_State_Discovery/01` HDP-HSMM branch. Do not modify existing data collectors, prior research or AM-T-HAR-Blend.

## Research objective

Discover **observable** conditions that produce distinct signed-return probability distributions; determine whether any such conditions exhibit nontrivial persistence, transition patterns or duration behavior; test whether conditioning a common heavy-tailed return density on those conditions adds information beyond both a no-state model and a direct continuous-feature regression. Method neutrality: rules/trees are discovery approaches, not favored over existing HMM/HSMM, which will be compared later on matched features/origins.

**No independent confirmation is claimed**: historical data and CS04-A family screening have been repeatedly examined. The last 20% of complete CS04-A days, overlapping already inspected CS03/CS03b TEST, remain unused by RD01 selection. Prospective observations after 2026-10-14 are needed to confirm a frozen hypothesis.

## Inputs and point-in-time rules

- Read only **local** `DERIBIT_DATA_ROOT/research/cs04a_feature_store/<ASSET>_features_and_forward_magnitude.parquet`. It contains Binance spot five-minute log returns, existing spot/futures inputs and 34-ish causally aligned CS04-A features. Do **not** download new market data.
- Primary BTC/ETH, secondary SOL/XRP. Validate unique, ordered, five-minute UTC timestamps. Require complete UTC days (288 feature-ready origins). No interpolating or bridging gaps. Feature columns are from *completed* bars at origin `t`. Regime assignments at `t` must never see future returns.
- Verify and reconstruct signed net future returns, `R_h(t)=sum_{j=1}^{h/5}r_{t+5j}` for `h=5,15,30,60` minutes. Reject all targets with missing intermediate bars or reaching the next UTC date. Audits must catch disagreements with existing CS04-A 5/15/30m labels.
- Use the previously saved `log_rv24h = log(rv_slow)` as a **causal trailing 24h variance normalizer**: `v_h=sqrt(exp(log_rv24h)*h/5)`. This is a **proxy, not a persisted AM-T-HAR-Blend ex-ante volatility forecast**, and no RD01 score can be described as an improvement over AM-T-HAR-Blend. A matched actual AM-T density comparison is reserved for RD03.

## Three-way chronological estimation and evaluation

- Take the earliest 80% of eligible historical complete days as discovery-eligible, excluding last 20%. Reuse CS04-A's **three expanding folds** with validation at 50%, 64%, 78% of discovery days, validation width 8%, and one whole day embargo.
- Within each fold's training days, assign first 70% to **structure estimation** and remaining 30% to **return-density calibration**, without overlap. Rule thresholds and tree splits are fitted only on earlier structure days. Baseline, direct Ridge, and regime-specific location/shrinkage parameters are fitted **only on subsequent calibration days**, making these location estimates use the same response sample. No validation targets enter any fit, threshold, tree or scaling step.
- Evaluate all four approaches on the exact same completed validation timestamps and horizon-specific target grid. One fixed feature/model specification; no historical hyperparameter tuning or winner-selected p-value.

## Observable regime methods

**Fixed rule grid:** For current signed 5m momentum (`mom_r5`), current 5m USD-M quote-weighted aggressive flow (`flow5`) and spot–futures basis (`basis_bps`), fit structural one-third/two-third quantiles on earlier structure training days. Assign L/N/H for each dimension and form exactly 27 interpretable, named bins `ML/FH/BN`, etc. All regime thresholds are learned from TRAIN only; bins are not hand-labeled momentum/reversal states before observing subsequent return behavior.

**Constrained CART:** Fit a `DecisionTreeRegressor` to **next 30m net signed spot log return divided by trailing expected 30m volatility proxy**. Tree structure training uses only UTC `:00`/`:30` origin anchors (nonoverlapping 30m outcomes), from the earlier structure days, and truncates 0.5% tails using **structure-only quantiles** for the split criterion. At most 8 leaves, depth 3, minimum leaf max(120,4% of eligible anchor origins), fixed random seed 2607. Predictors: `mom_r5`, `mom_r15_mean`, `flow5`, `flow30`, `basis_bps`, `basis_change_bps`, `last_1m_r`, `minute_accel`, `log_trades`. State assignment uses the fitted tree at every subsequent five-minute origin without accessing its return. Freeze and publish all tree rules and node counts by fold.

**Continuous-feature control:** Ridge regression with fixed alpha=100 on those same nine CART features, fit on the calibration block (standardization also fit there). This is the mandatory direct conditional-mean benchmark: partitions must outperform a non-regime regression, not just the no-state baseline.

## Frozen common-return-density test

At each horizon, define `y=R_h/v_h`. On calibration days only, use response clipping at the [0.5%, 99.5%] quantiles for conditional **location** estimates. Base unconditional standardized location is the clipped mean. For rule/tree states, compute a state mean on calibration data, shrunk toward the global mean by `n/(n+350)`, and use global mean for any unseen state. For Ridge, fit standardized predictors on calibration data and predict `y` directly. Fix all four models to **one common, robust (99.5% winsorized residual second moment), calibration-fitted Student-t(df=6) conditional scale** per asset/fold/horizon. Do **not** introduce state-specific scales in RD01. Score raw signed-return continuous-density NLL, conditional nonzero UP/DOWN log loss and Brier. Unknown-at-origin nonzero conditioning is a *diagnostic only*.

For paired comparisons of each conditional model versus baseline, use 300 resamples of complete UTC validation days (seed 2607) to produce descriptive 95% unadjusted intervals. Do not call intervals of winner-selected combinations 'confirmed'. Report results by horizon and asset, plus a simple equal-asset-fold average for descriptive overview.

## Persistence, transition and episode diagnostics

- On the **entire validation feature-ready state sequence** (not only horizon-scored rows), count genuine adjacent five-minute transitions, excluding gaps; record `P(next state | current state)` and each state's empirical occupancy.
- Define contiguous same-state episodes. A missing neighboring observation is a censored episode boundary. Do not count right-censored episodes as completed state exits; the first episode after an interval boundary is left-censored. Publish aggregate counts, uncensored duration mean/median/90th percentile, and age-conditioned exit hazards using at-risk episodes. No episode-level times/labels are committed.
- Compare observed one-bar state persistence against a simple **within-day iid occupancy reference** `sum_k p_k(day)^2`, clearly identified as a *descriptive* reference, not a null preserving serial autocorrelation or a formal significance test. CART state labels may be unstable across folds: state IDs `T###` are NOT meaningfully comparable across refits. Regime characterization must use model definitions from the same fold.
- For each state/horizon, report held-out `n`, days, realized UP/DOWN/ZERO frequencies, mean/median signed return in bps, standardized mean and 5%/95% return quantiles, predicted UP probability, and average state-conditional NLL delta. Mark cells with <200 observations or <20 days as **descriptively insufficient** (these are NOT individual inferential tests).

## Result interpretation and required outputs

- Rank or shortlist **candidate regimes** only when their conditional signed-return density improvement is stable and survives direct Ridge comparison. Distinct empirical distributions do not establish *persistent latent* regimes. Frequent 5m switching is evidence of state conditions, not durable structural episodes.
- The target is the signed **physical-world spot return**. Options economic value, risk-neutral density, the HSMM comparison and AM-T-HAR-Blend integration are entirely separate later steps.
- New repo files only: `src/data_engineering/modeling/observable_regime_discovery.py`, `tests/test_observable_regime_discovery.py`, `notebooks/Regime_Discovery/01_observable_regime_discovery.ipynb`, `notebooks/Regime_Discovery/README.md`, and this preregistration.
- When run locally, output under `notebooks/Regime_Discovery/01/`: compact `01_scores.csv`, `01_comparison_summary.csv`, `01_paired_bootstrap.csv`, `01_regime_characteristics.csv`, `01_state_forecasts.csv`, `01_transition_matrix.csv`, `01_episode_summary.csv`, `01_exit_hazards.csv`, `01_persistence.csv`, `01_rule_definitions.csv`, `01_tree_definitions.csv`, `01_coverage.csv` and `01_manifest.json`. Do not push per-origin market data, individual episode timestamps, or fitted raw return vectors. The optional `RD01_SYNTHETIC=1` mode runs a synthetic smoke test into a temporary directory and cannot support market inference.
