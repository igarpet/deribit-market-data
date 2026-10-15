# RD01-c — Observable regimes in conditional signed-return asymmetry

**Protocol established October 15, 2026 before any RD01-c real-history run. Version RD01c-0.1.0.** Historical discovery only, not confirmation, trading approval or authorization to change AM-T-HAR-Blend. RD01 and RD01-b remain unchanged.

## Motivation and falsifiable question

RD01 showed that a continuous return-location Ridge was usually competitive with or better than hard state partitions. RD01-b tested trees built on out-of-sample location residuals, with results pushed October 15: at its 30-minute primary horizon, the raw residual tree had negative average NLL relative to interaction Ridge of approximately **0.000150**, but only **8/12** favorable asset-folds, failing its 9/12 gate. Three-bar confirmation performed worse (3/12); mechanical persistence is no reason to favor regime models. Do **not** repeat another post-hoc search for an attractive conditional mean state on these same folds.

**RD01-c question:** Do causally observable market conditions identify future **asymmetric** return-distribution shapes that improve heldout signed-return predictive density *after fixing continuous location and a single common conditional scale*? A negative result is entirely acceptable. The skew-state comparison must beat both a globally skewed and a continuously feature-dependent skewed distribution; outperforming only a symmetric null does not establish a need for discrete regimes.

## Frozen source, chronological evaluation and targets

Existing local `DERIBIT_DATA_ROOT/research/cs04a_feature_store/<ASSET>_features_and_forward_magnitude.parquet` contains causal spot and USD-M futures current-bar features. BTC/ETH primary; SOL/XRP secondary. No new data downloads, orderbook tick data, option snapshots or model modifications. Reuse RD01 `forward_labels`, `full_feature_days`, `research_folds`, `volatility_proxy` and the exact nine `TREE_COLS`, and RD01-b `feature_design`, `ridge_predict` and UTC horizon-anchoring. Reconstruct strictly future signed log returns at horizons **5/15/30/60 minutes**, rejecting gaps and any UTC-day-crossing target. **30 minutes is primary; the other horizons are descriptive.**

Use the same previously studied **three expanding structure/calibration/validation folds** in the FIRST 80% of complete historical days, with one-day embargo; omit the last 20% from model selection entirely. STRUCTURE: early 65% of structure days for a preliminary continuous Ridge, later 35% to choose tree splits using non-overlapping h-minute anchor targets. CALIBRATION: disjoint later training days used for fixed full Ridge conditional-location fits, one common Student-t scale, and all skew parameters; skew fit origins use non-overlapping horizon anchors. VALIDATION: later 47-day windows, no refits or state-label lookahead. This remains historical *exploratory* because preceding RD01/b and CS04-A inspected closely related data and models.

`volatility_proxy = sqrt(exp(log_rv24h) × (h/5))` is a **causal trailing 24-hour square-return proxy, NOT the AM-T-HAR-Blend forecast**. No claim to beat AM-T-HAR-Blend is permitted at RD01-c.

## Normalized two-piece Student-t and isolated asymmetry

Use a common Student-t with fixed degrees of freedom **ν=6** and horizon-specific, CALIBRATION-fitted common residual scale, inherited from RD01-b. For skew parameter `k` bounded to **[-0.55,+0.55]**, raw two-piece Student-t has left width `a=exp(-k)` and right width `b=exp(k)`, with normalized density `2/(a+b) × tν(z/a)` below zero and `2/(a+b) × tν(z/b)` above zero. **Recenter by the exact mean and renormalize by its exact variance** so every skew parameter yields an innovation with E[ε]=0 and Var[ε]=ν/(ν−2). At k=0 this reduces exactly to standard symmetric t6. Thus ONLY signed-return asymmetry changes; the location and fixed variance are identical between all direct-location variants. Unit tests must check normalization, zero mean, constant variance, CDF and exact zero-skew recovery.

Five registered comparisons:

1. `direct_symmetric`: TRAIN-calibrated 9-variable continuous Ridge location plus symmetric Student-t.
2. `interaction_symmetric`: same Student-t and common scale, but Ridge location with four prespecified feature interactions, the strong RD01-b benchmark.
3. `direct_global_skew`: **same direct Ridge location**, a single CALIBRATION-fitted skew for all market observations.
4. `direct_continuous_skew`: same location, skew continuously dependent on TRAIN-scaled momentum, flow, and basis via penalized three-feature linear fit, fitted ONLY on CALIBRATION anchors. Fixed L2=.04, bounded parameters; no CV hyperparameter selection.
5. `direct_regime_skew`: same direct Ridge location and common scale, skew conditional on a shallow CART **asymmetry structure** learned on late STRUCTURE; per-state parameters fit on later CALIBRATION anchors, with n/(n+350) shrinkage toward global skew and sparse states using global exactly. All state labels use features ending at current 5m bar, never future outcomes.

CART target: from late-STRUCTURE anchored out-of-sample early-Ridge standardized signed residual `z`, use bounded third-moment contrast `clip(z,-3,3)^3 − 3clip(z,-3,3)`; this targets residual asymmetric shape instead of squared loss on return mean. Tree: max 5 leaves, max depth 3, min 8% of structure-anchor rows and at least 50 per leaf, seed 2610. This diagnostic is not guaranteed to isolate skewness from other misspecifications. All thresholds structure-fitted; parameter calibration never looks at VALIDATION.

Per-state skew calibration must use at least **200 non-overlapping anchors on ≥20 days**; otherwise fall back to global skew and explicitly mark `SPARSE_FALLBACK`. No tuning or rematching tree leaves after viewing outcomes.

## Outcomes, statistics and controls

**Primary:** raw signed-return predictive **negative log-likelihood** (NLL), including the exact conditional scale Jacobian. Equal-day paired losses with 300 UTC-day bootstrap resamples (historical, **unadjusted**), per asset/fold/horizon. **Secondary:** Brier for left/right standardized residual tail beyond CALIBRATION-only absolute q95 and descriptive P(positive raw return), state-wise residual left/right quantiles. The tail test does not equate conditional model residual sign to spot-UP probability.

CART state persistence, duration and transitions are evaluated at primary 30m with **raw (no confirmation) labels**; episodes are censored at each UTC-day boundary. Report observed stay vs within-day iid occupancy; no inference that persistence proves structural regimes. Minimum state diagnostic support 200 origins across 20 days. Predicted skew and observation summaries are aggregate-only. **No number of learned states is a result by itself.**

Primary comparisons: `direct_regime_skew` vs EACH of `direct_continuous_skew`, `direct_global_skew`, `interaction_symmetric`. A **purely descriptive screening candidate** requires on the 30-minute horizon negative equally weighted asset-fold NLL delta, at least **9/12** favorable asset-folds AND BTC and ETH each negative mean and at least 2/3 favorable folds for **all three** references. No missing assets may produce a pass. The fact that RD01-b already failed means RD01-c cannot be described as independent confirmation regardless of bootstrap intervals. A mere improvement against symmetric t, but not continuous skew, means *asymmetry* not *discrete regimes*.

## Artifacts and integrity

Files: `src/data_engineering/modeling/observable_regime_discovery_c.py`; synthetic regression tests; Jupyter notebook `notebooks/Regime_Discovery/01c_conditional_skew_regimes.ipynb`; this immutable preregistration and an `01c/README.md`. Aggregates only in `notebooks/Regime_Discovery/01c/`: scores, paired bootstrap, comparison summary, state diagnostics, state calibration, tree definitions, episode summaries, transitions, hazards, persistence, coverage, manifest and decision. No private per-origin source data or synthetic results in GitHub. `RD01C_SYNTHETIC=1` tests with a temporary folder, only BTC and 30m horizon; these outputs cannot substantiate empirical claims. No change to collectors, prior notebooks, HSMM/RSD branches or AM-T-HAR-Blend.