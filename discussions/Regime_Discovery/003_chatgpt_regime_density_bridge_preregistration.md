# RD03 — Regime-dependent return location against the actual VD18 AM-T-HAR-Blend density

**Research specification v0.1.0 · 2026-10-15 · HISTORICAL EXPLORATORY.** Designed after inspecting RD01, RD01-b, RD01-c and RD02 negative screening results. Historical results (including September–October 2026) have already been inspected elsewhere in this project. **No fresh or prospective significance claim is allowed.** No authorization to modify AM-T-HAR-Blend, its variance machinery, Deribit collectors, or live trading systems.

## Research question

Can observed or latent regimes inferred from *already collected* Binance perpetual/spot microstructure covariates improve **proper signed-return predictive log density** beyond the **exact frozen VD18 AM-T-HAR-Blend** return density and a direct continuous-feature return-location model, when every method sees **identical ex-ante VD18 conditional variance particles**? The primary horizon is 30 minutes, with secondary 5, 15 and 60 minutes. BTC/ETH are primary, SOL/XRP secondary. The structure of RD02 failed its historical return NLL screen; RD03 is a falsification/transfer exercise rather than an assumed integration success.

## Frozen baseline provenance, a critical no-lookahead constraint

Use the **actual serialized fitted parameters** in `notebooks/Volatility_Dynamics/18/18_manifest.json` from the `VD18-0.1.0` historical run whose `sample_last` precedes `2026-09-19T00:00Z`. Do **not** load the old `18_scores.csv` as point-in-time per-candle predictions; it stores aggregates only. Do **not** fit VD18 on the RD03 evaluation dates. Recompute the original native Student-t variance particles via `markov_return_horizon_calibration.variance_particles`, transform them using `markov_conditional_variance.component_variances` with the archived AM_T_BOTH calibration, then apply `markov_har_variance`'s archived selected HAR and dynamic anchor-weight adjustment **exactly as in VD18.evaluate_horizon**. No approximation to this model by trailing RV, HAR alone, a Gaussian with mean variance, or arbitrarily refitted Student-t.

The frozen VD18 model used a historical sample before September 19, 2026 and **TRAIN-normalized** clock/volatility features. Reconstruct VD14's original pre-September-19 chronological TRAIN/VALID boundaries before `make_features` on the subsequently extended spot history; never call `panelize` on the later full sample and silently use changed TRAIN clock normalization. Load the **six archived assets** (AVAX/BTC/ETH/SOL/TRX/XRP) to reproduce original split boundaries, even though RD03 scores only BTC/ETH/SOL/XRP. Fail closed if the recorded archive version, chronological fit cutoff or historical split fails to match. Regression check: with zero mean overlay, the new density NLL must equal `markov_conditional_variance.loss_components(...)["nll"]` to numerical tolerance on matched points.

The archived VD18 construction is horizon-specific and **not necessarily path-consistent** across different forecast horizons; RD03 does not make it path-consistent or change the volatility engine.

## Observation/target and period contract

- Existing spot close-to-close five-minute log returns from `DERIBIT_DATA_ROOT/history/price_5m/<ASSET>.csv.gz`. Input microstructure from local CS04-A Parquet `DERIBIT_DATA_ROOT/research/cs04a_feature_store/<ASSET>_features_and_forward_magnitude.parquet`; read **feature columns only**, never CS04 forward targets for model input. Join by exact UTC `state_time`. The sole response label is VD14's strictly later `R_h`; retain no target at time t or earlier. All horizons use exact five-minute future increments with no missing-gap bridging and original VD18 split-crossing censoring.
- Non-overlapping forecast **origins** on UTC 00/30 thirty-minute clock (not nonoverlapping 60-minute returns). Filter requires ready point-in-time VD18 features, HAR lags, local CS04 readiness and valid future response labels for **all** horizons. All methods score common origins. A 60-minute label may overlap the next anchor: bootstrap resamples whole days, not origins.
- Primary training/validation/evaluation fixed windows, no grid search: STRUCTURE 2026-03-01 through April 30 (exclusive May 1); RESPONSE calibration May 1 through August 31; diagnostic September 1–18 **report-only, no selection or refit**; EVALUATION September 19 through available October 2026 history, excluding October 14 midnight and later. There is no independent pristine holdout. Archived VD18 parameters were fitted and selected on periods before its original historical sample end; RD03 response overlays fit on data available before September 1, 2026.
- State probabilities use only market measurements **available by the current completed bar**; HMM and semi-Markov filter forward causally, restarting at missing 30-minute anchors. Never use Viterbi-smoothed states for future forecasts. Train-only Viterbi on STRUCTURE is allowed only to estimate the two-stage duration hazards.

## Frozen candidate response models

Every method receives the same zero-centered **VD18 mixture-of-Gaussians conditional on numerical Student-t mixing particles**. RD03 tests **location effects only**. All candidate conditional mean shifts are learned as a standardized return z = `R_h / sqrt(mean VD18 particles)` and applied as `mu=clipped(z_hat)*sqrt(mean VD18 particles)`, with |z_hat|<=0.5 ex-ante sigma. The **identical set of VD18 mixture variances** is used with every method; per-origin predictive variance is therefore fixed. No refitting Student-t degrees of freedom, mixing shape, HAR family/parameters or volatility-state transition parameters.

| Name | Incremental signed-return location |
|---|---|
| `AM_T_HAR_BLEND` | Original unmodified zero-centered VD18 baseline: mu=0 |
| `sign_only_mean` | CAL-only Ridge predictor using just the current five-minute return sign (simple return-side control) |
| `direct_mean` | CAL-only Ridge(α=100) regression on fixed continuous covariates |
| `observable_tree_mean` | Same direct model + residual state offsets; 5-leaf max, depth 3 CART rules fit on earlier STRUCTURE 30m standardized returns; state offsets CAL only with pseudo-count shrinkage 120 |
| `latent_static_mean` | Same direct model + CAL-fitted latent state offsets weighted by emission-only posterior; ordering removed |
| `latent_hmm_mean` | Same direct model + CAL-fitted state offsets weighted by causal HMM posterior |
| `latent_semi_markov_mean` | Same direct model + CAL-fitted state offsets weighted by causal age-dependent duration-overlay posterior |

HMM fitting reuses RD02's frozen 3-state, independent-t(6)-emission implementation with ten EM iterations on six causal STATE_FEATURES, standardizing on STRUCTURE only. Its duration overlay is **two-stage Viterbi-train hazard fitted** and NOT a jointly fitted HSMM. All latent methods use the same STRUCTURE-fitted HMM emission/transition parameters. All return-response parameters fit on CAL only, including latent residual offsets. The tree is structure-fitted; it does not see CAL outcomes to define thresholds. The direct predictor uses current momentum, flow, basis, intraminute path, activity columns. All model choices and shrinkage strengths were frozen before executing RD03 historical data.

**Additional mandatory control:** the signed-only return-location model must be scored on exactly the same origins as all other models. Neither a direct feature model nor a regime model can claim useful complexity merely by beating a zero-drift baseline when the sign-only rule performs equally well.

**Prediction limitation:** We mix HMM state means *before* applying a **single conditional-location shift** to the VD18 mixture, rather than integrating separate state-specific return densities. This retains exactly fixed conditional predictive variance and isolates only expected-return information; it does **not** measure state-dependent skewness, multimodality or full uncertainty over state-specific locations. These would require a separate distribution-shape experiment, not a silent extension here.

## Scoring and research gates

- Primary: matched 30m raw signed-return negative log likelihood (NLL), lower better. Additional 5/15/60-minute raw-return NLL and UP probability Brier, per-asset and pooled (equal-weight asset) reporting. A zero return counts as not UP in the unconditional probability Brier; interpret ties as explicit and separately report their frequency. `AM_T_HAR_BLEND` is recomputed pointwise, never approximated by the historical aggregate metrics.
- Pair each overlay against original VD18 on identical origin timestamps using a whole-UTC-day paired bootstrap (500 draws, seed 2613). Report effects, 95% nominal intervals and per-asset days, **exploratory only**. No post-selection p-values. Summarize direct-vs-observable, direct-vs-HMM and HMM-vs-static separately to identify incremental regime information beyond ordinary continuous conditioning.
- Regime advantage requires (i) lower signed-return NLL vs original baseline **and** vs direct_mean; (ii) favorable results in both BTC and ETH with no adverse sign Brier calibration; (iii) HMM temporal advantage over static emissions if claiming persistence; (iv) for explicit-duration benefit, semi-Markov beats HMM. A difference in state occupancy, duration or clustering alone is not evidence of forecasting information. RD01–RD02 already had negative results; negative RD03 is a valid successful completion.
- No confirmation and no trading claims regardless of nominal historical interval. Prospective freeze and truly fresh observations required. No modifications to VD18, options valuation or execution pipeline.

## Artifacts

New module `src/data_engineering/modeling/regime_density_bridge.py`, test `tests/test_regime_density_bridge.py`, notebook `notebooks/Regime_Discovery/03_return_density_bridge.ipynb`, README `notebooks/Regime_Discovery/03/README.md`. Save only aggregate CSV/JSON under `notebooks/Regime_Discovery/03/`: matched scores, paired-day bootstrap, comparison summaries, structure/response diagnostics, manifest and decision. Keep per-origin variance particles, returns, filtered posterior state probabilities and raw market data **local only**. Synthetic smoke mode `RD03_SYNTHETIC=1` validates scoring and output structure using temporary output; it never produces empirical evidence. Large cached posterior/variance arrays are never committed.
