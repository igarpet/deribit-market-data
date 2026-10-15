# RD04 — Historical robustness and falsification audit of the RD03 density overlays

**Frozen plan, October 15, 2026. Status: REPEATEDLY_INSPECTED_HISTORICAL_ROBUSTNESS.** RD03 reports no primary candidate with nominal improvement over both original VD18 AM-T-HAR-Blend and the direct continuous-feature mean model. This is not a license for new hyperparameter searches, changing regime definitions, or presenting retrospective intervals as prospective evidence. No model selection or integration into VD18 or collectors.

## Hypothesis and no-search contract

Question: Do the **unchanged** causal RD03 observable CART mean overlay, 3-state HMM mean overlay, or 2-stage age-hazard semi-Markov mean overlay deliver **stable incremental signed-return density** gains over BOTH (i) zero-mean frozen VD18 and (ii) the RD03 continuous-feature `direct_mean` across diagnostic (September 1–18, 2026) and evaluation (September 19–October 13, 2026), separately for BTC and ETH at **30 minutes**? Stability is necessary, not sufficient, for a future confirmation candidate. SOL/XRP secondary; 5/15/60 minutes descriptive. Never re-rank features or thresholds based on these periods.

The original RD03 model specification and archive are **unchanged**: STRUCTURE March 1–April 30, RESPONSE CAL May 1–August 31. Parameters frozen pre-September; native RD03 HMM and CART fit on STRUCTURE, response parameters fit only on RESPONSE. The September diagnostic interval is assessed for the first time *by this branch* but it is **not untouched project-wide**. Evaluation was inspected in RD03. No independent holdout remains; freshly collected subsequent data is needed.

## Mandatory point-in-time and model identity controls

Import, do not duplicate, `regime_density_bridge` v0.1.0 and its archived VD18 construction. Read `18_manifest.json` and call `confirm_parameters` / `archived_split_frame` to reconstruct exact original clock normalization. Read LOCAL `cs04a_feature_store` predictors, do NOT use stored future labels for features. Refit RD03 structure/response using identical frozen parameters, exact 00/30 anchor origins, same return labels and variance particles across methods. Retrieve native VD18 horizon-specific mixture variances from RD03 `vd18_particle_variances`. Fit RD03 models via `prepare_methods` once per asset, score both diagnostic and evaluation. **Abort** if every asset/horizon/method's recalculated evaluation NLL, sign Brier, and origin count do not reproduce RD03's committed `03_scores.csv` within strict tolerance. No trailing RV substitution.

## Frozen robustness views

1. **Daily paired NLL and UP Brier:** compare each candidate to original VD18, direct mean; HMM additionally versus static emissions; semi-Markov additionally versus HMM. All seven RD03 methods on exactly matching samples. Negative model-minus-reference difference is better.
2. **Chronological stability:** separate diagnostic and evaluation windows, each split into earlier/later halves based only on dates. Score whole-day equal-weight delta, share of favorable days, nominal paired bootstrap percentile intervals (1000 draws, seed 2614); minimum 12 days for full, 6 days for a half. Daily and half-window autocorrelation is not fully captured by a simple day bootstrap; confidence intervals are only descriptive.
3. **Ex-ante volatility stress:** define 33rd/67th tercile cutoffs of **forecast VD18 variance** on diagnostic origins, separately asset/horizon. Apply fixed cutoffs to diagnostic and evaluation, compare paired NLL and Brier in three buckets. At least five distinct UTC days per bucket; distinguish changes in event support from actual model gain. Do not condition buckets on future realized absolute returns or future variance.
4. **Multiplicity diagnostic:** six predeclared primary tests = 3 regime-vs-direct × BTC/ETH at 30 minutes in September diagnostic, one-sided Bonferroni bootstrap upper bounds (`1 - .05/6` quantile), 1000 draws. This is a limited within-RD04 screen only: it does **not** repair prior RD01–RD03 model selection and is not a valid prospective discovery p-value. Evaluate all windows and original/direct comparisons regardless of the screen.
5. **Decision:** a *nominal historical robustness candidate* requires negative upper 95% day-bootstrap bounds versus both VD18 and direct mean, in BOTH diagnostic and evaluation windows, separately BTC and ETH. No favorable pooled average or secondary asset can rescue failed primary cells. Regardless, `prospective_confirmation=false`, `AM_T_HAR_Blend_modified=false`. Report `NO_STABLE_INCREMENTAL_REGIME_EVIDENCE` if the gate fails. Even a gate pass would remain exploratory.

No retraining on September/October, no tuning regularization, feature families, state count, streak thresholds, emissions, Student-t df, blend weights, prediction horizon, or variance particles. The semi-Markov model remains an RD02-style **two-stage duration overlay**, not a jointly estimated HSMM. This RD04 module does not test latent regime covariance, mixtures of state-specific densities, return skew, or portfolio performance.

## Reproducible outputs

- `src/data_engineering/modeling/regime_robustness.py`
- `tests/test_regime_robustness.py`
- `notebooks/Regime_Discovery/04_regime_robustness.ipynb`
- `notebooks/Regime_Discovery/04/README.md`
- 04 outputs: `04_daily_pairs.csv`, `04_temporal_stability.csv`, `04_volatility_stress.csv`, `04_sigma_terciles.csv`, `04_primary_multiplicity.csv`, `04_rd03_reproduction.csv`, `04_summary.csv`, `04_manifest.json`, `04_decision.json`.

Push only these **aggregate** outputs after the real run. Do not upload raw/local feature stores, individual origin probabilities/returns/variance particles, sensitive .env, or synthetic output. `RD04_SYNTHETIC=1` smoke test must never be interpreted as market evidence.
