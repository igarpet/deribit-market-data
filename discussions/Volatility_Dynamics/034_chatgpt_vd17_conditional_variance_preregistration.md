# VD 034 — Module 17: State-conditional variance calibration of the Student-t RETURN Markov model

**Author:** ChatGPT · **Date:** 2026-10-14 · **Status:** preregistered implementation, *before the first VD17 real-market run*. **Research branch:** Volatility Dynamics. **No options strategy, no trading signal.**

## Origin of the question

The executed VD16 retrospective module improved 60-minute return negative log-likelihood (NLL): native `AM_T` −4.099177, `AM_T_BOTH` −4.148467, direct horizon-specific `M2_student_t` −4.166375. The one-hour `AM_T_BOTH` fitted horizon variance multiplier was 1.5692 and variance-mixture shape 1.7352. It improved nominal 95% interval miss rate to 5.20% but raised the training-threshold tail-event predicted frequency to 2.558% versus 1.866% realized; tail Brier 0.017723 vs M2 0.017404. A constant multiplier cannot express which **market states** require more or less variance. Follow-up of VD16 is already adaptive/hypothesis-generating research; previous market-data TEST period is no longer untouched.

## Specific scientific question

After fixing the Student-t innovations **in the 5-minute return transition** and the global horizon-specific VD16 shape/scale correction, do additional **causal state-dependent variance adjustments** improve return density, net-return variance scores and tail calibration? Are apparent improvements attributable to Markov state information or simply to importing M2's better direct horizon-specific conditional variance estimate?

**Scope:** Forecast variance corrections **only**, not re-estimation of AR(1) volatility transition matrices `phi`, their innovation covariance `Q`, the 5-minute return `nu`, return AR(1) `phi`, jump processes, or new options features. We intentionally distinguish the forecasting overlay from a structural state-process modification. VD17 calibrated outputs may violate cross-horizon Markov path consistency. Revisit structural variance-state dynamics only if this diagnostic separates its value from M2 anchoring.

## Data and independence

- Refit same-branch implementations `markov_lift_distributions` (VD14), `student_t_return_markov` (VD15), and `markov_return_horizon_calibration` (VD16) directly from raw Binance spot 5-minute closes. **No prior fitted artifacts, score CSVs or other research-branch models are imported into fitting.** No collector/DuckDB/Deribit-index write or tick storage.
- Assets BTC, ETH, SOL, XRP, AVAX, TRX; HYPE excluded if fewer than 365 historical days. Sample strictly earlier than 2026-09-19 UTC. TRAIN/VALID/TEST split by chronological UTC day 60/20/20. At any origin, use state derived at or before that bar; targets are `t+1 ... t+k` with missing/split crossings censored.
- All **7 horizons**: 5, 10, 15, 30, 60, 120, 240 minutes. Primary 60 min; others descriptive.
- VD16 global `(s_h,a_h)` fitted by penalized NLL on first 60% of VALID calendar dates (uniform maximum 512 origins per asset). Late VALID (remaining 40%) is time-separated and purged by at least the complete horizon at the boundary. VD15 original native return `nu` is still chosen on full VALID as in previous modules, so late VALID is not completely untouched for the *underlying model*. No claim of pristine held-out validation.
- Historical TEST is scored on common origins for all models, but **is previously inspected in VD13–VD16**. Therefore all positive TEST comparisons are `HISTORICAL_EXPLORATORY`, never prospective confirmation; cumulative historical model selection means bootstrap intervals cannot be interpreted as final out-of-sample alpha evidence.

## Six frozen models, unchanged return innovations and distribution shape

Let \(v_{t,h,i}\) be the `VD15` conditional-Gaussian QMC component return variances given origin \(t\), horizon \(h\), particle \(i\). VD16 `AM_T_BOTH` applies:

\[
\tilde v_{t,h,i}=s_h\bar v_{t,h}\,{v_{t,h,i}^{a_h}\over\overline{v_{t,h}^{a_h}}}
\]

where \(s_h,a_h\) are fitted from early VALID. VD17 **preserves the whole normalized particle-variance shape** and multiplies each component by \(\exp(\delta_{t,h})\). In particular the predicted variance becomes \(s_h\bar v_{t,h}\exp(\delta_{t,h})\); the conditional-return location stays zero. No new Student-t fit to volatility.

| Model | Variance treatment |
|---|---|
| `AM_T` | Native 5-minute Student-t Markov process propagated without horizon calibration |
| `AM_T_BOTH` | Refit exact VD16 horizon global scale + shape, no conditional correction |
| `AM_T_VAR_STATE` | Conditional log scale \(\delta=\mathrm{clip}(x_t'\beta,[-0.85,0.85])\). Predictors: causal hour-deseasonalized `slow`,`gap`,`now`,`slope15`, UTC clock sin/cos, pooled asset fixed effects. Standardize and clip predictors using **early VALID only**. Penalized early-VALID NLL with L2 0.025 and per-coefficient bounds [−0.6, 0.6]. |
| `AM_T_VAR_ANCHOR` | \(\delta=\mathrm{clip}(w\,z_{t,h},[-0.85,0.85])\), where \(z=\mathrm{clip}[\log(v_{M2,t,h}/v_{AM\_T\_BOTH,t,h}),[-1.5,1.5]]\). Estimate \(0\le w\le1.25\) with NLL penalty 0.003w² on early VALID. `v_M2` is the **TRAIN-only** fitted VD14 horizon-dependent predictive net-return variance, not realized future variance; no M2 *density* imported. |
| `AM_T_VAR_COMBO` | Joint \(\delta=\mathrm{clip}(x_t'\beta+wz_{t,h},[-0.85,0.85])\), same fit data/limits/penalties. |
| `M2_student_t` | Refit the original VD14 direct horizon-specific Student-t density benchmark, unchanged. |

At the fit stage, optimize `AM_T_VAR_STATE`, `ANCHOR`, `COMBO` separately. Score all on **late VALID** for a frozen, descriptive family selection, but report all candidates on historical TEST. Do not select a new variant by inspecting TEST wins.

## Correct interpretation of the anchor test

A high anchor weight plus good performance does **not** validate new independent information inside AM's Markov transition dynamics: it indicates that M2's variance forecast carries information the current Markov path integration does not capture. Any hybrid loses the claim of an independently specified native generative market process. The STATE model is the cleaner independent state-based test. A strong test of the structural Markov proposition requires the STATE model (not only a borrowed M2 anchor) to improve the previous baseline.

## Scoring, numerics, decisions

- Primary: common-origin mean negative log-likelihood of net log return, lower better. Paired resampling of **whole UTC days** with all assets and overlapping origin times kept together; 1,000 bootstrap replicates, descriptive 95% percentile intervals.
- Conditional net-return **variance** evaluation: \(\operatorname{QLIKE}_{t,h}=\log\hat v_{t,h}+R_{t,h}^2/\hat v_{t,h}\), appropriate for a zero-conditional-mean net-return variance forecast. A one-day correlated sequence is resampled as a block; cross-day dependence may still remain. QLIKE has no constant penalty term and can be negative.
- Tail proper log-loss/Brier and 95% central predictive interval miss from the **same fitted density**. Absolute-return tail indicator uses asset- and horizon-specific TRAIN q95 threshold. Report mean predicted/observed frequencies, per-asset results, and fitted variance correction coefficients.
- 256 scrambled Sobol QMC nodes with seeds inherited from VD15 and horizons fitted/evaluated at same nodes within each model. Numerical sensitivity 256→512 at 5, 60, 240 minutes on **late VALID only**, not TEST. Do not claim small changes are robust without convergence.
- Predeclared 60m comparisons: `STATE−BOTH`, `ANCHOR−BOTH`, `COMBO−BOTH`, `COMBO−ANCHOR`, `COMBO−STATE`, `COMBO−M2` (extra references `BOTH−AM_T`,`ANCHOR−M2`). A negative NLL difference and whole-day bootstrap interval entirely below zero is historical statistical support **only**; check tail Brier, logloss, 95% interval, and QLIKE before considering the model better calibrated.
- Calibrator family and seven-horizon comparisons are multiple exploratory tests; CI coverage unadjusted. No net option edge, jump discovery, risk-neutral calibration or HSMM state conclusion is allowed.

## Artifact contract

Commit **unexecuted** notebook `notebooks/Volatility_Dynamics/17_markov_conditional_variance.ipynb`, module `src/data_engineering/modeling/markov_conditional_variance.py`, tests `tests/test_markov_conditional_variance.py`, this protocol in `discussions/Volatility_Dynamics/034_chatgpt_vd17_conditional_variance_preregistration.md`. When executed locally, save only: `17_scores.csv`, `17_bootstrap.csv`, `17_per_asset.csv`, `17_variance_fit_grid.csv`, `17_particle_sensitivity.csv`, `17_decision.json`, `17_manifest.json` within numbered output `17/`. **No origin-level probabilities, particle paths, raw returns or market data dumps**.
