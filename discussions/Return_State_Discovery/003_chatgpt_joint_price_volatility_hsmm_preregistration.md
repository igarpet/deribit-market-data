# RSD 003 — Joint price–volatility coupling states over AM-T-HAR-Blend

**Date:** 2026-10-14 · **Branch:** Return State Discovery · **Status:** specified and synthetic-tested **before any Binance RSD03 execution**. Historical exploratory, not independent confirmation or trading authorization.

## Question and independence

RSD01 found 5–6 occupied return-path HDP-HSMM states and no forecast gain. RSD02 found 8–10 occupied volatility-only states, but no incremental return-density or directional gain over a continuous-volatility Student-t. Both negative controls occupied many states without true discrete switching. RSD03 asks a *different*, narrower hypothesis: **does the relationship between standardized signed price displacement and unexpected realized volatility form persistent states that improve the NEXT return distribution beyond continuous price–volatility interaction?** The answer may be no. Prior fitted state labels and parameter estimates are NEVER imported.

## Data, timestamps, chronology

- Source: same Binance spot 5-minute closes under `DERIBIT_DATA_ROOT/history/price_5m`; BTC and ETH only initially. Existing VD14–VD18 implementations and RSD01 pretest-only 30-minute return-density refit are reused as **code**, not previously fitted artifacts. No Deribit ticks, database changes, collectors or options signals.
- All samples strictly before `2026-09-19T00:00Z`, divided by original VD18 60/20/20 chronological full UTC days. Entire 150-day original VD18 historical TEST is already inspected by multiple prior research iterations; subdivide it 60% fit, 20% diagnostic, 20% return/sequence evaluation. Any bootstrap interval is retrospective/exploratory.
- Nonoverlapping UTC 00/30 half-hour blocks. At start `t`, obtain native VD18 *terminal* net-return predictive variance `Vret_t` and TRAIN-fitted HAR conditional **log realized-variance scale** `Vrv_t = exp(E_TRAIN[log RV30|past])` from the selected HAR feature family. This is *not* E[RV30]; predictive features retain this distinction. The selected family comes from pre-VD18-test VALID only; same family used to refit HAR on original TRAIN only. Require exact six subsequent observed five-minute returns for actual `R_t = sum(r5)` and `RV_t = sum(r5²)`; they become known only at block END. No missing-bar bridging. Assert return matches VD14 target. HAR forecast uses TRAIN-only transforms/asset scale (the net-return scale is explicitly undone to recover exp(E[log RV])). Numerical floors for RV are trained on RSD03 fit only, frozen before evaluation.

## Joint feature and construction

At the END of each completed block, define

- `Z_t = R_t / sqrt(Vret_t)`: signed standardized net return;
- `U_t = log(max(RV_t, epsilon_RSD03_TRAIN) / Vrv_t)`: log realized-variance surprise relative to HAR's log-RV scale;
- `C_t = Z_t * U_t`: signed coupling DIAGNOSTIC, not a univariate HSMM emission.

Primary HSMM observation is **bivariate** `X_t=(Z_t,U_t)`. Do NOT collapse the two dimensions into `C`: its sign conflates upward-expansion and downward-compression. Z is never clipped, since clipped net returns would invalidate proper return density scoring. The log RV surprise may be clipped to [-9,9] for numerical stability, and this must be reported. The joint features are standardized using robust TRAIN-only normalization. HAR forward realized variance and terminal return variance must never be interchanged. `U` and `Z` are mechanically dependent through the same six returns, so generic covariance/mixture structure under a Student-t null is NOT regime evidence.

## Primary Bayesian HDP-HSMM

Finite weak-limit hierarchical-DP explicit-duration sampler from RSD01, with STUDENT-T(6) **bivariate** emissions, state-dependent mean and unrestricted 2×2 covariance. `Kcap=16` is computational ceiling, **not** a K-grid choice or economic state count. Global beta ~ symmetric Dirichlet(gamma/Kcap); row-specific off-diagonal transitions, no self-transitions at episode boundaries; shifted negative-binomial duration per state. Age-state memory cap 96 half-hour blocks is not a compulsory transition. Primary priors inherited unchanged from RSD01 (`alpha=3`, `gamma=0.7`, `duration_shape=2`, duration Beta(2,8)); 2 independent chains, seed 1801 and 2909; 160 iterations, 80 burn, thin 4, 20 draws per chain; 4 representative draws per chain used for prediction. These are short exploratory chains, NOT a convergence demonstration. Report K posterior/histogram, chain agreement, at-cap probability, duration-boundary rate, truncated-state warning, state covariance/correlation, occupancy and episodes. Do not compare numerical internal labels across chains without matching.

## Causal forecasting and decomposition

All observations for block t are only available **AFTER** its net return is scored. For forecasting block t, posterior state probabilities come from completed blocks `<t`; current block ex-ante `Vret_t` may be used as forecast-scale Jacobian. At gap boundaries reset causal filter/lagged predictors. HSMM fitted only on first 60% of old TEST; no training or posterior state refit on EVALUATION.

**Primary predictive outcome**: next nonoverlapping 30-minute **raw signed return NLL**. For each selected posterior draw: filtered prior HSMM weights; fit state-specific Student-t(6) return location, previous return-AR and residual scales by probability-weighted, shrunk training regression (RSD02 method); score unseen net return, then update state filter with the completed `(Z,U)`. Average **densities** across posterior draws/chains, never state IDs. Use Jacobian `log(sqrt(Vret_t))` to score comparable raw R densities.

**Mandatory comparators on identical origins:**

1. Unchanged pretest-fitted AM-T-HAR-Blend 30m return likelihood.
2. Recent single Student-t return regression.
3. Continuous-volatility **coupling** Student-t with current ex-ante log V, previous completed Z, previous U, and explicit Z×U interaction in conditional mean, with continuous volatility/scale conditioning and ridge. Input scales fit on TRAIN only; no future U leak.
4. HSMM with state-specific scale but shared return mean/AR (directional ablation).
5. **Order-ablated state-conditioned return mixture**: identical fitted state-response distributions, static train-state weights instead of transition-filtered probabilities. This diagnoses whether ordering adds return information; it is not an independently refit mixture.

Score return NLL, sign Brier, paired 1000 UTC-day bootstrap for comparisons. Primary candidate requires HSMM-minus-continuous NLL upper interval <0 and HSMM-minus-VD18 NLL upper interval <0 for BTC/ETH, without severe posterior warnings; directional claim additionally needs improvement in sign Brier and conditional sign-permutation sanity. No fishing among horizons/assets.

## Separate JOINT observation sequence test

Score future bivariate `(Z,U)` joint Student-t prediction using (i) causal HSMM filter, (ii) the same HSMM emissions with **order-ablated, static occupation weights**, and (iii) independently fitted single joint Student-t. The second comparison is a controlled temporal ablation rather than a fully independently refitted static DPM; interpret appropriately. Joint-feature density NLL is in normalized bivariate feature space and must never be compared numerically with raw-return NLL. State-specific coupling correlations and state-conditioned next-block descriptive `(Z,U)` are secondary; require minimum weighted evaluation mass and no economic labels before stability checks. Use TRAIN normals only.

## Regime-free and directional controls

- **Conditional regime-free return generator:** freeze ex-ante historical `Vret` and HAR log-RV scales; generate independent Student-t(6) 5m innovations at the HAR scale, reconstruct actual six-return `R` and `RV`, and fit the same HSMM to bivariate features. This null **preserves the mechanical dependence** between return and RV. At least two null seeds per asset with short chains. It is not a self-consistent evolving volatility+HAR refit and must not be used for formal frequentist p-values.
- **Held-out sign-permutation diagnostic:** freeze predictive state/continuous probabilities generated from actual past observations, randomly flip the signs of held-out block outcomes 100 times. This only tests association of frozen sign probabilities with outcomes, not a refitted alternate HSMM world; never call it a full sign-randomized generative null.
- Report posterior K-cap warnings, chain disagreement, and how frequently the conditional Student-t null occupies multiple states. A count of occupied states by itself NEVER establishes economic regimes.

## Pre-registered decisions and failure interpretation

If HSMM does not improve raw-return NLL over the continuous interaction model and original AM-T-HAR, return **NO_INCREMENTAL_RETURN_STATE**. If joint-observation NLL improves but return NLL does not, it may identify **sequence structure in joint observations only**; this is not a return-predictive regime. If static/order-ablated mixture and HSMM are indistinguishable, no temporal-state effect is established. Any encouraging outcome is **historical exploratory** and requires fresh prospective confirmation, broader posterior chains/prior tests and a self-consistent continuous-volatility null before any options research.

## Artifacts and compatibility

- `src/data_engineering/modeling/hdp_hsmm_price_vol_coupling.py`
- `tests/test_hdp_hsmm_price_vol_coupling.py`
- `notebooks/Return_State_Discovery/03_joint_price_volatility_states.ipynb`
- `discussions/Return_State_Discovery/003_chatgpt_joint_price_volatility_hsmm_preregistration.md`
- `notebooks/Return_State_Discovery/README.md` (append new module; retain prior history)

Expected outputs under `notebooks/Return_State_Discovery/03/`: `03_manifest.json`, `03_posterior_chains.csv`, `03_state_profiles.csv`, `03_future_profiles.csv`, `03_model_scores.csv`, `03_bootstrap.csv`, `03_regime_free_null.csv`, `03_sign_null.csv`, `03_decision.json`. No origin-level raw returns, collector or database mutation, trading claim, or state smoothing at forecast time.
