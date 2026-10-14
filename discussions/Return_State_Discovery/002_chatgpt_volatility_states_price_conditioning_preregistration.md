# RSD 002 — Discover volatility states, then test subsequent price behavior

**Date:** 2026-10-14. **Author:** ChatGPT. **Status:** specified and synthetic-tested *before* the first real-data RSD02 run. Historical exploratory, not a pristine prospective confirmation and not a trading system.

## Hypothesis and rationale

RSD01's price-state HDP-HSMM occupied five or six states in BTC/ETH, but did not beat AM-T-HAR-Blend on subsequent 30-minute return likelihood; even regime-free Student-t price simulations occupied multiple states. The original HSMM 07–08 program failed to demonstrate discrete volatility regimes beyond HAR. A new, deliberately different hypothesis is: **a separately discovered volatility state might condition signed-return behavior** (drift, short-lag momentum versus reversal, scale or tail risk) *without* needing to improve volatility forecasting itself. The hypothesis can fail. The HDP is not a guarantee of real discrete regimes.

## Data, baselines and time

- Shared Binance 5m spot-close history, exact timestamp grid and checks from VD14–18, no added tick data, no new collectors, no DuckDB writes. Assets **BTC and ETH primary**; SOL/XRP reserved for a clearly secondary replication after primary results. AVAX/TRX excluded initially because of repeated-price/data-quality problems; HYPE lacks history.
- Refit VD18 **30-minute AM-T-HAR-Blend** using only its original pre-TEST TRAIN and VALID subsets, via existing RSD01's `build_historical_30m_blocks`. Do not import saved Module 18 fit objects or tune 30m HAR after looking at RSD02 results. Baseline forecast at block START uses only observations ending at/before start.
- Complete 30-minute returns comprise precisely six subsequent 5-minute bars, without missing bars; reconstruct `RV30=sum r5²` at the block **END**, assert net return is identical to 30m target. State emissions containing this RV are available **only after** those bars finish.
- Use VD18's already previously inspected 150-day historical TEST, split by full chronological UTC days into first 60% (~90d) for volatility-HSMM and return-response fitting, middle 20% (~30d) for descriptive diagnostics, final 20% (~30d) for return-density scoring. No future target crosses original VD18 TRAIN/VALID/TEST boundary. Repeat exploration on this previously inspected history is not independent evidence.
- Require 8 completed contiguous 30m blocks for the fast/slow volatility ratio; discard insufficient history and reset at missing-block boundaries. The small positive realized-variance floor is derived ONLY from the RSD02 training period then frozen. All robust feature centering/scaling fit only on RSD02 train.

## Anonymous VOLATILITY-ONLY observations

At the END of 30m block `t` define three scalar observations:

1. `log_predictive_variance = log(V_AM_T_HARBlend_30m(start_t))`: forecast before block.
2. `log_RV_surprise = log(max(RV30_t, eps_train) / V_AM_T_HARBlend_30m(start_t))`: observed variance surprise, revealed at end.
3. `log_RV_fast_slow_gap = log(max(RV30_t,eps_train) / max(mean(last eight completed contiguous RV30 blocks),eps_train))`.

**No signed return, signed path efficiency, direction, signed autocorrelation, future return or option quote enters the HSMM.** The state may reflect volatility LEVEL, TRANSITION or forecast miscalibration; names are assigned only after fitting. The first feature has deterministic time-of-day/asset influences, so characterize state-clock associations rather than treating them as economic regimes.

## Nonparametric HSMM inference

Reuse the same tested `hdp_hsmm_return_states.py` finite weak-limit hierarchical DP explicit-duration sampler, per asset. `Kcap=12` computational limit, not a market-state selection grid; report full occupied-K histogram over posterior draws, chain disagreement, cap mass and duration-cap diagnostics. Student-t(6) multivariate emissions (three volatility observations) with state-specific means and covariances; no self transitions at episode boundaries; shifted negative-binomial durations, capped *age representation* 96 blocks (48h) without compulsory switching. Two distinct chains, 140 iterations, burn 80, thinning 4. This is a **short exploratory run**, insufficient to establish convergence or a genuine state count. Explicit duration estimates remain sensitive to boundaries and model mismatch.

## Causal return-response test — the primary gate

For every block `t`, compute filtered **prior** state weights `w_(t,s)=P(S_t=s | volatility observations from blocks <t)`. Score block `t`'s return **before updating with its own observed RV**. This is the central no-leakage invariant. No retrospective smoothed state probability is ever used to forecast that same block. Use the preceding COMPLETED block's standardized net return as optional return-AR covariate, and current ex-ante 30m variance as Jacobian scaling.

Train state-specific Student-t(6) return `r30/sqrt(V_pred)` means, signed-return AR(1) slopes and residual scales by probability-weighted, ridge/pseudo-count-shrunk fit on RSD02 training blocks. State weights and parameters remain frozen during evaluation. For robustness, a **scale-only** state model uses the same state-specific scales but pooled signed-return intercept/AR, isolating any incremental directional contribution. State identities **cannot be averaged across chains or different posterior draws**. Fit one conditional-return response model per representative posterior draw and average *forecast densities* across six draw-specific models (three per chain). State-count summaries use all retained draws.

Mandatory common-origin comparators:
- Unchanged frozen VD18 AM-T-HAR-Blend 30m return-density NLL.
- Recently trained Student-t(6) with a common signed-return AR coefficient (no hidden states).
- Ridge-regularized continuous-volatility Student-t(6), with return mean depending on the last completed block's signed return, ex-ante log variance, previous vol surprise and previous fast/slow gap with signed-return interactions; **variance scale** also conditioned continuously on causal volatility covariates. Exact same return origins and Jacobian as HSMM.
- State scale-only variant, which removes state-dependent drift/momentum but retains state-dependent variance.

Primary statistic: 30m signed-return mean NLL, HDP-volatility-state minus the **continuous volatility Student-t** on evaluation blocks (lower is better). Also report the comparison to unchanged AM-T-HAR-Blend. Bootstrap by complete UTC day (1,000 draws), report asset-specific scores, CIs, sign logloss/Brier, and direction/reference coin-flip Brier. Report **posterior-weighted observed next-block** mean, sign probability, same-sign continuation, absolute path efficiency and within-block return dependence by the internally labeled state for each chain separately; these are descriptive outcomes on evaluation, not forecasts using future state labels. State-dependent price behavior should mean *forecasting incremental price dynamics*, not retrospective classification. Do not pool 5m overlapping horizons to inflate independent precision.

## Negative controls and interpretability

- **Conditional no-regime simulation:** retain each observed training origin's ex-ante variance time series, simulate six independent standardized Student-t(6) 5m innovations per 30m block; recompute observed RV and volatility emissions from simulated returns; run the same HSMM priors with two seeds. This is NOT a self-consistent HAR-refitted null, and is only a diagnostic of how many states a continuous variance path and Student-t innovations manufacture. Report posterior K, duration, convergence diagnostics. Do not claim a frequentist p-value from two simulations.
- **Block-sign randomization:** independently flip the sign of the whole **non-overlapping return block** in train/evaluation, keeping RV and volatility state observations unchanged. Refit only return-response models and continuous comparator; calculate the *state versus continuous* direction Brier advantage across 40 randomizations. Tests directional information, not volatility state existence. A sign-flip null retaining within-block shape is still imperfect; report the limitation.
- Descriptive profiles may use representative posterior labels, but state features here are only volatility summaries. Rare-state counts and chain label permutations are barriers to economic naming. Look for posterior states that have genuinely different subsequent signed-return AR slopes or directional calibration rather than merely different variance.
- Evaluate synthetic no-regime controls and no-lookahead unit tests before analyzing real outputs. BTC/ETH remain primary; SOL/XRP can be added only as labeled replication; no option signal or prospective confirmation is authorized.

## Decision / outputs

The experiment's primary claim requires a negative HSMM–continuous NLL difference, an upper session-day bootstrap bound below zero, reasonable directional controls and sensitivity to sign-flips/no-regime simulations; additionally report HSMM–AM-T-HAR unadjusted. If not, report **NO_INCREMENTAL_PRICE_STATE** without dressing up occupied-state count as economic regimes. Primary test is exploratory because chronology and hyperparameters were motivated by prior research.

Artifacts: `src/data_engineering/modeling/hdp_hsmm_volatility_price.py`; `tests/test_hdp_hsmm_volatility_price.py`; `notebooks/Return_State_Discovery/02_volatility_states_conditional_price.ipynb`; this numbered discussion; update `notebooks/Return_State_Discovery/README.md`. Notebook writes only compact aggregate CSV/JSON to `notebooks/Return_State_Discovery/02/`, never posterior paths, raw five-minute returns, prices or simulated tick-level data. Prior RSD01 and VD14–18 code is not modified.
