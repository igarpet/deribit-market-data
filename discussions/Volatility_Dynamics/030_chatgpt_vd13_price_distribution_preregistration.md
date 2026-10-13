# VD 030 — Module 13 pre-registration: fast–slow volatility, zero-drift price, and conditional return reversion

**Author:** ChatGPT · **Date:** 2026-10-13 · **Status:** pre-registered before any VD13 real-data run. Branch: `Volatility_Dynamics`; module: `13_two_timescale_price_distribution.ipynb`.

## 1. Why: compare, do not import, Regime Structure

The two immediately preceding repo commits are Claude's RS06 implementation `e72e862` and the real-data run `5ebfd37`. RS06's primary **15-minute taker delta-hedged long straddle** has mean session return on premium **−0.40987**, 95% session bootstrap **[−0.48425, −0.33676]**, with 22 sessions, 1,610 usable straddles: `NO_LONG_EDGE`. Its 15m zero-hedge-cost result is still **−0.19973**. Average hedge cost accounts for **0.2101** of premium. Therefore path-level reversion does **not** imply profitable gamma scalping; there is no authorization to reopen options trading.

RS05 found a 5m net/path variance ratio of 0.746 relative to a sign-randomized null near 0.996, and larger big-day giveback. RS06 independently measured Binance *perpetual* price variance ratios by resampling the two-year history. Test values at 60m: BTC 0.862, ETH 0.929, SOL 0.849, XRP 0.969; at 5m: BTC 0.857, ETH 0.934, SOL 0.781, XRP 0.933. Some price reversion persists under coarser sampling, but execution costs and microstructure still matter. Price is therefore not assumed to be an *exact* independent-increment random walk. Its **zero-drift forecast is a falsifiable null**.

The existing VD12 two-year study (discussion 027) found the 30m+4h two-EMA volatility baseline significantly beats climatology and fixed lags at predicting q99 *variance-onset* events; restricted rough-kernel features have no demonstrated incremental value. **VD13 does not repeat q99**. Instead it asks whether the return distribution and future realized variance are predictable, and whether conditional *signed* return mean reversion adds beyond a zero-drift price model.

Per `notebooks/README.md` branch-independence policy: the above serves as **motivation and external comparison only**. VD13 imports no model, learned parameter, result, regime label, threshold, or fitted feature from Regime Structure, Hurst Regimes, VD12, or VRP. Everything is fit afresh from shared raw price observations.

## 2. Data, chronology, guardrails

- **Raw source:** `DERIBIT_DATA_ROOT/history/price_5m/<ASSET>.csv.gz` from the project's shared historical price layer (Binance **spot** 5m closes, **not** the RS06 Binance USD-M perpetual series). `state_time` 5m UTC grid. Native log returns from **consecutive observed** closes. No interpolation of returns, no lookahead, no database writes.
- **Retrospective sample:** strictly before `2026-09-19T00:00:00Z` (as in VD12); assets with fewer than 365 days excluded. Starting candidates: AVAX, BTC, ETH, HYPE, SOL, TRX, XRP.
- **Important evidence-policy caveat:** The two-year history has already been explored by other modules; the chronological 60/20/20 split is a *retrospective out-of-sample diagnostic*, **not a truly untouched confirmation period**. A future prospective test begins no earlier than 2026-10-14 and requires an independently frozen plan.
- **UTC-day split:** 60% train / 20% validation / 20% historical audit/test. No future window crosses split boundary. Only train used for any parameter estimation, including asset-hour de-clocking, tiny variance epsilon, standardized features, regression coefficients, training q95 return thresholds and Gaussian variance calibration. Validation set is left untouched in v1.0 (fixed ridge constants; no parameter search).
- **Availability:** 288 elapsed *previous* 5m bars, >=80% finite return observations, >=one finite return in previous three bars (15m), and current bar finite. Missing bars contribute zero observation weight in EMA, never a fake zero return. Additional signed-return predictors require the *last six* bars to be finite (fair common complete-case support across models).
- **Future targets** for h=15,60,240 minutes: net log return `R_h=sum_{k=1}^{h/5}r_{t+k}`, integrated path variance `RV_h=sum r_{t+k}²`, and `|R_h|>=q95_train(|R_h|)` per asset and horizon. Fully finite future bars required; otherwise target censored.

## 3. Hypotheses and models (frozen)

At origin t the five-minute return at t is known. Fast/slow exponential memories of *train-hour-declocked log single-bar variance* use 30m and 240m half-lives and observed-weight normalization.

**K — Random-walk null:** zero drift, asset-specific train-only constant Gaussian net-return variance. **E — Two EMAs:** zero drift, Gaussian variance predicted by ridge regression on current log variance plus fast/slow memories and asset/time controls. **P — Volatility phase:** E plus fast-memory 15m slope, absolute fast/slow gap and nonlinear gap×slope. Since a bare fast−slow gap is collinear with the two existing EMAs, it is **not** added as a meaningless independent regressor. **M — Price reversion:** P variance plus train-fitted conditional price-return mean using recent r5 and previous cumulative 30m signed return. **PM — Conditional reversion:** M plus signed-return×volatility-gap/slope interactions.

Models are nested in their individual price-mean or variance dimensions. Models use fixed ridge regularization `alpha_var=0.02` and `alpha_mean=0.10` (penalty proportional to training sample size); no hyperparameter searches. Predictive integrated RV and Gaussian **net-price** variance are distinct: the Gaussian scale is adjusted *using train-only net-return squared-variance moments*, accommodating aggregate negative autocovariance without using future information. This is an empirical approximation, not a latent fBM/fOU/HSMM model, and normality can be false for crypto tails.

Primary **60m**. Secondary **15m and 240m**. Model-specific scores on identical complete-case origin indices:

1. Gaussian net-return **negative log-likelihood** (NLL), lower better.
2. **MSE of log future realized variance** (predicted using K climatology / E or P log-RV regression), lower better.
3. Proper log loss and Brier for `P(|R_h|>=training-only 95th percentile)` using **the same model-implied centered or noncentered Gaussian distribution**. Lower better.
4. Proper log loss for up/down sign `P(R_h>0)` from the same Gaussian model. Compare conditional means to the null's 0.5 probability. Lower better.

**Key pairwise tests** (all lower-is-better metrics): E−K; P−E; M−P; PM−M. Paired block bootstrap resamples whole UTC days and retains all cross-asset synchrony (B=1000; seed fixed from horizon/pair/metric). Difference and 95% bootstrap interval always reported. Training thresholds recalculated per h. Point estimates across overlapping 5m forecast origins **are not independent events**; day blocks address part of the correlation, not arbitrary long persistence.

## 4. Ex-ante interpretive flags, no trading gates

At 60m, only if both paired mean difference < 0 and 97.5% day-bootstrap bound < 0:
- **E_distribution_vs_K:** memory-based Gaussian price density improves on constant-volatility random walk.
- **P_phase_vs_E_rv:** volatility phase adds to future log-RV prediction.
- **P_phase_vs_E_nll / P_phase_vs_E_tail:** phase adds to price density / large-move probability.
- **M_price_vs_P_nll:** recent signed returns improve price density, conditional on P-scale.
- **PM_interaction_vs_M_nll:** volatility×signed-return interactions improve further.

Status is `INSUFFICIENT_DATA` below 30 test UTC days or 40 unique tail-positive origin timestamps, otherwise **`HISTORICAL_EXPLORATORY` regardless of score**, due prior data exposure. Negative results are useful; historical flags alone cannot authorize a signal or options hedge. No statistical conclusion about the structural Hurst exponent is permitted.

Potential issues: microstructure attenuation in 5m spot closes; non-Gaussian return tails; time-varying cross-asset dependence; distribution shift; the normal approximation may be poorly calibrated. Compare instruments/feeds explicitly before attributing RS06/VD13 differences to economics.

## 5. Retention and reproducibility

Implementation: `src/data_engineering/modeling/volatility_price_distribution.py`.
Notebook: `notebooks/Volatility_Dynamics/13_two_timescale_price_distribution.ipynb`.
Tests: `tests/test_volatility_price_distribution.py`.
Outputs only under `notebooks/Volatility_Dynamics/13/`: `13_scores.csv`, `13_paired_day_bootstrap.csv`, `13_decision.json`, `13_run_manifest.json`. No millions-of-origins raw predictions, high-resolution charts, or data lake mutations. Jupyter figures inline only. The notebook is committed without real-data outputs. Synthetic regression tests cover missing-data causality, split censoring, train-only thresholds, reproducible aligned scoring, and robust day-bootstrap mechanics.
