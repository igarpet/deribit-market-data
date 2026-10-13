# VD 031 — Module 14 preregistration: seven-horizon Markov lift, heavy tails and shock mixtures

**Author:** ChatGPT · **Date:** 2026-10-14 · **Status:** design and code frozen **before VD14 market-data run**. Real results: **not yet available**. Branch: `Volatility_Dynamics`. No use of models or results from other research branches. No changes to collectors or DuckDB.

## Research question

Does an augmented finite-dimensional Markov volatility state improve physical-world **return density, future realized variance and absolute-return tail probability** forecasts beyond a causal two-EMA Gaussian baseline, simple Student-t tails and constant/state-conditioned statistical jump mixtures?

**Important:** The word *jump* in M3, M4 and AM denotes a broad component of a conditional Gaussian mixture; it does **not** identify an actual price discontinuity from 5-minute closes. Markovity is imposed on the **augmented state representation**, not proven for price alone. Nonzero return drift is not fitted: this branch tests conditional risk, not directional alpha.

## Source, sample and time boundaries

- Historical physical data: `DERIBIT_DATA_ROOT/history/price_5m/<ASSET>.csv.gz`: five-minute **Binance spot** closes, not Deribit index/perpetual closes or options quotes; no import of VD12/VD13 predictors, parameters, fitted artifacts or scores. Assets initially AVAX, BTC, ETH, HYPE, SOL, TRX, XRP; exclude assets with <365 available calendar days.
- Historical upper bound **strictly before `2026-09-19T00:00:00Z`**, matching the already-viewed two-year price-history period. 60/20/20 chronological UTC-*day* split. Each forecast target uses `t+1 ... t+k`, with 5-minute grid continuity, no missing-bar bridging and no split-boundary crossing.
- The historical test set was already exposed in VD12/VD13. It is **retrospective exploratory** even though evaluation happens after chronological training. Future prospective confirmation must use **new post-2026-10-14 dates and frozen model specifications**, not reassign earlier test dates as untouched data.
- Causal features after observing the current 5-minute return: per-asset training-only hour-of-day de-clocking of `log(r_t² + eps_train)`, with `eps_train = .001 * median positive training r²`. Fast EMA half-life 30 minutes, slow EMA 240 minutes, `gap=fast-slow` and 15-minute fast-memory slope. Missing bars omitted from EMA observation weight; current return finite and prior 288 elapsed bars must have >=80% finite returns and at least one observation in the previous three bars.
- Uniform model support: all models use precisely the same forecast origins within each horizon. Larger horizons may censor more origins near UTC split boundaries.

## Seven horizons — user-requested, fixed

Exactly **5m, 10m, 15m, 30m, 60m, 120m and 240m**; primary inference horizon **60m**. Models evaluated at every horizon; secondary horizons are descriptive unless a separately adjusted multiplicity plan is frozen.

## Five models — frozen definitions

Let \(R_{t,h}=\sum_{j=1}^{h/5}r_{t+j}\) and \(RV_{t,h}=\sum_{j=1}^{h/5} r^2_{t+j}\). Zero conditional mean for all five models. Model-specific observed total net-return variance \(v_{t,h}\) is calibrated **using TRAIN ONLY**.

- **M1_gaussian** — Gaussian `N(0,v_EMA)` with log future RV ridge forecast using current de-clocked log r², 30m and 240m memory, asset fixed effects and UTC clock harmonics; train-calibrated net-return variance per asset. Ridge penalty 0.02 times training origin count, intercept unpenalized. This baseline intentionally mirrors the two-EMA *idea*, but coefficients are fitted independently and it does not import VD12/VD13 model objects.
- **M2_student_t** — M1 total variance with centered Student-t innovations. Degrees of freedom chosen **using VALID ONLY** from `{4,6,10,20,40}`; t scale is normalized so predicted total variance equals M1's.
- **M3_constant_mix** — variance-matched two-component Gaussian return density (ordinary + broad-tail), constant `p_h` for all origins within horizon, `p_h` fitted using TRAIN likelihood. Ratio of component standard deviations selected from `{3,5,8}` on VALID; no hindsight jump labels.
- **M4_conditional_mix** — same M3 variance normalization and broad ratio, but `p_{t,h}=logit^{-1}(intercept + beta_L slow + beta_F gap + beta_S slope15)` with coefficients estimated on TRAIN by penalized mixture likelihood. Each horizon fitted directly. This is *not* a two-state HMM.
- **AM_markov_lift** — separate augmented Markov state with `L_t=slow`, `F_t=fast-slow`. Fitted diagonal VAR(1) transitions in train with correlated innovation covariance; the two coefficients are constrained between 0.01 and 0.9999. Exact conditional first moment of future `exp(L+F)` is propagated analytically over k five-minute steps with transition-innovation uncertainty, then calibrated on TRAIN to the horizon's net-return variance. Statistical shock mechanism fit **once at 5m on TRAIN**: five-minute mixture width chosen on VALID and a per-asset logistic rare-shock probability conditioned on `(L,F)` fitted by TRAIN likelihood. Conditional future `(L,F)` means produce future one-step probabilities; `1-prod(1-p_j)` approximates probability of >=1 broad shock within the horizon. Broad-component variance increment from a *single* shock is diluted by `1/k`, giving horizon sd ratio `sqrt(1+(ratio_5m²-1)/k)`. The integrated-return density is a variance-matched mixture. **Approximation caveat:** no exact multiple-jump convolution, log-volatility path uncertainty beyond the first integrated-variance moment, leverage correlation, or risk-neutral option dynamics is modeled. A positive result would validate this forecast approximation, not prove a true latent Markov jump diffusion.

## Metrics and inferential procedure

1. **Primary:** out-of-sample mean **negative log-likelihood (NLL)** for signed net log returns, lower better.
2. Future realized variance: **MSE of log RV**. M1–M4 have the same log-RV forecast (by construction); AM uses its propagated-state log-RV forecast with train-estimated level offset.
3. Physical large move: train asset/horizon-specific 95th percentile of `|R_h|`, with **proper tail-event log loss and Brier** from the model-implied same density. Also report observed rate vs average forecast probability.
4. Two-sided predictive interval: proportion of test net returns whose model-implied two-sided survival probability is below 0.05; the nominal miss rate is 5%.
5. Report all seven horizons and all six primary pairwise comparisons: M2−M1, M3−M1, M3−M2, M4−M3, AM−M4, AM−M2. AM−M1 is supplementary. Paired **UTC-day block bootstrap**, 1000 draws, keeps assets and overlapping forecast origins in the same sampled UTC day; 95% percentile CI. Multiple horizons and metrics are **not** independent hypothesis tests.
6. Historical run always returns `HISTORICAL_EXPLORATORY` with >=30 test UTC days (else `INSUFFICIENT_DATA`). It must never print `CONFIRMED` or approve trading.

**Primary 60m scientific decision rules:**
- If M2−M1 has delta<0 and upper 97.5% CI<0, simpler heavy tails help.
- If M3−M2 similarly clears, the two-Gaussian mixture beats Student-t density.
- If M4−M3 clears, the conditional broad-component probability adds information.
- AM is a **strong prospective candidate** only if both AM−M4 and AM−M2 have negative delta and upper CI<0 on primary 60m NLL. In addition, **tail-event log loss and interval calibration must be inspected**, and an AM improvement is not described as solving tail risk if AM's tail calibration is worse.

A result selected because it is unusually good at 5m or 240m is a secondary discovery, requiring new preregistration. Proposing a different horizon or parameter grid after a result is specification search.

## Known threats to interpretation

- Reused and previously examined historical data: no truly independent confirmation.
- Clock/weekday, cross-asset dependence and overlapping origins; block bootstrap addresses only within-day dependence.
- Binance spot price quality: AVAX 2025-H2 flagged for suspect stale closes by Hurst Regimes 03; report each asset and do not attribute anomalous AVAX patterns to economics until source corrected.
- M3/M4/AM mixture components cannot physically identify jumps from 5m data.
- Conditional return variance is separately calibrated from integrated squared-return variance because aggregated returns can contain covariance; this can drift out of sample.
- AR(1) factor transition fitted on filtered EMAs is **not** a structural latent-state estimate, and the observed proxies are not guaranteed to be Markov. Multiple stochastic volatility processes can generate similar results.
- AM propagates conditional means and second moments but uses a moment-matched return-mixture approximation. It is **not** an exact continuous-time lifted Heston model, nor a fitted HSMM.
- No option price, 08:00 expiry P&L, spread, fees or risk-neutral probabilities. No trading claims or collector modifications authorized by these tests.

## Saved artifacts and execution

Code: `src/data_engineering/modeling/markov_lift_distributions.py`; notebook `notebooks/Volatility_Dynamics/14_markov_lift_distributions.ipynb`; synthetic tests `tests/test_markov_lift_distributions.py`. Notebook writes compact `notebooks/Volatility_Dynamics/14/14_scores.csv`, `14_day_bootstrap.csv`, `14_decision.json`, `14_run_manifest.json`. No origin-by-origin prediction files; one horizon at a time is scored and discarded.
