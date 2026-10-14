# VD 033 — Module 16: exploratory horizon calibration of AM-T return densities

**Author:** ChatGPT · **Date:** 2026-10-14 · **Status:** protocol frozen before Module 16 execution on actual market data; development is **exploratory specification search**, not fresh prospective confirmation.

## Motivation and exact clarification

VD15's AM-T does **not** hold five-minute variance constant at all horizons. It fits a native five-minute Student-t **return** transition scale and propagates 5m innovations and fast/slow Gaussian state shocks over `k=h/5` steps. Its integrated return variance therefore varies by origin and horizon, but the resulting distribution is *not independently recalibrated* for each horizon. A direct fitted terminal-horizon M2 Student-t forecast outperformed the native path-integrated AM-T from 10m to 240m, while AM-T beat AM-G at all horizons. VD16 tests whether that gap arises from inadequate aggregated **variance scale**, inadequate **shape/tail thickness**, or both. No new returns, macro, options, exchange datasets or branch outputs are introduced.

The user explicitly authorized exploratory horizon calibration and parameter search. This is not ex-ante proof. *Ex-post reuse of historically inspected test data creates data-snooping/selection bias; fitting on a genuinely later target is look-ahead leakage and is forbidden.* Market practitioners adapt models using information available at the time of their updates, and evaluate subsequent performance. VD16 therefore uses chrono-purged VALID subperiods for calibration and model selection. Its TEST window is historically revisited and cannot support formal confirmation, even though software never accesses TEST outcomes in parameter fitting.

## Data, states, models, exact horizons

- Start with same standalone Binance spot `<ASSET>.csv.gz` 5m history as VD15 (AVAX, BTC, ETH, SOL, TRX, XRP; HYPE excluded if too short); strict UTC history end `< 2026-09-19`. Preserve VD14's 60/20/20 UTC-day chronological split, causal hour de-clocking, original 30m/240m EMA state, missing-bar censoring and horizon-specific training-only q95 thresholds. No tick collection, trades, options, Deribit IV, or database writes.
- All seven forecasts: **5, 10, 15, 30, 60, 120, 240 minutes**. Primary 60m, secondary descriptive. All model comparisons are on identical TEST origin-time-asset sets within each horizon.
- Refit VD15 `AM_T` native 5m per-asset t `nu` from 5m VALID data, original path propagation and AM-G/M2 controls; no previous module fitted artifacts used as input. All `AM_T*` share exactly the same **unchanged native five-minute Student-t kernel** and same stochastic volatility Markov state process. `M2_student_t` is the original per-horizon refit control.
- Models: `AM_T`, `AM_T_SCALE`, `AM_T_SHAPE`, `AM_T_BOTH`, `AM_G`, `M2_student_t`. The three calibration variants alter **horizon-conditional return-density outputs**, not variance-state innovations or return transition parameters. They may destroy cross-horizon Chapman–Kolmogorov / semigroup consistency; do not call calibrated output a newly estimated globally coherent Markov process.

## Proper variance-versus-shape separation

VD15 represents the k-step predictive return density as a normalized mixture of `N(0,v_{t,h,i})` over deterministic QMC integration particles `i=1,...,N` (AM-T has `phi=0`). Define sample mean `m_{t,h}=mean_i v_{t,h,i}`. For horizon `h` and common *pooled-across-assets* calibration parameters `s_h>0` and `a_h>0`, transform each mixture component variance:

\[
 v'_{t,h,i}(s_h,a_h)=s_h m_{t,h}
   \frac{v_{t,h,i}^{a_h}}{N^{-1}\sum_j v_{t,h,j}^{a_h}}.
\]

Therefore `mean_i v'_i=s_h m`. `s` changes mean forecast variance; `a` adjusts *dispersion* of variance mixing, controlling predictive central peak and tails without mechanically changing that mean. The Student-t return process is left intact at native frequency; this is a forecast distribution recalibration overlay, not a horizon-dependent t innovation with a new `nu`.

- `AM_T`: `s=1`, `a=1` (exact original VD15 density with matching QMC nodes).
- `AM_T_SCALE`: `s` fitted, `a=1` fixed.
- `AM_T_SHAPE`: `s=1` fixed, `a` fitted.
- `AM_T_BOTH`: both `s` and `a` fitted.

Fit each horizon's **common pooled** `(s,a)` using only first 60% of eligible VALID calendar days; **purge last `h` minutes** before the chronological select boundary, so even the calibration target cannot overlap the selection subperiod. Retain an even-spaced time sample of at most 512 origin observations **per asset per subperiod**, independent of outcomes. Use 256 fixed-seed QMC particles. Fit by penalized density NLL with gentle anchor penalty `0.002*((log(s)/0.35)^2+((a-1)/0.5)^2)`. Bound `log(s)` to `[-0.70,+0.70]` (variance factor 0.497–2.014) and shape `a` to `[0.45,2]`; no test-based grid widening. Evaluate all fitted candidates on late VALID. Late-VALID NLL chooses a family for descriptive *prospective deployment*, but TEST reports every variant and does not choose an ex-post winner. Report fitted parameters, early calibration NLL, late selection NLL, and exact time boundary.

This arrangement investigates horizon-dependent calibration without letting TEST labels drive that calibration. Nonetheless, previously examined historical TEST outcomes mean this is *scientific exploration*, not confirmatory inference. Selecting a future version based on these TEST rankings requires genuinely new, untouched dates.

## Scoring and uncertainty

- Primary: held-out physical signed-return negative log-likelihood (NLL); lower better. Compare `AM_T_SCALE`, `AM_T_SHAPE`, `AM_T_BOTH` to original `AM_T`, and `AM_T_BOTH` to direct per-horizon `M2`. Additional paired comparisons identify whether both calibration knobs are necessary.
- Also return proper same-density tail-event log loss/Brier (TRAIN per-asset q95 of `|R_h|`), predicted vs observed large-move fraction, nominal 95% central-interval miss fraction, and per-asset score decomposition. The tail score uses the *same* transformed mixture densities, not a separately fitted event classifier.
- Pair origins one-to-one at each horizon; whole-UTC-day bootstrap resampling across assets, 1,000 replicates. Six other horizons are descriptive; families and horizons are a multiple-comparison search. All historical p-values/CIs are exploratory and not family-wise adjusted.
- Numerical stability: 256 vs nested 512 Sobol nodes on an evenly spaced *late-VALID-only* subsample for horizons 5, 60, 240; do not tune models using TEST-based quadrature differences. The shape warp relies on sampled variance-mixture tails, so numerical sensitivity is essential.
- Asset-specific price-quality caveat: AVAX prior suspicious prints; TRX peculiar tail behavior. Report BTC, ETH, SOL, XRP separately in per-asset CSV.

## Decision and next step

Never print `CONFIRMED`. Save `16_scores.csv`, `16_bootstrap.csv`, `16_per_asset.csv`, `16_calibration_grid.csv`, `16_particle_sensitivity.csv`, `16_decision.json`, `16_manifest.json` under `notebooks/Volatility_Dynamics/16/`. **No origin-level predictions or raw market data** are written.

The practical question: do modest horizon-by-horizon density corrections remove AM-T's gap to M2, or does M2's direct per-horizon feature mapping still dominate? Even if calibrated AM-T wins in the historical period, do not infer tradable options alpha or a proven five-minute generative Markov law.
