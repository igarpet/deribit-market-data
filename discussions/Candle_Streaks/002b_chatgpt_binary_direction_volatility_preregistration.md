# CS02b — Does past volatility improve UP/DOWN next-candle probabilities after excluding zeros?

**Date:** 2026-10-14. **Status:** historical exploratory; specified before running Binance historical CS02b. **Relation:** binary sensitivity study of CS02, not a change to its preregistration or output. Reuses frozen model families, volatility inputs, bins and chronology; removes future zero-return targets from the evaluation and refits a binary hierarchy. Historical market dates have already been inspected by CS01/CS02 and do **not** qualify as pristine prospective data.

## Specific falsifiable claim

CS02 improved next five-minute **DOWN/ZERO/UP** three-class log loss, but could have profited predominantly from forecasting unchanged prices, especially for SOL/XRP. CS02b isolates **direction conditional on an immediately following nonzero return** and tests whether currently observable volatility adds next-UP versus next-DOWN information after conditioning on current sign and/or exact ongoing signed streak length.

The target is **P(R[t+1] > 0 | R[t+1] != 0, F[t])**. The conditioning event `R[t+1] != 0` is not known at the prediction time. Therefore results are about directional discrimination among future nonzero moves only, **not** unconditional `P(next UP)` and not a live trading signal. The event means the **immediately next five-minute bar** is nonzero, *not* a jump to whichever later bar happens to move. No selection by direction or amplitude besides excluding exact zeros.

## Data and causal timeline

- Exactly the CS02 data source and assets: BTC/ETH primary, SOL/XRP secondary, two-year Binance five-minute close-to-close spot returns via the existing VD14 loader and cutoff 2026-10-14 00:00 UTC. Track the actual final timestamp separately. No tick data, new collectors, database writes, AM-T-HAR-Blend modification, or prior notebook output reuse as model fits.
- Calculate signed ongoing streaks with CS01's `build_events_and_runs` before target filtering. **Zeros always break directional streaks**; gaps and invalid bar returns also terminate/censor stretches. An event with current zero is not included because sign and streak length are undefined there. An event with current nonzero but immediately next zero is **excluded from binary fit/scoring**; it must NOT be stitched to a later nonzero event. Leave its observed time in the raw sequence.
- Use CS02's *past-only* fast and slow squared return means: 12 current/past five-minute bars for fast (1h) and 288 current/past bars for slow (24h), strict contiguous and finite. Features at the end of current bar t: `log RVslow` and `log(RVfast/RVslow)`; no realized return/variance from t+1 appears on the predictor side.
- Require complete UTC days with 288 valid five-minute returns from 00:00–23:55, as in CS02; exclude the partial historical first/last days. Split chronologically into 60% FIT, 20% VALID, 20% TEST; no label crosses a split boundary. Periods remain aligned with CS02. Do not use TEST or VALID to fit model hyperparameters, bin cutoffs or probability tables.
- **Bin compatibility:** Fit the `low/mid/high` and `compression/neutral/expansion` volatility tertiles exactly as CS02 did on **TRAIN's three-class eligible events before removing next zeros**, then freeze them and remove next-zero targets. This avoids post-hoc rebucketing when isolating the direction-only signal. Report zero exclusions by period, including later test.

## Fixed binary forecasting hierarchy

Training counts are `DOWN`/`UP` among **TRAIN next-nonzero observations only**, with prior smoothing weights inherited unchanged from CS02. All predictions `P(up|nonzero)` are strictly in `(0,1)` and `P(down|nonzero) = 1 - P(up|nonzero)`; preserve every observed exact streak age `k`, no K ceiling or 10-bar window. For missing/sparse cells, shrink toward registered parent distributions.

1. `random_50`: fixed 50/50 binary benchmark.
2. `iid_binary`: smoothed TRAIN up probability (Jeffreys 0.5 count on each class).
3. `sign_binary`: current sign only; pseudo-observation strength 20 toward unconditional probability.
4. `streak_binary`: `(sign,k)`; pseudo-observation strength 80 toward sign model.
5. `sign_vol_binary`: `(sign,vol_level,vol_phase)`; pseudo-observation strength 100 toward sign model.
6. `additive_binary`: arithmetic 50/50 mean of streak-only and sign+vol predictions.
7. `streak_vol_matrix_binary`: `(sign,k,vol_level,vol_phase)`; pseudo-observation strength 125 toward the additive model. Unseen cells use additive model unchanged.

**Controlled cross-version comparison:** On exactly the same later binary targets, separately refit CS02's original three-class models on all original TRAIN three-class outcomes (including zero), then compute `P(up|nonzero) = P3(up)/(P3(up)+P3(down))` for each model. These `cs02_normalized_*` predictions isolate whether re-estimating after zero exclusion changes performance, as opposed to just changing the evaluation metric. The original CS02 outputs are not imported as fitted artifacts.

## Primary predictive tests and precommitted decision

- Main proper scores: binary log loss and binary Brier, computed per **immediately next nonzero** label; all comparisons use matched origins and TRAIN-frozen models. Report empirical TEST up rate and forecast up probability mean.
- BTC and ETH primary: **volatility contributes binary directional information** only if `sign_vol_binary` beats both `sign_binary` and `iid_binary` on TEST log loss and the paired full-day 1000-bootstrap 95% **upper CI** is negative for each comparison in *both* assets.
- Separate stronger full-matrix hypothesis: `streak_vol_matrix_binary` must beat each of `streak_binary`, `sign_vol_binary`, `additive_binary` with negative 95% upper bootstrap bounds on BTC and ETH. SOL/XRP are only replications and cannot rescue a BTC/ETH failure.
- Report `streak_binary` versus `sign_binary` to test whether streak length itself adds signal; compare renormalized CS02 predictions with binary refit versions but **do not retune** weights or pick winners based on TEST.
- Paired whole-UTC-day bootstrap, 1000 repetitions, seed 2603; daily equal-weight mean and observed-event-weighted mean both reported, along with number of days and min/max event counts. No incomplete-day equal-weight problem from CS01.
- Secondary volatility-label randomization, 100 repetitions, within current sign × display-only coarse age groups; keep outcomes and trained model fixed. It is an alignment sanity check, *not* a model-based generated market null or trading p-value.

## Reporting and decision constraints

Save compact tables in `notebooks/Candle_Streaks/02b/`: binary TRAIN-fitted probability matrix with sample counts and exact unbounded streak ages; separate all-history descriptive matrix; monthly stability; held-out VALID/TEST binary scores; day-bootstrap comparisons; volatility-shuffle diagnostic; quality/period eligibility including omitted zero targets; train bin cutoffs; manifest and decision JSON.

If binary volatility forecasts are favorable, a future integration program must generate ex-ante **AM-T-HAR-Blend** return variance at the *matching five-minute horizon*, estimate `P0=P(next zero|F[t])` separately, combine `P(up)=(1-P0)P(up|nonzero)`, evaluate proper **unconditional** three-class log loss and full Student-t/mixed return-density NLL, directional calibration, tails/variance, transaction costs, and fresh truly prospective evidence. The AM component's return-density normalization and shape must be preserved. CS02b alone cannot approve this integration or justify saying that volatility adds useful unconditional direction.
