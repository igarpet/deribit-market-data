# VD 035 — Module 18: HAR variance versus Markov variance for native Student-t returns

**Author:** ChatGPT · **Date:** 2026-10-14 · **Status:** specified and synthetic-tested BEFORE any market-data Module 18 run. **Branch:** Volatility Dynamics. **Historical exploratory specification search, not independent prospective validation.**

## Research question

Can a separately TRAIN-fitted HAR predictor of future net-return variance improve the VD15 native five-minute Student-t RETURN Markov model (with fixed VD16 tail-shape calibration) beyond VD17's causal continuous-state variance correction and beyond VD14's horizon-specific two-EMA Student-t benchmark M2? Is a HAR forecast valuable **independently** of borrowing M2's direct variance mapping?

## Scope and retained definitions

- Data: the **same** `DERIBIT_DATA_ROOT/history/price_5m/<ASSET>.csv.gz` files with `state_time`, `close`, Binance spot five-minute returns. Assets initially AVAX, BTC, ETH, HYPE, SOL, TRX, XRP. Apply VD14's >=365-calendar-day history and missing-bar requirements. Strict cutoff **before 2026-09-19 00:00 UTC**. No Deribit tick data, new API calls, database modifications, trading claims, or leakage from a separate research branch.
- Forecast origins are after `r_t` observed; all predictors use timestamps <=t, targets use `(t,t+h]`, no bridge over missing bars and no target crossing TRAIN, VALID or TEST day splits. Same 60/20/20 UTC-day chronological split as Modules 14–17. Strict complete-history HAR windows are additionally required **for all models**; benchmark scores are refitted on this common-origin support.
- Seven fixed horizons: **5, 10, 15, 30, 60, 120, 240 minutes**, primary **60m**. Historical TEST (roughly 150 days) has been inspected repeatedly in VD13–17, so every score and day-bootstrap interval is *exploratory*. New prospective periods with frozen algorithms are required to confirm anything.
- Underlying five-minute Student-t RETURN transition, fitted native 5m degrees of freedom and fast/slow Gaussian Markov state process remain unchanged from VD15. VD16's per-horizon scale+shape parameters are refitted and inherited unchanged. Forecast variance overlays can destroy multi-horizon path consistency; do NOT call them a new global Markov transition law.

## HAR forecast — causal and TRAIN-only

Use past completed five-minute returns to form realized variance sums at **30m, 60m, 240m, 1440m** lookbacks. Include the current five-minute return and no future returns. Compute strict complete windows of squared returns; **any missing bar invalidates its overlapping HAR feature**. Features are `log(max(past_RV,1e-16))`, with train/valid/test standardized using **TRAIN only**. No regular-grid zero imputation of missing realized volatility.

Fit two candidate models separately at every horizon:

- **HAR_2:** past 30-minute and 240-minute log realized variance, plus UTC clock sin/cos and asset fixed effects. This is the minimal memory set supported by VD12.
- **HAR_4:** past 30m, 60m, 240m and 1440m log realized variance, plus the same controls. This tests richer but correlated memory, penalized by train-only ridge coefficient 0.02, rather than assuming more lookbacks must help.

Predict the future path's **realized variance** `RV_{t,h}=sum_{j=1}^k r_{t+j}²` by training ridge on `log(max(RV_{t,h}, eps_{asset,h,train}))`, with the floor set to 0.001 times median positive training RV (minimum 1e-16). Then TRAIN-only calibrate to the net terminal return variance `E[R_{t,h}²]` with per-asset capped (99.5%) return-square mean divided by mean exp(HAR log RV prediction), clipping calibration ratio to `[0.01,100]`. This explicitly separates future integrated squared-return variance from terminal net-return variance.

### HAR selection and density transforms

The original AM-T QMC distribution represents origin-/horizon-dependent Gaussian component variances `v_i`. VD16's fitted `AM_T_BOTH` shape mixture `vtilde_i` is preserved across HAR models. Define `m=mean_i(vtilde_i)` and `z=log(v_HAR/m)`.

- **AM_T_HAR_DIRECT:** `v_i^new=vtilde_i*exp(z)` with *unclipped z*. Its mean variance equals the HAR net-return variance **exactly**; its conditional dispersion/tail shape remains VD16's normalized particle-mixture shape. This is the cleanest independent HAR-level test.
- **AM_T_HAR_BLEND:** `v_i^new=vtilde_i*exp(clip(w_h*clip(z,-1.5,1.5),-.85,.85))`; fit one pooled horizon-specific `w∈[0,1.25]` on EARLY VALID by penalized return NLL, penalty `.003*w²`, matching VD17.
- **AM_T_HAR_STATE:** `v_i^new=vtilde_i*exp(clip(X_t*beta+w_h*clip(z,-1.5,1.5),-.85,.85))`, where `X_t` is exactly VD17's causal continuous-state design `(slow,gap,now,slope15,clock,asset controls)`, with same `.025 ||beta||² + .003 w²` penalties and coefficient bounds, fitted on early VALID.

**Selection:** HAR_2 and HAR_4 candidate blend weights are separately fitted on the first 60% of VALID days (max 512 equally spaced eligible origins per asset), purging forecast targets near the time boundary. Compare their blend NLL on remaining 40% of VALID (same max per asset); **only that later VALID** selects which HAR engine will feed HAR_DIRECT, HAR_BLEND and HAR_STATE on the historical TEST. HAR_STATE joint coefficients fit on EARLY VALID *after* the HAR engine family is chosen using LATE VALID. The resulting late-VALID HAR_STATE score is not independent because this validation part helped pick the HAR engine. Full five-minute return df selection already used complete VALID in VD15. All scores are exploratory.

## Seven held-out comparison families

| Code | Conditional variance source |
|---|---|
| `AM_T_BOTH` | Refit exact VD16 global horizon calibration; constant variance scale, unchanged shape |
| `AM_T_VAR_STATE` | Refit VD17 continuous-state correction, no M2 anchor |
| `AM_T_VAR_ANCHOR` | Refit VD17 M2 variance anchor, diagnostic **not independent of M2** |
| `AM_T_HAR_DIRECT` | Independent HAR forecast variance; native student-t return mixture shape unchanged |
| `AM_T_HAR_BLEND` | HAR forecast blended with AM-T variance using early VALID NLL |
| `AM_T_HAR_STATE` | HAR anchor plus VD17 continuous-state correction |
| `M2_student_t` | Refit VD14 original horizon-specific Student-t benchmark |

Refit original implementations, don't train on previously saved fitted artifacts. Every model scored on **exactly identical eligible return origins**, with non-leaky causal past HAR windows.

## Evaluation, diagnostics and decision criteria

- Main proper score: mean TEST signed-return NLL (lower better), with **paired whole-UTC-day block bootstrap** (1,000 draws). Primary comparison at 60m: HAR_DIRECT/HAR_BLEND/HAR_STATE each vs AM_T_BOTH, HAR_BLEND vs VD17 STATE, HAR_BLEND vs VD17 M2-ANCHOR, and HAR_BLEND/HAR_STATE vs M2. Secondary horizons descriptive; many families create selection bias even without software leakage.
- Additional proper scores: terminal-return variance QLIKE `log(v)+R²/v`, tail event log loss, tail Brier, predicted versus observed TRAIN-defined 95th-percentile absolute-return events, and 95% central predictive-interval miss rate. Per-asset score breakdown mandatory, especially BTC, ETH, SOL, XRP and the earlier AVAX data-quality warning.
- HAR diagnostic files: fit coefficients/scale calibration, HAR2-vs-HAR4 late-VALID decision, fitted blend weights, separate direct-HAR late-VALID score and any zero-RV floors. Particle precision is audited at 5m, 60m and 240m on **late VALID only**, with 256 vs 512 Sobol nodes (up to 96 origins per asset); no test-informed tuning of quadrature or model parameters. Differences smaller than numerical instability are not credited as model gains. Avoid retaining full origin×particle arrays or raw tick data.
- Improvement against M2 requires a negative 60m paired NLL difference with a negative upper day-bootstrap bound, **and** scrutiny of tail calibration/QLIKE. Historical selection cannot establish a prospective win or trading alpha. If HAR_DIRECT wins independently, pursue coherent HAR-augmented state simulation; if only blends or M2 anchor win, interpretation is narrower.

## Files and execution

- `src/data_engineering/modeling/markov_har_variance.py`
- `tests/test_markov_har_variance.py` (synthetic; with VD14–VD17 regression suite)
- `notebooks/Volatility_Dynamics/18_har_markov_variance.ipynb`
- `discussions/Volatility_Dynamics/035_chatgpt_vd18_har_vs_markov_variance_preregistration.md`

Notebook writes only small `18_scores.csv`, `18_bootstrap.csv`, `18_per_asset.csv`, `18_har_family_selection.csv`, `18_particle_sensitivity.csv`, `18_decision.json`, `18_manifest.json` under `notebooks/Volatility_Dynamics/18/`. No future-delivery promise; market-history runs require user-local raw data.
