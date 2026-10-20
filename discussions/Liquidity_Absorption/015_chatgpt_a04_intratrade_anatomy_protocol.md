# Liquidity Absorption 015 — A04 intratrade anatomy protocol

**Author:** ChatGPT · **Date:** 2026-10-20 · **Status:** synthetic-tested implementation, historical runs pending · **Scope:** exploratory outcome-conditioned diagnostics, not a new trading edge.

## Why this module

A03–A03d show whether fixed structural profit targets are eventually reached under specified stops/timeouts, but do not directly explain the *intra-position dynamics*. Studying only winning trajectories after exit invites label leakage and survivorship bias. A04 therefore examines **completed bars prior to fixed landmark times** for trades still open at each time and compares eventual TARGET vs STOP vs TIMEOUT after that landmark. This does **not** identify a causal mechanism or prospective classifier.

### Existing preliminary diagnosis (A03b, 1:1 variant only)

413 completed BTC/ETH entries: 116 TARGET, 112 STOP, 185 TIMEOUT, 2 censored. The median A03b gross target distance is ~39.8bp for TARGET, ~45.2bp for STOP, and ~145.5bp for TIMEOUT. When each target distance is divided by the independent point-in-time VD18 HAR_4 forecast of **240m terminal return sigma**, the group medians are ~0.47, ~0.61, ~1.96. The substantial timeout association is at least partly mechanical: unreachable barrier distances within a prechosen 240m horizon. This is not a standalone alpha result, and not a finding from A03d.

## Frozen A04 design

1. **Primary event universe:** precisely the A03d policy `entry_kind=TAKER_FIRST_POST_SIGNAL`, `model_mode=VD18_VOL_FIXED`, `hold_minutes=240`, and 452 original A02 IDs. Keep all skipped and censored events in population accounting. **No choice of exit mode/horizon is made from outcome comparisons.** Other policies should be separately preregistered studies.
2. **Market source:** existing Binance UM **one-minute OHLC + quote volume + taker-buy quote volume** from `BTCUSDT_um_1m.parquet` and `ETHUSDT_um_1m.parquet`. No new tick downloads, replication or bid/ask assumptions. Causal 1m candle ends at `open_time + 60 seconds`.
3. **Checkpoint times:** 1, 5, 15, 30, 60 minutes after the already-completed signal close. For time `t` require exact complete consecutive minute closes `(signal_close, t]`. No completed candle after checkpoint may enter features. If historical trade already exited by `t`, do NOT treat its features as if the position remained open. No forward fill across missing minutes; preserve `INCOMPLETE_MINUTE_WINDOW` and other statuses.
4. **Same entry and target:** original A03d first-tape-print entry, preassigned structural target and VD18-protected stop are read unchanged from simulation results. No refitting or change in trade payoff.
5. **Early features:** signed close markout (bp) and target/stop normalized progress; one-minute OHLC max favorable/adverse *available by checkpoint* (intraminute crossing order not known); ratio of favorable closes relative to defended level; fractions of candles overlapping the zone; kline quote turnover relative to completed 30m pre-signal; *signed taker-buy quote imbalance* aligned with trade side; 60-minute VD18 forecast sigma and target/sigma geometry. These are **proxies** for price acceptance, volume imbalance and retracement, not direct latent passive-liquidity identity.
6. **Labels:** retrospective eventual TARGET, STOP, TIMEOUT; among positions at risk at each landmark. Stops and targets before checkpoint are reported separately as exits, not hidden. Output *outcome-conditioned descriptive medians*, plus cumulative exit counts using the original 452 denominator. No classifier is fit, no thresholds selected on the observed period.
7. **Cross-asset stratification:** BTC and ETH separate; any combined interpretation requires accounting for side, target source, level family, forecast volatility and target-feasibility effects. Signals may overlap, so observations are not independent; prospective uncertainty should block by UTC day and asset.

## Deliverables

- `src/data_engineering/modeling/liquidity_absorption_stage_a04.py`: read-only one-minute cohort processor.
- `tests/test_liquidity_absorption_stage_a04.py`: synthetic strict-as-of, risk-set, missing-minutes, stop/target geometry, flow sign tests.
- `notebooks/Liquidity_Absorption/04_intratrade_path_anatomy.ipynb`: local run in `deribit-data` kernel. Outputs under `notebooks/Liquidity_Absorption/04/`:
  - `04_primary_event_outcomes.csv`
  - `04_landmark_panel.csv`
  - `04_landmark_summary.csv`
  - `04_competing_exits.csv`
  - `04_competing_exits.png`
  - `04_landmark_features.png`
  - `04_decision.json`

## Interpretive gates / limitations

- **Do not** call post-selection TARGET-versus-STOP path differences predictions. It is not enough to demonstrate that winners progressed toward their targets by +30m; this is expected for a path approaching a barrier.
- **Target feasibility first:** estimate target distance divided by expected sigma, existing stop distance, and underlying target choice. An apparent signal can be mechanical proximity to a barrier. Terminal-return sigma is not a first-passage probability.
- **No tick/L2 or realized fills:** Binance 1-minute taker quote volume identifies directional *aggressive activity*, not true passive absorption, queue position or bid/ask execution. Capturing minute high/low does not reveal the intraminute order in which barriers would cross.
- **Censoring / attrition:** the active risk-set population shrinks at each checkpoint. Strong-looking 60m winner features condition on surviving to 60m; do not pool them with 1m features as though the samples are identical.
- **Spot/perp distinction:** VD18 spot return variance forecast is used as frozen context. Actual trades and aggressive volume proxies are from Binance perps. Transfer calibration remains unverified.
- **Future confirmation:** July–September 2026 has been examined repeatedly. Frozen early-feature hypotheses need new unseen forward data, day-blocked evaluation and costs before any trading conclusion.

## Proposed next module after results

If +5m/+15m features separate outcomes *after controlling for target/sigma and level family*, preregister one small set of features from the present **risk set** and test a longitudinal conditional target-before-stop probability model (cause-specific hazards / competing risks), with a simple geometry-only benchmark. Do not optimize stop/target on the same discovered subset.
