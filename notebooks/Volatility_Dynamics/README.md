# Volatility Dynamics

Independent research branch for the observable dynamics of physical volatility.

## Independence rule

This branch starts from the shared research database only. It does **not** import research outputs, fitted models, frozen specifications, selected features, thresholds, state definitions, or empirical assumptions from HSMM, Volatility Regression, Volatility Risk Premium, Pairwise Feature Regression, or any other research branch.

Cross-module inheritance inside this branch is also not automatic: each module states explicitly whether it consumes an earlier module. Module 02 does **not** consume Module 01 outputs or conclusions.

## Research sequence

### 01 — Volatility baseline from first principles

`01_volatility_baseline.ipynb`

Historical first standalone module. It constructed several rolling realized-variance horizons from raw five-minute returns. Its results remain useful descriptively, but rolling 30m episode durations are not suitable as primitive duration evidence because the window itself induces persistence.

### 02 — Native five-minute volatility dynamics

`02_native_5m_volatility.ipynb`

This is the authoritative primitive-timescale study for volatility persistence and duration.

It reads only:

- `state_time`
- `underlying`
- `return_5m`

and defines:

[
v_t=r_t^2
]

for the single native five-minute bar.

There is no rolling realized-variance block.

Module 02 measures:

- five-minute variance distribution and tails;
- missingness by date and exact five-minute UTC slot;
- exact-slot five-minute volatility clock structure;
- lag dependence between individual five-minute variance observations;
- one-step empirical volatility-quantile transition matrices;
- 90th / 95th / 99th percentile single-bar tail persistence;
- consecutive runs of extreme native five-minute bars;
- five-minute event paths around run onset;
- same-timestamp cross-asset extreme-volatility synchronization.

Any future claim about duration or persistence should start from Module 02, not from the rolling-window duration in Module 01.


### 03 — q99 precursor discovery

`03_q99_precursor_discovery.ipynb`

Recomputes native five-minute q99 volatility onsets internally, then asks whether **any numeric/boolean market-state field observable before the event** is systematically unusual.

Predictors are discovered directly from the current `time_series_feature_state` schema at runtime. No feature list or ranking is imported from another research branch or from Module 01/02 outputs.

For each q99 onset, controls are nearby-date non-q99 observations from the same asset and exact five-minute UTC clock slot. Event values are expressed as percentiles versus those controls from -120m through +120m.

Only negative relative times can enter precursor rankings. The event bar and post-event bars are descriptive anatomy only.

The exploratory ranking penalizes candidates whose direction is inconsistent across assets or across the pre-event lags (-60, -30, -15, -10, -5 minutes). Simultaneous multi-asset q99 timestamps are averaged before pooled ranking so one broad market move cannot receive seven times the weight.

If coherent candidates emerge, a later module must freeze and test them chronologically before any predictive claim.


### 04 — Conditional q99 precursor discrimination

`04_conditional_q99_precursors.ipynb`

Tests whether the frozen Module 03 candidate families remain unusual after controlling for the pre-existing volatility environment.

The matching baseline remains native five-minute data. Each event/control origin is represented by exact single-bar squared-return observations at -120, -60, -30, -15, -10 and -5 minutes.

Controls must be the same asset and exact five-minute UTC clock slot, then are selected for similarity in that six-point native-volatility history. No rolling 30m/60m realized-variance field is used for matching.

The frozen primary candidates are:

- native intrabar range;
- USD volume and trade count;
- ATM, 25d put and 25d call IV levels;
- book spread;
- book update count;
- OBI zero-cross / transition / roughness instability measures.

Module 04 recomputes q99 from `return_5m`; it does not read Module 03 output files. It explicitly compares the original clock-only control result with the native-volatility-conditioned result, including candidate-level attenuation/retention and family summaries.

A surviving candidate is still only an incremental precursor candidate. Predictive claims require a later chronological prospective test.

## Data window

Both modules default to the full available database history unless their module-specific start/end environment variables are deliberately set.

## Run Module 02

Open:

`notebooks/Volatility_Dynamics/02_native_5m_volatility.ipynb`

It writes compact outputs to `notebooks/Volatility_Dynamics/02/`.

The notebook reads `DERIBIT_DATA_ROOT/database/research.duckdb` read-only.
