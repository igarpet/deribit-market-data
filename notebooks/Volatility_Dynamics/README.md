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


### 05 — The −95 minute q99 precursor motif

`05_minus95m_q99_motif.ipynb`

Characterizes the visually discovered structure around roughly 95 minutes before q99 onset.

The motif is frozen as the five native candles at:

- -105m
- -100m
- -95m
- -90m
- -85m

rather than treating one exact plotted point as a magic lag.

The primary sample excludes any q99 onset that had **any prior q99 bar in the preceding 120 minutes**. Eligible matched controls are subjected to the same exclusion. This is designed to distinguish a genuine pre-event motif from the after-effects of an earlier extreme-volatility shock.

Module 05 rebuilds the same-asset, exact-clock, native-volatility-matched controls used in Module 04. It then reports:

- all-event versus clean-event profiles;
- feature-level local motif level, center deviation and rebound;
- family-level activity, IV, liquidity/book-activity and L2-instability components;
- descriptive bootstrap stability over unique event timestamps;
- leave-one-asset-out stability;
- exact -95m asset-level diagnostics.

Because the -95m motif was discovered by viewing the same sample, Module 05 is explicitly **post-selection exploratory analysis**. It cannot establish prospective predictability. A surviving clean motif must be frozen and tested on later/non-discovery observations.


### 06 — Latent market-state HMM

`06_latent_market_state_hmm.ipynb`

Introduces the first latent-state model in the Volatility Dynamics branch.

The core model is a first-order diagonal-Gaussian HMM fit to a compact emission vector motivated by Modules 03–05:

- intrabar range;
- USD volume;
- trade count;
- ATM forward IV;
- 25d put IV;
- 25d call IV;
- book spread;
- book update count.

The sparse L2 zero-cross/transition/roughness variables are tested only in a **separate L2-enhanced HMM**, so their limited history does not shorten the main core model.

Key design constraints:

- q99 is **not** used to fit the HMM;
- `return_5m` is not an emission feature;
- no MR/MOM/VOL/Bridge/Clean/Noisy labels are forced;
- no HSMM duration distribution is imposed;
- deterministic asset/clock scaling is estimated from the training period only;
- model fitting uses a chronological train/test split;
- candidate state count is selected using training BIC with an occupancy guard;
- held-out q99 hazard is calculated from **filtered** state probabilities only;
- smoothed states are not used for held-out/live-style evaluation;
- sequences break on asset changes, missing required emissions, or non-five-minute gaps, but not at UTC midnight.

For each selected model, Module 06 reports:

- candidate-K train/test likelihood and training BIC;
- state emission profiles;
- transition probabilities;
- held-out filtered-state q99 hazard at 30/60/90/120m horizons;
- held-out hard-state transition q99 hazard;
- event-aligned filtered state probabilities around q99;
- empirical filtered-state durations versus the HMM geometric-duration implication;
- age-specific empirical exit hazard versus the HMM constant exit hazard.

An HSMM is not the automatic next step. It is justified only if duration/age diagnostics indicate material HMM misspecification and the richer model subsequently improves chronological out-of-sample performance.

## Data window

Both modules default to the full available database history unless their module-specific start/end environment variables are deliberately set.

## Run Module 02

Open:

`notebooks/Volatility_Dynamics/02_native_5m_volatility.ipynb`

It writes compact outputs to `notebooks/Volatility_Dynamics/02/`.

The notebook reads `DERIBIT_DATA_ROOT/database/research.duckdb` read-only.
