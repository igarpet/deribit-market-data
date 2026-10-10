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

## Data window

Both modules default to the full available database history unless their module-specific start/end environment variables are deliberately set.

## Run Module 02

Open:

`notebooks/Volatility_Dynamics/02_native_5m_volatility.ipynb`

It writes compact outputs to `notebooks/Volatility_Dynamics/02/`.

The notebook reads `DERIBIT_DATA_ROOT/database/research.duckdb` read-only.
