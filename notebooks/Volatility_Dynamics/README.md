# Volatility Dynamics

Independent research branch for the observable dynamics of physical volatility.

## Research sequence

### 01 — Volatility state panel

`01_volatility_state_panel.ipynb`

Builds causal expanding-session HAR+clock forecasts at 30m/60m and separates:

- expected high-volatility episodes;
- positive volatility-surprise / shock candidates;
- forecast-high misses.

It then studies exact-clock-normalized flow, liquidity, basis/OI and IV diagnostics around declustered event origins, from 60 minutes before through 120 minutes after.

This module does **not** fit HMM/HSMM states and does not test direction.

Outputs live in `01/` and are intentionally compact.

### Proposed 02 — Absorption / Bridge hypothesis

Only after 01 is reviewed. Test whether unusually large activity with unusually small contemporaneous price response predicts later volatility expansion and/or directional movement.

### Proposed 03 — MOM vs MR conditional on volatility environment

Only after 01–02. Test whether displacement has different forward-return slopes in stable/contracting versus expanding/surprise volatility environments.

## Data window

This branch is exploratory and may use all currently available research data. By default Module 01 runs from `VD_START=2026-09-22` through the latest row in the database. Set `VD_END=YYYY-MM-DD` only when a deliberate historical cutoff is desired.

## Run

Open and run:

`notebooks/Volatility_Dynamics/01_volatility_state_panel.ipynb`

The notebook reads `DERIBIT_DATA_ROOT/database/research.duckdb` read-only.
