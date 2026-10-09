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

## Holdout protection

Volatility Regression has a frozen confirmation sample beginning with expiry sessions on 2026-10-09. Module 01 therefore defaults to `VD_END=2026-10-08` and refuses to inspect later outcomes unless `VD_ALLOW_CONFIRMATION_CONTAMINATION=1` is explicitly set after Ivan releases that holdout.

## Run

Open and run:

`notebooks/Volatility_Dynamics/01_volatility_state_panel.ipynb`

The notebook reads `DERIBIT_DATA_ROOT/database/research.duckdb` read-only.
