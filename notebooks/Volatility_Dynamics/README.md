# Volatility Dynamics

Independent research branch for the observable dynamics of physical volatility.

## Independence rule

This branch starts from the shared research database only. It does **not** import research outputs, fitted models, frozen specifications, selected features, thresholds, state definitions, or empirical assumptions from HSMM, Volatility Regression, Volatility Risk Premium, Pairwise Feature Regression, or any other research branch.

Module 01 reads only:

- `state_time`
- `underlying`
- `return_5m`

from `time_series_feature_state`.

Any later complexity must be motivated by results produced inside Volatility Dynamics itself.

## Research sequence

### 01 — Volatility baseline from first principles

`01_volatility_baseline.ipynb`

Starts without a forecasting model or latent-state assumption. It constructs realized variance directly from contiguous five-minute index returns and measures:

- distributions at 5m, 15m, 30m, 60m and 4h backward horizons;
- 30m/60m forward realized variance for descriptive current-vs-future association;
- UTC clock structure, estimated internally rather than assumed;
- volatility persistence at several lags;
- high-volatility episode thresholds and contiguous episode durations;
- buildup and decay paths around the first threshold crossing.

No option, flow, order-book, HAR, HMM/HSMM, MOM/MR/VS/Bridge or prior-branch variable enters Module 01.

Outputs live in `01/` and are compact.

### Later modules

The next module is **not precommitted**. Module 01 results decide what comes next.

Possible future questions include forecast baselines, volatility surprise, absorption/Bridge, or volatility-conditioned MOM/MR, but none is imported as an assumption.

## Data window

The default is the entire available database history. Set `VD_START` or `VD_END` only when a deliberate historical cutoff is desired.

## Run

Open and run:

`notebooks/Volatility_Dynamics/01_volatility_baseline.ipynb`

The notebook reads `DERIBIT_DATA_ROOT/database/research.duckdb` read-only.
