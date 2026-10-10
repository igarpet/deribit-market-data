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

## b-run (02b–06b) — review re-run, VD 010

Parallel re-runs of Modules 02–06 with the fixes from the Claude review (discussion `010`). The original notebooks and outputs are unchanged. Outputs go to `02b/` … `06b/`. Shared vectorized core: `volatility_vd_b_core.py`. Tests: `tests/test_volatility_dynamics_b.py`.

| Notebook | Question | Pre-registered rule |
|---|---|---|
| `02b_clock_null.ipynb` | How much of the exact-slot clock profile is sampling noise? (per-day rotation null, split-half reliability, slot vs hour) | descriptive |
| `03b_q99_precursor_null.ipynb` | Is the Module 03 ranking beyond selection noise? (exact Module 03 scan on day-relabel placebo events; per-field and family-wise p) | descriptive; volatility fields are expected to beat this null |
| `04b_repaired_vol_conditioning.ipynb` | Does anything survive real volatility conditioning? (causal 15m/1h/4h/24h RV + range environment, ±30 min clock pool, bias-corrected matching, placebo band) | gate: RV1h/RV4h retention ≤ 0.20 under M1_bc and M2_bc; survivors need Holm p < 0.05 under M2_bc; survival is necessary, not sufficient |
| `05b_prospective_motif95.ipynb` | Does the frozen −95m L2 notch appear in clean q99 onsets after 2026-10-08 08:00 UTC? | PASS iff ≥ 20 unique timestamps, mean notch < 0, one-sided sign-flip p < 0.05 |
| `06b_hmm_vs_benchmark.ipynb` | Do HMM states forecast q99 beyond a trivial volatility benchmark? | h = 60 min: logloss(M3) − logloss(M1) < 0 with day-bootstrap q97.5 < 0 and ≥ 20 unique test onset timestamps |

### 06c — Student-t emission HMM (VD 012)

`06c_student_t_hmm.ipynb` — 06b pipeline with a multivariate Student-t emission (diagonal scale, per-state ν, shared latent scale per bar, missing fields marginalized). Pre-registered: same forecast decision as 06b; K grid 2–10 with an interior-K test; Student-t vs Gaussian validation likelihood on identical sequences. Outputs: `06c/`.


### 07 — q99 pre-shock HMM taxonomy

`07_q99_preshock_hmm_taxonomy.ipynb`

Changes the latent-state question after the inconclusive 06b/06c forecasting results.

Module 07 does **not** ask whether an HMM predicts q99 better than volatility. It conditions on q99 having occurred and asks whether the strictly pre-shock trajectories themselves separate into recurring types.

For each asset-level q99 onset, the HMM sees exactly 24 native five-minute bars:

-120m through -5m.

The q99 event bar at t0 and every post-event observation are excluded from the HMM and from shock-type clustering.

The emission information is frozen from the Module 05 -95m research families:

- activity: intrabar range, USD volume, trade count;
- IV level: ATM forward IV, 25d put IV, 25d call IV;
- liquidity/book activity: spread, book update count;
- L2 instability: OBI zero-cross count/rate, transition count, path roughness.

Source features are transformed and normalized by asset x UTC hour using only the discovery-era panel, then averaged within family. The family representation reduces duplicated IV/L2 variables before the diagonal-Gaussian HMM is fit. Missing L2 dimensions are marginalized rather than deleting the shock window.

The analysis has two latent objects:

1. **HMM states** — recurring phases inside the 120-minute precursor window.
2. **Shock types** — clusters of the complete 24-bar posterior-state trajectory.

Simultaneous multi-asset q99 shocks are kept as separate asset instances but weighted so each exact event timestamp contributes total weight one to HMM fitting and trajectory clustering.

Shock sign, q99 magnitude, simultaneous-asset count, prior-q99/aftershock status, asset, date, and collector era are not used in fitting. They are post-hoc descriptors only.

The chronology is split by unique q99 timestamp into discovery and validation samples. Validation shocks are assigned to frozen discovery shock-type centroids.

Because 06b identified a potential collector shift in `book_update_count`, Module 07 reruns the full taxonomy with the liquidity family defined from spread alone and reports Adjusted Rand Index agreement.

A positive result would be a stable taxonomy of q99 precursor paths. It would **not** establish q99 predictability or an economic regime count.

### 08 — Jump arrivals as a point process (VD 017)

`08_jump_point_process.ipynb` — jumps = returns > 4 local-volatility units (two-sided jump-robust bipower, deseasonalized). Homogeneous Poisson vs clock/volatility-inhomogeneous Poisson vs discrete Hawkes with self- and cross-asset excitation (exponential kernel). Pre-registered: 5-fold blocked CV by day, logLik gain with day-bootstrap lower bound > 0; c grid 3–5 reported only. Explanatory, not a real-time forecaster. Outputs: `08/`.

## Data window

Both modules default to the full available database history unless their module-specific start/end environment variables are deliberately set.

## Run Module 02

Open:

`notebooks/Volatility_Dynamics/02_native_5m_volatility.ipynb`

It writes compact outputs to `notebooks/Volatility_Dynamics/02/`.

The notebook reads `DERIBIT_DATA_ROOT/database/research.duckdb` read-only.
