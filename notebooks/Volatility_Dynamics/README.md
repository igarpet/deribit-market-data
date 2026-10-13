# Volatility Dynamics

Independent research branch for the observable dynamics of physical volatility.

## Independence rule

This branch starts from the shared research database only. It does **not** import research outputs, fitted models, frozen specifications, selected features, thresholds, state definitions, or empirical assumptions from HSMM, Volatility Regression, Volatility Risk Premium, Pairwise Feature Regression, or any other research branch.

**Approved exception (Ivan, 2026-10-10):** Module 10 reuses the Volatility Risk Premium branch's Black-76 conventions (`option_pricing`) and its pricing-audit findings (VRP 006), and never touches sessions from the VRP confirmation boundary (2026-11-07) onward. Module 10b inherits this through Module 10's priced quotes. No other module uses VRP outputs.

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

### 09 — Compound-Poisson jumps (VD 019)

`09_compound_poisson_jumps.ipynb` — Module 08 detection unchanged (c = 4). It produces:

- a jump catalogue;
- size distributions in bps and in local-σ units (exponential vs GPD tail);
- a common vs idiosyncratic arrival decomposition (pre-registered P1: co-jump bars vs independence, Poisson p < 0.01);
- Garwood rate CIs and $P(\ge1\text{ jump in }\tau)$;
- an index tick-discreteness check (P2: median jump ≤ 3 ticks means an artifact);
- half-sample stability.

Outputs: `09/`.

### 10 — Physical jump model vs 0DTE option wings (VD 019, cross-branch)

`10_jump_model_vs_option_wings.ipynb` prices every two-sided OTM 0DTE quote (60–1200 min, 15-min grid, non-floor) under a compound-Poisson + diffusion model. The diffusion variance is ex-ante: backward bipower × clock, with settlement-TWAP weights. Jump rate and size are leave-one-session-out. Quotes are compared in price space.

The primary, pre-registered statistic is the median over sessions of the session median $\ln(\text{mid}/P^{jump})$ for $z\in[2,5)$, giving RICH / CHEAP / CONSISTENT by session bootstrap.

Also reported:

- all buckets and per-asset results;
- the no-jump model;
- sensitivities: diffusion ×0.85, λ ×0.5 and ×2;
- hold-to-settlement P&L.

Outputs: `10/`.

### 10b — Model-free physical benchmark for the wings (VD 021)

`10b_model_free_wing_benchmark.ipynb` re-prices Module 10's primary rows from the realized distribution of the standardized move to settlement, $u=\ln(S_T/I_t)/\sqrt{D_t}$, using Module 10's own ex-ante $D$, so any bias in $D$ cancels. The $u$ pool is leave-one-session-out, with the raw outer tails kept and a per-quote martingale normalization.

Pre-registered rule: $M$ is the median of session-median $\ln(\text{mid}/P^{emp})$ for $z\in[2,5)$. The confidence interval comes from a joint session bootstrap that resamples quotes and outcomes together, giving RICH / CHEAP / CONSISTENT. Rows priced at zero are dropped, which is conservative against RICH. The verdict becomes INSUFFICIENT_TAIL if more than 25% are dropped.

Reported only: the variance mixture, the TTE-conditioned pool, per asset, buckets and tail frequencies. 10b consumes `10/10_priced_quotes.csv.gz`. Outputs: `10b/`.

### 10c — Long-history physical tail for the wings (VD 023)

`10c_long_history_wing_benchmark.ipynb` replaces 10b's 18-session outcome pool with about two years of 5-minute history taken strictly before the first quote. The history is Binance spot `<ASSET>USDT`, falling back to the Deribit perpetual, and is downloaded with `tools/download_price_history.py` to `DERIBIT_DATA_ROOT/history/price_5m/`. $D$ uses exactly Module 10's clock.

A consistency gate compares the long-history $u$ with 10b's index $u$ on the quote window (corr ≥ 0.90, sd ratio in [0.85, 1.15]).

- **Decision V_bid:** the median of session-median $\ln(\text{bid}/P)$, with a CI from a joint bootstrap over quote sessions × history dates (B = 1000).
- **V_mid:** the same at the mid, for continuity.
- **Reported:** net-of-fee bid, break-even disaster mass, and a refitted-clock sensitivity.

Outputs: `10c/`.

### 10d — Wings priced by volatility regime × time to expiry (VD 028)

`10d_regime_conditioned_wings.ipynb` — 10c's two-year pool, clock and gate; each wing quote priced from its own (own-asset volatility tercile × TTE bucket) cell. Two pre-registered one-sided decisions with Holm (α = 0.05): calm tercile CHEAP at the ask net of fee (long edge), volatile tercile RICH at the bid net of fee (short edge). No rejection closes the wing line. **Result (VD 029): `NOT_CHEAP` (calm ask +0.54 log over fair) / `NOT_RICH` (volatile bid −0.23) — wing line closed.** Code: `volatility_wings_regime.py`; tests: `tests/test_volatility_wings_regime.py`; outputs `10d/`.

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_volatility_wings_regime.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks\Volatility_Dynamics\10d_regime_conditioned_wings.ipynb
git add notebooks\Volatility_Dynamics
git commit -m "VD 10d: regime-conditioned wings run"
git push
```

### 11 — Lifted rough-kernel volatility memory (VD 11)

`11_lifted_rough_memory.ipynb` — An independent prospective test of whether a finite exponential lift of native 5-minute squared-return memory improves q99-onset hazard prediction beyond fixed volatility lags and ordinary exponential-memory baselines. It does **not** fit a latent lifted-Heston model, imply a validated roughness estimate, modify the HSMM, or authorize option trades.

The notebook uses read-only `time_series_feature_state` (`state_time`, `underlying`, `return_5m`) and writes compact research artifacts to `11/`. Methodology: training-only variance floor and asset/UTC-hour de-clocking; 60/20/20 chronological fit/validation/test split; strict five-minute continuity; asset-specific training q99 thresholds; fully observed 30/60/90/120-minute forward-label windows; validation-selected `H`, lift dimension, and ridge; paired UTC-day bootstrap. Primary 60-minute confirmation requires improvements versus **both** fixed-lag and equal-timescale exponential-memory baselines, at least 20 unique held-out q99 onsets and at least 10 test days.

Implementation: `src/data_engineering/modeling/volatility_lifted_memory.py`. Tests: `tests/test_volatility_lifted_memory.py`. This module is a separate memory-representation research test; it does **not** reopen the option-wing backtest that was paused after Module 10c. No empirical Deribit result is claimed until the notebook is executed on the current research database.

### 12 — Module 11 on two years of history, with unrestricted and climatology benchmarks (VD 026)

`12_lifted_memory_long_history.ipynb` — Module 11 v0.1.1's pipeline and rules, unchanged, on the two-year Binance-spot 5m history (`DERIBIT_DATA_ROOT/history/price_5m`, from `tools/download_price_history.py`), restricted to data **before 2026-09-19** (the Module 11 window is excluded). Adds **K** (climatology: asset effects + time-of-day only) and **U** (every factor EMA as its own regressor). Pre-registered 60m status ladder: `PASS_ROUGH_SPECIFIC` → `MULTISCALE_MEMORY_NOT_ROUGH_SPECIFIC` → `TWO_EMA_MEMORY_SUFFICIENT` → `SKILL_WITHOUT_LONG_MEMORY` → `NO_SKILL`. Synthetic power check: a truly rough panel ($H$ = 0.08) returns `TWO_EMA_MEMORY_SUFFICIENT`, so this task cannot detect roughness (VD 026 §3).

**Result (run `55c46c4`, VD 027): `TWO_EMA_MEMORY_SUFFICIENT`.** 445 onsets / 150 test days; ~21% log-loss skill over climatology at 60m; two EMAs beat 2h lags; rough kernel, equal weights and unrestricted weights add nothing (unrestricted is worse); $H$ unidentified.

Code: `src/data_engineering/modeling/volatility_lifted_memory_long.py` (+ `volatility_lifted_memory.py`). Tests: `tests/test_volatility_lifted_memory_long.py`. Outputs: `12/12_*`. Runtime about 5–15 min.

Run (PC, Windows CMD; close the notebook in Jupyter first):

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_volatility_lifted_memory_long.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks\Volatility_Dynamics\12_lifted_memory_long_history.ipynb
git add notebooks\Volatility_Dynamics
git commit -m "VD 12: long-history lifted memory run"
git push
```

## Data window

Both modules default to the full available database history unless their module-specific start/end environment variables are deliberately set.

## Run Module 02

Open:

`notebooks/Volatility_Dynamics/02_native_5m_volatility.ipynb`

It writes compact outputs to `notebooks/Volatility_Dynamics/02/`.

The notebook reads `DERIBIT_DATA_ROOT/database/research.duckdb` read-only.


### 11 v0.1.1 — Missing-aware lifted volatility memory (2026-10-12)

`11_lifted_rough_memory.ipynb` and `src/data_engineering/modeling/volatility_lifted_memory.py`.
Original 2026-10-12 execution in commit `16487be` **failed during validation**: after requiring
288 **consecutive** finite returns the shared support was train 2,216 / 25,536,
validation 0 / 8,064, test 0 / 8,736. The failure is a filter-design bug, **not a
negative forecast result**. No valid real-data scores or PASS/FAIL verdict exist.

v0.1.1 retains 288 elapsed 5m slots, requires >=80% *observed* returns over the
strictly previous 24h plus at least one valid return in the previous 15m, and
uses observed-weight-normalized causal EMAs. Missing returns contribute zero
observation weight; no data are interpolated or fabricated. Test labels still
require fully observed future windows and cannot cross chronological splits.
The 80% / 15m policy is a documented engineering repair, **not an empirically
optimized research threshold**. Daily-gap synthetic tests now protect against
zeroing validation/test support; seven automated tests pass locally.

Results are **PENDING RERUN** on `research.duckdb`, after pulling the fix.
With the original roughly 21-day observed window (five test UTC days), the
pre-registered >=10-test-day confirmation gate will still return
`INSUFFICIENT_DATA` even if predictive scores can now be computed.

### 13 — Two timescales, a price martingale and conditional local reversion

`13_two_timescale_price_distribution.ipynb`

Independent research starting from shared Binance spot 5-minute close history (`DERIBIT_DATA_ROOT/history/price_5m`), using no fitted model/output from other research branches. Historical input is strictly before 2026-09-19. Pre-registration: `discussions/Volatility_Dynamics/030_chatgpt_vd13_price_distribution_preregistration.md`.

Primary 60-minute forecast (secondary 15m, 240m): integrated future realized variance, signed net log return, and the probability of a large absolute net move (95th-percentile threshold learned on training only). Compares a zero-drift constant-variance random walk K, the established two-EMA memory baseline E, nonlinear volatility-phase model P, signed-return mean correction M, and phase-conditioned return mean PM. All models use exactly the same uncensored future labels and complete-case forecast origins. Model statistics: Gaussian return log score, log-RV MSE, tail log loss/Brier, and up/down log loss; paired UTC-day bootstrap.

**Status: pre-registered, synthetic-tested, not run on real data.** The previously analyzed two-year historical sample is deliberately labeled retrospective/exploratory; new prospective data are required for confirmation. Regime Structure 06's `NO_LONG_EDGE` gamma-scalping result is an external motivation, **not** imported research evidence or an authorized option-trading signal. Results are compact in `13/`.

### 14 — Seven-horizon Markovian lift and physical return-distribution tests

`14_markov_lift_distributions.ipynb`

Freshly registered study over **5m, 10m, 15m, 30m, 60m, 120m and 240m** net-return horizons, with 60m primary. Compares M1 two-EMA Gaussian, M2 Student-t, M3 variance-matched constant broad-tail mixture, M4 phase-conditional mixture, and **AM**: augmented Markov fast/slow volatility-state transitions with 5m statistical shock hazard propagated across horizons. Separate fit code `src/data_engineering/modeling/markov_lift_distributions.py`; five synthetic tests `tests/test_markov_lift_distributions.py` pass. All models use same within-horizon origins; UTC-day paired bootstrap and tail calibration. Descriptive/exploratory historical sample before 2026-09-19; no real-data outputs yet, no option backtest, no trading claim. Registration: `discussions/Volatility_Dynamics/031_chatgpt_vd14_markov_lift_preregistration.md`. Compact outputs under `14/` after execution.
