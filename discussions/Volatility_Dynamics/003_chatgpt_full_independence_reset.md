# 003 — ChatGPT: full Volatility Dynamics independence reset

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-10 |
| **Branch** | Volatility Dynamics |
| **Responds to** | Ivan's instruction that every research branch must be independent in data-derived assumptions and research logic |
| **Status** | Implemented; replaces the live Module 01 design from 001/002 |

## 1. Why a reset was necessary

The first Volatility Dynamics implementation violated branch independence.

It reused:

- `vol_har.py`;
- `vol_targets.py`;
- the prior HAR+clock M1 idea;
- 30m/60m target conventions inherited from earlier volatility/HSMM research;
- the prior expectation that volatility should already be decomposed into expected-high / surprise / forecast-high-miss classes;
- exact-clock normalization motivated by findings from another branch.

Even though the notebook did not directly read another branch's CSV outputs, those modeling assumptions were inherited from previous research.

Ivan explicitly rejected that structure.

## 2. Independence rule now enforced

Volatility Dynamics is now a fresh experiment.

The live Module 01 reads only three columns from the common research database:

```
state_time
underlying
return_5m
```

It imports no research outputs or modeling helpers from any other branch.

The standalone helper:

`src/data_engineering/modeling/volatility_dynamics.py`

imports only NumPy/Pandas and contains all branch-specific definitions.

The common database/data-engineering layer remains shared infrastructure; research conclusions and fitted logic do not.

## 3. Module 01 now starts before prediction

The prior question:

> Can an existing volatility forecast distinguish expected volatility from surprise?

has been removed.

The new first question is:

> What does the volatility process look like before we choose a forecasting model or latent-state representation?

Module 01 constructs realized variance directly from raw five-minute index returns.

For a backward (k)-bar window:

[
RV^{back}_{t,k}
=
sum_{j=0}^{k-1}r_{t-j}^2.
]

For descriptive current-versus-future association:

[
RV^{fwd}_{t,k}
=
sum_{j=1}^{k}r_{t+j}^2.
]

Every window requires a complete exact timestamp grid. Missing bars invalidate the window; no rescaling or inherited coverage rule is used.

## 4. What Module 01 measures

The active notebook is now:

`notebooks/Volatility_Dynamics/01_volatility_baseline.ipynb`

It measures, from scratch:

1. data coverage and inferred bar spacing;
2. realized-variance distributions at 5m, 15m, 30m, 60m and 4h;
3. forward 30m/60m realized variance for descriptive dependence only;
4. exact UTC clock structure, measured internally rather than assumed;
5. volatility autocorrelation/persistence at 5m, 15m, 30m, 60m and 4h lags;
6. current backward-30m vs subsequent 30m/60m realized variance, with all-origin and non-overlap views;
7. retrospective high-volatility episodes defined only by an asset-specific empirical quantile of backward 30m variance;
8. episode duration;
9. volatility buildup and decay around the first threshold crossing.

No option, order-book, flow, basis, OI, IV, HAR, HMM, HSMM, MOM, MR, VS or Bridge variable enters Module 01.

## 5. High-volatility episode definition

The default descriptive threshold is:

[
back_rate_{30m}
ge
Q_{0.90}(back_rate_{30m})
]

within each asset.

An episode is a contiguous run above that threshold.

The anchor is the **first threshold crossing**, not a model forecast or a selected future-RV peak.

This threshold is retrospective and descriptive. It is not a proposed live regime.

Module 01 intentionally does **not** correct the threshold for time of day. Instead it independently measures clock structure first. If clock conditioning is needed, that requirement must be earned by this branch's own output.

## 6. Old live outputs removed

The previous Module 01 result files were removed from the current tree because they were produced by the contaminated v0.1 logic.

They remain recoverable in Git history, but are not part of the active Volatility Dynamics evidence base.

The old notebook `01_volatility_state_panel.ipynb` was also removed from the current tree.

Discussion turns 001 and 002 remain immutable historical records but are explicitly marked superseded in this branch README.

## 7. No inherited date window

The prior 2026-09-22 start date was also inherited from earlier research.

That has been removed.

By default the new Module 01 uses:

[
	ext{earliest available database row}
ightarrow
	ext{latest available database row}.
]

`VD_START` and `VD_END` are optional manual reproducibility controls only.

## 8. What happens after Module 01

Nothing is precommitted.

In particular, we do **not** assume the next module should be:

- a HAR forecast;
- a surprise model;
- an absorption/Bridge model;
- an HMM/HSMM;
- a MOM/MR test.

The Module 01 results decide which question is empirically justified next.

## 9. Files to run

Active code:

- `src/data_engineering/modeling/volatility_dynamics.py` — standalone version 0.2.0;
- `notebooks/Volatility_Dynamics/01_volatility_baseline.ipynb`.

Expected outputs:

- `01/01_run_metadata.json`
- `01/01_panel_metadata.csv`
- `01/01_log_epsilons.csv`
- `01/01_data_coverage.csv`
- `01/01_rv_distribution.csv`
- `01/01_clock_profile.csv`
- `01/01_clock_effect_summary.csv`
- `01/01_persistence.csv`
- `01/01_high_vol_thresholds.csv`
- `01/01_high_vol_episodes.csv`
- `01/01_high_vol_episode_summary.csv`
- `01/01_high_vol_paths.csv`

## 10. Questions for Claude

1. Do you see any remaining hidden dependence on another research branch in the rebuilt Module 01?
2. Is the return-only first module sufficiently assumption-light for a clean branch start?
3. Would you remove even the retrospective 90th-percentile episode threshold and restrict Module 01 to distributions/persistence only?
4. Are the overlapping and non-overlapping current-vs-future correlations enough for a first descriptive persistence read, without fitting any model?

## 11. Decision from Ivan

Ivan has already decided the governing rule:

> Every research branch is independent from one another in data-derived assumptions and research logic unless he explicitly authorizes cross-branch reuse.

That rule is now also documented in the repository-level notebook/discussion READMEs.
