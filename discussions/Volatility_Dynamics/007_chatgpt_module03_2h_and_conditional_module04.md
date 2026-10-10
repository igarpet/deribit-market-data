# 007 — ChatGPT: extend Module 03 to ±2h and build conditional precursor Module 04

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-10 |
| **Branch** | Volatility Dynamics |
| **Responds to** | Ivan's instruction to extend Module 03 to 24 five-minute candles before/after and build Module 04 |
| **Status** | Implemented; both notebooks ready to run |

## 1. Module 03 anatomy window extended

The active Module 03 notebook is still:

`notebooks/Volatility_Dynamics/03_q99_precursor_discovery.ipynb`

Its anatomy window is now 24 native five-minute bars before and after the q99 onset:

-120 minutes through +120 minutes.

This produces 49 relative timestamps including the q99 bar at t0.

The exploratory precursor-ranking lags remain unchanged in Module 03:

-60, -30, -15, -10 and -5 minutes.

The extension is for richer pre/post event anatomy and does not retroactively change which lags produced the original Module 03 candidate ranking.

The helper version is now `volatility_q99_precursors 0.2.0`.

All notebook execution outputs were cleared after the logic change so stale ±60m displays are not confused with the new run.

## 2. Why Module 04 is needed

Module 03 established a broad pre-q99 constellation: realized volatility, native range, volume, trade count, IV level, spread and book activity were elevated, with sparse but strong L2-instability candidates.

The unresolved question is whether volume, IV, spread, or L2 instability contain information beyond the fact that the market was already volatile.

Module 04 therefore asks:

> Among similarly volatile pre-event environments, what still distinguishes q99 onsets?

## 3. Native five-minute volatility matching

Module 04 does not condition on a rolling 30m or 1h realized-volatility object.

For each event or control origin, the native-volatility environment vector contains the log of the exact single-bar five-minute variance observed at:

-120, -60, -30, -15, -10 and -5 minutes.

Each component is one exact native five-minute return squared. No sum or moving average is created.

For each q99 event, candidate controls must first satisfy:

- same asset;
- exact same five-minute UTC clock slot;
- non-q99 origin;
- prior exact bar is not q99.

Within that set, controls are ranked by robustly scaled Euclidean distance in the available native-volatility environment dimensions.

At least four of the six environment dimensions must overlap by default.

Up to eight nearest volatility-matched controls are retained.

## 4. Frozen candidate set

Module 04 does **not** scan all 205 fields again.

The following primary candidates are frozen from Module 03 before Module 04 results are observed:

### Native price activity
- `intrabar_range_bps`

### Trading activity
- `volume_usd`
- `trade_count`

### Option IV level
- `option_iv_atm_forward`
- `iv_put_25d`
- `iv_call_25d`

### Liquidity / book activity
- `book_twa_spread_bps`
- `book_update_count`

### L2 instability
- `book_obi_20_raw_zero_cross_count`
- `book_obi_20_transition_count`
- `book_obi_20_raw_zero_cross_rate`
- `book_obi_20_path_roughness_per_min`

This freeze prevents Module 04 from performing a second unrestricted feature-mining pass.

## 5. Diagnostics and contrast checks

Rolling realized-volatility fields:

- `realized_vol_30m`
- `realized_vol_1h`
- `realized_vol_4h`

are included only as diagnostics.

If native-volatility matching is working, their event/control deviation should attenuate relative to Module 03-style clock-only controls.

Weak/inconsistent Module 03 channels are retained as contrast checks:

- `trade_flow_imbalance`
- `signed_volume_usd`
- `open_interest_return_5m`
- `perpetual_mid_vs_index_bps`

They are not promoted to primary candidates.

## 6. Two parallel event studies

Module 04 recomputes q99 directly from `return_5m**2`; no Module 03 output file is read.

It constructs:

1. **Clock-only controls** — same asset/exact clock, nearest eligible dates, reproducing the Module 03 comparison.
2. **Native-volatility-matched controls** — same asset/exact clock, selected by distance in the six-point native-volatility history.

Every frozen feature is then profiled from -120 minutes through +120 minutes.

Only negative lags can count as precursor evidence.

## 7. Conditional ranking

The Module 04 ranking lags are:

-120, -60, -30, -15, -10 and -5 minutes.

The same dependence protections remain:

- asset-events at the same exact q99 timestamp are collapsed before pooled ranking;
- cross-asset direction consistency is recorded;
- cross-lag direction consistency is recorded.

For each frozen candidate, Module 04 reports how much of the original clock-only deviation from 0.5 remains after volatility matching.

This is descriptive attenuation/retention, not a significance test.

## 8. Interpretation

Three outcomes are possible.

### Candidate disappears

If its conditional percentile moves back toward 0.5, the Module 03 feature was largely an environment marker.

### Candidate partially survives

If the anomaly shrinks but remains coherent, the feature may carry some incremental information.

### Candidate survives strongly

If a feature remains substantially unusual with consistent direction across assets and pre-event lags after matching on native volatility, it becomes a much stronger precursor candidate.

Even then, no predictive claim is made.

## 9. Active files

Extended Module 03:

`notebooks/Volatility_Dynamics/03_q99_precursor_discovery.ipynb`

Module 03 helper:

`src/data_engineering/modeling/volatility_q99_precursors.py`

New Module 04:

`notebooks/Volatility_Dynamics/04_conditional_q99_precursors.ipynb`

Module 04 helper:

`src/data_engineering/modeling/volatility_q99_conditional.py`

Expected Module 04 outputs:

- `04/04_run_metadata.json`
- `04/04_candidate_contract.csv`
- `04/04_q99_thresholds_and_counts.csv`
- `04/04_q99_event_origins.csv`
- `04/04_vol_match_support.csv`
- `04/04_vol_matched_controls.csv`
- `04/04_native_vol_environment_balance.csv`
- `04/04_clock_only_profiles.csv`
- `04/04_vol_conditioned_profiles.csv`
- `04/04_vol_conditioned_by_asset.csv`
- `04/04_clock_only_rankings.csv`
- `04/04_vol_conditioned_rankings.csv`
- `04/04_candidate_comparison.csv`
- `04/04_family_summary.csv`
- `04/04_environment_diagnostic_attenuation.csv`
- `04/04_top_conditional_event_scores.csv`

## 10. Interpretation boundary

Module 04 answers whether a frozen Module 03 candidate remains unusual after controlling for a comparable native-volatility history.

It does not establish out-of-sample q99 prediction, a trading rule, option profitability, or a latent state/HSMM.

If one or more candidates survive, the natural next step is a frozen chronological prospective test.
