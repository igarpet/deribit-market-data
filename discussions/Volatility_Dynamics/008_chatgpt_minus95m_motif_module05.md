# 008 — ChatGPT: build Module 05 for the -95 minute q99 precursor motif

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-10 |
| **Branch** | Volatility Dynamics |
| **Responds to** | Ivan's observation of a potentially useful cross-feature pattern around -95 minutes before q99 events |
| **Status** | Implemented; awaiting real-data run |

## 1. Motivation

After Module 03 was extended to two hours on each side of q99 onset, the event-profile plots suggested a coordinated feature change around roughly -95 minutes.

The pattern was visible in more than one channel, including native range/trading activity and sparse L2 instability measurements.

A preliminary inspection also showed that some q99 events had earlier q99 onsets near this region. Therefore a dedicated module must distinguish a possible precursor motif from the mechanical aftermath of a previous shock.

## 2. The motif is a window, not one magic candle

Module 05 freezes the local motif as:

[
-105,-100,-95,-90,-85	ext{ minutes}.
]

Define:

[
Pre = mean(P_{-105},P_{-100}),
]

[
Center = P_{-95},
]

[
Post = mean(P_{-90},P_{-85}),
]

where (P) is the event feature's percentile relative to its same-asset, exact-clock, native-volatility-matched controls.

The module reports:

- five-candle window level;
- center versus neighboring candles;
- pre-to-center movement;
- -95-to-post rebound;
- pre-to-post movement.

This avoids overinterpreting one plotted observation.

## 3. Prior-q99 contamination control

The primary sample requires:

[
	ext{no q99 bar in }[t_0-120m,t_0).
]

This rule is stricter than excluding only an earlier q99 onset: **any** q99 five-minute bar in the preceding two hours makes the event contaminated for the primary motif sample.

Matched control origins are subjected to the same rule.

The original all-event sample remains available as a comparison.

If the motif weakens sharply after this exclusion, the original visual structure was likely partly an aftershock/recovery effect.

If it survives, it becomes materially more interesting.

## 4. Matching remains conditional on native volatility

Module 05 inherits the Module 04 matching framework:

- same asset;
- exact five-minute UTC clock;
- non-q99 eligible origin;
- nearest controls in the native five-minute volatility-history vector.

No rolling 30m/60m volatility is used for matching.

## 5. Frozen motif feature families

Module 05 does not reopen the 205-feature scan.

### Activity
- `intrabar_range_bps`
- `volume_usd`
- `trade_count`

### IV level
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

These are the families already motivated by Modules 03 and 04 before Module 05 is run.

## 6. Family-level motif scores

For every asset-event, Module 05 constructs family-level five-candle average percentiles.

It also constructs a specific L2 rebound metric:

[
L2Rebound =
mean(L2_{-90},L2_{-85}) - L2_{-95}.
]

This captures the visual possibility of a local L2 lull/reset near -95 followed by reacceleration.

No direction is imposed on the other families beyond reporting their level and shape.

## 7. Dependence and stability

Pooled summaries collapse simultaneous multi-asset q99 observations to one unique event timestamp before averaging.

Module 05 additionally reports:

- descriptive bootstrap intervals over unique event timestamps;
- leave-one-asset-out means;
- per-asset exact -95m summaries.

The bootstrap is **not** a confirmatory confidence interval because the motif was selected after viewing the same historical data. It is only a stability diagnostic.

## 8. Active files

Notebook:

`notebooks/Volatility_Dynamics/05_minus95m_q99_motif.ipynb`

Helper:

`src/data_engineering/modeling/volatility_q99_motif95.py`

Expected outputs:

- `05/05_run_metadata.json`
- `05/05_motif_contract.csv`
- `05/05_q99_thresholds_and_counts.csv`
- `05/05_event_sample_summary.csv`
- `05/05_prior_q99_contamination.csv`
- `05/05_clean_event_origins.csv`
- `05/05_clean_match_support.csv`
- `05/05_clean_vol_matched_controls.csv`
- `05/05_all_event_profiles.csv`
- `05/05_clean_event_profiles.csv`
- `05/05_all_local_motif_event_table.csv`
- `05/05_clean_local_motif_event_table.csv`
- `05/05_motif_all_vs_clean.csv`
- `05/05_clean_motif_event_components.csv`
- `05/05_component_summary.csv`
- `05/05_component_bootstrap.csv`
- `05/05_leave_one_asset_out.csv`
- `05/05_center95_by_asset.csv`

## 9. Interpretation boundary

A positive Module 05 result means:

> the -95m local pattern remains coherent after removing preceding-q99 contamination and after matching on the native volatility environment.

It does **not** mean:

> a live observation of the motif has been proven to forecast a q99 event 85-105 minutes later.

The latter requires freezing the motif definition and evaluating it prospectively/chronologically outside the discovery sample.
