# 014 — ChatGPT: Module 07 q99 pre-shock HMM taxonomy

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-10 |
| **Branch** | Volatility Dynamics |
| **Responds to** | Ivan's request to separate q99 shocks across assets and test whether their strictly pre-shock -120m trajectories differ |
| **Status** | Implemented; awaiting real-data run |

## 1. Why this is not another Module 06

The 06b and 06c experiments asked whether full-chronology HMM state probabilities improve prospective q99 forecasts beyond a volatility benchmark.

They did not.

Module 07 does not relax or retry that failed claim.

Instead it asks a different conditional question:

> Given that a q99 shock happened, are there several recurring types of pre-shock trajectory?

Every sequence in Module 07 is therefore a shock sequence.

There is no normal/non-event class and no q99 forecast benchmark.

## 2. Strict information window

For q99 onset t0, the HMM sequence is exactly:

[
[t_0-120m, t_0-115m, ldots, t_0-5m].
]

That is 24 native five-minute bars.

The event bar t0 is excluded.

No post-event bar is used.

The event bar is retained only outside the HMM so the discovered shock types can later be described by shock sign and magnitude.

## 3. Event definition

The shock object returns to the event definition used in the original Module 03-05 discovery sequence:

- per-asset full-sample q99 of native five-minute variance;
- consecutive q99 bars reduced to run onsets.

This is intentional: Module 07 is a retrospective taxonomy of the already-discovered shock population, not a prospective threshold-estimation test.

Each asset-level event is retained separately.

## 4. Features

Ivan requested that Module 07 use the features that formed the pattern around -95m.

The source fields are therefore frozen from Module 05:

### Activity
- intrabar range;
- USD volume;
- trade count.

### IV level
- ATM forward IV;
- 25d put IV;
- 25d call IV.

### Liquidity / book activity
- book spread;
- book update count.

### L2 instability
- OBI raw zero-cross count;
- OBI raw zero-cross rate;
- OBI transition count;
- OBI path roughness.

No new full-schema scan is performed.

## 5. Why family emissions are used

Several Module 05 fields are close representations of the same economic quantity.

A diagonal HMM fed ATM IV plus two wing-IV levels can spend states fitting repeated IV information rather than temporal structure.

The same issue exists among L2 zero-cross and transition measures.

Module 07 therefore:

1. applies monotone log/log1p transforms where appropriate;
2. normalizes every source feature by asset x UTC hour using robust parameters estimated before the validation-shock period;
3. averages normalized fields within the four frozen Module 05 families.

The HMM emission vector is:

[
X_t =
[
Activity_t,
IVLevel_t,
LiquidityBook_t,
L2Instability_t
].
]

Missing L2 information is marginalized by the Gaussian emission likelihood.

## 6. Synchronized shock weighting

A market-wide shock may produce q99 onsets in all seven assets at the same timestamp.

The asset paths are scientifically useful and remain separate.

But they are statistically dependent.

Each event therefore receives sequence weight:

[
w_e = rac{1}{N_{assets}(t_e)}.
]

Thus every unique q99 timestamp has total HMM fitting weight one.

The same weights are used in shock-trajectory clustering.

## 7. Chronological discovery and validation

Unique q99 timestamps are sorted chronologically.

The default split is 70% discovery / 30% validation.

All asset-events sharing one timestamp stay on the same side of the split.

Asset x UTC-hour normalization is fitted on panel rows before the first validation shock timestamp.

The HMM is fit only on discovery shock windows.

Validation sequence likelihood is a diagnostic and does not refit the HMM.

## 8. HMM states are phases, not shock types

A selected HMM with K states describes recurring phases within the 24-bar pre-shock windows.

It does not directly classify shock instances.

For every event, Module 07 computes the smoothed posterior state trajectory using only its -120m to -5m sequence.

This is retrospective taxonomy, so smoothing within the pre-shock window is legitimate: by t0-5m all 24 observations already exist, and neither t0 nor future bars enter.

## 9. Shock trajectory embedding

Each event is represented by:

- average posterior state occupancy in six consecutive 20-minute blocks;
- exact posterior state probabilities at -95m;
- terminal posterior state probabilities at -5m.

If the HMM has K states, the event embedding therefore has 8K dimensions.

The -95m point is preserved explicitly because it motivated the experiment.

## 10. Shock types

Discovery embeddings are clustered by weighted k-means.

Candidate shock-type counts are:

[
Cin\{2,3,4\}.
]

Selection uses weighted silhouette with a minimum effective support of five unique-timestamp equivalents per type.

Validation shocks are assigned to the nearest frozen discovery centroid.

Validation does not refit the type centroids.

## 11. Post-hoc descriptors

The following variables are forbidden from HMM fitting and shock-type clustering:

- t0 return sign;
- t0 variance and q99-threshold multiple;
- number of simultaneous q99 assets;
- whether another q99 bar occurred in the preceding 120m;
- time since the previous q99 bar;
- asset identity;
- event date;
- L2 collector era.

They are inspected only after the unsupervised taxonomy exists.

This lets the data answer questions such as:

- Is one type mostly aftershocks?
- Is one type disproportionately synchronized?
- Do types differ in q99 magnitude or sign?
- Is a type actually just one asset or one date?

## 12. Collector sensitivity

06b raised evidence that `book_update_count` changed measurement regime around the October 4 collector update.

Module 07 therefore performs a full sensitivity rerun where the liquidity/book family contains spread only.

Primary and sensitivity event assignments are compared with Adjusted Rand Index.

A taxonomy that disappears under this change should not be interpreted economically.

## 13. What would constitute an interesting result

A positive exploratory result requires more than a good discovery silhouette.

We would want:

1. at least two non-trivial shock types;
2. clearly different observed family trajectories;
3. clearly different HMM posterior-state paths;
4. non-trivial validation support for the same frozen types;
5. tolerable validation centroid drift;
6. no type dominated by one asset, date, collector era, or aftershock status;
7. meaningful agreement with the spread-only sensitivity.

If these fail, shock heterogeneity is not yet stable.

## 14. Active files

Notebook:

`notebooks/Volatility_Dynamics/07_q99_preshock_hmm_taxonomy.ipynb`

Helper:

`src/data_engineering/modeling/volatility_q99_preshock_hmm.py`

Expected outputs:

- `07/07_run_metadata.json`
- `07/07_feature_contract.csv`
- `07/07_q99_event_catalog_all.csv`
- `07/07_q99_event_catalog_usable.csv`
- `07/07_window_audit.csv`
- `07/07_source_hour_normalizer.csv`
- `07/07_family_window_coverage.csv`
- `07/07_hmm_model_comparison.csv`
- `07/07_hmm_emissions.csv`
- `07/07_hmm_transitions.csv`
- `07/07_event_state_paths.csv`
- `07/07_event_embeddings.csv`
- `07/07_shock_type_model_comparison.csv`
- `07/07_shock_assignments.csv`
- `07/07_shock_type_family_profiles.csv`
- `07/07_shock_type_state_profiles.csv`
- `07/07_shock_type_descriptors.csv`
- `07/07_shock_type_asset_composition.csv`
- `07/07_shock_type_date_composition.csv`
- `07/07_validation_stability.csv`
- `07/07_sensitivity_hmm_comparison.csv`
- `07/07_sensitivity_shock_type_comparison.csv`
- `07/07_collector_sensitivity.csv`

## 15. Interpretation boundary

Module 07 can discover a candidate taxonomy of q99 precursor trajectories.

It cannot establish:

- that the types predict whether q99 will occur;
- that the HMM state count is a market-regime count;
- that any type is an economic regime;
- that the taxonomy has trading value.

Those are separate questions.
