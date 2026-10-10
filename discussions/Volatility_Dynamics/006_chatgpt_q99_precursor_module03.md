# 006 — ChatGPT: q99 precursor discovery Module 03

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-10 |
| **Branch** | Volatility Dynamics |
| **Responds to** | Ivan's request to inspect all collected market information before and after native five-minute q99 volatility events |
| **Status** | Implemented; awaiting real-data run |

## 1. Research question

The module asks only:

> Using information observable before a q99 five-minute volatility onset, is anything in the collected market state repeatedly unusual relative to comparable non-event moments?

No attempt is made to classify the event as US-open, market-wide, idiosyncratic, news-driven, MOM, MR, Bridge, or any other economic type.

## 2. Event definition

Module 03 does not read Module 02 output files.

For each asset it recomputes:

[
v_t=r_t^2
]

and:

[
c_i=Q_{0.99}(v_{i,t}).
]

A q99 bar is:

[
v_{i,t}ge c_i.
]

If several q99 bars are consecutive, only the first is an event onset.

Thus the research target remains the native five-minute extreme-volatility event, but the module is reproducible without an output dependency on Module 02.

## 3. Predictor discovery

No feature family is hand-selected.

The notebook executes:

`DESCRIBE time_series_feature_state`

and includes every runtime-discovered numeric or boolean column except explicit identifiers. `return_5m` is also retained as a valid predictor **only when lagged**.

String/categorical fields remain in the inventory but are not quantitatively ranked in this first precursor scan.

This means the scan can include, wherever the common database contains them:

- price/return/range quantities;
- volume, trade count and flow quantities;
- perpetual/index/basis quantities;
- open-interest quantities;
- spread/depth/OBI/microprice and instability measurements;
- option IV, skew, curvature, surface-change and option-liquidity quantities;
- boolean readiness/quality/integrity flags;
- other numeric fields added to the common feature table.

No ranking or selection from Pairwise Regression, HSMM, Volatility Regression, VRP, or another branch enters.

## 4. Hard information boundary

For an event at (t_0):

[
t<t_0
]

is precursor information.

[
t=t_0
]

and:

[
t>t_0
]

are anatomy only.

The precursor ranking uses only:

[
-60,-30,-15,-10,-5	ext{ minutes}.
]

The complete anatomy profile is retained from -60m through +60m.

## 5. Controls

The event study should not rediscover ordinary intraday seasonality as a precursor.

For each q99 event, controls are therefore selected from:

- the same asset;
- the exact same five-minute UTC clock slot;
- non-q99 origins;
- origins whose immediately previous exact bar was not q99.

Among eligible controls, the nearest dates are selected first, up to eight by default.

The matching rule does not classify or explain the event. Clock is only a nuisance condition.

## 6. Unitless precursor score

For each event, feature and relative bar, the event feature value is ranked against its matched-control values:

[
P_{e,f,k}
=
	ext{percentile of }X^{event}_{e,f,k}
	ext{ among matched controls}.
]

Interpretation:

- (0.5): typical;
- near (1): unusually high before the event;
- near (0): unusually low before the event.

A secondary robust effect is:

[
D_{e,f,k}
=
rac{X^{event}-Median(X^{controls})}
{IQR(X^{controls})}.
]

No imputation is used.

## 7. Dependence handling

Many q99 events can occur at the same exact timestamp across assets.

Ivan does not want these events split into economic categories, but statistically they are dependent.

Therefore, before the pooled ranking, all asset-event scores sharing the same timestamp are averaged.

One timestamp receives one unit of weight rather than up to seven.

## 8. Feature-fishing guard

The discovery ranking evaluates pre-event lags:

[
-60,-30,-15,-10,-5m.
]

A feature is not scored only by whichever one lag looks strongest.

The ranking also records:

- direction consistency across assets;
- direction consistency across the five candidate lags;
- average absolute deviation across those lags.

The exploratory score is approximately:

[
|ar P_{best}-0.5|
	imes
AssetConsistency
	imes
LagConsistency.
]

This is a prioritization device only.

It is **not a p-value** and does not establish forecasting ability.

## 9. Outputs

Active notebook:

`notebooks/Volatility_Dynamics/03_q99_precursor_discovery.ipynb`

Standalone helper:

`src/data_engineering/modeling/volatility_q99_precursors.py`

Expected compact outputs:

- `03/03_run_metadata.json`
- `03/03_feature_inventory.csv`
- `03/03_feature_coverage.csv`
- `03/03_q99_thresholds_and_counts.csv`
- `03/03_q99_event_origins.csv`
- `03/03_matched_controls.csv`
- `03/03_feature_event_profiles.csv`
- `03/03_precursor_by_asset.csv`
- `03/03_precursor_rankings.csv`
- `03/03_top_candidate_event_scores.csv`

The full event×feature×lag score matrix remains in memory and is not committed by default.

## 10. What a positive result would mean

A feature whose percentile trajectory rises before (t_0), remains directionally consistent across assets/events, and differs from same-clock controls becomes a **candidate precursor**.

It does not yet become a predictor.

The next step would freeze the candidate(s) and test:

[
P(q99_{t+h}mid X_t)
]

chronologically against non-event origins.

## 11. What a negative result would mean

If no field shows coherent pre-event behavior, that is also useful evidence:

> With the market information currently collected, native q99 volatility events may not have a stable detectable precursor at the five-minute resolution.

That outcome must be accepted rather than forcing a regime story.

## 12. Questions for Claude after the run

1. Do the leading candidates remain coherent across assets or are they driven by one instrument?
2. Are the candidate paths gradual into the event or isolated at only -5m?
3. Are option variables leading realized volatility or merely reacting in the final bar before q99?
4. Do flow/liquidity variables provide information beyond lagged return/volatility quantities?
5. Which candidates, if any, deserve a frozen chronological predictive test?
