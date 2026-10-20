# Liquidity Absorption 030 — LA09 level availability, first touch, second touch and subsequent reaction

**Author:** ChatGPT · **Date:** 2026-10-21  
**Version:** LA-LIFECYCLE-09-0.1.0  
**Status:** protocol and synthetic-tested implementation; real historical notebook run pending local Binance archive  
**Scope:** strictly descriptive formation/path research, no new entry rule or independent alpha claim.

## Motivation and distinction from earlier modules

A05a showed −240 to +240 minutes around eventual A02 signal close, but marked the eventual level retrospectively in hours when many pivots **were not known**. Claude's A06 and W01 focus on entry-time derivatives and future CVD management. This module returns to the original A02/A03b/A03c research objective: **what does the original two-touch setup look like once its level exists, through its two encounters, and during the following market reaction?** No replacement signal or simulated policy.

**Primary membership:** original A02b 452 price-only candidate IDs and original A03b `FIXED` (one-for-one gross) outcome per ID: 116 TARGET / 112 STOP / 185 TIMEOUT / 37 NO_STRUCTURAL_TARGET / 2 CENSORED_TAPE_END in the previously recorded sample; code validates 452 identities, not hard-coded counts of winners. Primary outcome is not A03d's volatility-protected policy. A03c `MAKER_TOUCH_3RD` is joined **only** as separate hypothetical queue-eligibility metadata: never treated as realized fills.

## Recovered *causal* level clock from frozen A02 code

- For 5-minute pivots, A02 `get_reference_levels` detects two prior and two subsequent five-minute extremes, and sets `available_at = pivot_bar_close + 2 × 5 minutes`. `level_source_time` is the **pivot bar close**, not when the strategy could see the pivot. The **10-minute delay is mandatory**. The level can be used only at and after this confirmed availability time.
- For `prior_day` extrema, `level_source_time` is the previous day UTC midnight and the completed prior UTC day becomes available at **next UTC midnight** (`source_time + 24h`). This is a completed-day reference, not the opening day's high/low known in advance.
- Compare `available_at ≤ first_touch_close < second_touch_close = signal_time_utc`, all strict minute-close units. Rejection level visibility is never retrojected into the earlier context. The first or second *intraminute* price touch cannot be timestamped more exactly from 1m OHLC; the event labels refer to each candle **close**, not a particular transaction.
- Historical original 452 A02b candidates have already been metadata-audited: **373 pivot_5m** and **79 prior_day**, with **no first touches earlier than calculated availability**; **16 have first-touch close exactly equal to pivot availability**, so the touch may have occurred before pivot confirmation within that minute and is flagged as not independently actionable. Median `available → first` about 16 minutes, median `first → second` about 7 minutes. These descriptive figures came from GitHub source CSVs, not the new module's execution.

## Frozen stage taxonomy

The eight **non-overlapping** market phases use exact completed one-minute candle times `(start,end]` (no forward fills) and are interpreted as follows:

1. `PRE_SOURCE_60`: 60 completed minutes ending at the source reference time. The eventual selected level is **not available**; level-specific features are intentionally NaN.
2. `SOURCE_TO_AVAILABLE`: after the source stamp through the eventual level-availability close: for pivots the **10-minute confirmation interval**, and for prior-day extremes the **entire previous calendar day**. Level-specific features are intentionally NaN before confirmation. This is NOT a homogeneous microstructural phase across families; keep families separate.
3. `AVAILABLE_TO_FIRST`: completed bars strictly after availability through and including the first rejection candle close. The entire stage follows observable level availability.
4. `FIRST_TO_SECOND`: first rejection close exclusive, second rejection/signal close inclusive. The earliest moment the two-touch rule is actionable is the final close.
5. `AFTER_SIGNAL_1_5`: +1 to +5 completed minutes.
6. `AFTER_SIGNAL_6_15`: +6 to +15 completed minutes.
7. `AFTER_SIGNAL_16_60`: +16 to +60 completed minutes.
8. `AFTER_SIGNAL_61_240`: +61 to +240 completed minutes.

For each phase, report exact expected/observed minute counts and a complete-window flag; incomplete stages contribute no medians. Metrics: signed OHLC price movement, taker-buy quote-volume imbalance aligned to original side, mean quote turnover, quote activity vs trailing 60 complete candles, average absolute one-minute log return, and zone overlap / favorable-side closing fractions **only when the level was observable**. OHLC overlap is not tick-by-tick visit duration or L2 absorption.

Separately align fixed `−15..+30` completed minutes around **level source**, **availability**, **first touch**, and **second touch**; report aligned price paths, completed-minute aggressive flow proxy, and an explicit `level_known` field. Any calculated level distance before availability is NaN. Events exited before later bars still contribute market price observations but are marked `post_exit_market_only`; **not** continuing P&L or a live risk-set.

## Source files and outputs

Read-only inputs: `notebooks/Liquidity_Absorption/02/02_feature_contract.json`, `02b/02b_touch_features.csv`, `03b/03b_1to1_events.csv`, `03c/03c_events.csv`, and the local `BTCUSDT_um_1m.parquet` and `ETHUSDT_um_1m.parquet` specified by `BINANCE_UM_DIR`. Each Binance file is loaded once, timestamps converted explicitly to epoch milliseconds to prevent the A04 pandas precision bug. No Deribit tick archive, regenerated signals, 03c maker fills, or new collectors are required. A future option can rerun this specific *descriptive timeline* on Claude's EXT cohort after independent identity and venue contracts are applied; do not silently mix original Binance and extended Deribit fills.

Outputs under `notebooks/Liquidity_Absorption/09/`:

- `09_decision.json` — provenance, cohort cardinality, missing-window accounting.
- `09_cohort_timing.csv` — level source, availability lag, first/second touch intervals, outcome, maker annotation.
- `09_lifecycle_stages.csv`, `09_lifecycle_summary.csv` — original ID × fixed phase and outcome-conditioned medians.
- `09_landmark_minute_panel.csv.gz`, `09_landmark_summary.csv` — event × source/level-available/first/second milestone × −15..+30 minute observations and group medians.
- `09_milestone_price_paths.png`, `09_lifecycle_phase_changes.png` — per-asset examples of group median trajectories and phase changes, not expected payoffs.

## Limitations and next scientific design

- **Event selection / controls:** these 452 IDs were selected *because* they ultimately received a second touch, so this cohort cannot answer `P(second touch | newly available level)` or show which levels never get touched. A meaningful control universe must be generated at level **availability** and retain untouched / single-touched / invalidated levels without selecting on later outcomes. That is the next module if lifecycle signatures look coherent.
- Original Jul–Sep 2026 and Claude EXT 2024–2026 histories have been repeatedly inspected. Grouped TARGET vs STOP is descriptive; there is **no untouched holdout** in this dataset.
- Mean per-minute aggressive trade imbalance is an observable quote-volume proxy, not proof of passive replenishment. Same-candle order of touch, rejection and any stop barrier cannot be established from OHLC.
- Strong A03b target-distance/volatility confounding still exists; early phase patterns must be compared within exact symbol, side, level family, target geometry and time regime before any usable filter is claimed.
- A03b assumes taker print fills, adverse execution offsets and symmetric 1:1 stop, with no funding/book/capital coupling. A03c passive eligibility is not filled-order evidence.

## Execution

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_liquidity_absorption_stage_a09.py" -v
jupyter lab
```

Open `notebooks/Liquidity_Absorption/09_level_lifecycle_anatomy.ipynb`, choose **Restart Kernel & Run All**, then commit the executed notebook and `09/` outputs to GitHub `main` for review. This module does not modify earlier results.
