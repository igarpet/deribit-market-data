# Liquidity Absorption 005 — A02b touch-specific flow and dwell features

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-20 |
| **Branch** | Liquidity_Absorption |
| **Responds to** | `004_chatgpt_a02_results_review.md`, user approval of Stage A02b |
| **Status** | Implemented exploratory feature enrichment; **no P&L, no selection based on outcomes** |

## Purpose and source

A02 selected 452 price-only retest events (BTCUSDT 212, ETHUSDT 240) under exploratory settings. In A02, 165 of the 452 first-touch candles occurred at least ten minutes before the second rejection, outside the preceding 10-minute rolling profile comparison. A02b corrects that *measurement mismatch* without changing signal membership, first/second touch times or source levels. The input is `02/02_price_only_candidates.csv`, not the prior flow-selected enriched file. Source version is validated to be `LA-A02-0.1.0`.

The model uses Binance USD-M perpetual aggTrades (`time_ms`, actual `price`, `qty`, and `is_buyer_maker`) and the existing Binance USD-M 1-minute OHLC plus taker-buy quote-volume fields. It does **not** observe historical passive resting book, hidden order IDs, maker queue position, or exchange-wide liquidation stops.

## Protocol, with all intervals half-open

For one A02 event let the first candle **close** at \(t_1\), the second close/signal at \(t_2\), the pre-established level \(P\), and the A02 halfwidth \(w\):

- First-touch candle: \([t_1-1\text{min},t_1)\).
- Second-touch candle: \([t_2-1\text{min},t_2)\).
- Between-touch encounter: \([t_1-1\text{min},t_2)\), including both touches. Candidate A02 event times are unchanged.
- Historical profile pre-first: \([t_1-31\text{min},t_1-1\text{min})\). The 30-minute length is exploratory, not tuned to downstream returns.
- Executions contribute zone quote volume iff their **actual print prices** satisfy \(P-w\leq p\leq P+w\), inclusive. No midpoint-bin approximation enters first/second-touch volume attribution.
- Buy aggressor: `is_buyer_maker == False`, sell aggressor: `True`. Signed quote delta is buyer-initiated quote volume minus seller-initiated quote volume, in USDT.
- Opposing aggressive flow for a **long support** candidate is sell-initiated zone quote volume; for a **short resistance** candidate it is buy-initiated zone quote volume.

Features per candidate include both touch-candle near-zone total, buy, sell, signed and opposing quote volumes; relative near-zone shares of each candle's total executed quote flow; touch-to-touch differences and a growth ratio (undefined if first touch volume=0); cumulative all-market quote delta across the actual encounter; and first/second candle return and first-to-second closing displacement in basis points.

### Separating time away from the level

To avoid mistaking elapsed time for volume accumulation, the **zone visiting** proxy counts 1-minute bars from the first touch through the second whose OHLC ranges intersect \([P-w,P+w]\). The encounter is split into an early and late half by candle index. We compute exact in-zone executed quote volume in each half, divide by the respective **count of visiting minutes**, and compare the resulting rates. If one half has no visiting minute or the earlier rate is zero, the rate ratio is `NaN`.

The reported distinct `active_print_seconds` measure counts 1-second buckets containing one or more in-zone executions. This is not the true time the market traded or rested at that level, and the distinct-second rate is diagnostic only. Neither volatility nor session-normalization is fitted here; normalized quote-volume shares allow cross-asset comparisons without fabricating a universal crypto tick size.

### Historical volume-profile reference

A **pre-first-touch-only** 30-minute profile is built using 1bp logarithmic *price* bins. The bin with highest historical executed quote volume is the point of control (POC); output its approximate price, distance in bp to the tested A02 level, relative POC-bin volume, and exact pre-first volume share executed inside the candidate zone. Missing pre-first-day files or no historical trades get explicit status flags and null POC values. POC values do **not** modify candidate-event selection: adding a third rolling profile-node **candidate family** would change the population and requires separate approval before a new experiment.

### Missing-data semantics and integrity

The code streams one available Parquet day at a time, reading bounded batches and reducing them directly into event-level counters plus a small 30-minute histogram; it does not create a duplicate multi-month trade archive or load all trades into RAM. UTC-day boundaries are reconciled for events spanning midnight. Missing **touch** day marks touch data as incomplete and masks partial trade features. Missing **pre-first** day is reported separately and masks pre-first POC estimates; it does not invalidate available touch flow.

Timestamp \(t_2\) is an information barrier: each trade read for encounter features must have timestamp **strictly less** than \(t_2\), and the POC profile cannot use a trade after the first candle starts. Candle context uses only complete bars through \(t_2\). The code compares aggregated-trade quote volume for the first and second candles with their kline quote volume as a descriptive source-alignment diagnostic, without choosing a threshold post hoc.

## Implementation and output contract

- Module: `src/data_engineering/modeling/liquidity_absorption_stage_a02b.py` (`LA-A02b-0.1.0`)
- Tests: `tests/test_liquidity_absorption_stage_a02b.py` — synthetic timestamp, future exclusion, near-zone buy/sell sides, half-open boundaries, missing data, first-vs-second intensity and provenance tests
- Notebook: `notebooks/Liquidity_Absorption/02b_touch_flow.ipynb`
- Outputs: `notebooks/Liquidity_Absorption/02b/02b_touch_features.csv`, `02b_coverage.csv`, `02b_feature_contract.json`, `02b_decision.json`

## Interpretation and next decision

Passing tests on invented trades does not validate the actual 92-day Binance execution archive. The owner runs the notebook in the local `deribit-data` conda kernel and pushes its outputs. Stage A02b **cannot estimate trading returns or a statistically significant signal**. In particular, even strong *sell* delta against a price-stable support zone is only **consistent with** passive absorption, not proof of iceberg or parent orders. No fixed 15–20 MNQ tick stop or +10 MNQ point break-even rule is translated mechanically into BTC/ETH dollars.

The exploratory July–September 2026 window is already inspected, and neither A02b nor later A03 results on it may be described as a fresh holdout. For a later frozen test, reserve future data and explicitly specify crypto stop-size mapping, one-minute swing confirmation timestamp, structural target selection, same-minute order-of-events, market-order stop slippage, fees and funding.

## Questions for Claude

1. Is volume per OHLC-intersecting minute sufficiently robust as a *price-visitation proxy*, or should a competing duration proxy be made before Stage A03?
2. What independent control events would best distinguish true effort-vs-result asymmetry from ordinary delta reversals near five-minute pivots?

## Decisions needed from Ivan

1. Run LA A02b, push outputs, and decide whether these encounter-specific features match the discretionary observations sufficiently for A03.
2. Approve or reject an **additional causal volume-profile-derived level family** in a future module. The current 452 price-only A02 reference levels remain unchanged until such approval.
3. Approve a separate A03 executable entry/stop/target protocol before any P&L testing.
