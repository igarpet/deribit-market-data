# 019 — Deribit History Trade Tape Validator & LA-A05b Millisecond Microstructure Study

**Author:** ChatGPT · **Date:** 2026-10-20 · **Status:** IMPLEMENTED, UNIT-TESTED, **historical market results NOT YET RUN** · **Scope:** read-only archival QA + cross-venue descriptive research. No strategy optimization or collection changes.

## 1. Why two separate components?

The file `tools/download_deribit_history_trades.py` writes UTC-calendar-day Parquet files:

`<DERIBIT_DATA_ROOT>/history/trades/{BTC,ETH}/future/YYYY/YYYY-MM-DD.parquet`

Its `kind=future` means **dated futures and perpetual futures are mixed in a currency/day file**. Each row represents an actual exchange-reported trade (not a one-millisecond OHLC bar) with an exchange timestamp **quantized to milliseconds**, `trade_seq` per instrument, `trade_id`, `instrument_name`, `direction`, `price`, `amount`, etc. Several actual executions can share an identical millisecond. Historical order book / best bid-ask / queue data are **not downloaded**. Files download via `*.tmp` -> atomic replace `.parquet` after a UTC day is complete.

**Validator:** `tools/validate_deribit_history_trades.py` scans Parquet **in batches** without modifying or redownloading. The local QA identifies missing/in-progress daily files, schema failures, out-of-day and out-of-order timestamps, numeric/side validity, instrument mix, optional exact duplicate `trade_id`, suspicious per-instrument `trade_seq` span gaps, daily instrument coverage, and buckets with >=1,000 prints in **one millisecond** across the currency file. It also reports adjacent-day instrument sequence-span differences. CSV summary and detailed JSON are written separately.

**Hard caveat:** The downloader paginates its `get_last_trades_by_currency_and_time` requests with `count=1000` using timestamp-inclusive cursors and trade-id de-duplication. The downloader **may miss executions** when more than a page of trades share a millisecond. The validator's `max_trades_same_ms_currency` and `millisecond_page_limit_buckets` are **risk heuristics**, not a rigorous completeness certificate: even a clean file with no suspicious bucket can have missing rows. Sequence-span gaps are useful but not a formal guarantee; block/combo behavior and exchange sequence conventions should be checked. A conservative `PASS_WITH_LIMITATIONS` **does not** mean complete remote data. For absolute assurance, separate instrument `trade_seq` source reconciliation or controlled overlapping historical API audit is necessary; this module does not redownload anything automatically.

**Cost:** Full January 2024–October 2026 audit must scan all local future trade rows. It can be CPU/disk expensive but uses bounded Parquet batches and one daily file at a time. The optional `--exact-ids` creates a Python set of all IDs in one daily file and increases memory substantially; use only when acceptable. No full tick copy is generated.

## 2. Fixed A05b design

Source candidate universe is read only from `notebooks/Liquidity_Absorption/05a/05a_cohort_metadata.csv` (452 original A02 candidates: TARGET/STOP/TIMEOUT plus no-target and censored). Retain the original side and labels, no rescore/refit. Two exact **UTC minute-candle-close anchors** per original signal:

1. `FIRST_TOUCH` = `first_touch_close` from A02b/A05a.
2. `SIGNAL` = completed second-touch candle close = `signal_time_utc`.

**Critical timing distinction:** These two timestamps identify **Binance one-minute candle closes**, not exact within-candle moments of first/second interaction; the microstudy cannot claim it identifies the millisecond at which an order first interacted with the level. The Deribit venue has a different order book and executions; do **not** compare the absolute `level_price` of Binance USDT-M futures to Deribit price. The study examines contemporaneous Deribit market reaction around the same *UTC* anchors, not literal Binance liquidity-level prints or execution fills.

Only **`BTC-PERPETUAL` and `ETH-PERPETUAL`** from the corresponding Deribit currency/future archive files are matched, using ±60 seconds around each anchor. An UTC-day boundary is handled by loading both days. Files are scanned in Parquet batches and **only short event neighborhoods** are accumulated. Event-neighborhood aggregation uses `trade_seq` for ordering trades within an exchange timestamp; there is no claim of nanosecond ordering or order book reconstruction. Every 1ms/10ms/100ms/1000ms output row represents a bucket with **at least one actual execution**. Empty milliseconds are not fabricated or treated as observed price updates.

Metrics for each anchor, instrument and eventual outcome:

- **Sparse OHLC executed-trade price** with trade counts in each occupied bucket, direction-adjusted markout (bp) relative to the most recent trade at or before anchor when <=30s stale, min/max price;
- **Aggressor count imbalance**, directional sign and Deribit-reported amount imbalance (do not assume `amount` universally equals USDT notional);
- Six frozen phase summaries: [-60,-10), [-10,-1), [-1,0), [0,1), [1,10), [10,60) seconds. Median event phase returns, trade arrival rate, count/amount alignment are grouped by eventual outcome, BTC/ETH, first versus second anchor;
- Earliest-candidate-ID **illustrative sparse 1ms** plots ±2s around SIGNAL; not a representative or cherry-picked exemplar, and absence of ticks is visible as gaps;
- Explicit calendar-day availability, zero prints and stale reference statuses. A missing neighboring day marks an anchor as untrustworthy; such cases are excluded from grouped summaries.

**Outputs** under `notebooks/Liquidity_Absorption/05b/`:

- `validator/deribit_daily_qc.csv` and `validator/deribit_daily_qc.json` (separate validator; event-date sample or entire period);
- `05b_sparse_trade_buckets.csv.gz` (only occupied bins, all four granularities; may be large), `05b_event_phases.csv`, `05b_phase_summary.csv`, `05b_anchor_coverage.csv`, `05b_day_scan_status.csv`, `05b_decision.json`, `05b_first_touch_millisecond_phase_reactions.png`, `05b_signal_millisecond_phase_reactions.png`, `05b_signal_sparse_1ms_examples.png`.

## 3. Research questions, and what conclusions are **not** supported

1. Are taker-buy/taker-sell execution imbalances, burst rates, and executed-price changes similar around first and second touch *timestamps*? Are post-signal bursts different among eventual TARGET, STOP, TIMEOUT cohorts?
2. Does a 1s or 10s reaction show patterns that one-minute candle OHLC previously hid, and are there periods with **no trades**? Does Deribit's perpetual market consistently react in the same direction as the Binance signal?
3. Is there a systematic *cross-venue* delay around the signal? **A05b does not estimate a latency lead/lag by default:** exchange timestamps, sampling, order arrival delays and clock accuracy need a dedicated design.

**Do not claim:** historical Deribit perpetual execution is a Binance futures fill; Deribit's market level is the Binance defended level; the signal was known at the first touch; the size or direction of trades reveals resting orders or cancellation; adverse/positive returns observed after the event are executable edge; an observed retrospective outcome-conditioned difference is predictive or causal.

Before fitting any next strategy filter, separately test an ex-ante baseline controlling for initial target/stop geometry, volatility, asset, side, level type, and chronology. Use later untouched dates for confirmation. This module changes **no trading logic, stops, signal detector, earlier notebook outcomes, or collection code**.

## 4. Runbook (Windows `deribit-data` environment)

From repository root:

```bat
python -m unittest discover -s tests -p "test_deribit_history_validator.py" -v
python -m unittest discover -s tests -p "test_liquidity_absorption_stage_a05b.py" -v

REM Example event-study date range (90 days per currency; large files):
python tools\validate_deribit_history_trades.py --currencies BTC ETH --kind future --start 2026-07-01 --end 2026-09-30
```

**IMPORTANT:** The actual validator uses singular `--kind future`, not downloader's plural `--kinds`. Use:

```bat
python tools\validate_deribit_history_trades.py --currencies BTC ETH --kind future --start 2026-07-01 --end 2026-09-30 --out notebooks\Liquidity_Absorption\05b\validator
```

For your entire completed historical range (file source begins Jan 2024), set `--start 2024-01-01 --end 2026-10-19` explicitly or later verified date. The downloader by default stops at **yesterday**, not today's incomplete UTC day. **Do not run a destructive re-download on a daily integrity warning.** Notebook `05b_deribit_millisecond_microstructure.ipynb` runs focused sample validation and the corresponding event study, and writes its own outputs. Both scripts accept explicitly configured roots if `.env` paths do not match.

**Caveat about archival storage:** `05b_sparse_trade_buckets.csv.gz` may still be very large if the Deribit perpetual tape is dense; compression of occupied bucket summaries reduces cost substantially compared with re-storing raw multi-year trades. If GitHub rejects the result file due to size, commit the smaller summaries and provenance, leaving the raw sparse buckets local or in appropriate data storage. Never bypass repo size limits by silently dropping rows.
