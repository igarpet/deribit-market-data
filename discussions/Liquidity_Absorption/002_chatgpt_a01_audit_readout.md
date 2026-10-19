# Liquidity Absorption 002 — LA-A01 audit readout and corrective patch

**Date:** 2026-10-20  
**Status:** PARTIALLY VERIFIED — rerun required under LA-A01-0.1.2  
**Scope:** Review the user-executed LA-A01-0.1.1 outputs. No strategy results or profitability claims.

## Findings genuinely supported by committed outputs

- BTCUSDT and ETHUSDT Binance USD-M perpetual minute series each contain **1,067,040** rows from **2024-10-01 00:00 UTC** through **2026-10-11 23:59 UTC**, with zero reported null/duplicate/out-of-order times, internal minute gaps, invalid-price rows and negative-volume rows.
- BTCUSDT and ETHUSDT aggregated-trade archives each have **92 daily files**, **2026-07-01** to **2026-09-30**, with no missing *filenames* inside that interval.
- Six sampled trade days (first/middle/last for each asset) have zero invalid rows, zero null aggressor-side flags and zero descending trade timestamps. All six contain trades in every UTC minute. This sample is *not* proof of completeness for all trade days.
- Reported `spot BAD_SCHEMA` rows arise because the existing `download_binance_basis_data.py` intentionally retains `open_time, close, volume, quote_volume, trades, taker_buy_volume, taker_buy_quote_volume`, without `open/high/low`. The audit incorrectly required full OHLC.
- The flow-comparison CSV contains **six `DUPLICATE_KLINE_MINUTES` statuses**, each with **1,353** false duplicate minute keys and **zero matched minutes**. It therefore contains *no valid flow-consistency evidence*, even though the decision JSON displayed `ready_for_signal_design=true`.

## Root cause and patch

The `parse_kline_open_time` helper correctly interpreted numerical Binance timestamps, but `compare_minute_flow` then used `timestamp.astype("int64") // 1_000_000` to construct epoch-ms keys. On the user's installed pandas version, parsed UTC datetimes retain **millisecond** precision; their raw int64 values are already milliseconds. Dividing them by one million collapsed approximately 1,000 seconds into one join key, giving 87 unique keys in a 1,440-minute day (1,353 duplicates).

**LA-A01-0.1.2** converts UTC timestamps explicitly with `timestamp.dt.as_unit("ms").astype("int64")`, supports the compact spot schema, and includes spot quality (when spot is present) and sampled flow-alignment success in the `ready_for_signal_design` gate. Numerical delta/volume error thresholds remain diagnostic; we did not optimize them after seeing the results.

The local synthetic suite now has **10 passing tests**, including a 1,440-minute `datetime64[ms, UTC]` regression and a decision-gating regression. No local Binance Parquet data were accessible to the assistant, so the **actual corrected flow errors are not known**.

## Required next action

1. `git pull` and restart the `deribit-data` Jupyter kernel.
2. Run all cells in `notebooks/Liquidity_Absorption/01_data_audit.ipynb` under **LA-A01-0.1.2**.
3. Commit and push the replaced `01/*.csv`, `01_decision.json` and executed notebook.
4. Inspect **`01_trade_kline_flow_comparison.csv`**: expect a `PRESENT` status and **1,440 matched minutes** per sampled day if archives agree. Examine median and p95 quote-volume and delta discrepancies before interpreting the aggregate-flow measurement as reliable.
5. Do not treat the prior v0.1.1 `ready_for_signal_design=true` as a valid green light.

Following a successful real-data rerun, freeze the causal feature/event contract for LA-A02. The previously inspected historical sample is exploratory, not prospective confirmation.
