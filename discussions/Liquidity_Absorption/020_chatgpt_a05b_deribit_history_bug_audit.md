# Deribit trade-history downloader, validator and LA-A05b: bug audit and repairs

**Date:** 2026-10-20  
**Scope:** `tools/download_deribit_history_trades.py`, `tools/validate_deribit_history_trades.py`, `src/data_engineering/modeling/liquidity_absorption_stage_a05b.py`, `notebooks/Liquidity_Absorption/05b_deribit_millisecond_microstructure.ipynb`, their three regression test modules, and runbook.  
**Status:** Source-level and synthetic audit; **not a validation of the user's private 2024–2026 Parquet bytes**, which are on the user's computer only. No collector/backtest history or raw archived Parquet files were edited.

## Confirmed defects and applied changes

| Severity | Confirmed defect | Correction | Regression coverage |
|---|---|---|---|
| Blocking | Executed A05b notebook cell 3 contains invalid `json.dumps(...) + ,encoding=...` and raises `SyntaxError` before the historical validator runs | Replace with explicit newline string `+ '\\n'`, commit corrected notebook, clear stale traceback | New `test_notebook_every_code_cell_compiles` compiles **all** notebook code cells |
| High, historical data integrity | Downloader timestamp-only paging did `s += 1` on non-advancing full pages. If 1,000 or more rows fall in the cursor millisecond, this **can silently omit later prints** but still write the day's Parquet, causing future reruns to skip it as complete | Fail closed with an explicit `RuntimeError` and **no new day file** instead of advancing over an ambiguous millisecond; preserve existing completed files. Do not automatically re-download old files | Two downloader regression tests: crowded ms must raise; exception must not publish an incomplete UTC daily parquet |
| Blocking in empty / incomplete sample | A05b produced zero-column CSV files when no prints or usable phases were available, causing `pd.read_csv` to raise `EmptyDataError` | Create schema-bearing empty `05b_phase_summary.csv` and `05b_event_phases.csv` | End-to-end synthetic zero-print archive test reads both CSVs |
| High, interpretation | A05b's injection path and completeness accounting could call a trade window complete even when a necessary adjacent UTC day is reported as missing | Track all required UTC days per event, and gate phase summary membership on successful scans for **every** relevant day | Synthetic midnight-crossing missing-file regression |
| Medium, mislabeled QA | Unexpected option instrument in a `kind=future` daily archive was counted but did not force validator failure | Treat the mismatch as `FAIL` | Future-kind archive containing an option fixture fails |
| Medium, untested grouping | Frozen near/far target-feasibility groups added after the initial suite had no regression on the published aggregate grouping | Assert near/far/ALL phase summary counts | New regression test |
| Test runner gap | Direct execution of `test_liquidity_absorption_stage_a05b.py` invoked `unittest.main()` above later test classes, so direct-script runs skipped them | Move the script guard to the file end | Confirmed 20 A05b tests run both by discovery and direct script |

## Test evidence and unverified checks

**33 local focused regression tests passed** for A05b plus the Deribit validator, including an end-to-end **synthetic** archive scan: 20 A05b tests and 13 validator tests. The GitHub downloader suite has **five** tests after the pagination fix; its exact remote changed script/test pair was reviewed but could not be executed against the user's local archive from this environment. Notebook cells were cleared of error outputs; the corrected historical notebook still needs a fresh local run.

The tests **cannot establish real history completeness**. In particular, the downloader previously saved days that may have advanced over ambiguous same-millisecond pages; the new safeguard applies only to future runs. The validator's per-instrument `trade_seq` gaps, same-ms concentration, cross-day bounds, and missing days are useful **audit indicators**, but absence of flags is not an independent record-by-record reconciliation with Deribit.

The archive download may still be running in the user's Python process. Updating GitHub source does **not** alter code already loaded in that process. Do not terminate an hours-long download just to pick up the patch; finish it and audit results first. Existing Parquet days are not altered by these changes. Redownload only specific dates supported by a concrete anomaly review.

## Remaining engineering / research risks

1. **Memory and throughput:** downloader's `paginate_day` currently accumulates all trades and a `seen` set for a day before writing Parquet. Very busy days may use substantial RAM. A future independent streaming-row-group refactor would need a separate tested migration; no attempt was made to change the active collector's file format while a download is ongoing.
2. **Remote completeness:** per-instrument historical trade sequence can occasionally skip numbers; `WARN` does not automatically mean the downloaded data are corrupt. Timestamp-cursor pagination has an unresolvable within-millisecond boundary without a stronger exchange cursor or independent replay reconciliation. Treat `PASS_WITH_LIMITATIONS` exactly as labeled.
3. **No exact Binance touch millisecond:** first-touch/signal anchors are **Binance candle-close times** from A05a. A05b measures independent **Deribit perpetual trade prints** centered on those times. It does not reconstruct the exact Binance touch, resting quotes, or order-book defense.
4. **Perps vs dated futures:** the downloaded `kind=future` day mixes these instruments; A05b intentionally filters **BTC-PERPETUAL and ETH-PERPETUAL**. Both have ms timestamp precision. Including dated futures requires an independent, per-instrument/front-expiry design, not pooling trade prices across expiries.
5. **Historical empirical outputs pending:** no real daily archive was accessible for this audit; evaluate day-level QA summaries, coverage, sparse-bucket results, event plots, and any new tracebacks after the user reruns.

## Reproduction from repository root

```bat
git pull
python -m unittest discover -s tests -p "test_download_deribit_history_trades.py" -v
python -m unittest discover -s tests -p "test_deribit_history_validator.py" -v
python -m unittest discover -s tests -p "test_liquidity_absorption_stage_a05b.py" -v
jupyter lab
```

Restart kernel and run `notebooks/Liquidity_Absorption/05b_deribit_millisecond_microstructure.ipynb` from its first code cell. Do not treat a successful synthetic run as a certificate that downloaded 2024–2026 trade history is complete.
