# LA-A05b and Deribit historical futures tape — independent second-pass code audit

**2026-10-20 · Scope:** The current repository implementation and synthetic execution of `tools/validate_deribit_history_trades.py`, `tools/download_deribit_history_trades.py`, `src/data_engineering/modeling/liquidity_absorption_stage_a05b.py`, the A05b notebook, and targeted tests. **No private local 2024–2026 archive was available**; empirical tape completeness and millisecond trading results remain unverified.

## Confirmed current status

1. The previously executed notebook `SyntaxError` in writing `deribit_daily_qc.json` has been corrected in main to append `'\\n'`. Previously saved traceback outputs were removed; the notebook is **not yet re-executed on actual Deribit trades**. A synthetic regression compiles every notebook code cell to prevent reoccurrence.
2. The downloader's timestamp-cursor overflow path no longer silently advances one millisecond after a non-advancing page. It raises `RuntimeError` before committing that incomplete UTC day. **Previous saved Parquet days are not automatically repaired**; the patch only applies to processes started after the updated source is loaded. Mid-run Python download processes have old code until restarted.
3. The A05b event-study previously called `drop_duplicates(['trade_id'])` after sorting each ±60-second window. That is **silent data correction** and hides real tape corruption, so it now raises on any duplicate or null trade ID within a captured window. Distinct legitimate prints at the same millisecond still count separately. A test's fabricated later print reused `trade_id='b0'`; this test fixture was corrected to use a distinct ID.
4. A05b previously allowed no-pre-anchor-print or stale-pre-anchor-print windows into the aggregate phase comparisons if their calendar days existed, despite being unable to calculate a reliable anchored reference price. New grouped medians **require `status=='PRESENT'` and `complete_calendar_days==True`**. All windows, including excluded ones, retain coverage/detail rows so exclusions remain auditable.
5. Existing fixes for readable empty phase summary CSVs, explicit missing calendar day exclusion, fatal invalid future-vs-option schema, and per-instrument sequence checks remain in main.

## Reproduction and evidence

A focused local synthetic suite was executed twice after the new changes:
- `test_liquidity_absorption_stage_a05b.py` → **23/23 passed**.
- `test_deribit_history_validator.py` → **13/13 passed**.
- The A05b engine and QA Python modules compiled, and the existing notebook's code cells were independently syntax-checked with the published JSON newline fix.
- The separately modified downloader test module currently contains **5 tests** in GitHub. Those test cases were reviewed, but unlike the two suites above, **its exact final remote file/test pair was not executed in this local audit environment**. Do not describe 41/41 or all project tests as passed.
- The tests use fake archives and synthetic Deribit prints. They do not validate the user's local historical download.

## Remaining known research and archive limitations

**Completeness:** A pass in a local Parquet scan is not a certificate that the original Deribit API returned every trade. The old timestamp-based downloader could have saved incomplete UTC days after a crowded-ms pagination event. The new fail-closed guard prevents this specific skip in *future* downloads but does not fix past days; an authoritative exchange-side sequence/retrieval reconciliation remains the sounder next quality gate. Deribit documents `trade_seq` per instrument, including occasional natural sequence skips; sequence-span warnings require investigation, not automatic redownload.

**What the notebook actually measures:** The A05b windows center on **one-minute Binance first/second-touch candle-close timestamps** and read **Deribit BTC-PERPETUAL and ETH-PERPETUAL executed prints** from the BTC/ETH `future` trade files. It does *not* compare dated futures yet, detect exact touch milliseconds, reconstruct L2 orders, or prove passive absorption. Empty ms buckets are intentionally not invented.

**Runtime scale:** The downloader still accumulates a whole currency-kind-day list and trade ID set before writing Parquet. This can consume substantial memory for busy perpetual days. A row-group streaming redesign would need its own independent implementation and tests, preferably after the current long-running download finishes; it is not silently introduced during this fix.

**No empirical results:** The notebook's actual event coverage, archive fatal/warning counts and statistical outcomes cannot be asserted until it executes on the user's Windows archive. The user should commit the lightweight validator outputs and executed notebook for a third-stage empirical audit. Large raw trade files or gigabyte sparse-bucket outputs should remain local.

## Runbook (Windows CMD from repository root)

```bat
git pull
python -m unittest discover -s tests -p "test_download_deribit_history_trades.py" -v
python -m unittest discover -s tests -p "test_deribit_history_validator.py" -v
python -m unittest discover -s tests -p "test_liquidity_absorption_stage_a05b.py" -v
jupyter lab
```

After restarting `deribit-data` kernel, execute `notebooks/Liquidity_Absorption/05b_deribit_millisecond_microstructure.ipynb` from the first cell. The validator output is `notebooks/Liquidity_Absorption/05b/validator/deribit_daily_qc.csv` and `deribit_daily_qc.json`; the study outputs `05b_anchor_coverage.csv`, `05b_phase_summary.csv` and `05b_decision.json` should be examined before drawing conclusions. This read-only study does not modify archived trade data or prior LA backtests.
