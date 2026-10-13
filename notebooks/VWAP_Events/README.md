# VWAP Events notebooks

Pre-registration: `discussions/VWAP_Events/001_claude_branch_preregistration.md`.

| Notebook | Content | Status |
|---|---|---|
| `01_vwap_touch_event_study.ipynb` | Real VWAP touches (08:00 UTC session, 4h) vs offset and random-phase placebo touches; time-of-day-normalised realised volatility after the touch; Bonferroni gates on the test window | Built, synthetic-validated, awaiting first real run |

## Data

Binance USD-M perpetual 1-minute klines, one parquet per symbol (`<ASSET>USDT_um_1m.parquet`), from `tools/download_binance_um_klines.py`. Default location: `<parent of DERIBIT_DATA_ROOT>/Binance`, or `BINANCE_UM_DIR`.

## Run (PC, Windows CMD)

```
python tools\download_binance_um_klines.py --months 24
jupyter nbconvert --to notebook --execute --inplace notebooks\VWAP_Events\01_vwap_touch_event_study.ipynb
```

Environment: `VE01_ASSETS` (comma list), `VE01_N_PLACEBO` (default 20), `VE01_SYNTHETIC=1` (smoke run to a temp dir). Assets without a parquet file are skipped.

Outputs go to `01/`: coverage, event counts, primary readout, decision, horizons, per-asset test readout, direction, and the manifest. All are small CSV/JSON.
