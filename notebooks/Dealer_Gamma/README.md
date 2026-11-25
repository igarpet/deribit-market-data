# Dealer Gamma research branch

**Question:** does option-dealer hedging leave a footprint in BTC/ETH? Dealer gamma is rebuilt from the taker side of every Deribit option print since 2024-01 and tested three ways. Discussions: `../../discussions/Dealer_Gamma/` (pre-registration DG 001). **Closed (DG 003): no hedging footprint in discovery (DG 002); holdout never run.**

| Notebook | Content | Status |
|---|---|---|
| `01_data_gate.ipynb` | Validator summary; reconstructed 24h volume and \|net\| vs the collector's 24h volume and open interest (0DTE/1DTE overlap). **Gate: FAIL stops the branch** | Run 75c7105: data PASS, positioning FAIL (DG 002) |
| `02_build_gex.ipynb` | One pass over all prints: 5-minute dealer gamma (5 variants), perpetual bars, 05:00 strike tables, settlement proxy. Cached outside the repo | Run 75c7105: 1,058 days/currency, no missing files |
| `03_regime.ipynb` | GEX at 08:05 → next-session log RV and AC1 of 5-min returns; HAC; terciles | **FAIL** (DG 002) |
| `04_hedge_flow.ipynb` | Predicted hedge −100·G·r → next-15-min signed perp taker flow; clustered SE; **decides whether hedging is established** | **FAIL**, BTC wrong sign (DG 002) |
| `05_pinning.ipynb` | Settlement vs dealer-long-gamma strike, mirrored placebo; daily / weekly / monthly / quarterly | **FAIL** (DG 002) |

Code: `src/data_engineering/modeling/dealer_gamma.py`. Tests: `tests/test_dealer_gamma.py`. Outputs: `NN/NN_*`.
Periods: burn-in to 2024-06-30, **discovery** 2024-07-01 → 2025-06-30, **holdout** 2025-07-01 → last day (only with `DG_HOLDOUT=1`).

## Running (PC, Windows CMD), in this order

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest tests.test_dealer_gamma -v

REM 1. Bring the trade archive up to date (resumes from the last complete day)
python tools\download_deribit_history_trades.py --currencies BTC ETH --kinds option future --start 2024-01-01

REM 2. Validate the archive (options and futures)
python tools\validate_deribit_history_trades.py --kind option --start 2024-01-01 --out notebooks\Dealer_Gamma\01\validator_option
python tools\validate_deribit_history_trades.py --kind future --start 2024-01-01 --out notebooks\Dealer_Gamma\01\validator_future

REM 3. Pull the latest collector files from the VM
python tools\sync_deribit_data.py

REM 4. Gate. Stop here and report if 01_gate.json says FAIL
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 notebooks\Dealer_Gamma\01_data_gate.ipynb

REM 5. Build (slow, cached in DERIBIT_DATA_ROOT\derived\dealer_gamma)
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=86400 notebooks\Dealer_Gamma\02_build_gex.ipynb

REM 6. Tests, discovery period only
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 notebooks\Dealer_Gamma\03_regime.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 notebooks\Dealer_Gamma\04_hedge_flow.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 notebooks\Dealer_Gamma\05_pinning.ipynb

git add notebooks\Dealer_Gamma
git commit -m "DG 01-05 discovery run"
git push
```

Holdout, only after the discovery readout is posted: `set DG_HOLDOUT=1` and rerun 03–05.

Environment: `DG_CACHE` (cache folder), `DG_FORCE=1` (rebuild the cache), `DG_HOLDOUT=1`, `DG_SYNTHETIC=1` (smoke run on synthetic data, writes to the temp folder only).
