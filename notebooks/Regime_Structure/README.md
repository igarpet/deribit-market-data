# Regime Structure notebooks

Pre-registration: `discussions/Regime_Structure/001_claude_branch_preregistration.md`.

| Notebook | Content | Status |
|---|---|---|
| `01_regime_existence.ipynb` | Hourly SHOCK / DISP / NOISE / RANGE labels; persistence and transition lifts vs a sign-randomised null; gates on the test window | v1.2 run 6854f48 (binding, RS 004): no persistent regime states 15m–4h; C3 up-DISP→SHOCK passes (1h, 15m); 4h E2 passes test only |
| `02_c3_options.ipynb` | C3 signal: Part A normalised move after signal vs hour-matched times (60 min, to settlement); Part B executable long ATM 0DTE straddle / call / 60-min exit on collector quotes (VRP 02 code, approved) | Run 9c21ecd (RS 006): Part A ratios 1.04–1.11 (all >1); Part B INCONCLUSIVE, signal straddle −0.21 vs −0.29 all entries |

## Run (PC, Windows CMD)

```
jupyter nbconvert --to notebook --execute --inplace notebooks\Regime_Structure\01_regime_existence.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks\Regime_Structure\02_c3_options.ipynb
```

RS 02 also reads the research DuckDB (`DERIBIT_DATA_ROOT/database/research.duckdb`) and fetches Deribit delivery prices (cached in `02/`; `RS02_OFFLINE=1` uses the cache).

Uses the Binance parquets from `tools/download_binance_um_klines.py` (`<parent of DERIBIT_DATA_ROOT>/Binance` or `BINANCE_UM_DIR`). Environment: `RS01_ASSETS` (default BTC,ETH,SOL,XRP), `RS01_N_NULL` (default 399), `RS01_SYNTHETIC=1`. Outputs in `01/` (small CSV/JSON).
