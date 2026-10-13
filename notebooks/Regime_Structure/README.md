# Regime Structure notebooks

Pre-registration: `discussions/Regime_Structure/001_claude_branch_preregistration.md`.

| Notebook | Content | Status |
|---|---|---|
| `01_regime_existence.ipynb` | Hourly SHOCK / DISP / NOISE / RANGE labels; persistence and transition lifts vs a sign-randomised null; gates on the test window | Built, synthetic-validated, awaiting first real run |

## Run (PC, Windows CMD)

```
jupyter nbconvert --to notebook --execute --inplace notebooks\Regime_Structure\01_regime_existence.ipynb
```

Uses the Binance parquets from `tools/download_binance_um_klines.py` (`<parent of DERIBIT_DATA_ROOT>/Binance` or `BINANCE_UM_DIR`). Environment: `RS01_ASSETS` (default BTC,ETH,SOL,XRP), `RS01_N_NULL` (default 199), `RS01_SYNTHETIC=1`. Outputs in `01/` (small CSV/JSON).
