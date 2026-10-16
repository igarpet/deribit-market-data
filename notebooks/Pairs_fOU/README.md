# Pairs fOU — Ivan's thesis pipeline (fOU statistical arbitrage) at 5 minutes on crypto

| Notebook | Description | Status |
|---|---|---|
| `02_basis_fou.ipynb` | PF02: the same engine on construction-tied spreads: perp − spot (about 30 assets) and quarterly − perp, β = 1, ADF screen, real funding, taker costs. Protocol 003 | Built; awaiting download and run |
| `01_fou_pairs_stage_a.ipynb` | Stage A: rolling 7-day Engle–Granger + Hurst + fOU, conditional first-passage T70 ≤ 4 h, perpetual legs at 12 bp round trip; OU and z-score benchmarks; calibration | Built; awaiting local run |

Protocol: `../../discussions/Pairs_fOU/001_claude_stage_a_preregistration.md`. Inputs: `<BINANCE_UM_DIR>/<ASSET>USDT_um_1m.parquet` (BTC, ETH, SOL, XRP). Run `python -m pytest -q tests/test_fou_pairs.py`, then execute notebook 01. Expect about 1 hour (formation tests are cached across the three modes). Outputs go to `01/`. `PF01_SYNTHETIC=1` runs a synthetic smoke test. Requires `statsmodels`.

## PF02 run order

1. `python tools/download_binance_basis_data.py --months 26` downloads spot, perps (only those missing), funding and quarterly files for the 30-asset universe into `<BINANCE_UM_DIR>/basis/` (several GB).
2. `python -m pytest -q tests/test_basis_fou.py`
3. Execute `02_basis_fou.ipynb`. It takes 1–3 hours and checkpoints per asset in `02/parts/`, so a re-run resumes. Push `02/` (`parts/` may be left out if too large).
