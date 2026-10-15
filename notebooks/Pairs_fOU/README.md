# Pairs fOU — Ivan's thesis pipeline (fOU statistical arbitrage) at 5 minutes on crypto

| Notebook | Description | Status |
|---|---|---|
| `01_fou_pairs_stage_a.ipynb` | Stage A: rolling 7-day Engle–Granger + Hurst + fOU, conditional first-passage T70 ≤ 4 h, perpetual legs at 12 bp round trip; OU and z-score benchmarks; calibration | Built; awaiting local run |

Protocol: `../../discussions/Pairs_fOU/001_claude_stage_a_preregistration.md`. Inputs: `<BINANCE_UM_DIR>/<ASSET>USDT_um_1m.parquet` (BTC, ETH, SOL, XRP). Run `python -m pytest -q tests/test_fou_pairs.py`, then execute notebook 01. Expect about 1 hour (formation tests are cached across the three modes). Outputs go to `01/`. `PF01_SYNTHETIC=1` runs a synthetic smoke test. Requires `statsmodels`.
