# Signed Jumps — notebooks

| Notebook | Description | Status |
|---|---|---|
| `01_signed_hawkes.ipynb` | Volatility engine on bipower (1m) + signed discrete Hawkes ladder (a)–(e); gates G1 excitation, G2 sign (parent sign / truncated skew), G3 15-minute signed jump forecast; c sensitivity; jump sizes by sign | Run `8a62395` (SJ 002): **G1, G2a, G2b, G3 all PASS**; ρ = 0.155, half-life 1 min; economics untested (SJ 02a proposed) |

Protocol: `../../discussions/Signed_Jumps/001_claude_sj01_preregistration.md`. Code: `src/data_engineering/modeling/signed_jumps.py`.

Inputs: `<BINANCE_UM_DIR>/basis/spot/<A>USDT_spot_1m.parquet` (BTC, ETH, SOL, XRP), from `tools/download_binance_basis_data.py`.
Run `python -m pytest -q tests/test_signed_jumps.py`, then execute notebook 01 (about 30–60 min). Outputs go to `01/`. `SJ01_SYNTHETIC=1` runs a short synthetic smoke test.
