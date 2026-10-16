# Signed Jumps — notebooks

| Notebook | Description | Status |
|---|---|---|
| `01_signed_hawkes.ipynb` | Volatility engine on bipower (1m) + signed discrete Hawkes ladder (a)–(e); gates G1 excitation, G2 sign (parent sign / truncated skew), G3 15-minute signed jump forecast; c sensitivity; jump sizes by sign | Run `8a62395` (SJ 002): **G1, G2a, G2b, G3 all PASS**; ρ = 0.155, half-life 1 min; economics untested (SJ 02a proposed) |
| `02_post_jump_paths.ipynb` | SJ 02a: SJ 01 jumps (no refit, `01/01_fits.csv`) → cluster starters → signed Binance perp returns 1–60 min, immediate and delayed entry; rule chosen on DEV; TEST verdict NET_EDGE / GROSS_EDGE_ONLY / NO_EDGE at 12 bp | Run `2f99c39` (SJ 004): **NO_EDGE**, TEST gross +0.13 bp [−0.63, +0.87]; path flat at all horizons |
| `03_flow_heat.ipynb` | SJ 03: order-flow "heat" (trade/volume acceleration, \|imbalance\|) added to SJ 01 (b)/(e); F1/F2/F3 gates; 15-min hazard lift; event-time activity profile around jumps | Run `11f8369` (SJ 006): F1 PASS, F2 FAIL, F3 FAIL — flow is a sharper volatility thermometer, not a separate boil signal |
| `04_seconds.ipynb` | SJ 04: SJ 01 jump minutes located to the second with spot 1s klines; P1 activity build-up vs hour-matched placebo, P2 split events, P3 post-onset continuation grid | Run `342ea73` (SJ 008): P1 PASS, P2 MOSTLY_DISTINCT_EVENTS, P3 withdrawn (look-ahead in design) |
| `05_realtime_trigger.ipynb` | SJ 04b: real-time seconds trigger, forward returns over latency δ ∈ {1,2,5,10,30} s × hold H ∈ {10,30,60,300} s, c ∈ {3,4,5}; splits by asset/direction | Run `6733112` (SJ 010): NO_CONTINUATION, flat at every latency/hold/threshold |
| `06_jump_hazard_vd19.ipynb` | SJ 05: expected SJ 01 jumps from VD 19 entries to the 08:00 expiry; encompassing regression with VD 19's x; R_net split by jump hazard | Built; awaiting run (reads `Volatility_Dynamics/19/19_rows.csv`) |

Protocol: `../../discussions/Signed_Jumps/001_claude_sj01_preregistration.md`. Code: `src/data_engineering/modeling/signed_jumps.py`.

Inputs: `<BINANCE_UM_DIR>/basis/spot/<A>USDT_spot_1m.parquet` (BTC, ETH, SOL, XRP), from `tools/download_binance_basis_data.py`.
Run `python -m pytest -q tests/test_signed_jumps.py`, then execute notebook 01 (about 30–60 min). Outputs go to `01/`. `SJ01_SYNTHETIC=1` / `SJ02_SYNTHETIC=1` / `SJ03_SYNTHETIC=1` / `SJ04_SYNTHETIC=1` / `SJ04B_SYNTHETIC=1` / `SJ05_SYNTHETIC=1` run short synthetic smoke tests. Notebook 02 also needs `<BINANCE_UM_DIR>/<A>USDT_um_1m.parquet` (perps).
