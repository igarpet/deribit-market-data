# Liquidity Absorption — independent research line

MNQ-inspired BTC/ETH Binance perpetual *historical exploratory* research. 5-minute context and 1-minute entry/management. Existing production collection pipelines are untouched. Versioned reusable engines live under `src/data_engineering/modeling/`. Local Binance root comes from `BINANCE_UM_DIR` or the parent of `DERIBIT_DATA_ROOT` (`.env`).

- **01_data_audit.ipynb** — Binance 1m/perp/spot, aggTrades first/middle/last checks, flow reconciliation; outputs in `01/`.
- **02_causal_features.ipynb** — independent price-only two-touch candidates and signal-close flow features; outputs in `02/`.
- **02b_touch_flow.ipynb** — same A02 candidates; first-vs-second level-executed volume, aggressor flow, zone dwell proxies, pre-first profile; outputs in `02b/`.
- **03_execution_simulation.ipynb** — same A02 candidates under FIXED / BREAKEVEN / TRAIL_1M, trade-print ordered stop/target and *assumed* fills; outputs in `03/`.

A03 uses explicitly **provisional** crypto stop/BE/target/fee parameters, and does **not** replicate the user's MNQ point/tick distances, simulate low-volume-node targets, fund positions, identify executable bid/ask liquidity, apply portfolio capital constraints, or claim prospective trading significance. Its historical sample has already been examined for features. Every trade retained in its output, even censored/skipped events.

Run in the `deribit-data` conda environment after `git pull`:

```
python -m unittest tests.test_liquidity_absorption_stage_a03
jupyter lab
```

Open and **Restart Kernel & Run All** in `notebooks/Liquidity_Absorption/03_execution_simulation.ipynb`, then commit the notebook and **four** generated `03/` CSV/JSON files to GitHub for independent review. Data are read-only and aggregate trade files are streamed by symbol/day; CPU cost depends on tape volume.

Research discussions under `discussions/Liquidity_Absorption/` document the hypotheses, non-frozen assumptions, integrity caveats, and owner decisions needed before a genuinely untouched holdout experiment.
