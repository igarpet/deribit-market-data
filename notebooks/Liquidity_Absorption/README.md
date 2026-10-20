# Liquidity Absorption — independent research line

MNQ-inspired BTC/ETH Binance perpetual *historical exploratory* research. 5-minute context and 1-minute entry/management. Existing production collection pipelines are untouched. Versioned reusable engines live under `src/data_engineering/modeling/`. Local Binance root comes from `BINANCE_UM_DIR` or the parent of `DERIBIT_DATA_ROOT` (`.env`).

- **01_data_audit.ipynb** — Binance 1m/perp/spot, aggTrades first/middle/last checks, flow reconciliation; outputs in `01/`.
- **02_causal_features.ipynb** — independent price-only two-touch candidates and signal-close flow features; outputs in `02/`.
- **02b_touch_flow.ipynb** — same A02 candidates; first-vs-second level-executed volume, aggressor flow, zone dwell proxies, pre-first profile; outputs in `02b/`.
- **03_execution_simulation.ipynb** — same A02 candidates under FIXED / BREAKEVEN / TRAIL_1M, trade-print ordered stop/target and *assumed* fills; outputs in `03/`.
- **03b_mirrored_stop_equity.ipynb** — **keep the original A03 structural TP and entry**, set initial stop the identical price distance away on the losing side (exact 1:1 gross RR). Executes historical prints, compares paired closed outcomes to original fixed-stop A03, and produces gross/net BTC/ETH/combined virtual equity curves in `03b/`. +1R BE/trailing are redundant with 1R TP; only FIXED is meaningful. No actual trade performance has been generated until the local notebook is run.

- **03c_reverse_passive_third_touch.ipynb** — independent replay of **reversed-side 1:1** using A03b frozen barriers, and original-side **passive limit orders at the liquidity level** awaiting a third post-second-close revisit. Separate opposing-aggressor touch and stricter one-tick-through maker-eligibility proxies. Outputs in `03c/`: all 452 signals per scenario, skip/censor statuses, net expectancy per closed trade and per signal, fee-only sensitivities and gross/net BTC/ETH/combined virtual equity curves. **Passive fills are hypothetical, NOT evidenced maker queue fills**; passive avoids taker fees only on entry, exits remain taker under the research assumptions.


- **03d_volatility_risk_integration.ipynb** — causal, read-only bridge to the saved **VD18 HAR_4** 15/60/240-minute spot terminal-return variance forecasts. Liquidity Absorption retains **all existing directions, entries and frozen structural targets**. A protective stop is set beyond both the second rejection wick/zone and one unit of VD18 60m forecast sigma. Reports risk-budgeted hypothetical position sizing and compares 60m vs 240m hold policies for A03 taker FIXED/BE/TRAIL and A03c touch/through maker *eligibility* (not known maker fills). No in-sample optimization; forecasts require strict complete past 5m bars. Outputs in `03d/`, source protocol `discussions/Liquidity_Absorption/013_chatgpt_a03d_vd18_risk_bridge.md`.

- **04_intratrade_path_anatomy.ipynb** — A04 entry-to-exit study on the frozen A03d fixed-stop 240m taker cohort (452 original candidates). Compares still-active TARGET/STOP/TIMEOUT paths at causal +1/+5/+15/+30/+60m checkpoints, using existing Binance UM 1m candles and taker-buy quote-volume flow proxies. Produces risk-set panels, group summaries and competing-exit charts under `04/`. Outcome-conditioned **descriptive** diagnostics only; no new tick archives, reoptimized exits or confirmed prediction. Engine: `liquidity_absorption_stage_a04.py`; tests: `tests/test_liquidity_absorption_stage_a04.py`; protocol: `discussions/Liquidity_Absorption/015_chatgpt_a04_intratrade_anatomy_protocol.md`.

A03 uses explicitly **provisional** crypto stop/BE/target/fee parameters, and does **not** replicate the user's MNQ point/tick distances, simulate low-volume-node targets, fund positions, identify executable bid/ask liquidity, apply portfolio capital constraints, or claim prospective trading significance. Its historical sample has already been examined for features. Every trade retained in its output, even censored/skipped events.

Run in the `deribit-data` conda environment after `git pull`:

```
python -m unittest tests.test_liquidity_absorption_stage_a03
jupyter lab
```

Open and **Restart Kernel & Run All** in `notebooks/Liquidity_Absorption/03_execution_simulation.ipynb`, then commit the notebook and **four** generated `03/` CSV/JSON files to GitHub for independent review. Data are read-only and aggregate trade files are streamed by symbol/day; CPU cost depends on tape volume.

Research discussions under `discussions/Liquidity_Absorption/` document the hypotheses, non-frozen assumptions, integrity caveats, and owner decisions needed before a genuinely untouched holdout experiment.

Run A03b with `python -m unittest tests.test_liquidity_absorption_stage_a03b`, then open `03b_mirrored_stop_equity.ipynb` and run all cells, commit `03b/` CSV/JSON/PNG outputs and executed notebook. Graphs assume fixed USDT 10,000 notional per independent event without capital/overlap constraints; they are *not* executable account equity. The historical tape lacks bid/ask and funding.

Run A03c with `python -m unittest tests.test_liquidity_absorption_stage_a03c` then open `03c_reverse_passive_third_touch.ipynb` in the `deribit-data` kernel, run all, and commit `03c/` outputs. The prior A03b results are the required source and remain untouched. A03c assumes 1-second order-placement latency, 90-minute passive-order lifetime, 240-minute holding after maker entry, 2bp maker-entry fee and 6bp taker-exit fee for A03 comparability, plus a fee-only sensitivity grid. The historical aggTrade print side/price establishes *possible* queue eligibility but cannot prove a passive order filled. Review `discussions/Liquidity_Absorption/011_chatgpt_a03c_reverse_and_third_touch_passive.md` before interpreting returns.

Run A03d after pulling updates with `python -m unittest tests.test_liquidity_absorption_stage_a03d`; open `03d_volatility_risk_integration.ipynb` in the `deribit-data` kernel and **Restart & Run All**. In addition to `BINANCE_UM_DIR`, the frozen VD18 forecast bridge needs local `DERIBIT_DATA_ROOT/history/price_5m/{BTC,ETH}.csv.gz`; results remain exploratory and no actual full historical A03d run has been performed in the assistant's environment. Upload the executed notebook and generated `03d/` artifacts for audit. Holding-time scenarios are compared descriptively; no adaptive timeout is claimed before testing.
