# CS03b — Signed trade-flow vs trading-activity attribution

Run `notebooks/Candle_Streaks/03b_trade_flow_attribution.ipynb` from a JupyterLab kernel in the `deribit-data` environment after running:

```bat
cd /d D:\Projects\deribit_market_data
git pull origin main
conda activate deribit-data
python -m pytest -q tests\test_candle_streak_flow_attribution.py
```

The experiment reuses the original five-minute **Binance spot** streaks and your already-downloaded **Binance USD-M futures one-minute Parquet files**, using the exact CS03 end-of-candle alignment. No new depth data or downloads are required. Expected inputs: `DERIBIT_DATA_ROOT/history/price_5m/<ASSET>.csv.gz` and `BINANCE_UM_DIR/<ASSET>USDT_um_1m.parquet` (or a sibling `Binance/` directory).

Four TRAIN-fitted multiclass logistic models compare price+streak+volatility alone, adding **signed aggressive trade imbalance** alone, adding **quote volume / trade count** alone, and both together. The complete model mathematically matches CS03. All eligible forecast origins are evaluated as **DOWN/ZERO/UP**. Separate conditional moving-bar UP/DOWN, running-streak reversal, flat termination, and fresh post-flat start scores are provided with count thresholds and paired full-day bootstraps.

Read the frozen [`CS03b protocol`](../../../discussions/Candle_Streaks/003b_chatgpt_trade_flow_attribution_preregistration.md) before interpreting outputs. This same historical TEST period was already inspected in CS03, so results are only **exploratory**; AM-T-HAR-Blend and collectors remain unchanged.

Notebook writes nine small aggregate CSV/JSON outputs alongside this README, under `notebooks/Candle_Streaks/03b/`. Run every cell, then push the aggregate outputs to GitHub for review. The optional `CS03B_SYNTHETIC=1` mode runs a synthetic smoke test into a temporary directory and **must not** be treated as empirical evidence.
