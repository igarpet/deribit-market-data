# Breakout Trend research branch

**Question:** does the hourly breakout strategy (100 h anchor, band $\pm1.5A$, stop-and-reverse at $1.5A$, 14 h trailing exit) survive Deribit costs on crypto perpetuals, beat random entries, and how does a long ATM option version compare? Discussions: `../../discussions/Breakout_Trend/`.

| Notebook | Content | Status |
|---|---|---|
| `01_breakout_backtest.ipynb` | Exact rules on 5m Binance OHLC (HYPE: Deribit perp); perpetual leg net of fees and slippage; delta-matched long ATM option leg (model-priced, calibrated on collector quotes; real-quote check in the collector window); dev 2024-09 → 2025-12, test 2026; coin-flip null (200 runs); equity curves per asset (perpetual vs options) | Pre-registered (BT 001); awaiting first run |

Code: `src/data_engineering/modeling/breakout_trend.py`. Tests: `tests/test_breakout_trend.py`. Outputs: `01/01_*`.

Data (BT 002): Binance USD-M perpetual 1m parquet (`tools/download_binance_um_klines.py`, `<parent of DERIBIT_DATA_ROOT>/Binance` or `BINANCE_UM_DIR`) aggregated to 5m, BTC/ETH/SOL/XRP; plus the collector's `daily/option_grid*` parquet files. The registered spot source is still available with `set BT01_SOURCE=ohlc_5m` (needs `tools/download_ohlc_history.py --start 2024-08-25`).

## Running 01 (PC, Windows CMD)

Close the notebook in Jupyter first.

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_breakout_trend.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebooks\Breakout_Trend\01_breakout_backtest.ipynb
git add notebooks\Breakout_Trend
git commit -m "Breakout Trend 01: run"
git push
```

Optional before the `jupyter` line: `set BT01_ASSETS=BTC,ETH`, `set BT01_N_NULL=50` (quick look; the decision uses 200).
