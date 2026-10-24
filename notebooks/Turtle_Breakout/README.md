# Turtle Breakout research branch

**Question:** does the turtle breakout (20-day high entry, max(entry − 2 ATR, 10-day low) exit, 2% high-water-mark risk, long only) beat luck and buy-and-hold on BTC/ETH/SOL/XRP perpetuals after fees, slippage and funding? Discussions: `../../discussions/Turtle_Breakout/`.

| Notebook | Content | Status |
|---|---|---|
| `01_turtle_breakout.ipynb` | Exact rules on 5m perp bars (17:00 New York days), portfolio with 2% HWM sizing; G1 vs 1,000 random-entry portfolios, G2 Sharpe vs buy-and-hold at the same mean leverage; trade twins, variants (no funding, short, long+short, 1h), single assets, years, halves | Run `6f8ce4a` (TB 002): `TREND_EDGE` as registered; G1 robust, G2 benchmark flawed |
| `02_turtle_benchmarks.ipynb` | Unchanged TB 01 portfolio vs constant-mix benchmarks (perp monthly/daily with funding, spot proxy monthly; reported: inverse-vol, equal-vol, alpha/beta, years, halves) | Pre-registered (TB 003); awaiting run |

Code: `src/data_engineering/modeling/turtle_breakout.py`, `turtle_benchmarks.py`. Tests: `tests/test_turtle_breakout.py`, `test_turtle_benchmarks.py`. Outputs: `01/01_*`, `02/02_*`.

Running 02: same as 01 without the downloads, with `02_turtle_benchmarks.ipynb`.

Data: `<BINANCE_UM_DIR>\long\<A>USDT_um_1m.parquet` and `long\basis\funding\<A>USDT_funding.parquet` (7 years; separate folder so the 2-year files of other branches are untouched). Override with `TB01_DATA_DIR`.

## Running 01 (PC, Windows CMD)

Close the notebook in Jupyter first.

```bat
cd /d D:\Projects\deribit_market_data
git pull
python tools\download_binance_um_klines.py --months 84 --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT --out D:\MarketData\Binance\long
python tools\download_binance_basis_data.py --months 84 --assets BTC ETH SOL XRP --skip spot perp quarterly --out D:\MarketData\Binance\long
python -m unittest discover -s tests -p "test_turtle_breakout.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebooks\Turtle_Breakout\01_turtle_breakout.ipynb
git add notebooks\Turtle_Breakout
git commit -m "Turtle Breakout 01: run"
git push
```
