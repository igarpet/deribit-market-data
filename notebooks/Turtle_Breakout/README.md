# Turtle Breakout research branch

**Question:** does the turtle breakout (20-day high entry, max(entry − 2 ATR, 10-day low) exit, 2% high-water-mark risk, long only) beat luck and buy-and-hold on BTC/ETH/SOL/XRP perpetuals after fees, slippage and funding? Discussions: `../../discussions/Turtle_Breakout/`.

| Notebook | Content | Status |
|---|---|---|
| `01_turtle_breakout.ipynb` | Exact rules on 5m perp bars (17:00 New York days), portfolio with 2% HWM sizing; G1 vs 1,000 random-entry portfolios, G2 Sharpe vs buy-and-hold at the same mean leverage; trade twins, variants (no funding, short, long+short, 1h), single assets, years, halves | Run `6f8ce4a` (TB 002): `TREND_EDGE` as registered; G1 robust, G2 benchmark flawed |
| `02_turtle_benchmarks.ipynb` | Unchanged TB 01 portfolio vs constant-mix benchmarks (perp monthly/daily with funding, spot proxy monthly; reported: inverse-vol, equal-vol, alpha/beta, years, halves) | Run `18e355e` (TB 004): `FAILS_B1_B2_B3` |
| `03_turtle_intraday_engine.ipynb` | 9 strategies: daily / 4h candles / last 4 hours × ATR / engine / engine + stop sizing (frozen VD18 engine, causal calibration); G1/G2 per strategy, contrasts C1–C8 | Run `f41eaf7` (TB 006 §0): no strategy beats the benchmarks; original daily remains best |
| `04_trade_features.ipynb` | Winners vs losers of the original strategy: 41 entry features + path features; discovery on 26 other USDT perps, confirmation on the four; BH-FDR screen, Holm confirmation, ridge-logistic model with permutation test | Run `98224f4` (TB 007 §0): `ENTRY_SIGNAL_CONFIRMED` via model only (AUC 0.605); no single feature; TON flat-series artefact |
| `05_fractal_timeframes.ipynb` | Original rules on 1, 2, 3, 5, 10, 15, 30 min, 1, 2, 4, 8, 12 h, 1 day; gross edge in R vs random entries, net bp at taker, break-even cost, trades/day | Run `0119f2c` (TB 008 §0): not fractal; edge grows with candle size; only daily passes the net gate |
| `06_put_hedge_and_calls.ipynb` | Daily turtle on BTC/ETH: perpetual vs perpetual + put at the entry strike vs ATM / +1 ATR calls (38-day tenor, DVOL pricing, Deribit fees); Sharpe and drawdown on daily cumulative-R curves | Run `2cca94c` (TB 009 §0): Q1 put hedge `WORSENS` (Sharpe 0.53 vs 0.76), Q2 ATM call `NO_DIFFERENCE` (0.68) |
| `07_stop_and_reverse.ipynb` | What price does after losing exits (event study vs winning exits and random flat days) and a mirrored-turtle short opened at each losing stop exit; G1 mean R > 0, G2 vs random flat-day shorts | Run `9293f61` (TB 010 §0): `BEATS_RANDOM_SHORTS_ONLY` — +0.17R per reversal, beats random shorts (p 0.03), G1 fails ([−0.14, 0.53]) |
| `08_risk_per_trade_btc.ipynb` | BTC only: risk per trade 1%–100% vs 2% (125× cap, HWM sizing primary; 3× cap and equity sizing reported); growth difference with Holm over 99; Kelly bootstrap | Run `1e23d58` (TB 011 §0): `2PCT_NOT_BEATEN` in all sweeps; best in-sample 12% (CAGR 88%, DD −95%), ruin from 13% (HWM sizing) |
| `09_risk_per_trade_btc_reverse.ipynb` | Descriptive: TB 08's sweep (P / C3 / EQ, 1%–100%) for long only vs long + stop-and-reverse on BTC; side-by-side drawdowns and equity curves | Note TB 011; awaiting run |

Code: `src/data_engineering/modeling/turtle_breakout.py`, `turtle_benchmarks.py`, `turtle_intraday.py`, `turtle_trade_features.py`, `turtle_fractal.py`, `turtle_options.py`, `turtle_reverse.py`, `turtle_sizing.py`. Tests: `tests/test_turtle_breakout.py`, `test_turtle_benchmarks.py`, `test_turtle_intraday.py`, `test_turtle_trade_features.py`, `test_turtle_fractal.py`, `test_turtle_options.py`, `test_turtle_reverse.py`, `test_turtle_sizing.py`. Outputs: `01/01_*` … `09/09_*`. TB 06 needs `tools/download_deribit_dvol.py`. TB 04 needs extra downloads (TB 006 §6).

Running 07: `git pull`, `python -m unittest discover -s tests -p "test_turtle_reverse.py" -v`, then nbconvert `07_stop_and_reverse.ipynb` (timeout 7200); no download. Pre-registration: `discussions/Turtle_Breakout/009`.

Running 02 / 03: same as 01 without the downloads, with `02_turtle_benchmarks.ipynb` / `03_turtle_intraday_engine.ipynb` (03: timeout 14400; `set TB03_NULL_SCALE=0.1` for a quick look).

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
