# 001 — Claude: Turtle Breakout 01 pre-registration — the turtle breakout on crypto perpetuals

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-24 |
| **Branch** | Turtle Breakout (new, independent) |
| **Responds to** | Ivan, 2026-10-24: test Richard Dennis' turtle strategy as presented by Rallio Trading and reviewed by Algo Vibes (two transcripts supplied) |
| **Status** | pre-registration + code (validated on synthetic data); awaiting Ivan's download and run |

## 1. What the sources claim

**Rallio's simplified version.**
- Long only, no pyramiding, risk 2% of the high-water mark per trade.
- Markets: S&P 500, NASDAQ, gold, silver and Bitcoin, over about 11 years: 27.3% a year, max drawdown 37.6%, profit factor 1.88.
- Bitcoin was the best market: profit factor 3.01 on 66 trades.

**The Algo Vibes review** on futures since 2020 (Bitcoin since 2017) reproduced those numbers and then asked two questions:
1. **Does it beat doing nothing?** On return, no. Buying the same markets at the same leverage (about 1.13×) earned about the same, 32.8% against 33.5% a year. The strategy's advantage was a smaller drawdown, 27.7% against 42.7%.
2. **Does it beat luck?** Not on 10 futures markets: the same exits with random entries did as well. **Bitcoin was the exception**, with random entries matching it in only 4.8% of runs. With 11 markets tested, the reviewer calls that interesting, not proven.

We have crypto perpetuals with real funding, so the review's Bitcoin exception can be tested out of sample on BTC, ETH, SOL and XRP.

## 2. Rules (fixed in `turtle_breakout.py`, `Params`)

**Setup.**
- The trading day runs 17:00 → 17:00 New York, including DST.
- Levels use completed days only and move once a day.

**Entry.** A stop order at $H^{20}_d=\max(\text{high})$ over the last 20 days. It fills at $\max(H^{20}_d,\ \text{open of the 5m bar that crosses it})$.

**Stops.**
- **Fixed stop:** $S^{fix}=P_{entry}-2\,\mathrm{ATR}_{20}$, never moved. ATR is the Wilder average of the daily true range.
- **Trailing stop:** $L^{10}_d=\min(\text{low})$ over the last 10 days, recalculated each day.
- **Exit** when price trades through $\max(S^{fix},L^{10}_d)$. The fill is at $\min(\text{stop},\ \text{bar open})$.

**Other rules.**
- **Same-day stops:** a stop can be hit on the entry day, but only on 5m bars after the entry bar.
- **Re-entry:** a re-entry after an exit waits for the next day.

**Size.** $u=0.02\,\mathrm{HWM}/(2\,\mathrm{ATR}_{20})$, where HWM is the highest marked equity at a prior day's close.

**Leverage cap (my addition).** Position notional is capped at 3× the last marked equity, and no entries are taken at equity ≤ 0. The cap does not bind in the daily version (synthetic mean leverage 0.4×). It exists for the hourly variant, where ATR is tiny.

**Costs.**
- Taker fee 0.05% + 1 bp slippage per side (the Deribit perpetual schedule, as in BT 001).
- Binance USD-M funding, charged on the held notional at every funding time. Long pays positive funding.
- **Funding matters here.** A long-only trend strategy holds through bull markets, which is when funding is highest. The review used futures, where carry sits in the price.

**Data.**
- Binance USD-M perpetual 1m klines from late 2019 (BTC, ETH, XRP) and September 2020 (SOL), aggregated to 5m bars for intraday fills.
- Funding history from the same source.

**Not tested: Rallio's "day-trading" variant** (flat from 16:30 to 18:00 New York). It exists to avoid CFD swap charged at 17:00. Perpetual funding is charged at 00, 08 and 16 UTC, so on our instrument the variant would only add fills.

## 3. Primary specification and windows

- **Primary specification:** daily, long only, BTC + ETH + SOL + XRP in one account, net of all costs.
- **Primary window:** from 2020-10-15, the first date all four perpetuals are listed with warm-up, to the end of the data.
- **Out of sample:** nothing is fitted. The parameters are the published turtle/Rallio values, so the whole window is out of sample for them.
- **Single-asset runs** use each asset's full history and are reported only.

## 4. Decision (binding)

**N2, random portfolio.**
- The exits, sizing and costs are identical to the strategy.
- The entry is a coin. On each eligible flat day the strategy enters at that day's 17:00 open with probability equal to the real run's entry rate for that asset.
- 1,000 runs.

**Buy-and-hold benchmark.**
- Equal notional in the four perpetuals at the strategy's mean gross leverage, never rebalanced.
- The same entry cost, with funding.

**Gates:**

| Gate | Rule |
|---|---|
| **G1, beats luck** | share of N2 runs with CAGR ≥ real < 0.05 |
| **G2, beats doing nothing** | Sharpe(strategy) − Sharpe(buy-and-hold), daily returns, 21-day block bootstrap (2,000), q2.5 > 0 |

**Verdict:**

| Gates passed | Verdict |
|---|---|
| G1 and G2 | `TREND_EDGE` |
| G1 only | `TIMING_ONLY` |
| G2 only | `RISK_PROFILE_ONLY` |
| Neither | `NO_EDGE` |

**Reported only:**
- **N1 trade twins:** each real trade redrawn at a random 5m bar, with the same holding time, side and costs; 2,000 runs, per asset and pooled. There are 4 assets, so a per-asset share below 0.0125 is the Bonferroni-equivalent of 5%.
- **Variants:** no funding; short only; long + short (the original turtles); 1-hour bars (20/10/20 hours). Each has its own random-entry null of 100–200 runs.
- **Per asset:** single-asset runs.
- **Time splits:** calendar years and two halves (split at 2023-07-01).
- **Buy-and-hold at 1×.**

## 5. Validation (synthetic)

`tests/test_turtle_breakout.py`: 14 tests, all passing.

**Mechanics and causality:**
- The 17:00 New York boundary falls at 21:00 UTC in summer and 22:00 UTC in winter.
- 5m bars are labelled at the bar END.
- Levels use completed periods only: they are unchanged when later data is removed, and the trade list is causal up to a cut.
- The breakout bar is the first bar at or above the level.
- Stop-order fills are never better than the open.
- Sizing equals 2% HWM / 2 ATR.
- Accounting closes: final equity equals start + Σ trade P&L + the exit fees of the last open position.
- Funding sign: longs pay positive funding; shorts receive it.

**Nulls and modes:**
- With a zero random-entry probability there are no trades.
- On a driftless random walk, G1 = FAIL.
- The hourly mode runs, and the leverage cap holds.

**End-to-end run** on 4 synthetic random walks with funding of 1 bp per 8 h: `NO_EDGE`.
- Random entries matched or beat the real CAGR in 28% of runs.
- Trade twins: 15–68% per asset.
- The short side lost money, as expected under a driftless random walk with positive funding.
- The hourly version was ruined by costs (4,112 trades).

## 6. Run (PC, Windows CMD)

The long history goes to a separate folder, so the 2-year files used by VWAP Events, VWAP Drift and Breakout Trend are not overwritten. Close the notebook in Jupyter first.

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

- **Download.** About 340 monthly zips, roughly 1 GB including the zip cache. Months before a contract was listed are reported as missing and skipped.
- **Notebook.** About 10–20 minutes, mostly the null runs and the hourly variant.
- **Outputs** go to `notebooks/Turtle_Breakout/01/`:
  - `01_decision.json`, `01_vs_buy_hold.csv`;
  - `01_trade_twins.csv`, `01_null_portfolios.csv`;
  - `01_variants.csv`, `01_single_asset.csv`, `01_by_asset.csv`;
  - `01_yearly.csv`, `01_halves.csv`;
  - `01_equity_daily.csv`, `01_trades.csv.gz`;
  - `01_coverage.csv`, `01_manifest.json`.

## 7. Questions for ChatGPT

1. Binance funding is used as a proxy for Deribit perpetual funding. Is the difference material for a strategy that holds for weeks?
2. The buy-and-hold benchmark is never rebalanced. Should it be rebalanced monthly to equal weight, which is closer to what the strategy's sizing does?

## 8. Decisions needed from Ivan

1. Confirm that §4 is binding: primary = daily long-only portfolio; G1 and G2 as written.
2. Confirm the two additions to the published rules: the 3× per-position leverage cap and funding included in the primary.
3. Run §6.
