# 016 — Claude: TB 14 pre-registration — the frozen specification on US equities (S&P 500 constituents and index ETFs)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-26 |
| **Branch** | Turtle Breakout |
| **Responds to** | Ivan, 2026-10-26: stop testing on coins (crypto rallies are too correlated); test on stocks and indexes, Yahoo Finance data, using the constituent list of the thesis repo |
| **Status** | pre-registration + code (validated on synthetic data); awaiting Ivan's download and run |

## 0. TB 13: not run, closed by Ivan

- **Universes:** the rule-based universes were chosen and committed (`c1d0194`): 184 eligible new perps (60 drawn) and 20 spot pairs for 2017–19.
- **Notebook:** the run (`9c02960`) found no 1m files and stopped. The 864 files of about 30 KB in the data folder are the daily klines of the *screening* stage. The 1m `download` stage had not produced data.
- **Decision:** Ivan has closed the crypto out-of-sample test; TB 13 has no result. The universe file stays in the repo, so the test can still be run later as specified.

## 1. Question

Does the frozen specification of TB 12/13 work on a market with different data and much lower cross-sectional correlation?
- **Components:** one account; TB 01 daily longs; TB 07 reverse short at each losing stop exit; no portfolio rule; risk per trade $f \times$ equity.
- **Correlation:** the median pairwise daily correlation of S&P 500 stocks is far below crypto's (Ivan's thesis).

The two claims tested are the ones TB 13 would have tested:
- **(A)** the strategy grows the account;
- **(B)** the reversal improves Calmar over long only.

## 2. Data and universe

**Universe.**
- **SP500:** the 503 symbols of `data/raw/constituents.csv` in `igarpet/thesis` @ `f967975`, copied to `notebooks/Turtle_Breakout/14/14_constituents_thesis.csv`. `.` becomes `-` for Yahoo, as in the thesis.
- **INDEX:** SPY, QQQ, DIA, IWM. The ETFs are tradable, unlike `^GSPC` and `^NDX`.

**Prices.** Yahoo Finance daily OHLC, adjusted for splits and dividends (`auto_adjust=True`), from 2000-01-03 to the last session; 30 sessions of warm-up.

**Survivorship bias.** The list is a 2026 snapshot (the thesis notes the same limitation). Stocks that left the index (bankruptcies, takeovers, demotions) are missing, which flatters anything long. The equal-weight buy-and-hold of the same names carries the same bias and is reported next to the strategy. For **B** the bias works *against* the reversal, since surviving stocks went up after their failed breakouts.

## 3. Adapting the 5-minute engine to daily bars (`turtle_stocks.py`)

**Calendar.** Trading sessions are mapped to consecutive pseudo-days, so the 20/10/20-period windows are 20/10/20 *trading* days; results are mapped back to real session dates.

**Fill order.**
- Each daily bar becomes three ordered pseudo-bars, O→H, H→L, L→C.
  - A long that enters at the 20-day high and whose stop is below that day's low is filled in the adverse order: entry first, stop second.
  - A gap through a stop fills at the open.
- Daily highs, lows, closes and the Wilder ATR are unchanged (unit test).

**Costs.**
- 5 bp per side (commission, spread and slippage for S&P 500 names); 10 bp sensitivity.
- Short borrow 0.5% a year, charged at the exit (marks before the exit omit it); 2% sensitivity.
- No funding, and no interest on idle cash.

**Sizing.**
- Equity sizing; at most 3× notional per position.
- **New: total entry notional of open positions ≤ 2× equity** (Reg-T overnight margin). An entry beyond it is cut to fit, or skipped if less than 25% fits.
- **Base risk:** with about 500 names, many positions are open at once.
  - SP500 uses $f \in \{0.025\%, 0.05\%, 0.1\%\}$, primary **0.05%**. With roughly a third of the stocks in a trade and a 2-ATR stop of about 4%, gross notional is around 2× at 0.05%.
  - INDEX (4 ETFs) uses $f \in \{0.25\%, 0.5\%, 1\%\}$, primary **0.5%**.
  - The rate of skipped and scaled entries is reported, because TB 12 showed that skipping clustered breakouts costs return.

## 4. Decision (binding)

Per set (SP500, INDEX), at its primary $f$; one-sided, Bonferroni over 4 ($\alpha = 0.0125$), 21-day circular block bootstrap (4,000 draws), annualised on 252 sessions:

| | Test |
|---|---|
| **A** the strategy grows | long + reverse: mean daily log growth, lower bound > 0 |
| **B** the reversal helps | Calmar(long + reverse) − Calmar(long), lower bound > 0 |
| **Verdict per set** | `HOLDS` / `EDGE_ONLY` / `REVERSAL_ONLY` / `FAILS` |

**Reported:**
- the other values of $f$; 10 bp and 2% borrow sensitivities;
- equal-weight daily-rebalanced buy and hold of each set; SPY buy and hold;
- long only at the exposure of long + reverse;
- sub-periods 2000–09 (two bear markets), 2010–19 and 2020–;
- trade statistics (count, mean and median R, win rate, share of R ≥ 3);
- curves.

**Prior.**
- **Weak evidence for trend following on single stocks:** stocks mean-revert more than futures or crypto, and the turtle's success was in commodity and currency futures.
- **A on SP500** could pass mainly through survivorship and the 2010s bull market. That is why the buy-and-hold comparison matters even though it is not a gate.
- **B is the cleanest test here.** If the reversal helps in a market where failed breakouts do not cluster, the crypto mechanism (clusters of failed breakouts become clusters of shorts) was not the whole story. If it fails, the crypto result was specific to correlated markets.
- **Expected:** INDEX `EDGE_ONLY` or `FAILS` (four instruments, few trades); SP500 B `FAILS`.

## 5. Validation

`tests/test_turtle_stocks.py`: 8 tests, all passing.
- Pseudo-bars: three per session, one consecutive period per session.
- Period OHLC and ATR equal the daily ones.
- Pseudo-to-real mapping.
- Missing sessions and repair of inconsistent bars.
- Same-day entry-then-stop is adverse.
- Borrow charges shorts only.
- The gross cap binds and lowers leverage.
- Session-dated metrics.

`test_turtle_portfolio.py` still passes: the `gross_cap` default is off, so TB 11/12 are unchanged.

**Smoke run** (18 synthetic tickers with a common factor, no trend persistence): both sets `FAILS`; the notebook runs end to end.

## 6. Run (PC, Windows CMD)

```bat
cd /d D:\Projects\deribit_market_data
git pull
pip install yfinance
python -m unittest discover -s tests -p "test_turtle_stocks.py" -v
python tools\download_tb14_stocks.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 notebooks\Turtle_Breakout\14_stocks_and_indexes.ipynb
git add notebooks\Turtle_Breakout
git commit -m "Turtle Breakout 14: run"
git push
```

- **Download:** about 507 tickers of daily data, a few minutes. The tool writes `D:\MarketData\Stocks\tb14\stocks_daily.parquet` and a `download_report.csv` listing any ticker Yahoo did not return.
- **Notebook:** trade lists for about 500 tickers and the one-account simulations take roughly 20–40 minutes.

**Outputs** go to `notebooks/Turtle_Breakout/14/`: `14_decision.json`, `14_results.csv`, `14_benchmarks.csv`, `14_subperiods.csv`, `14_trades_summary.csv`, `14_coverage_{sp500,index}.csv`, `14_curves.csv.gz`, `14_manifest.json`.
