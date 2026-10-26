# 017 — Claude: TB 14 readout and TB 15 pre-registration — the turtle on every S&P 500 stock, each in its own account

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-26 |
| **Branch** | Turtle Breakout |
| **Responds to** | TB 14 run `5fe843f`; Ivan, 2026-10-26: TB 14 (one portfolio) was not the test he asked for. TB 15 = long and long + reverse on each S&P 500 constituent as if each stock had its own account, same TB 01 rules (2% risk per trade), to see whether the strategy holds on completely different data. A correlated portfolio is decided afterwards |
| **Status** | pre-registration + code (validated on synthetic data); awaiting Ivan's run |

## 0. TB 14 in brief (run `5fe843f`; one account, not the requested design)

Verdict: both sets `FAILS`.

**SP500, one account at 0.05% of equity per trade, gross ≤ 2×:**

| | CAGR | Max DD |
|---|---|---|
| Long | 9.3% | −55% |
| Long + reverse | **−8.6%** | **−92%** |
| SPY buy and hold | 8.4% | −55% |
| Equal-weight buy and hold of the same names | 11.9% | −58% |

- **Trades:** 76,796 longs at mean R +0.118 (win rate 38%); 47,166 reverse shorts at **−0.240R**.
- **The 2× cap** skipped 21,074 long entries at 0.05%.

**INDEX (SPY/QQQ/DIA/IWM) at 0.5%:**
- Long: 2.5%, DD −9%, Calmar 0.28. Long + reverse: 0.7%, DD −21%.
- Longs +0.20R; reverses −0.25R.

**Reading.**
- The reversal loses money on equities in every set, period and cost case. Stock breakouts that fail do not go on to fall.
- The long side is positive per trade, but in one account it does not beat buy-and-hold.

**Limits** that keep TB 14 from being the answer:
- It mixed the strategy with portfolio construction (gross cap, f chosen for 500 names).
- It used a survivorship-biased list.

## 1. TB 15 design (`turtle_stock_accounts.py`)

**Per stock, one account, the TB 01 rules unchanged.**
- Risk 2% of the account's high-water mark per trade: units $= 0.02\,\mathrm{HWM}/(2\,\mathrm{ATR})$.
- Notional ≤ 3× equity.
- Entry at the 20-day high (stop order); exit at the higher of entry − 2 ATR and the 10-day low.
- **Long + reverse:** at every losing stop exit, a short with the mirrored exits (TB 07).
- No portfolio rule: each stock is a separate account.

**Same data handling as TB 14.**
- Universe: the thesis S&P 500 list (506 tickers returned by Yahoo; WBD empty), daily, adjusted, 2000 (or first session) + 30 sessions → 2026-10-23.
- Sessions are consecutive periods; each day is filled in the order O→H→L→C.
- Costs: 5 bp per side; borrow 0.5%/yr on shorts.
- The 4 index ETFs are run the same way and reported, outside the decision.
- The single-stock account reproduces `turtle_breakout.simulate` exactly (unit test).

## 2. Decision (binding)

Bonferroni over 2 tests, $\alpha = 0.025$ each, one-sided. Test set: the S&P 500 stocks.

| | Test |
|---|---|
| **G1** timing edge | pooled mean net R of all real long trades vs random-entry longs on the same stocks: same exits, costs and per-stock entry rate, 50 runs. Passes if z-test p < 0.025 **and** the difference ≥ **0.10R** |
| **G2** reversal | number of stocks with Calmar(long + reverse) > Calmar(long): sign test p < 0.025 |
| **Verdict** | `EDGE_AND_REVERSAL` / `TIMING_EDGE_ONLY` / `REVERSAL_ONLY` / `NO_EDGE` |

**Why the 0.10R floor on G1.**
- The daily-bar fills are not perfectly neutral. On driftless synthetic stocks whose OHLC come from an intraday random-walk path, breakout longs beat random-entry longs by **+0.04R** (sd 0.035 per 8 stocks), whatever the intraday order (adverse, nearest, random).
- With about 500 stocks, a z-test alone would declare that artefact significant. The floor is 2.5× the artefact.
- For scale: on daily crypto the edge vs random entries was +0.78R (TB 05).
- A first synthetic generator (opens not taken from the path) produced −0.15R and was discarded as unrealistic.

**Calibration of G2.** On 60 driftless synthetic stocks, long + reverse beat long on Calmar in 22/60 (p 0.99): no false positive.

**Reported:**
- **Distributions across stocks** (p10/p25/median/p75/p90/mean) of CAGR, max DD, Calmar and Sharpe for long, long + reverse and buy and hold of the same stock over the same window.
- **Shares of stocks:**
  - with positive CAGR;
  - where long beats buy and hold on Calmar and on Sharpe;
  - where long + reverse beats long on CAGR and on drawdown;
  - ruined.
- **Per-stock timing:** stocks whose real mean R exceeds their own random-entry mean, with a sign test.
- **Breakdowns:** by GICS sector; the index ETFs.
- **Average of the per-stock accounts:** each stock 1/N of capital, so not a portfolio test.
- **Figures.**

## 3. What to expect, stated in advance

- **G1:** TB 14 measured +0.118R per long; random entries on survivors will also be positive (the stocks went up). Whether the timing adds ≥ 0.10R is open. Equities are known to trend less than futures and crypto.
- **G2: expected `FAIL`.** TB 14's reverse shorts averaged −0.24R across 47,166 trades.
- **Long vs buy and hold:** with survivors and 2% risk per trade (mean gross well below 1×), buy and hold will usually have the higher CAGR; the drawdown and Calmar columns are the fair comparison.
- **Crypto comparison (TB 10, 30 coins, 2% of HWM):** median coin CAGR 4.5% long and 6.1% long + reverse; median max DD −22%.

## 4. Validation

`tests/test_turtle_stock_accounts.py`: 7 tests, all passing.
- Trade R matches the book's R.
- Borrow charges shorts only.
- The account equals TB 01's single-asset simulation.
- The random-entry null runs.
- Sign and z tests.
- The decision logic, including the effect floor.
- **Driftless synthetic stocks do not pass G1.**

`turtle_stocks.pseudo_bars` gained an `order` option (`adverse` default, unchanged; `nearest`, `random` for the calibration) and a path-based synthetic generator.

**Smoke run** (12 synthetic stocks + 4 ETFs): `NO_EDGE`; the notebook runs end to end.

## 5. Run (PC, Windows CMD)

The TB 14 data file is reused; no download.

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_turtle_stock_accounts.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 notebooks\Turtle_Breakout\15_stocks_own_accounts.ipynb
git add notebooks\Turtle_Breakout
git commit -m "Turtle Breakout 15: run"
git push
```

**Runtime:** roughly 25–45 minutes; the 50 random-entry runs per stock dominate.

**Outputs** go to `notebooks/Turtle_Breakout/15/`: `15_decision.json`, `15_per_stock.csv`, `15_distributions.csv`, `15_summary.csv`, `15_sectors.csv`, `15_average_of_accounts.csv`, `15_null.npz`, `15_manifest.json`.
