# 019 — Claude: TB 16 readout and TB 17 pre-registration — BTC-USD daily since 2014

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-28 |
| **Branch** | Turtle Breakout |
| **Responds to** | TB 16 run `1c7280e`; Ivan, 2026-10-28: is the crypto edge only a product of the bull markets BTC went through? Test only BTC, from 2014 onwards, with Yahoo data |
| **Status** | pre-registration + code (validated on synthetic data); awaiting Ivan's run |

## 0. TB 16 in brief (run `1c7280e`): `NO_EDGE` on futures

**40 Yahoo continuous futures** (6N was not returned), each in its own account, TB 01 rules:
- Real longs +0.091R, random-entry longs +0.083R: difference +0.009R.
- Per-contract t-test against the +0.045R fill artefact: t = −2.2.
- The reversal is better on 10 of 40.

**18 futures-proxy ETFs:** +0.002R.

**Average of the futures accounts:** long 0.8% CAGR (Calmar 0.05), buy and hold 3.4%.

**By sector:**
- Metals, grains and equity indices have positive R, but random entries match it.
- FX and rates are negative.
- Roll gaps (`big_gaps`) concentrate in livestock, PL/PA and 6M.

**Across about 580 non-crypto instruments since 2000, the 20-day breakout shows no timing edge.** The only market where it does is crypto (daily +0.78R over random entries, TB 05), and every crypto test so far starts in 2019–2020.

## 1. Question

Does the BTC edge exist before 2020, through the 2014–15 and 2018 bear markets, or is it a feature of the 2020–2026 cycle?

## 2. Design (`turtle_btc_history.py`, `tools/download_tb17_btc.py`)

**Data.** Yahoo `BTC-USD` daily, from 2014-09-17, its first bar. It is a spot index traded 7 days a week, so every calendar day is a session.

**Rules.** TB 01 exactly: one account, 2% of HWM per trade, notional ≤ 3× equity, long and long + reverse (TB 07).
- Fills: O→H→L→C as TB 14–16.
- Costs: 6 bp per side (taker 0.05% + 1 bp, as TB 01); 15 bp sensitivity.
- No funding (spot) and no borrow on shorts. Before 2019 a BTC short meant margin lending or BitMEX swaps, whose costs are not modelled.

**Windows.**

| | Window | Role |
|---|---|---|
| **PRE** | 2014-09-17 → 2019-12-31 | **decision**: data no earlier test used |
| FULL | 2014-09-17 → last bar | reported |
| POST | 2020-01-01 → last bar | reported; overlaps the data of TB 01–12 and checks that Yahoo daily reproduces the perpetual result |

Each window starts a fresh account after 30 days of warm-up.

## 3. Decision (binding, PRE)

Bonferroni over 2 tests ($\alpha = 0.025$ each).

| | Test |
|---|---|
| **G1** timing | mean net R of the real longs − mean R of random-entry longs (same exits, costs and entry rate; 1,000 runs) **≥ 0.48R** |
| **G2** reversal | Calmar(long + reverse) − Calmar(long): 21-day circular block bootstrap, lower bound (2.5%) > 0 |
| **Verdict** | `EDGE_AND_REVERSAL` / `TIMING_EDGE_ONLY` / `REVERSAL_ONLY` / `NO_EDGE` |

**Why 0.48R.** With a single instrument there is no cross-section to average out the fill artefact and path noise.
- **Calibration:** on 300 driftless synthetic BTC-like series (4% daily vol, 48 intraday steps, no overnight gap, 1,932 days, the length of PRE), real − random averaged +0.078R, sd 0.183.
- **97.5% quantile: 0.48R.** That is the threshold.
- For FULL (4,425 days, 150 paths) the same quantile is 0.33R; reported only.
- For scale: daily crypto 2020–26 showed +0.78R (TB 05).

**Power.** PRE holds about 45 long trades, so only an edge similar to 2020–26's can pass. A smaller real edge would show as a positive difference below the threshold. The regime table then shows where it comes from.

**Reported:**
- **Each window:** trades, real vs random R and empirical p, mean R of reverses, CAGR / max DD / Calmar / Sharpe for long, long + reverse and buy and hold, and the 15 bp sensitivity.
- **By regime (FULL)**, using cycle labels dated ex post from BTC's tops and bottoms:

| Regime | Dates |
|---|---|
| bear 2014 | to 2015-01-14 |
| base 2015 | to 2015-10-31 |
| bull 2015–17 | to 2017-12-17 |
| bear 2018 | to 2018-12-15 |
| bull 2019–21 | to 2021-11-10 |
| bear 2022 | to 2022-11-21 |
| bull 2023–25 | to 2025-10-06 |
| after the 2025 top | to the end |

  For each regime: long trades, mean and median R, win rate, random-entry R (200 runs), **timing R** (real − random), reverse trades and R, and each variant's return and max DD inside the regime.
- **Calendar-year returns.**
- **Figures:** equity and drawdown with bear phases shaded, R per trade over time, and real vs the random distribution.

**What the answer to Ivan's question looks like.**
- *Edge only in bull markets:* positive timing R in the bull regimes, about zero or negative in the bears, and G1 fails on PRE.
- *Edge that holds in bears:* positive timing R in bear 2014, bear 2018 and bear 2022. Note that in a bear market a long-only turtle can have positive timing R while losing money, because random longs lose more.

## 4. Validation

`tests/test_turtle_btc_history.py`: 5 tests, all passing.
- The regimes are contiguous and labelled.
- The PRE window is truncated at 2019-12-31.
- The cost override lowers equity.
- Random entries by regime.
- The 0.48R threshold keeps false passes on driftless series at ≤ 2 of 12.

**Smoke run** (a synthetic series with drift): `NO_EDGE`; the notebook runs end to end.

## 5. Run (PC, Windows CMD)

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_turtle_btc_history.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebooks\Turtle_Breakout\17_btc_since_2014.ipynb
git add notebooks\Turtle_Breakout
git commit -m "Turtle Breakout 17: run"
git push
```

- The first run downloads `BTC-USD` to `D:\MarketData\Stocks\tb17\btc_daily.parquet`.
- **Runtime:** a few minutes, mostly 3 × 1,000 random-entry runs and 200 regime runs.

**Outputs** go to `notebooks/Turtle_Breakout/17/`: `17_decision.json`, `17_windows.csv`, `17_regimes.csv`, `17_yearly.csv`, `17_trades_full.csv`, `17_curves.csv`, `17_null.npz`, `17_manifest.json`.
