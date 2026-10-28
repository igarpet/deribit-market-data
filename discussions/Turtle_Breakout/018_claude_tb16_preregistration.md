# 018 — Claude: TB 15 readout and TB 16 pre-registration — the turtle on futures, each contract in its own account

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-28 |
| **Branch** | Turtle Breakout |
| **Responds to** | TB 15 run `b002265`; Ivan, 2026-10-28: test option 1 first — the turtle on futures, its native market |
| **Status** | pre-registration + code (validated on synthetic data); awaiting Ivan's run |

## 0. TB 15 in brief (run `b002265`): `NO_EDGE` on S&P 500 stocks

500 stocks, each in its own account, TB 01 rules (2% of HWM), 2000 → 2026.

**G1 (timing).**
- Real longs +0.115R, random-entry longs +0.097R: difference **+0.018R**.
- The z-test is significant (z 5.9), but the difference is below the 0.10R floor and below the +0.04R fill artefact. **No timing edge on stocks.**
- For scale, daily crypto showed +0.78R (TB 05).

**G2 (reversal).** The reversal improves Calmar on 24/500 stocks (p = 1); reverse shorts average −0.23R.

**Median stock:**

| | CAGR | Max DD | Calmar | Sharpe |
|---|---|---|---|---|
| Long | 1.4% | −28% | 0.05 | 0.21 |
| Long + reverse | −0.8% | −54% | −0.02 | 0.02 |
| Buy and hold | 11.6% | −68% | 0.18 | 0.52 |

- Long beats buy and hold on Calmar for 14% of stocks.
- **Average of the 500 accounts over about 26 years:** long ×1.43, long + reverse ×0.46, hold ×19.7.
- **Every GICS sector** has the same picture.
- **Reported:** SPY longs made +0.31R against +0.07R for random entries (162 trades). QQQ, DIA and IWM were weaker.

**Conclusion.** The turtle's crypto edge is not a general property of prices. Single stocks show none, which matches the literature (stocks mean-revert more; trend-following is documented in futures). A stock portfolio has nothing to combine.

## 1. Question

Does the turtle, with exactly the TB 01 rules, have a timing edge on the market it was designed for? And does the TB 07 reversal help there?

## 2. Universe and data (`16/16_universe.csv`, `tools/download_tb16_futures.py`)

**FUT (decision): 41 continuous front-month futures on Yahoo (`=F`).**

| Sector | Contracts |
|---|---|
| Energy | CL, BZ, NG, HO, RB |
| Metals | GC, SI, HG, PL, PA |
| Grains | ZC, ZW, KE, ZS, ZM, ZL |
| Softs | KC, SB, CC, CT, OJ |
| Livestock | LE, GF, HE |
| Rates | ZB, ZN, ZF, ZT |
| FX | 6E, 6J, 6B, 6A, 6C, 6S, 6M, 6N, DX |
| Equity indices | ES, NQ, YM, RTY |

- **Window:** 2000 (or first session) + 30 sessions → last session.
- **Per contract:** each uses its own trading calendar (sessions as consecutive periods; futures holidays differ), and O→H→L→C fills as TB 14/15.
- **Costs:** 2 bp per side (5 bp sensitivity); no borrow, since a short future costs the same as a long; no funding; no interest on margin cash.

**ETF (reported, not decided): 18 futures-proxy ETFs.**
- GLD, SLV, CPER, USO, UNG, DBA, DBC, TLT, IEF, SHY, FXE, FXY, FXB, FXA, FXC, UUP, EEM, EFA.
- 5 bp per side; borrow 0.5%/yr.

**Data caveats.**
1. **Roll gaps.** Yahoo `=F` series splice front contracts *without back-adjustment*, so each roll can create an artificial gap (contango or backwardation) that can trigger a false breakout or stop. Back-adjusted continuous series, the turtles' standard, are not free.
   - **Diagnostic:** sessions per contract with an open gap > 3 ATR (`big_gaps`).
   - The ETFs roll internally without price gaps but carry roll costs; they are the cross-check.
2. **Negative prices** (WTI, April 2020) are dropped.
3. **No survivorship issue:** these are standard contracts, all still listed.

## 3. Rules (unchanged, as TB 15)

- Per contract, one account.
- Risk 2% of HWM per trade, units $= 0.02\,\mathrm{HWM}/(2\,\mathrm{ATR})$, notional ≤ 3× equity.
- Entry at the 20-day high; exit at the higher of entry − 2 ATR and the 10-day low.
- Long + reverse adds the TB 07 short at every losing stop exit.

## 4. Decision (binding, FUT)

Bonferroni over 2 tests, $\alpha = 0.025$ each, one-sided.

| | Test |
|---|---|
| **G1** timing edge | pooled mean net R of real longs − random-entry longs (same exits, costs, per-contract entry rate, 50 runs) **≥ 0.10R**, **and** the per-contract differences exceed the calibrated fill artefact of +0.045R (t-test, p < 0.025) |
| **G2** reversal | contracts with Calmar(long + reverse) > Calmar(long): sign test p < 0.025 |
| **Verdict** | `EDGE_AND_REVERSAL` / `TIMING_EDGE_ONLY` / `REVERSAL_ONLY` / `NO_EDGE` |

**Why G1 changed from TB 15.** TB 15's z-test compares the real mean with random entries *on the same price paths*, so it ignores the path noise across instruments.
- With 500 stocks that did not matter: the floor decided.
- With 8 synthetic driftless ETFs in the smoke run it passed spuriously.

**Calibration (20 sets × 41 driftless synthetic contracts, 26 years, daily vol 1–3%):**
- Per-contract real − random averaged +0.045R (sd 0.086).
- The pooled difference ranged 0.021–0.065R (99th percentile 0.064R), so it never reached 0.10R.
- The paired t-test against 0.045 rejects in 5% of sets, as designed.
- `decide_paired` is new; TB 15's `decide` is unchanged.

**Reported:**
- per contract: years, big gaps, long trades, mean R real vs random, win rate, share of R ≥ 3, mean R of reverses, and CAGR / max DD / Calmar for long, long + reverse and buy and hold;
- distributions; shares; per-contract timing sign test;
- by sector;
- 5 bp sensitivity;
- average of the per-contract accounts (each 1/N; not a portfolio test);
- the ETF set with the same statistics.

**Caution on ETFs:** with fewer instruments and shorter histories, the 8-ETF synthetic smoke run passed G1 by chance. ETF G1 numbers are descriptive.

**Prior.**
- **G1:** trend-following's long record in futures (time-series momentum) makes it the most likely place for G1 to pass. Note, though, that it is a long-only 20-day breakout with random-entry nulls, not a diversified long/short CTA book.
- **G2:** the reversal is a short-side trend bet after a failed breakout. On futures, where downtrends are as tradable as uptrends (energy 2008, 2014, 2020; rates 2022), it has its best chance outside crypto.
- **Expected:** G1 PASS is plausible; G2 FAIL likely.

## 5. Validation

`tests/test_turtle_stock_accounts.py`: 8 tests, all passing (7 from TB 15 + `decide_paired`: a clear edge passes; the artefact alone does not).

The downloader was tested against a mocked yfinance (59 tickers, renaming to `futures_daily.parquet`).

**Smoke run** (8 + 8 synthetic instruments): FUT `NO_EDGE`; the notebook runs end to end.

## 6. Run (PC, Windows CMD)

The notebook downloads the data on its first run if it is missing; about 59 tickers, a few minutes.

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_turtle_stock_accounts.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 notebooks\Turtle_Breakout\16_futures_own_accounts.ipynb
git add notebooks\Turtle_Breakout
git commit -m "Turtle Breakout 16: run"
git push
```

**Runtime:** roughly 5–10 minutes after the download.

**Outputs** go to `notebooks/Turtle_Breakout/16/`: `16_decision.json`, `16_per_instrument.csv`, `16_distributions.csv`, `16_summary.csv`, `16_sectors.csv`, `16_average_of_accounts.csv`, `16_manifest.json`. The data goes to `D:\MarketData\Stocks\tb16\`.
