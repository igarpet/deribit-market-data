# 020 — Claude: TB 17 readout and TB 18 pre-registration — is BTC mean-reverting while the turtle is out?

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-28 |
| **Branch** | Turtle Breakout |
| **Responds to** | TB 17 run `c0b2690`; Ivan, 2026-10-28: the turtle is a trend strategy, so look for a mean-reverting strategy for the time it is out of the market. Diagnostic first |
| **Status** | pre-registration + code (validated on synthetic data); awaiting Ivan's run |

## 0. TB 17 in brief (run `c0b2690`): `TIMING_EDGE_ONLY`; the BTC edge predates 2020

**Timing vs random entries.**

| Window | Real longs | Random longs | Difference | Gate |
|---|---|---|---|---|
| PRE 2014–19 (unused data) | +3.30R | +1.54R | **+1.76R** | threshold 0.48R, p ≈ 0.001: **G1 PASS** |
| FULL | | | +0.92R | |
| POST | | | +0.49R | |

**Reversal.** Calmar +0.04, lower bound −0.30: **G2 FAIL**.

**FULL window:**

| | CAGR | Max DD | Calmar | Sharpe |
|---|---|---|---|---|
| Long | 29.9% | −24.6% | **1.22** | **1.22** |
| Buy and hold | 56.0% | −83.4% | 0.67 | 0.84 |

Mean exposure is 0.18×, which is why raw CAGR is lower.

**Timing R by regime.**
- Positive in the bulls: 2015–17 **+3.11**, 2019–21 **+1.25**, 2023–25 +0.46; base 2015 +0.53; after the 2025 top +0.62.
- Slightly negative in the bears, with few trades: 2014 −0.61 (1 trade), 2018 −0.09 (6), 2022 −0.40 (9).
- **Bear-market losses were small:** 2018 −9.7% vs −83% for buy and hold; 2022 −12% vs −76%.

**Concentration.** The 5 best trades carry 53% of long R.

**Reversal by regime.** The reversal added +8.1R in bear 2022 and lost 4.2R in bull 2023–25: a bear-market hedge, as in crypto 2020–26.

**Conclusion.** A real trend-capture edge on BTC across three cycles, earned in bull runs; the protection comes from stopping out quickly when breakouts fail.

## 1. Question and the case against the idea, stated in advance

Ivan's idea: a mean-reversion strategy for the days the turtle is out of the market. Before designing one, this checks whether those days are mean-reverting at all.

**Prior (Claude, in chat on 2026-10-28):**
1. Many FLAT days fall in downtrends (2018, 2022), where buying dips is the wrong trade.
2. Capital is idle even while the turtle trades (0.18× exposure), so a second strategy would not need to fit into the turtle's gaps.
3. Crypto mean reversion failed in VWAP Drift and was near a random walk intraday (TB 05).

## 2. Design (`turtle_flat_diag.py`)

**Data.** The TB 17 Yahoo `BTC-USD` daily file, 2014-09-17 → last bar; close-to-close log returns $r_t$.

**State of day $t$.**
- From the TB 17 FULL long-only run (TB 01 rules): **IN** if a long is held at the close ($\text{entry} \le t < \text{exit}$), **FLAT** otherwise, after the 30-day warm-up.
- FLAT days are also labelled by a causal trend filter:
  - `down`: close < MA50 and 20-day return < 0;
  - `up`: close > MA50 and 20-day return > 0;
  - `range`: otherwise.

**Statistics, conditioned on the state of day $t$** (what a strategy knows at $t$):
- **Variance ratios** of the next $q$ days, $q$ = 2, 5, 10, 20 (below 1 = mean reversion, above 1 = trend):
$$\mathrm{VR}(q) = \frac{\mathrm{Var}(r_{t+1} + \dots + r_{t+q})}{q\,\mathrm{Var}(r_{t+1})}$$
- **Autocorrelations** $\mathrm{AC}(k) = \mathrm{corr}(r_t, r_{t+k})$, $k$ = 1…10.
- **Reversal spreads** $\mathrm{REV}(k,k)$, $k$ = 1, 5: mean next-$k$-day return of the top quintile of the past $k$-day return minus the bottom quintile (negative = reversion).
- **z-score table:** next-5-day return by bucket of $z = (C - \mathrm{MA}_{20}) / (\mathrm{MA}_{20}\,\sigma_{20})$.

**Null.** The daily returns are permuted 2,000 times with the state mask kept, which destroys serial dependence. Limitation: the permutation also destroys volatility clustering, so p-values for VR are approximate.

## 3. Decision (binding, FLAT days)

**`MR_PRESENT`** if both hold:
- **(a)** $\min_q$ p(VR(q) < 1) < 0.0125 (Bonferroni over 4), **and**
- **(b)** $\min_k$ p(REV(k,k) < 0) < 0.025 ($k$ = 1, 5; Bonferroni over 2).

Otherwise **`NO_MEAN_REVERSION`**, and no mean-reversion strategy is built on BTC FLAT days.

**Why two reversal horizons.** REV(5,5) alone has little power against lag-1 reversion: an AR(1) with $\phi = -0.25$ gives REV(5,5) p ≈ 0.19. REV(1,1) catches it.

**Reported:**
- the same statistics for IN days, all days, and FLAT days split by trend;
- the share of FLAT days in each trend label;
- FLAT spell lengths; FLAT share by year;
- reversal and z-score tables;
- price with the IN periods shaded; VR curves by state.

**If `MR_PRESENT`:** the next step is a pre-registered mean-reversion rule on FLAT days (TB 19), judged against random entries. **If `NO_MEAN_REVERSION`:** the options raised in chat remain, combining the turtle with Ivan's fOU pairs book, or the forward shadow test.

## 4. Validation

`tests/test_turtle_flat_diag.py`: 8 tests, all passing.
- Forward and past sums, including a leading NaN.
- A random walk gives `NO_MEAN_REVERSION`, with VR ≈ 1.
- AR(1) with $\phi = -0.25$ gives `MR_PRESENT` (VR5 < 0.85), also with a leading NaN.
- AR(1) with $\phi = +0.25$ gives VR > 1.15 and no MR.
- The mask conditions correctly: reverting days vs trending days.
- Turtle state and trend labels.
- The tables.

**Smoke run** (synthetic random walk with drift): `NO_MEAN_REVERSION`, VR ≈ 1 in every state; the notebook runs end to end.

## 5. Run (PC, Windows CMD)

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_turtle_flat_diag.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks\Turtle_Breakout\18_flat_days_diagnostic.ipynb
git add notebooks\Turtle_Breakout
git commit -m "Turtle Breakout 18: run"
git push
```

- It uses the TB 17 data file.
- **Runtime:** a few minutes (6 states × 2,000 permutations).

**Outputs** go to `notebooks/Turtle_Breakout/18/`: `18_decision.json`, `18_stats_by_state.csv`, `18_composition.csv`, `18_state_daily.csv`, `18_manifest.json`.
