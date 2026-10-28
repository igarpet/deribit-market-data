# 021 — Claude: TB 18 readout and TB 19 pre-registration — a 1-day reversal rule on turtle-flat BTC days

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-28 |
| **Branch** | Turtle Breakout |
| **Responds to** | TB 18 run `5cd4f64`; Ivan, 2026-10-28: try the reversal rule |
| **Status** | pre-registration + code (validated on synthetic data); awaiting Ivan's run |

## 0. TB 18 in brief (run `5cd4f64`): `MR_PRESENT`, but only at a 1-day horizon

FLAT days are 59% of all days since 2014. IN days behave the opposite way, so the turtle's state works as a regime filter.

| | FLAT days | IN days |
|---|---|---|
| AC1 | **−0.091** (95% band ±0.039) | +0.082 |
| VR(2) | 0.90 (p 0.0005) | 1.09 |
| VR(5) | 0.90 (p 0.014) | 1.13 |
| VR(10) | 0.89 (p 0.051) | 1.24 |
| VR(20) | 0.96 | 1.30 |
| REV(1,1) | **−0.49%/day** (p 0.012) | +0.84% |
| REV(5,5) | −0.32% (p 0.36) | +1.89% |
| Drift | −1%/yr | +109%/yr |

**What the reversion looks like on FLAT days.**
- After a bottom-quintile day (mean −5.1%): next day **+0.35%**, up 58% of the time.
- After a top-quintile day (+4.2%): next day −0.15%.
- No reversion toward the 20-day mean: next-5-day returns are flat across z buckets.
- **By trend:** down 35% of all days (VR p 0.004, REV(1,1) p 0.031), range 13%, up 10%; none of these passes alone.

**Caveats:**
- The permutation null ignores volatility clustering.
- Yahoo's close is a composite index, so the reversal could be a close artefact. G3 below checks this.

## 1. The rule (frozen; nothing here is tuned)

- **Signal:** at the close of day $t$, the turtle (TB 01, long only) holds no long at that close (FLAT) **and** $r_t = \ln(C_t/C_{t-1})$ is at or below the 20th percentile of $r$ over the previous 365 FLAT days (≥ 100 needed, strictly before $t$).
- **Trade:** buy at $C_t$, sell at $C_{t+1}$. Long only, one day, no stop.
- **Size:** notional $= \min(0.02/(2\,\mathrm{ATR}_t/C_t),\ 3) \times$ equity, which is the turtle's 2% risk unit (ATR = Wilder 20 through day $t$).
  - Results are also given in $R = \text{net return} / (2\,\mathrm{ATR}_t/C_t)$.
- **Costs:** 6 bp per side. On the perpetual, the long pays the funding of the holding day.
- **The 20% threshold** is TB 18's quintile, so the rule trades exactly the observation that TB 18 tested.

## 2. Data and gates

| Gate | Data | Turtle state |
|---|---|---|
| **G1** | Yahoo `BTC-USD`, entries 2014-10 → 2019-12 | TB 17 FULL run |
| **G2** | Yahoo `BTC-USD`, entries 2020-01 → end | TB 17 FULL run |
| **G3** | Binance BTCUSDT perpetual, daily bars by UTC day from 1m klines, 2019-09 → end | its own turtle run on those bars |

**Each gate passes only if both hold:**
- the mean net R of the rule's trades is above the mean R of 1-day longs on random FLAT days (pool = all FLAT days, same trade count, 10,000 draws), with p < 0.05/3;
- the mean net return per trade > 0.

**Verdict:** `REVERSAL_EDGE` if G1, G2 and G3 all pass; otherwise `NO_TRADABLE_REVERSAL`.

**What it can and cannot show.** All three windows were used by TB 18 to find the pattern, so this is an *in-sample* check.
- **It can show** that the rule survives costs, holds in both halves, and is not specific to Yahoo's composite close (G3).
- **It cannot show** that the pattern will persist. A pass would have to be confirmed forward.

**Reported:**
- variants (q = 10% and 30%, 2-day hold, 15 bp per side, the short leg after top-quintile days), per half and source;
- by year;
- accounts: reversal alone, turtle alone, combined with half the capital in each, and buy and hold;
- the daily correlation of reversal and turtle;
- curves.

## 3. Prior, stated in advance

- **Size:** REV(1,1) of −0.49% splits into about +0.35% on bottom-quintile days, and the cost is 0.12% a round trip. The expected net is roughly +0.2% per trade, about 40 trades a year.
- **Account effect:** at the turtle's 2% unit (BTC daily ATR ≈ 4%, so notional about 0.25× equity), that adds only about 2% a year. That is small next to the turtle.
- **Likely failure modes:**
  - the effect is a Yahoo close artefact, in which case G3 fails;
  - it lives only in 2014–2019, in which case G2 fails.
- The combined account is where any value would show: lower drawdown through near-zero correlation, not higher return.

## 4. Validation

`tests/test_turtle_reversal1d.py`: 5 tests, all passing.
- **Signals are causal:** changing future data leaves past signals unchanged. They fire on about 20% of FLAT days and never on IN days.
- **Trade math:** gross, costs and funding sign, long and short.
- **A random walk fails the gate; an AR(1) with $\phi = -0.25$ passes it.**
- Account curve; daily bars from 1m; daily funding sums.

**Smoke run** (synthetic random walk with drift): `NO_TRADABLE_REVERSAL`; the notebook runs end to end.

## 5. Run (PC, Windows CMD)

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_turtle_reversal1d.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks\Turtle_Breakout\19_flat_day_reversal.ipynb
git add notebooks\Turtle_Breakout
git commit -m "Turtle Breakout 19: run"
git push
```

- It uses the TB 17 Yahoo file and `BTCUSDT_um_1m.parquet` plus the funding file from the TB 01 data folder.
- **Runtime:** a few minutes, mostly reading the 1m file.

**Outputs** go to `notebooks/Turtle_Breakout/19/`: `19_decision.json`, `19_variants.csv`, `19_trades_{yahoo,binance}.csv`, `19_accounts_{yahoo,binance}.csv`, `19_manifest.json`.
