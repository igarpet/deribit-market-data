# 007 — Claude: TB 05 pre-registration — is the turtle breakout fractal? 1 minute to 1 day

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-25 |
| **Branch** | Turtle Breakout |
| **Responds to** | TB 04 run `98224f4`; Ivan, 2026-10-25: can we fractalize the market, i.e. does the breakout also work on 1-minute, 2-minute … candles? Preference for rapid trades over holding |
| **Status** | pre-registration + code (validated on synthetic data); awaiting Ivan's run |

## 0. TB 04 in brief (run `98224f4`)

**Verdict.** `ENTRY_SIGNAL_CONFIRMED` through the ridge model only. Confirmation AUC 0.605 [0.511, 0.698], permutation p 0.015. No single feature survives FDR (minimum q 0.41).

**Data artefact.** TON is a flat series from 2026-08-07: 77 trades with ATR < 1% and 5-minute holds. Without them the model gives AUC 0.602 [0.507, 0.696], permutation p 0.045.

**Model drivers:**

| Effect on the chance of a winner | Feature |
|---|---|
| higher | 24h volume surge |
| higher | BTC 20-day return |
| higher | 24h basis |
| lower | funding |

**Model-score terciles are non-monotonic** (high / mid / low mean R 2.5 / −0.2 / 1.0). It is not usable as a filter yet.

**Path.** A trade at −0.5R or worse after 5 days ends a winner only 0–7% of the time.

**Exploratory, after TB 04** (not pre-registered):
- No persistent regimes. Monthly mean-R autocorrelation is 0.14.
- The win rate is flat at 29–37% across market states.
- Mean R rises with BTC's 20-day return (0.04 → 0.64) and with the volume surge.

## 1. Question

Two separate questions, measured separately:

1. **Fractal edge.** Does the same rule earn the same edge **in risk units** at every candle size? If the breakout pattern repeats at all scales, the edge per trade, $R_{gross}$ above random entries, should be flat across timeframes.
2. **Tradeable edge.** Fees are a fixed number of bp, but the size of a candle shrinks roughly with $\sqrt{\text{timeframe}}$. So costs in R grow as candles shrink: for BTC, about 0.02R at 1 day, 0.09R at 1h, 0.3R at 5m and 0.7R at 1m. Even a fractal edge can be untradeable at short scales.

**Prior evidence, stated before the run.** Hurst Regimes 02 (price H ≈ 0.5 from 5 min to 7 days), Signed Jumps SJ 02a and 04b (no post-jump continuation from seconds to 60 min), and TB 01/03 (hourly and last-4h roughly break even before costs) all point to the edge decaying at short scales.

## 2. Design (`turtle_fractal.py`)

**Timeframes.** 1, 2, 3, 5, 10, 15, 30 min; 1, 2, 4, 8, 12 h; 1 day. Thirteen in total.
- Candles are UTC-anchored.
- The 1-day candle uses 17:00 New York, as in TB 01.

**Rules.** TB 01 on every timeframe:
- entry on the 20-candle high;
- exit at max(entry − 2 ATR, 10-candle low);
- ATR is the EWM of the true range with α = 1/20;
- long only, one position per asset;
- no re-entry in the candle of an exit.

**Execution and sizing:**
- **Unit notional per trade.** There is no sizing or compounding; the object of study is the edge per trade.
- **Base bars.** 1-minute bars for candles of 1–3 min, 5-minute bars otherwise. A stop is checked from the bar after the entry bar.
- **Fills: the close of the base bar in which the trigger trades.** This changes TB 01's level fills.
  - **Why.** Filling exactly at the level ignores the overshoot of a discrete crossing. On a synthetic martingale this added about +0.07R per trade at 1–60 min, an artefact that would grow as candles shrink and fake a "fractal edge".
  - **Check.** Bar-close fills are unbiased: mean R is 0.006 ± 0.027 at 5 min and 0.014 ± 0.045 at 15 min over 8 martingale seeds, and real − null is 0.009 / 0.001 / 0.003 at 1/2/3 min over 10 seeds.
  - **Daily case.** The entry bars of the daily case reproduce TB 01 exactly.

**Data.** Binance USD-M 1m klines for BTC/ETH/SOL/XRP and funding; window 2020-10-15 → end.

**Per-trade measures:**

| Measure | Definition |
|---|---|
| $R_{gross}$ | $(P_x-P_e)/(2\,\mathrm{ATR})$ |
| $bp_{gross}$ | $10^4(P_x/P_e-1)$ |
| $bp_{net}$ | $bp_{gross}$ − (0.05% + 1 bp) × 2 sides − funding |
| maker scenario | 1 bp per side, reported |
| break-even round-trip cost | mean of ($bp_{gross}$ − funding) |

**Null.** Random entries at candle opens with the real entry rate and the same exits. Runs per timeframe: 50 (1–3 min), 100 (5–15 min), 200 (30 min – 12 h), 400 (1 day).

## 3. Decision (binding)

Per timeframe, with Bonferroni over 13: $a = 0.05/13 = 0.0038$.

| Gate | Rule |
|---|---|
| **G_edge, fractal edge** | $z=(\bar R_{real}-\overline{R}_{null})/\mathrm{sd}_{null}$; one-sided $p=1-\Phi(z)<a$ |
| **G_net, tradeable** | pooled mean $bp_{net}$ (taker) > 0, with a calendar-month cluster bootstrap $(1-a)$ interval whose lower bound is > 0 |

- **Why a z-test for G_edge.** The normal approximation is used because 50–400 null runs cannot resolve $p<0.004$ empirically. The empirical share is reported alongside.
- **Verdicts:** `EDGE_TRADEABLE` (both gates), `EDGE_GROSS_ONLY` (G_edge only), `NO_EDGE`.

**Reported only:**
- real − null $R$ against log timeframe (flat = fractal; decaying = not);
- break-even cost against taker (12 bp) and the maker scenario (2 bp);
- trades per day and median hold;
- per-asset results;
- trades for 30 min and longer.

## 4. Validation (synthetic)

`tests/test_turtle_fractal.py`: 8 tests, all passing.

- **Candles and levels:** candle ids, completed-candle levels, and the first breakout bar are correct.
- **Daily vs TB 01:** daily entry bars equal TB 01's, and exits agree on ≥ 80% of trades.
- **Fills and costs:** fills are bar closes, and the cost arithmetic holds.
- **Unbiasedness:** martingale mean R is within noise.
- **1-minute base:** runs correctly.
- **Daily trend:** a trending synthetic beats the null.
- **Summary:** keys present, and `NO_EDGE` on a martingale.
- **Smoke run.** The notebook runs end to end on 4 synthetic martingales (90 days), giving `NO_EDGE` at all 13 timeframes. The smallest p was 0.009 at 2 min, above $a$.

## 5. Run (PC, Windows CMD)

No new download. Close the notebook in Jupyter first.

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_turtle_fractal.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=28800 notebooks\Turtle_Breakout\05_fractal_timeframes.ipynb
git add notebooks\Turtle_Breakout
git commit -m "Turtle Breakout 05: run"
git push
```

**Runtime.** Long, about 2–4 hours. The 1–3 minute timeframes generate hundreds of thousands of trades, and each null run re-simulates them. Two ways to check it first:
- `set TB05_TFS=5,15,60,240,1440` before the `jupyter` line, for a subset;
- `set TB05_NULL_SCALE=0.2`, for a quick pass.

The decision uses the full run.

**Memory.** About 1–2 GB, for the 1m bars of four coins.

**Outputs** go to `notebooks/Turtle_Breakout/05/`:
- `05_decision.json`, `05_summary.csv`, `05_by_asset.csv`;
- `05_null_runs.csv`, `05_trades_30m_plus.csv.gz`;
- `05_manifest.json`.

## 6. Decisions needed from Ivan

1. Confirm §3 is binding, including the fill change in §2.
2. Run §5.
