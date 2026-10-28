# 022 — Claude: TB 19 readout and TB 20 pre-registration — flat-day legs held with turtle exits (long after a sharp drop, short at a stalled peak)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-28 |
| **Branch** | Turtle Breakout |
| **Responds to** | TB 19 run `61b9a50`; Ivan, 2026-10-28: hold with the turtle's rules (no TP, trailing stop), allow shorts at a flat peak, turtle sizing — **no time limit on trades** |
| **Status** | pre-registration + code (validated on synthetic data); awaiting Ivan's run |

## 0. TB 19 in brief (run `61b9a50`): `NO_TRADABLE_REVERSAL`

The run used the registered parameters (q 20%, 365 FLAT-day lookback, 1-day hold, 6 bp per side; commit `4f7087e`).

| Gate | Trades | Mean net per trade | Win rate | Mean R vs random FLAT longs | p (need < 0.0167) |
|---|---|---|---|---|---|
| G1 Yahoo 2014–19 | 189 | +0.57% | 59% | +0.024 vs −0.017 | 0.074 |
| G2 Yahoo 2020– | 299 | +0.42% | 55% | +0.014 vs −0.011 | 0.119 |
| G3 Binance perp | 273 | +0.39% | 56% | +0.014 vs −0.011 | 0.102 |

- **Same sign everywhere, none significant.** It is not a Yahoo close artefact (Binance agrees), but the effect is too small for the trade count.
- **Accounts** at the turtle's 2% unit: reversal alone 1.3%/yr (DD −12.6%) on Yahoo, 1.0%/yr (DD −10.1%) on Binance. Correlation with the turtle ≈ 0.
- Combined (half each), Binance: 8.0%/yr, DD −7.8% vs turtle alone 13.1%, −14.7%. Lower drawdown, mostly by halving turtle exposure.

## 1. The change Ivan asked for

TB 19 traded the 1-day reversal and exited the next close. TB 20 keeps the same kind of entry but **holds with the turtle's exits and no time limit**, and adds a short leg.

## 2. Rules (frozen; `turtle_flat_legs.py`, `LParams`)

**Common to both legs.**
- Signals only at the close of a FLAT day $t$ (the TB 01 long-only turtle holds no long at that close). Entry at $C_t$.
- **Exits = the turtle's, nothing else:**
  - fixed stop at entry $\mp 2\,\mathrm{ATR}_t$ (Wilder 20 through day $t$);
  - trailing stop at the 10-day low (long) / 10-day high (short) of completed days;
  - the closer of the two applies; fill at the stop, or at the open if the day gaps through it.
  - **No take-profit. No time limit: a trade stays open until a stop is hit, however long that takes.** Open trades at the end of data are marked `end` and excluded from the gate.
- One position per leg at a time; signals while a position is open are ignored.
- **Size = turtle:** units $= \min\!\big(0.02\,\mathrm{HWM}/(2\,\mathrm{ATR}_t),\ 3\,E/C_t\big)$, each leg in its own account.
- Costs 6 bp per side; on the perpetual, funding paid/received for each held day (shorts receive positive funding).
- $R = \text{net return} / (2\,\mathrm{ATR}_t/C_t)$.

**Leg A (long, after a sharp drop).** $r_t = \ln(C_t/C_{t-1})$ at or below the 20th percentile of $r$ over the previous 365 FLAT days (≥ 100, strictly before $t$). This is TB 19's signal unchanged.

**Leg B (short, at a stalled peak).** Both:
- $C_t \ge \max(H_{t-54},\dots,H_t) - \mathrm{ATR}_t$ (within 1 ATR of the 55-day high);
- $\big|\ln(C_t/C_{t-20})\big| \le 0.5\,\sigma_{20}$, with $\sigma_{20} = \sqrt{20}\times$ sd of daily log returns over the previous 365 days (the 20-day move is at most half its usual size: price is high but has stopped rising).

## 3. Decision (binding)

Per leg, three windows: **Yahoo 2014–19**, **Yahoo 2020–**, **Binance perpetual** (daily UTC bars from 1m klines, its own turtle state and funding).

| | Test |
|---|---|
| Null | the same leg's trade from **every** FLAT day (same direction, exits, costs), 10,000 draws of the real trade count |
| Window passes | mean net R of real trades > null with **p < 0.01**, **and** mean net return per trade > 0 |
| Leg verdict | `EDGE` only if all three windows pass; otherwise `NO_EDGE` |

**Why p < 0.01.** These windows have now been used by TB 17–19; this is a third rule on the same days, so the bar is higher than TB 19's 0.0167.

**Reported:** trades, hold-day distribution, exit reasons, top-5 share of R, results by regime (Yahoo) and by year (Binance); accounts for turtle, A, B, turtle + A, turtle + B, turtle + A + B (equal capital per sleeve), buy and hold; daily correlations; curves.

## 4. Prior, stated in advance

1. **Horizon mismatch.** TB 18 found reversion at 1 day only (VR back to 0.96 at 20 days). A trailing-stop hold turns leg A into a trend bet entered after a drop; its edge depends on whether drops on FLAT days start new up-moves, which TB 18 did not test. Expected: most trades stop out within days at about −1R (the 2 ATR stop is wide relative to the +0.35% day-1 bounce), with a few long winners.
2. **Leg B fights BTC's drift** (about +56%/yr over 2014–26). FLAT-day drift is about −1%/yr, which helps, but a short with no time limit and a trailing high can still be carried into a breakout. The turtle will usually go long on that same breakout, so B partly hedges the turtle.
3. **Leg B is rare.** Its two conditions together fire on a few percent of FLAT days; expect tens of trades, so low power, and a pass needs a large effect.
4. **Expected:** both legs `NO_EDGE`. Value, if any, would show as lower drawdown in the combined accounts, not as an edge.

## 5. Validation

`tests/test_turtle_flat_legs.py`: 6 tests, all passing.
- No time limit: a trade in a steady trend stays open until the trailing stop is hit.
- Fixed stop and gap fills.
- Short funding sign and costs.
- Leg B signal conditions.
- One position per leg; turtle sizing and the 3× cap.
- A random walk passes neither leg.

**Smoke run** (synthetic random walk with drift): both legs `NO_EDGE`; leg A median hold 1 day (p90 13), leg B 20 trades. The notebook runs end to end.

## 6. Run (PC, Windows CMD)

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_turtle_flat_legs.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks\Turtle_Breakout\20_flat_day_legs.ipynb
git add notebooks\Turtle_Breakout
git commit -m "Turtle Breakout 20: run"
git push
```

- Uses the TB 17 Yahoo file and `BTCUSDT_um_1m.parquet` plus the funding file from the TB 01 data folder (as TB 19).
- **Runtime:** a few minutes, mostly reading the 1m file.

**Outputs** go to `notebooks/Turtle_Breakout/20/`: `20_decision.json`, `20_trades_{yahoo,binance}_{A,B}.csv`, `20_accounts_{yahoo,binance}.csv`, `20_manifest.json`.
