# Funding Carry 004 — FC 02 Stage A readout: funding is predictable, by persistence alone; the engine adds nothing

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-18 |
| **Branch** | Funding Carry |
| **Responds to** | FC 003 pre-registration; Ivan's run 2952fd5 |
| **Status** | Result. Registered run: 30 assets; OI coverage 100%, so all blocks were used. The verdicts in §1–§3 are binding. §4 is post hoc. |

## 1. Q1: predictable? **Yes. $R^2_{OOS} = 0.51$ at 7 days, from persistence alone**

TEST: 2025-10-01 → 2026-09-30, 10,641 asset-days.

| Target | Persistence P | Full model (all blocks) |
|---|---|---|
| next 1 day | 0.537 | 0.537 |
| **next 7 days** | **0.512** [ΔMSE vs FE CI excludes 0] | 0.507 |
| next 30 days | 0.456 | 0.361 |

**Coefficients (7d):**

$$\hat y_7 = 0.014 + 0.287 f_1 + 0.425 f_7 + 0.097 f_{30} + \text{asset FE}$$

The persistence weights sum to 0.81, so forecasts revert about 19% toward each asset's mean.

**By asset:** $R^2$ ranges from 0.75–0.78 (BTC, ETH, XRP, APT) down to below zero (SUI −0.22, NEAR −0.15).

**"By how much", as realized next-7-day funding (%/yr) per quintile of the TEST forecast:**

| Quintile | Predicted | Realized | Share negative |
|---|---|---|---|
| 1 (lowest) | −12.0 | **−11.4** | 82% |
| 2 | −1.3 | −2.1 | 55% |
| 3 | +1.9 | +0.5 | 32% |
| 4 | +4.1 | +2.4 | 17% |
| 5 (highest) | +6.8 | **+4.1** | 10% |

## 2. Q2: which drivers add? **None beyond persistence**

Premium, momentum, OI/positioning, breadth, naive volatility and the engine all fail to add in the nested sequence at every horizon, both in sequence and as unique contributions.

The one exception is the premium's unique contribution at 30 days, which is tiny (ΔMSE +0.00016). The OI block **hurts** at 30 days (R² 0.456 → 0.378).

The premium adding nothing is expected in hindsight: funding is computed from the premium, so $f_1$ already contains it.

## 3. Q3: the volatility engine. **Q3a no; Q3b no.** Not useful for carry

**Q3a: next-24h / next-7d spot RV, QLIKE on TEST**

| | 24h | 7d |
|---|---|---|
| ENGINE | 1.853 | 1.412 |
| NAIVE | 1.686 | 1.916 |
| HAR_D | **1.339** | **1.206** |
| Engine better than NAIVE and HAR_D? | no | no; HAR_D significantly better [CI −0.387, −0.011] |

**Q3b:** adding E to base + V makes funding forecasts slightly *worse* at all horizons (ΔMSE CI just below 0).

## 4. Post hoc (not binding): the crash, and what the engine really is

The means in §3 are dominated by **10 October 2025**, the liquidation crash. On that day, altcoin spot RV over the next 24h was 300–4,800× the forecast for every model (TON's QLIKE alone is 4,771).
- **Medians (24h):** ENGINE 0.058, NAIVE 0.064.
- **Excluding 3–10 Oct origins:**

| | 24h | 7d |
|---|---|---|
| ENGINE | 0.175 | 0.205 |
| NAIVE | 0.236 (engine better, CI [0.042, 0.075]) | 0.291 (n.s.) |
| HAR_D | **0.174** (tie, CI [−0.014, 0.015]) | **0.173** (HAR_D better, CI [−0.068, −0.002]) |

- **BTC/ETH/SOL/XRP only, crash included:** the same pattern. The engine beats naive at 24h (0.332 vs 0.503) and ties HAR_D (0.334); HAR_D is better at 7d.

**Reading (opinion):**
- The engine is a sound 24h forecaster: Mincer–Zarnowitz slope 0.94, clearly better than naive.
- It is **not better than a properly calibrated daily HAR**.
- Beyond 24h it is worse, because its longest lag is one day.

VD20's large engine advantage over daily HAR (0.200 vs 0.296) needs revisiting: VD20's own notes flag that its HAR_D level was off by about 1.6×. Here HAR_D includes a TRAIN smearing correction, and the gap disappears.

## 5. Consequences (opinion)

1. **For carry:** predictability doesn't rescue the long-carry trade. Even the top forecast quintile realized only **+4.1% a year** gross funding per unit of notional in TEST. On 1.5 units of capital that is 2.7% before costs, below cash. Stage B, a long-carry timing rule, has no room to pass in the current regime, so I don't recommend building it.
2. **The bottom quintile is the visible asymmetry:** −11.4% a year realized, negative 82% of the time. A **reverse carry** (short spot, long perpetual) would collect that. Two caveats:
   - it needs borrowed spot, whose cost and availability on Binance Margin are not in our data;
   - negative funding coincides with heavy short demand, when borrow rates are typically highest.

   It is the only untested idea this produced. It should be checked against actual borrow-rate history before any backtest.
3. **For the engine:** treat AM-T-HAR-Blend as a well-calibrated 24h variance forecaster equivalent to a daily HAR, not a superior one. For sizing (the Kelly denominator), either works. For horizons beyond a day, a daily HAR is better.

## Questions for ChatGPT

1. Do you agree VD20's engine-vs-HAR_D gap was a calibration artifact of HAR_D? If so, should VD 041 be amended?
2. Is Binance Margin borrow-rate history obtainable (API or archive) for a reverse-carry feasibility check?

## Decisions needed from Ivan

1. Accept FC 02 Stage A as binding. Skip Stage B (long-carry timing). (Recommended.)
2. Do a feasibility check of reverse carry: borrow-rate history for the negative-funding assets first, then a backtest only if borrow is cheaper than the funding received.
