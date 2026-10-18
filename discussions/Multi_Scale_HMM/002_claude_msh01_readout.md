# Multi-Scale HMM 002 — MSH 01 readout: no momentum or reversal states at 1h, 4h or 1D

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-18 |
| **Branch** | Multi-Scale HMM |
| **Responds to** | MSH 001 pre-registration; Ivan's run 60ff884 |
| **Status** | Result. Registered run (99 / 199 / 199 null draws). The verdict is binding. |

## 1. Verdict: `NO_TRADEABLE_MOMENTUM_STATES` at every scale

| Scale | K, ν | S: always-in gross bp/bar | Null median | p | Assets > 0 | E: net bp/day [95% CI] | States vs K=1 | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1h | 6, 15 | −0.05 | +0.03 | 0.57 | 2/4 | −0.6 [−3.3, +1.3] | no | NO_SKILL / NO_NET_EDGE |
| 4h | 6, 15 | +0.63 | +0.46 | 0.46 | 3/4 | +2.6 [−1.7, +7.3] | no | NO_SKILL / NO_NET_EDGE |
| 1D | 4, 7 | −20.9 | +1.7 | 0.93 | 1/4 | −13.6 [−34.7, +7.4] | no | NO_SKILL / NO_NET_EDGE |

- **Hit rates on TEST:** 1h 0.49–0.52, 4h 0.47–0.48, 1D 0.45–0.48. Out-of-sample $R^2$ is at or below zero for all 12 asset × scale cells.
- **Prediction recorded in MSH 001 §10:** no skill at 1h and 4h; 1D uncertain. 1h and 4h came out as predicted. 1D failed as well.

## 2. What the HMM found: volatility and bar-shape states, not direction

- **Selection:** VALID likelihood kept improving with $K$ and chose the grid edge ($K=6$) at 1h and 4h. As registered, this is reported and the grid is not extended.
- **What separates the states:** the sign and size of the current bar ($z$, efficiency, close location), the volatility level ($\log\sigma$) and the volatility surprise. Typical examples:
  - a slow high-volatility regime (1h state 0: 17 h duration; 4h state 5: 62 h duration);
  - short "big up bar" and "big down bar" states lasting 1–2 bars.
- **Direction:** **none of the 16 states has a significant momentum or reversal slope.** All $|b_k/\text{se}| < 2$. The largest are 1h states 0 and 2 at about −1.7.
- **Drift:** one state shows a nominal drift: 4h state 4 (after large down bars, 10% occupancy) with $a=-0.145$, $t\approx-2.0$. With 16 states × 2 coefficients tested, one $|t|\approx2$ is the expected number by chance.

Persistent states exist, but they are volatility states. This is the same finding as RS01, HSMM 06–08 and RSD 01–03, now confirmed at horizons 12 to 288 times longer.

## 3. The 1D result: overfitting, not anti-skill

- The 1D HMM has about 275 parameters on 1,220 TRAIN bars.
- The train-to-valid log-likelihood gap is −7.75 vs −9.63 per bar, against about −7.5 vs −8.2 at 1h.
- Its TEST forecast is worse than a coin flip (−20.9 bp/bar), but inside the null spread (sd 15.7 bp).

This was the power problem flagged in MSH 001 §8: 1D does not have enough history for a 10-dimensional, full-covariance, multi-state model.

## 4. Baselines (reported, not gates; post hoc)

| Scale | K=1 slope $b$ (t) on TRAIN+VALID | K=1 always-in gross bp/bar on TEST | TSMOM gross bp/bar | Drift-only |
|---|---|---|---|---|
| 1h | +0.009 (1.3) | +0.17 | +0.83 | +0.02 |
| 4h | +0.024 (2.0) | +1.29 | −0.28 | +0.07 |
| 1D | +0.050 (1.9) | +18.5 | +5.0 | −2.9 |

There is a weak, pooled, linear daily-momentum slope. It is the same sign as the crypto time-series momentum literature, and the K=1 forecast was positive on TEST at 1D.

This is **not** a finding:
- It was not a gate.
- It is one of several baselines reported.
- Its TEST significance is low: about 252 days, a pooled daily sd of roughly 250 bp, so $t\approx1.2$.

If Ivan wants to pursue it, it has to be pre-registered and judged prospectively from today (MSH 001 §8 sets 180 days for 1D). It cannot be promoted from this table.

## 5. Data notes

- Minute data ends **2026-10-11** (not 10-18). TEST is therefore 2026-02-01 → 2026-10-11. The prospective window still starts 2026-10-18 as registered.
- Funding ends **2026-09-30**, so `funding_missing` is True at 4h and 1D. It does not change any verdict, since E fails on P&L at every scale. A rerun with updated funding is not needed.
- The 1h net rule traded only 2–23 times per asset. Forecasts rarely exceeded the 12 bp threshold, which is consistent with $R^2\approx0$.

## 6. Consequence for the project (opinion)

Direction has now failed independently from 1 minute (CS, SJ, PSH) through 5 minutes – 4 hours (RS, RSD, RD, BT, PFR, VWAP) to 1 hour – 1 day (this branch):
- with supervised and unsupervised models;
- with price, flow, order book, options and jump features.

I propose closing this branch and treating "a model discovers directional states" as answered for these four assets on Binance perpetuals.

The remaining routes to positive expected value do not depend on forecasting direction:
1. risk premia: VRP selling with defined risk, funding/basis carry;
2. at most, one prospective pre-registered daily-momentum test (§4), sized with the volatility engine.

## Questions for ChatGPT

1. Do you read any of the 16 states as a momentum or reversal regime that the regression parametrization ($a_k + b_k z_t$) could be missing, for example a nonlinear response to $|z|$ or to `er`? If so, specify it before anyone looks at TEST again. Otherwise this is a new test on a used window.
2. Do you agree the 1D failure is variance/overfitting rather than a sign problem?

## Decisions needed from Ivan

1. Accept MSH 01 as binding and close the branch? (Recommended.)
2. Pre-register a prospective daily-momentum test (K=1 pooled slope / TSMOM, volatility-engine sizing, scored on data after 2026-10-18 only)? Or move to the risk-premium routes?
