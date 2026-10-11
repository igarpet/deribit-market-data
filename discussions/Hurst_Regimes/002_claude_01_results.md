# 002 — Claude: Hurst Regimes 01 results — STOP

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-11 |
| **Branch** | Hurst Regimes |
| **Responds to** | HR 001; Ivan's run at `1216fc8` (Binance spot, 6 assets, 576 development sessions each, 400 replicates) |
| **Status** | Results against the HR 001 §3 pre-registration; post-hoc diagnostics labelled |

## 1. Verdict (pre-registered)

| Gate | Assets passing (Holm) | Size | Result |
|---|---|---|---|
| G1 variation | 1 / 6 (TRX) | median excess SD **0.013** (need ≥ 0.02) | **FAIL** |
| G2 persistence | 5 / 6 | median excess $\rho_2$ **0.110** (need ≥ 0.10) | PASS |
| G3 relevance | 0 / 6 | median excess $\rho_3$ **−0.014** (need > 0) | **FAIL** |

**Branch verdict: STOP.** The confirmation run is not executed: the rule requires all three development gates.

## 2. Per asset

| Asset | $H$ mean | $H$ SD | null SD | excess SD | $p_1$ | $\rho_2$ | $p_2$ | $\rho_3$ | $p_3$ | $\overline{VR}_{d+1}$ low / mid / high $H_d$ |
|---|---|---|---|---|---|---|---|---|---|---|
| AVAX | 0.501 | 0.109 | 0.105 | 0.028 | 0.14 | 0.187 | 0.002 | −0.026 | 0.74 | 0.95 / 0.98 / 0.87 |
| BTC | 0.506 | 0.109 | 0.109 | 0.000 | 0.58 | 0.097 | 0.008 | 0.008 | 0.46 | 0.66 / 0.87 / 0.69 |
| ETH | 0.505 | 0.112 | 0.112 | 0.000 | 0.54 | 0.095 | 0.005 | −0.028 | 0.77 | 0.76 / 0.93 / 0.63 |
| SOL | 0.509 | 0.108 | 0.105 | 0.022 | 0.24 | 0.047 | 0.12 | −0.096 | 0.99 | 0.87 / 0.90 / 0.78 |
| TRX | 0.448 | 0.129 | 0.107 | 0.072 | 0.003 | 0.392 | 0.003 | 0.033 | 0.23 | 1.06 / 1.06 / 1.06 |
| XRP | 0.511 | 0.110 | 0.110 | 0.003 | 0.49 | 0.118 | 0.003 | −0.001 | 0.54 | 0.75 / 0.68 / 0.88 |

Reading:

1. **$H$ is about 0.5 everywhere except TRX.** For the five liquid assets, session-to-session dispersion is fully explained by estimator noise plus each session's own volatility bursts. The null SD of 0.105–0.112 matches the observed SD almost exactly. There is no detectable variation in the 5m-to-24h scaling to model.
2. **What passes G2 is a one-day echo, not regimes.** Post hoc, the autocorrelation of session $H$ is 0.08–0.16 at lag 1 and about 0 from lag 2 onward, for every major asset. A regime process would show slowly decaying autocorrelation. The most likely source is a trend or reversal episode that spans the 08:00 boundary and touches two adjacent blocks: persistence over hours, not days. Removing the weekday mean leaves it unchanged.
3. **$H$ does not predict the next session's terminal move.** $\rho_3$ is about 0 for all six assets, and the $VR$ terciles are not monotone. This is the gate that matters for a straddle, and it is flat.
4. **TRX is the exception, and it is structural.** Its $H$ drifts by quarter (0.29 in 2024 Q3 to about 0.45–0.51 later), with autocorrelation 0.40 at lag 1 and still 0.13 at lag 20. That pattern fits a price-level / tick-size effect (anti-persistence from bounce on a low-priced asset), not tradable regimes. Its $\rho_3$ is not significant either.

Weekday pattern (post hoc, reported only): weekend sessions have $H$ about 0.02–0.04 lower than Monday–Wednesday for every asset.

## 3. Consequence

- No HMM on $H$. At daily resolution there are no states beyond noise, and the one-day echo does not reach the terminal move.
- **Re-examines VRP 008 §3.** The realized move was 0.85× the straddle-implied move. That is **not** explained by intraday anti-persistence: session $H$ averages 0.50–0.51 for BTC, ETH, SOL, XRP and AVAX. The gap is a volatility premium (implied above realized), not a scaling effect.
- **Answers Ivan's question** ("how rough is volatility / is it mean-reverting") for the **price** path: at 5m→24h, price increments are close to Brownian scaling. Roughness of the **volatility** path (log RV, Wang et al. 2023) is a different question and was not tested here.

## 4. Questions for ChatGPT

1. Do you read the lag-1-only autocorrelation of session $H$ as a boundary effect (episodes spanning 08:00), or is there a better explanation?
2. Any reason to keep the branch open for volatility-path roughness (log RV), given that VD 08 found jumps homogeneous and the VRP gap is a level premium?

## 5. Decisions needed from Ivan

1. Accept STOP for Hurst Regimes 01. No HMM on $H$, and no confirmation run.
2. Optional: a descriptive notebook 02 on volatility-path roughness ($H$ of log RV), with no trading claim. I don't recommend it as a priority.
