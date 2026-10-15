# VD 039 — VD19 v0.1.1: the information is real on the Deribit index and replicates at 06:00; still no money at the ask

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-16 |
| Branch | Volatility Dynamics |
| Responds to | VD 038; Ivan's run 2e7c28e |
| Status | Result. The primary is unchanged (`MODEL_ADDS_INFORMATION`). The specification is frozen as v0.1.1 for the prospective re-score. |

## 1. The spot-noise confound is ruled out

- Deribit-index path RV and Binance-spot path RV correlate at **0.995–0.999** over these windows, with a mean log ratio of only 0.01–0.02.
- The index-based slope is at least as strong:

| Entry | β, spot RV | β, Deribit index RV |
|---|---|---|
| 04:00 (240 min), primary | 1.23 [0.39, 2.15] | **1.45 [0.51, 2.52]** (21 sessions) |
| 06:00 (120 min) | 1.23 [0.47, 1.94] | **1.25 [0.45, 1.92]** |
| 12:00 (1200 min), extension | 0.29 [−0.35, 1.04] | 0.06 [−0.52, 0.79] |

## 2. Reading

- **2–4 hours to expiry:** the HAR variance level predicts realised volatility to expiry beyond what the ATM straddle implies.
  - **Encompassing regression:** at 04:00 the model's coefficient is 1.23 [0.15, 2.25] against implied's 0.76 [−0.18, 1.63]; at 06:00 the model's is 1.15 [0.21, 1.85].
  - **Not independent confirmation:** 06:00 is a sub-window of 04:00 on the same sessions.
- **20 hours (12:00):** no information. Implied dominates ($b_{imp}$ = 1.70 [0.87, 2.29], $b_{mod}$ = 0.24). The market prices the overnight-to-morning horizon well, but under-uses recent intraday volatility in the last few hours.
- **Per asset (04/06 pooled, index RV):**

  | Asset | β |
  |---|---|
  | BTC | 1.39 |
  | ETH | 1.97 |
  | SOL | 0.94 |
  | XRP | **−0.63** |

## 3. Economics

| Entry | Selected (model ≥ 1.25 × implied) | Not selected |
|---|---|---|
| 04:00 | −0.15 [−0.57, +0.37], n=19 | −0.50 [−0.61, −0.37] |
| 06:00 | −0.11 [−0.46, +0.25], n=15 | −0.50 [−0.64, −0.34] |
| 12:00 | −0.23 [−0.47, +0.01], n=32 | −0.24 [−0.41, −0.05] |

On selected trades, realised variance is about **1.3× implied variance**: the model's call is right. The settlement move averages 1.1–1.3× the implied move. Even so, the long straddle still loses 11–15% at the ask plus fees. The remaining gap is the option spread.

## 4. Status and next step

- **Information:** confirmed historically at 2–4 hours, robust to the measurement concern. This is the project's first result where the model knows something the option price doesn't.
- **Tradeability:** not at the ask.
- **Frozen as v0.1.1.**
- **Prospective re-score:** sessions from 2026-10-15 onward, with no specification changes, after about 20 new sessions (around mid-November). It confirms only if:
  - β at 04:00 on index RV has a lower bound > 0; and
  - selected-trade $R_{net}$ is reported, but it is not expected to pass.
- **Open economic question (not a test yet):** what entry price would make the selected 04:00/06:00 straddles break even? It needs the bid/ask/mid per row, which the run doesn't save. A small v0.1.2 output-only change could add it without changing any statistic.
