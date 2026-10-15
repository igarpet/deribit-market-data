# VD 038 — VD19 readout: the model predicts realised volatility better than implied, but it does not pay

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-16 |
| Branch | Volatility Dynamics |
| Responds to | VD 037 pre-registration; Ivan's run 538cea0 (outputs archived in `19/v0.1.0/`) |
| Status | Primary result binding. Secondary entries and one diagnostic pending (v0.1.1). |

## 1. Primary (04:00 UTC entry, 240 min to expiry)

85 session-assets over 23 sessions (2026-09-21 → 10-13). All 85 settlements come from the exchange.

**Verdict: `MODEL_ADDS_INFORMATION`.**

| Outcome | β on $x=\log(\sigma_{model}/\sigma_{imp})$ | 95% session CI |
|---|---|---|
| **log(realised path variance / implied variance)**, primary | **1.23** | **[0.39, 2.15]** |
| log(settlement move / implied move) | 0.44 | [−1.05, 1.77] |

**Encompassing regression.** $\log RV = a + b_{imp}\log\sigma_{imp} + b_{mod}\log\sigma_{model}$ gives:
- $b_{mod}$ = **1.23 [0.15, 2.25]**;
- $b_{imp}$ = 0.76 [−0.18, 1.63].

With both in the regression, the model's forecast carries the information and implied's does not reach significance.

**Robustness (post-hoc diagnostics, not gates):**
- Within-asset β (asset means removed) is 1.24 [0.15, 2.37].
- Leave-one-session-out β ranges from 0.98 to 1.41.
- Per asset: BTC 1.15, ETH 2.30, SOL 1.25; **XRP −1.05**.

## 2. Economics: it does not pay

Straddle $R_{net}$ at the ask plus fees:

| Subset | Sessions | Rows | Mean $R_{net}$ | 95% CI |
|---|---|---|---|---|
| All 04:00 straddles | 23 | 85 | −0.39 | [−0.55, −0.22] |
| Model ≥ 1.25 × implied (pre-registered rule) | 9 | 19 | **−0.15** | [−0.57, +0.37] |
| Model < 1.25 × implied | 21 | 66 | −0.50 | [−0.61, −0.37] |

**Descriptive terciles of x** (cut-offs from this sample):

| Tercile of x | Mean $R_{net}$ |
|---|---|
| low | −0.44 |
| mid | −0.52 |
| high | −0.21 |

- **Selection works in the right direction:** the selected straddles lose about 35 points less than the rest.
- **It doesn't make the trade positive.** The 04:00 straddle costs about 39% on average, and the model's selection is not enough to overcome the spread.
- **The two outcomes disagree, and that's expected:** the realised-path result is significant, but the settlement move is not. The straddle pays on the single move to expiry, which is about twice as noisy (VD 037 §4).

## 3. A confound to rule out before believing §1

$y_{rv}$ is measured on **Binance spot 5m** returns, the same data family the model's inputs come from.
- Spot 5m RV contains microstructure noise that persists: variance ratio 0.78–0.93 (RS 06). HAR forecasts that noise; the option market prices the Deribit index and ignores it.
- So part of β could be "HAR predicts Binance 5m noise", which is not tradeable.
- v0.1.1 adds the pre-specified diagnostic $y_{rv,index}$: the same regression using the **Deribit index** 5m path from the collector.
- If β survives on the index path, the information is real. If it collapses, §1 is a measurement artefact.

## 4. Bug: the 06:00 and 12:00 entries are missing

- **Cause:** the DuckDB query filtered `EXTRACT(hour …)` in the PC's local time zone (UTC+2), so hours 4, 6 and 12 local became 02, 04 and 10 UTC. Only the 04:00 UTC rows (local 06) survived the UTC check in `mv.entries`.
- **Primary not affected:** its rows are genuine 04:00 UTC entries, TTE-checked to within 10 minutes.
- **Fix (v0.1.1):** sets `TimeZone='UTC'` on the connection, as PFR 01 does.

## 5. What this means

- **The first positive result on options in the project:** the model's variance forecast contains information the 0DTE price does not fully use.
- **The economics are still negative.** The model narrows the loss of a long straddle but does not remove it at retail costs.
- **It is a 23-session historical sample.** The prospective test is sessions after 2026-10-14, with the specification frozen as v0.1.1.

## 6. Next steps (for Ivan)

1. Run v0.1.1 now. It reports the index-path diagnostic and the 06:00 and 12:00 entries, and the primary is unchanged.
2. If the index diagnostic holds, freeze v0.1.1 and re-score on post-10-14 sessions after about four weeks of new collector data. No more TEST-window changes.
3. **Possible next questions, not yet proposed:**
   - Does the edge improve with mid-price execution? VRP 03 says that's not achievable passively.
   - Does a larger threshold (1.5×) help? That would be specification search: prospective only.
