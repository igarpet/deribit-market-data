# Regime Structure 006 — RS 02 readout: the signal is real, but it does not pay for the spread

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-13 |
| Branch | Regime Structure |
| Responds to | RS 005; Ivan's run, commit 9c21ecd |
| Status | Result. The primary verdict is binding. |

## 1. Part A — Binance, 2 years: the signal raises the move, modestly and robustly

Normalised |move| after a 15-minute up-displacement, relative to hour-matched other times (97.5% CI):

| Window | 60 min | To 08:00 settlement |
|---|---|---|
| Development | 1.043 [1.017, 1.070] | 1.047 [1.017, 1.078] |
| **Test** | **1.068 [1.011, 1.131]** | **1.107 [1.058, 1.155]** |

- Both windows and both horizons are above 1.
- Per asset, the settlement ratio is 1.09–1.14, with every CI above 1.
- The effect does not fade before settlement: the move to 08:00 is 5–11% larger than at comparable times. That is the quantity a held straddle is paid on.

## 2. Part B — collector quotes, 20 sessions (2026-09-21 → 10-10), 329 signal straddles of 16,717

| Long ATM 0DTE, net of ask + fee | Signal entries | All entries | Signal − all |
|---|---|---|---|
| **Straddle to settlement (primary)** | **−0.210 [−0.403, +0.018]** | −0.286 [−0.435, −0.121] | +0.076 [−0.015, +0.179] |
| Straddle at mid (not executable) | −0.068 [−0.29, +0.20] | −0.156 [−0.33, +0.04] | +0.088 [−0.02, +0.21] |
| ATM call only to settlement | −0.428 [−0.70, −0.15] | −0.301 [−0.58, +0.01] | **−0.127 [−0.22, −0.04]** |
| Straddle sold at the bid after 60 min | −0.318 [−0.35, −0.29] | −0.314 [−0.34, −0.29] | −0.004 [−0.03, +0.02] |
| Realised / implied move | 0.92 | 0.83 | +0.09 [−0.00, +0.20] |

**Primary verdict: `INCONCLUSIVE`.** M = −0.21, 95% CI [−0.41, +0.04]. 4 of 20 sessions are positive.

## 3. Reading

1. **The signal does what Part A says.** After a signal, the realised move relative to the straddle's implied move rises from 0.83 to 0.92. That is about the 10% Part A measured, and options don't price it in. The straddle loss shrinks by about 8 points of premium.
2. **It is not enough.** Even at the mid, signal straddles lose about 7%. At the ask they lose about 21%. Breaking even at the ask would need the realised move about 25–30% above implied; the signal delivers about 10%. The CI touches zero only because 20 sessions are few; the point estimate is a clear loss.
3. **The call-only version is worse, and significantly so** (−0.13 against all entries). Although the next large 5-minute bar leans upward, by settlement the up-move tends to be partly given back, or calls are richer right after an up-move. Directional long calls on this signal lose.
4. **Exiting after 60 minutes loses about 32% whatever the signal.** Paying the spread twice dominates everything. Short holds are not viable at the ask/bid.

## 4. What this closes

- The only robust structure found in the Regime Structure branch is real but small. It improves long-straddle P&L by about 8 points and does not make it positive at executable prices.
- The single route that could change this is **execution cost**. At the mid, the signal straddle is near break-even with a wide CI. Getting filled closer to mid, for example passive limit orders, is the lever. This branch has not tested that, and VRP 03's execution results are not imported here.

## 5. Questions for ChatGPT

1. Is the call-only result skew (calls rich after up-moves) or path (partial reversal to settlement)? Splitting it by delta-matched put/call implied vol would tell.
2. Does any reason remain to continue C3 beyond an execution study?

## 6. Decisions needed from Ivan

1. Accept the result: the C3 signal is real but does not pay for long 0DTE options at the ask.
2. Next:
   - (a) an execution study: can signal straddles be bought near mid with limit orders, and at what fill rate?
   - (b) wait for more collector sessions and re-run RS 02 unchanged on sessions after 2026-10-10 as confirmation;
   - (c) close the branch.
