# 008 — Claude: VRP 02 results — no unconditional long edge

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-11 |
| **Branch** | Volatility Risk Premium |
| **Responds to** | VRP 007; Ivan's run at `8fc765d` |
| **Status** | Results against the VRP 007 §2 pre-registration. Post-hoc reading labelled. |

## 1. Verdict (pre-registered)

| | Value |
|---|---|
| Sessions / session-assets / straddles | 20 (2026-09-21 → 10-10) / 140 / 2,515 |
| Settlement | Exchange delivery price for all 140 session-assets |
| $M$ (net return on premium) | **−0.266**, 95% CI [−0.389, −0.122] |
| Leave-one-session-out range | [−0.311, −0.252] |
| Sessions with positive mean | 3 of 20 |
| **Verdict** | **NO_LONG_EDGE** |

Settlement check: our approximate TWAP agrees with the exchange price to within 1 bp on average, with a session sd of 0.6–6.9 bps. This is immaterial against the result.

## 2. Where the 27% goes (reported, no decision)

| Metric | $M$ | 95% CI |
|---|---|---|
| Net (ask + fee) | −0.266 | [−0.39, −0.12] |
| Gross (ask, no fee) | −0.229 | [−0.36, −0.07] |
| Mid entry (not executable) | −0.137 | [−0.28, +0.04] |
| Realized move / straddle-implied move | 0.85 | [0.70, 1.02] |
| Spread / mid | 0.25 | [0.23, 0.27] |
| Short at the bid, net (bps of F) | +1.0 | [−21, +20] |

Reading:

- **Execution cost is about half the loss.** The bid-ask spread is about 25% of the straddle mid, rising from 14% at 16–20h to 49% at 2–4h. Paying half of it plus fees costs about 13% of premium.
- **The rest is a volatility premium.** At the mid, the realized move runs about 15% below the straddle-implied move. The CI touches 1, so this is not established on 20 sessions.
- **This reverses VRP 006 Part D**, which put realized above implied by +0.36 to +0.58 log. Part D compared summed 5m index RV (inflated 5–26% by noise, Part E) with ATM IV, as a median of logs. The executable straddle settles on the actual move and has none of those problems. **Part D should not be cited as evidence that options are cheap.**
- **Every asset loses except AVAX.** AVAX: +0.04, CI [−0.26, +0.43]. TRX is worst at −0.44. **Every TTE bucket loses.**
- **The short side is not a free lunch either.** Selling at the bid nets about +1 bps, with a CI spanning zero. The spread sits roughly around fair value, as VD 10c found for the wings.

## 3. Consequence

**Branch result:** unconditionally, a liquidity-taking buyer of 0DTE ATM straddles loses about a quarter of the premium, net. Any long strategy needs a conditioning signal that selects sessions where the realized move exceeds the implied move by about $1/(1-0.27)\approx1.35\times$ on average, after costs.

That bar is high relative to what the project has found:
- VD 011–016: precursors reduce to the volatility level, which the price already contains.
- VD 018: jumps arrive as a homogeneous Poisson process.

## 4. Proposed next steps (need Ivan's approval)

1. **Execution cost first (post-hoc question; cheap and decisive).** The mid result (−0.14, CI touching 0) says the spread is the larger, controllable part of the loss.
   - Measure how often a **passive bid at the mid**, or mid minus k ticks, would have filled within the next 5–15 minutes. Use the 5m quote history, which needs no new data.
   - If passive fills are common and not adversely selected, re-score $R$ at the realised fill.
   - If passive fills are rare or adversely selected, the long-0DTE route is closed at current sample size.
2. **Do not open conditional long tests** (OBI H2, VD 10d at the ask) until step 1 shows the cost hurdle can be cut. Otherwise each has to clear a ~35% realized-over-implied lift.
3. **Re-run 02 unchanged on the confirmation window** (sessions from 2026-11-07), as already planned. A 20-session verdict at CI −0.12 is unlikely to flip.

## 5. Questions for ChatGPT

1. Do you agree that VRP 006 Part D is superseded by the executable result for any "options cheap" claim?
2. For the passive-fill study, is a 5-minute quote grid adequate, or does adverse selection need trade-level data?

## 6. Decisions needed from Ivan

1. Accept NO_LONG_EDGE for the unconditional long ATM straddle.
2. Approve or decline the passive-execution study (§4.1). Also hold the conditional long tests until it reads out (§4.2).
