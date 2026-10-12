# 011 — Claude: VRP 03 rerun (v0.1.1) — the sell-side verdict

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-12 |
| **Branch** | Volatility Risk Premium |
| **Responds to** | VRP 010 §1.2; Ivan's rerun (commit 8170dd2) |
| **Status** | Final. Buy `INCONCLUSIVE`, sell `PASSIVE_HURTS`. The execution line is closed. |

## 1. Consistency check

The sell side now adds up: policy (−9.1%) minus aggressive on the same entries (−5.8%) equals ΔR (−3.3%).

At the primary setting, 2,418 of 2,515 entries are used across all 20 sessions. The 97 excluded entries (3.9%) had no executable bid at $t+W$. Excluding them flatters the policy slightly, since a seller there could not trade at all.

The buy side is unchanged from v0.1.0, as expected.

## 2. Primary (q = 0.5, W = 15 min)

| Side | ΔR (return units) | Δ (bps of forward) | Policy R | Verdict |
|---|---|---|---|---|
| Buy | −0.15 pp [−0.95, +0.80] | −1.4 [−2.9, +0.1] | −26.8% | `INCONCLUSIVE` |
| Sell | **−3.3 pp [−5.2, −1.9]** | **−3.2 [−4.0, −2.4]** | −9.1% | `PASSIVE_HURTS` |

On the sell side:
- the result is negative in all 8 (q, W) cells, in every TTE bucket and in all 7 assets;
- the cost grows with W (−1.9 bps at 5 min to −5.1 bps at 60 min);
- only 2% of straddles fill passively on both legs.

The resting offer fills when the market lifts it. That is when realized volatility is arriving, and the seller is then left short into the move.

## 3. Conclusion

- A passive limit near the mid does not recover the spread on either side. Adverse selection exceeds the half-spread saved.
- Any 0DTE strategy in this branch must clear its hurdle at taker prices.
- At taker prices:
  - the long straddle loses about 27% of premium;
  - the unconditional short straddle loses about 6% of credit (it was +1 bp of forward on VRP 02's statistic: positive in bps, negative as a mean of per-trade ratios).
- The only open item in the branch is VRP 04's conditional test, which runs on the confirmation window.

## 4. Decision needed from Ivan

None for VRP 03. The decisions on the VRP 04 spec (VRP 010 §3) are still due before 2026-11-06.
