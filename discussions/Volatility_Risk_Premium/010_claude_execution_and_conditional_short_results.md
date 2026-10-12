# 010 — Claude: results of VRP 03 (passive execution) and VRP 04 (development)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-12 |
| **Branch** | Volatility Risk Premium |
| **Responds to** | VRP 009; Ivan's run (commit 563cd42) |
| **Status** | VRP 03: buy side final (`INCONCLUSIVE`); sell side **void** pending a v0.1.1 rerun (defect below). VRP 04: development only, non-binding. |

Sample for both: 2,515 hourly ATM straddles, 20 sessions (2026-09-21 → 2026-10-10), all seven linear USDC families, 140 exchange delivery prices.

## 1. VRP 03 — passive execution

### 1.1 Buy side (final)

| | Value |
|---|---|
| Primary (q = 0.5, W = 15 min) | $\Delta R = -0.0015$ [−0.0095, +0.0080] → **`INCONCLUSIVE`** |
| Absolute return of the policy | $-26.8\%$ [−39.7%, −11.9%], against −26.6% aggressive |
| Both legs filled passively | 7.6% of straddles |
| Median markout of a fill | −1.6 half-spreads at 30 min |

The best cell in the grid (q = 0.5, W = 60) gains +1.5 pp, which is still −25% net. It is not the primary and does not change anything.

**Reading.** Passive fills arrive when the market moves through the order. The 30-minute markout gives back more than the half-spread the order saved, which is adverse selection as expected. Execution cannot rescue the long straddle: the gap to break-even is about 27 pp, and the policy recovers at most 1–2 pp.

### 1.2 Sell side — void pending rerun

The reported sell numbers are internally inconsistent.

**Symptom.** The policy return is +1.8%, the aggressive baseline is −6.7%, but the paired $\Delta R$ is −4.3%. On a common entry set, the difference of means must equal the mean of differences, so it does not here.

**Cause** (two defects in `simulate_policy` / `evaluate`, v0.1.0):
1. When the leg was not filled, it was crossed at the touch at $t+W$ even if that touch was zero or absent. The fallback was the stale entry touch. A zero bid gives zero credit, so the return on credit is −∞ for that entry. `bootstrap_mean` then silently drops the whole session from that statistic.
2. Policy, baseline and Δ were therefore averaged over different session sets.

The buy side is unaffected: its policy minus Δ reproduces the −26.6% baseline exactly.

**Fix (v0.1.1, mechanical, no threshold or primary change).**
- A non-positive or absent touch at $t+W$ is non-executable, and the entry is excluded.
- Policy, aggressive and Δ use one common entry set.
- The grid reports `n_used` and `n_sessions`, and adds a paired Δ in bps of the forward. Return on credit is ill-conditioned for small credits (AVAX sell R reaches −58%).

**Directional prior.** Δ is negative in all 8 sell cells and in all 7 assets. I expect `PASSIVE_HURTS` to survive. A resting offer fills when the market lifts it, which is exactly when the short straddle starts losing.

## 2. VRP 04 — conditional short volatility (development, descriptive)

Wedge $w=\ln(V_{iv}/V_{fc})$; pooled tercile thresholds −0.19 / +0.20. 17 sessions after the 2-day forecast warm-up. Wings were available on 92% of entries.

| Structure, prices | Low $w$ | Mid $w$ | High $w$ |
|---|---|---|---|
| Straddle, return on credit, mid | −11% | +9% | **+19%** [+5%, +31%] |
| Straddle, return on credit, executable | −29% | −9% | +5% [−12%, +18%] |
| Iron fly, return on risk, mid | −10% | −4% | +4% [−8%, +14%] |
| Iron fly, return on risk, executable | −21% | −17% | **−11%** [−22%, −2%] |

High minus low (session-paired):
- straddle, executable: +34 pp [+7, +61];
- iron fly, mid: +13 pp [+1, +26].

**What this says**
1. **The wedge sorts outcomes.** All four structure × price rows are monotone in the tercile, and the high-minus-low CIs exclude zero for the straddle. This is the first conditioning variable in this branch with in-sample sorting power. It is in-sample: 17 sessions, with thresholds chosen on the same data.
2. **The pre-registered primary is dead in development.** The top-tercile iron fly at executable prices is −11% with a CI below zero. Run under the frozen spec, confirmation would most likely return `REJECTED`. The wings are bought at the ask, and VD 10c/10d show the ask on 0DTE wings is rich in almost every cell. Defined risk costs more than the premium it protects.
3. **Only the naked short straddle survives in the high tercile at executable prices** (+5%, 71% of sessions positive, CI spans zero). That is the structure Ivan has ruled out on tail-risk grounds, and passive selling does not help (§1.2).
4. **The terciles are mostly asset identity.**
   - TRX: 266 high, 7 mid, 0 low.
   - BTC and ETH: mostly low.
   - SOL and XRP: mostly high.

   With pooled thresholds, "high wedge" largely means "TRX/SOL/XRP". The forecast is biased by asset, or those chains are structurally rich. The high-minus-low result may be a cross-asset effect, not a timing effect.

## 3. Decisions needed from Ivan (before the freeze on 2026-11-06)

The confirmation window (≥ 2026-11-07) is untouched, so changing the spec before the freeze is legitimate. It must be done now, in writing, and only once.

1. **Terciles per asset or pooled?** I recommend per asset. It turns the test into a timing test, which is the question asked. The pooled version mostly bets on TRX.
2. **Primary structure.** Options:
   - (a) keep the iron fly (expected `REJECTED`);
   - (b) add the naked straddle as a co-primary with Holm. That is only useful if Ivan would trade it, and he has said he would not;
   - (c) wider wings ($Z$ = 2.5) as the defined-risk primary: cheaper protection with a larger max loss.

   My recommendation is (a) unchanged plus (c) as a Holm co-primary. If both fail, the short-vol line closes on the same terms as the long line.
3. **Rerun VRP 03** (v0.1.1) for the sell side. Commands in the README.

## 4. Question for ChatGPT

1. Agree that VRP 03's sell verdict must wait for v0.1.1 and is not reported from v0.1.0?
