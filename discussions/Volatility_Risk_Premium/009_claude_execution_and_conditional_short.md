# 009 — Claude: passive execution (VRP 03) and conditional short volatility with defined risk (VRP 04)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-12 |
| **Branch** | Volatility Risk Premium |
| **Responds to** | VRP 008 §4; Ivan, 2026-10-12: build the four recommended steps (execution study, conditional short test, Module 10d, defined-risk pricing). Module 10d is in VD 028. |
| **Status** | Built: `vrp_execution.py` v0.1.0, `vrp_conditional_short.py` v0.1.0; tests 9/9 (+ VRP 02's 6/6). Both notebooks run end to end on synthetic DBs, including VRP 04 freeze → confirm → refusal to overwrite. No real-data output. |

## 1. Context

VRP 02 (20 sessions, at executable prices):
- the long straddle at the ask lost 27% of premium;
- the short straddle at the bid made about 0 (+1 bp of forward);
- at the mid, the seller would have made about +14%.

The volatility premium exists, but crossing the spread gives it away. Two things can change that: paying less spread (VRP 03), and trading only when implied is rich against a forecast (VRP 04). Ivan's concern about selling is tail risk, so VRP 04 prices a defined-risk iron fly from the start.

## 2. VRP 03 — passive execution (pre-registered)

| Item | Value |
|---|---|
| Entries | VRP 02's (hourly ATM straddles, TTE 2–20 h, sessions < 2026-11-07) |
| Order | Each leg rests at $P_b=\text{mid}-q(\text{mid}-\text{bid})$ (buy) or $P_s=\text{mid}+q(\text{ask}-\text{mid})$ (sell), rounded to the family tick away from the mid; $q\in\{0,0.5\}$; $W\in\{5,15,30,60\}$ min |
| Fill proxy | At a later 5m snapshot within $W$: the opposite touch reaches $P$, **or** a trade printed (24h volume rose or last price changed) at or through $P$. $P$ strictly improves the touch, so the order has queue priority; one-tick spreads join the touch and are flagged. Optimistic at a 5-minute grid. |
| Policy | Passive-then-cross: a leg not filled by $t+W$ is crossed at that snapshot's touch. Every entry trades, so it is paired with VRP 02's aggressive entry. |
| Primary | $q=0.5$, $W=15$ min. Per side, paired $\Delta R$ = policy − aggressive; mean of session means; 95% session bootstrap (4,000, seed 9003) |
| Verdict | CI > 0 `PASSIVE_HELPS`; CI < 0 `PASSIVE_HURTS`; else `INCONCLUSIVE` |
| Reported | Absolute policy R per side; fill rates by q, W, TTE and asset; markouts (mid 30 min after the fill, in half-spreads); both-legs-filled subsample; adverse-selection gap |

On synthetic data (Black-76 quotes, 24% spread, random trades): buy-side fills rise from 1% to 55% of straddles as $W$ goes 5 → 60 min; $\Delta R$ is +0.02 at the primary; markouts are negative (the expected adverse selection).

## 3. VRP 04 — conditional short volatility (pre-registered)

**Forecast (VRP-local; nothing imported from other branches).**
- $\hat v_t$ is the mean of causal EMAs (half-lives 1 h, 4 h, 24 h) of index $r^2$ de-clocked by a development-only time-of-day profile.
- Forecast variance to expiry: $V_{fc}=\hat v_t\sum_{j\in(t,T]}s^2(\text{slot}_j)$.
- Implied: $V_{iv}=\sigma_{\text{straddle,mid}}^2T$.
- Wedge: $w=\ln(V_{iv}/V_{fc})$.

**Structures.**
- Short ATM straddle.
- **Short iron fly:** the straddle plus long wings at the listed strikes nearest $Fe^{\pm1.5\sigma\sqrt T}$.
  - Max risk = widest wing − credit + fees.
  - Return on risk RoR = P&L / max risk.
- Prices: executable (sell at bid, buy at ask) and mid. Fees on every leg. Held to the delivery price.

**Windows and decision.**

| Step | Rule |
|---|---|
| Development | Sessions < 2026-11-07; terciles of $w$ in-sample; **descriptive only** |
| Freeze | Once, after 2026-11-06 (`VRP04_FREEZE=1`): clock profiles, tercile thresholds and constants written to `04/04_frozen_spec.json`; never overwritten |
| Confirmation | Sessions ≥ 2026-11-07 (the VRP holdout from VRP 002 §2.4), frozen spec, **binding at ≥ 20 sessions** |
| Statistic | Top-tercile short iron fly at executable prices: mean over sessions of session-mean RoR; 95% session bootstrap (4,000, seed 9004) |
| Verdict | CI > 0 `CONDITIONAL_PREMIUM_CONFIRMED`; CI < 0 `REJECTED`; else `INCONCLUSIVE` |

Reported: straddle and iron fly at executable prices and at mid, by tercile; high − low tercile (session-paired); iron-fly credit as a share of straddle credit (about 0.79 on synthetic data, so the wings cost about 21% of the credit).

## 4. Order of execution

1. **Now:** run VRP 03, and VRP 04 in development mode (both read the research DB; minutes).
2. **After 2026-11-06:** VRP 04 freeze.
3. **From 2026-11-07, at ≥ 20 sessions (about end of November):** VRP 04 confirm; VRP 02 re-run unchanged on the same window.

## 5. Questions for ChatGPT

1. Is the 5-minute fill proxy acceptable as an upper bound, or should we collect a trade tape (Deribit `trades` channel) before trusting any passive-fill number?
2. Z = 1.5 straddle-sigma for the wings: any objection before the freeze?

## 6. Decisions needed from Ivan

1. Run VRP 03 and VRP 04 (development); commands in `notebooks/Volatility_Risk_Premium/README.md`.
2. Confirm the maker-fee assumption (maker = taker = 0.03% capped at 12.5%). If Deribit charges makers less, VRP 03 is conservative.
