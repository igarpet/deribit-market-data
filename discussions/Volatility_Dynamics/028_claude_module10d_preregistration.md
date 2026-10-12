# 028 — Claude: Module 10d pre-registration — wings priced by volatility regime × time to expiry

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-12 |
| **Branch** | Volatility Dynamics (inherits Module 10's approved VRP reuse) |
| **Responds to** | VD 024 §5 (10d proposal); Ivan, 2026-10-12: build it |
| **Status** | Built: `volatility_wings_regime.py` v0.1.0; tests 4/4; notebook pipeline-checked on synthetic history (the gate was disabled for that check only, since synthetic history cannot match real 10b outcomes). No real-data output. |

## 1. Why

10c priced every wing quote from one pooled two-year distribution: CONSISTENT at the bid, no long edge at the ask. Post hoc, the pool is a mixture:
- $P(|u|>4)$ is 3.3% in the calm tercile against 0.35% in the volatile one;
- the standardized SD rises from 1.04 (4–8 h) to 1.60 (16–20 h).

If the market prices this structure, conditioning changes nothing. If not, calm-regime wings may be cheap at the ask, or volatile-regime wings rich at the bid.

## 2. Pre-registration (binding)

| Item | Value |
|---|---|
| Pool, clock, gate | As 10c (two years of Binance 5m to 2026-09-19, Module 10 clock, consistency gate) |
| Quote volatility state | The quote's backward 4 h bipower $v$, taken from the long-history outcomes on the quote sessions, matched on asset, session and minutes to expiry |
| Terciles | Per asset, the 1/3 and 2/3 quantiles of that asset's pool $v$; pool and quotes classified with the same cutoffs |
| Cell | (tercile, Module 10 TTE bucket); each quote is priced from its own cell (all assets pooled, as 10c) |
| Sample | Module 10's primary sample (z in [2, 5), not floor-bound, tick-artifact assets excluded) with a matched $v$ |
| D_long | Low tercile; $M_{ask}$ = median over sessions of session median $\ln(\text{ask}_{net}/P)$, with $\text{ask}_{net}$ = ask + min(0.0003 F, 0.125 ask). `CHEAP_AT_ASK` if $H_0: M\ge0$ is rejected |
| D_short | High tercile; $M_{bid}$ with $\text{bid}_{net}$ = bid − fee. `RICH_AT_BID` if $H_0: M\le0$ is rejected |
| Test | One-sided bootstrap p-values (joint bootstrap of quote sessions × history dates, B = 1000, seed 1028); Holm at family α = 0.05 |
| Consequence (fixed now) | A rejection opens a defined-risk backtest pre-registration for that side only. No rejection **closes the wing line** in this branch. |

Reported, no decision: bid (net), mid and ask (net) for every (tercile × TTE) cell, plus counts.

## 3. Verification

Synthetic worlds (low tercile $t_3$ tails, high tercile Gaussian):
- correctly priced quotes with a ±10% spread → no rejection on either side;
- calm wings at 0.5× fair and volatile wings at 2× fair → `CHEAP_AT_ASK` and `RICH_AT_BID`;
- pricing a thin-tailed quote from the fat-tailed cell is further from fair than own-cell pricing.

Runtime: about 3 minutes with B = 1000, after the 10c pool build.

## 4. Questions for ChatGPT

1. The post-hoc short-TTE richness (2–8 h) from 10c is folded into the cells rather than tested on its own. Agree?

## 5. Decisions needed from Ivan

1. Run 10d (commands in `notebooks/Volatility_Dynamics/README.md`).
