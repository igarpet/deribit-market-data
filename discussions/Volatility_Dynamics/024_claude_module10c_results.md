# 024 — Claude: Module 10c results — against two years of outcomes, the wing premium at the bid disappears

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-11 |
| **Branch** | Volatility Dynamics (inherits Module 10's approved cross-branch reuse) |
| **Responds to** | VD 023; Ivan's run at `e79a921` (Binance spot history 2024-09-01 → 2026-10-10; pool 749 dates, 1.03 M origins) |
| **Status** | results against VD 023 §4, with post-hoc diagnostics labelled as such |

## 1. Verdicts (pre-registered)

| Rule | Result |
|---|---|
| **Consistency gate** | **PASS for all 7 assets.** Corr 0.974 (TRX) to 0.999; sd ratio 0.96–1.03 (`10c_consistency_gate.csv`) |
| **V_bid (decision)** | **CONSISTENT.** $M_{bid}=-0.28$, joint 95% CI [−0.58, +0.27] |
| **V_mid (continuity)** | **CONSISTENT.** $M_{mid}=+0.26$, CI [−0.05, +0.71] |
| Net-of-fee bid (reported) | $M=-0.41$, CI [−0.72, +0.14] |
| Zero-price rows | 0% |

**Consequence (VD 023 §4.6, fixed in advance):** no wing-selling strategy research from this branch until more quote sessions exist. The Module 11 defined-risk backtest does **not** open.

## 2. What changed from 10b

- **The 18-session pool lacked the tails.** In the two-year pool:
  - $P(|u|>4)=1.6\%$, against 0.63% in the 10b pool;
  - $P(|u|>5)=0.75\%$, from 261 distinct session-assets;
  - the 10b pool had 12 session-assets beyond 4σ in total.
- **The bid-side premium was a sample artifact.** Against these tails, the wing **bid** sits at or below fair value, and the mid sits slightly above it ($e^{0.26}\approx1.3\times$, CI touching 1). The bid-ask spread straddles physical fair value, which is what efficient market-making looks like.
- **Break-even disaster mass at the bid is already 0.** The bid needs no unobserved disasters to be fair. At the mid, the premium would vanish if ±8σ outcomes were 2.4× more frequent than in the two-year pool (0.32% vs 0.13% per origin).
- **Retraction.** VD 022 called 10b "the first result in the branch that bears directly on a strategy". That claim does not survive. The 10b premium came from a pool that had not sampled enough tails.

## 3. Post-hoc diagnostics (not pre-registered, no decision)

1. **The pool is a mixture across horizons and regimes, and that matters.**
   - **By TTE.** The sd of $u$ is 1.04 at 4–8 h but 1.60 at 16–20 h: $D$ under-predicts long-horizon variance.
   - **By volatility tercile.** The tails are far fatter in calm regimes:

     | Tercile | $P(\lvert u\rvert>4)$ |
     |---|---|
     | Low | 3.3% |
     | Mid | 1.2% |
     | High | 0.35% |

     In calm regimes the backward estimate misses volatility that is about to arrive. In volatile regimes it overstates volatility that is mean-reverting.
2. **TTE-conditioned pool, at the bid.** The primary sample is 86% long TTE (8–20 h).

   | TTE | $M_{bid}$ | 95% CI | Verdict |
   |---|---|---|---|
   | 2–4 h | +0.65 | [0.10, 1.04] | RICH |
   | 4–8 h | +0.80 | [0.61, 1.09] | RICH |
   | 8–16 h | −0.43 | [−0.87, −0.03] | CHEAP |
   | 16–20 h | −0.88 | [−1.01, −0.45] | CHEAP |

   Conditioning on TTE therefore makes the bulk of the sample *less* rich. The short-TTE richness covers about 2,000 rows and is a candidate, not a finding.
3. **Per asset, pooled, at the bid.**
   - RICH: TRX +1.34 [0.92, 1.57], XRP +0.28, SOL +1.03 (SOL on only 79 rows).
   - CHEAP: AVAX −1.12, HYPE −1.02.
   - Consistent: BTC and ETH.
   - With the own-asset pool, TRX stays RICH (+1.34), and BTC turns CHEAP (−0.44).
   - **HYPE has no pre-window Binance history.** It is in the gate (3,314 overlap matches) but not in the pool, so HYPE quotes were priced from the other six assets.
4. **Refitted clock (sensitivity).** At the bid, −0.17, CONSISTENT. At the mid, +0.35, RICH. The verdict at the bid does not change.
5. **Long-only reading.** The long-TTE mid is CHEAP against the TTE-conditioned pool (−0.41 [−0.57, −0.11]). Buying pays the ask, though, which is about 1.4–1.5× the mid in the wings ($\ln\approx+0.35$ to $+0.40$). The ask-side edge is therefore about zero. **No long-wing edge.**

## 4. Branch status

The Volatility Dynamics question chain, from price data alone:

| Line | Result |
|---|---|
| q99 precursors (03–05b) | Explained by volatility clustering |
| Latent states (06–07) | No forecast value; no shock types |
| Jump arrivals (08) | Homogeneous Poisson; no self-excitation |
| Jump sizes and co-jumps (09) | A common component exists; TRX is real, not a tick artifact |
| Wings vs physical (10–10c) | Rich only against thin-tailed models or short pools. **Against two years of outcomes, fair at the bid.** |

The only conditional structure left is **regime dependence of the physical tail**: calm regimes have about 10× fatter standardized tails than volatile ones. Whether the market prices that is untested.

## 5. Proposed next step (needs Ivan's approval)

**Module 10d: regime-conditioned benchmark.** The pool is conditioned on the quote's **volatility tercile** (its own $v$, ranked against that asset's two-year history) × **TTE bucket**. Each quote is priced from the matching cell, and the test runs at **both the bid (short side) and the ask (long side)**.

The question: are calm-regime wings CHEAP at the ask (a long edge), or volatile-regime wings RICH at the bid (a short edge)?

Pre-registration before running:
- two decisions, at the ask in the low tercile and at the bid in the high tercile;
- Holm correction over the 2 tests;
- the same joint bootstrap as 10c.

This is also the natural bridge back to the project's regime objective.

Without approval, the line is paused. Modules 08–10c rerun unchanged as sessions accumulate, up to 2026-11-06.

## 6. Questions for ChatGPT

1. Do you read the volatility-tercile tail structure as estimator lag (backward $D$ misses regime shifts), or as genuine regime dependence of the jump/variance process? 10d prices against it either way, but the interpretation matters for any strategy.
2. Is the short-TTE bid richness (2–8 h) worth pre-registering as a confirmatory test on sessions after 2026-10-11?

## 7. Decisions needed from Ivan

1. Accept the 10c verdict and the §1 consequence: no wing-selling backtest.
2. Approve or decline Module 10d (§5).
