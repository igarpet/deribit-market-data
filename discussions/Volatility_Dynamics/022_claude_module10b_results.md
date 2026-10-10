# 022 — Claude: Module 10b results — wings rich against realized moves, but the margin is within reach of a rare tail

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-11 |
| **Branch** | Volatility Dynamics (inherits Module 10's approved cross-branch reuse) |
| **Responds to** | VD 021; Ivan's run at `3f7f2c3` (outcomes 2026-09-19 → 2026-10-10; 18 settled sessions, 126 session-assets, 27,986 origins) |
| **Status** | results against VD 021 §3, plus post-hoc diagnostics labelled as such |

## 1. Verdict (pre-registered)

**Primary: RICH.** $z\in[2,5)$ gives $M=1.05$, joint-bootstrap 95% CI [0.48, 2.34]. That covers 14,805 rows over 19 sessions, with 0% of rows dropped as zero-price. The typical wing mid is about $e^{1.05}\approx2.9\times$ the realized-outcome price.

| Benchmark on the same rows | $M$ | 95% CI | Verdict |
|---|---|---|---|
| Module 10 jump model | 2.98 | — | RICH |
| **Empirical, pooled, LOSO (primary)** | **1.05** | **[0.48, 2.34] joint** | **RICH** |
| Variance mixture | 0.60 | [0.18, 1.22] joint | RICH |
| Empirical, TTE-conditioned | 1.33 | [0.97, 1.92] quote-only | RICH (6.8% zero-price) |

**Calibration and shape across buckets.** The near-the-money buckets ($z<1$) are CONSISTENT in 4 of 5 TTE buckets. Richness rises with moneyness:

| $z$ bucket | $M$ |
|---|---|
| [1, 2) | −0.07 to 0.50 |
| [2, 3) | 0.70–1.07 |
| [3, 5) | 2.0–3.0 |
| [5, 8) | 4–6, but 30–45% of rows priced zero, so not interpretable |

By asset, all seven are RICH, ranging from AVAX 0.40 [0.05, 0.83] to TRX 2.44.

## 2. What 10b showed about physical tails

Realized tails are fat, and Module 10's model was far too thin:

| | Realized | Normal |
|---|---|---|
| $P(\lvert u\rvert>2)$ | 8.8% | 4.6% |
| $P(\lvert u\rvert>3)$ | 2.9% | 0.27% |
| $P(\lvert u\rvert>4)$ | 0.63% | 0.006% |

Realized variance relative to ex-ante $D$ has an sd of $\log k$ of 0.79. Moving from the jump model to the empirical benchmark removes about two thirds of Module 10's gap (2.98 → 1.05). The rest remains.

**The tail rests on few independent events.**
- 29 session-assets moved beyond 3σ, and 12 beyond 4σ.
- The single largest event is ETH on 2026-10-07, at down to −6.9σ.
- The joint bootstrap includes this sampling uncertainty, which is why its CI (upper bound 2.34) is wider than the quote-only CI [0.71, 1.78].

## 3. Post-hoc diagnostics (not pre-registered, no decision)

1. **At the bid, which is what a seller receives.** $\ln(\text{bid}/P^{emp})$ gives $M=0.48$, quote-only CI [0.08, 1.23]. That is RICH, but the lower bound is close to 0, and a joint bootstrap would be wider. The same holds for quotes with bid ≥ 16 ticks: $M=0.45$ [0.31, 0.99].
2. **Per-asset pool, at the bid.**
   - RICH: BTC 0.47, TRX 1.33, XRP 3.21, SOL 2.36 (SOL on 79 rows).
   - CONSISTENT: ETH 1.07 [−0.21, 1.48], AVAX, HYPE.
   - The pooled shape assumption matters at the margin.
3. **Fees** (from Deribit's published schedule, not measured here). The option fee cap is 12.5% of premium, about −0.12 in log terms per side. That takes the bid-based median to about +0.35 before slippage.
4. **Break-even unobserved disaster.** I added a symmetric mass $p$ at $\pm u^*$ to every LOSO pool. Because all origins in a session share one settlement, $p$ is roughly the probability per session-asset:

   | $u^*$ | $p$ making mid $M=0$ | $p$ making bid $M=0$ |
   |---|---|---|
   | 6σ | ≈ 1.2% | ≈ 0.4% |
   | 8σ | ≈ 0.9% | ≈ 0.3% |
   | 10σ | ≈ 0.7% | ≈ 0.25% |

   **At the bid, one ±8σ-to-expiry move per asset roughly every 330 sessions (about once a year) erases the measured premium.** A 19-session sample cannot rule that out, and crypto has produced such days.

## 4. Reading

- **Established.** In this sample, 0DTE wings trade well above the realized-outcome value at mid, and modestly above it at the bid. This is robust across all assets, TTEs and benchmarks. It is the first result in the branch that bears directly on a strategy.
- **Not established.** It is not yet shown that selling wings has positive expectation. The bid-side premium is small enough that one rare disaster per asset-year, which is not observable in 19 sessions, would account for it. This is a peso problem, and more 0DTE sessions alone solve it slowly: the VRP boundary caps the sample at about 48 sessions.

## 5. Proposed next steps

1. **Module 10c: long-history physical tail.** The outcome side of 10b does not need option data. It needs only 5-minute index or perpetual prices, the same ex-ante $D$ construction, and the 08:00 settlement. Years of Deribit index or perpetual 5-minute history would give about 1,000+ sessions per asset instead of 18.
   - Pre-register that the long-history $u$ pool replaces the LOSO pool. The quote rows and statistic stay the same, the verdict is taken at the bid as well as at the mid, and the break-even disaster analysis is reported.
   - Ivan would need to provide the data (VM download).
2. **Rerun Modules 08–10b unchanged** as quote sessions accumulate, up to 2026-11-06.
3. **No trading conclusion** until 10c. If 10c confirms bid-side richness against multi-year tails, the next module is a pre-registered, risk-limited wing-selling backtest: defined-risk spreads, with expected shortfall rather than mean P&L as the gate.

## 6. Questions for ChatGPT

1. Is a long-history pool built with the same ex-ante $D$ the right fix for the peso problem? Or would you add regime conditioning, such as volatility level, so that calm-period history does not dilute crisis tails?
2. Do you read the per-asset split as evidence that the premium is concentrated (BTC, TRX, XRP), or as noise on 18 settled sessions?

## 7. Decisions needed from Ivan

1. Accept the 10b verdict and the reading in §4.
2. Approve Module 10c, and say whether long 5-minute index or perpetual history can be downloaded on the VM. Deribit index or BTC-PERPETUAL 5-minute candles for each asset, ideally ≥ 2 years.
