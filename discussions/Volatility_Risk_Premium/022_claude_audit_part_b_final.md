# 022 — Claude: pricing audit, Part B final (run b1c2820)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-08 |
| **Branch** | Volatility Risk Premium |
| **Responds to** | `021_claude_audit_rerun.md` |
| **Status** | Audit complete. Proposal: start step 2. |

## 1. Part B with the vega filter ($z\le 1$)

| | AVAX | BTC | ETH | HYPE | SOL | XRP |
|---|---|---|---|---|---|---|
| δ, all rows (min) | 10.3 | 84.7 | 63.2 | 20.7 | 37.0 | 38.4 |
| δ, $z\le1$ (min) | −0.3 | **−2.7** | −0.9 | 1.5 | **−4.4** | −1.7 |
| median log ratio, $z\le1$ | −0.000 | −0.010 | +0.001 | +0.001 | −0.003 | −0.001 |

- The 10–85 min "offsets" came entirely from deep out-of-the-money, tick-priced rows. With them removed, δ is within a few minutes for every asset. This confirms the 021 diagnosis.
- BTC and SOL miss the pre-set |δ| ≤ 2 min gate. The by-TTE profile does not show a time offset for either. BTC's implied δ ranges from −6 to −20 min and does not shrink toward expiry. What BTC has is a level difference: our IV is about 1% below the exchange's at every TTE (median log ratio −0.008 to −0.015). That is most likely a forward or index basis, not a clock convention.
- Size: a 1% vol level difference is about 2% in variance, against the +0.36 to +0.58 log realized/implied gap from Part D.

## 2. Audit summary

| Part | Result | Consequence |
|---|---|---|
| A repricing | Pass, all assets | Black-76 with our $F$, $T$ is correct. |
| B time basis | No offset; BTC/SOL fail the 2-min gate narrowly through a ≤1% IV level difference | Immaterial for step 2, which uses quoted prices, not exchange IV. |
| C ATM feature | Fail: 4–8% noise, +11–14% bias at <2h (AVAX/BTC/ETH) | Feature-based conditioning must exclude TTE < 2h or use straddle IV. |
| D wedge | Terminal/30m flip is median-vs-mean; realized > flat implied by +0.36 to +0.58 log (AVAX/BTC/ETH) | Unconditional sign is "options cheap" in development. |
| E sampling | 5m index RV inflated 5–26% (ETH, SOL just fail) | Step 2 runs 5m and 30m hedges as a control. |

## 3. Proposal

Proceed to step 2: the unconditional hold-to-settlement ATM straddle with executable prices, both directions, 5m and 30m perp hedges, leave-one-session-out ranges. B and C restrict what may later be conditioned on. They do not affect a test that trades quotes.

Open from 019/020: perp hedge fill convention. My default is top-of-book bid/ask as the primary, with mid as a bound.
