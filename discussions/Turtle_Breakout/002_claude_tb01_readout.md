# 002 — Claude: TB 01 readout — TREND_EDGE as registered; beats luck robustly, the buy-and-hold gate is not credible

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-25 |
| **Branch** | Turtle Breakout |
| **Responds to** | TB 001; Ivan's run `6f8ce4a` (Binance USD-M 1m + funding, 2020-01 → 2026-10-24) |
| **Status** | Result. Pre-registered verdict `TREND_EDGE`; G2 judged not credible (benchmark flaw), so TB 02 retests it |

## 1. Primary

**Run:** daily, long only, BTC/ETH/SOL/XRP, 2020-10-15 → 2026-10-24, net of fees, slippage and funding.

| | Turtle | N2 random entries (1,000) | B&H ×0.41, never rebalanced |
|---|---|---|---|
| CAGR | **74.5%** | median 23.9%, max 58.0% | 27.7% |
| Sharpe | **1.57** | q95 1.33, max 1.63 | 0.77 |
| Max drawdown | −36.1% | median −32.4% | −98.0% |

**Activity:**
- 180 trades, 40.6% winners, profit factor 2.42, average hold 17.6 days.
- Mean gross leverage 0.41, maximum 2.24.
- Costs: fees $67k, funding $288k.

**Gates:**

| Gate | Result | Status |
|---|---|---|
| **G1** | 0 of 1,000 random portfolios reach the real CAGR. On Sharpe the share is 0.003, so the gap is not explained by the higher exposure (0.41 vs 0.28). | **PASS** |
| **G2** | Sharpe difference +0.80 [+0.04, +1.45] | **PASS**, but see §2 |

## 2. Why G2 is not credible

- **The flaw.** The benchmark buys equal notional once and never rebalances.
  - SOL's notional grew about 80× in 2021, and funding was paid on that grown notional.
  - At 1× leverage the benchmark's equity reaches **−$77k** on 2022-11-09.
  - Its Sharpe (0.77) and drawdown (−98%) are therefore products of the construction, not of "doing nothing".
- **Pre-registration.** I raised this choice as a question in TB 001 §7 but did not fix it before the run. The registered verdict stands as recorded. The substantive answer to "does it beat doing nothing" moves to **TB 02** (TB 003), which uses constant-mix benchmarks.

## 3. Reported (no decision)

**N1 trade twins** (share of random twins ≥ real):

| Asset | Share |
|---|---|
| ALL | **0.0045** |
| XRP | 0.0125 |
| SOL | 0.055 |
| ETH | 0.14 |
| BTC | 0.21 |

Bitcoin, the review's only significant market, is the weakest here.

**Variants:**

| Variant | CAGR | Sharpe | Max DD | Note |
|---|---|---|---|---|
| No funding | 85.7% | 1.74 | −35.2% | |
| Short only | −9.0% | 0.15 | −90% | |
| Long + short | 70.3% | 1.34 | −43% | |
| Hourly (long) | ruined | — | — | 4,906 trades, fees $6.0M |

**Single assets, own history:** CAGR 11–20%, Sharpe 0.88–1.19, max DD −13 to −22%.

**Halves:**

| Window | CAGR | Sharpe |
|---|---|---|
| 2020-10 → 2023-06 | 64.9% | 1.45 |
| 2023-07 → end | 83.3% | 1.66 |

**Concentration.**
- One XRP trade (2024-11-07 → 2025-01-27, +394% on notional) is 42% of dollar P&L.
- The top 5 trades are 79% of dollar P&L.
- Per unit notional, returns cluster in 2021 (+8.5, mostly SOL), 2024 (+3.9) and 2025 (+4.1); 2022 is −1.0.

## 4. Reading

- **What is established:**
  - Breakout timing on these four perpetuals beat random entries with the same exits and sizing, by a wide margin, in both halves.
  - The effect is long-side only; the short side loses.
- **What is not yet established:**
  1. That it beats a proper buy-and-hold → TB 02.
  2. That it is not a selection artefact: the four assets were chosen in 2026, after SOL's and XRP's runs → TB 03 (point-in-time universe including delisted perps).
  3. That it survives without its few largest trends → TB 03.

## 5. Decisions needed from Ivan

1. Accept this readout and the TB 02 design in TB 003.
