# 004 — Claude: TB 02 readout — better Sharpe than every benchmark, none of it significant

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-25 |
| **Branch** | Turtle Breakout |
| **Responds to** | TB 003; Ivan's run `18e355e` |
| **Status** | Result. Pre-registered verdict **`FAILS_B1_B2_B3`** |

## 1. Gated tests

**Strategy:** the TB 01 primary, unchanged and reproduced, over 2020-10-15 → 2026-10-24.

**Benchmarks:** at the strategy's mean leverage (0.41). Δ = Sharpe(strategy) − Sharpe(benchmark), with a 21-day block bootstrap.

| | Sharpe | Δ Sharpe [95%] | Pass | CAGR | Max DD |
|---|---|---|---|---|---|
| **Turtle** | **1.57** | — | — | 74.5% | −36% |
| B1 perp, monthly | 1.06 | +0.50 [−0.08, +1.11] | no | 32.1% | −46% |
| B2 perp, daily | 0.94 | +0.63 [−0.03, +1.29] | no | 26.8% | −48% |
| B3 spot proxy, monthly | 1.21 | +0.35 [−0.32, +0.97] | no | 38.2% | −47% |

## 2. Reported

- **B4, inverse-vol:** Sharpe 0.98; Δ = +0.59 [−0.05, +1.18].
- **B1 at the strategy's volatility** (leverage 0.55): CAGR 41%, max DD −58%, against 74.5% and −36% for the turtle.
- **Alpha vs B1 at 1×:** 39%/yr (simple), Newey–West t = 2.9, β = 0.34. Not the registered test; noted only.
- **Halves vs B3:**
  - 2020-10 → 2023-06: Sharpe 1.449 vs 1.444, Δ = 0.00.
  - 2023-07 → end: Δ = +0.67 [−0.15, +1.38].
- **Years:**
  - 2021: the strategy trailed (+124% vs +182–234%).
  - 2022: it lost less (−20% vs −37%).
  - 2023–2026: it was ahead every year.

## 3. Reading

- **What holds:** the breakout timing beats random entries (TB 01 G1).
- **What does not:** over six years it is **not distinguishable** from a rebalanced long position in the same four coins. This is the same conclusion the reviewed video reached for futures.
- **Where the strategy is ahead:** every point estimate (Sharpe, drawdown, equal-risk return). The standard error of a Sharpe difference over six years (≈0.3) is too large to confirm a difference of about 0.5.

## 4. Next

TB 03 per Ivan (TB 005): intraday windows and volatility-engine sizing.
