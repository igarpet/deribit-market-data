# Pairs fOU 001 — Stage A: the thesis fOU pairs pipeline at 5 minutes on crypto perpetuals (pre-registration)

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-16 |
| Branch | Pairs fOU (new) |
| Source | García Petrov (2026), *Fractional Ornstein–Uhlenbeck Statistical Arbitrage with Synthetic Equity Options* (UNED master's thesis) |
| Ivan's choices | Rolling 7-day formation; 4-hour (48-bar) maximum horizon; options (Stage B) only if Stage A shows a gross edge |
| Status | Pre-registration. Written before any real-data run. |

## 1. Prior

The thesis itself found no established edge on daily equities:
- predicted 70% convergence against 33% realised;
- HAC alpha p = 0.65;
- 44th-percentile placebo rank;
- 58% maximum drawdown.

This stage asks whether the same machinery finds convergence that pays at 5 minutes on crypto perpetuals, before any option layer.

## 2. Thesis → 5-minute translation

| Thesis (daily equities) | Stage A (5m crypto perps) |
|---|---|
| 464 stocks, correlation pre-screen, 3,828 candidates | BTC, ETH, SOL, XRP: all 6 pairs, no pre-screen |
| Frozen 2016–2022 formation | **Rolling**: previous 2016 bars (7 days), refit every UTC day; parameters frozen for that day's entries |
| Fixed orientation, alphabetically first = y | Same |
| ADF integration screen (5%); Engle–Granger p ≤ 1%; β > 0 | Same, on the 7-day formation window |
| H by change of frequency, clipped [0.01, 0.99]; H < 0.5 | Same |
| fOU moments: μ = mean, $\sigma^2=\overline{(\Delta^2X)^2}/(4-2^{2H})$, $\kappa=(\sigma^2\Gamma(2H+1)/2\mathrm{Var}X)^{1/2H}$, 0 < κ < 2 | Same (Δt = 1 bar) |
| Structural T70 ≤ 252 sessions from Z0 = 1.5; top-40 ranking | Structural T70 ≤ 252 bars (2,000 paths); **no ranking** (only 6 pairs) |
| Entry \|Z\| ≥ 1.5, conditional T70 ≤ 126 sessions, 60-innovation memory, 5,000 paths | Entry \|Z\| ≥ 1.5, **conditional T70 ≤ 48 bars**, 60-innovation memory, 5,000 paths (common random numbers per pair-day) |
| Execute at next close; exit the close after convergence is observed, or at expiry | Same, with expiry = signal bar + T70 |
| Long put + long call (synthetic BS), 10 bp slippage | **Perpetual legs**: −sign(Z) on y and +sign(Z)·β on x; 6 bp per side per leg, i.e. 12 bp round trip of gross notional |

One position per pair at a time. A position may run past midnight on its entry day's parameters. P&L is in bp of gross notional (1 + β).

## 3. Benchmarks (same engine)

- **OU:** H fixed at 0.5 (no fractional memory, no H filter); κ from the OU moment identity.
- **z-score:** cointegrated pair-days only; enter at |Z| ≥ 1.5, exit at μ or after 48 bars, no simulation.

## 4. Sample and evaluation

- **Data:** existing Binance USD-M 1m files → complete 5m bars (bar-end timestamps), 2024-10 → 2026-10.
- **Split:**
  - DEV = entries before 2025-10-01;
  - TEST = entries from 2025-10-01.

  Nothing is fitted across days, so the split separates design looking from scoring.
- **Primary statistic:** fOU, TEST, all pairs pooled, mean net bp per trade. The 95% CI comes from a whole-UTC-day bootstrap (4,000 draws, trade-weighted).
- **Calibration** (thesis §11): mean predicted P(convergence by T70) against the realised share on complete horizons, plus the Brier score.

| Verdict | Condition |
|---|---|
| `NET_EDGE_CANDIDATE` | net CI lower bound > 0 |
| `GROSS_EDGE_ONLY` | gross CI lower bound > 0 (this unlocks Stage B, options) |
| `NO_EDGE` | gross or net CI upper bound < 0 |
| `INCONCLUSIVE` | otherwise |

The OU and z-score benchmarks are reported alongside. fOU's added value is judged by comparing it with them, not by a separate gate.

## 5. Validation before the real run

Tests pass:
- H recovers 0.5 for Brownian motion and 0.3 for anti-persistent fGn;
- κ is recovered for an OU;
- the conditional fGn operator has no memory at H = 0.5 and negative memory at H < 0.5;
- first-passage and T70 arithmetic;
- a planted mean-reverting pair trades with positive gross P&L;
- a random-walk spread is mostly not cointegrated;
- trade P&L sign.

## 6. Expectation (stated in advance)

`NO_EDGE` or `INCONCLUSIVE`. Crypto pairs on a 7-day window will often fail the 1% Engle–Granger test. When they pass, the 5-minute spread moves from 1.5σ to the mean are worth a few bp per gross notional, against 12 bp of costs. The calibration gap seen in the thesis is likely to repeat. A positive gross edge would make Stage B (long 0DTE put + call, optionally filtered by the VD19 model-vs-implied signal) worth building.
