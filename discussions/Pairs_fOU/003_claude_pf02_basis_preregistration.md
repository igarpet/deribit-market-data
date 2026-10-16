# Pairs fOU 003 — PF02: the thesis fOU pipeline on construction-tied spreads (pre-registration)

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-16 |
| Branch | Pairs fOU |
| Responds to | Pairs fOU 002 (Stage A failed at the formation stage); Ivan: "as many as we can, all available derivatives, options too"; top ~30 USDT assets; taker costs as the gate |
| Status | Pre-registration. Written before any data for these instruments was downloaded. |

## 1. Why construction-tied spreads

Stage A failed because crypto majors are not cointegrated: the equilibrium was estimated and then drifted. Here the equilibrium is enforced by arbitrage, through funding and expiry convergence, so the fOU machinery has a stable target. The open question is whether the swings exceed taker costs.

## 2. Families and universe

**Universe (fixed now, before data):** BTC ETH SOL XRP BNB DOGE ADA LINK DOT LTC BCH UNI NEAR APT ARB OP SUI FIL ATOM ETC INJ AAVE TON WIF HBAR XLM SEI TIA ENA WLD. AVAX, TRX and HYPE are excluded, per Ivan earlier.

| Family | y | x | β | Funding | Taker cost per side per leg | Round trip, per gross notional |
|---|---|---|---|---|---|---|
| **SP** (primary) | USD-M perpetual | spot | 1 | on y (actual Binance rates) | 6 bp (perp), 11 bp (spot) | 17 bp |
| **PQ** | front USD-M quarterly (rolled 14 days before expiry; formation never spans a roll) | perpetual | 1 | on x | 6 bp, 6 bp | 12 bp |

Costs are 0.05% futures taker plus 1 bp, and 0.10% spot taker plus 1 bp. They are fixed now; Ivan should check them against his account tier.

**Options** (put-call parity against the forward or perp) are a separate diagnostic (PF03) on Deribit collector data. Trading a parity violation needs a short option leg, so it needs Ivan's decision first.

## 3. Engine changes from PF01 (nothing else changes)

- **Screen:** the Engle–Granger regression is replaced by β = 1 and ADF on the basis level, p ≤ 1% (unit root rejected), over the 7-day formation window. H, fOU moments, structural T70, conditional T70 ≤ 48 bars, |Z| ≥ 1.5, next-bar execution and exits are all unchanged (`fou_pairs` v0.1.1 hooks; PF01 behaviour unchanged).
- **P&L in bp of gross notional:**
  $$\text{net}=\text{gross}+\text{funding}-2(c_y+c_x)/2$$
  Funding is −position_perp × Σ rates over (entry, exit].
- **Sides:**
  - `short_y` (short perp, long spot) needs no borrowing.
  - `long_y` needs short spot via margin. Its borrow cost is not modelled, so both sides are reported separately.

## 4. Evaluation

- **Split:** DEV = entries before 2025-10-01; TEST = from 2025-10-01.
- **Primary statistic:** SP, fOU, TEST, all assets pooled, mean net bp per trade, 95% whole-day bootstrap (trade-weighted).

| Verdict | Condition |
|---|---|
| `NET_EDGE_CANDIDATE` | net lower bound > 0 |
| `GROSS_EDGE_ONLY` | gross lower bound > 0 |
| `NO_EDGE` | net upper bound < 0 |
| `INCONCLUSIVE` | otherwise |

- **PQ** is reported with the same rule as secondary.
- **Also reported:**
  - OU and z-score benchmarks;
  - per-asset results;
  - per-side results;
  - funding and cost decomposition;
  - calibration (predicted vs realised convergence);
  - the basis scale: median rolling 7-day s.d. in bp per asset. This is the number to compare with costs.

## 5. Validation before the real run

Tests pass:
- the ADF screen accepts a stationary basis and rejects a random-walk basis;
- quarterly expiries (last Friday 08:00 UTC) and front-contract rolling with a forced gap;
- funding parsing, and funding signs (short perp receives positive funding);
- per-leg cost arithmetic;
- the downloader parses spot microsecond timestamps and both header styles.

**Synthetic smoke run** (basis s.d. 25 bp, half-life about 8 bars):
- gross is about 20 bp per gross notional, against 17 bp of costs;
- calibration is 71% predicted vs 79% realised.

The engine is calibrated when the model is true.

## 6. Expectation (stated in advance)

- **BTC/ETH/SOL/XRP SP:** a basis s.d. of a few bp at 5 minutes, so `NO_EDGE`.
- **Altcoin SP:** larger swings during squeezes, the only plausible source of gross edge. Cost (17 bp per gross notional) is very likely binding.
- **PQ:** BTC/ETH only, with a small calendar-basis s.d.; likely `NO_EDGE`.

A passing SP result driven by `long_y` alone (short spot) would need borrow costs before it means anything.
