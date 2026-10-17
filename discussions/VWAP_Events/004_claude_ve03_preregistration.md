# VWAP Events 004 — VE03 preregistration: does a VWAP fade scalp have gross edge, and is it VWAP-specific?

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-17 |
| **Branch** | VWAP Events |
| **Responds to** | VWAP Events 003; Ivan's request to test a VWAP scalping strategy on perps |
| **Status** | Preregistration. Fixed before any real-data output. Code `vwap_fade.py` (VE03-0.1.0), tests `tests/test_vwap_fade.py` (5 pass) |

## 1. Question

VE02 showed that price returns to the VWAP within minutes after most crosses, but identically at placebo anchors. **Can a fade scalp turn that into gross edge at the real VWAPs, beyond placebo VWAPs and above costs?**

**Prior** (stated in the chat, 2026-10-17): on a driftless price, any target/stop rule has expected gross P&L ≈ 0, with win rate ≈ S/(S+T); the return to VWAP is partly the average moving toward price.

## 2. Data, events, trade

**Data:** BTC and ETH Binance USD-M perpetual 1m, the full history used in VE02. Anchors and VWAP as VE01/VE02:
- **real:** session 08:00 UTC; 4H 00/04/…/20 UTC;
- **placebo:** session 02/14/20 UTC; 4H offset by +1/+2/+3 h.

**Event (stretch).** After each cross, the first bar $t$ of that crossing segment, at least 30 minutes after the anchor, with
$$|\log C_t-\log VWAP_t|\ \ge\ k\,\sigma_{t-1}\sqrt{15}$$
- $\sigma$ is the causal 1m EWMA volatility (VE01);
- one event per crossing segment;
- $k\in\{1,2\}$.

**Trade.**
- **Position:** fade, $d=-\text{side}$, betting on reversion toward VWAP. The variant "follow" ($d=+\text{side}$) is reported.
- **Entry:** close of bar $t+1$, i.e. 1 minute of latency.
- **Exit, first of:**
  - **target:** $T\in\{5,10,20,40\}$ bp, or **"vwap"**: a touch of the VWAP level, exiting at that level;
  - **stop:** $S\in\{5,10,20,40\}$ bp;
  - **time:** 60 minutes.
- **Same-bar target and stop:** the stop is assumed.
- **Fills:** exits at exact levels, with no gap slippage. This is optimistic.

## 3. Primary and gates

**Primary cell (fixed):** $k=1$, fade, exit at the VWAP touch, stop 20 bp, max 60 min; BTC+ETH and session+4H pooled.

**Gates** (day-block bootstrap, 2,000 draws):

| Gate | Condition |
|---|---|
| **G1, VWAP-specific** | **hour-matched** mean gross (real) − mean gross (placebo) > 0, with q2.5 > 0. Within each UTC entry hour, weighted by real counts; the unmatched difference is reported |
| **G2, tradeable** | mean gross (real) q2.5 > 7 bp (cheapest perp taker round trip) |

| Verdict | Condition |
|---|---|
| `VWAP_FADE_EDGE` | G1 and G2 |
| `VWAP_SPECIFIC_BUT_UNTRADEABLE` | G1 only |
| `GENERIC_EDGE_NOT_VWAP` | G2 only |
| `NO_EDGE` | neither |

**Reported, not gated:**
- the full grid (k × mode × target × stop, real and placebo), with win rate, mean win, mean loss and n;
- net at 4 bp (maker both legs, fills assumed; optimistic), 7 bp and 12 bp;
- per asset / anchor;
- a pre-2026-04-01 / post split.

## 4. Synthetic validation

| Check | Result |
|---|---|
| Hand-built paths | target, stop, time exit, stop-wins-ties, VWAP-touch exit pays exactly the distance to the level, and no phantom target if price never returns |
| Random walk, no VWAP effect | aggregated from a true 2-second path (`synthetic_path_klines`), with volatility seasonality peaking near 08:00 and 14:30 UTC: primary gross within ±1.3 bp across seeds; verdict never `VWAP_FADE_EDGE` |
| G1 calibration | **0 false passes in 40 null runs**; z mean −0.08, s.d. 0.95 |
| Win rates | follow $S/(S+T)$ (e.g. T=10/S=40 → 0.77 vs 0.80 theory), and every cell's gross is within about ±1 bp |

**Changes found on synthetic data before any real run:**
1. **Matched by hour.** The first smoke run (seasonality at 08:00) gave a false G1 with the unmatched difference, so G1 is now hour-matched.
2. **Path-true synthetic data.** The shared `synthetic_klines` generator has high/low wiggles that are not a price path, which biased gross by about +1.4 bp. Validation now uses path-true synthetic data. Real klines are true path extremes.

## 5. Expectation (stated in advance)

`NO_EDGE`:
- primary gross 0 ± 2 bp;
- real equal to placebo within ±1 bp;
- win rates near $S/(S+T)$ across the grid;
- every cell below 7 bp.

The VWAP-exit cell will show a high win rate (60–80%) with losses larger than wins: the negative-skew profile that looks attractive and isn't.

## Decisions needed from Ivan

1. Approve and run. Expect about 5–10 minutes.
