# VWAP Events 005 — VE03 readout: the VWAP fade scalp has no edge, and the real VWAP is identical to placebo

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-17 |
| **Branch** | VWAP Events |
| **Responds to** | VWAP Events 004; Ivan's run `a7431db` |
| **Status** | Result. **`NO_EDGE`.** G1 FAIL, G2 FAIL. |

## 1. Primary

Fade the first ≥1σ stretch after a cross; exit at the VWAP touch, a 20 bp stop or 60 min. BTC+ETH, session+4H pooled; 38,174 real trades.

| Quantity | Value |
|---|---|
| Real gross | **−0.16 bp** [−0.43, +0.11] |
| Placebo gross | +0.03 bp |
| Real − placebo, hour-matched | −0.14 bp [−0.39, +0.11] |
| Win rate | 47.5% |
| Net | −4.2 bp at 4 bp, −7.2 at 7 bp, −12.2 at 12 bp |

## 2. Grid (reported only; `03_grid.csv`)

**Every cell for both k values, fade and follow, is between −1.2 and +0.8 bp gross.** The best real cell is k=2, fade, VWAP exit, 5 bp stop: +0.81 bp [0.53, …]. The same cell at the placebo anchors gives +0.81.

The small positive cells (+0.4 to +0.8 bp) have three things in common:
- they fade with a tight stop and a wider target;
- they appear identically at the placebo anchors;
- they are about a tenth of the cheapest round trip.

This is generic sub-bp short-horizon reversion, likely bid/ask bounce in 1m closes, not a VWAP property.

**Win rates follow the stop/target geometry.** For k=1 fade:

| Target / stop (bp) | Realised win rate | $S/(S+T)$ |
|---|---|---|
| 10 / 40 | 0.76 | 0.80 |
| 5 / 40 | 0.86 | 0.89 |
| 40 / 5 | 0.16 | 0.11 |
| 10 / 10 | 0.50 | 0.50 |

**Mean win × win rate ≈ mean loss × loss rate in every cell.**

**Correction to my expectation (VE 004 §5).** I predicted the VWAP-exit cell would show a 60–80% win rate. At a 20 bp stop it is 48%, because the average distance back to VWAP (about 20 bp) is as large as the stop. Only with a 40 bp stop does the win rate reach 59%, and then the mean loss of −31.5 bp against a mean win of +20.8 bp gives −0.5 bp.

## 3. Splits

By asset, anchor and real/placebo, the primary cell is within ±0.3 bp, with every interval spanning zero.

**Reporting bug (no effect on any gate or grid number):** the pre/post-2026-04 split labelled every trade "pre". Day stamps are microsecond epochs and were parsed as nanoseconds. Only the split table is affected.

## 4. Reading

- **A VWAP fade is a random-walk bet.** Its P&L is fixed by the stop/target geometry and is gross ≈ 0. The real 08:00 and 4H VWAPs give the same result as arbitrary anchors.
- **This closes the VWAP question three ways:**

| Study | Finding |
|---|---|
| VWAP 002 | touches do not precede violent moves |
| VE02 | touch, cross and return statistics equal placebo |
| VE03 | fading the stretch has no gross edge and nothing VWAP-specific |

- **Branch status: VWAP Events closed.**
