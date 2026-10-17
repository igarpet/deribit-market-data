# VWAP Events 007 — VE04 readout: fading extreme displacement has no edge either; VWAP branch closed

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-17 |
| **Branch** | VWAP Events |
| **Responds to** | VWAP Events 006; Ivan's run `6a27404` |
| **Status** | Result. **`NO_EDGE`.** G1 FAIL, G2 FAIL. |

## 1. Selection and verdict

**DEV-selected cell** (highest DEV lower bound): ≥5σ displacement, exit at VWAP touch, 20 bp stop, 60 min max. DEV: +1.23 bp [0.32, 2.12], n = 9,839.

**TEST** (from 2026-04-01; 3,276 trades):

| Quantity | Value |
|---|---|
| Real gross | **+0.75 bp** [−0.39, +2.02] |
| Placebo | +0.37 bp |
| Real − placebo, hour-matched | +0.19 bp [−1.09, +1.46] |
| Win rate | 41% |
| Net | −6.2 bp at 7 bp; −11.2 bp at 12 bp |

The DEV edge shrank by about 40% out of sample, and the TEST interval includes zero.

## 2. All 144 cells (reported only; `04_all_cells.csv`)

- **DEV real gross ranges from −1.2 to +1.2 bp; TEST from −2.3 to +2.5 bp.** No TEST lower bound in any cell is above zero.
- **Real minus placebo** across all TEST cells averages +0.13 bp (range −1.2 to +2.9).
- **The largest TEST values** (+1.7 to +2.5 bp) are wide-stop (80 bp), 4-hour VWAP-exit cells. Their intervals run from about −2.4 to +6 bp: noise in the tail.
- **The DEV ranking does not carry to TEST.** The best DEV cells (k5 / p1 with 20 bp stops) are +0.1 to +0.8 on TEST.

**Event counts were far higher than I expected.** I expected hundreds; there are thousands (real anchors, DEV / TEST):

| Threshold | DEV | TEST |
|---|---|---|
| ≥5σ | 9,839 | 3,276 |
| ≥1% | 21,009 | 4,392 |

Moves of 1% or more from VWAP are common in crypto, so the null result is well powered.

## 3. Reading

- **Larger displacement adds no reversion beyond placebo anchors** and no gross edge near fees.
- **The VWAP question has now been answered four ways**, all against arbitrary-anchor placebos:

| Study | Finding |
|---|---|
| VWAP 002 | touch → violence |
| VE02 | touch / cross / return statistics |
| VE03 | fade after a cross |
| VE04 | fade extreme displacement |

- **Branch closed.**
