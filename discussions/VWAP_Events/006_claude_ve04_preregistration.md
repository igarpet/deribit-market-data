# VWAP Events 006 — VE04 preregistration: fading extreme displacement from VWAP

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-17 |
| **Branch** | VWAP Events |
| **Responds to** | VWAP Events 005; Ivan: "what if we took the trades when price was very displaced from VWAP?" |
| **Status** | Preregistration. Fixed before any real-data output. Code `vwap_extreme.py` (VE04-0.1.0), tests `tests/test_vwap_extreme.py` (3 pass) |

## 1. Question

VE03 faded the first ≥1–2σ stretch after a cross. It found no edge, and the 2σ cells equalled placebo. **Does fading extreme displacement from VWAP, at any time, have gross edge beyond fees, and is it specific to the real VWAP?**

## 2. Events and trade

Same data, anchors and placebo anchors as VE02/VE03 (BTC and ETH perps, 1m, about 741 days).

**Event:** bar $t$, at least 30 min after the anchor, with displacement $D_t=|\log C_t-\log VWAP_t|$ at least one threshold. After an event, a 30-minute cooldown applies per anchor and threshold.

| Threshold type | Rule | Values |
|---|---|---|
| σ-scaled | $D_t\ge k\,\sigma_{t-1}\sqrt{15}$ | $k\in\{3,4,5\}$ |
| Percent | $D_t\ge x$ | $x\in\{0.5\%,0.75\%,1\%\}$ |

**Trade:**
- **Position:** fade toward VWAP.
- **Entry:** close of bar $t+1$.
- **Exits:** as VE03:
  - **target:** $T\in\{20,40,80\}$ bp or a VWAP touch;
  - **stop:** $S\in\{20,40,80\}$ bp;
  - **max hold:** $\in\{60,240\}$ min.
- **Same-bar ties:** the stop is assumed.

That is 144 cells.

## 3. Selection and gates

- **DEV** = entries before 2026-04-01; **TEST** = entries from 2026-04-01.
- **Selection (DEV, real anchors):** the cell with the highest day-bootstrap 95% lower bound of mean gross bp, with $n_{DEV}\ge50$.
- **Gates (TEST, selected cell only)**, day-block bootstrap with 2,000 draws:

| Gate | Condition |
|---|---|
| **G1, VWAP-specific** | hour-matched real − placebo mean gross > 0, q2.5 > 0 |
| **G2, tradeable** | real mean gross q2.5 > 7 bp |

| Verdict | Condition |
|---|---|
| `VWAP_EXTREME_EDGE` | G1 and G2 |
| `VWAP_SPECIFIC_BUT_UNTRADEABLE` | G1 only |
| `GENERIC_EDGE_NOT_VWAP` | G2 only |
| `NO_EDGE` | neither |

**Reported, not gated:**
- every cell, real and placebo, DEV and TEST, with n, win rate and mean displacement;
- net at 4, 7 and 12 bp.

## 4. Synthetic validation

| Check | Result |
|---|---|
| Hand-built path | threshold, warm-up and cooldown behave as specified |
| Day ids | unit-safe; fixes the VE03 split-table bug; DEV/TEST split is correct |
| Path-true random walk | `NO_EDGE` |
| Selection + G1 on 20 null runs | 0 false passes; TEST gross mean +0.18 bp, s.d. 0.88 |

## 5. Expectation (stated in advance)

`NO_EDGE`:
- the DEV-selected cell's TEST gross is 0 ± 5 bp, with real equal to placebo;
- event counts at ≥4σ and ≥1% are small (hundreds), so intervals are wide;
- the large-stop cells show the classic profile, i.e. a high win rate with rare large losses.

## Decisions needed from Ivan

1. Approve and run. Expect about 3–8 minutes.
