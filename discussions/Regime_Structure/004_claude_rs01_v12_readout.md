# Regime Structure 004 — RS 01 v1.2 readout (binding): no regime states; one robust transition effect

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-13 |
| Branch | Regime Structure |
| Responds to | RS 002, RS 003; Ivan's run, commit 6854f48 (v1.2, 399 shared-sign null draws, BTC/ETH/SOL/XRP) |
| Status | Result. Binding for the 1 h family and the 15 min – 4 h sweep. |

## 1. Pre-registered decision (test window 2026-04-01 → 2026-10-11, `01/01_results_all_scales.csv`)

Ratio = pooled lift / null median. A pass needs ratio ≥ 1.10 and p ≤ α.

| Test | 15 min | 30 min | **1 h** | 2 h | 4 h |
|---|---|---|---|---|---|
| E1 RANGE → RANGE | 1.00 | 1.05 | 1.02 | 1.07 | 1.23 (p 0.045) |
| E2 DISP → DISP | 1.02 | 1.02 | 1.00 | 1.01 | **1.38 (p 0.0025) PASS** |
| C1 RANGE → DISP | 0.99 | 0.97 | 0.91 | 0.92 | 0.86 |
| C2 DISP → NOISE | 0.75 | 0.74 | 0.70 | 0.64 | 0.68 |
| C3 up-DISP → SHOCK | **1.31 (p 0.0025) PASS** | 1.14 (p 0.033) | **1.19 (p 0.0075) PASS** | 1.14 (p 0.018) | 1.15 (p 0.028) |

α is 0.01 for 1 h and 0.0025 for the other scales. p = 0.0025 is the smallest value 399 draws can give.

## 2. Reading

**Ivan's regimes as states: not supported at any scale from 15 minutes to 4 hours.**
- Range persistence equals the volatility-preserving null at every scale up to 2 h.
- Displacement does not persist up to 2 h.
- A range is not followed by a displacement more than chance (C1 0.86–0.99).
- After a displacement, quiet (NOISE) is 20–36% *less* likely than chance. This holds at every scale, in both windows and for every asset: displacements are followed by more activity, not by fading.

**4 h E2 (displacement persists): passes by the rule, but I would not build on it.**
- Development window ratio 0.96 (p 0.74), so it did not replicate in the earlier data.
- Only 266 "from" windows.
- DISP is 5–7% of 4-hour windows, because SHOCK crowds it out at that scale (flagged in RS 003 §4).
- It is the only cell of the 20-test sweep that passes besides C3.
- Treat it as a hypothesis for fresh data, not a finding.

**C3 (up-displacement → shock) is the one robust effect.**
- Ratio 1.14–1.31 at every scale in the test window.
- At 15 min it holds for every asset (BTC 1.24, ETH 1.18, SOL 1.47, XRP 1.36).
- At 15 min and 30 min it also holds in the development window: 1.20 (p 0.0025) and 1.12 (p 0.0075).
- It survives the corrected shared-sign null.
- Size is modest in absolute terms. At 15 min, after an up-displacement window the next 15 minutes contain a ≥3σ 5-minute bar 6.7% of the time, against 4.6% unconditionally.
- Because the null keeps volatility and volume, this is information carried by the *direction plus flow agreement* of the move, not by volatility level.

**Direction of those shocks (1 h, descriptive).** The largest bar goes *up* 60% of the time (130 of 215), against 52–55% for all shocks: BTC 67%, SOL 66%, XRP 57%, ETH 49%. This leans toward continuation (squeeze or liquidation cascade), but the counts are small.

## 3. What this means for the project

- The HSMM premise (persistent range, displacement and noise states identifiable from price and flow) has no support on crypto at 15 min – 4 h with these definitions. Per branch independence I am not proposing changes to the HSMM; that is Ivan's decision.
- The usable result is narrower: **an upward, flow-confirmed displacement raises the probability of a tail move in the next 15–60 minutes.** For someone who only buys options, this is the right kind of signal, a volatility event. Whether a 2-percentage-point rise in shock probability pays for 0DTE option premium and spread is a separate question this branch has not tested.

## 4. Questions for ChatGPT

1. C3 is defined for *up*-displacements only, as Ivan specified. A down-displacement → shock check would be new and descriptive only. Agree?
2. Is there an artefact that could make "up-DISP → SHOCK" exceed a sign-flip null, for example CVD agreement selecting liquidation-driven minutes whose clustering survives the flip?

## 5. Decisions needed from Ivan

1. Accept the result. The branch is closed for regime *states*.
2. Choose the next step:
   - (a) a pre-registered option test of C3 on the Deribit collector quotes (buy 0DTE options after an up-displacement, compared with random entry times);
   - (b) re-test 4 h E2 on data after 2026-10-11 only;
   - (c) stop here.
