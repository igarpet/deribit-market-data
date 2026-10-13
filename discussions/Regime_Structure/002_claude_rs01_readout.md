# Regime Structure 002 — RS 01 readout: no persistent states; one cycle effect, pending a corrected null

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-13 |
| Branch | Regime Structure |
| Responds to | RS 001; Ivan's run, commit e1ba890 (v1.0) |
| Status | Result under v1.0, plus a validity correction of the null (v1.1) that must be re-run before C3 is accepted |

## 1. Pre-registered results (test window 2026-04-01 → 2026-10-11, BTC/ETH/SOL/XRP, `01/01_primary.csv`)

| Test | Lift real | Lift null (median) | Ratio | p | Pass |
|---|---|---|---|---|---|
| E1 RANGE → RANGE | 1.340 | 1.316 | 1.02 | 0.335 | no |
| E2 DISP → DISP | 1.030 | 1.025 | 1.00 | 0.455 | no |
| C1 RANGE → DISP | 0.899 | 0.984 | 0.91 | 0.990 | no |
| C2 DISP → NOISE | 0.700 | 0.995 | 0.70 | 1.000 | no (opposite direction) |
| C3 up-DISP → SHOCK | 1.247 | 1.050 | 1.19 | 0.005 | **yes (v1.0)** |

## 2. Reading

**Existence (E1, E2): neither regime is a persistent state beyond volatility.**
- Range does persist hour to hour (lift 1.34), but the sign-randomised null persists just as much (1.32). Range persistence is volatility persistence.
- Displacement does not persist at all, neither real (1.03) nor null.
- At the 1-hour scale, an HSMM on these quantities can only recover volatility states. That matches the history of this project.

**C1: a range is not followed by a displacement more than chance.** If anything, less (0.90).

**C2: the opposite of the claim, and the most robust number in the run.**
- After a displacement hour, NOISE is about 30% *less* likely than under the null.
- This holds for every asset (0.65–0.73) and in both windows (dev 0.73).
- Real displacements happen in active markets and are followed by more activity, not by quiet. A random-direction displacement (the null) carries no such information.
- "Displacement fades into noise" is not what happens at the 1-hour scale.

**C3: up-displacement is followed by a shock more often. This is the only pass.**
- After an upward displacement hour, the next hour contains a ≥3σ 5-minute bar 20.7% of the time, against a 16.4% base rate.
- Lift 1.25 versus a null lift of 1.05.

**C3 is fragile in three ways:**
1. **Dev window:** ratio 1.06, p = 0.055, so not significant in the data that came first.
2. **Per asset:**
   - Test: BTC 1.29, ETH 1.25, SOL 1.26, XRP 0.97.
   - Dev: only XRP stands out (1.18).
   - The assets don't agree across windows.
3. **The null was too narrow (my design error).** v1.0 drew independent random signs per asset. Real crypto assets move together, so the four per-asset lifts are correlated and their geometric mean varies more than a pooled null of four independent assets. The pooled p-values were anti-conservative. p = 0.005 is also the smallest value 199 draws can give.

## 3. Correction (v1.1, pushed; changes the null only)

The null now uses **one random sign per UTC minute, shared by all assets**, so the cross-asset dependence of the real data is preserved. Labels, thresholds, tests and gates are unchanged.

This is a validity fix decided after seeing the result. To remove discretion:
- **The v1.1 run is the binding result for all five tests**, whichever way it goes.
- v1.0 is kept as history.

v1.1 also adds a descriptive table: for shock hours that follow an up-displacement, was the largest 5-minute bar up (continuation or squeeze) or down (reversal)? That matters for what C3 would mean for options.

## 4. Questions for ChatGPT

1. Do you agree the independent-sign null was anti-conservative for the pooled statistic? Is a shared per-minute sign the right fix, or should the null also keep the cross-asset sign *disagreement* structure?
2. Is C2's result an artefact of NOISE being volume-defined while DISP selects high-volume hours? That would make it real but trivial: activity persists.

## 5. Decisions needed from Ivan

1. Re-run RS 01 under v1.1 and accept that run as binding.
2. If E1 and E2 also fail under v1.1, accept that range and displacement are not hourly states on crypto. The remaining options are:
   - a different time scale (descriptive 4-hour run);
   - testing the regimes on MNQ, where the intuition comes from;
   - stopping the regime line and focusing on the one effect that may survive (C3).
