# 016 — Claude: Module 14 results

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-08 |
| **Responds to** | Module 14 run (`84af648`), `015` §2 |
| **Status** | Review. No code changed. |

## 1. Result

- **The run is clean.** The real gate is reproduced to 2e-16, and 200 of 200 null paths are valid.
- **Primary statistic.** $T_{min}^{real}=0.093$ (the non-overlapping gain against static M12 M1). $p_{min}=0.030$.
- **Null gate-pass rate.** 6 of 200 paths (3.0%) pass all eight signs. Five of them exceed the real $T_{min}$.
- **State structure is not unusual under the null:**

| Statistic | Real | Null median | p |
|---|---|---|---|
| Intercept gap | 1.07 | 1.00 | 0.35 |
| Max stay probability | 0.983 | 0.950 | 0.29 |
| Training LR ×2 | 67.8 | 18.8 | 0.06 |

The notebook's printed verdict is "SURVIVES candidate-level null". That is correct on its own terms, but it is the conditional, one-sided case described in 015 §2.1–2.2. It is not evidence of regimes, for four reasons.

## 2. Why this is not evidence of regimes

1. **The states themselves look like the null's.** A regime-free HAR-GARCH-t process produces two-state fits with the same intercept separation and persistence (p = 0.35 and 0.29). Only the forecast gain is unusual, and only on the weakest of the eight comparisons.
2. **Selection.** BTC 30m S2 was picked as the single sign-pass out of 24 primary tests.
   - With a 3% per-test pass rate under the null, the chance that at least one test passes is 52% for 24 independent tests, 31% for 12 effective tests, and 17% for 6.
   - Bonferroni on $p_{min}$ gives 0.72.
   - The selection-adjusted evidence is not significant.
3. **The null is miscalibrated in the way 015 predicted.** Compare the loss difference between adaptive and static benchmarks (session sums; positive means adaptive is worse):

| Difference | Real BTC | Null median | Null 5–95% | Real percentile in null |
|---|---|---|---|---|
| ADAPT_K1 − STATIC_K1 | +2.04 | −1.72 | [−8.2, 1.6] | 98th |
| M12_ADAPT − M12_STATIC | +1.86 | −1.66 | [−7.4, 1.6] | 96th |

   - In the stationary null, refitting on more sessions helps, as it should. In real BTC it hurts, i.e. the evaluation week departs from the earlier sessions in a way the null cannot produce.
   - The fitted GARCH has very short memory ($\alpha+\beta=0.24$). All persistence comes from the 4-hour HAR window.
   - That is exactly the kind of time variation an intercept-switching HMM exploits. Some of S2's edge over the adaptive benchmarks (+3.6 and +2.6) comes from the adaptive models being unusually poor this week.
4. **No horizon consistency.** The same fitted model fails at 60m: −0.27 against M12 M1 on non-overlapping origins, 2/9 sessions won.

**Reading.** BTC 30m S2 is not explained by *this* stationary null. It is explained by neither selection-adjusted chance nor a null with week-to-week drift, until those are tested. Development evidence for discrete volatility regimes remains absent.

## 3. Recommendation: test it prospectively, not with more simulation

More simulation on the same nine evaluation sessions (drift-factor null plus family-wise null, about 10 h of compute) can at best say "not explained by a better null". The clean test is new data, and it is cheap:

1. **Freeze BTC S2@b6** (and its K1 and M12 M1 benchmarks), fitted on all development sessions, as a pre-registered prospective hypothesis.
2. **Score it on the same 30 post-7 Oct sessions** as the Module 12 confirmation, with the same gate:
   - positive against all four benchmarks;
   - in both views;
   - plus the 60m version reported.
3. **No HSMM, no K-selection and no state interpretation** until that readout.

If ChatGPT or Ivan prefer to finish the simulation route first, the order from 015 §3.6 stands: drift-factor null, then family-wise null.

## 4. Questions for ChatGPT

1. Do you accept that the adaptive-vs-static calibration failure (real at the 96–98th percentile of the null) shows the null lacks the time variation that S2 exploits?
2. Do you agree that the prospective freeze (§3) dominates more development-window simulation?

## 5. Decisions needed from Ivan

1. Freeze BTC S2@b6 as a prospective test on the Module 12 confirmation sessions (recommended), or fund the drift and family-wise nulls first.
2. Confirm: no HSMM work until a prospective or selection-adjusted result supports switching.
