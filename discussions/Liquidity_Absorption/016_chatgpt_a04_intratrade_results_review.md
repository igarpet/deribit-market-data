# Liquidity Absorption 016 — A04 intratrade anatomy results review

| Field | Value |
|---|---|
| Author | ChatGPT |
| Date | 2026-10-20 |
| Sources | Executed `04_intratrade_path_anatomy.ipynb`, `04/{04_decision.json,04_primary_event_outcomes.csv,04_landmark_panel.csv,04_landmark_summary.csv,04_competing_exits.csv}` |
| Historical status | **EXPLORATORY outcome-conditioned descriptive analysis, not a causal explanation or independently validated signal** |
| Research cohort | A03d `TAKER_FIRST_POST_SIGNAL`, `VD18_VOL_FIXED`, fixed 240-minute holding |
| Changes | Review document only; no research inputs or simulation code modified |

## 1. Completion and cohort integrity

A04 `LA-A04-0.1.1` executed successfully after the date-unit hotfix. The notebook recorded 16 passing synthetic regression tests. There are 452 A02 candidates, 415 assumed first-print entries, 414 completed trade outcomes (**119 TARGET, 202 STOP, 93 TIMEOUT**), 37 no-structural-target skips and 1 censored trade. Every candidate has a landmark at 1, 5, 15, 30, 60 minutes, resulting in 2,260 rows, of which **1,691** are PRESENT and 0 are missing-minute-window failures. As-of features use only closed one-minute perpetual candles.

The risk set shrinks substantially; 38 rows are outside the completed-outcome cohort at every horizon:

| Post-signal landmark | Still-at-risk rows | Exited by landmark | Not in completed cohort |
|---|---:|---:|---:|
| +1 minute | 411 | 3 | 38 |
| +5 minutes | 401 | 13 | 38 |
| +15 minutes | 354 | 60 | 38 |
| +30 minutes | 303 | 111 | 38 |
| +60 minutes | 222 | 192 | 38 |

These are **different survivor populations**. Do not interpret paths of their medians as longitudinal trajectories of an unchanging set of trades, nor compare a 60m survivor with a trade exited in minute 5.

## 2. Descriptive early price markouts and aggressive flow

Combining BTC and ETH but stratifying conceptually because their conditional regimes differ:

| Minutes | Surviving TARGET | Surviving STOP | Surviving TIMEOUT | Median signed return TARGET / STOP / TIMEOUT (bp) | Median aligned flow TARGET / STOP / TIMEOUT |
|---|---:|---:|---:|---|---|
| 1 | 117 | 201 | 93 | -0.012 / 0.000 / 0.000 | +0.012 / -0.001 / -0.039 |
| 5 | 111 | 197 | 93 | +0.480 / -0.107 / +1.228 | -0.013 / -0.056 / -0.005 |
| 15 | 96 | 165 | 93 | +6.466 / -1.200 / +3.693 | +0.063 / +0.004 / +0.023 |
| 30 | 78 | 132 | 93 | +7.507 / -2.581 / +5.048 | +0.046 / +0.004 / +0.055 |
| 60 | 49 | 80 | 93 | +10.389 / +1.810 / +12.583 | +0.039 / +0.005 / +0.060 |

Note that eventual TIMEOUT trades can initially move **favorably**, often more than eventual stopouts, even though their fixed structural targets are not reached. Negative results are therefore a mixture of directional rejection failures and *target infeasibility / insufficient time*. Aggressive flow differences are much smaller than price markouts. It is a one-minute taker-buy volume proxy, NOT actual bid/ask replenishment, absorption, limit-order queues, or causal identity of liquidity providers.

## 3. Geometry dominates the raw contrast

Among +15m survivors, grouping on the *known-at-entry* target distance divided by VD18 forecast **60m** terminal-return sigma:

| Group | At risk | Eventual TP | Eventual SL | Eventual timeout | Fraction TP |
|---|---:|---:|---:|---:|---:|
| target/sigma60 <= 1 | 71 | 42 | 22 | 7 | 59.2% |
| target/sigma60 > 1 | 283 | 54 | 143 | 86 | 19.1% |

This is a powerful historical **association**, not a causal discovery. There are at least three mechanical confounders: the preselected structural target distance, the fixed 240m expiration, and the use of a **60m** forecast sigma rather than a horizon-matched first-passage model. The ratio's raw outcome separation may reflect reachable targets, rather than predictive sign/absorption quality.

## 4. At +15m: provisional incremental flow/price hypotheses

Among all +15m survivors, target rates are **62/198 = 31.3%** when signed markout > 0, versus **34/156 = 21.8%** when markout <= 0. With flow alignment > 0: **59/190 = 31.1%** eventual TP; flow alignment <= 0: **37/164 = 22.6%**. These bins deliberately use intuitive zero thresholds, yet **they were examined after outcomes** and are not preregistered strategies.

Within less feasible setups (target/sigma60 > 1), there is still a descriptive association: positive signed markout and flow alignment at +15m had **35 TARGET among 127 survivors (27.6%)**, versus **19 TARGET among the other 156 (12.2%)**. These groups are not matched on exact target distance, stop distance, BTC/ETH, side, or target source, so they **do not establish independent predictive lift**, nor show profit after fees from acting 15m later. Within the closer target/sigma60<=1 bin, small samples can exhibit different relations.

The +1m signed return had virtually no target-versus-stop separation. +5m separation was small/inconsistent (BTC conditional target median signed return negative; ETH positive). At +15m, both assets' eventual target survivors had a more favorable median signed price return than stop survivors (BTC +7.97 vs -0.065bp; ETH +6.47 vs -3.47bp), so there is an interesting time-local research hypothesis.

## 5. Technical reporting caution: 240-minute cumulative chart

`04_competing_exits.csv` counts `time_to_exit_min <= h` *exactly*. Although there are **93** eventual TIMEOUT outcomes, the row at **h=240** counts only **42** timeouts, because another **51** exits execute on the first available print marginally **after** the 240-minute limit (median +0.0041 min among >240). All 119 TARGET and 202 STOP exits are correctly counted by 240; the 240m cumulative row thus has 363 completed-by-exact-time versus 414 full completed positions. **Do not interpret the 51 as additional censored/unentered trades or a data loss**, and do not describe the h=240 panel as terminal outcome percentages. A later reporting revision could append a clearly labeled terminal/all-exits row, preserving the time-specific curve.

## 6. Interpretive limits, recommendation

- The eventual outcome is joined only for retrospective group labels, and `time_to_exit_min` and `reason_final` must **never** enter features for a deployed checkpoint predictor.
- Minute high and low reveal observed range in a fully completed minute, not within-minute barrier crossing sequence. All A03d stop/target labels derive from the earlier aggTrade exit replay; maker eligibility/queue is outside A04.
- Risk sets and overlapping trades are clustered; a full comparison needs symbol, day, side and target family stratification, with block uncertainty estimates.
- The July–September 2026 sample is repeatedly inspected; the above threshold groupings are hypotheses only.

**Recommended next study:** preregister one geometry-only 15m landmark conditional baseline using ex-ante target/risk distances and VD18 60/240m forecasts. Compare against a *minimal* augmented model adding signed completed-minute markout and normalized aligned aggressive quote-flow; use later strictly unexamined dates or a clearly marked exploratory blocked walk-forward test. Evaluate out-of-sample Brier/log score and calibrated competing outcome probabilities (TARGET vs STOP vs TIMEOUT), then actual incremental **after-cost P&L** of any continuation/hold/filter decision. Do not optimize indicator cutoffs on these A04 outcomes.

**Verdict:** historical signal *path anatomy* is successfully measured. Geometry is dominant in the descriptive groups; +15m markout plus taker-flow direction may carry additional information but is **not confirmed**, and no positive net EV has been established.
