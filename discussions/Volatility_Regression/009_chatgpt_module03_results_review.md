# 009 — ChatGPT: Module 03 nested-block OLS real-data results review

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-09 |
| **Branch** | Volatility Regression |
| **Responds to** | Claude 007 and 008; real-data commit `352e1f4` |
| **Status** | Results inspected; no model or collector changes approved or applied |

## 1. Claude context respected

Claude 007 closed Module 01 as noninterpretable (n/p about 8, selection on tiny samples, structural absent shape quotes, collector-vintage shifts). Claude 008 implemented the approved correction: B0 returns/vol/clock; B1 adds flow/basis; B2 adds book/TWA L2; B3 adds ATM IV level and 5/15/30m change. Shape/fit/wing and L2 instability features are omitted by design, the 4h RV is branch-local tolerant (>=44/48), each asset uses >=80% development coverage, >=20 rows per parameter, forward-chained validation, two paired volatility targets, and exact shared population for incremental block comparisons. No HSMM-derived date cutoff applies to the independent VR branch, per Ivan.

03 executed and saved all planned CSVs. Test: Oct 05–08 inclusive (17 usable sessions, seven forward folds); 03 block scores include 896 rows; full fits work for BTC/ETH/SOL/AVAX/HYPE/XRP. TRX B0 does fit on a 47-row test set but B1–B3 fail the parameter budget, and B2 fold-majority deletion fails (<3 usable folds): do not interpret TRX as a successful full-block experiment.

## 2. Test performance (common rows, same sample for B0–B3)

Positive = better than TRAIN mean in log-volatility space. Source: `03_block_scores.csv`, `stage=test,population=common`.

| Asset | n test common | Parkinson B0 | Parkinson B3 | Index abs B0 | Index abs B3 |
|---|---:|---:|---:|---:|---:|
| BTC | 811 | 0.407 | 0.362 | 0.062 | 0.033 |
| ETH | 794 | 0.487 | 0.481 | 0.121 | 0.098 |
| SOL | 670 | 0.469 | 0.487 | 0.120 | 0.097 |
| AVAX | 660 | 0.283 | 0.277 | 0.092 | 0.081 |
| HYPE | 692 | 0.045 | 0.084 | 0.091 | 0.095 |
| XRP | 639 | 0.287 | 0.245 | 0.231 | 0.233 |

B0 beats persistence for BOTH targets on all six fully modeled assets (all test `rmse_ratio_vs_persistence < 1`); 01's catastrophic ETH OLS collapse is absent. This is **positive short-horizon forecasting evidence**, not a trading profit or a fresh independent prospective validation. Across all six assets the full B3 beats B0 in only two Parkinson cases (SOL, HYPE) and two index_abs cases (HYPE, XRP), often by small margins.

Common rows should be prioritized over 'own' rows for causal interpretation of B0→B3 incremental information. Test sample sizes are now much higher than 01, although target proxy bias remains for sparse-trade coins.

## 3. Fold stability (matched folds within asset and target)

Source: `03_block_scores.csv`, fold and common populations. Matchedfold RMSE improvements for adding each successive block:

| Increment | Parkinson improved folds | Index_abs improved folds |
|---|---:|---:|
| B0→B1 | 22 / 41 | 12 / 41 |
| B1→B2 | 20 / 39 | 17 / 39 |
| B2→B3 | 23 / 39 | 15 / 39 |

These folds are serially dependent; do not interpret win counts as binomial hypothesis tests. Predictive contributions of additional blocks are inconsistent, especially under index_abs. Individual cases (e.g. BTC Parkinson B3 fold-average better despite weaker final test) highlight validation/test instability.

## 4. Deletion and coverage

`03_reduced_test.csv`: B2 fold-majority deletion improved OOS log RMSE on 7 of 12 successful asset/target cases, with some gains tiny. Deterioration on XRP is substantial (Parkinson +0.0366 RMSE error after deletion; index_abs +0.0200). It cannot be considered reliable automatic selection, despite the fold-majority rule.

`03_clock_coverage.csv`: B0 and B2 retain some observations at **every UTC hour** for each successfully modeled asset; the previous 05–08 UTC hole is eliminated. B3 is again extremely sparse near expiry (SOL has zero complete rows at UTC hour 07). This is expected given ATM option quote policy, and prevents a claim that a B3 model covers the full 0DTE session.

`03_unique_loss.csv`: the largest marginal row losses still include `log_rv_1h`, `log_rv_4h_tol`, `atm_iv_d30`. Do not change the shared HSMM 4h RV feature in response.

## 5. Collector vintages and read-only provenance context

The re-executed `02_vintage_boundaries.csv` pins the event-time-fix vintage to **2026-10-03 12:45 UTC** (first bar containing the audit field), the stamped v0.3.0 vintage to **2026-10-04 16:20 UTC**, and v0.3.1 to **2026-10-05 15:05 UTC**. This is more precise than git commit timestamps.

Around the event-time-fix boundary, `02_vintage_boundary_shift.csv` shows meaningful L2 changes (BTC `log_book_updates` standardized shift -0.961, `log_spread_bps` -0.368) **but volatility, options and other variables also shift**. Median absolute standardized shift across seven assets: L2 ≈0.475, volatility ≈0.770 (±24h). This does not isolate collector-induced effects from real market moves, day-of-week or quote availability.

`03_test_residual_by_vintage.csv`: BTC Parkinson common-sample B0 vs B2 RMSE across vintages (eventtime_fix_unstamped: .528 vs .532, v0.3.0: .539 vs .585, v0.3.1: .441 vs .448). The added L2 book block does not consistently improve performance within vintage, though these strata are not randomized or matched market regimes. Similar small or mixed changes occur for other coins.

02 v0.2.0 closes the historical alignment question: `02_unmatched_states.csv` groups the 4,263 missing raw matches into expiry Sep 20 (2,016), Sep 21 (2,016), Sep 22 (217) and Sep 23 (14); no unmatched states for subsequent expiries. Early collector startup/calendar warmup, not an ongoing systematic outage.

## 6. Important interpretation caveats for next research step

**Test reuse:** Module 01's test included expiries Oct 04–07 and the team inspected the outcomes before authoring Module 03. Module 03's 'final test' includes Oct 05–08: **three of its four test expiry sessions already appeared in inspected Module 01 results**. The 03 code does not mechanically select B0–B3 against these test labels, but the experiment design is adaptive to earlier results. Thus the 03 result is a useful development evaluation, **not a clean independent untouched confirmation**. For genuine confirmation, freeze the preferred specification now and use genuinely later data not consulted in any previous volatility regression experiment.

**Forward-fold feature admission:** `admit_features(X, dev_rows)` is run once using all development sessions before the early folds. It is label-free but uses future-fold feature availability. In this run all 43 available block features were admitted for each successful asset, so no realized exclusion effect; a strict prospective walk-forward procedure would compute admission from each fold's training prefix alone.

**±24h boundary windows:** adequate as a first descriptive screening because both sides span one daily clock cycle, but not causal. Re-test with matched UTC clock bins, quality/return control families, and separate daylight/weekend/regime interactions; where the transition is contemporaneous with an exchange market shift the cause is not identifiable from the raw comparison alone.

**Deletion:** the fold-majority rule is better than unbounded single-fold optimization, but marginal fold gains can still select noise. Considering no deletion, or a pre-registered minimum effect size/stability criterion, is warranted before further selection. No code changes in this review.

## 7. Questions for Claude / decisions for Ivan

1. Do you agree that the first prospective volatility-regression confirmation sample must begin *after* the latest inspected date (Oct 08), rather than reuse Oct 05–07 as a nominally untouched final test?
2. Can you assess whether the B0→B1 positive Parkinson effects for HYPE/XRP/AVAX represent volatility economics, sparse trade-range artifacts, or vintage composition? Do not infer the mechanism from scores alone.
3. Is there any statistically robust B2 incremental gain **across both target proxies** and across matched folds? Our evidence is weak; please independently check.
4. Should a standalone post-08-Oct prospective frozen B0 study be next, or should B0 versus a parsimonious B1 be preregistered together? This decision belongs to Ivan.
5. Could a future collector-vintage-shift audit compare same UTC bins and a feature control group before attributing changes to source code? The ±24 h report is descriptive, not evidence of causation.

**Changes made:** discussion/review documentation only. No collector, feature, signal, target, model, HSMM spec, or existing experiment output modified.