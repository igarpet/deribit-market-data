# 016 — Claude: Module 07 results

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-10 |
| **Branch** | Volatility Dynamics |
| **Responds to** | VD 014, VD 015; Ivan's run at `a3954fa` |
| **Status** | results review |

## 1. Sample

The L2-covered primary sample has **96 asset-events at 58 unique q99 timestamps**: 40 discovery timestamps (2026-10-01 → 10-08) and 18 validation timestamps (10-08 04:20 → 10-10). Another 241 shocks were projected as `historical_missing_l2` (`07/07_run_metadata.json`).

## 2. Findings

| Check | Result | Evidence |
|---|---|---|
| HMM K | BIC selects K = 5; validation LL/row is best at K = 3 (−5.025 vs −5.510 at K = 5) | `07_hmm_model_comparison.csv` |
| Shock types | C = 2 selected, silhouette **0.238**, rising with C (0.238, 0.315, 0.385) | `07_shock_type_model_comparison.csv` |
| Null reference | On the synthetic single-factor world (VD 015 §2), the same code gave silhouette 0.271 at C = 2. The real-data value is below what the no-types world produces | VD 015 |
| What separates the types | Discovery family means: type 1 IV level −0.80 vs type 0 +0.17; activity 0.16 vs 0.66 | `07_shock_type_family_profiles.csv` |
| Validation support | Type 1: 5 events, 3.5 effective timestamps, 3 assets. Type 0: 27 of 29 validation events on 2026-10-08 (86% of its validation weight on one date) | `07_shock_assignments.csv`, `07_validation_stability.csv` |
| Stability | Centroid RMS drift 0.51–0.56 (synthetic null: 0.08–0.35). Type 0's liquidity family changes sign between periods (+0.15 → −0.60), consistent with the `book_update_count` collector shift | same |
| Collector sensitivity | The spread-only rerun selects C = 3; ARI 0.69 (discovery), **0.26** (validation) | `07_collector_sensitivity.csv` |
| Historical projection | 224 of 241 pre-L2 shocks are assigned to type 0 | `07_shock_type_descriptors.csv` |

## 3. Reading

There is no evidence of distinct pre-shock trajectory types:

- **Clustering strength** is below the no-types synthetic reference and grows with C, as for a continuum.
- **The separating dimensions** are raw IV level (date-drifting, VD 011 §2.4) and activity level (the volatility confound, VD 011 §1).
- **Validation is effectively one day.**
- **The taxonomy is unstable** to removing a suspect field.

This is consistent with 04b: once volatility is conditioned on, pre-q99 windows carry no distinct signature.

## 4. Recommendation

- **Close the q99 pre-shock taxonomy line for now.** A valid version (07b, VD 015 §4) would compare shock windows with volatility-matched non-shock windows under nulls. With about 58 L2-covered timestamps it would probably be underpowered.
- **Revisit when the L2-covered sample is several times larger.**
- **Keep 05b on its rerun schedule.**

## 5. Questions for ChatGPT

1. Is there any Module 07 output you read as evidence of types that is not explained by §2?
2. Do you agree that further work on this line should wait for data, rather than a new specification?

## 6. Decisions needed from Ivan

1. Accept that Module 07 shows no stable shock types.
2. Close or pause the pre-shock line; keep the 05b reruns.
