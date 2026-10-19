# Liquidity Absorption 004 — LA-A02 first real-data results review

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-20 |
| **Branch** | Liquidity_Absorption |
| **Responds to** | `003_chatgpt_a02_causal_feature_contract.md`, `notebooks/Liquidity_Absorption/02/02_*.csv`, `02_feature_contract.json`, `02_decision.json` |
| **Status** | Review of user-executed descriptive features; **no P&L verdict, no code changes** |

## A02 completed; data and detector output

The committed `02_decision.json` reports `LA-A02-0.1.0`, `FEATURE_EXTRACTION_ONLY`; per symbol it loads 136,800 one-minute candles including 3 warmup days (2026-07-01 to 2026-09-30 event window). The price-only and enriched CSVs each contain **452 identical candidate rows**, comprising **212 BTCUSDT** and **240 ETHUSDT**. Every candidate reports `profile_status=PRESENT`; `missing_trade_day_events=0`. This means the feature processor had inputs for the candidate windows, not that genuine absorption was proved.

| Descriptive fact | BTCUSDT | ETHUSDT |
|---|---:|---:|
| Long / short candidates | 98 / 114 | 130 / 110 |
| Confirmed 5m pivot / previous-day references | 176 / 36 | 197 / 43 |
| Median first–second touch separation | 9 min | 6 min |
| Range of separation | 3–88 min | 3–90 min |
| Days with candidate events | 76 | 82 |
| Median last-5m near-level quote-volume share | 15.41% | 12.66% |
| Opposing **overall** 5m signed quote delta (% candidate events) | 62.3% | 55.0% |
| Opposing **near-level** signed quote delta (% candidate events) | 47.2% | 50.4% |
| Median recent / previous 5m near-level volume ratio, conditional on defined ratio | 0.521 | 0.643 |
| Previous-5m near-level quote volume = 0, so ratio undefined | 76 (35.85%) | 61 (25.42%) |
| Recent-5m near-level quote volume = 0, despite profile file present | 9 | 4 |
| Recent / previous 5m near-level ratio > 1, conditional on ratio defined | 48/136 (35.3%) | 66/179 (36.9%) |

The two CSVs show **452 distinct symbol/side/signal-time** combinations; all signal times align to one-minute UTC boundaries, and all level source timestamps precede signal time. The 5m quote volume calculated from candles and trade bins matches closely at candidate windows: median relative error approximately zero, p95 **0.00175% BTC** and **0.00237% ETH**; largest candidate error approximately 0.119% BTC and 0.165% ETH. These are additional descriptive consistency checks; they do not certify historical executable quotes.

## Interpretation: candidacy is not confirmed absorption

These events pass a *price-only* second-touch-with-departure pattern, with 2bp zone halfwidth, 35% wick-fraction floor, 3–90-minute retest separation and a 30-minute cooldown. The flow signals are attached **after** price-only selection. Only 221/452 (~48.9%) show aggressor quote delta opposite the rejection direction locally; 264/452 (~58.4%) show opposite aggregate 5m delta. Thus a wick does not mechanically entail the opposing aggressive flow described in the user's discretionary method. We have not tested whether conditioning on that flow improves outcomes.

The `profile_status=PRESENT` flag means *input trade minutes exist*, not that trades occurred in the target zone; **13** events have zero near-zone volume in the last five minutes. The growth ratio is undefined for **137/452 (30.3%)** because prior-5m near-zone volume equals zero. Undefined growth must remain an explicit category; replacing it with zero or infinity would mischaracterize the setup.

## Material measurement mismatch and missing feature contracts

**A02 does not yet measure first-touch vs second-touch accumulation.** Its rolling near-level comparison is [signal-10min, signal-5min) versus [signal-5min, signal). For **165/452 candidates (36.5%; 97 BTC, 68 ETH)** the two rejection close timestamps are separated by **10 or more minutes**: the first touch candle itself ends at or before signal-10min, so its trades lie outside the pre-signal 10m window. Other cases include variable amounts of intervening *time away from the level*, so even a defined 5m growth ratio is not necessarily an intensity change conditional on dwelling at the zone. This is a fidelity limitation, not evidence against the hypothesis.

**Proposed Stage A02b — still feature-only, subject to owner approval before change:** compute (a) first-touch and second-touch near-zone aggressively signed quote volume and volume-at-price per actual candle and/or fixed local encounter window, (b) zone dwell minutes and normalized volume accumulation *per minute at/near zone*, (c) two-touch delta divergence, delta slope and price response per normalized opposing flow, (d) explicit profile-undefined and zero-volume flags, and (e) causal rolling high-volume / low-volume price nodes as an optional third level family, with levels timestamped at their actual availability. This proposal requires a carefully bounded streaming aggregation to avoid whole-day tick accumulation in RAM and must avoid using future profile volume in level definition. The existing A02 event table remains an exploratory baseline; do not silently replace prior definitions.

**The user's structural stops and targets are not represented.** A03 execution semantics will still need owner-approved crypto-normalized stop distances, +10 MNQ-point analog (not a literal $10 crypto translation), next executable entry prices, causal one-minute swing confirmation and updates, opposing structural targets, bid/ask costs, and conservative actual trade sequencing. No full P&L or post-entry outcome was examined or altered here.

## Research decision

**A02 feature generation: SUCCESS. A02 faithful reconstruction of user's absorption observation: PARTIAL. Trading profitability, significance, and readiness for live orders: NOT EVALUATED.** Recommend improving the touch-specific measurement in A02b prior to A03 P&L to avoid testing the wrong feature. Historic July–September research is exploratory, not independent confirmation. Do not tune candidate thresholds based on downstream returns without preregistration.

## Questions for Claude

1. Is the 10-minute rolling window vs variable first-touch/separation gap a material measurement problem for testing real *replenishment/absorption*, and what causal encounter-window alternative is practical?
2. For volume profiles, how should we normalize by zone dwell time and instrument volatility while preventing price-level binning artifacts and retrospective level identification?

## Decisions needed from Ivan

1. Approve or reject the proposed **A02b touch-specific flow/dwell refinement** (without changing the existing production collector), before executing A03.
2. Decide whether the level families should remain only previous-day high/low and confirmed 5m pivots for the first execution test, or also include *causal historical volume-profile nodes* (closest to the user's discretionary setup).
