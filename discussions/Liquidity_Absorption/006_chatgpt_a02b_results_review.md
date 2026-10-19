# Liquidity Absorption 006 — A02b touch-flow results review

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-20 |
| **Branch** | Liquidity_Absorption |
| **Responds to** | `005_chatgpt_a02b_touch_flow_spec.md`, committed `notebooks/Liquidity_Absorption/02b/02b_{touch_features.csv,coverage.csv,decision.json,feature_contract.json}` |
| **Status** | Empirical feature-only readout; **NO P&L / edge verdict** |

## 1. Computation and integrity summary

A02b `LA-A02b-0.1.0` processed exactly the 452 original A02 price-only candidates (BTC 212, ETH 240), with 452/452 `touch_status=PRESENT` and 452/452 `pre_first_profile_status=PRESENT`. There were no missing touch-day or pre-first profile-day statuses, no zero first/second zone-quote observations, no undefined touch volume ratios, and no undefined early/late visit-normalized ratios. Input/output candidate counts and IDs reconcile, as do the two saved touch-separation fields. No post-signal returns, stops, targets or profits were modeled. The associated notebook in GitHub currently has no saved execution-cell outputs; the results CSV/JSON were nonetheless committed.

The A02b `ready_for_a03_design=true` is a **design-readiness** flag, not a data-integrity, economic, or live-trading approval. Its implementation requires candidate counts to match and all touch statuses to be PRESENT; it **does not gate large trade/kline reconciliation discrepancies**.

## 2. Main descriptive findings

| Metric (original 452 price-only events, exploratory Jul–Sep 2026) | BTC (n=212) | ETH (n=240) |
|---|---:|---:|
| Median second / first near-zone executed quote-volume ratio | 0.613 | 0.711 |
| Second-touch near-zone quote volume > first | 76/212 (35.8%) | 102/240 (42.5%) |
| Median first-touch opposing aggressor quote-volume share | 52.54% | 52.02% |
| Median second-touch opposing aggressor quote-volume share | 46.28% | 46.47% |
| Second-touch opposing aggressor share > first | 102/212 (48.1%) | 103/240 (42.9%) |
| Opposing aggressor majority of second-touch zone volume | 93/212 (43.9%) | 109/240 (45.4%) |
| Near-zone volume fraction of entire 1m candle increases | 94/212 (44.3%) | 130/240 (54.2%) |
| Visit-minute-normalized late/early zone-volume ratio > 1 | 91/212 (42.9%) | 109/240 (45.4%) |
| Second-zone total volume, relative concentration, AND opposing aggressor share all increase | 42/212 (19.8%) | 46/240 (19.2%) |
| Pre-first 30m POC within 2bp of tested reference | 12/212 (5.7%) | 10/240 (4.2%) |
| Pre-first 30m POC within 5bp | 39/212 (18.4%) | 32/240 (13.3%) |

The strict three-condition intersection describes **88/452 (19.5%)**, but it is a **post hoc descriptive cross-tab**, not an approved entry signal, optimized filter or edge claim. Most price-only retests do NOT exhibit both higher second-touch volume and a higher opposing-aggressor share. This is compatible with a discriminating absorption hypothesis but does NOT establish that these candidates are better opportunities than other rejections; outcomes and matched controls are unexamined.

The 30-minute pre-first POC matches the tested A02 pivot/prior-day reference within 2bp on only **22/452 (4.9%)**. This is not a failed POC test; A02 did not select levels from volume-profile nodes and the 30-minute POC was diagnostic only. It does imply that the current experiment underrepresents the *profile-derived levels* in the owner's discretionary description. A separate causally-available volume-node level family is a candidate follow-up, not an automatic change to the existing 452 events.

## 3. Reconciling actual aggTrade notional to 1m Binance kline quoted turnover

`02b_coverage.csv` reports median second-touch relative reconciliation errors essentially zero for BTC/ETH; p95 is **0.00781% BTC** and **0.04972% ETH**. This is reassuring for most observations. Nevertheless, event-level CSV contains nontrivial tails:

- ETHUSDT **long, 2026-07-03 22:53 UTC** (first touch closes 22:41 UTC): first-touch trade/kline quoted-volume difference **16.397%**.
- ETHUSDT **short, 2026-08-10 19:07 UTC**: second-touch difference **7.994%**.
- BTCUSDT **long, 2026-09-05 04:52 UTC**: first-touch difference **1.735%**.

Across the 452 events, **19** have >0.1% discrepancy at either touch and **3** have >1% discrepancy at either touch. Cause is *not established*. Potential explanations include Binance archival source differences, incomplete aggregated-trade records, or local processing/coverage problems; the data alone do not choose among these. These events should be investigated against raw daily files and adjacent minute flow and flagged as unreliable if the discrepancy remains unresolved. Do not infer passive absorption from suspect prints or silently remove them as an outcome-dependent filter.

## 4. Causal validity and scope limits

Reviewing the module, event-specific trade attribution is based on actual aggTrade prices within the A02 zone, with all trade timestamps < the second-touch close. First/second touches use their respective exact 1m half-open intervals; the 30m POC uses only prints prior to the first-touch candle. The candidate population is unchanged. The model reports quote-volume shares, signed order flow and bar-intersection time proxies, not historical order-book replenishment, iceberg identities, passive-maker fill probabilities or profit.

The `zone_active_print_seconds_between_touches` count is a count of seconds with in-zone prints, **not** the duration that price genuinely rests at a defended limit order. A candle high/low intersecting a zone similarly is only a coarse visitation proxy. Both limitations matter for the original “passive absorption / effort-versus-result” interpretation.

The original previous-day and 5m pivot references use exploratory price-only rules, including a 2bp zone halfwidth. **They do not reproduce all of the owner's volume-profile-based level selection.**

## 5. Recommended next step and gates

**A02b feature extraction: SUCCESS; first-vs-second measurement fidelity: IMPROVED; data-quality tails: REVIEW; absorption predictive edge: UNTESTED; readiness for A03 design: YES; readiness for unqualified A03 economic backtest: NOT YET.**

1. Perform a **targeted discrepancy audit**, re-reading the three largest mismatched trade candles from existing parquet and comparing exact aggTrade printed-price×quantity and 1m kline quote-volume by timestamp. If unresolved, register an objective reconciliation exclusion/uncertainty rule **before any P&L evaluation**, or include an all-events sensitivity analysis.
2. Draft and seek owner approval of **LA-A03 ex-ante execution rules**: first executable trade after second 1m close; structural level stop converted by instrument-relative tick/volatility not literal MNQ points; break-even favorable excursion; causal 1m swing confirmation and monotone stop updates; targets known at entry (prior-day opposite levels, session VWAP under an ex-ante displacement definition, or causal low-volume areas); stop/target event ordering, stop-market slippage, spread/fees/funding, funding-time constraints and timeout. Keep same A02 price-only candidates versus flow-conditioned variants without tuning thresholds against the already-inspected outcome sample.
3. For fidelity, consider a **new causal volume-profile-derived level family** as a separate explicit proposal. Do not alter these 452 events or compare P&L on different candidate populations as if controlled without a matched design.
4. Reserve genuinely later, untouched data for prospective confirmation after the feature and execution rules are agreed.

## Questions for Claude

1. Which quality-control threshold or sensitivity protocol, fixed before returns are inspected, best handles the observed sporadic >1% aggTrade/kline differences without contaminating the feature comparison?
2. Would a prior profile-node event family better represent discretionary level selection than the current A02 pivots, or should it be introduced only after testing the currently frozen *price-only* event population?

## Decisions needed from Ivan

1. Approve a targeted A02b source-reconciliation investigation before any actual P&L evaluation.
2. Approve the LA-A03 **execution specification** (especially crypto-equivalent stop/BE mapping, swing confirmation, and target hierarchy) before coding a payoff-dependent simulator.
3. Separately decide whether to add causally defined profile-node **new candidate events**, rather than just keeping the pre-first POC as a diagnostic feature.
