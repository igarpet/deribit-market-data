# Liquidity Absorption 031 — Module 09 observable-level lifecycle results review

**Date:** 2026-10-21  
**Notebook:** `notebooks/Liquidity_Absorption/09_level_lifecycle_anatomy.ipynb`  
**Inputs:** Original A02b 452 two-touch candidate IDs, A03b FIXED 1:1 eventual outcomes, A03c maker-touch *eligibility-only* annotation and local Binance UM 1m OHLC + taker-buy quote.  
**Status:** Descriptive historic analysis; no causal mechanism established and no out-of-sample alpha claim.  
**Scope of change:** This review document only; no research code, outputs, collector files or strategy rules edited.

## 1. Completion and timing contract

`09_decision.json` confirms `LA-LIFECYCLE-09-0.1.0`: 452 candidates, **3,616 original-ID × 8 phase rows**, **83,168** minute milestone rows (452 × 4 milestone anchors × 46 minutes), **zero missing landmark minute bars**, and **16 empty lifecycle stages**, each the zero-duration `AVAILABLE_TO_FIRST` interval for a touch candle that closes at the same instant as pivot confirmation (not corrupt OHLC data). These 16 are **not strictly actionable first-touch observations**, since the intraminute touch might precede availability. All events satisfy availability <= first-touch *close*, but 16 fail the stricter first-touch close > availability relation.

The two level families are **373 pivot_5m** and **79 prior_day**. Pivots are recognized only **10 minutes after their source 5m candle close** (two subsequent 5m bars); prior-day levels become available at the *next UTC midnight*. Median times: 16m available→first touch across all 452; 7m first→second touch. Within pivot_5m, availability→first touch median is 12m; within prior_day it is ~810m. First-to-second medians: pivot 7m, prior-day 9m.

The stage taxonomy is `PRE_SOURCE_60`, `SOURCE_TO_AVAILABLE`, `AVAILABLE_TO_FIRST`, `FIRST_TO_SECOND`, `AFTER_SIGNAL_1_5`, `AFTER_SIGNAL_6_15`, `AFTER_SIGNAL_16_60`, and `AFTER_SIGNAL_61_240`. All phase windows use completed 1m candles of form (start,end]. Retroactively identified level/zone statistics are **NaN before actual availability**.

## 2. Important cohort distinction

The eventual outcomes here belong to **A03b fixed 1:1**, NOT the volatility-risk-protected A03d policy used in A04/A05a. `09_decision.json` has **116 TARGET, 112 STOP, 185 TIMEOUT, 37 NO_STRUCTURAL_TARGET, 2 CENSORED_TAPE_END**. Therefore differences in TARGET/STOP/TIMEOUT rates or grouped shapes compared with A05a are not an automatic strategy improvement or dataset inconsistency: the simulated stop/target rule has changed. A03c maker eligibility is joined solely as an annotation; actual fills and queue priority are not observed.

## 3. Outcome-conditioned median phase price changes

Numbers below are medians of **per-event** direction-adjusted phase price changes (basis points), recomputed from `09_lifecycle_stages.csv`, excluding zero/missing phases; they are not differences of cohort-median price curves.

| Stage | TARGET | STOP | TIMEOUT |
|---|---:|---:|---:|
| PRE_SOURCE_60 | -8.49 | -14.81 | -5.21 |
| SOURCE_TO_AVAILABLE | +2.67 | +3.37 | +2.29 |
| AVAILABLE_TO_FIRST | -12.32 | -13.81 | -10.01 |
| FIRST_TO_SECOND | -0.01 | -0.27 | -0.43 |
| AFTER_SIGNAL_1_5 | +2.41 | -1.76 | +1.23 |
| AFTER_SIGNAL_6_15 | +3.88 | -3.93 | +0.36 |
| AFTER_SIGNAL_16_60 | +17.09 | -16.53 | +1.11 |
| AFTER_SIGNAL_61_240 | +26.00 | -13.13 | +1.75 |

Because previous A03b trades may already have hit stops/targets, every full after-signal phase is a **market** trace, not holding-period P&L, active-risk-set evidence or an executable continuation rule. `SOURCE_TO_AVAILABLE` pools fundamentally different phase lengths across level families; analyze pivot and prior-day separately.

## 4. Pre-signal market microstructure hypotheses

During `AVAILABLE_TO_FIRST`, the three outcomes look similar in adverse drift and opposing aggression: median signed move TARGET -12.32bp, STOP -13.81bp, TIMEOUT -10.01bp. During `FIRST_TO_SECOND`, all median signed moves are near zero, reflecting the retest-based selection.

The more interesting *candidate* difference appears inside `FIRST_TO_SECOND`:

| Outcome | n | Median zone-overlap share | Median direction-aligned aggressive quote imbalance | Median stage length |
|---|---:|---:|---:|---:|
| TARGET | 116 | 0.423 | -0.028 | 7m |
| STOP | 112 | 0.298 | +0.025 | 7m |
| TIMEOUT | 185 | 0.333 | +0.020 | 8m |

This pattern is **consistent with** the hypothesis that eventual targets exhibit more repeat candle interaction near the selected zone despite net aggressive executions opposing the eventual trade direction, i.e. possible price resistance to aggression. It does **not** demonstrate passive absorption, order-book replenishment or a predictive filter. Zone overlap is based on 1m high/low intersection; it cannot indicate time spent in-zone or within-minute order. A02b's signal selection itself favors zone interactions. Across pivot-only cases, TARGET median FIRST_TO_SECOND zone-overlap ~0.433 vs STOP ~0.333, with imbalance -0.055 vs ~0.000; prior-day target n=20 and stop n=29 are too small for confident comparison.

Post confirmation, already by +6..+15 minutes, TARGET median direction-adjusted move +3.88bp vs STOP -3.93bp, and aligned quote imbalance +0.055 vs -0.059. By +16..+60m, the median move is +17.09 vs -16.53bp. These retrospective differences also depend on the eventual target/stop labels and must not be interpreted as positive trading expectancy.

## 5. Key remaining methodological issues

1. **Touch/action timing:** 16 same-close first touches may be pre-confirmation intraminute; audit or exclude before any causal simulation or pre-entry feature inference. The exact touching trades are not reconstructed from Binance 1m OHLC.
2. **Signal-time hindsight:** `FIRST_TO_SECOND` includes the completed second-touch signal bar (available *only at the end of the phase*). A model using its full-stage statistics may act at t=0 after confirmation, but not during the phase. The milestone chart's `signed_price_from_anchor_bp` for negative relative offsets is normalized to a **future** anchor close; this is acceptable as retrospective visualization only, not an ex-ante negative-time feature.
3. **Selected-only cohort:** every included setup was selected *because* its second touch occurred. No denominator for all available levels, levels with zero/one touch, failed/invalidated levels or signals avoided by a rule. Thus no `P(second touch | level available)` or genuine signal-discovery hazard can be inferred.
4. **Target feasibility:** fixed 1:1 A03b outcomes are heavily confounded by different structural target distances/risk; this module does not match by target-to-sigma, exact stop distance, asset, side, session or level family. Medians alone cannot certify independent explanatory value or statistical significance.
5. **Comparability:** never pool 10-minute pivot confirmation intervals with 1440-minute prior-day reference accumulation as though they are a common microstructure state.
6. **Economic boundary:** no refreshed taker/maker execution, funding, stop-fill, queued maker outcome or after-cost net EV is established by these charts. Historical July–September 2026 has already been inspected; forward independent dates are required.
7. **Code audit:** implementation and 20 synthetic regression tests reviewed at source. The executed notebook's entire raw JSON could not be retrieved via the GitHub file connector due to its size; the result artifacts and recorded decision were inspectable and internally consistent.

## 6. Decision and recommended next step

**Verdict:** module 09 achieves its intended *observable level lifecycle decomposition* and successfully avoids attributing the selected level to periods before it existed. The promising next diagnostic is level-centered, rather than signal-only: construct a complete universe of available pivot/prior-day levels, retain **untouched, first-touched, invalidated, and twice-touched** levels, and measure time-to-event and encounter hazard from availability. Then compare the existing two-touch signals to timestamp/asset/level-family/volatility-matched non-signal levels, separately stratify by frozen target geometry, and estimate block-bootstrap uncertainty. This is a research protocol, not a new strategy specification or automated trading decision.
