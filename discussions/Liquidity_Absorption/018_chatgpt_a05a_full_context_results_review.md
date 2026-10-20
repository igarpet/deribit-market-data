# Liquidity Absorption 018 — A05a full -240/+240m signal context results

| Field | Value |
|---|---|
| Author | ChatGPT |
| Date | 2026-10-20 |
| Inputs | Executed `05a_signal_context_event_study.ipynb`; `05a/{05a_decision.json,05a_coverage.csv,05a_context_checkpoints.csv,05a_phase_panel.csv,05a_phase_summary.csv,05a_cohort_metadata.csv}` |
| Version | `LA-A05a-0.1.0` |
| Status | **Historical descriptive event study only — no causal or independently validated predictive conclusion** |
| Modifications | This review only; no collector, simulator, or prior data modified |

## 1. Successful run and population

The recorded `05a_decision.json` reports **452 frozen A04/A03d candidate signals × 481 event-relative minutes = 217,412 minute rows**; exactly 2,712 event × six phase rows, 452/452 complete -240m windows and 452/452 complete +240m windows, **zero missing candle rows**. Outcomes: TARGET 119, STOP 202, TIMEOUT 93, NO_STRUCTURAL_TARGET 37, CENSORED_TAPE_END 1. Exactly 414 completed outcomes; the event-study summary charts condition on these TARGET/STOP/TIMEOUT labels, whereas the raw full panel also retains the 38 no-completed-outcome cases.

Signal **t=0 is the close of the second-touch 1m candle**, not the first post-signal aggTrade entry. The series' signed basis-point directions are aligned to the future trade side (long +, short -) and referenced to the t=0 close. After a trade's exit, the *market* path continues to +240m; this is NOT continuing strategy P&L and must never be used as a trader's actionable path.

## 2. Why the prehistory and the actual setup must be separated

Metadata confirm **373 pivot_5m levels** and **79 prior_day levels** (224 short signals, 228 long signals). All 79 prior-day references predate the -240m study window, while all 373 pivot level references are created **inside** it. The **median** pivot age at confirmation is **31 minutes** (10th–90th percentile about **19–61m**); **334/452** eventual level references are younger than 60 minutes and **373/452** are younger than 240 minutes. First touch is within the window for **all 452 signals**, a median **7m** before confirmation (median pivot 7m; prior-day 9m). Eventual level price is *retrospectively* projected back over the entire -240..+240 price graph; for 373 pivot signals, a price/level relationship before the reference's formation is NOT historical decision information.

Therefore the study contains three separate timestamps, which should be treated distinctly in future research: level formation/availability, first touch, second touch/signal. A clean process study should align on each of these boundaries instead of interpreting the entire past four hours as a four-hour liquidity defense.

## 3. Signed median price paths (bp from t=0 close)

| Event-relative minute | TARGET (n=119) | STOP (n=202) | TIMEOUT (n=93) |
|---:|---:|---:|---:|
| -240 | +24.67 | +20.82 | +17.19 |
| -60 | +11.67 | +11.76 | +13.05 |
| -30 | +9.87 | +9.22 | +9.60 |
| -15 | +5.07 | +6.55 | +3.79 |
| -5 | +1.93 | +1.40 | +2.39 |
| 0 | 0 | 0 | 0 |
| +5 | +1.92 | -0.41 | +1.19 |
| +15 | +8.77 | -4.13 | +3.68 |
| +60 | +19.23 | -17.39 | +12.54 |
| +120 | +36.08 | -27.68 | +17.35 |
| +240 | +54.11 | -32.59 | +28.04 |

For all three outcome groups, the center of the retrospective path tends to drift *against* the intended trade direction from earlier positive signed prices into the t=0 level/rejection. **This common approach is strongly conditioned on the A02 two-touch-rejection selection rule**, not independent evidence of a discovered mechanism. At +15m targets diverge positively and stops negatively, but a key share have already exited by then, and post-exit marks are included in A05a medians. Do not read the +240 row as trade returns, live risk sets, or historical 240-minute trading positions.

## 4. Fixed-phase medians: early context, formation and reaction

Medians of each **event's** phase price change are not identical to difference of cross-event median path endpoints. Values below come from the individual event phase-panel records, not subtracting curve medians.

| Phase (minutes) | TARGET median signed price change (bp) | STOP | TIMEOUT |
|---|---:|---:|---:|
| context (-240..-61) | -5.20 | -9.13 | -5.58 |
| approach (-60..-16) | -2.40 | -6.25 | -7.25 |
| formation (-15..0) | -5.07 | -6.55 | -3.79 |
| immediate (+1..+15) | +8.72 | -3.95 | +4.83 |
| early (+16..+60) | +15.03 | -11.11 | +8.55 |

Formation-phase **aligned aggressive quote-volume fractions** are approximately -0.069 TARGET, -0.060 STOP, -0.061 TIMEOUT. Pre-signal 15m realized volatilities are nearly equal around **16.2 bp** in this combined grouping. The negative aligned flow and adverse drift **do not separate TARGET and STOP by much before the signal**. At +1..+15m, aligned fractions become +0.067 TARGET, -0.042 STOP, +0.023 TIMEOUT. This supports an *exploratory* reaction/continuation contrast, but is not a causal absorption proof; aligned taker quote volume is not passive liquidity replenishment.

Broad pre-signal price paths and pre-signal vol/activity trends do **not** present a reliably discriminating common winner signature in the displayed medians. BTC and ETH should not be assumed interchangeable; several phase outcomes differ in magnitude.

## 5. Existing target-feasibility confounding continues

Original A03d entry-target distance divided by frozen VD18 forecast **60m terminal-return** volatility has median:
- TARGET (119): **0.917 sigma**, 62/119 at or below one sigma;
- STOP (202): **2.152 sigma**, 30/202 at or below one sigma;
- TIMEOUT (93): **3.954 sigma**, 7/93 at or below one sigma.

This ex-ante target-geometry difference is much larger than broad pre-signal flow contrasts. In the far-target (ratio >1) subset, median -60..-16 phase price change is **-7.33bp for TARGET (n=57)** versus **-6.68bp for STOP (n=172)**. In the nearer-target subset, -15..0 phase medians are **-7.82bp for TARGET (n=62)** versus **-7.99bp for STOP (n=30)**. These comparisons are not carefully matched on exact target distance, symbol, level family or session, but illustrate how generic pre-signal price patterns can overlap.

A terminal 60m spot sigma is NOT the price-barrier first-passage scale for a 240-minute strategy. Geometry still needs 240m/hazard calibration.

## 6. Technical and inferential issues

- All 481 relative prices are aligned to the **eventual signal**. Retrospective labels/levels and pre-level source periods are visualizations, not contemporaneously executable features. The 79 prior-day levels and 373 future-created pivot levels must be handled separately.
- A05a keeps **post-exit prices**, unlike A04's still-active landmark data. At +240m, 119 TARGET and 202 STOP outcomes have already exited; all 93 TIMEOUT positions may still be active at that exact signal-clock timestamp because timed-out trades execute on the next print slightly after the 240m threshold. A05a market curve is informative for post-rejection market behavior but not risk-set prediction or account equity.
- Minute OHLC / taker-buy quote are *flow and price movement proxies*, not L2/queue-identified passive absorption or per-print crossing order.
- The whole sample (July–September 2026) has been repeatedly inspected, including price/flow and geometry. Outcome-conditioned shapes are hypotheses, not historical validation; no new trading rules, maker fills or positive net trading edge are demonstrated.
- A05a tests completeness of 1m price data, but does not validate independent causal *level availability* for all negative relative times. Anchoring the next study at the actual level-source timestamp and first touch would address this issue.

## 7. Recommended next research question

Rather than instantly fitting a complex flow classifier, first decompose **the actual formation process** into: (1) market context before level formation, (2) level source to first touch, (3) first to second touch, and (4) immediate reaction *at surviving risk-set landmarks*. Explicitly prevent leakage when the eventual level did not yet exist. Compare original A02 candidates to a timestamp-compatible non-signal control universe where possible (to distinguish signal-generation mechanics from baseline market behavior), and normalize target reachability. Then validate any proposed pre-entry quality feature on genuinely uninspected later days.

**Verdict:** A05a worked technically and documented a broadly common adverse-direction approach into signal confirmation. The stronger TARGET/STOP path differences appear after the signal and are heavily entangled with outcome-selection mechanics and target feasibility. The economically interesting next step is level-formation/first-touch-aligned research, not in-sample cutoffs on the -240..0 means.
