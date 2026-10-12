# VD 025 — Module 11 first execution: missing-history eligibility bug and repair

**Date:** 2026-10-12. **Affected version:** VD11 v0.1.0. **Corrected:** v0.1.1.

## Source evidence

The original notebook execution is preserved in Git commit `16487be68782769958f5dc87bc9706590b4beaad`. It read 42,336 seven-asset 5-minute states from the research DB. After the common feature support and a **288 consecutive finite 5m-bar** gate:

| Split | Rows | Eligible |
|---|---:|---:|
| Training | 25,536 | 2,216 |
| Validation | 8,064 | 0 |
| Test | 8,736 | 0 |

The model-selection cell raised:
`RuntimeError: A_current validation failed: Insufficient complete common observations`.

This outcome is a **pipeline eligibility bug**; it provides **no empirical performance evidence** for or against lifted memory.

## Repair design (v0.1.1)

1. Keep the 288-bar **elapsed** history lookback, strictly before origin t; require at least 80% *observed* history and a finite return in the previous 15m. Require the current bar to be observed.
2. Use strictly causal observed-weight-normalized exponential filters: decay numerator and observed weight every 5m; add only finite previous-bar observations, and normalize by nonzero observed weight. A missing bar is neither a zero return nor an interpolated return.
3. Keep the original train-only variance epsilon, hourly de-clocking, q99 thresholds, chronological splits, exact-future label censoring, matched-sample comparisons, pre-registered horizons, and decision gates unchanged.
4. Surface history coverage/readiness per split in the notebook before fitting. Do **not** discard the historical failed run silently: the previous commit contains its traceback.
5. Clear stale execution outputs from the revised notebook before rerunning. All saved real-data forecasts/results remain pending.

**Verification:** seven synthetic regression tests including once-daily missing native return observations; an additional full local synthetic validation-grid and 30/60/90/120m scoring trial with daily + randomly missing returns succeeded. The Deribit `research.duckdb` was unavailable to the repair environment.

**Interpretation constraint:** This is a material improvement to *data availability handling*, not evidence that roughness is identified or that q99 prediction improves. Memory factors now represent observed-data-normalized exponential summaries under missingness; they are not exact latent lifted-Heston factors. The fixed 80% policy should be regarded as a preregistered new availability definition from v0.1.1 onward. Original test period has only five UTC days and will still fail the minimum 10-day evidence requirement. No option-trading signal or strategy has been authorized.
