# 011 — Claude: pre-registered confirmation (04/05) and mid-path pass-through

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-09 (committed ≈ 00:30 UTC, before any 2026-10-09 expiry outcome is complete or inspected) |
| **Branch** | Volatility Regression |
| **Responds to** | 010; Ivan's approval of 010 §7 items 1 and 2 (2026-10-09) |
| **Status** | Protocol fixed in code (`vol_ols_confirm.py` v1.0.0). Tests 6/6. 04/05 executed on synthetic data only. Item 3 of 010 §7 (collector index gaps) not approved; untouched. |

## 1. Pre-registration (binding)

Fixed in `src/data_engineering/modeling/vol_ols_confirm.py`. Notebook 05 refuses to run if the constants differ from those serialised in `04_frozen_spec.json`.

| Item | Value |
|---|---|
| Question | Does ATM implied-vol information improve next-5m log-volatility forecasts beyond past volatility and clock? |
| Models | **B0** = 03's B0 (returns, 30m/1h/tolerant-4h RV, current range, clock, TTE). **B0_ATM** = B0 + `log_atm_iv`, `atm_iv_d5/d15/d30`. No flow, no L2 (no `log_book_updates`), no deletion. |
| Training | Expiries ≤ **2026-10-08**. Fitted **once**; coefficients, scaling, ε and train mean frozen in `04_frozen_spec.json`. Feature admission (≥ 80% coverage) on training rows only. Budget n ≥ 20p. |
| Confirmation | Expiries ≥ **2026-10-09**, never inspected. No refitting. |
| Primary | Target `parkinson`; assets **BTC, ETH, SOL** (0% zero-range bars). |
| Secondary (never decisive) | `index_abs` (sign check), `mid_rv` (10-second mid-path RV; trained on ≈ 4 sessions), all other assets. |
| Rows | Paired-valid rows complete for both models. A session counts if it has ≥ 100 such rows. |
| Asset rule | **Pass** if B0_ATM's session log-RMSE < B0's in ≥ 2/3 of counted sessions **and** pooled log-RMSE is lower. |
| Overall rule | **Confirmed** if ≥ 2 of BTC/ETH/SOL pass. |
| Readout | **Binding at ≥ 20 counted sessions** (about three weeks). Earlier runs of 05 are interim and non-binding. |

The win-share and pooled rule is the threshold proposed in 010 §6 and approved by Ivan. Three items were not specified there and were fixed here by Claude: the 20-session readout, the 100-row session minimum and the 2-of-3 asset rule. Ivan can amend any of them **before 04 is run**. After that, a change means a new version and a later window.

`mid_rv` per bar: \(\sigma^{\text{mid}} = \sqrt{RV_{\text{mid}} \cdot 29 / n_{\text{inc}}}\), requiring at least 26 of the 29 within-bar 10-second increments.

## 2. Mid-path pass-through (approved pipeline change)

`market_time_series.py` **v0.5.0** (time-series schema **0.6.0**) passes these raw collector fields through to `market_time_series_state` and `time_series_feature_state`:
- `mid_path_valid_points`, `mid_path_increment_count`, `mid_path_realized_variance`;
- zero/non-zero increments, sign flips, max abs log return;
- `book_out_of_order_update_count`, `collector_version`.

They are NULL before v0.3.0 (2026-10-04 16:20 UTC). No existing column changes. Canonical already carried them (02 `stage_schema`).

## 3. Order of execution

1. Rebuild features (schema 0.6.0).
2. Run 04 and commit `04_frozen_spec.json`. 04 loads only expiries ≤ 2026-10-08 and refuses to overwrite an existing spec.
3. Run 05 whenever new sessions exist. It loads expiries ≥ 2026-10-08, uses 10-08 only for lag features, and scores ≥ 10-09.

## 4. Questions for ChatGPT

1. Any objection to the rule or the readout size before 04 is run?
2. 05 compares the two models on the same rows but does not report uncertainty. Should a session-block bootstrap of the pooled difference be added as a *reported* (non-decisive) statistic?

## 5. Decisions needed from Ivan

1. Confirm or amend the three Claude-fixed items in §1 **before running 04**.
2. Rebuild features and run 04 (commands in `notebooks/Volatility_Regression/README.md`).
