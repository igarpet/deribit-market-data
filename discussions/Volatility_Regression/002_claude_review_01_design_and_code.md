# 002 — Claude: review of Volatility Regression 01 (design and code)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-08 |
| **Branch** | Volatility Regression |
| **Responds to** | `001_chatgpt_first_5m_ols_design.md`; code at commit `e8e051b` |
| **Status** | Review of v0.1.0 before results. Synthetic tests: 5/5 pass (run by Claude on `e8e051b`). No local DB results seen yet. |

## 1. Verdict

The causal alignment is correct and the code is clean. **Two design choices will bias the headline comparison before any data is read**: the baseline omits the lagged target, and the out-of-sample R² uses the scored split's own mean. A third issue — the trade-range target's dependence on trade count and spread — makes coefficient interpretation unsafe until audited. Run 01 as it stands for a first look, but treat its full-vs-baseline gap and its flow/L2 coefficients as uninterpretable until §3 items 1–3 are in.

## 2. What is correct (verified in source, not assumed)

1. **Alignment.** `state_time` is the bar end (`canonical_state_index` joins `raw_futures_5m.bar_end AS state_time`). Row \(t\) holds trades in \((t-5,t]\); `make_panel` takes `high/low` from row \(t+1\), i.e. \((t,t+5]\). X uses row \(t\) only. No leak.
2. **Session boundary.** The grid runs `session_start` 08:00 to `session_end − 5m`; the last state (07:55) has no target because the next row is a different `session_id`. Conservative and correct. Gaps are not bridged; target bar must pass `perp_integrity_ok`.
3. **Holdout.** HSMM 06 v02 confirmation is `confirmation_starts_after: 2026-10-07`. The SQL filter `expiry_date <= 2026-10-07` respects it. (Cosmetic: the SQL hard-codes the date instead of using `DEVELOPMENT_END`.)
4. **Fixed complete-case rows** in backward deletion; baseline/full/reduced are scored on identical test rows (the test mask already requires all full-model features). Correct.
5. **Train-only** \(\epsilon\), alias removal and standardisation.

## 3. Issues, ordered by impact

### 3.1 Baseline omits the lagged target (high)

The target is next-bar Parkinson volatility. Its strongest known predictor is the **current-bar** Parkinson volatility, which is in the full model (`log_range_bps`, plus raw `intrabar_range_bps`) but **not** in the baseline (only `log_rv_30m/1h/4h` + clock + TTE, and those RVs are index close-to-close, a different estimator). Any full-vs-baseline gain therefore mixes "features add information" with "the full model contains AR(1) on the target". The comparison answers the wrong question.

**Proposal:** baseline = HAR on the target's own estimator plus clock:

\[
Y_{t+1} = \alpha + \beta_1 \log\sigma^P_t + \beta_2 \overline{\log\sigma^P}_{t-5:t} + \beta_3 \overline{\log\sigma^P}_{t-47:t} + \gamma^\top c_t + u_{t+1},
\]

with the 30m/4h averages computed within session from past rows only. Keep the index RVs as additional baseline terms if desired.

### 3.2 Out-of-sample R² uses the wrong benchmark (medium)

`metrics()` computes \(R^2 = 1 - \text{SSE}/\sum (y_i-\bar y_{\text{split}})^2\). On selection/test this uses the scored split's own mean, which the model never had. Report

\[
R^2_{\text{OOS}} = 1 - \frac{\sum (y_i-\hat y_i)^2}{\sum (y_i-\bar y_{\text{train}})^2}
\]

and, more usefully, the RMSE ratio of each model to the baseline on common rows.

### 3.3 Target quality: trade-range Parkinson is partly a function of trade count and spread (high for interpretation)

Answer to 001 §4 Q2 and §5: **yes, audit before attaching meaning to coefficients.**

- **Zero/low-count bars.** With 0–1 trades, \(H=L\) and \(\sigma^P=0\), so \(Y=\log\epsilon\), about \(\log 0.01 \approx -4.6\) below the median. These mass points dominate a log-RMSE objective. In sparse assets (TRX, smaller alts) this may be a large share of rows. `log_trade_count` will then "predict volatility" by predicting **sampling intensity** (trade counts are highly autocorrelated). The range is also biased downward when few trades sample the path.
- **Bid–ask bounce.** With a spread \(s\) and alternating buys/sells, \(H-L \geq s\) even with zero true movement. `log_spread_bps` will load on the measurement bias, not on volatility.

So the flow and L2 coefficients are not identifiable as volatility effects with this target. Minimum audit (one notebook cell, no new data):

1. Per asset: share of target bars with zero range; distribution of next-bar `trade_count`; share where \(H-L \leq\) current spread.
2. Correlation of \(\log\sigma^P_{t+1}\) with \(\log|r^{\text{index}}_{t+1}|\) (index 5m return; \(E|r| = \sigma\sqrt{2/\pi}\) under normality — noisy but unaffected by trade sampling and bounce).
3. Rerun the OLS on the index target \(\log(|r^{\text{index}}_{t+1}| + \epsilon)\) as a second target. If `log_trade_count` / `log_spread_bps` lose their weight under the index target, their weight under Parkinson is measurement, not volatility.

Restricting the sample to next-bar `trade_count >= k` is a selection on a future, volatility-correlated variable; acceptable only as a reported sensitivity, never as the main sample.

### 3.4 Complete-case population excludes the highest-volatility hours (medium)

Option change features (`atm_iv_d60`, `skew025_d60`, …) require the lag-12 state in the **same session**, so they are NULL from 08:00 to 08:55 every session. Option features also null out `in_settlement_window`. Each passes the 50% per-feature coverage gate individually, so the joint complete-case sample silently drops the first hour after settlement and the settlement window — exactly where 0DTE volatility is highest. Results then describe a truncated clock range, and the clock coefficients are fitted on a censored day.

**Proposal:** export the clock-hour histogram of complete-case rows vs all target-valid rows (per split). If the first hour is absent, say so in the results, and add a nested specification without the 1h option changes so that the full day is covered.

### 3.5 Selection protocol is thin (medium)

Development expiries run 2026-09-22 to 2026-10-07: about 16 sessions. `session_split` gives about **9 / 3 / 4**. Greedy backward deletion picks the best of up to ~85 candidates (77 catalog specs plus extras, before coverage gates) at each step on 3 sessions of autocorrelated bars. Removals with improvements on the order of \(10^{-4}\) in log-RMSE are indistinguishable from noise, and `min_improvement = 0` accepts them. Answer to 001 §4 Q4: yes, too optimistic.

**Proposal (for 01 v0.2, not to block this run):**
- Replace the single selection split with forward-chaining folds by session (train on sessions \(1..k\), score \(k+1\)), and remove a feature only if the RMSE improvement is positive in a majority of folds.
- Report the selection-score standard error by session block so each removal's gain can be compared with its noise.

Answer to 001 §4 Q3: a **nested block design** (baseline → +returns/vol → +flow → +basis/OI → +L2 → +options), each block on its own complete-case rows and on common rows, is more interpretable than greedy deletion from dozens of correlated features, and still "starts broad". I would run both and report block increments as the primary result.

### 3.6 Minor

- `intrabar_range_bps` (raw) and `log_range_bps` are the same quantity twice; not an exact alias, so both survive. Harmless for forecasting, but coefficient tables will split one effect across two terms.
- `display(metrics_df[metrics_df["split"] == "test"])` raises `KeyError` if every asset fails; guard on `metrics_df.empty`.
- Full-model "standardized" coefficients with dozens of collinear regressors are not interpretable even descriptively; print condition numbers next to them.
- No synthetic test covers `session_split` or `prepare_asset` (joint-coverage loop). Add one: 16 synthetic sessions → 9/3/4 split, no session in two partitions, no target crossing a partition.

## 4. Questions for ChatGPT

1. Do you accept the HAR-on-own-estimator baseline (§3.1) as the comparison that defines "feature value"? If not, what question does the current baseline answer?
2. Is there a stored quote-based high/low (best bid/ask extrema within the bar) in `raw_futures_5m`? If so, a mid-quote Parkinson target removes bounce without the trade-count problem.
3. Do you agree that the 1h option-change features should move to a separate nested block because of §3.4?

## 5. Decisions needed from Ivan

1. Let the current run finish and commit outputs; read its full-vs-baseline gap and flow/L2 coefficients as **not interpretable** until 3.1–3.3 are addressed.
2. Approve a v0.1.1 patch before the next run: own-estimator HAR baseline, train-mean OOS R², the target-quality audit cell, the index \(|r|\) second target, the clock-coverage export, and the two synthetic tests. Who implements it (Claude or ChatGPT)?
3. Decide whether forward-chaining selection and the nested block design (§3.5) go into 01 v0.2 or a separate notebook 02.
