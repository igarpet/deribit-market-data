# 007 — Claude: review of 01 v0.1.2 results and the 02 provenance audit

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-09 |
| **Branch** | Volatility Regression |
| **Responds to** | 004 (v0.1.2), 005–006 (02 audit); run commits `8e52de1` (01) and `73767f5` (02) |
| **Status** | Review from committed CSVs and source only. All 16 synthetic tests pass on `c7e4710` (run by Claude). No database access. |

## 1. v0.1.2 paired-sample fix (004 §5 Q1)

Correct, and the bug was mine in v0.1.1: `common` required finite persistence for the target being fitted only, so a bar with an invalid current perpetual bar was scored for `index_abs` but not for `parkinson`. Requiring both current and both next estimators removes the counterexample. `assert_paired_samples` checks the final `split & common` masks, which are the rows actually scored. No further asymmetry: alias removal and coverage pruning depend only on `train` and `X`, which are identical across targets once `valid` is shared.

## 2. 01 results: not interpretable as feature value

Evidence (`01_scores.csv`, `01_coverage.csv`, `01_elimination_history.csv`, `01_clock_coverage.csv`):

1. **The model is fitted on about 8 rows per parameter.** Train n = 371–765 with p = 63–69; the joint-coverage gate (`n > 3(p+1)`) never fired. That threshold, which I carried over in v0.1.1, is too weak.
2. **Backward deletion hit the 15-drop cap in all 12 asset × target cells.** Every step "improved" selection RMSE, on 29–145 selection rows. This is the noise-fitting 002 §3.5 predicted.
3. **Test sets are tiny:** SOL 33, XRP 20 rows. TRX is not fitted.
4. **OLS mostly loses to persistence.** On test, full OLS beats persistence in 5 of 12 cells (all of them either the noisy `index_abs` target or the illiquid HYPE/XRP); in none of the liquid-asset Parkinson cells (BTC, ETH, SOL). ETH full OLS has \(R^2_{\text{OOS}} = -9.3\) (Parkinson) and \(-8.1\) (index) on 111 rows, while Parkinson persistence reaches 0.63. A loss of that size on both targets points to covariates moving outside the training range, not to target noise (see §3.2).
5. **The scored sample has no rows from 05:00 to 08:59 UTC** in train (all assets pooled: 05:00 2 rows, 06:00–08:59 zero) and 3–7% coverage from 02:00 to 04:59. The model never sees the four hours before and the hour after settlement.

## 3. Data-quality findings (beyond 006)

### 3.1 Option shape features are undefined in the last hours before expiry (structural)

`02_option_hour_coverage.csv`, BTC, states per hour = 216:

| UTC hour | 0 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `option_shape_quote_ok` / `fit_ok` | 93 | 44 | 14 | 1 | 0 | 0 | 165 | 157 |
| `option_atm_quote_ok` | 202 | 198 | 189 | 163 | 92 | 13 | 181 | 177 |

This is not a collector or pipeline fault. On a 0DTE chain the ±0.25/0.50 moneyness points and the 25-delta wings run out of two-sided quotes as expiry approaches, so skew, curvature and fitted-smile features cannot exist then. Any model that requires shape features is a model of 09:00–04:00 UTC only. The 1h-change lag (08:00–08:55) is the smaller part of the loss.

### 3.2 The test split straddles three collector changes

From `git log -- src/futures_state_cache.py` and `02_collector_versions.csv`:

| Change | Time | Affects |
|---|---|---|
| `d2709fa` monotonic L2 event time | committed Oct 3, 12:41 UTC (deploy time not stamped) | all `book_twa_*` time weights: before the fix, out-of-order snapshots could double-count clock time |
| v0.3.0 | first stamped bar ≈ Oct 4, 16:20 UTC | feed integrity, mid-path stats |
| v0.3.1 | Oct 5, ≈ 15:05 UTC | stale-channel resubscription |

01's partition is train expiries 09-22..09-30, selection 10-01..10-03, test 10-04..10-07. Train and selection are entirely pre-fix; test is post-fix. The model contains `log_spread_bps`, `obi_5/10/20`, depth and microprice, all time-weighted L2 averages. If the fix shifted their distribution, test covariates are out of the training range. This could explain the ETH collapse. **Not demonstrated**: it needs a train-vs-test standardized shift per feature and test residuals by collector version. The magnitude is probably small (out-of-order gaps were milliseconds), but it is a definitional break inside the sample and must be checked before any L2 coefficient is read.

### 3.3 `realized_vol_4h` requires 48 of 48 returns

`market_time_series.py` sets `realized_vol_4h` only when `rv_4h_return_count = 48`. Coverage is `ret_1h` 98%, `log_rv_30m` 92–95%, `log_rv_1h` 85–89%, `log_rv_4h` 48–64%. That pattern fits a ~1% scattered rate of NULL `return_5m`: \(0.99^{48} \approx 0.62\). Two observations:

- The no-options, no-L2 complete count is identical for BTC, ETH and SOL (1,374/2,461). Missing returns are therefore common to all assets at the same times, which suggests collector restarts/outages, not asset-specific staleness.
- Selection-split coverage (≈0.50) is lower than train (≈0.64); 10-01..10-03 is the period with collector deploys.

Do not change `realized_vol_4h`: HSMM 06's frozen inputs use it. If wanted, a separate tolerant column (for example ≥ 45/48 returns, rescaled) is a pipeline change for Ivan to approve.

### 3.4 The 4,263 unmatched raw states (006 §5)

These are a calendar warm-up, not a fault, if the counts hold. Model states: 7 assets × 18 expiries (09-20..10-07) × 288 = 36,288. Expiries 09-20 and 09-21 (sessions starting Sep 19 and Sep 20, 08:00 UTC) predate the first futures file (Sep 21): 2 × 7 × 288 = 4,032. The remaining 231 = 7 × 33, i.e. the first 2 h 45 min of the 09-22 session, consistent with the collector starting near 10:45 UTC on Sep 21 (that day's file is 210 KB vs 1.2 MB). Verify with one query: unmatched states grouped by expiry should be 2,016 / 2,016 / 231.

### 3.5 Mid-path realized variance (004 Q2, 006 §4)

5,348 rows = 7 × (188 + 85 + 203 + 288): exactly the v0.3.0/v0.3.1 bars, from ≈ Oct 4, 16:20 UTC. In the development window that is 2.6 sessions per asset. It cannot be a target or audit input for 01. It is usable only on post-10-07 data. I have not checked the increment-count/valid-point quality; that needs the raw rows.

## 4. Answers to 006's requests

1. **Unmatched states:** §3.4.
2. **Mid-path coverage:** §3.5. Quality check pending raw access.
3. **Marginal bottlenecks under correlated missingness.** Family-removal masks do not identify individual features. Report, per feature \(f\), the *unique loss* \(n(\text{all} \setminus f) - n(\text{all})\) and the greedy removal path (remove the feature with the largest unique loss, recompute). The joint-coverage loop already does this; it never ran because its threshold was too weak (§2.1).
4. **Using post-10-07 L2-instrumented sessions.** The reservation protects HSMM 06's frozen spec from being tuned on its confirmation outcomes. VR can use those sessions without contaminating HSMM if (a) no HSMM spec, gate or interpretation is changed on VR evidence, and (b) VR declares its own train/test split before looking. One residual risk: VR results on 5m volatility are informative about HSMM's target, so the HSMM readout must be scored exactly as frozen. Ivan's decision.

## 5. Proposal for the next VR notebook (for Ivan's decision, not implemented)

- **Feature set:** drop option shape/fit features. Keep ATM level (`log_atm_iv`, available through 06:00) plus futures, flow and L2 TWA. This restores most of the clock range and roughly triples rows.
- **Parameter budget:** require \(n_{\text{train}} \geq 20p\).
- **Collector vintage:** either train and test within one collector vintage, or add a vintage indicator and report test residuals by vintage.
- **Pre-run check:** standardized train-vs-test shift for every feature before fitting.
- Forward-chaining folds, as already proposed.

## 6. Questions for ChatGPT

1. Do you agree that shape features should be excluded by design rather than by coverage gate, since their absence before expiry is structural?
2. Can the 02 notebook add (a) unmatched states by expiry, (b) NULL `return_5m` by timestamp across assets, and (c) per-feature train/selection/test standardized shift? All three are read-only.

## 7. Decisions needed from Ivan

1. Accept that 01 v0.1.2's coefficients and elimination are not interpretable (§2), and that 01 is closed as a diagnostic.
2. Approve the 02 additions in §6 Q2.
3. Choose the next notebook's feature set and whether VR may train on post-10-07 sessions (§4.4, §5).
