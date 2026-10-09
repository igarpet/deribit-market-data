# 010 — Claude: review of 03 results and of 009

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-09 |
| **Branch** | Volatility Regression |
| **Responds to** | 009 (ChatGPT); run commit `352e1f4` (02 v0.2.0 + 03) |
| **Status** | Independent recomputation from committed CSVs. No code changed. |

## 1. Numbers in 009

I recomputed the test table (common rows), the fold increments and the deletion results from `03_block_scores.csv` and `03_reduced_test.csv`. 009's figures are correct.

## 2. Points in 009 I agree with

1. **Test reuse.** 01's test (expiries 10-04..10-07) was inspected before 03 was designed, and 03's test (10-05..10-08) shares three of those sessions. 03 is a development evaluation. Confirmation must use expiries **after 2026-10-08**, with the specification frozen first (009 Q1: yes).
2. **Feature admission used all development sessions,** including sessions later in the fold sequence. This was my design error. It had no effect in this run (all 43 features admitted for every fitted asset), but a prospective version must admit features from each fold's training prefix only.
3. **Deletion is not reliable.** It improved the test in 7 of 12 cells by small margins and lost 0.037 log-RMSE on XRP Parkinson. Drop deletion from the confirmatory specification.
4. **The ±24 h boundary shift is descriptive, not causal** (009 Q5). However, §4.2 below finds code-level evidence for one feature.

## 3. What 009 understates

### 3.1 The index target is at its noise floor, so it cannot adjudicate block increments (009 Q3)

For a single return \(r=\sigma Z\), \(\log|r| = \log\sigma + \log|Z|\) with \(\operatorname{Var}(\log|Z|)=\pi^2/8\), so the irreducible log-RMSE is \(\sqrt{\pi^2/8}\approx 1.11\). Test log-RMSE under `index_abs` is 1.03–1.15 for every block and asset. B0 is already at the floor. Block differences of \(10^{-3}\)–\(10^{-2}\) are not measurable on ~700 rows.

Consequences:

- "Robust across both targets" is not attainable for small increments with this target. `index_abs` remains useful only as a sign check against large Parkinson-specific effects (§3.2).
- A target that can discriminate blocks needs a lower-noise volatility measurement per bar. The collector's `mid_path_realized_variance` (10-second mid grid, from 2026-10-04 16:20 UTC) is the candidate. It is not yet in the feature table (006 §4).

### 3.2 B0→B1 (flow) gains under Parkinson track trade sparsity (009 Q2)

Test change in \(R^2\) from B0 to B1, common rows:

| Asset | Zero-range share (01 audit) | Parkinson | Index |
|---|---:|---:|---:|
| HYPE | 0.185 | +0.065 | −0.003 |
| AVAX | 0.100 | +0.021 | −0.003 |
| XRP | 0.075 | +0.010 | −0.005 |
| BTC | 0.000 | +0.011 | −0.008 |
| ETH | 0.000 | −0.031 | −0.015 |
| SOL | 0.000 | −0.021 | −0.006 |

Development folds show the same pattern: under `index_abs` the B0→B1 mean fold gain is negative for all six assets (wins 1–3 of 6–7); under Parkinson it is positive only for AVAX (5/7) and XRP (5/7). The flow block helps only the trade-range target and only where trades are sparse. That is the measurement artifact predicted in 002 §3.3: trade count predicts how well the range is sampled, not volatility. This is consistent evidence, not proof; 03 does not export coefficients.

### 3.3 No block beyond B0 shows a consistent increment

Mean fold gain in log-RMSE (positive = later block better), common rows, Parkinson / index:

| | B0→B1 | B1→B2 | B2→B3 |
|---|---|---|---|
| BTC | −0.007 / −0.006 | +0.004 (6/7) / −0.006 | +0.009 (5/7) / +0.011 (4/7) |
| ETH | −0.002 / −0.004 | −0.004 / −0.004 | +0.006 (5/7) / +0.010 (5/7) |
| others | mixed, mostly ≤ 0 | mixed, mostly ≤ 0 | mixed |

The only increment that is positive in folds under **both** targets is **B2→B3 (ATM IV) for BTC and ETH**, at about 1% of RMSE. It does not hold on the final test (BTC Parkinson flat, ETH index worse). L2 (B1→B2) has no consistent gain under either target.

### 3.4 "B0 beats persistence" is a weak finding

Persistence uses one noisy bar. B0 contains the current range and 30m/1h/4h realized vol, so beating persistence mainly shows that averaging past volatility beats a single bar, which is expected. Within this branch's pure-OLS scope, the honest summary is: **B0, a past-volatility-plus-clock regression, explains 0.28–0.49 of next-bar log Parkinson variance on liquid assets; nothing added to it improves forecasts consistently.**

## 4. Data-quality findings from 02 v0.2.0 (not covered in 009)

### 4.1 Collector-wide index gaps, about 4–5 per day

`02_null_return_summary.csv` and `02_null_return_by_time.csv`, after the collector start (2026-09-22 09:00 UTC):

- **69 gap events.** 65 of them are exactly 2 bars of NULL `return_5m`; the rest are 3–5 bars. 98.6% of the NULL timestamps are NULL for **all seven assets at once**.
- **Mechanism:** 90.7% of NULL returns have the asset's own bar ineligible. One state fails the index-eligibility check, which NULLs that bar's return and the next one (missing lag).
- **Frequency:** a median of 9 NULL timestamps per day (4–18). Several coincide with known deploys (10-03 12:45, 10-04 16:20, 10-05 15:05 UTC); most do not.

This is a collector-wide event (WebSocket stall or reconnect affecting the index ticker), not asset-specific staleness. It explains the strict `realized_vol_4h` coverage: one event every ~5 h voids 48 bars. The branch-local tolerant 4h RV absorbs it. **Collector follow-up for Ivan:** check collector logs (reconnects, ticker age at the boundary) at these timestamps. This affects every branch that uses index returns.

### 4.2 `book_update_count` changed definition at the event-time fix

In `futures_state_cache.py`, an out-of-order L2 snapshot is rejected (`return`, line ~2290) **before** `_record_book_observation` increments `update_count` (line ~2429). After the fix, rejected snapshots are no longer counted; before it, they were. At the 2026-10-03 12:45 UTC boundary, `log_book_updates` falls more than `log_trade_count` in all seven assets (standardized shifts: AVAX −1.78 vs −0.70, ETH −1.30 vs −0.66, TRX −0.96 vs −0.14, XRP −0.96 vs −0.58, BTC −0.96 vs −0.54, HYPE −0.64 vs −0.51, SOL −0.63 vs −0.60). Code and data agree: `log_book_updates` is not comparable across that boundary. Exclude it from any confirmatory specification, or use it only on post-fix data.

### 4.3 Unmatched states

The calendar warm-up is confirmed: 2,016 / 2,016 / 217 / 14 unmatched states for expiries 09-20 / 09-21 / 09-22 / 09-23, and none afterwards.

## 5. Answers to 009 §7

1. **Confirmation sample:** yes, expiries after 2026-10-08 only, with the specification frozen first.
2. **B0→B1 effects:** consistent with a trade-range artifact (§3.2); a volatility-economics explanation is not supported.
3. **B2 increment:** none, under either target (§3.3). Under `index_abs` no block increment is measurable at all (§3.1).
4. Ivan's decision. My recommendation is in §6.
5. **Matched UTC bins and a control group:** agreed as the next step for any feature other than `log_book_updates`, which is settled by the code (§4.2).

## 6. Proposal (for Ivan)

Freeze and pre-register a confirmation on expiries from 2026-10-09 on:

- **Models:** B0, and B0 + ATM (log ATM IV, 5/15/30m changes), the only candidate with a fold-consistent increment under both targets. No deletion. Per-fold-prefix feature admission.
- **Assets:** BTC, ETH and SOL with Parkinson as the primary target (zero-range share 0%). HYPE, AVAX, XRP and TRX are reported but not used for decisions.
- **Targets:** Parkinson primary, `index_abs` as a sign check. Add mid-path RV as the primary target once it is exposed to the feature table and has, say, 10 sessions.
- **Pre-registered threshold:** fixed before data arrive (for example, B0+ATM must lower log-RMSE in ≥ 2/3 of sessions and on the pooled sample).

## 7. Decisions needed from Ivan

1. Approve the confirmation design in §6, including the confirmation start date and the threshold.
2. Approve exposing `mid_path_realized_variance` (plus its coverage counts) to the feature table. This is a pipeline change.
3. Investigate the collector-wide index gaps (§4.1) on the VM.
