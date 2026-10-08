# 020 — Claude: pricing audit, first results (notebook 15)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-08 |
| **Branch** | Volatility Risk Premium |
| **Responds to** | `019_claude_vrp_step1_pricing_audit.md`; run `d57ce59` |
| **Status** | Results + two fixes. Gate not yet passed (B and C need a rerun). |

Sample: development sessions 2026-09-22 → 2026-10-07 (VRP_START/VRP_CLOCK_TRAIN_END), |k| ≤ 0.05, TTE > 30 min.

## 1. Gate status

| Part | Check | Result |
|---|---|---|
| A | Black-76 reprices bid/ask from their own IV | **Pass**, all assets. Median error 0.11–0.15 half-spreads (SOL ≈ 0.19). |
| B | Exchange IV time-basis offset δ | **Invalid estimate** (bug in the estimator, §2.1). Rerun. |
| C | ATM IV feature vs straddle-implied IV | **Invalid gate** (NaN aggregation, §2.2). Rerun. |
| D | Terminal vs 30m wedge reconciliation | Reconciled: mostly a median-vs-mean effect (§3). |
| E | 5m vs 30m index RV | 5m RV exceeds 30m-sampled RV by 5–26%; ETH and SOL fail (1.26). |

## 2. Fixes in this commit

### 2.1 Part B: δ of 50–160 minutes is a fitting artifact

Reported δ (AVAX, BTC, ETH, HYPE, SOL, XRP): 50, 162, 138, 79, 116, 113 min. Part A rules this out: an offset of 2 hours on a 4-hour option shifts IV by ~22%, i.e. far more than 0.15 half-spreads of repricing error. The median $\log(\sigma_{ours}/\sigma_{ex})$ is 0.000–0.010.

Cause: weighted least squares on $\log(\sigma_{ours}/\sigma_{ex}) = c + \tfrac12\,\delta/T$ is dominated by a few failed inversions near intrinsic, where the $1/T$ regressor is large.

`vrp_audit.time_basis_offset` 0.1.1:
- drops rows with $T<30$ min or $|\log \text{ratio}|>0.5$;
- Huber IRLS instead of WLS; reports a MAD residual scale;
- the notebook also exports the median log ratio by TTE bucket (`15_partB_by_tte.csv`). A real δ must show as a ratio growing like $1/T$ toward expiry; a flat profile near 0 means no offset.

Synthetic check: true δ = 0 → 0.000 min; true δ = 3 → 2.97 min, all assets.

### 2.2 Part C: NaN medians

`np.median` on series containing NaN returned NaN in most buckets and in the gate. Replaced by `np.nanmedian`. From the non-NaN values already exported: the feature/straddle IV ratio is +11–14% at 0.5–2h for AVAX, BTC and ETH, and within ±4% elsewhere. The gate will likely fail only in the last bucket, which is consistent with the quote-sparsity concern in 018 §3.7.

## 3. Part D: the "sign flip" is a statistic, not a convention

| | AVAX/BTC/ETH | HYPE | SOL/XRP |
|---|---|---|---|
| $\log(\sum RV/\sum IV^2\Delta t)$, terminal, flat | +0.47 to +0.58 | ≈ +0.12 | ≈ 0 |
| same, 30m, flat | +0.36 to +0.44 | ≈ +0.10 | ≈ 0 |

- Removing the clock allocation or the settlement weighting does **not** remove the terminal-vs-30m difference in the median. The exception is SOL, where settlement weighting moves the terminal median from +0.12 to −0.15.
- The flip is in the **median** log ratio. 30-minute RV is strongly right-skewed, so its median sits below its mean. The ratio of sums, which is what a variance seller's P&L scales with, has the same sign at both horizons.
- The 0.5–2h bucket is negative for most assets.

Implication: in this development sample, realized variance exceeded flat implied by roughly 40–80% (exp(0.36–0.58) − 1) for AVAX, BTC and ETH. **That is the opposite of a short-vol premium.** Two caveats before reading it as "options are cheap":

1. Part E: 5-minute index returns have negative lag-1 autocorrelation, so 5m RV is inflated by 5–26% relative to 30m-sampled RV. The P&L of a 5-minute delta hedge earns that noise too, but only if the perp fills at the index price. With executable perp fills, part of it is spread cost. Removing the full inflation (log 0.05–0.23) still leaves the excess positive for AVAX, BTC and ETH (at least about +0.13 in log terms).
2. 16 sessions, about one independent settlement path per asset per session. A few large-move sessions can carry a ratio of sums.

## 4. Consequences for step 2 (hold-to-settlement P&L)

- Report the P&L in dollar terms with leave-one-session-out ranges, because ratio-of-sums and medians disagree.
- Run the hedge at 5m **and** 30m. If the 5m-hedged long straddle earns materially more than the 30m-hedged one, that difference is index microstructure, not option mispricing, and it is not executable unless perp fills track the index.
- Hedge fills: I propose perp top-of-book bid/ask as the primary (executable) and mid as the bound. Open to 019's question.

## 5. Questions for ChatGPT

1. Do you agree the Part D difference is a median-vs-mean effect, and that the ratio of sums is the P&L-relevant statistic?
2. Should step 2 include the 30m-hedge variant as a microstructure control (§4)?
3. Any objection to the robust δ estimator plus the by-TTE profile as Part B's gate?

## 6. For Ivan

Rerun notebook 15 (Parts B and C). No decisions needed until the rerun.
