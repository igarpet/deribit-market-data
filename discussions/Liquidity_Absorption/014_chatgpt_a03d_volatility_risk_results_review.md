# Liquidity Absorption 014 — A03d volatility-conditioned risk results review

| Field | Value |
|---|---|
| Author | ChatGPT |
| Date | 2026-10-20 |
| Source | Executed `03d_volatility_risk_integration.ipynb`, `03d/{03d_decision.json,03d_contract.json,03d_summary.csv,03d_event_forecasts.csv,03d_paired_baseline.csv,03d_paired_summary.csv}` |
| Status | HISTORICAL EXPLORATORY, NOT A CONFIRMED EDGE |
| Code/data changes | **None**; this document is a review only |

## 1. Status, coverage, and integrity

- The A03d `LA-A03d-0.1.0` notebook completed without an error and generated the expected output files and two PNG plots.
- There are **452 unchanged candidate IDs and 452 present causal VD18 HAR_4 spot-return variance forecasts**. 4,520 policy output rows = 452 original signals × (3 taker exit policies + 2 maker eligibility variants) × (60m + 240m).
- Executed logs: 171 futures daily aggTrade files in A03 taker replay and 170 in A03c maker replay. `BTCUSDT:2026-10-01` and `ETHUSDT:2026-10-01` are listed as missing and candidate results can be censored where relevant; a paired comparison excludes unmatched/nonclosed observations.
- Forecast origins `floor(signal_time_utc, 5min)` have no post-signal timestamps, with 0–4 minutes lag; spot HAR requires 30m, 60m, 240m and 1440m contiguous past realized variance. All 452 records report `PRESENT`; no future bars or stale spot forecasts are used per the implemented point-in-time join.
- Median forecast **terminal 60m spot return sigma**: **BTC 34.10 bp, ETH 42.83 bp**. Median 15m BTC 16.25/ETH 20.14 bp; median 240m BTC 71.09/ETH 94.40 bp. These are *terminal return risk*, not stop hit probability or maximum adverse excursion.
- A03d adopts frozen **VD18 TRAIN-fitted HAR_4** coefficients; it does NOT directly run the entire VD18 AM-T-HAR particle distribution or the daily VD20 24-hour forecasting workflow. The historical VD18 60-minute blended forecast did not conclusively beat the M2 Student-t benchmark; VD20's stronger 24-hour QLIKE results must not be transferred to 60-minute first-passage stop prediction.

## 2. Policy performance — all negative after assumed costs

The `03d_summary.csv` reports *per-closed-position* hypothetical net basis points; the 60m and 240m horizons are different model variants with some differences in censoring/exit sample.

| Entry / exit policy | BTC 60m (bp) | ETH 60m (bp) | BTC 240m (bp) | ETH 240m (bp) |
|---|---:|---:|---:|---:|
| Taker FIXED | -11.696 | -13.182 | -13.807 | -12.639 |
| Taker BREAKEVEN | -11.707 | -13.368 | -14.041 | -11.827 |
| Taker TRAIL_1M | -12.056 | -13.925 | -13.426 | -10.701 |
| Maker TOUCH_3RD (eligibility only) | -8.198 | -10.378 | -8.771 | -9.985 |
| Maker THROUGH_3RD (eligibility only) | -8.437 | -10.176 | -9.191 | -9.775 |

Closed-count-weighted combined means:

| Mode | 60m net bp | 240m net bp |
|---|---:|---:|
| Taker FIXED | -12.480 | -13.189 |
| Taker BREAK-EVEN | -12.584 | -12.869 |
| Taker TRAIL_1M | -13.042 | -11.989 |
| Maker TOUCH_3RD (eligibility) | -9.360 | -9.418 |
| Maker THROUGH_3RD (eligibility) | -9.365 | -9.502 |

All 10 strategy × horizon policies remain negative after the assumed transaction fees and execution adjustments, before funding. No model supports live trading.

## 3. Actual paired improvements are small and heterogeneous

A03d pairs the frozen **240m** FIXED and maker-touch variants against their corresponding 240m A03/A03c baselines on the **same candidate IDs where both trades closed** (source `03d_paired_baseline.csv`). This comparison is more meaningful than comparing different populations' averages.

| Comparison (vol minus original, mean net bp) | BTC | ETH | Paired combined |
|---|---:|---:|---:|
| VD18 240m FIXED versus A03 original FIXED | **-0.208** (n=195) | **+1.007** (n=219) | **+0.435** (n=414) |
| VD18 240m maker TOUCH versus A03c maker TOUCH | **+1.831** (n=169) | **-0.560** (n=193) | **+0.556** (n=362) |

On the 414 taker pairs the change improves **145**, worsens **211**, and is unchanged for **58**; **median difference = -6.64bp** despite a modest positive *mean*, meaning favorable outliers drive the average. Maker pairs improve 105, worsen 81, and are unchanged 176 (median 0bp). These correlated events are not independent significance trials. Both new models continue to have substantially negative absolute net expectancy.

## 4. What the volatility risk overlay is actually changing

- Taker A03 prior median initial price stop risk was roughly **6.16bp**; current A03d median stops are **~33.9bp BTC, ~42.2bp ETH** for 240m FIXED. This meaningfully changes the fraction of a risk unit consumed by taker execution costs and reduces the frequency of very tight stop-outs; this does *not* guarantee a profitable gross directional edge.
- Maker A03c original symmetric target-mirrored risk was typically **~83bp median combined**; A03d replaces it with a zone/wick/60m-spot-sigma protected stop, generally **~34bp BTC, ~42bp ETH**, while preserving the old structural take-profit and hypothetical maker eligibility. The new geometry is not a 1:1 RR test; stops can move closer and stop exits become more frequent. Do NOT describe A03d's maker improvement as pure volatility forecasting value: the stop and risk:reward contracts change.
- Illustration uses a **USDT 10,000 reference equity** and **0.25% (25 USDT) risk allowance per hypothetical trade**, cost allowances 14bp (taker), 9bp (maker eligibility), and a USDT 10,000 notional cap. Risk-budgeted virtual realized P&L is NOT a portfolio equity curve; positions can overlap with no aggregate capital, margin, liquidation, or funding accounting.
- Current A03d did **not** implement an adaptive volatility-conditioned holding period. It runs independent fixed **60-minute and 240-minute** variants.
- Maker mode does not demonstrate a separate third encounter, post-only acceptance, historical queue fill, or order-size capacity. AggTrades touch/through proxies remain insufficient for actual execution claims.

## 5. Risk budget illustration (NOT account equity)

Summing completed-trade *virtual* net P&L in `03d_summary.csv` under 25 USDT per-hypothetical-trade budget gives:

| Model | BTC 60m / 240m USDT | ETH 60m / 240m USDT | Combined 60m / 240m USDT |
|---|---:|---:|---:|
| Taker FIXED | -1,208.60 / -1,353.28 | -1,333.81 / -1,286.31 | -2,542.41 / -2,639.58 |
| Maker TOUCH | -830.47 / -837.22 | -1,041.04 / -941.11 | -1,871.50 / -1,778.33 |

These values sum *nominal per-trade risk budgets reused independently*, not sequential account equity. Fewer maker-filled positions also reduce realized absolute loss total.

## 6. Caveats and next research gates

1. **Calibration / transfer**: evaluate the ratio of ex-ante VD18 spot sigma to subsequent **perpetual** realized volatility and maximum adverse excursion/stop-first probabilities on timestamp-matched causal windows. Distinguish clean terminal variance skill from first-passage risk sizing. Do not fit on the displayed July–September outcomes and then claim validation there.
2. **Pair-aware uncertainty**: quantify uncertainty using whole-day/event-cluster resampling, stratified by symbol, signal side and origination date. The mean +0.43/+0.56bp is tiny compared with dispersion (around 40bp standard deviation for paired differences) and may not survive robust inference.
3. **Cost and target geometry**: audit ratio of target distance to new risk distance (fixed target unchanged), costs divided by stop risk, causal target feasibility, and whether a volatility-derived stop reclassifies trades as short/long stops sooner or later. Do not interpret a 60m versus 240m performance difference as an optimized dynamic holding rule.
4. **Maker realism**: separate the third distinct encounter via a causal excursion-and-return rule and stop identifying trade-through as certain queue fill; actual historical bid/ask or order book would be needed for stronger execution claims.
5. **Account portfolio**: sequential exposure, concurrent order reservation, position size caps, funding, liquidity/fees by date, and drawdown. The `03d_virtual_equity.png` image illustrates virtual outcomes only.
6. **Sample integrity**: the July–September 2026 candidate population is extensively inspected; reserve later unexamined dates for freezing the risk transformation and out-of-sample comparison. Keep the Volatility Dynamics research track read-only.

Data/report caution: `03d_summary.csv` column `mean_net_bp_per_452_signals_partial` is a per-*whole-universe* (452 signal) contribution **even within individual BTC/ETH rows**, not the mean per BTC signal or per ETH signal; do not present it as per-asset opportunity EV. Ongoing censoring also prevents complete per-signal EV.

**Verdict:** integration executed correctly under the registered synthetic/inference checks; material widening of aggressive entry stops and per-trade cost/risk relief; **small mixed paired improvement, still no positive net expectancy or confirmed alpha**. Recommended follow-up is a dedicated risk-forecast quality/first-passage audit, rather than more in-sample stop multiplier optimization.
