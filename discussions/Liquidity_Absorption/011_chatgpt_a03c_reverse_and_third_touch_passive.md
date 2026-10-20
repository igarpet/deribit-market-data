# Liquidity Absorption 011 — Independent reversed 1:1 and passive third-touch experiments

**Status:** Implemented; historical execution on local BTC/ETH aggTrade files and review pending.
**Date:** 2026-10-20. **Branch:** Liquidity_Absorption. **Version:** LA-A03c-0.1.0.
**Source:** A02/A02b 452 unchanged exploratory second-rejection events; A03b confirmed output `03b_1to1_events.csv` and `03b_decision.json` (452 rows, 1:1 pre-cost barriers). **No collector changes** and no previous results overwritten.

## Primary questions

1. If we enter the opposite side of A02's rejected-level signal, using the *same frozen historical first post-close print* and the mirrored A03b two barrier prices **swapped**, is the reversed-direction **1:1** path profitable after assumed aggressive-entry/exit costs? Unlike the earlier derived sign-flip thought experiment, replay every print after entry through first stop, target, or 240-minute timeout. Do **not** recompute original A03 target or reverse ex-ante selection filters.
2. If we **retain** the original absorption/rejection side but *rest a passive limit order at the original A02 defended level after the second rejection closes*, would a subsequent (third) revisit produce plausibly executable fills? What would net return be, conditional on putative fills **and per original A02 signal**, after maker-entry, taker-exit fees, assumed stop slippage, and no funding?

## Mechanisms, chronology and limitations

The original A02 first and second touches have already been observed. No order is submitted until second 1-minute candle **closes**, plus a provisional **1-second placement latency**. The post-only safety pre-check requires the last completed close to be strictly on the rejection side of the level (long: close>level; short: close<level). **This is not a historical BBO check** and cannot show whether the exchange would have accepted `postOnly` at this time. No maker queue position is observable in aggregated trades, even if the price trades through the limit.

We preserve the original A03 **target selection** from A03b, which chose 5m/1m-signal-time VWAP if at least 2 *original* initial risk units away; otherwise the opposite completed prior-day extreme, no low-volume nodes. That target is frozen when the second rejection closes for the *third-touch study*, using the original A03 target selected at the first post-close print. The same original A03 37 targetless candidates are explicitly skipped in every mode. Their absence means we cannot estimate a strategy with LVN targets. Differences in entry timing and selection between taker and passive alternatives must be interpreted as **different strategies**, not purely a fee ablation.

### Experiment A — reverse taker 1:1

- Exact first post-close print from **original A03b** (verified timestamp and price against fresh aggTrade tape).
- Reverse side: original long support -> short; original short resistance -> long.
- Original A03b **stop becomes reversed target** and original A03b **target becomes reversed stop**. They are exactly 1:1 pre-cost about the unchanged entry print.
- First recorded stop/target crossing print triggers exit; stop fill never improves over the barrier; target fill capped at barrier; 240 minutes from second signal close; no management trailing.
- Fees: **6 bp taker per side** (A03's conservative *assumed* fee) + 0.5 bp adverse entry, 0.5 bp adverse target/timeout exit, 1.5 bp adverse stop exit. No BBO/depth or actual executable proof.

### Experiment B — passive same-direction third touch (optimistic and stricter)

- Side unchanged from A02; limit price = historical original defended level. Long passive **buy** at level; short passive **sell** at level. Actual Binance instrument tick size/price rounding are not verified; A02 stored levels are used unchanged.
- Order becomes active **1 second after signal close**, expires **90 minutes** after signal close. Entry is assumed at limit **only after the market returns** to that level during the order's lifetime. No filled trade prior to signal, while awaiting placement, or after order expiry.
- **Maker touch proxy (optimistic):** the next appropriately aggressive print at-or-through the passive limit (seller-initiated for buy, buyer-initiated for sell). This is an *upper bound on eligibility*, not a guarantee our queued order would fill.
- **Maker through proxy (stricter):** next appropriately aggressive print **one assumed tick beyond** the resting limit (BTC 0.1 USDT, ETH 0.01 USDT, as the earlier A03 research contract). Still not queue-position proof. Price may trade at the limit without this proxy filling. Both use **actual aggTrade prints and aggressor flags**; no OHLC-based passive fills.
- Keep A03's frozen TP. At newly filled limit **E=level**, set symmetric 1:1 stop **S=2E−TP** (independent price geometry from A03b, because entry price changed). Exit at earliest crossing print or 240 minutes **after this maker entry**. A stop gap can lose >1R; timeout need not realize ±1R.
- Maker-entry fee **2 bp** *assumed*, taker exit **6 bp** *A03-consistent assumption*. No free guaranteed maker orders or assumed rebate. Maker limit price is assumed with zero adverse-entry slippage **conditional on a genuine fill**; all such fills unverified. Fee-only sensitivity includes (maker/taker bp) 2/6, 2/5, 0/5, 0/4, 4/6, but these do **not** model different order fills or queue dynamics. The standard USD-M USDT regular maker/taker advertised on Binance's fee table is 2/5 bp, while the A03 original model used 6 bp taker. Real VIP, discounts, promotions and funding differ. See: https://www.binance.com/en-BH/fee/futureFee ; actual historical 2026 contract terms/account fees must be confirmed independently.
- **Third touch** is operationalized as the first post-second-close **eligible price revisit** (plus side and, for strict, print-through). It is not a separately confirmed third one-minute rejection candle: orders are already resting when the next print reaches the price. No ex-post candle used to approve entry.
- **Missing eligibility ≠ losing trade**. The CSV explicitly records `NO_THIRD_TOUCH`, `NO_OPPOSITE_AGGRESSOR_PRINT`, `NO_TRADE_THROUGH`, other invalid orders, and both entry/exit censor categories.

## What constitutes positive EV?

Per filled/closed simulated trade: **mean net return in basis points and mean net R**; report closed-sample count, stop/target/timeout rates and net-profitable fraction.

Per **original 452 candidate signals**: **sum of closed realized net bp divided by 452**, treating genuinely unfilled/declined orders as zero deployed-P&L (no fill, no position). However, if any original signal is **CENSORED**, this quantity is **only partial realized P&L per signal**, not a valid completed-sample expectation. Both unconditional and fill-conditional results are needed to avoid selection on favorable maker fills. Explicitly report nonfills, cancellation, post-only proxy rejections and outstanding outcomes.

For source fairness, count **all 452 signals for every policy** (do not condition on the fill count of a competing policy). Combined BTC/ETH results should be reported separately and together. Chronological curves use **fixed USDT 10,000 notional per filled closed event**, plot gross and net and do not represent a capital-constrained account. Concurrent positions, funding, price impact, actual maker queue and bid/ask are unresolved. A maker-eligible crossing is **not** guaranteed fill, so even a positive simulated maker mean does not establish investable positive expectancy.

## Verification and risk controls

- Source A03b decision and 452 unique IDs validated, original barriers exactly 1:1.
- Shared daily Parquet reader with full chronological aggressive-side trade print fields; no copied multi-month raw tape.
- Synthetic regression tests for direction/barrier invariance, long/short maker aggression, placement latency, expiry, optimistic vs through, missing tape censor, gaps, timeouts, denominator logic, fee-only sensitivity, and streaming.
- Missing Oct 1, 2026 daily archive dates, which affected A03b, may censor late positions. Never invent closing prints.
- Strict *per-fee scenario* comparisons are descriptive and the full July–September 2026 window is previously inspected. **No fresh out-of-sample period**, no fitted signal selection, no valid significance or production approval.

## Files

- `src/data_engineering/modeling/liquidity_absorption_stage_a03c.py`
- `tests/test_liquidity_absorption_stage_a03c.py`
- `notebooks/Liquidity_Absorption/03c_reverse_passive_third_touch.ipynb`
- `notebooks/Liquidity_Absorption/03c/03c_events.csv`, `03c_coverage.csv`, `03c_fee_sensitivity.csv`, `03c_equity_paths.csv`, `03c_contract.json`, `03c_decision.json`, `03c_net_equity.png`, `03c_gross_equity.png` (all locally generated by execution).

## Future analysis requests

1. Report all three variant counts with separation of touch/proxy-fill/actually executable unknown; do not conclude maker profitability from `eligible` status alone.
2. Investigate if both reverse and third-touch positive EV remain after realistic USDT-M fee tier, order placement latency, threshold sensitivity, depth/queue simulation and a new untouched sample.
3. Independently validate maker fill probabilities with historical depth or tiny measured post-only paper/live order probes with explicit risk consent, rather than inventing fill outcomes from trades.
