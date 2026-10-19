# Liquidity Absorption 003 — LA 02 causal feature and event contract

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-20 |
| **Branch** | Liquidity_Absorption |
| **Responds to** | `002_chatgpt_a01_audit_readout.md`, corrected LA 01 v0.1.2 outputs, and Ivan's authorization to proceed |
| **Status** | Implemented exploratory feature contract; **not frozen for P&L** |

## Observed historical readiness

`01_decision.json` now reports `LA-A01-0.1.2` and `ready_for_signal_design=true`. Both BTCUSDT and ETHUSDT USD-M perpetual candle histories contain 1,067,040 contiguous minute bars (2024-10-01 through 2026-10-11). There are 92 BTC and 92 ETH aggTrade daily files (2026-07-01 through 2026-09-30), zero missing *filenames* within that window. Six sampled trade-to-kline comparisons reconcile 1,440 minutes per sample day; median quote volume errors are zero and the maximum sampled p95 relative quote volume error is 0.00062015 (~0.062%). These outcomes support feature extraction. A three-day probe per asset does **not** certify every trade day.

## Scope, provenance, and independent status

The branch explores Ivan's discretionary MNQ pattern translated to BTC/ETH Binance perpetuals, **not** transferred historical claims from other repo branches. Context is 5m; confirmed retest, cumulative delta, and stop management occur at 1m. This notebook constructs *candidates and contemporaneously available features only*. It neither chooses profitable thresholds nor evaluates realized trades.

**Available discovery universe:** 2026-07-01 through 2026-09-30 UTC, BTCUSDT and ETHUSDT, 1m Binance USD-M perp candles and matching aggTrade files. Three days of candle warmup enable five-minute pivots and previous completed UTC day reference levels. The much longer 1m bars can be used later for a separate price-only frequency baseline, but NOT called an unseen confirmation set if they were previously analyzed elsewhere. External later prospective data must be explicitly reserved before fitting any payoff-sensitive decisions.

## Exact A02 operational proxy — not yet Ivan's full discretionary zone recognition

1. **Reference level, known before touch.** Two independent candidate sources per side: (a) previous **fully completed UTC day** high/low; (b) latest fully confirmed five-minute swing pivot, strictly more extreme than the previous 2 and subsequent 2 completed five-minute bars. The pivot becomes available only at the **close of the second following five-minute bar** (10 minutes after the pivot candle close), remains eligible for 720 minutes. The pivot is not backdated to its forming candle.
2. **Tolerance** is level price × 2 basis points on either side. This is an **untuned instrument-relative proxy**, not MNQ's tick width and not an outcome-optimized threshold. First and second support touches require the candle *low* within [level − 2×width, level + width], a close strictly above level + width, and lower wick at least 35% of full one-minute candle range. Resistance mirrors with upper wick and close below level − width.
3. **Independent second touch:** minimum three one-minute closes between first and second event timestamps, maximum 90 minutes, with **an intervening completed candle** closing at least 3×width away in the expected rejection direction. Breach by a close more than 2×width past level invalidates/reset first touch. A new source level also resets it. Thirty-minute cooldown for the same reference stream after a recorded second-touch candidate.
4. **Signal timestamp** is exactly the **second 1m rejection candle close**; all measured features end at this timestamp. A02 does not assert an executable fill at the same close.
5. **Delta** = 2 × Binance one-minute taker-buy quote volume − one-minute quote volume (USDT); 5- and 15-minute rolling delta, quote flow, and UTC-midnight anchored CVD are stored separately. Delta is **not used to select price-only events**, so we can later test incremental value against a true price-only baseline.
6. **Traded volume near level** is computed only from prior trades in [signal close − 10 minutes, signal close). Trade records are compressed into 1bp logarithmic price bins, recording quoted volume and aggressive buy/sell quote volume by minute/bin. Bins whose *centers* lie within the frozen event-level tolerance are counted. Recent 5m and previous 5m near-level quote volume, recent 5m share of total traded quote volume, growth ratio if denominator positive, and local aggressor delta are recorded. These are **approximations** (up to half a price bin) rather than exact executions at a quoted price. Missing UTC-day files or missing trade minutes are marked with explicit missing status, never silently treated as zero.

The detector currently treats previous-day extremes and recent pivots as **candidate reference levels**. It does *not* derive levels from a fully evolving volume-profile node, which Ivan sometimes uses; implementing such nodes would need a further causal rule, validated with only previously observed trade prices/volumes. Nor can trades identify iceberg order identities, book replenishment, or latent stops.

## Outputs and provenance

- `notebooks/Liquidity_Absorption/02_causal_features.ipynb`
- `src/data_engineering/modeling/liquidity_absorption_stage_a02.py` (version `LA_A02_VERSION=LA-A02-0.1.0`)
- `notebooks/Liquidity_Absorption/02/02_price_only_candidates.csv` — **no delta or volume columns**
- `notebooks/Liquidity_Absorption/02/02_causal_flow_features.csv` — identical price-defined events, now with flow and trade-profile features
- `02_coverage.csv`, `02_feature_contract.json`, `02_decision.json`
- `tests/test_liquidity_absorption_stage_a02.py` — synthetic causal/no-lookahead and timestamp regressions

Neither feature file includes future returns, stop/target outcomes, or P&L. The complete new module's aggTrade streaming path requires the PC's existing `pyarrow`/pandas environment; synthetic price-only tests run offline.

## Planned LA 03 and restrictions

LA 03 will require Ivan's approval of the reference-level proxy and the **actual trade-management semantics**. Specifically: post-close entry price; initial stop offset relative to zone; crypto-normalized equivalent of MNQ's fixed offsets; break-even at an instrument-relevant favorable excursion; swing high/low recognition/confirmation at the available timestamp (not hindsight pivots); never loosen trailing stop; realistic stop-market order slippage, fee/funding, and conservative same-minute sequencing. No parameter retuning against prior outcomes can be described as preregistered confirmation. Original high-RR target logic involves prior-day opposing zones, displaced VWAP and low-volume areas and must be specified causally before ranking trades.

No collectors, live orders, prior branch model fits, historical stage-01 outputs or experimental thresholds from other branches were modified for this module.

## Questions for Claude

1. Should a causal rolling volume-profile **node** be added as a third candidate reference family, and how would one prevent level-selection leakage while keeping computation tractable?
2. Which synthetic intrabar stop-order sequencing tests are essential for LA 03's break-even and structural trailing rules?

## Decisions needed from Ivan

1. Review the **proxy level sources and 2bp tolerance** after LA 02 output; authorize any structural changes *before* outcome evaluation. The user may prefer dynamic volume-profile nodes to pivots.
2. Decide when the A02 event universe and A03 execution rules are frozen for any prospective confirmation. Merely pushing A02's descriptive outputs is not approval of a profitable trading system.
