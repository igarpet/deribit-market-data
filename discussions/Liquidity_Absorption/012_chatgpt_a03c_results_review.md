# Liquidity Absorption 012 — A03c reverse 1:1 and passive third-touch results

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-20 |
| **Inputs** | Executed `notebooks/Liquidity_Absorption/03c_reverse_passive_third_touch.ipynb`; `03c/{03c_decision.json,03c_contract.json,03c_events.csv,03c_coverage.csv,03c_fee_sensitivity.csv,03c_equity_paths.csv}` |
| **Status** | **EXPLORATORY HISTORICAL RESULT, NOT VALIDATED TRADABLE EDGE** |
| **Change policy** | Review document only; no collectors, trade simulators or original results modified |

## 1. Successful execution and inputs

A03c `LA-A03c-0.1.2` completed on 452 unchanged A02 candidate IDs, producing 1,356 scenario rows (452 times three). It separately simulates `REVERSE_TAKER_1TO1`, `MAKER_TOUCH_3RD`, and `MAKER_THROUGH_3RD`. The independent tape replay read 170 daily aggregated-trade files. The missing 2026-10-01 BTC/ETH daily archives generated exactly **two censored positions per scenario** rather than invented results. The simulator preserves A03b's structural target decisions for the original-side maker variants; reverse swaps A03b's symmetric price barriers and direction. Both passive modes retain the ORIGINAL rejected-level side.

A03c's fee contract is **2bp maker entry / 6bp taker exit** (no assumed maker-entry price impact), plus 0.5bp adverse target/timeout exit or 1.5bp adverse stop exit. Reversed aggressive entry uses **6bp taker entry / 6bp taker exit**, plus adverse entry/exit costs. Funding, actual best-bid/ask, post-only acceptance, queue priority and portfolio capital constraints remain unobserved/unmodeled. Printed trade side and crossing price do not prove the researcher's limit order would fill.

## 2. Combined results from committed historical replay

| Scenario | Entered | Closed | TP/SL/timeout | Mean gross bp/closed | Mean **net** bp/closed | Net R/closed | Net-positive/closed | Mean partial net bp/452 original signals |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Reversed taker 1:1 | 415 | 413 | 112/116/185 | +2.471 | **-10.814** | -0.2951 | 178/413 (43.1%) | **-9.881** |
| Maker touch third | 364 | 362 | 95/85/182 | -1.130 | **-9.869** | -0.1080 | 178/362 (49.2%) | **-7.904** |
| Maker through third | 363 | 361 | 95/85/181 | -1.231 | **-9.970** | -0.1090 | 177/361 (49.0%) | **-7.963** |

*The per-452 original signal metric counts definitively unfilled or ineligible orders as 0 realized P&L but is marked PARTIAL due to two censored observations per variant. Do not report it as fully identified expected value. The three groups have different entries, directions and fills, so closed-trade means are not paired causal comparisons.*

| Asset | Reverse net bp/closed | Maker touch net bp/closed | Maker through net bp/closed |
|---|---:|---:|---:|
| BTC (195 / 169 / 168 closed) | -8.833 | -10.602 | -11.028 |
| ETH (218 / 193 / 193 closed) | -12.587 | -9.227 | -9.049 |

**Net-negative under the recorded assumptions in both assets for all variants.** Reverse long-side cases look less negative than reverse short-side cases, but this is an exploratory conditional decomposition and not evidence for a new long-only edge.

## 3. Distinguish fee reductions from a predictive signal

Per closed trade, the cost subtraction from gross price-return is **13.285bp** for reverse, **8.739bp** for optimistic maker touch and **8.739bp** for maker through. Passive entry improves *modeled* costs by ~**4.55bp**, largely by replacing a 6bp assumed aggressive entry fee with 2bp maker fee and eliminating entry adverse execution, but the **gross return for BOTH maker modes is already negative**.

Fee-only reprice with the same assumed crossings, **0bp maker fee and 4bp taker exit fee**, still yields mean net: reverse **-6.813bp**, maker touch **-5.867bp**, maker through **-5.969bp** per completed trade. (The reverse model does not use the maker fee.) Thus no tested fee scenario gives positive net EV; this does not establish what realized fills or refunds would be.

## 4. Maker eligibility and third-touch fidelity

- Optimistic maker touch: **364/452** eligible entered (80.5%); through: **363/452** (80.3%). Accounting: `NO_STRUCTURAL_TARGET` 37, `TARGET_REACHED_BEFORE_MAKER_FILL` 27, `NO_THIRD_TOUCH` 24, one additional `NO_TRADE_THROUGH` (BTC), and 2 censored positions per mode. These are *model* statuses, not empirical queue fill probabilities.
- In the optimistic maker-touch study, **104/364** assumed entries (~28.6%) occur within the first 60 seconds after the already-confirmed second touch; median wait after second signal close is ~**166 seconds**. The module regards any *post-close return to limit price* as the third touch; **it does NOT require a separately confirmed third encounter following a new departure**. This can confound a lingering second-touch episode with a genuine third independent test of a level.
- The `MAKER_THROUGH_3RD` 1-tick threshold is very small versus BTC/ETH price (assumed BTC 0.1 USDT; ETH 0.01 USDT). The stricter requirement eliminates **only one BTC entry** and **zero ETH entries** relative to the optimistic model, so it barely brackets real maker queue risk. Even prints through the level are insufficient proof of queue priority or post-only order acceptance.
- The post-only eligibility proxy uses the *second one-minute candle close*, not a contemporaneous BBO. AggTrade prices and aggressor side are insufficient to verify maker order eligibility, time priority, executed size or commission classification.
- Passive exits remain modeled as **taker** stops and targets, and stop/target/timeout can occur after maker eligibility without evidence of an actual maker fill.

Before promoting passive results, implement a genuinely separate encounter segmentation rule (price must depart beyond the defined zone and return only after a new minimum-time/price excursion, causally); tiered market-trade volume-through-at-level plus quote/queue or MBP snapshots; and order-size-aware fill capacity where available. Do not optimize threshold from already-inspected returns.

## 5. Virtual equity (NOT account equity)

The existing `03c_equity_paths.csv` cumulatively sums USDT 10,000 hypothetical notional per *completed trade* in exit-time order, allowing overlapping positions, with no margin reservation, funding, liquidations or capital constraints.

| Market | Reverse final net (USDT) | Maker touch final net (USDT) | Maker through final net (USDT) |
|---|---:|---:|---:|
| BTC | -1,722 | -1,792 | -1,853 |
| ETH | -2,744 | -1,781 | -1,746 |
| **Combined** | **-4,466** | **-3,572** | **-3,599** |

The maker curves lose less total dollar notional partly because **fewer assumed trades close**; therefore cumulative net P&L alone does **not** rank trade quality. Use per-closed and per-original-signal normalization and report censoring. The original A03b results are not altered.

## 6. Research assessment and next gates

**A03c execution: SUCCESS. Reverse direction: gross directional advantage too weak vs assumed costs. Passive third touch: lower modeled fee burden, but negative gross and net average returns; optimistic/strict fills barely separated; no genuine maker fill data. Evidence of profitable trading EV: NONE.**

Recommended next work is **A03d validation-first**, rather than threshold optimization on the inspected Jul–Sep 2026 period:
1. Define genuine third-*encounter* separation instead of a post-second-close price touch (causal departure > zone buffer, new approach), carry forward A02 level identity and original signal time. Freeze these choices before outcome testing.
2. Use a set of defensible maker fill assumptions from observable data, prioritizing relevant bid/ask or order-book sampling and queue-ahead proxy where available. Treat trade-through as eligibility only, never an assured fill.
3. Model actual position size, order lifetime, post-only rejection, cancelled orders, hedging/stop order fees, funding and overlapping capital. Keep a fee/spread/impact stress grid and drawdown controls.
4. Reserve truly untouched later data, because repeated analysis of July–September 2026 now exhausts its confirmatory value.
5. If the owner requests a causal new event family at LVN/HVN levels, register separately; current price-only pivots and prior-day levels are not equivalent to the owner's full discretionary strategy.

## Decisions needed from owner

- Approve a genuine *third distinct encounter* test with a causal departure/return rule and a conservative uncertainty range on passive fills.
- Choose whether to prioritize obtaining compact historical BBO/L2 snapshots or to limit research claims to print-only eligibility without fill conclusions.
- Approve capital/risk/funding rules before interpreting the virtual equity as a live-executable account outcome.
