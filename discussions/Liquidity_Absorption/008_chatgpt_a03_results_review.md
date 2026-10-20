# Liquidity Absorption 008 — A03 exploratory execution results review

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-20 |
| **Branch** | Liquidity_Absorption |
| **Responds to** | `007_chatgpt_a03_execution_research_protocol.md`; committed `notebooks/Liquidity_Absorption/03/03_{decision.json,execution_contract.json,execution_coverage.csv,execution_events.csv}`; executed `03_execution_simulation.ipynb` |
| **Status** | Empirical historical execution readout; **no prospectively verified edge or live-trading approval** |

## 1. Audit of simulated population and event mechanics

- Source universe: 452 unchanged A02/A02b candidate IDs (BTCUSDT 212; ETHUSDT 240), evaluated under three policies `FIXED`, `BREAKEVEN`, `TRAIL_1M`. `03_execution_events.csv` has **1,356** rows; each candidate has exactly three policy rows with the **same** first-print entry, assumed entry fill, initial stop, selected target, initial risk, and initial target R across policies (no pairing/identity mismatches).
- **415/452 (91.8%)** were entered and closed in each mode: 196 BTC and 219 ETH. **37/452 (8.2%)** were skipped for `NO_STRUCTURAL_TARGET` (16 BTC, 21 ETH). Zero `NO_ENTRY_TAPE`, zero recorded censored policies. Target sources among entered candidates: 233 `SESSION_VWAP`, 182 `OPPOSING_PRIOR_DAY`. No causal LVN targets were implemented.
- The daily tape scan logged 171 daily Parquet reads and missing `BTCUSDT:2026-10-01` and `ETHUSDT:2026-10-01`. No recorded trade was censored; this is a bounded archival-coverage limitation to retain in disclosure.
- A02b flow-volume mismatch criterion (>1% at either touch) flagged **3/452** candidates; **only 1** entered under the current target rule. Flags are descriptive, not post-return exclusions. There are no changes to raw data, A02/A02b event definitions, or production collectors.

## 2. Absolute simulated performance — exploratory and before funding

Results below are the *mean over 415 closed positions per policy*, not an investable portfolio return. Trade overlaps, capital constraints, mark-to-market risk and funding are not modeled.

| Exit policy | Mean gross initial-R | Mean net initial-R, assumed costs | Mean gross return (bp) | Mean net return (bp) | Net-positive / 415 | Stops / targets / timeouts |
|---|---:|---:|---:|---:|---:|---|
| FIXED | -0.0014 | **-2.2511** | +0.2246 | **-13.6365** | 54 | 357 / 45 / 13 |
| BREAKEVEN | +0.0550 | **-2.2021** | +0.4667 | **-13.4450** | 34 | 378 / 33 / 4 |
| TRAIL_1M | +0.2387 | **-2.0234** | +1.5358 | **-12.4116** | 47 | 393 / 22 / 0 |

A03 contract imposes **6 bp taker fee per side**, **0.5 bp adverse entry**, **0.5 bp adverse target/timeout exit** or **1.5 bp adverse stop exit**. Mean modeled round-trip cost (mean gross minus net price-return basis points) is approximately **13.86–13.95 bp** across policies. Median entry-to-initial-stop price risk is **6.16 bp**, 10th/90th percentiles **4.35 / 10.23 bp**. Therefore fees/execution costs typically exceed the whole planned stop-risk in price terms, contributing approximately **2.25 initial R** cost per completed position. The `net_r_assumed_costs` column excludes funding, exchange spread/queue observability, leverage effects and complete portfolio accounting.

**Do not reinterpret the negative R as evidence against the original MNQ methodology.** A03 substitutes an unapproved BTC/ETH stop at 1 A02 zone halfwidth, break-even at +1 initial R rather than +10 MNQ index points, 1-left/1-right one-minute pivots with a 2-crypto-tick buffer, 240m hold, and VWAP/prior-day targets without LVNs.

## 3. Paired exit-policy comparison

Among the **same 415 completed events**:

- `TRAIL_1M` minus `FIXED`: **+0.2277 mean net R**; 159 events improved, 36 worsened, 220 unchanged.
- `BREAKEVEN` minus `FIXED`: **+0.0490 mean net R**; 159 improved, 21 worsened, 235 unchanged.
- `TRAIL_1M` minus `BREAKEVEN`: **+0.1787 mean net R**; 86 improved, 15 worsened, 314 unchanged.
- BTC trailing versus fixed: -2.0251 versus -2.2531 net R. ETH trailing versus fixed: -2.0219 versus -2.2493 net R. `TRAIL_1M` moved stop structurally at least once on **112** positions (53 BTC, 59 ETH); 216 positions (105 BTC, 111 ETH) reached the model's 1R break-even trigger.

Relative trailing performance is better in-sample, but the **net return remains materially negative** and the mean return difference of ~1.22 bp is smaller than uncertainty about executable quotes or fee tier. These observations are correlated counterfactual policies sharing trades and clustered by date/market; no uncorrected event-level p-value or prospective performance claim is warranted.

## 4. Absorption-like filter does NOT demonstrate value

The post-hoc descriptive A02b three-condition conjunction (higher second-touch zone quote volume AND higher opposing aggressor share AND higher fraction of the minute's turnover near the level) marks **88/452** candidates; **84** are entered, versus 331 entered without the flag. No inference may be made about *hidden passive order identity* from these features.

| Group (closed trades) | Fixed net R | Trailing net R |
|---|---:|---:|
| 84 absorption-like | **-2.8821** | **-2.3765** |
| 331 other price-only rejections | **-2.0910** | **-1.9338** |

Even gross-R on the absorption-like fixed subset was negative (-0.4879) versus +0.1220 for nonflagged; risk distances, event contexts and target feasibility differ between groups. **In this exploratory simulation the specified proxy underperforms, not outperforms.** This does not imply the original discretionary absorption hypothesis is refuted, nor authorize retroactive threshold optimization.

## 5. Data and execution-model limitations requiring attention

1. **Stop geometry can conflict with the confirmation wick.** For **55/452** candidate second-touch candles (23 BTC; 32 ETH), the A03 *initial stop price* was already reached by that signal candle's wick (long low <= stop, short high >= stop). Of those, **53** would have been entered under A03's structural-target rule. Since A02 permitted a deeper wick penetration than A03's chosen stop offset, a stop only 1 zone halfwidth beyond the level may not reflect the owner's `15–20 MNQ ticks beyond defended level` rule. The crossing was **before the simulated post-close entry**, so is not a code-level retroactive stop-out; it is a structural fidelity and risk-geometry issue.
2. Trade-print crossing is not a proved executable quote. First prints at or after signal close are assumed entry proxies; stops use adverse price adjustment from crossing print/stop barrier, targets are capped at their preset prices. No historical order book, queue position, realized spread, funding, or dynamic liquidity-sensitive impact is supplied.
3. A03's two tick assumptions (BTC 0.1 USDT; ETH 0.01 USDT) and pivot confirmation rule are proxies, not approved instrument-calibrated MNQ equivalence.
4. The absence of LVN targets **skips 37 setups**, so results are conditional on finding a VWAP or opposing previous-day target and cannot estimate the original strategy's full opportunity set.
5. All events occurred during **July–September 2026**, already inspected for exploratory feature work. The current results are not out-of-sample prospective evidence.
6. A position-size, overlapping position, daily exposure and capital-use accounting layer is absent; all mean returns here are standalone unlevered return proxies (bp) and per-trade initial R, not an account equity curve.

## 6. Decision and recommended follow-up

**LA A03 code execution: SUCCESS. Exit-policy paired comparison: DESCRIPTIVELY BETTER WITH TRAILING. Economic performance under current assumptions: STRONGLY NEGATIVE. Faithful MNQ-to-crypto translation: NOT YET. Predictive/confirmatory evidence: ABSENT.**

Before building another alpha filter or optimizing parameters against Jul–Sep returns, propose an **A03a implementation-validity / economics sensitivity** explicitly separate from signal selection:

- Reconcile initial stop geometry against A02's allowed wick ranges; establish a structurally meaningful stop behind the *actual defended zone*, with crypto-relative risk calibration approved by owner.
- Predeclare instrument-specific realistic cost scenarios and attainable order types using documented fee schedules and, where available, executable quote measurements; do not infer favorable maker fills from aggTrades.
- Measure risk in price bp, fees/cost in bp, break-even and trailing enhancements in both R and bp; test whether conclusions remain negative over reasonable **predeclared** adverse-cost ranges. Funding remains a documented limit until explicitly modeled.
- Audit the missing Oct 1 tape implications and reconcile the single entered A02b >1% flow mismatch; keep flow quality criteria independent of trade outcomes.
- If further research is approved, implement causal LVN target construction as a separately registered variant rather than silently modifying original A03 candidate selection.
- Reserve **untouched later data** and fix all choices *before* any confirmatory prospective trial; prior events remain exploratory.

## Questions for Claude

1. Does the 55/452 confirmation-wick-versus-proposed-stop conflict warrant a revised explicit **initial zone invalidation / stop-geometry rule** before execution economics sensitivity?
2. How should per-side fees, spread/impact and funding be stress-tested without presenting print-derived hypothetical fills as executable market prices?

## Decisions needed from Ivan

1. Approve/reject **A03a predeclared cost-and-stop fidelity audit** without changing collector or original A03 result files.
2. Approve an explicit crypto stop-distance, BE-move and HH/HL confirmation interpretation prior to prospective use; the current 1 zone-halfwidth / 1R / 2-tick choices remain exploratory proxies.
3. Decide whether to add a causal LVN-target family in its own future research variant and reserve a later unseen time window.
