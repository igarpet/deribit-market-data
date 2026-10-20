# Liquidity Absorption 013 — A03d VD18 volatility-to-risk bridge

| | |
|---|---|
| Author | ChatGPT |
| Date | 2026-10-20 |
| Track | **Independent** `Liquidity_Absorption`; read-only external VD18 fitted artifact |
| Status | IMPLEMENTED **EXPLORATORY** sensitivity study; live profitability **NOT** established |
| Requires | Existing executed A02/A02b, A03/A03b and A03c inputs; user's local Binance/Deribit-linked spot+UM historical archives |

## 1. Approved research architecture

**Liquidity Absorption determines whether and in which direction a trade occurs; Volatility Dynamics informs risk scale and risk budget.** A03d DOES NOT use volatility forecasts to reverse directions, predict the market's sign, cherry-pick entry candidates, or choose a new price target. It does not edit the Volatility Dynamics branch, data collectors, or earlier LA results.

- **Frozen strategy reference:** 452 BTC/ETH A02 two-touch candidates, original first-post-close taker entry, or A03c's already-defined passive-limit touch/through *eligibility* at the original liquidity level. Existing A03/A03b structural target remains unchanged. It can become less favorable relative to a wider stop (R:R not constrained to 1:1), which is deliberately exposed in outputs.
- **Volatility reference:** original TRAIN-fit HAR_4 coefficients, design centering/scaling, asset effects and return-variance rescaling in `notebooks/Volatility_Dynamics/18/18_manifest.json`. For forecast horizons **15/60/240 minutes**, `HAR_4` is VD18's previously selected family. The only inference information is Binance **SPOT** five-minute closes with `state_time` at bar END. Every forecast uses the most recent *exact* completed five-minute bar `floor(signal_time,5min)` plus strictly contiguous past windows 30m/60m/240m/1440m. No bar ending after signal, no forward-fill, no test-period refit.
- **Time horizon:** 60m versus 240m holding windows are reported separately; no in-sample selection or profit-driven adaptive timeout. The 15m and 240m predictions are retained as diagnostics to support later horizon-policy design.
- **Volatility transfer:** spot forecast -> perp-risk policy is **NOT YET validated**. VD18 HAR forecasts **terminal net-return variance**, not maximum adverse excursion, stop-first probability, first-passage time, or direction. A03d is testing a specific mapping rather than assuming this difference is irrelevant.

## 2. Fixed initial stop mapping (one primary, no in-sample grid)

At the entry price `E` (first prompt taker print, or assumed limit at A02 reference level), direction `d` (+1 long, −1 short), wick extreme `W` and zone halfwidth `w`, require a protective stop that lies beyond *both* the level's outside edge and the signal-confirmation wick, and at least `k=1.0` 60m forecast sigma away from entry. The fitted forecast `vhat_60` is a **dimensionless net log-return variance**; `sigma_bp = 10,000 sqrt(vhat_60)`.

- Long: `stop_raw = min(level − w − tick, signal_low − tick, E − k·E·sqrt(vhat_60))`; round DOWN to assumed exchange tick.
- Short: `stop_raw = max(level + w + tick, signal_high + tick, E + k·E·sqrt(vhat_60))`; round UP to assumed exchange tick.
- Tick assumptions inherit A03 (BTCUSDT 0.1, ETHUSDT 0.01 USDT), unvalidated against venue's historical symbol precision. No ATR or terminal sigma is misrepresented as a predicted adverse excursion.

**Taker source invariant:** A03's prior-day/VWAP priority runs using the **original A03 provisional stop basis**; the selected take-profit and first trade-print entry are then checked against the immutable A03b source. ONLY the stop/risk budget changes. This is intentional to avoid target-selection circularity. The original A03 FIXED, BE(+1 new initial risk), and TRAIL_1M policies are rerun at **60 and 240-minute timeouts**.

**Maker source invariant:** A03c's currently existing *post-close level revisit* and one-tick through eligibility are rerun with the SAME old target and volatility stop computed at the limit price. If target or stop is hit before an eligible maker print, the unfilled order is cancelled. Maker **order fill and queue rank remain UNOBSERVED**, and a post-close level revisit is NOT necessarily a distinct third encounter (A03c's known issue). Both maker modes also use 60 and 240-minute holds after assumed entry.

## 3. Position sizing is illustrative and cost aware

Reference equity = 10,000 USDT, nominal risk budget = 0.25% = **25 USDT per hypothetical position**, notional cap = 10,000 USDT. Estimated all-in cost allowance is 14bp taker round trip, 9bp maker-entry+taker-exit. For `risk_bp = |E−S|/E×10,000`, the reference notional is

`N = min(10,000, 25 × 10,000 / (risk_bp + cost_allowance_bp))`.

Per-event **virtual realised P&L** is `N × assumed_net_return_bp/10,000`. It assumes each entry can separately use the reference risk budget. Trades overlap; liquidity, margin, capital reservation, funding, realized BBO slippage, and concurrent loss constraints are absent. Curves are **not account equity**, and maker scenarios are **not executed fills**. Compare net bp per trade and per original signal, plus risk-conditioned reference P&L, alongside the traditional fixed-$10k notional baseline.

## 4. Inputs, QC and output contract

- Read-only CSVs from `notebooks/Liquidity_Absorption/02b/`, `03b/` and `03c/`, and A03 A02+A02b event VWAP inputs; candidate IDs, side, timestamps, reference levels and first print/target checked for exact drift.
- Read-only VD18 fitted model from `notebooks/Volatility_Dynamics/18/18_manifest.json` and local `DERIBIT_DATA_ROOT/history/price_5m/{BTC,ETH}.csv.gz` (user may supply `--spot-history`). Spot feature windows must be strictly complete and have the exact as-of completed five-minute bar. A missing forecast gets a documented **NO_COMPLETE_CAUSAL_5M_HISTORY** status, **never** substituted with a current or future volatility reading.
- Read-only Binance historical futures candles and daily aggTrades from `BINANCE_UM_DIR` (or A01 `binance_root`). Unchanged A03/A03c timestamp-ordered replay. A03c scan enforces missing-day censoring; A03's existing scan is reused (its gap caveats remain inherited).
- Outputs under `notebooks/Liquidity_Absorption/03d/`:
  - `03d_event_forecasts.csv` — forecast origin/status, sigma/variance for 15/60/240, 452 untouched candidate IDs
  - `03d_risk_execution.csv` — all simulations, event outcomes and reference position size
  - `03d_summary.csv` — coverage, net bp/R and risk-budgeted virtual P&L stratified by asset/policy/hold
  - `03d_paired_baseline.csv`, `03d_paired_summary.csv` — completed ID-paired A03 FIXED and A03c optimistic maker vs vol-stop 240m counterparts (both closed only)
  - `03d_virtual_equity.png`, `03d_forecast_vs_stop.png` — virtual realised P&L by mode and stop-vs-forecast diagnostic
  - `03d_contract.json`, `03d_decision.json` — immutable policy and model provenance.

**Gate / interpretation**: Avoid claiming that a negative or positive A03d net expectancy is a validation of predictive ability or genuine maker fills. July–September 2026 is an already-inspected exploratory sample. A newly frozen model-to-risk mapping needs **later, uninspected data**, and ideally a spot-vs-perp variance calibration and actual bid/ask/queue-aware execution model. Do NOT select sigma multipliers or holding windows by in-sample P&L. The existing passive notion of a "third touch" should be improved with a distinct encounter departure/return segmentation in a separate controlled variant.

## 5. Owner choices for subsequent studies

1. Approve or change the provisional sigma multiplier and the 0.25% nominal per-trade risk for a later **frozen prospective** protocol, not retroactively based on A03d P&L.
2. Decide whether to keep the frozen VWAP/prior-day targets, add causally derived LVNs as their own candidate/target family, or test a separately preregistered volatility-derived take-profit.
3. Decide whether to build the **genuine third separate encounter** and tighter passive fill-capacity/queue study before attempting live order-execution inference.
