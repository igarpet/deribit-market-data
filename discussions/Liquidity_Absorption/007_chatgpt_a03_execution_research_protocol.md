# Liquidity Absorption 007 — A03 execution-research protocol

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-20 |
| **Branch** | Liquidity_Absorption |
| **Responds to** | `006_chatgpt_a02b_results_review.md`; Ivan's instruction to build A03 |
| **Status** | Implementation committed for exploratory historical simulation; **NOT an approved/frozen MNQ-to-crypto replication** |

## Objective and evidence boundary

Evaluate the same **452 original A02 price-only candidates**, now enriched with **A02b first/second touch-flow** data, under three distinct but identical-entry execution policies: `FIXED`, `BREAKEVEN`, and `TRAIL_1M`. Follow the original five-minute-context/one-minute-execution narrative. **No A02 candidate membership or trade-flow fields are changed**. Candidate feature measurements are taken at/before the signal close; *only the simulator* examines post-signal prices.

No genuine holdout remains within July 1–September 30, 2026. These dates have been inspected for feature extraction and are **historical exploratory data**, irrespective of whether performance is favorable. Inference about a tradable edge requires a new, locked protocol, later untouched data, and proper executable quotes.

## Explicit, unfrozen approximations

The user described an MNQ stop **15–20 MNQ ticks beyond the defended level**, break-even after **+10 NQ index points**, and trailing after successively confirmed **one-minute higher high/higher low** (reverse short), with the stop **2 ticks beyond the retracement wick**. These **MNQ absolute values do not transfer directly** to BTCUSDT/ETHUSDT. A03's *exploratory implementation proxies*, specified before viewing A03 returns, are:

| Decision | A03 provisional implementation |
|---|---|
| Entry | First recorded Binance USD-M aggTrade print at/after second-touch **1m bar close**, requiring print within 60 seconds of signal; no fill at the just-closed bar's close |
| Initial stop | Long: level minus 1× A02 halfwidth; short: level plus 1× halfwidth; rounded *outward* to assumed instrument tick (BTC 0.1 USDT, ETH 0.01 USDT). A02 halfwidth = 2bp; **this is not MNQ 15–20 tick equivalence** |
| BE | When an observed print travels **1.0 initial entry-to-stop R** favorably, move stop to **unadjusted first entry print** (fees may make realized exit negative). Only `BREAKEVEN` and `TRAIL_1M` do this |
| Structural trailing | After BE, detect a strictly higher **1m swing high** (1 completed left bar + 1 completed right bar confirmation), then a strictly higher **1m swing low** confirmed after that high; improve long stop to swing low minus **2 assumed instrument ticks**. Shorts use lower swing low followed by lower swing high, stop plus 2 ticks. Monotonic, no retroactive updates. Initial reference is signal candle high/low |
| Target | If completed-bar UTC-session VWAP is **at least 2 initial R on the favorable side of entry**, select VWAP; otherwise select the opposite **previous fully completed UTC-day extreme** (long prior-day high; short prior-day low) when correctly directional. If no valid structural target, skip and record. **Low-volume-node targets are NOT implemented** because no causal LVN level definition was approved |
| Timeout | 240 minutes from second-touch signal; market-price proxy at first observed print at/after deadline; if tape incomplete label censored; no invented exit |
| Fees and print-to-fill approximation | 6bp fee per side, +0.5bp adverse entry, 0.5bp adverse target/timeout exit, 1.5bp adverse stop exit. Stop exit no better than stop; target exit no better than target; outputs retain observed print and assumed fill separately |
| Funding | **Omitted**. `net_r_assumed_costs` is net **only of the specified transaction/impact assumptions**, not complete economic performance |
| Observed 02b volume quality | Flag any event with >1% aggTrade-to-kline quoted volume discrepancy in first or second touch, **retain** event, carry QC flag; no outcome-dependent exclusion |

All model constants are serialized into `03_execution_contract.json`. These are research hypotheses for **design comparison**, **not** faithfully user-supplied crypto stop/BE distances, true exchange tick-size rules over history, exact stop-market fills, measured spread, stop depth, or funding schedules.

## Causal event order and exits

Only prints at or after the A02 second-touch close can trigger simulated entry/management. For each later print, the simulator first incorporates 1-minute bars whose close timestamp is **<= print timestamp**; pivots are only confirmed after the required right bar **closes**. The simulator then applies eligible BE adjustments from the observed favorable excursion, followed by stop and target checks on the print. The first print after a pivot confirmation can trigger a revised stop but the bar's earlier wicks cannot retroactively do so. This makes confirmed-trailing versus fixed/BE treatment intentionally asymmetric, as in real discretionary monitoring. **Within a millisecond, the stable Parquet print ordering is used; true order-book event/queue sequencing is unknowable from aggregate trades.**

Exact stop-market order fill is not identified by last-traded prices. The simulation assumes adverse execution offsets; such assumptions are stress parameters and are not execution evidence. One-minute OHLC highs/lows are used for confirming swings only, never for intrabar stop/target ordering.

Three exit policies share identical ex-ante candidate, first print, stop/target geometry and starting position. Each trade/policy produces exit reason, entry/exit times, observed prints, assumed fills, event-specific risk/target R, break-even/structural stop revisions, gross R and transaction-cost net R. Cross-policy observations are correlated; **sum of per-candidate R is not a portfolio backtest** because concurrent trades are allowed and capital constraints are not applied.

## Flow conditioning without cherry-picking

The `absorption_like_exploratory` column is true when **(second/first zone volume > 1) AND (opposing zone share rises) AND (zone volume share of the whole candle rises)**. This exact conjunction was explored descriptively in A02b (88/452). It is included to make comparisons reproducible; **it is already an inspected, post-hoc proxy, not preregistered predictive evidence**. All 452 events are simulated, and quality flags are retained. Do not call a favorable subset confirmatory or infer significance from naïve event-level t-tests.

## Data and output contract

Inputs: 

- `notebooks/Liquidity_Absorption/02/02_causal_flow_features.csv`, `02_decision.json`
- `notebooks/Liquidity_Absorption/02b/02b_touch_features.csv`, `02b_decision.json`
- local Binance BTCUSDT/ETHUSDT USD-M perpetual 1m klines and daily aggTrades.

Outputs in `notebooks/Liquidity_Absorption/03/`:

- `03_execution_events.csv` — exactly 3 policy rows per A02 candidate, **including skipped and censored**; explicitly carries print/assumed-fill and quality flags.
- `03_execution_coverage.csv` — events, valid closed trades, skipped/censored, stop/target/timeout counts, descriptive gross and execution-cost net R per market/policy.
- `03_execution_contract.json` — exact config and limitations.
- `03_decision.json` — data coverage and **exploratory** run record; no profitability gate or promotion to live trading.

Execution engine: `src/data_engineering/modeling/liquidity_absorption_stage_a03.py`; synthetic tests in `tests/test_liquidity_absorption_stage_a03.py`; notebook `notebooks/Liquidity_Absorption/03_execution_simulation.ipynb`.

Performance: each symbol/day historical tape is opened at most once, streamed through Parquet batches, and processed without creating a full copied tick archive. CPU runtime will depend on trade counts and exit-time horizons. New exceptions must not be silently filled from OHLC data.

## Next review after user executes A03

Check source-event count, identical initial entries across three policies, exit/fill causality, censored/missing last-day cases, sample sizes by structural target, **net transaction costs versus risk distance**, impact of flagged A02b flow-volume outliers, and sensitivities to execution fees/spreads and the MNQ-to-crypto stop/BE mapping. Evaluate the absolute fixed/BE/trailing exit differences before conditioning on flow.

**No statistical-significance / economic-edge claim will be made from unheld July–September data.** Do not promote favorable in-sample return profiles to production or ask the user to trade them.

## Questions for Claude

1. Does the 1-left/1-right *bar-close* pivot protocol approximate the owner's real-time 1m HH/HL adjustments well enough for exploratory inference, or should a separate conservative delayed-pivot policy be run?
2. How should we bound plausible maker/taker execution economics from only Binance aggregate trades, and what future bid/ask snapshot data would most reduce identification uncertainty without tick archives?

## Decisions needed from Ivan

1. Review/approve or replace the **crypto-relative** stop, 1R BE, pivot confirmation, 240-minute timeout, and 2R-VWAP target thresholds **before any prospective confirmation**. These currently remain explicitly *exploratory*.
2. Authorize a causally constructed historical **low-volume-node target** as a distinct additional model once its window, binning and tie-breaking are specified.
3. If exploratory A03 results justify deeper study, reserve a later untouched period and freeze a complete cost/instrument-specific specification for actual out-of-sample evaluation.
