# Liquidity Absorption 001 — Stage A kickoff and data audit

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-19 |
| **Branch** | Liquidity_Absorption |
| **Responds to** | Ivan's MNQ absorption/retest methodology; historical-data review |
| **Status** | Implementation — data audit, no trading verdict |

## Objective and boundaries

Reconstruct a *causal*, executable version of Ivan's discretionary second-touch absorption strategy from existing locally held Binance data. It is an independent branch: no model selections or research conclusions are inherited from existing branches. No changes to collectors, and no trades.

User-described rules, as observed (not adjusted or optimized):

- Five-minute market context; one-minute candle confirmation and real-time monitoring.
- Support/resistance indicated by repeated wicks (approximately 2–4 MNQ ticks), cumulative-delta slope and rapid growth of volume at the price zone.
- Entry after the **second** one-minute rejection candle closes, at a *subsequently executable* price.
- Initial stop 15–20 MNQ ticks beyond the zone, then move to entry after price advances 10 MNQ index points; trail long positions two ticks below a *confirmed* 1m retracement low after higher high/higher low; mirror for shorts.
- Targets are opposing prior-day liquidity zones, a displaced VWAP, or low-volume areas. All levels/target rules must be fixed using information known at entry.

**MNQ point/tick distances are not portable to crypto.** Price-zone tolerance, exit buffers, 1m swing confirmation, target selection and crypto tick values remain *unfrozen research-design decisions*; they must be registered before strategy comparisons or holdout use. We explicitly do not retrofit these choices based on audit output.

## Stage A plan

1. **LA 01 — Inventory and consistency audit (implemented):** BTCUSDT/ETHUSDT Binance USD-M 1m perpetual candles, optional spot 1m, and daily perpetual aggregated trades. Verify paths, minute coverage, timestamp monotonicity, duplicate minutes, malformed prices/volumes, day coverage. Probe first/middle/last available aggTrade day per symbol with bounded-size Parquet batches; reconcile derived aggressor buy/sell *quote* flow to Binance minute kline taker-volume fields. Two-basis-point logarithmic profile bins are **diagnostic storage bins only**, not zone-tolerance selections.
2. **LA 02 — Causal feature contract (next):** 5m context, price-only two-touch events, and price-binned aggressive buy/sell trade flow. CVD resets/session anchors, zone aging, wick/body definitions, and no-lookahead timestamps need explicit design before scoring.
3. **LA 03 — Execution research (after LA 02):** same entries under fixed stop, move-to-break-even, and fully causal trailing stop; compare target-first probabilities, net realized P&L and event-matched nulls. Trade-time sequencing for BTC/ETH must take precedence over 1m-bar assumptions. Actual historical order-book quotes/queue position unavailable: slippage and maker fills cannot be identified precisely.
4. **External confirmation:** leave truly unseen later samples untouched after the protocol is frozen. Historical days already explored across other branches are not automatically independent confirmation data.

## Local data paths

Defaults inferred from `DERIBIT_DATA_ROOT` or `BINANCE_UM_DIR`, can be overridden by `--binance-root`:

```
D:\MarketData\Binance\BTCUSDT_um_1m.parquet
D:\MarketData\Binance\ETHUSDT_um_1m.parquet
D:\MarketData\Binance\basis\spot\<SYMBOL>_spot_1m.parquet
D:\MarketData\Binance\aggTrades\<SYMBOL>\<SYMBOL>-aggTrades-YYYY-MM-DD.parquet
```

The two-year minute coverage and 92 days of aggTrades were established by *existing committed notebook manifests*, **not** by direct access to the PC. LA 01 will verify files actually present at run time. The Deribit historical downloader is outside Stage A and need not finish before LA 01.

## Run and review

From repository root in the `deribit-data` environment:

```
python -m unittest tests.test_liquidity_absorption_stage_a
python -m src.data_engineering.modeling.liquidity_absorption_stage_a
```

or execute `notebooks/Liquidity_Absorption/01_data_audit.ipynb`. Commit the CSV/JSON outputs from `notebooks/Liquidity_Absorption/01/` then request a review. The audit returns no edge or profitability conclusion.

## Questions for Claude

1. Is the diagnostic first/middle/last-day trade-flow reconciliation sufficient to proceed to LA 02, or should day-level historical completeness be established by full metadata scanning first?
2. Which order- and trade-data limitations create the most material ambiguity when transferring MNQ absorption concepts to Binance/Deribit?

## Decisions needed from Ivan

1. After viewing LA 01's actual local data outputs, approve the LA 02 feature/event contract before any pre-registered strategy comparison.
2. Keep the collector untouched unless the historical-trade quality findings justify a separate proposal.
