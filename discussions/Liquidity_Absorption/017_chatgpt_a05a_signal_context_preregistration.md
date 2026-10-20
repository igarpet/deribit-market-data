# Liquidity Absorption 017 — A05a signal-centered −240 to +240 minute event study

**Author:** ChatGPT · **Date:** 2026-10-20 · **State:** IMPLEMENTED / historical outputs PENDING user's local run · **Scope:** retrospective exploration only, not a causal, profitable, or predictive model.

## 1. Motivation and chosen reference time

A04 established that early +15m price/flow behavior differs across eventually TARGET/STOP/TIMEOUT outcomes, and that original-target distance divided by forecast volatility strongly correlates with completion. Before fitting a conditional predictive model, **A05a measures how the market develops before the signal itself**, as well as after it.

The fixed clock reference is **t=0 = confirmed A02 second-touch one-minute candle CLOSE** (the predeclared `signal_time_utc`), **not** the first aggTrade entry fill. Use all integer candle closes **t=−240, −239, …, −1, 0, +1, …, +240**, exactly **481 one-minute records per signal**. Each candle at t records only information available at its completed close. The signal bar itself is available only at t=0. Actual entry, first touch, level-source timestamp, second touch, exit, and censor status have separate annotations.

This design is not a new strategy or a replacement for A04's active-position landmark risk sets.

## 2. Fixed sample, source contracts, and reference limitations

- Frozen source population: **452 A02 candidate IDs** as materialized in `04/04_primary_event_outcomes.csv`, A04 decision version `LA-A04-0.1.1`. A04's primary policy is A03d's `TAKER_FIRST_POST_SIGNAL` / `VD18_VOL_FIXED` / 240m. Retain **119 TARGET, 202 STOP, 93 TIMEOUT, 37 NO_STRUCTURAL_TARGET, 1 CENSORED_TAPE_END**, as recorded in the user's A04 sample; *the code verifies original identities and class names, not any fixed performance number.* All original candidates remain in the minute panel and completeness accounting; only TARGET/STOP/TIMEOUT enter outcome-conditioned medians.
- Restore contemporaneous second-touch OHLC and first/second touch times from `02b/02b_touch_features.csv`, via one-to-one candidate IDs, strict symbol/side/signal timestamp/level checks. The `level_source_time` in A02b is a retrospective event annotation; **it does not prove the liquidity level was actionable at that timestamp** for every level type. Before this time, plotting price relative to the eventually selected level is a retrospective visualization, *not a point-in-time available trading signal*. Even after it, original pivot confirmation delays might require additional proof.
- Raw market source: existing Binance USDT-M perp `BTCUSDT_um_1m.parquet` and `ETHUSDT_um_1m.parquet`. Read each ONCE. No new aggTrades downloads, no tick/L2 archiving, no Deribit/volatility model refitting.
- The original `signal_time_utc` close is checked directly against that event's Binance 1m close; mismatch fails loudly. Currency-time epochs are always normalized via `.dt.as_unit('ms').astype('int64')` to avoid the pandas ms-vs-ns A04 bug.
- Missing one-minute candles NEVER forward-fill, borrow future bars, or silently count as price touches. Rolling features require consecutive candles and reset after a gap. Explicit per-event 240m pre/240m post coverage flags are retained.

## 3. Measurements — *price first*, flow as context, not alpha selection

Fixed per-minute fields:

1. **Signed market price** relative to signal-close price; signed price relative to eventual defended liquidity level; 1m candle body, log return and high/low excursions. Signed means + is the *original* A02 trade direction, and - is adverse.
2. **Liquidity-level context**: minute range intersects `level_price ± zone_halfwidth`, favorable-side close, trailing 15 fully completed minute zone-overlap rate. First/second touch and the level-source timestamp are marked separately. No later touch is assumed to be a distinct third encounter.
3. **Volatility**: trailing 15m and 60m root-summed-square *perpetual* 1m log returns, strictly historical at each minute, no future 5m use. The existing VD18 60m **spot terminal** sigma forecast from A03d is retained as frozen event metadata for grouping (not recalculated at t<0); it must not be presented as a pre-formation feature.
4. **Activity and flow**: 1m and trailing 15m aligned taker-buy quote imbalance; contemporaneous quote activity divided by mean of the **preceding 60 complete minutes**; ratio of trailing 15m activity to earlier 60m reference. These are Binance kline aggressive-volume proxies, not passive limit-order absorption, queue rank, or hidden participants.
5. **Trade-execution annotations**: first-print assumed entry and historical exit markers, `trade_active_at_t`, and `trade_exited_by_t`. **The post-exit market path is still plotted through +240m**, for full market context. Prices after historical exit MUST NOT be treated as ongoing trade P&L, MAE/MFE, or at-risk positions.

The module additionally emits fixed **phase windows**, all candle-close-relative and inclusive: `[-240,-61]`, `[-60,-16]`, `[-15,0]`, `[+1,+15]`, `[+16,+60]`, `[+61,+240]`. Phase price changes, weighted aligned flow, quote activity, realized-volatility proxy, and level overlap are descriptive; if *any* candle in a phase is absent, mark its phase incomplete and price-change aggregate unknown.

## 4. Historical outcome comparisons (NOT selection criteria)

Median per-minute market paths stratified by **BTC / ETH** and eventual **TARGET / STOP / TIMEOUT**; further separate the frozen `target_to_sigma60 <= 1` versus `> 1` geometries. More finely stratified symbol/side/target family may be assessed later, but do not optimize conditional filters on this repeatedly examined sample.

- For every `(eventual outcome, t)`, output `n_candidates`, `n_price_present`, `n_trade_active`, and `n_trade_exited`. This makes the distinction between the FULL market path and A04's shrinking survivor cohort inspectable. Do not depict the median as a fixed pool of active trades.
- For pre-signal conditions, even though all observations are before t=0, grouping by outcome is inherently hindsight-conditioned; neither the eventual outcome, eventual stop/target, nor the eventual level should be inserted into ex-ante features earlier than they were actually established.
- **Survivor bias**: present in any ACTIVE-ONLY post-signal subset. A05a full-market post-signal comparisons explicitly include markets after exits, and separately counts active positions. Sample sizes can still vary by missing minute coverage.
- No alpha, first-passage model, threshold grid, regime labels, flow classifier, profitability optimization, or parameter refit is performed.

## 5. Output files and visuals

Source engine: `src/data_engineering/modeling/liquidity_absorption_stage_a05a.py`.
Tests: `tests/test_liquidity_absorption_stage_a05a.py` (strict timestamp units, negative-time no-leakage, missing-minute rolling reset, known event markers, phase accounting, full post-exit path and identity joins).
Notebook: `notebooks/Liquidity_Absorption/05a_signal_context_event_study.ipynb`, executed locally in the `deribit-data` kernel.

Expected outputs under `notebooks/Liquidity_Absorption/05a/`:
- `05a_event_minute_panel.csv.gz` (481 per original signal, ~217,412 rows; compressed to avoid unnecessary repository bloat)
- `05a_median_context_curves.csv` (per-minute outcome, asset, and target/sigma geometry medians plus coverage and trade-active counts)
- `05a_context_checkpoints.csv` (fixed −240, −120, −60, −30, −15, −5, −1, 0, +1, +5, +15, +30, +60, +120, +240 comparisons)
- `05a_phase_panel.csv`, `05a_phase_summary.csv` (six predeclared phases)
- `05a_coverage.csv`, `05a_cohort_metadata.csv`, `05a_decision.json`
- `05a_price_paths.png`, `05a_flow_context.png`, `05a_activity_context.png`, `05a_rv_context.png`, `05a_level_context.png`
- `05a_BTCUSDT_event_heatmap.png`, `05a_ETHUSDT_event_heatmap.png`, `05a_example_event_timelines.png` (earliest complete candidate ID per symbol/outcome, **illustrations not representative samples**)

Notebook launch: `python -m unittest tests.test_liquidity_absorption_stage_a05a -v`; then **Restart Kernel & Run All**. Push the executed notebook, results and logs. No earlier modules or collectors are modified.

## 6. Next decision only AFTER inspecting the descriptive outputs

Focus on whether different TARGET/STOP/TIMEOUT paths have **different pre-signal approach slopes, compression levels, and zones/touch arrangements** once grouped by the known (post-signal) target-feasibility ratio. If differences appear, preregister how a feature could be measured at the actual decision timestamp, not at the creation of a future level. Confirm prospectively on *unseen* chronological data with cluster uncertainty and trading costs before interpreting any price/flow associations as market structure alpha.
