# Liquidity Absorption — independent research line

MNQ-inspired BTC/ETH Binance perpetual *historical exploratory* research. 5-minute context and 1-minute entry/management. Existing production collection pipelines are untouched. Versioned reusable engines live under `src/data_engineering/modeling/`. Local Binance root comes from `BINANCE_UM_DIR` or the parent of `DERIBIT_DATA_ROOT` (`.env`).

- **01_data_audit.ipynb** — Binance 1m/perp/spot, aggTrades first/middle/last checks, flow reconciliation; outputs in `01/`.
- **02_causal_features.ipynb** — independent price-only two-touch candidates and signal-close flow features; outputs in `02/`.
- **02b_touch_flow.ipynb** — same A02 candidates; first-vs-second level-executed volume, aggressor flow, zone dwell proxies, pre-first profile; outputs in `02b/`.
- **03_execution_simulation.ipynb** — same A02 candidates under FIXED / BREAKEVEN / TRAIL_1M, trade-print ordered stop/target and *assumed* fills; outputs in `03/`.
- **03b_mirrored_stop_equity.ipynb** — **keep the original A03 structural TP and entry**, set initial stop the identical price distance away on the losing side (exact 1:1 gross RR). Executes historical prints, compares paired closed outcomes to original fixed-stop A03, and produces gross/net BTC/ETH/combined virtual equity curves in `03b/`. +1R BE/trailing are redundant with 1R TP; only FIXED is meaningful. No actual trade performance has been generated until the local notebook is run.

- **03c_reverse_passive_third_touch.ipynb** — independent replay of **reversed-side 1:1** using A03b frozen barriers, and original-side **passive limit orders at the liquidity level** awaiting a third post-second-close revisit. Separate opposing-aggressor touch and stricter one-tick-through maker-eligibility proxies. Outputs in `03c/`: all 452 signals per scenario, skip/censor statuses, net expectancy per closed trade and per signal, fee-only sensitivities and gross/net BTC/ETH/combined virtual equity curves. **Passive fills are hypothetical, NOT evidenced maker queue fills**; passive avoids taker fees only on entry, exits remain taker under the research assumptions.


- **03d_volatility_risk_integration.ipynb** — causal, read-only bridge to the saved **VD18 HAR_4** 15/60/240-minute spot terminal-return variance forecasts. Liquidity Absorption retains **all existing directions, entries and frozen structural targets**. A protective stop is set beyond both the second rejection wick/zone and one unit of VD18 60m forecast sigma. Reports risk-budgeted hypothetical position sizing and compares 60m vs 240m hold policies for A03 taker FIXED/BE/TRAIL and A03c touch/through maker *eligibility* (not known maker fills). No in-sample optimization; forecasts require strict complete past 5m bars. Outputs in `03d/`, source protocol `discussions/Liquidity_Absorption/013_chatgpt_a03d_vd18_risk_bridge.md`.

- **04_intratrade_path_anatomy.ipynb** — A04 entry-to-exit study on the frozen A03d fixed-stop 240m taker cohort (452 original candidates). Compares still-active TARGET/STOP/TIMEOUT paths at causal +1/+5/+15/+30/+60m checkpoints, using existing Binance UM 1m candles and taker-buy quote-volume flow proxies. Produces risk-set panels, group summaries and competing-exit charts under `04/`. Outcome-conditioned **descriptive** diagnostics only; no new tick archives, reoptimized exits or confirmed prediction. Engine: `liquidity_absorption_stage_a04.py`; tests: `tests/test_liquidity_absorption_stage_a04.py`; protocol: `discussions/Liquidity_Absorption/015_chatgpt_a04_intratrade_anatomy_protocol.md`.

- **05a_signal_context_event_study.ipynb** — signal-close-anchored **−240 to +240 minute** one-minute Binance UM event study across all **452** A02 candidate IDs, explicitly annotating first touch, second touch/signal, original level-source timestamp, entry and exit. Retrospective full-market paths (including after an exit, **not** ongoing P&L), outcome/target-feasibility groupings, fixed formation/approach/post phases, volume/flow/realized-volatility context, heatmaps and earliest-complete-ID examples. Uses A04 primary cohort and A02b joins, without new tick data or future bar imputation. Outputs under `05a/`; historical results require local notebook run. **Any preformation reference to the eventual liquidity level is hindsight context, not a known historical signal**.

A03 uses explicitly **provisional** crypto stop/BE/target/fee parameters, and does **not** replicate the user's MNQ point/tick distances, simulate low-volume-node targets, fund positions, identify executable bid/ask liquidity, apply portfolio capital constraints, or claim prospective trading significance. Its historical sample has already been examined for features. Every trade retained in its output, even censored/skipped events.

Run in the `deribit-data` conda environment after `git pull`:

```
python -m unittest tests.test_liquidity_absorption_stage_a03
jupyter lab
```

Open and **Restart Kernel & Run All** in `notebooks/Liquidity_Absorption/03_execution_simulation.ipynb`, then commit the notebook and **four** generated `03/` CSV/JSON files to GitHub for independent review. Data are read-only and aggregate trade files are streamed by symbol/day; CPU cost depends on tape volume.

Research discussions under `discussions/Liquidity_Absorption/` document the hypotheses, non-frozen assumptions, integrity caveats, and owner decisions needed before a genuinely untouched holdout experiment.

Run A03b with `python -m unittest tests.test_liquidity_absorption_stage_a03b`, then open `03b_mirrored_stop_equity.ipynb` and run all cells, commit `03b/` CSV/JSON/PNG outputs and executed notebook. Graphs assume fixed USDT 10,000 notional per independent event without capital/overlap constraints; they are *not* executable account equity. The historical tape lacks bid/ask and funding.

Run A03c with `python -m unittest tests.test_liquidity_absorption_stage_a03c` then open `03c_reverse_passive_third_touch.ipynb` in the `deribit-data` kernel, run all, and commit `03c/` outputs. The prior A03b results are the required source and remain untouched. A03c assumes 1-second order-placement latency, 90-minute passive-order lifetime, 240-minute holding after maker entry, 2bp maker-entry fee and 6bp taker-exit fee for A03 comparability, plus a fee-only sensitivity grid. The historical aggTrade print side/price establishes *possible* queue eligibility but cannot prove a passive order filled. Review `discussions/Liquidity_Absorption/011_chatgpt_a03c_reverse_and_third_touch_passive.md` before interpreting returns.

Run A03d after pulling updates with `python -m unittest tests.test_liquidity_absorption_stage_a03d`; open `03d_volatility_risk_integration.ipynb` in the `deribit-data` kernel and **Restart & Run All**. In addition to `BINANCE_UM_DIR`, the frozen VD18 forecast bridge needs local `DERIBIT_DATA_ROOT/history/price_5m/{BTC,ETH}.csv.gz`; results remain exploratory and no actual full historical A03d run has been performed in the assistant's environment. Upload the executed notebook and generated `03d/` artifacts for audit. Holding-time scenarios are compared descriptively; no adaptive timeout is claimed before testing.

Run A05a with `python -m unittest discover -s tests -p "test_liquidity_absorption_stage_a05a.py" -v`, then open `05a_signal_context_event_study.ipynb` in the `deribit-data` kernel and select **Restart Kernel & Run All**. Push the executed notebook and `05a/` results to `main`. A05a reuses the locally generated `04/04_primary_event_outcomes.csv` and existing Binance UM 1m Parquet files; it neither refits VD18 nor rescans aggregate trade tick archives. Research protocol: `discussions/Liquidity_Absorption/017_chatgpt_a05a_signal_context_preregistration.md`.

- **06_deribit_entry_context.ipynb** — A06: Deribit-only context around the frozen A04 **entries**, −240…+240 min at 1-minute resolution: perpetual liquidations by side, mark–index premium, blocks, taker imbalance and price; short-dated ATM IV, put–call skew and taker-signed option delta flow; nearest dated-future basis. TARGET/STOP (by target geometry), WIN/LOSS and hour-matched non-signal CONTROL windows; BH-FDR. Pre-entry candidates go to the LA 022 holdout only. Engine `liquidity_absorption_stage_a06.py`; tests `tests/test_liquidity_absorption_stage_a06.py`; protocol `discussions/Liquidity_Absorption/023_claude_a06_deribit_entry_context_protocol.md`; outputs `06/`.

- **EXT_full_period_deribit.ipynb** — LA-EXT: the whole module (A02 → A02b → A03 → A03b → A03c → A03d → A04 → A05a → A06) re-run **venue-native on Deribit** perpetual prints from 2024-01-01 to the latest downloaded day, engines unchanged. Pre-registered kill gate on A03b forward 1:1 taker with Deribit fees, BTC/ETH × two holdout halves; Jul–Sep 2026 reported separately as in-sample; bridge vs the 452 Binance signals. Orchestration `liquidity_absorption_extended.py`; tests `tests/test_liquidity_absorption_extended.py`; protocol `discussions/Liquidity_Absorption/024_claude_extended_deribit_full_period_preregistration.md`; outputs `EXT/`.

## Deribit historical trade QA and LA-A05b millisecond context

The Deribit historical archive downloader (`tools/download_deribit_history_trades.py`) writes individual reported trade prints for **dated futures and perpetuals combined** under `DERIBIT_DATA_ROOT/history/trades/{BTC,ETH}/future/YYYY/YYYY-MM-DD.parquet`. Trades may share the same millisecond timestamp. The new **read-only** `tools/validate_deribit_history_trades.py` audits file completeness/availability, per-instrument trade-sequence gaps, invalid numeric/timestamp fields, duplicate IDs (optional exact mode), and overloaded millisecond print buckets. Local QA cannot prove that the timestamp-paginated downloader captured every exchange trade; `PASS_WITH_LIMITATIONS` is not a remote completeness certificate. A partially downloading day remains `.tmp`, not a completed Parquet file.

Use the validator on the full completed range, for example:

```bat
python -m unittest discover -s tests -p "test_deribit_history_validator.py" -v
python tools\\validate_deribit_history_trades.py --currencies BTC ETH --kind future --start 2024-01-01 --end 2026-10-19 --out notebooks\\Liquidity_Absorption\\05b\\validator_all_history
```

**A05b microstructure notebook:** `05b_deribit_millisecond_microstructure.ipynb` reuses 452 frozen A05a signals and their first-touch / second-touch **one-minute candle close** times. Its independent Deribit **BTC-PERPETUAL and ETH-PERPETUAL** windows (±60s) preserve only occupied 1/10/100/1000-ms trade buckets with aggressor count/amount imbalances and executed-price movements. Anchors are *Binance candle times*, **not** precise millisecond touch times; Deribit and Binance markets have different price/level/queue states. No trading signals, strategy fills, collector code, or previous results are changed. Run:

```bat
python -m unittest discover -s tests -p "test_liquidity_absorption_stage_a05b.py" -v
jupyter lab
```

In Jupyter open `notebooks/Liquidity_Absorption/05b_deribit_millisecond_microstructure.ipynb` and **Restart & Run All**. The notebook runs focused daily QA for the original July–September 2026 signal windows and then the tick-resolution study. This can scan large existing Parquet files twice; it doesn't redownload. Share committed lightweight `05b/` results and executed notebook for review; keep oversized `05b_sparse_trade_buckets.csv.gz` in the local archive rather than forcing it through GitHub's file-size limit. Protocol: `discussions/Liquidity_Absorption/019_chatgpt_deribit_history_validator_and_a05b_microstudy.md`.

# LA 07 — W01: anatomy of the A03b 1:1 trades (winners vs losers)

`07_winner_anatomy_1to1.ipynb`, engine `src/data_engineering/modeling/liquidity_absorption_w01.py`, tests `tests/test_liquidity_absorption_w01.py`, protocol `discussions/Liquidity_Absorption/027_claude_w01_winner_anatomy_preregistration.md`, outputs `07/`.

Every EXT A03b 1:1 trade with a −240…+240 min per-minute panel from Deribit prints (price/high/low, taker flow, large-print flow, volume), A06 context (liquidations, premium, basis, ATM IV, skew, option flow) and the 18:00 ET session profile at entry. Binding H1: CVD toward the target in the first 15 min predicts the rest of the trade beyond price progress (landmark design, MINING < 2025-07-01, CONFIRM after). Also: landmark exit rules (+15 min losing cut, low-CVD cut), burst onset study, pre-entry and landmark feature screens. Needs EXT outputs; the panel parquet goes to `DERIBIT_DATA_ROOT\la_w01\`. `LAW01_SYNTHETIC=1` for a smoke run.

## LA-08 — Original 1:1 strategy with Claude's Deribit methodology

**Notebook:** `08_original_strategy_claude_method_bridge.ipynb`; engine `src/data_engineering/modeling/liquidity_absorption_bridge_08.py`; tests `tests/test_liquidity_absorption_bridge_08.py`; protocol `discussions/Liquidity_Absorption/029_chatgpt_la08_original_strategy_claude_method_bridge.md`.

Tests our *unchanged* original EXT A03b forward 1:1 entry, structural target, mirrored stop and 240-minute trade-print exits with Claude V2's **signal-close** CVD divergence and causal **18:00 New York** volume-profile target-availability gates (CVD only, profile only, both) alongside unfiltered A03b. A separate reference is Claude's full CVD/profile/VD18 stop/TRAIL_1M strategy, **whose target, stop, and management differ** and therefore cannot be a one-factor control. The same 2,564 EXT candidate IDs must match; the notebook fails safely on drift or invalid gross 1:1 geometry. Reprices EXT/V2 already replayed trades under Claude's frozen Deribit taker cost assumption without re-reading the historical tick archive. Signal-time filters never examine future outcomes.

```bat
git pull
python -m unittest discover -s tests -p "test_liquidity_absorption_bridge_08.py" -v
jupyter lab
```

Open the new notebook and **Restart Kernel & Run All**, then commit the executed notebook and lightweight `08/` outputs to GitHub main. Inputs: `EXT/03b/03b_1to1_events.csv`, `V2/V2_signals.csv`, `V2/V2_events.csv`. Results are *historical exploratory attribution on previously inspected data*, not new confirmation or portfolio equity. Censored outcomes are kept distinct; paired results exclude unresolved trades.

## LA09: observable level-source → confirmation → first touch → second touch → reaction

Open [`09_level_lifecycle_anatomy.ipynb`](09_level_lifecycle_anatomy.ipynb) to follow the **original 452 Binance A02 price-only candidates**, joined to the original A03b **fixed 1:1** outcomes (A03c maker-touch only as unfilled eligibility metadata). A02's five-minute pivot is **not available until 10 minutes after its source candle closes**; prior-day highs/lows become available at the next UTC midnight. First touches whose candles close *exactly at availability* are specifically flagged because their actual intraminute touch time is unknown.

The read-only engine [`liquidity_absorption_stage_a09.py`](../../src/data_engineering/modeling/liquidity_absorption_stage_a09.py) uses existing Binance USDT-M 1-minute Parquet files, and eight fixed market phases: the 60 minutes before source, source→availability, availability→first touch, first→second touch, then +1–5, +6–15, +16–60 and +61–240 minutes after the signal. It also aligns **four different clocks** (source / actual level availability / first touch / second touch), with explicit NaN level-specific features before availability, missing-minute flags, and post-exit market observations clearly marked as non-P&L. No current Deribit historical download or original simulator is modified.

Run after `git pull`:

```bat
python -m unittest discover -s tests -p "test_liquidity_absorption_stage_a09.py" -v
jupyter lab
```

Open the 09 notebook in your `deribit-data` kernel and **Restart & Run All**, then commit the notebook and generated `09/` reports to `main`. Protocol: [`030_chatgpt_a09_observable_level_lifecycle_protocol.md`](../../discussions/Liquidity_Absorption/030_chatgpt_a09_observable_level_lifecycle_protocol.md). A09 is **retrospective descriptive research**, not an independently confirmed entry filter, trading strategy, or actual A03c maker fill.

## LA10 — unselected, causally observable level universe

Open `10_unselected_level_universe.ipynb` after `git pull`. It studies **every active A02 reference episode** (BTC/ETH prior-day support/resistance and most recently confirmed five-minute pivots), including levels with **no valid rejection touch**, **only an initial rejection**, or a **confirmed second-touch signal**. The reference clock respects the 10-minute pivot confirmation delay or prior-day next-midnight availability. A second study clock starts at the first recognized touch and records invalidation, reference replacement, 90-minute retest expiry, valid second touch or archive-end censoring. **A hard replay gate requires the generated signal events to match all 452 frozen A02 events exactly**, otherwise the run aborts and publishes no results. No trader P&L or maker fills are inferred for untouched levels.

```bat
python -m unittest discover -s tests -p "test_liquidity_absorption_stage_a10*.py" -v
jupyter lab
```

Restart and run `notebooks/Liquidity_Absorption/10_unselected_level_universe.ipynb`; push the executed notebook and `10/` CSV/JSON/PNG artifacts to GitHub `main`. Requires existing local `BTCUSDT_um_1m.parquet` / `ETHUSDT_um_1m.parquet` through `BINANCE_UM_DIR`, and original `02/` frozen outputs. This is a descriptive *denominator* study, not a matching-adjusted or prospective predictive test. See `discussions/Liquidity_Absorption/032_chatgpt_la10_unselected_level_universe_protocol.md` for risk-set/censoring limitations and next matching protocol.
