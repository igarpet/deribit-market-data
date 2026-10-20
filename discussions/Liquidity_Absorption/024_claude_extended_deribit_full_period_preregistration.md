# Liquidity Absorption 024 — Pre-registration: the whole module, venue-native on Deribit, 2024-01-01 → latest

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-20 |
| **Branch** | Liquidity_Absorption |
| **Responds to** | Ivan (2026-10-20): "test from 01-01-2024 until the latest data available, the whole research module"; LA 022 §3–§4 (holdout proposal and kill gate) |
| **Status** | Pre-registration. Written and pushed before any real-data run. Code: `liquidity_absorption_extended.py` (LA-EXT-0.1.0); notebook `EXT_full_period_deribit.ipynb`; outputs `notebooks/Liquidity_Absorption/EXT/` |

## 1. Design decision: venue-native Deribit

Signals, bars and the execution tape are all built from the downloaded **Deribit BTC-PERPETUAL / ETH-PERPETUAL prints**. Three reasons:

1. The A02 zone is ±2 bp around a level. Binance and Deribit prices differ by the same order, so Binance-derived levels cannot be executed on Deribit prints without changing the engines (barrier translation), which the frozen design forbids.
2. Binance aggTrades for 2024–2026 are not on disk (about 50–80 GB), and Binance is not available to Ivan (MiCA, July 2026).
3. Deribit is the venue Ivan can actually trade.

This is LA 022 §3(b) (the venue-native arm), promoted to primary. Consequence: the signals are **not the same events** as the 452 Binance signals. §5 measures the overlap on Jul–Sep 2026.

## 2. Inputs built from the prints (resumable per day, under `DERIBIT_DATA_ROOT\la_ext_venue`)

| File | Content |
|---|---|
| `aggTrades/<SYM>/<SYM>-aggTrades-<day>.parquet` | time_ms, price, qty = USD amount / price, is_buyer_maker = (taker sold); perpetual only, ordered by (time, trade_seq) |
| `<SYM>_um_1m.parquet` | 1m OHLCV in the Binance kline schema (open = minute open; quote volume = USD; taker-buy quote = USD bought by takers); minutes without prints = previous close, zero volume; missing days = gaps |
| `price_5m/<CUR>.csv.gz` | 5m close of the **Deribit index** (last print's `index_price` in each bar), state_time = bar end; replaces Binance spot as the VD18 input |

Labels BTCUSDT / ETHUSDT are kept because the engines expect them; here they mean the Deribit perpetuals.

## 3. Stages: engines unchanged, wrappers without the 452-count checks

| Stage | What runs | Change |
|---|---|---|
| A02 | `run_a02` with `Config(start=2024-01-01, end=last day)` | period only (warm-up 3 days, already in Config) |
| A02b | `read_candidates` + `enrich_a02_candidates` | repo-path coupling removed |
| A03 | `EventSimulation` × FIXED/BE/TRAIL, `scan_event_tapes` | none |
| **A03b** | `MirroredStopSimulation.fixed_record` | none — **primary** |
| A03c | `TapeStudy` × REVERSE / MAKER_TOUCH / MAKER_THROUGH, `scan_tapes` | none |
| A03d | VD18 HAR_4 frozen fits (repo), `risk_stop`, taker and maker classes as in `build_results` | input = Deribit index 5m |
| A04 / A05a | primary cohort, landmarks, context panel | descriptive |
| A06 | Deribit derivatives context on the EXT cohort, 1 control per event | descriptive |
| A05b | not re-run | closed as null in LA 022 |
| A01 | replaced by `EXT_venue_coverage.csv` and the existing history validator | — |

## 4. Gate (binding; LA 022 §4 adapted to the venue)

- **Primary:** A03b forward 1:1 taker, FIXED, closed trades (TARGET/STOP/TIMEOUT).
- **Net costs:** fees repriced to Deribit (taker 5 bp/side; the engines' 6 bp replaced). The assumed adverse execution bps stay as frozen (entry 0.5, target 0.5, stop 1.5, timeout 0.5). Funding is not modelled, as in A03.
- **Holdout:** every signal **outside** 2026-07-01 … 2026-09-30. That quarter has been inspected since A02; it is reported separately as `INSAMPLE_JUL_SEP_2026`.
- **Halves:** H1 = signals before 2025-04-01; H2 = from 2025-04-01, excluding the in-sample quarter.
- **Pass:** mean net bp per closed trade $>0$ **and** its one-sided 95% lower bound $>0$, using a bootstrap over signal days ($B$ = 2000), in **each of** BTC-H1, BTC-H2, ETH-H1, ETH-H2. The holdout lower bound must also be $>0$ per symbol.
- **Otherwise:** `FAIL_CLOSE_LA` — the branch closes.
- **Everything else is descriptive:** A03 FIXED/BE/TRAIL, A03c reverse and maker (maker fills unverified, maker fee 0), A03d variants, A04–A06. No variant, filter or parameter is promoted from these tables on this data.

## 5. Bridge (descriptive)

For Jul–Sep 2026, count how many of the 452 Binance signals have a Deribit-native counterpart:
- same symbol and side;
- signal within ±5 min;
- level within 10 bp.

For the matched ones, show the agreement of A03b outcomes. Low overlap means the strategy's events are venue-specific. It does not invalidate the gate, which tests the Deribit-native strategy.

## 6. Known deviations and caveats

- **Tick sizes:** the engines' assumed ticks (0.1 / 0.01) are finer than Deribit's perpetual ticks. Stop rounding is marginally tighter than executable.
- **VD18 fit:** the HAR_4 fit's TRAIN sample covers part of 2024–2026, so the volatility forecasts (used only for A03d stop width) are partly in-sample. They are not P&L-fitted. The Deribit index replaces Binance spot as input.
- **Thinner tape:** ETH-PERPETUAL prints about 0.4 times a second. 1m bars have more zero-volume minutes than Binance, and A02's wick/zone tests see coarser candles.
- **Fills:** a stop is taken at the first print through it; maker fills are eligibility only, as in A03c.
- **Repo test:** the existing `test_liquidity_absorption_stage_a02b` suite fails in my environment on unmodified code (`Missing 1m candle within event`). It is unrelated to this change, and the EXT path builds complete days.

## 7. Validation before this turn

- **`tests/test_liquidity_absorption_extended.py`, 4 tests pass:**
  - day-input schema;
  - empty-minute fill and index bar alignment;
  - period labels and the cluster bootstrap;
  - Deribit fee repricing;
  - the full chain A02 → A04 plus evaluation on synthetic Deribit prints, including the 1:1 contract and resumability.
- **A06 tests:** still pass.
- **Synthetic smoke run:** the notebook runs end to end.

## 8. My prediction (opinion, before the run)

`FAIL_CLOSE_LA`. Gross per trade will be near zero in every cell, as in Jul–Sep 2026 (−2.5 ± 3.0 bp) — a near-martingale first-passage on symmetric barriers. Net will be about −12 bp after Deribit taker fees and the adverse-execution assumptions.

## 9. Running it (PC, Windows CMD)

Close the notebook in Jupyter first. The first run builds the venue inputs from about 1,000 days × 2 perpetuals (a few GB under `la_ext_venue`). After that, every stage is skipped on re-run unless `set LAEXT_FORCE=1`. A06 reads the options files for the whole period and is the slowest part; `set LAEXT_SKIP_A06=1` runs everything else first.

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest tests.test_liquidity_absorption_extended -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=86400 notebooks\Liquidity_Absorption\EXT_full_period_deribit.ipynb
git add notebooks\Liquidity_Absorption
git commit -m "LA EXT full-period run"
git push
```

If a single output file exceeds GitHub's 100 MB limit (most likely `EXT/03d/03d_risk_execution.csv` or `EXT/06/06_phase_panel.csv`), leave it local and push the rest.

## Questions for ChatGPT

1. Do you agree that venue-native is the only coherent way to run A02's ±2 bp zones on Deribit?
2. Is the 2025-04-01 half split acceptable? It is roughly the middle of the holdout signal count.

## Decisions needed from Ivan

1. Approve this pre-registration and gate as binding before running.
