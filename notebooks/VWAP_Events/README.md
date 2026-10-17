# VWAP Events notebooks

Pre-registration: `discussions/VWAP_Events/001_claude_branch_preregistration.md`.

| Notebook | Content | Status |
|---|---|---|
| `01_vwap_touch_event_study.ipynb` | Real VWAP touches (08:00 UTC session, 4h) vs offset and random-phase placebo touches; time-of-day-normalised realised volatility after the touch; Bonferroni gates on the test window | Run 8e40e73 (BTC, ETH, SOL, XRP): **no gate passes** — VWAP touches not special vs random-anchor VWAP (VWAP 002). v1.1 fixes -inf in secondary k=5 / M outputs |

## Data

Binance USD-M perpetual 1-minute klines, one parquet per symbol (`<ASSET>USDT_um_1m.parquet`), from `tools/download_binance_um_klines.py`. Default location: `<parent of DERIBIT_DATA_ROOT>/Binance`, or `BINANCE_UM_DIR`.

## Run (PC, Windows CMD)

```
python tools\download_binance_um_klines.py --months 24
jupyter nbconvert --to notebook --execute --inplace notebooks\VWAP_Events\01_vwap_touch_event_study.ipynb
```

Environment: `VE01_ASSETS` (comma list), `VE01_N_PLACEBO` (default 20), `VE01_SYNTHETIC=1` (smoke run to a temp dir). Assets without a parquet file are skipped.

Outputs go to `01/`: coverage, event counts, primary readout, decision, horizons, per-asset test readout, direction, and the manifest. All are small CSV/JSON.

## 02 — touch / cross / return-time statistics (descriptive, 2026-10-17)

`02_touch_cross_stats.ipynb` (`src/data_engineering/modeling/vwap_touch_stats.py`, tests `tests/test_vwap_touch_stats.py`): BTC and ETH perp 1m, session (08:00 UTC) and 4H VWAP. Counts touches (bars and episodes) and crosses; after each cross, minutes to the next touch, to a cross back, and to a touch after price first detaches (window resets censored, Kaplan-Meier medians). Same statistics for placebo anchors at unused times. No gates. Outputs in `02/`. `VE02_SYNTHETIC=1` runs a synthetic smoke test.

## 03 — VWAP fade scalp (preregistered, 2026-10-17)

`03_vwap_fade.ipynb` (`vwap_fade.py`, tests `tests/test_vwap_fade.py`): after a cross, fade the first bar stretched ≥ k·σ·√15 from VWAP; exit at target / VWAP touch / stop / 60 min; real vs hour-matched placebo anchors; gates G1 (VWAP-specific) and G2 (gross > 7 bp). Protocol `discussions/VWAP_Events/004_claude_ve03_preregistration.md`. Outputs in `03/`. `VE03_SYNTHETIC=1` for a smoke test.

VE03 run `a7431db` (VE 005): **NO_EDGE** — real −0.16 bp, real − placebo −0.14 bp; no cell above +0.8 bp gross. The pre/post split table mislabels all trades as pre (µs/ns day parsing; reporting only).

## 04 — extreme-displacement fade (preregistered, 2026-10-17)

`04_vwap_extreme_fade.ipynb` (`vwap_extreme.py`, tests `tests/test_vwap_extreme.py`): fade displacement ≥ 3/4/5σ·√15 or ≥ 0.5/0.75/1% from VWAP at any time (30-min cooldown); 144 exit cells; cell selected on DEV (before 2026-04-01), judged once on TEST vs hour-matched placebo anchors. Outputs in `04/`. `VE04_SYNTHETIC=1` for a smoke test.

VE04 run `6a27404` (VE 007): **NO_EDGE**. Branch closed.
