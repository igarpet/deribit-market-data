# Liquidity Absorption — independent research branch

This branch studies the MNQ-inspired second-touch absorption/rejection mechanism on Binance BTC/ETH perpetual data, with five-minute context and one-minute execution context. Existing collectors and other research branches are unchanged.

- **LA 01**: `01_data_audit.ipynb` — raw input integrity and trade-to-candle reconciliation. Corrected LA-A01-0.1.2 audit passed locally.
- **LA 02**: `02_causal_features.ipynb` — 452 descriptive, *price-only* second-touch candidate events, enriched with rolling delta/zone quote flow. Does not score profitability.
- **LA 02b**: `02b_touch_flow.ipynb` — *same A02 candidate set*, but quote-volume and aggressor-side flow attributed separately to first-touch versus second-touch one-minute rejection candles. Adds normalized zone-visitation intensity and a historical pre-first-touch POC diagnostic. **No future returns, trade signals, fills, stops, or P&L.**

The A02b inputs are `notebooks/Liquidity_Absorption/02/02_price_only_candidates.csv`, `02_decision.json`, historical Binance USD-M perpetual candle Parquets and existing daily `aggTrades` Parquets. The module scans trades in bounded batches without storing additional raw tick datasets. Full output is under `notebooks/Liquidity_Absorption/02b/` after running locally.

Use the `deribit-data` conda kernel. From the repo root run `python -m unittest tests.test_liquidity_absorption_stage_a02b`; then run all cells in `02b_touch_flow.ipynb`. `DERIBIT_DATA_ROOT` or `BINANCE_UM_DIR` supplies the Binance data path.

This historical period is exploratory, not a fresh validation holdout. Event conditions remain provisional; no future information is allowed to select the price-only candidates or enter the first/second-touch flow windows. The trading-system payoffs, after-cost economics and prospective confirmation are separate later steps requiring owner-approved definitions.
