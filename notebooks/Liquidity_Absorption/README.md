# Liquidity Absorption — independent research branch

Study the MNQ-inspired second-touch absorption/rejection mechanism on BTC/ETH Binance perpetual historical data. The 5m price regime provides context; 1m candles trigger and manage trades. Price-at-level cumulative delta and volume-profile growth are candidate explanatory signals, not assumed sources of profitable edge.

**01_data_audit.ipynb** runs the read-only input validation and writes its diagnostics under `01/`. It does not score signals, choose trading parameters, or execute orders. Reusable audit code lives in `src/data_engineering/modeling/liquidity_absorption_stage_a.py`.

Set `DERIBIT_DATA_ROOT` or `BINANCE_UM_DIR` in `.env` (or supply an explicit `--binance-root` via CLI). Run in the `deribit-data` conda environment.

Research discipline: no historical leakage, no full-sample normalization, no silent imputation, and no promotion of tested history into prospective confirmation. The actual order and price of stop/target fills must be represented conservatively.
