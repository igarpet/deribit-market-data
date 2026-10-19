# Liquidity Absorption — independent research branch

Investigate the MNQ-inspired second-touch absorption/rejection hypothesis on BTC/ETH Binance USD-M perpetual futures. Five-minute price context identifies preliminary levels; one-minute candle closes confirm candidate retests. Signed aggressor volume and trade-derived volume at price are candidate explanatory features, **not** presumed trading signals or evidence of iceberg orders.

- **LA 01** (`01_data_audit.ipynb`) audits historical input readiness in `01/`. Its corrected `LA-A01-0.1.2` output is signal-design ready; sampled 1m trade reconciliation is strong.
- **LA 02** (`02_causal_features.ipynb`) produces timestamped price-only two-touch candidates and contemporaneous 1m flow / 1bp-binned trade-profile features in `02/`. **No future returns, P&L, targets, stops or executable fills** are calculated. The module is `src/data_engineering/modeling/liquidity_absorption_stage_a02.py`.
- **LA 03** (not yet implemented) is reserved for executable fixed versus dynamic stop/target research, once the event and stop-recognition contract is approved.

Set `DERIBIT_DATA_ROOT` or `BINANCE_UM_DIR` in the repository `.env` and run in the `deribit-data` Jupyter kernel, with `pyarrow` available. Existing Binance data files are read-only; no additional downloads or modifications to Deribit collectors are necessary for LA 02.

**Caveats:** LA02's prior-day extremes and last confirmed 5m pivot are *proxies*, not the user's complete discretionary volume-profile level identification. The 2bp level tolerance is provisional, not MNQ's 2–4 tick width. No lookahead, no full-sample fitting, no reclassification of reused historical data as unseen validation. For details see `discussions/Liquidity_Absorption/003_chatgpt_a02_causal_feature_contract.md`.
