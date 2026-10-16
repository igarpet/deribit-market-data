# Predictive_State_HMM 01 — volatility memory states, futures AND options

**First experiment:** Are latent volatility states inferred from two independently validated *feature-family* memories (`ema_30m`, `ema_240m`, VD12) useful beyond smooth continuous memory for future 60-minute native q99-onset probabilities? If yes, are their durations genuinely different from geometric HMM predictions? Outcomes are explored on spot history, while a **separate** 30-minute futures-direction/cost scenario uses actual Binance USD-M perpetual contract candles. No Deribit option-price claim can be made without matched historical options quotes.

Protocol: `../../discussions/Predictive_State_HMM/001_chatgpt_psh01_preregistration.md`.

The module reuses `volatility_lifted_memory` and `latent_regime_tournament` utilities explicitly approved for this new branch. It does not update existing HSMM/VD19 predictions or collector tables. Filter states use observations at the origin or earlier (the EMA emissions themselves use strictly earlier bars); TEST is never used to fit any HMM, transformation or outcome response.

From repo root on Windows CMD:

```bat
cd /d D:\Projects\deribit_market_data
conda activate deribit-data
python -m pytest -q tests\test_predictive_state_hmm.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks\Predictive_State_HMM\01_predictive_state_hmm.ipynb
```

Data inputs (read only): `DERIBIT_DATA_ROOT/history/price_5m/<ASSET>.csv.gz` (Binance spot) and Binance USD-M `<BINANCE_UM_DIR>/<ASSET>USDT_um_1m.parquet`, or `<parent of DERIBIT_DATA_ROOT>/Binance/` by default. Full historical processing capped before 2026-10-17 UTC; historical TEST was previously studied in other research and is **exploratory, not clean confirmation**. Missing perpetual file produces `INSUFFICIENT_PERP_ROWS`, never spot-price-substituted simulated execution.

Notebook outputs under `01/`: `01_forecast_scores.csv`, `01_state_profiles.csv`, `01_transition_matrix.csv`, `01_duration_hazards.csv`, `01_coverage.csv`, `01_perp_economics.json`, `01_decision.json`, `01_manifest.json`. Commit those along with the executed notebook for review. No raw-per-candle labels or local feature caches are committed.

**Interpretation limits:** Expected 12bp round-trip hypothetical perp cost is an assumption and not an observed Deribit fill; no funding, position constraints or liquidation risk are evaluated. The HMM is **not** an HSMM. Stage A uses only vol memories, not the statistically significant signed-flow / jump-hazard candidate modules, until a new test is registered. No signal is deployable.