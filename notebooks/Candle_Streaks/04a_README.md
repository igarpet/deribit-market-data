# CS04-A — Feature-family discovery and magnitude labels

**Status:** code and synthetic regression validation only; no actual historical CS04-A empirical results in this commit.

CS04-A scans all 175 single/pair/triple combinations of ten feature families plus a shared baseline, on locally downloaded Binance **spot 5-minute** closes and **USD-M perpetual 1-minute** OHLCV and taker-buy data. The model gives next-5m three-class DOWN/ZERO/UP forecasts, with a conditional UP/DOWN diagnostic, and independent predictions of the **absolute net log-return size** over the *future 5m, 15m and 30m*.

The local feature store also retains future signed log returns, magnitude, squares, upside/downside and future path variance separately at these horizons. These are ***labels*** that must never appear in forecast predictors. The model does not forecast option prices or modify the current AM-T-HAR-Blend.

## Data / dependencies

Runs in the existing `deribit-data` Conda environment, which includes scikit-learn >=1.6, pandas, numpy, PyArrow, matplotlib/ipykernel/pytest. Requires:

- `DERIBIT_DATA_ROOT/history/price_5m/<ASSET>.csv.gz` with spot close and aligned 5m history.
- `<parent of DERIBIT_DATA_ROOT>/Binance/<ASSET>USDT_um_1m.parquet`, or `BINANCE_UM_DIR` override; one-minute file must include open/high/low/close, total quote volume, taker-buy quote volume and trades.
- BTC and ETH perpetual data both needed to construct contemporaneous cross-market peer features.

No Binance or Deribit download is invoked. Intraminute book depth and resting OBI are **not** claimed.

## Run on Windows (from the repo root)

```bat
cd /d D:\Projects\deribit_market_data
git pull origin main
conda activate deribit-data
python -m pytest -q tests\test_candle_streak_feature_matrix.py
jupyter lab
```

Open `notebooks/Candle_Streaks/04a_feature_combination_discovery.ipynb`, **Restart Kernel and Run All Cells**. The full run fits 176 models × 4 assets × 3 folds = **2,112 multinomial models** plus three magnitude Ridge models per fit, and can be computationally substantial. The notebook checkpoints each asset/fold/combo under `04a_checkpoint_<ASSET>.csv` so it may resume after interruption. Never reuse a checkpoint after changing the model specification, input history or time-fold definitions; delete stale checkpoints first.

To smoke-test the loader and notebook against synthetic data use `CS04A_SYNTHETIC=1` in the environment; to run a partial real-data workflow use `CS04A_QUICK=1`. Neither is a full empirical evaluation. Make sure both variables are unset/`0` for the registered historical scan.

### Output locations

- **Small shareable aggregate CSV/JSON**: `notebooks/Candle_Streaks/04a/` (`04a_fold_scores.csv`, `04a_ranked_combinations.csv`, `04a_coverage.csv`, `04a_quality.csv`, `04a_feature_catalog.csv`, `04a_manifest.json`). Push these small outputs and the notebook if you want the results analyzed.
- **Private local per-origin magnitude/feature cache**: `DERIBIT_DATA_ROOT/research/cs04a_feature_store/<ASSET>_features_and_forward_magnitude.parquet`. **Do not push this cache to GitHub**; it contains feature columns and future labels separated explicitly in naming. Never leak any future magnitude label into a forecast feature.

The rank #1 discovery model is NOT an independently validated candidate. Review the exact frozen [preregistration](../../discussions/Candle_Streaks/004a_chatgpt_feature_matrix_preregistration.md) and reproduce the fold boundaries before drawing conclusions. The previously inspected legacy TEST is excluded from selection, but the whole dataset has been involved in prior research, so entirely new market data is needed for final confirmation.
