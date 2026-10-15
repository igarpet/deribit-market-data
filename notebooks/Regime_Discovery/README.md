# RD01 — Observable Regime Discovery

This is the **new observable-regime research line** (RD01), not the earlier `Return_State_Discovery/01` Bayesian HDP-HSMM experiment.

RD01 compares a 27-cell momentum × aggressive-flow × spot/futures-basis grid against an eight-leaf supervised regression tree, on exactly matched, causal five-minute Binance spot / perpetual observations. It reports return directions and signed-return density NLL, plus state frequencies, episode durations, exit hazards, transition matrices, and stability across folds. A direct Ridge return-mean model and a no-regime Student-t density are mandatory controls.

RD01 uses the local CS04-A feature store: `DERIBIT_DATA_ROOT/research/cs04a_feature_store/<ASSET>_features_and_forward_magnitude.parquet` for BTC/ETH/SOL/XRP. Run CS04-A first. **No download needed.** The first 80% of historical complete days only are used in the three RD01 expanding development folds; last 20% are excluded from model selection. This history has been seen in other project research, so every conclusion is exploratory.

```bat
cd /d D:\Projects\deribit_market_data
git pull origin main
conda activate deribit-data
python -m pytest -q tests\test_observable_regime_discovery.py
jupyter lab
```

Open `notebooks/Regime_Discovery/01_observable_regime_discovery.ipynb` and run all. The notebook writes **13 aggregate-only CSV/JSON outputs** into `notebooks/Regime_Discovery/01/`; push those files to GitHub for review. Raw per-candle price/feature/label data and individual episode timestamps never leave the local feature store. `RD01_SYNTHETIC=1` is a synthetic-only end-to-end test, writing into a temporary directory.

**Important scope:** `log_rv24h` in the CS04-A store is used only to construct a trailing historical variance proxy for normalization. It is **not the AM-T-HAR-Blend point-in-time volatility prediction**. RD01 therefore **does not test whether a regime improves AM-T-HAR-Blend**. That matched comparison requires an additional RD03 density-evaluation module. Do not change AM-T-HAR-Blend or interpret selected historical probabilities as live signals.

[Research protocol](../../discussions/Regime_Discovery/001_chatgpt_observable_regime_discovery_preregistration.md)
