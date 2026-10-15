# RD01-c — Return-distribution asymmetry conditional on observable regimes

RD01-b did not establish incremental predictive return-**location** states. RD01-c instead tests whether a *shallow causally learned partition* identifies conditional **upside/downside asymmetry** that improves the full signed-return Student-t density over both (a) a common skew and (b) continuous covariate-dependent skew. The primary horizon is **30 minutes**, with 5/15/60-minute diagnostics. Variance and conditional mean are fixed across the direct-location symmetric/global/continuous/regime alternatives. There is **no fitted per-regime scale**.

Read the immutable [protocol](../../../discussions/Regime_Discovery/001c_chatgpt_conditional_skew_regimes_preregistration.md) before inspecting output. All evaluations reuse historical discovery folds and the RD01 trailing variance proxy, **not a point-in-time AM-T-HAR-Blend forecast**. Any improvement is exploratory. The old RD01 and RD01-b sources, collectors and AM-T-HAR-Blend remain unchanged.

From `D:\Projects\deribit_market_data` in your `deribit-data` environment:

```bat
python -m pytest -q tests\test_observable_regime_discovery_c.py
jupyter lab
```

Run `notebooks/Regime_Discovery/01c_conditional_skew_regimes.ipynb` top to bottom. It reads local private `<DERIBIT_DATA_ROOT>/research/cs04a_feature_store/<ASSET>_features_and_forward_magnitude.parquet` (BTC/ETH/SOL/XRP), no downloads. It saves **small aggregate CSV/JSON** results to this directory; push aggregate outcomes but **not** the per-origin Parquet files.

Set `RD01C_SYNTHETIC=1` in the Jupyter environment for a fast isolated synthetic smoke test (BTC, 30m only); generated synthetic results are written to a temporary folder and **must not** be committed as market evidence.