# RD04 — Regime robustness and falsification

This module **does not discover more regimes**. It uses RD03's frozen observable and latent regime return-location overlays and exactly reproduces its archived VD18 AM-T-HAR-Blend predictive variance and evaluation losses before running historical stability checks.

**Inputs:** LOCAL `DERIBIT_DATA_ROOT/research/cs04a_feature_store/` (four feature/label Parquets), historical Binance spot price files, and committed VD18 `18_manifest.json` and RD03 `03_scores.csv`. No downloads. No modifications to collectors or baseline models. All reported historical periods have already been inspected elsewhere and are **not prospectively confirmatory**.

**Run from project root:**

```bat
cd /d D:\Projects\deribit_market_data
git pull origin main
conda activate deribit-data
python -m pytest -q tests\test_regime_robustness.py
jupyter lab
```

Then run `notebooks/Regime_Discovery/04_regime_robustness.ipynb` in the `deribit-data` kernel with no special environment flags. To test synthetic-only plumbing, set `RD04_SYNTHETIC=1` before launching Jupyter (outputs go to a temporary directory).

**Outputs:** Nine small aggregates written under `notebooks/Regime_Discovery/04/`. Commit those after the real historical run. The minute-by-minute pointwise scores and labels remain local/in memory, and are not written to the repo. Audit reproduces RD03 exact evaluation NLL/Brier/count grid or halts.

**Primary question:** Do observed CART or latent HMM/semi-Markov models improve 30-minute signed-return NLL beyond *both* exact frozen VD18 and simple continuous feature conditioning in BTC and ETH, in *both* September diagnostic and later evaluation? Historical positive signs alone do not establish predictive regimes. Explicit-duration overlay is not jointly fitted HSMM. Fresh prospective data is required.

[Research protocol](../../../discussions/Regime_Discovery/004_chatgpt_regime_robustness_preregistration.md)
