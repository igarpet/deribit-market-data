# RD01-b — Residual and causally confirmed observable regimes

This module tests whether a horizon-specific, tree-discovered **residual regime correction** improves next-5/15/30/60-minute signed-return forecasts beyond continuous Ridge and Ridge with explicit feature interactions. The primary horizon is **30 minutes**. The return density uses RD01's common Student-t(6) scale and **trailing-RV proxy**, not the AM-T-HAR-Blend variance forecast. RD01-b does **not** change AM-T-HAR-Blend.

Run `notebooks/Regime_Discovery/01b_residual_regime_validation.ipynb` from your `deribit-data` Jupyter kernel after pulling the repository. It reuses the local CS04-A Parquet feature store under `DERIBIT_DATA_ROOT/research/cs04a_feature_store`. No network downloads. Synthetic smoke testing: `RD01B_SYNTHETIC=1` in the kernel environment; synthetic outputs go to a temporary directory and must not be pushed.

Folds and training/censoring rules are preregistered at `discussions/Regime_Discovery/001b_chatgpt_residual_regime_preregistration.md`. Tree structures are trained only on late STRUCTURE days using a model fitted on earlier STRUCTURE days. Return corrections and continuous benchmarks use separate CALIBRATION days. Validation never refits anything. Confirmed labels require three observed consecutive five-minute raw labels and initialize at each new day or gap.

After the real run, commit the aggregate CSV and JSON results from this folder; do **not** commit the raw CS04-A feature store. The screening is historical exploratory and does not establish independently verified regimes or a trading strategy.
