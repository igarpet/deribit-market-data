# HSMM research branch

This folder contains the complete latent-regime / HSMM research line developed after the general exploratory notebooks.

## Sequence

- `06b_displacement_tests.ipynb` — displacement-anchor diagnostics.
- `07_regime_feature_contract.ipynb` — causal regime feature contract.
- `08_hsmm_exploratory.ipynb` — first HSMM exploratory fit.
- `09_factor_augmented_hsmm.ipynb` — factor-augmented HSMM.
- `10_joint_sdf_hmm.ipynb` — joint SDF/HMM test.
- `11_full_information_price_model.ipynb` — directional-return benchmark used to test whether the regime thesis earns a mean-return role.
- `12_volatility_har.ipynb` — non-switching physical-volatility benchmark and frozen confirmation specification.
- `13_volatility_hmm_prototype.ipynb` — volatility-state HMM switching gate.
- `14_btc_s2_har_garch_null.ipynb` — BTC S2 continuous-volatility no-regime null.

Generated Module 12–14 CSV/JSON artifacts are stored here beside the notebooks that produced them.

## Current research status

The branch is still research, not a production regime model. Module 12 provides the strongest positive forecasting evidence. Module 13 found no broad cross-asset regime evidence; BTC S2 30m was the sole candidate advanced to Module 14. Module 14 rejected its first stationary continuous-volatility null conditionally, but selection and slow-drift concerns remain, so no HSMM duration model is yet authorized.

Related research debate: `../../discussions/HSMM/`.
