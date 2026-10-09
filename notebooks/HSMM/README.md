# HSMM research branch

The latent-regime / HSMM research line built on `../initial_research/`. Discussions: `../../discussions/HSMM/`.

## Sequence

| Notebook | Former | Content |
|---|---|---|
| `01_regime_feature_contract.ipynb` | 07 | Causal regime feature contract |
| `02_hsmm_exploratory.ipynb` | 08 | First HSMM exploratory fit |
| `03_factor_augmented_hsmm.ipynb` | 09 | Factor-augmented HSMM |
| `04_joint_sdf_hmm.ipynb` | 10 | Joint SDF/HMM test (framework v3.2) |
| `05_full_information_price_model.ipynb` | 11 | Directional-return benchmark |
| `06_volatility_har.ipynb` | Module 12 | Non-switching volatility HAR benchmark; frozen confirmation spec `06/06_frozen_spec_v02.json` |
| `07_volatility_hmm_prototype.ipynb` | Module 13 | Volatility-state HMM switching gate |
| `08_btc_s2_har_garch_null.ipynb` | Module 14 | BTC S2 continuous-volatility no-regime null |

Outputs `06_*`, `07_*`, `08_*` (CSV/JSON) live in the subfolders `06/`, `07/`, `08/`. `06/06_frozen_spec_v02.json` is the binding spec; `06/06_frozen_spec.json` is the superseded v0.1 spec and must not be used for confirmation.

## Status

Research, not a production regime model. HSMM 06 holds the strongest positive evidence (30–60m and to-settlement variance is forecastable); its binding confirmation uses `06/06_frozen_spec_v02.json` after 30 post-freeze sessions (about 6 Nov 2026). HSMM 07 found no broad cross-asset regime evidence; BTC S2 30m was the only candidate. HSMM 08 rejected its first continuous-volatility null only conditionally (selection and miscalibration concerns), so no HSMM duration model is authorized yet.
