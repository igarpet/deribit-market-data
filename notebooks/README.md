# Notebooks

One folder per research branch, matching `discussions/`. Notebooks are numbered per folder from `01`; each notebook writes its results (CSV/JSON) beside itself with the same number prefix.

| Folder | Content |
|---|---|
| [initial_research](./initial_research/) | Foundational analysis used by all branches: feature inventory, market/surface distributions, microstructure, cross-feature relationships, clock/expiry effects, displacement tests. |
| [HSMM](./HSMM/) | Regime feature contract, exploratory HSMM/HMM, volatility HAR benchmark and frozen spec, volatility-state HMM, regime null test. |
| [Volatility_Risk_Premium](./Volatility_Risk_Premium/) | Option pricing audit, forecast vs implied variance, executable option P&L. |

## Mapping from the old global numbers (renumbered 2026-10-08)

| Old | New |
|---|---|
| `01`–`06` (root) | `initial_research/01`–`06` (same names) |
| `HSMM/06b_displacement_tests` | `initial_research/07_displacement_tests` |
| `07_regime_feature_contract` | `HSMM/01_regime_feature_contract` |
| `08_hsmm_exploratory` | `HSMM/02_hsmm_exploratory` |
| `09_factor_augmented_hsmm` | `HSMM/03_factor_augmented_hsmm` |
| `10_joint_sdf_hmm` | `HSMM/04_joint_sdf_hmm` |
| `11_full_information_price_model` | `HSMM/05_full_information_price_model` |
| `12_volatility_har` + `12_*` outputs (Module 12) | `HSMM/06_volatility_har` + `06_*` (incl. `06_frozen_spec_v02.json`) |
| `13_volatility_hmm_prototype` + `13_*` (Module 13) | `HSMM/07_volatility_hmm_prototype` + `07_*` |
| `14_btc_s2_har_garch_null` + `14_*` (Module 14) | `HSMM/08_btc_s2_har_garch_null` + `08_*` |
| `15_vrp_pricing_audit` + `15_*` | `Volatility_Risk_Premium/01_vrp_pricing_audit` + `01_*` |

Only file names and the paths in notebook code changed. Notebook prose, saved cell outputs, `src/` docstrings and discussion files may still use the old numbers. A note at the top of each renumbered notebook gives its former number.
