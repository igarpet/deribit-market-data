# RD02 — Latent return-regime tournament

RD02 compares a **three-state Student-t-emission HMM**, its **two-stage age-dependent semi-Markov duration overlay**, a temporally order-ablated state mixture, and matched continuous/observable-tree return-skew baselines on causal 30-minute forecast origins. Primary outcome: 30-minute **signed-return predictive density NLL**. Also evaluates 5, 15, and 60 minutes and 30-minute state durations/transitions. **AM-T-HAR-Blend is unchanged.** RD02 uses the same *trailing realized variance proxy* as RD01, not saved ex-ante AM-T forecasts; no direct AM-T improvement claim is possible yet.

Read [RD02 frozen research protocol](../../../discussions/Regime_Discovery/002_chatgpt_latent_regime_tournament_preregistration.md) before interpreting results. RD01-b and RD01-c did not establish a regime advantage over continuous forecasting; RD02 explicitly evaluates whether latent temporal ordering and duration give such an advantage.

### Local run

```bat
cd /d D:\Projects\deribit_market_data
git pull origin main
conda activate deribit-data
python -m pytest -q tests\test_latent_regime_tournament.py
jupyter lab
```

Execute `notebooks/Regime_Discovery/02_latent_regime_tournament.ipynb`. Input: `DERIBIT_DATA_ROOT/research/cs04a_feature_store/<ASSET>_features_and_forward_magnitude.parquet`; run CS04-A first. The normal notebook processes BTC, ETH, SOL, XRP and saves compact aggregate tables alongside this README.

| File | Contents |
|---|---|
| `02_scores.csv` | asset × fold × horizon × method return NLL, sign Brier, UP probabilities |
| `02_comparison_summary.csv` | equal asset-fold NLL gains versus continuous-skew baseline |
| `02_paired_day_bootstrap.csv` | exploratory whole-day paired comparisons |
| `02_state_diagnostics.csv` | HMM state occupancy priors/stay estimates, fit diagnostics |
| `02_return_response.csv` | CAL-fitted state location and skew, effective sample |
| `02_transition_matrix.csv` | fitted HMM state transition matrices |
| `02_duration_hazards.csv` | age-dependent semi-Markov hazards, exposures, exits |
| `02_filtered_episode_summary.csv` | aggregate VALID causal state episode durations, no timestamps |
| `02_validation_state_behavior.csv` | descriptive next-30-minute returns by filtered state |
| `02_observable_tree_definitions.csv` | supervised observable comparison rules and date boundaries |
| `02_coverage.csv` | evaluated assets/folds/samples |
| `02_manifest.json` | fixed configuration, data provenance, historical caveats |
| `02_decision.json` | screening verdict, explicitly not independent confirmation |

`RD02_SYNTHETIC=1` runs a temporary synthetic smoke scenario and **must not be confused with market evidence**. Do not upload local per-origin data, state probabilities, full decoded timestamped episodes or synthetic outputs to GitHub. HMM state IDs can permute across folds; compare prediction scores, not numeric labels.
