# Return State Discovery

Independent exploratory research branch initiated at the user's request on 2026-10-14.
Reuses AM-T-HAR-Blend (VD18) as the **ex-ante** predictive baseline on common Binance spot 5-minute prices; it does not assume K=2 or import a discovered volatility-regime model.

- `01_hdp_hsmm_return_states.ipynb`: finite weak-limit hierarchical DP explicit-duration HSMM, heavy-tailed multivariate emissions, non-overlapping price blocks, conditional no-regime null and strictly sequential return-density evaluation.
- Protocol: `../../discussions/Return_State_Discovery/001_chatgpt_hdp_hsmm_return_states_preregistration.md`.

The finite Kcap is a computational ceiling; the posterior occupied-state histogram is not itself proof of economic states. The negative control and prospective forecast gains determine whether latent price-state interpretation is warranted. No Deribit/tick collection, database changes, or trading authorization.
