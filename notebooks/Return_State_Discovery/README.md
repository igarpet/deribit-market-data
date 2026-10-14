# Return State Discovery — independent research branch

This branch studies **latent states in return and volatility processes**, without imposing economic regime labels or a fixed market-state count. The underlying Binance five-minute price data are shared; prior branch outputs aren't imported as fitted labels.

- `01_hdp_hsmm_return_states.ipynb` — initial unsupervised HDP-HSMM on standardized **price-path features**, exploratory negative predictive result. Its numbered research dialogue remains immutable.
- `02_volatility_states_conditional_price.ipynb` — **new, independent hypothesis**: infer HDP-HSMM states using *volatility features only*, then examine subsequent **signed-return dynamics** using causal, pre-block state probabilities. Benchmarks are VD18 AM-T-HAR-Blend, recent Student-t and continuous-volatility Student-t. Reports conditional heavy-tail null, sign-flip null and posterior state-count uncertainty.

For Module 02, run `python -m pytest -q tests/test_hdp_hsmm_volatility_price.py`, then open notebook. It writes compact CSV/JSON to `notebooks/Return_State_Discovery/02/` and does not change collectors or retain raw price history. All historical results are **exploratory**, not prospectively confirmed, and no options strategy is approved.

- `03_joint_price_volatility_states.ipynb` — **new bivariate joint price–volatility coupling hypothesis:** use AM-T-HAR-Blend's terminal return variance to normalize price displacement and TRAIN-fitted HAR log realized-variance forecast to measure its co-occurring RV surprise. Fit a Bayesian nonparametric Student-t HDP-HSMM on non-overlapping joint `(Z,U)` blocks; test whether sequential state probabilities add future signed-return NLL over continuous `Z × U` coupling, prior VD18, and order-ablated distributions. Also report joint-feature temporal NLL (distinct scoring units), posterior state-count uncertainty and conditional heavy-tail nulls. Frozen research protocol `../../discussions/Return_State_Discovery/003_chatgpt_joint_price_volatility_hsmm_preregistration.md`.

For Module 03, run `python -m pytest -q tests/test_hdp_hsmm_price_vol_coupling.py`, then execute notebook 03. Output is compact CSV/JSON under `03/`; the original Binance history and collectors are not modified. All historical results remain exploratory.
