# Return State Discovery — independent research branch

This branch studies **latent states in return and volatility processes**, without imposing economic regime labels or a fixed market-state count. The underlying Binance five-minute price data are shared; prior branch outputs aren't imported as fitted labels.

- `01_hdp_hsmm_return_states.ipynb` — initial unsupervised HDP-HSMM on standardized **price-path features**, exploratory negative predictive result. Its numbered research dialogue remains immutable.
- `02_volatility_states_conditional_price.ipynb` — **new, independent hypothesis**: infer HDP-HSMM states using *volatility features only*, then examine subsequent **signed-return dynamics** using causal, pre-block state probabilities. Benchmarks are VD18 AM-T-HAR-Blend, recent Student-t and continuous-volatility Student-t. Reports conditional heavy-tail null, sign-flip null and posterior state-count uncertainty.

For Module 02, run `python -m pytest -q tests/test_hdp_hsmm_volatility_price.py`, then open notebook. It writes compact CSV/JSON to `notebooks/Return_State_Discovery/02/` and does not change collectors or retain raw price history. All historical results are **exploratory**, not prospectively confirmed, and no options strategy is approved.
