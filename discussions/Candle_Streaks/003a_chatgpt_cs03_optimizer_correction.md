# CS03 v0.1.1 — numerical optimizer correction

**Implementation amendment after the failed real-data run on October 14, 2026.** This does **not** replace or change the frozen CS03 research protocol. The first executed notebook stopped at the `sign_streak_vol_flow` multinomial logistic fit: scikit-learn L-BFGS hit the configured 300-iteration ceiling (`tol=1e-7`). Features had already been standardized, so the warning was not caused by missing scaling.

## Changes
- Retain the identical three-class conditional model, feature design, chronological splits, TRAIN-only scaling and regularization strength `C=0.25` and tolerance `1e-7`.
- Replace only the numerical solver with scikit-learn's `newton-cholesky`, appropriate for a dense matrix of at most 12 feature columns and three outcomes. Use `max_iter=120` Newton iterations, not 120 L-BFGS iterations. Multinomial support requires scikit-learn >=1.6.
- Continue to fail loudly on any convergence warning or an exhausted solver iteration budget, rather than interpreting unfinished estimates as successful.
- Add tests for highly correlated trade-flow terms, rare zero-return outcomes, and agreement with a fully converged original L-BFGS optimizer.

## Local verification
- The 300-step L-BFGS ceiling was reproduced on a realistic large synthetic design; the new solver converged in nine iterations under the same objective and regularization.
- CS03 suite: 20 passed, 1 skipped (PyArrow not available in this execution container).
- No real-market CS03 scores can be inferred from the failed run. The notebook must be restarted and rerun against local historical data. Existing historical TEST periods remain exploratory, not prospectively independent.

No collector, AM-T-HAR-Blend model, dataset, fitted research parameter, statistical comparison, primary outcome or trading gate was changed.
