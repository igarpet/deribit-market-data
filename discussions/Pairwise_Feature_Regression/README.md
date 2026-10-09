# Pairwise Feature Regression — ChatGPT / Claude research dialogue

Independent exploratory branch for **same-five-minute-bar, one-feature-at-a-time OLS** matrices of slope and residual error. Notebook: `../../notebooks/Pairwise_Feature_Regression/01_pairwise_feature_ols_matrices.ipynb`.

- **001**: `001_chatgpt_pairwise_ols_v1.md` — v1 design, equation, sample masks, matrix interpretation, tests and open methodological questions.
- **002**: `002_chatgpt_windows_test_runner_fix.md` — resolve Windows delayed-DLL subprocess crash by running nine synthetic tests inside Jupyter's existing kernel.
- **003**: `003_chatgpt_v1_results_review.md` — executed v1 output review, structural aliases, return/IV/LOB relationships, sample masks and v2 considerations.
- **004**: `004_chatgpt_exact_five_minute_clock_v2.md` — exact 288 five-minute UTC clock regressions for every asset, matrix archives and small-N safeguards.
- **005**: `005_chatgpt_mixed_timezone_fixture_fix.md` — change pandas mixed-offset fixture so UTC normalization is tested inside the clock engine rather than in test setup.
- **Next turn:** `006` unless a newer numbered file is committed first.

Follow `../README.md`: one immutable discussion file per turn; evidence before interpretation; any proposed change requires explicit approval.
