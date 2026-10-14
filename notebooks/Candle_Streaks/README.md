# Candle Streaks — independent price-sign research

`01_running_sign_streaks.ipynb` discovers the **actual** lengths of all completed consecutive five-minute positive and negative close-to-close return runs across the existing Binance price history, alongside each live reached-k opportunity. No arbitrary fixed 10-bar samples, no truncation of long runs and no random-walk assumption built into the observed counts.

The first benchmark is an independent 50/50 sign sequence; the more important test asks whether streak length forecasts the **immediately next nonzero candle's sign** beyond observed marginal sign frequency and a first-order sign Markov baseline. Exact-length empirical continuation rates are shrunk to the Markov model; a smooth log-length hazard is also estimated. Censored gaps and zero returns are handled explicitly. Scores use chronological 60/20/20 full-UTC-day partitions and paired day bootstrap, plus an iid-sign permutation null for long runs.

Research protocol: `../../discussions/Candle_Streaks/001_chatgpt_candle_streaks_preregistration.md`.

Run `python -m pytest -q tests/test_candle_streaks.py` from repo root in the `deribit-data` environment, then execute notebook 01. It reads only the price files under `DERIBIT_DATA_ROOT/history/price_5m` and writes aggregate CSV/JSON under `notebooks/Candle_Streaks/01/`.

**Do not merge this into AM-T-HAR-Blend without independent forecast gains and matched-horizon calibration testing.** Historical evaluation is exploratory and previously observed prices are not a pristine independent holdout.

## Module 02 — Volatility-conditioned directional probability matrix

`02_streak_volatility_matrix.ipynb` asks whether the observed streak-length continuation effect varies with **causal volatility level and expansion/compression**. The current signed streak remains unbounded in length, while volatility is represented by the TRAIN-tertile intersection of past 24h realized variance and past 1h/24h relative variance (nine observable volatility conditions). Fits a TRAIN-smoothed **three-class DOWN/ZERO/UP** conditional probability matrix over `(current sign, full signed streak length, volatility level, volatility phase)`. It compares matrix probabilities with IID, sign-only, streak-only, sign+vol, and additive non-interaction models in VALID/TEST. Every row includes counts and all three probabilities, not just conditional-on-next-nonzero direction.

**Important:** Volatility features are observed trailing realized-variance summaries, *not* a new fit of AM-T-HAR-Blend. The module does not yet integrate a directional overlay into AM-T-HAR-Blend. The evaluation excludes incomplete UTC days and prohibits a future observation from crossing a split boundary. BTC/ETH are the preregistered primary assets; TEST monthly stability, complete-day paired bootstraps and volatility-alignment randomization must be reviewed before any integration experiment. See [`CS02 protocol`](../../discussions/Candle_Streaks/002_chatgpt_streak_volatility_probability_matrix_preregistration.md).

From the repo root run `python -m pytest -q tests/test_candle_streak_volatility.py` and execute notebook 02. The notebook saves compact aggregate CSV/JSON under `02/` and does not modify any previously fitted model, collection logic or database.
