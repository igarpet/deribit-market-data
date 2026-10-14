# Candle Streaks — independent price-sign research

`01_running_sign_streaks.ipynb` discovers the **actual** lengths of all completed consecutive five-minute positive and negative close-to-close return runs across the existing Binance price history, alongside each live reached-k opportunity. No arbitrary fixed 10-bar samples, no truncation of long runs and no random-walk assumption built into the observed counts.

The first benchmark is an independent 50/50 sign sequence; the more important test asks whether streak length forecasts the **immediately next nonzero candle's sign** beyond observed marginal sign frequency and a first-order sign Markov baseline. Exact-length empirical continuation rates are shrunk to the Markov model; a smooth log-length hazard is also estimated. Censored gaps and zero returns are handled explicitly. Scores use chronological 60/20/20 full-UTC-day partitions and paired day bootstrap, plus an iid-sign permutation null for long runs.

Research protocol: `../../discussions/Candle_Streaks/001_chatgpt_candle_streaks_preregistration.md`.

Run `python -m pytest -q tests/test_candle_streaks.py` from repo root in the `deribit-data` environment, then execute notebook 01. It reads only the price files under `DERIBIT_DATA_ROOT/history/price_5m` and writes aggregate CSV/JSON under `notebooks/Candle_Streaks/01/`.

**Do not merge this into AM-T-HAR-Blend without independent forecast gains and matched-horizon calibration testing.** Historical evaluation is exploratory and previously observed prices are not a pristine independent holdout.
