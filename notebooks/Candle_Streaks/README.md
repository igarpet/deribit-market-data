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

## Module 02b — UP/DOWN-only volatility-conditioned probabilities

`02b_binary_direction_volatility.ipynb` is an **independent binary sensitivity experiment** answering whether the volatility gains observed in CS02 survive after *immediately next zero-return targets* are excluded. **A zero bar still breaks a running sign streak**; no flat bar is skipped to a later nonzero candle. The 1h/24h causal realized-variance predictors, TRAIN-fitted nine volatility bins, and chronological complete-UTC-day splits are identical to CS02. The binary refits include 50/50, unconditional, current sign, signed streak length, sign+volatility, additive, and exact streak×volatility matrices. They are evaluated on matched binary origins, including a side-by-side comparison against CS02's separately refitted three-class probabilities normalized onto the nonzero outcomes. Separate TRAIN-frozen and all-history descriptive matrices report exact ages and sample counts.

Preregistered protocol: [`CS02b`](../../discussions/Candle_Streaks/002b_chatgpt_binary_direction_volatility_preregistration.md). Run `python -m pytest -q tests/test_candle_streak_volatility_binary.py`, then execute notebook `02b_binary_direction_volatility.ipynb` and push the aggregate `02b/` outputs. Importantly, this is **P(next UP | the immediately next return is nonzero)**, which is *not* an actionable unconditional UP probability; no AM-T-HAR-Blend code, database or collectors are changed.

## Module 03 — Binance USD-M aggressive trade flow versus spot streak transitions

`03_trade_flow_streak_transitions.ipynb` aligns the **existing downloaded** USD-M futures 1-minute `taker_buy_quote_volume`, `quote_volume` and `trades` with original Binance **spot** five-minute return streaks. It examines whether executed buyer/seller flow predicts the next completed candle's **continuation, opposite-sign reversal/new streak or flat termination**, and also models **fresh UP/DOWN starts after a current zero**. No full L2 history is presumed. Five-minute and trailing 30-minute signed trade-flow measures are known by the current bar's close and never measured on the target bar.

The notebook evaluates TRAIN-frozen IID, sign, sign+streak, sign+streak+volatility, sign+streak+trade-flow and combined volatility+flow logistic models on aligned complete UTC dates using three-class log loss and binary next-nonzero diagnostics. Interpretative tables are stratified by TRAIN-defined flow tertiles and unbounded-age *display* groups; input feature age stays continuous and uncapped. Metadata distinguishes spot prices from perpetual trade-flow information.

Protocol: [`CS03`](../../discussions/Candle_Streaks/003_chatgpt_trade_flow_streak_transitions_preregistration.md). Existing inputs: `DERIBIT_DATA_ROOT/history/price_5m/<ASSET>.csv.gz` and `BINANCE_UM_DIR/<ASSET>USDT_um_1m.parquet` (default: `<parent of DERIBIT_DATA_ROOT>/Binance`). No new downloads, collector changes or AM-T-HAR-Blend changes. Run `python -m pytest -q tests/test_candle_streak_trade_flow.py`, then execute notebook 03. Results under `notebooks/Candle_Streaks/03/`. Set `CS03_SYNTHETIC=1` only for an isolated synthetic smoke test; synthetic results are not market evidence.
