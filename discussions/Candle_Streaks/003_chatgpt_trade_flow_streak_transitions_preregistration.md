# CS03 — Historical aggressive trade flow and five-minute candle-streak transitions

**2026-10-14. HISTORICAL_EXPLORATORY.** Protocol frozen before real market data evaluation. No changes to collectors or AM-T-HAR-Blend.

## Research question and data compatibility

Our existing CS01–02b streaks are defined on **Binance spot five-minute close-to-close returns**. Keep that definition unchanged. Overlay already-downloaded **Binance USD-M perpetual one-minute klines** (`<ASSET>USDT_um_1m.parquet` from `tools/download_binance_um_klines.py`), which contain `open_time`, `quote_volume`, `taker_buy_quote_volume` and `trades`. Buyer-initiated executed notional = taker-buy quote volume; seller-initiated = total quote minus taker-buy quote. This is **executed trade-flow imbalance, NOT L2 OBI/resting liquidity**. Spot/futures basis and price-market differences are acknowledged. Report synchronized overlapping days per asset.

## Strict time semantics

- A Binance 1m kline is timestamped by **OPEN** time `m`; the interval is `[m,m+1m)`. Five 1m candles with opens `[t−5m, ..., t−1m]` aggregate into feature timestamp `t` (**five-minute CLOSE**). This timestamp agrees with `state_time` returned by the existing price downloader/VD14. Require precisely five distinct minute opens; duplicated opens, negative taker volume, taker buy > total, NaNs and missing minutes are not silently interpolated or filled.
- Five-minute `flow5_t=(2Σtaker_buy_quote−Σquote)/(Σquote)` and past-30m aggregate flow `flow30_t=(Σ6 signed_quote)/(Σ6 quote)`. Volume/trade intensity: `log1p(5m quote_volume)` and `log1p(5m trades)`. All feature timestamps are **≤ t** and no future bar is used. Rolling flow requires all previous 6 matched bars; volatility requires 1h and 24h contiguous spot return history through t.
- Spot current close-to-close `sign_t = sign(r_t)` is −1, 0, +1; live **exact streak age** `k_t` increments without any cap, resets on sign reversal, flat return or data gap. Flat current candles are included with `sign=0, k=0`, so the model can study streaks **starting after a flat candle**. Next target is sign of the immediately next contiguous spot five-minute return `r_{t+5m}`, not a later nonzero one.
- Outcomes conditional on current `sign_t≠0`: `continue` (same sign), `reverse_new_streak` (opposite sign), `stop_flat` (next exactly zero). Conditional on `sign_t=0`: `start_up_after_flat`, `start_down_after_flat`, `flat_stays`. Unknown next bar is censored; never count it as a reversal or flat.
- Require *complete UTC days* containing all 288 five-minute spot returns, valid exactly aligned futures flow and warmed-up features (using preceding raw days for warmup). Split the **intersection** chronologically into 60% TRAIN, 20% VALID, 20% TEST. Training targets never cross a split; a feature may use previously observed bars in an earlier split.

## Frozen forecasting comparisons

Multinomial DOWN/ZERO/UP log loss (primary) and three-class Brier; secondary binary log loss for the immediate next nonzero return (where zero is only excluded for that diagnostic). All candidates use independent per-asset TRAIN-fitted multinomial L2 logistic regressions with fixed `C=0.25` and TRAIN-only feature standardization. Features are continuous; no arbitrary maximum streak age and **no pre-selected winning trade-flow threshold**.

1. `iid`: empirical TRAIN down/zero/up probabilities, Jeffreys smoothing.
2. `sign`: categorical current −1/0/+1 only.
3. `sign_streak`: sign plus signed and absolute `log1p(k_t)`.
4. `sign_streak_vol`: sign+streak plus `log` trailing 24h squared-return mean and 1h/24h ratio.
5. `sign_streak_flow`: sign+streak plus 5m/30m quote-weighted taker imbalance, each imbalance × current sign, log quote volume, log trade count.
6. **`sign_streak_vol_flow`**: all signed-streak, volatility and flow features.

Primary incremental gate: for BOTH BTC and ETH, `sign_streak_vol_flow − sign_streak_vol` must have negative TEST three-class log-loss delta with negative 95% upper *paired-day bootstrap* CI. Also require `sign_streak_vol_flow − sign_streak_flow` to have negative CI if interpreting the flow/volatility interaction, and inspect binary log loss before a directional claim. SOL/XRP secondary. Use 1000 bootstrap draws over complete UTC days, fixed seed 2604. Historical TEST reused across prior work: exploratory only, not prospective confirmation.

## Interpretation and controls

- Descriptive **flow tertiles** fitted on TRAIN only and age display buckets `0,1,2,3,4–5,6–8,9+` form a table of starts/stops, with counts by TRAIN/VALID/TEST. This table is for interpretation only: forecasting uses continuous features and the exact unbounded streak age. Report cells with counts to avoid reading rare groups as reliable.
- Negative control: 100 permutations of the entire flow feature vector between TEST origins of the same hour-of-day, with trained models frozen and sign/streak/volatility labels unchanged. This diagnostic breaks genuine time alignment but does not represent a fully specified market null or support standalone significance claims.
- Future leakage red flags: same-bar price/flow association is not a forecast; all predictive outcomes must be strictly later. Validate no minute overlap, no missing-bar forward fill, no cross-split target, no training-derived normalization on VALID/TEST. If data coverage is poor or BTC/ETH absent, report exclusions and do not issue a primary decision.
- A gain in three-class NLL may be exclusively a flat-candle prediction gain; require separate next-nonzero sign scores for any *directional* claims. Additional test must forecast the actual spot next-candle return distribution (preferably aligned to AM-T-HAR-Blend horizon) and assess option bid/ask economics before trading.

## Artifacts

New notebook `notebooks/Candle_Streaks/03_trade_flow_streak_transitions.ipynb`, modeling module `src/data_engineering/modeling/candle_streak_trade_flow.py`, tests and this immutable preregistration. Compact CSV/JSON saved in notebook `03/`. Do not commit parquet data or synthetic results, and do not modify AM-T-HAR-Blend, Deribit collector or earlier Candle Streaks notebooks.
