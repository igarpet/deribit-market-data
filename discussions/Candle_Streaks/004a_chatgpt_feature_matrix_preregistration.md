# CS04-A — Large feature-family subset scan with future return magnitude

**Date:** 2026-10-15. **Status:** historical exploratory / discovery only. Model and feature definitions frozen before any CS04-A historical market evaluation. **Existing CS03/CS03b TEST periods have already been inspected**, and must not be recycled as independent confirmation. No Deribit collector, option pipeline or AM-T-HAR-Blend modifications.

## Research goals

CS03b found small historical directional improvements from Binance USD-M perpetual *signed aggressive execution imbalance* beyond trailing spot volatility, but incremental probabilities remained near 50/50. CS04-A searches systematically for additive combinations of price, intra-candle, signed trade-flow, market activity, cross-market and clock features from existing local files. **Magnitude is a co-primary research output**, not an afterthought: forward signed log return, absolute net move, squared net move, upside/downside magnitude and realized path variance must be stored for horizons 5/15/30 minutes. The corresponding magnitude predictor is the TRAIN-fitted conditional **mean absolute net return**, scored out of sample; it is not a complete option-return density.

### Data and point-in-time contract

- Assets: BTC and ETH primary; SOL and XRP secondary. Binance **spot** five-minute close-to-close log returns from `DERIBIT_DATA_ROOT/history/price_5m/<ASSET>.csv.gz`, exactly via `markov_lift_distributions.load_returns`, with cutoff `2026-10-14T00:00:00Z`.
- Binance USD-M perpetual 1m klines already downloaded with `tools/download_binance_um_klines.py` in `<ASSET>USDT_um_1m.parquet`. Required input columns: `open_time,open,high,low,close,quote_volume,taker_buy_quote_volume,trades`; no fresh downloads. Futures timestamps represent **minute OPEN**. Precisely five minute candles `[t-5,t-4,t-3,t-2,t-1]` (minute offsets) form features known at completed five-minute timestamp `t`. Duplicates, incomplete minute buckets, impossible prices/volumes, and off-grid times are rejected, never forward-filled.
- Spot close and futures OHLC are different markets; book OBI or true tick-level executed trades are **not** present. Cross-market basis estimates are diagnostic and subject to exchange quote conventions.
- BTC uses same-timestamp ETH perpetual price/flow as its peer. ETH, SOL and XRP use same-timestamp BTC perpetual data as peer. **Peer timestamps must match exactly**; missing data invalidate the feature origin.
- All features (price, 1m path, volume, flow, basis, peer, calendar) use observations ending at or before the **current bar close**. No end-of-target-period features or rolling features that bridge gaps.
- Future outcome `r_{t,t+h}` is the *net sum of future spot five-minute **log returns*** for h=5,15,30 minutes. Require all intermediate bars and strictly subsequent times. Reject horizons crossing UTC day boundaries (also removes boundary-label crossings by construction). Do not equate realized path variance to square of net return; `sq_h` is **squared net return**; a future extension can separately preserve realized path variance.
- Current zero-return candles stay in class 0 and break streak age; all observed labels in {-1,0,+1} use the immediate next five-minute candle. No future zero filtering for the primary three-class fit.

## Frozen ten families

All models include the *same* baseline: sign dummies, unbounded signed/absolute `log1p(streak age)`, trailing 24h log variance, and past 1h/24h log variance ratio.

| Family | Additional predictors |
|---|---|
| `momentum` | current 5m log return, past 3-bar and 12-bar return means, current vs 3-bar mean |
| `candle_geometry` | futures body log-return, range/open, upper and lower wick fraction |
| `minute_path` | last one-minute log body return, within-bar sum of squared 1m log bodies, last-vs-mean 1m body return |
| `signed_flow` | 5m and 30m taker quote-notional imbalances and sign interactions |
| `flow_timing` | first-four-minute imbalance, final-minute imbalance, final-minus-early imbalance |
| `trade_activity` | log total 5m quote volume, log trade count, log average quote volume per trade |
| `flow_price_impact` | contemporaneous flow-minus-tanh(futures 5m return/current trailing vol), flow × futures price direction |
| `spot_futures_basis` | futures/spot close basis bps, past-only basis change, contemporaneous futures minus spot 5m log return |
| `peer_market` | contemporaneous peer futures 5m return and taker flow |
| `utc_calendar` | sine/cosine UTC 24h clock and seven-day weekday cycle |

The family comparison is **additive** in the model's linear predictor, but each listed family may contain predefined within-family transforms or sign interactions. No arbitrary nonlinear pairwise cross-family interactions in CS04-A.

All ten singleton, 45 pair and 120 triple family subsets are enumerated lexicographically from the stable family list, **175 combinations plus the baseline**. Subset size is not chosen from observed losses. There is **no unregistered hyperparameter sweep**.

## Model / magnitude evaluation

- Per asset/per fold, standardize all features using FIT rows only. Fit unconditional three-class logistic regression `newton-cholesky`, fixed `C=.25`, `tol=1e-6`, `max_iter=120`. Convergence failure is fatal; do not silently raise iterations or tune models during historical scanning.
- Independently fit Ridge regressions with `alpha=10` to predict **mean absolute future net log return** at h=5,15,30, in bps. Clip invalid negative magnitude forecasts to zero. Evaluate validation MAE in bps and MSE in bps², along with predicted vs observed mean absolute move and RMSE. Magnitude models are trained strictly on available FIT outcomes; no validation-fitted scaling or transformations.
- Primary directional proper score: unconditional three-class log loss; secondary three-class Brier and binary log loss of UP vs DOWN **conditional on immediate next nonzero**. Conditional binary loss is a diagnostic only, not an ex-ante deployable event probability.
- Every subset is scored against the **identical common baseline on identical evaluation origins** per asset and fold. Each result saves `asset, fold, combination, number_of_features, sample sizes, log loss, binary loss, forecast-zero, observed-zero, magnitude MAE/MSE/RMSE at each horizon, predicted and observed mean absolute return`.

## Time splits and selection bias

- Require complete UTC days with all 288 five-minute spot returns and valid future-independent inputs. Retain all consecutive full eligible days; exclude incomplete days. First 80% of eligible days are available for **discovery**; **last 20% (legacy, already-read TEST) are not used in this scan**.
- On that first 80%, form three expanding training folds, validation starting at 50%, 64% and 78% of discovery days. Each validation window lasts 8% of discovery days, minimum five; **one full day embargo** between FIT and VALID. Outcomes are censored at each UTC day boundary, so no future target can cross a fold split.
- Rank by mean *equal asset-fold weighted* three-class log-loss improvement relative to baseline. Also rank independently by 5m magnitude MSE gain; report 15m and 30m magnitude gains and share of favorable folds. No purported post-selection bootstrap p values from the ranked matrix.
- Historical results do not support choosing and declaring a 'confirmed' model. After exploration, freeze a **small set of finalists** and test on fresh, chronologically new data with multiplicity control, calibration and stability diagnostics. As a separate economic experiment, compare at the real option decision horizon with AM-T-HAR-Blend return-density NLL and bid/ask/fees. Do not change the deployed AM-T-HAR-Blend density based on the scan.

## Artifacts and reproducibility

- New module `src/data_engineering/modeling/candle_streak_feature_matrix.py` with defined registry and targets, regression tests `tests/test_candle_streak_feature_matrix.py`, and notebook `notebooks/Candle_Streaks/04a_feature_combination_discovery.ipynb`.
- GitHub-shareable **aggregate-only** notebook outputs under `notebooks/Candle_Streaks/04a/`: `04a_fold_scores.csv`, `04a_ranked_combinations.csv`, `04a_coverage.csv`, `04a_quality.csv`, `04a_feature_catalog.csv`, `04a_manifest.json`, optional partial aggregate checkpoints. Manifest labels synthetic/quick and records exact fold boundaries. Do not commit generated synthetic results or any raw/local per-origin prices.
- **Local-only reusable feature-label Parquet:** `DERIBIT_DATA_ROOT/research/cs04a_feature_store/<ASSET>_features_and_forward_magnitude.parquet`, including feature matrix and signed/absolute/squared/upside/downside/path-realized-variance/path-absolute-sum net log return at three horizons. Do not upload this file to GitHub: it is too large and includes forward outcomes that must never be accessed at forecast time. Keep strict segregation of feature columns from label columns.
- Notebook options: `CS04A_SYNTHETIC=1` temporary synthetic validation; `CS04A_QUICK=1` scan only first ten family subsets for integration smoke; normal real run with neither flag executes all 175 candidates plus baseline for BTC, ETH, SOL, XRP, and writes resumable aggregate score checkpoints. Never interpret synthetic/quick output as full-market CS04-A evidence.
