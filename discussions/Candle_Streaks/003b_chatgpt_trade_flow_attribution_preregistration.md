# CS03b — Signed aggressive flow vs market activity in five-minute streak transitions

**Protocol established October 15, 2026 before executing CS03b against real downloaded data. HISTORICAL_EXPLORATORY.** CS03's identical underlying dates, labels, and fitted full model have already been examined, so no part of CS03b's historical TEST is a pristine independent test. No permission to change collectors or AM-T-HAR-Blend.

## Research question and frozen data contract

CS03's combined five-minute *spot* sign/streak, trailing volatility, and Binance USD-M futures *taker execution* variables improved unconditional three-class held-out log loss in BTC and ETH, with some ETH conditional directional gain. Yet CS03 mixed **signed taker imbalance** and **gross activity** (quote volume and trade count), so the effect cannot be attributed to buying/selling pressure. CS03b asks which component provides the information and whether it forecasts (a) reversals of an ongoing streak, (b) flat termination, or (c) restarting UP/DOWN after a flat origin.

Use the **same downloaded datasets** and preparation as CS03: `DERIBIT_DATA_ROOT/history/price_5m/<ASSET>.csv.gz` spot 5m close-to-close returns via VD14; `BINANCE_UM_DIR/<ASSET>USDT_um_1m.parquet` (default sibling Binance folder) with USD-M perpetual `open_time`, `quote_volume`, `taker_buy_quote_volume`, `trades`. Do **not** download new L2, trades, bars or raw prices. BTC/ETH primary; SOL/XRP secondary. Prices and executed flow refer to **different spot and futures markets**. Taker quote volume is *executed trade flow*, not resting-book OBI.

Each five-minute `state_time=t` includes **exactly the five USD-M one-minute candles opening at t−5m through t−1m**, with no fill for missing minutes. Features are known as of the completed price/flow bar at t; targets are the **immediately next** spot return at t+5m. Current sign in {−1,0,+1}; signed running streak length k is uninterrupted and uncapped; current zero means age 0 and can initiate a new run on the next candle. Missing/gapped bars and incomplete UTC dates cannot create artificial transitions. Reuse CS03 `one_minute_futures_flow`, `read_futures_parquet`, `build_origin_panel`, `full_days`, and `split_panel` functions, and identical features/predictors. `full_days` requires 288 complete aligned five-minute rows including warmed-up price volatility and 30m signed flow. Split complete observed intersection dates into **60% TRAIN / 20% VALID / 20% TEST** by day, with no target crossing period boundaries. Fixed historical CS03 loader cutoff: `2026-10-14T00:00:00Z`; report true last available bar separately if needed.

## Frozen 4-way attribution, multinomial objective

All four fits use identical eligible TRAIN rows, three-class outcomes DOWN/ZERO/UP, TRAIN-only feature standardization, scikit-learn **multinomial LogisticRegression(newton-cholesky, C=.25, tol=1e-7, max_iter=120)**, following the numerical CS03 v0.1.1 correction. Convergence warnings remain fatal. Missing classes or nonfinite predictors are fatal. There is **no parameter sweep, refit on VALID/TEST, elastic net, tuned sampling weights, or post-hoc length truncation**. If a model fails, report it; do not silently tune it.

| Name | Input features (via CS03 raw_design) |
|---|---|
| `price_vol` | categorical sign dummies, signed/absolute log(1+exact age), lagged 24h variance level, 1h/24h variance ratio |
| `price_vol_signed` | `price_vol` plus quote-weighted taker imbalance over 5m and 30m and both flow×current-sign interactions |
| `price_vol_activity` | `price_vol` plus `log1p(5m quote notional)` and `log1p(5m trade count)` |
| `price_vol_both` | `price_vol` plus **both** sets; mathematically and numerically matches CS03 `sign_streak_vol_flow` |

Verify numerical equality of `price_vol` with CS03 `sign_streak_vol` and of `price_vol_both` with CS03 `sign_streak_vol_flow` on identical synthetic TRAIN rows. This is an ablation, **not** a revision to the original CS03 objective.

## Fixed outcome metrics and loss attribution

**Primary proper score**: unconditional immediate-next **three-class multinomial log loss** and Brier across all forecast origins, including current flat and next flat events. **Directional diagnostic**: proper binary UP/DOWN log loss and Brier **conditional on immediately next bar being nonzero** (not forward-jumping to the next nonzero bar). This conditioning event is unknown at the forecast decision instant; it is not actionable without a separate zero-return model.

Prespecified start/stop breakdowns use those **same frozen three-class predictions**, not new separately trained classifiers:

- `flat_vs_move`: binary P(next zero) / 1−P(next zero), scored over all **currently nonflat** origins; flat termination of ongoing streaks.
- `reversal`: P(next opposite sign | next nonzero, currently nonflat), scored over **currently nonflat origins with next nonzero**; continuing vs reversing.
- `start_up`: P(next up | next nonzero, currently flat), scored over **currently flat origins with next nonzero**; upward vs downward initiation after zero. Current flat and next flat origin counts are included in unconditional three-class evaluation; no claim from small conditional start samples.

Each diagnostic reports counts, dates, log loss and Brier; cells without at least **200 origin events on at least 20 distinct UTC days** are marked sparse, and no bootstrap gate may use them. Historical TEST descriptive tables summarize observed starts/stops by signed flow **TRAIN-only tertiles** and display-only exact-age groups (0, 1, 2, 3, 4–5, 6–8, 9+); these groups are **never model input truncations**. Forecasts are always frozen TRAIN-model probabilities; observed TEST frequency is never substituted into a prediction.

## Six frozen model comparisons and evaluation controls

All proper-score deltas take `first model − second model`; **negative** favors the first.

1. `price_vol_signed − price_vol`: is signed execution pressure helpful beyond price/variance?
2. `price_vol_activity − price_vol`: is gross activity helpful beyond price/variance?
3. `price_vol_both − price_vol`: does the original complete flow family help?
4. `price_vol_both − price_vol_signed`: does activity add after signed flow?
5. `price_vol_both − price_vol_activity`: does signed flow add after activity?
6. `price_vol_signed − price_vol_activity`: direct component comparison (descriptive, not nested).

Use **1,000 full-UTC-day paired bootstrap resamples (seed 2605)** for each VALID/TEST loss comparison; report equal-day and origin-weighted differences, CI_2.5%, CI_97.5%, number of paired days and minimum/maximum daily event counts. The registered **signed attribution candidate** requires, **for both BTC and ETH**, negative TEST 95% upper confidence bounds for (1) and (5) on **unconditional three-class log loss**. A **signed conditional directional candidate** additionally requires negative upper bounds for those same pairs on conditional binary log loss. An **activity attribution candidate** separately requires negative TEST upper bounds for (2) and (4) on three-class log loss. SOL/XRP cannot rescue a failed BTC/ETH primary gate. Any missing primary asset or unscorable comparison blocks the respective gate. This is a **historical exploratory screening gate**, not a prospective significance claim; six comparisons and multiple outcome metrics create multiplicity, so any positive gate requires wholly new data and proper multiplicity handling before confirmation.

Extra diagnostic: freeze all TRAIN fit parameters and shuffle the **signed flow feature pair** (5m/30m) or **activity feature pair** together across TEST observations within UTC hour-of-day (100 repetitions, seed 2605), leaving sign/streak/volatility/outcomes fixed. The shuffle is a label-alignment check, **not** a generative market null or valid causal p-value. Signed interactions are recomputed from permuted flow and unchanged origin sign.

## Fixed research outputs and unchanged components

Notebook `notebooks/Candle_Streaks/03b_trade_flow_attribution.ipynb` and module `src/data_engineering/modeling/candle_streak_flow_attribution.py` with tests and this immutable protocol. Save **aggregate-only** files under `notebooks/Candle_Streaks/03b/`: `03b_quality.csv`, `03b_coverage.csv`, `03b_scores.csv`, `03b_paired_day_bootstrap.csv`, `03b_transition_calibration.csv`, `03b_model_coefficients.csv`, `03b_component_shuffle.csv`, `03b_manifest.json`, `03b_decision.json`. No local raw prices, one-minute futures data, origin-level predictions or synthetic evidence are pushed. Keep CS01, CS02, CS02b, CS03, collectors, and AM-T-HAR-Blend **unchanged**. Study signal economic relevance and fresh prospective evaluation separately before any integration.
