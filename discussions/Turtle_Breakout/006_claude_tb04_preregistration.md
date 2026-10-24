# 006 — Claude: TB 04 pre-registration — what separates winning from losing trades of the original turtle

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-25 |
| **Branch** | Turtle Breakout |
| **Responds to** | TB 03 run `f41eaf7`; Ivan, 2026-10-25: analyse the original strategy's trades with everything available, to see what differs internally between winners and losers. Design choices: discover on a wide universe and confirm on the four coins; use all data sources |
| **Status** | pre-registration + code (validated on synthetic data); awaiting Ivan's downloads and run |

## 0. TB 03 in one paragraph (run `f41eaf7`)

- **Benchmarks.** None of the 9 strategies beats the TB 02 benchmarks.
- **Luck.** The original daily/ATR strategy and the three 4h-candle versions beat random entries. Daily with engine sizing does not (random-entry share 0.055 and 0.027, against 0.0056). Last-4h is ruined by costs: about 20,600 trades and CAGR −99.6%.
- **Engine vs ATR.** No significant difference in any of the 6 comparisons. The engine sizes positions smaller at breakouts, which is when recent volatility is high, and that lowers exposure by about 30%.
- **4h vs daily.** Sharpe 1.27 vs 1.57, not significant. 4h has a −76% drawdown and very large costs.
- **Conclusion.** The original daily version stays the reference.

## 1. Question and the overfitting problem

The original strategy has 180 trades on BTC/ETH/SOL/XRP, 73 of them winners, and about 40 candidate features. Mining 40 features on 73 winners would produce several "significant" differences by chance. So the protocol separates discovery from confirmation:

| Set | Trades | Rules |
|---|---|---|
| **Discovery** | the same original rules on 26 other Binance USD-M USDT perpetuals, each over its full history (expected about 1,000–1,500 trades) | BNB DOGE ADA LINK DOT LTC BCH UNI NEAR APT ARB OP SUI FIL ATOM ETC INJ AAVE TON WIF HBAR XLM SEI TIA ENA WLD |
| **Confirmation** | BTC/ETH/SOL/XRP from 2020-10-15 (the TB 01 primary trades) | — |

- The discovery universe is the Pairs fOU list. It was also chosen in 2026, so it carries survivorship. That matters less here, because the question is *which trades* win, not whether the strategy pays.
- Trades still open at the end of the data are excluded, because their outcome is unknown.

## 2. Labels

- **Primary:** $R=\dfrac{\text{net return per notional}}{2\,\mathrm{ATR}/P_{entry}}$, the R-multiple net of fees, slippage and funding. **win** = $R>0$.
- **Secondary (reported):** big = $R\ge3$. The strategy's profit comes from a few large trends.

## 3. Features

**Entry features** use 5m bars ending **strictly before** the entry bar, and levels from completed days. All are dimensionless, so they pool across assets. There are 41:

| Group | Features |
|---|---|
| price / trend | breakout gap (ATR); distance from yesterday's close to the level (ATR); ATR %; ATR vs its 100-day median; 20-day range (ATR); 20-day return (ATR); 60-day log return; distance to MA50 / MA200 (ATR); above MA200; days since the previous breakout; time of day of the breakout; previous trade won; consecutive losses |
| flow / CVD | 24h volume vs the 20-day mean; 1h volume vs mean; taker-buy ratio 1h and 24h; 4h CVD / volume; 24h trade count vs mean; average trade size vs mean |
| funding | last rate; 3-day mean; 3-day vs 30-day z-score |
| volatility engine | engine 1-day σ × price / ATR; engine 60-min σ vs its 30-day median; 24h realised vol / ATR; largest 5m move in 24h / engine σ |
| OI / positioning | OI change over 24h and 7 days; OI value / 24h volume; top-trader long/short ratio and its 24h change; global long/short ratio; 24h taker long/short volume ratio |
| spot–perp basis | basis now; 24h mean; 7-day z-score |
| market | BTC 20-day return; BTC in breakout; breadth (share of the universe whose last close was at or above its 20-day high) |

**Notes on the feature groups:**
- **Engine.** Assets outside the VD18 fit use the baseline asset effect. The engine features are ratios, so that level offset largely cancels.
- **OI and basis.** OI and positioning start around December 2021 (Binance metrics), and the basis needs the spot download. Missing values are left missing: in the model they are imputed with the median plus a missing-indicator; univariate tests use the available trades only.

**Path features** are measured during the trade and are **descriptive only**; they cannot filter entries:
- MFE and MAE in R;
- days to MFE;
- R after 1, 3 and 5 days;
- taker-buy ratio over the first 24h and 72h;
- volume over the first 3 days;
- OI change over the first 3 days;
- mean funding during the trade;
- holding days.

## 4. Decision (binding)

**Univariate:**
1. **Discovery.** For each entry feature, compute the AUC (win vs loss) with a **calendar-month cluster bootstrap** (2,000 draws), because trades in the same month share the market move. The p-value is two-sided, and Benjamini–Hochberg FDR is applied over the 41 features. A feature survives at $q<0.10$. Features need at least 100 non-missing trades.
2. **Confirmation.** Each survivor is tested on the four coins: same direction, one-sided cluster-bootstrap $p$, with Holm correction over the survivors at 5%.

**Model:**
- Ridge logistic regression (λ = 1) on all entry features. Inputs are standardised and winsorised at 1/99% on discovery, median-imputed, with missing indicators. It is fitted on discovery only.
- **Pass:** confirmation AUC 95% lower bound > 0.5 **and** label-permutation p < 0.05 (200 refits on shuffled discovery labels).

**Verdict:** `ENTRY_SIGNAL_CONFIRMED` if at least one feature is confirmed or the model passes; otherwise `NOTHING_CONFIRMED`.

**Reported only:**
- the `big` label (same screen);
- per-group models;
- confirmation trades by model-score tercile (win rate, mean R);
- path features for winners vs losers;
- final win rate by R reached after 1/3/5 days (early-path table);
- exit reasons.

**Any trading rule** suggested by these results, such as an early exit, an entry filter or engine-based management, needs its own pre-registered test (TB 05) on held-out data.

## 5. Validation (synthetic)

`tests/test_turtle_trade_features.py`: 8 tests, all passing.

- $R\cdot2\mathrm{ATR}/P = $ net return, and win ⇔ net return > 0.
- **No lookahead.** Scaling flow, spot, OI and funding after the bar before entry leaves every entry feature unchanged.
- **Coverage and direction:** entry-feature coverage is > 80%, and MFE is larger for winners.
- **Market features:** breadth is in [0, 1].
- **Statistics:** AUC and BH are correct on known cases.
- **Planted signal:** discovered (q < 0.10), confirmed, and the model passes.
- **Pure noise:** the model does not pass.
- **Notebook:** runs end to end on 10 synthetic assets with no planted signal, and gives `NOTHING_CONFIRMED` with 0 discovery survivors.

## 6. Run (PC, Windows CMD)

Downloads go into the same `long` folder as TB 01, about 10–12 GB in total including the zip cache. Existing files are reused or skipped.

```bat
cd /d D:\Projects\deribit_market_data
git pull
python tools\download_binance_um_klines.py --months 84 --out D:\MarketData\Binance\long --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT BNBUSDT DOGEUSDT ADAUSDT LINKUSDT DOTUSDT LTCUSDT BCHUSDT UNIUSDT NEARUSDT APTUSDT ARBUSDT OPUSDT SUIUSDT FILUSDT ATOMUSDT ETCUSDT INJUSDT AAVEUSDT TONUSDT WIFUSDT HBARUSDT XLMUSDT SEIUSDT TIAUSDT ENAUSDT WLDUSDT
python tools\download_binance_basis_data.py --months 84 --skip perp quarterly --out D:\MarketData\Binance\long
python tools\download_binance_um_metrics.py --start 2021-12-01 --out D:\MarketData\Binance\long\basis\metrics
python -m unittest discover -s tests -p "test_turtle_trade_features.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 notebooks\Turtle_Breakout\04_trade_features.ipynb
git add notebooks\Turtle_Breakout
git commit -m "Turtle Breakout 04: run"
git push
```

- **Downloads.** Slow: 30 symbols × 84 months of 1m perpetual and spot data, plus about 50,000 daily metrics files. Symbols or months that do not exist are skipped and reported.
- **Notebook.** About 30–60 minutes, roughly 1 minute per asset to load and compute features, plus the bootstraps.
- **Outputs** go to `notebooks/Turtle_Breakout/04/`:
  - `04_decision.json`;
  - `04_univariate_discovery.csv`, `04_confirmation.csv`;
  - `04_univariate_big.csv`, `04_confirmation_big.csv`;
  - `04_group_models.csv`, `04_model_terciles.csv`;
  - `04_path_features.csv`, `04_early_path.csv`;
  - `04_trade_features.csv.gz`, which holds every trade with every feature, for any later look;
  - `04_coverage.csv`, `04_manifest.json`.

## 7. Decisions needed from Ivan

1. Confirm §4 is binding.
2. Run §6.
