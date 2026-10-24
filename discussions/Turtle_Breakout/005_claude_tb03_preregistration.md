# 005 — Claude: TB 03 pre-registration — intraday windows and volatility-engine sizing

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-25 |
| **Branch** | Turtle Breakout |
| **Responds to** | Ivan, 2026-10-25: replace the 20-day window with a 4-hour window. Test both readings (4-hour candles; the last 4 hours), each with and without the volatility engine as position sizer, and both engine roles (sizing only; sizing and stop), against the original |
| **Status** | pre-registration + code (validated on synthetic data); awaiting Ivan's run |

## 1. Nine strategies

All nine are long only on BTC/ETH/SOL/XRP perpetuals.
- **Kept from TB 01:** 2% HWM risk, fees + slippage + funding, the 3× per-position cap, 5m intraday fills.
- **Window:** 2020-10-15 → end of data.
- **Grid:** 3 timeframes × 3 sizing modes.

**Timeframes:**

| Timeframe | Bars | Entry | Trailing stop | ATR |
|---|---|---|---|---|
| `daily` | 17:00 NY days | 20-day high | 10-day low | 20 days |
| `h4_candles` | 4h, UTC 00/04/…/20 | 20 bars = 80 h | 10 bars = 40 h | 20 bars |
| `last_4h` | 1h | high of the last 4 h | low of the last 2 h | 4 h |

`last_4h` keeps the 20:10:20 proportions. It is the reading in which the window itself is 4 hours.

**Sizing modes:**

| Mode | Units | Fixed stop |
|---|---|---|
| `atr` (original) | $0.02\,\mathrm{HWM}/(2\,\mathrm{ATR})$ | entry − 2 ATR |
| `engine` | $0.02\,\mathrm{HWM}/(2U)$ | entry − 2 ATR |
| `engine_stop` | $0.02\,\mathrm{HWM}/(2U)$ | entry − 2U |

The trailing stop is unchanged in every mode.

**Engine unit:**
$$U = c_k\,\sigma^{\text{eng}}_{t}\,P_t$$
- $\sigma^{\text{eng}}_t$ is the frozen VD18 HAR_4 (AM-T-HAR-Blend variance level) sd of the log return over one bar of the timeframe: 1,440, 240 or 60 min.
  - It is taken from the 5m bar before the entry bar.
  - 1,440 min is beyond the fitted horizons, so it uses the flat extrapolation of the 240-min rate.
- $c_k$ is the expanding median of $\mathrm{ATR}/(\sigma^{\text{eng}}P)$ over past periods, causal and with at least 30 observations.
  - $c$ makes the engine change **when** risk is large without changing its **average** scale. Without it, engine sizing would differ from ATR mostly by leverage (ATR ≈ 1.6 σ).
- If $U$ is unavailable, sizing falls back to ATR, and the fallback is counted.

## 2. Decision (binding)

**Per strategy.** Bonferroni over 9: $\alpha=0.05/9$.

| Gate | Rule |
|---|---|
| **G1, beats luck** | share of random-entry portfolios with CAGR ≥ real < $\alpha$, **and** the strategy's net CAGR > 0 |
| **G2, beats benchmarks** | Sharpe difference vs B1, B2 and B3 (TB 02 constant-mix), each with a two-sided $1-\alpha$ circular 21-day block-bootstrap interval whose lower bound is > 0; all three must pass |

- **G1 details.** The random portfolios use the same exits, sizing and costs. There are 1,000 runs for daily, 400 for h4 and 200 for last_4h.
- **Why net CAGR > 0 is in G1.** Random portfolios in the last_4h design are ruined by costs. "Beating" ruined portfolios while losing money is not an edge, and the synthetic run showed exactly that.
- **G2 benchmark leverage:** min(mean leverage, 1). Sharpe is almost leverage-free, and the cap stops a 4× buy-and-hold from being ruined artificially.
- **Verdicts:** `BEATS_LUCK_AND_BENCHMARKS`, `BEATS_LUCK_ONLY`, `BEATS_BENCHMARKS_ONLY`, `NO_EDGE`.

**Contrasts.** Paired daily-return Sharpe differences, with two-sided intervals at $1-0.05/8$. Each result is `A_BETTER`, `B_BETTER` or `NO_DIFFERENCE`.

| Contrast | Comparison | Question |
|---|---|---|
| C1–C3 | `engine` − `atr`, per timeframe | does engine sizing help? |
| C4–C6 | `engine_stop` − `atr`, per timeframe | does engine sizing + stop help? |
| C7 | `h4_candles|atr` − `daily|atr` | intraday vs original |
| C8 | `last_4h|atr` − `daily|atr` | intraday vs original |

**Reported only:**
- trades, win rate, profit factor and hold time;
- mean leverage and capped entries;
- engine-use and fallback counts;
- fees and funding;
- per-asset P&L;
- years;
- equity curves.

## 3. Expectations, stated in advance

- **`last_4h`.** About 17,000 trades in the synthetic run. ATR over 4 hours is so small that the 3× cap sets the size on about 90% of entries. Costs alone are likely to ruin it, and engine sizing barely matters there.
  - This is a property of the design, not a bug. TB 01's hourly variant (20/10/20 hours) was ruined the same way.
- **`h4_candles`.** About 1,000 trades and about 1× leverage. Fees and funding are large, so it is the version to watch.

## 4. Validation (synthetic)

Tests: `test_turtle_intraday.py` has 8 tests. `test_turtle_breakout.py` and `test_turtle_benchmarks.py` are unchanged and pass, so the defaults reproduce TB 01 and TB 02. 33 tests pass in total.

- **Periods and levels:**
  - h4 boundaries are at UTC multiples of 4 h;
  - last_4h levels are the max of 4 hourly highs and the min of 2 hourly lows.
- **Engine unit:**
  - the engine σ is finite and positive;
  - $c$ and σ are causal, unchanged when future data is removed;
  - the median $U/\mathrm{ATR}$ is ≈ 1.
- **Sizing modes:**
  - all three give identical entries;
  - `engine` keeps the ATR exits;
  - units × 2U = 2% HWM.
- **Decision code:** evaluation and contrasts run end to end.
- **Benchmark panel fix:** TB 02's benchmark panel now always uses daily 17:00 New York closes, whatever the strategy's timeframe. It previously assumed daily periods. TB 02's daily results are unaffected, because the same bars are used.
- **Synthetic notebook** (trending market, nulls cut to 5%):
  - daily and h4 beat luck;
  - last_4h is ruined;
  - no engine contrast is significant.

## 5. Run (PC, Windows CMD)

No new download. Close the notebook in Jupyter first.

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_turtle_intraday.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 notebooks\Turtle_Breakout\03_turtle_intraday_engine.ipynb
git add notebooks\Turtle_Breakout
git commit -m "Turtle Breakout 03: run"
git push
```

**Runtime:** about 30–60 minutes, mostly the 600 last_4h random-entry runs. For a quick look first, `set TB03_NULL_SCALE=0.1`; the decision needs the full run.

**Outputs** go to `notebooks/Turtle_Breakout/03/`:
- `03_decision.json`, `03_strategies.csv`, `03_contrasts.csv`, `03_benchmark_tests.csv`;
- `03_per_asset.csv`, `03_yearly.csv`;
- `03_equity_daily.csv`, `03_trades.csv.gz`, `03_null_runs.csv.gz`;
- `03_manifest.json`.

## 6. Decisions needed from Ivan

1. Confirm the last_4h reading (4 / 2 / 4 hours on 1h bars) and the engine-unit calibration $c$.
2. Confirm §2 is binding.
3. Run §5.
