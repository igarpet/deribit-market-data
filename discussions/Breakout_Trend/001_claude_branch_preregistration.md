# 001 — Claude: Breakout Trend branch, pre-registration of BT 01

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-13 |
| **Branch** | Breakout Trend (new, independent) |
| **Responds to** | Ivan, 2026-10-13: test a third-party hourly breakout strategy (YouTube review of an index-futures strategy) on our data, on perpetuals and on options, with equity curves per asset |
| **Status** | pre-registration + code (validated on synthetic data); awaiting Ivan's download and run |

## 1. Question

Does the strategy make money **after Deribit costs** on crypto perpetuals, and is its entry better than a coin flip? Buying ATM options on the same signals is a second leg. It is reported, but no decision rides on it.

The source reported the following on US index micro futures:
- 2021–2025 (in sample): profitable;
- 2020 (out of sample): it held;
- 2026 (out of sample): it lost on every index, and its entry was no better than random entries.

## 2. Rules (fixed in `breakout_trend.py`, `Params`)

Hourly bar $h$ ends at HH:00 and contains the twelve 5-minute bars that end HH:05 … (HH+1):00. At a 5-minute decision time $t$, only completed hours are used ($\text{end}(h)\le t$).

$$
M_h=\mathrm{SMA}_{100}(C),\qquad A_h=\mathrm{SMA}_{100}(H-L),\qquad U_h=M_h+1.5A_h,\quad L_h=M_h-1.5A_h
$$

$$
S_t=0.5\,P_t+0.5\,\mathrm{SMA}_3(C)_h,\qquad P_t=\text{close of the 5-minute bar ending at }t
$$

- **Entry from flat.** Go long if $S_t>U_h$ and short if $S_t<L_h$. The fill is at $P_t$.
- **Stop-and-reverse.** If $P_t$ moves $1.5A_{\text{entry}}$ against the entry, the position flips to the opposite side at $P_t$. The new position gets fresh levels (Ivan).
- **Trailing exit.** Once $P_t$ has been $\ge 1.0A_{\text{entry}}$ in favour, the position exits to flat when $P_t$ crosses $\mathrm{SMA}_{14}(C)_h$ against it.

**Ivan's answers (2026-10-13):**
- $A$ is the mean high−low.
- $P_t$ is the current 5-minute price.
- A flip gets fresh levels.
- Costs are Deribit's perpetual fees.

**Assumptions (mine, configurable, fixed before any real-data output):**
1. **Length of $A$ = 100 hours,** the same as the anchor. The source does not give it.
2. **Re-entry after a trailing exit.** The same side needs $S_t$ back inside the band before it can trigger again (`reentry="cross"`). The source describes the flip on a stop but not re-entry after a profit exit. `reentry="state"` is the alternative, where the position re-enters while $S$ is still outside the band.
3. **No intrabar fills.** Stops and exits are evaluated on 5-minute closes only ("checked every 5 minutes").
4. **Unusable hours.** An hour with fewer than 10 of its 12 bars is dropped. An SMA needs at least 90% of its window. No decision is taken if the last completed hour is more than 2 h old.

## 3. Data and costs

- **Price.** 5-minute OHLC from Binance spot `<ASSET>USDT`, 2024-09-01 to now (`tools/download_ohlc_history.py`, new). This is the same source as VD 10c, but VD 10c kept only closes, and $A$ needs high/low.
  - **HYPE** falls back to Deribit `HYPE_USDC-PERPETUAL` candles. VD 024 found no pre-window HYPE history on Binance.
  - Binance spot is used as a proxy for the Deribit perpetual price path.
- **Perpetual leg.** One unit of underlying per trade. Each side costs a taker fee of 0.05% of notional plus 1 bp of slippage, so a flip pays both sides.
  - **Funding is ignored.** Bias: a persistent positive funding rate would penalize longs and pay shorts.
- **Option fee.** 0.03% of the underlying per option, capped at 12.5% of the price. The daily delivery fee is assumed to be 0.
- **Verification.** I could not verify the fee schedule from the build environment, as in VRP 02 (Ivan to confirm).

## 4. Options leg

- **Contract.** On every perpetual entry, including flips, buy the ATM call for a long signal or the ATM put for a short signal. The expiry is the first daily expiry (08:00 UTC) at least 24 h away, so TTE at purchase is in [24 h, 48 h).
- **Exit.** The option is sold when the perpetual position closes. If it expires first, it settles at intrinsic and the position rolls into the next expiry.
- **Size.** Delta-matched, $q=1/|\Delta|$ at purchase, about 2 contracts.
- **Model-priced over the full history.** No option quotes exist before 2026-09-19, so the pricing is Black-76 with
$$
\sigma = k\,\hat\sigma_{\text{EWMA}}(t_0),
$$
where $\hat\sigma_{\text{EWMA}}$ is 5-minute realized volatility with a 24-h half-life.
  - $\sigma$ is held fixed over the option's life, so the model has **no vega P&L**.
  - The option is bought at $\sigma(1+h)$ and sold at $\sigma(1-h)$, where $h$ is the relative IV half-spread by TTE bucket (0–6, 6–24, 24–48 h).
  - $k$ and $h$ are calibrated per asset on the collector's quotes. $k$ is the median ATM mid IV/$\hat\sigma$ for quotes with TTE ≥ 12 h, from the 0DTE grid since 2026-09-19 and the 1DTE grid since 2026-10-06.
  - **Limitation.** These parameters come from a 3–4 week window inside the test period. They are applied backwards to 2024–2025, so the options curve is a cost-and-convexity transformation of the perpetual signal, not an executable history.
- **Real-quote check.** In the collector window, every option piece is re-priced at the real ask (entry) and bid (exit) of the nearest-forward strike, with fees, and compared with the model. This is reported only; the window is short.

## 5. Train/test

**The parameters are not fitted to crypto**, so 2024-09 → 2026-10 is out of sample in its entirety for the published settings. The split matters only for modifications:

- **Development:** 2024-09-01 → 2025-12-31, about 16 months.
- **Test:** 2026-01-01 → end of data. This matches the source's failing year.

Any modification must be chosen on development only. The test window is then read **once per pre-registered modification set**. Tuning on development and re-reading test until something passes is the optimization trap the source's review demonstrates (the top-10 settings had a 2026 Sharpe of −0.36).

## 6. Coin-flip null and decision (pre-registered)

- **Null.** 200 runs (seed 1201) per asset with the same exits and costs. The entry from flat is replaced by a coin: on each valid flat bar, enter with probability $p$ equal to the real run's entries-from-flat per flat bar, and pick long or short 50/50.
- **Statistic.** Sharpe ratio of daily changes in mark-to-market equity, annualized with $\sqrt{365}$.
- **Null share.** The share of null runs whose Sharpe is at least the real Sharpe, computed per window.
- **Decision (perpetual leg, net).** A window is `EDGE_CANDIDATE` iff net P&L $>0$ **and** null share $<0.05$; otherwise it is `NO_EDGE`.
- **Primary verdict:** the pooled portfolio (equal-weight sum over all available assets, null runs paired by index) on the **test** window. Per-asset and development verdicts are reported only, to avoid choosing the best of 7 assets after the fact.
- **Options.** P&L, Sharpe and drawdown are reported with no decision.

## 7. Outputs (`notebooks/Breakout_Trend/01/`)

| File | Content |
|---|---|
| `01_equity_daily.csv` | Daily equity (perpetual, options model, buy & hold) per asset and pooled |
| `01_trades.csv.gz` | Every trade: perpetual net/gross return, exit reason, option return and size |
| `01_metrics.csv` | Per asset × window × leg: P&L, Sharpe, max drawdown, trades/yr, win rate, average win/loss, long/short split, time in market, buy & hold |
| `01_null_decision.csv` | Coin-flip null and verdicts |
| `01_null_runs.csv.gz` | Coin-flip runs |
| `01_option_calibration.csv` | Option calibration ($k$, $h$, coverage) |
| `01_option_real_vs_model*.csv` | Real-vs-model option comparison |
| `01_decision.json` | Primary verdict and registration |
| `01_run_manifest.json`, `01_coverage.csv` | Run manifest and data coverage |

The equity figure (one panel per asset with the perpetual and options lines, plus a pooled panel) is displayed inline in the notebook.

## 8. Validation (synthetic)

`tests/test_breakout_trend.py` has 14 tests, all passing:
- No lookahead: indicators at $t$ are unchanged when later bars are removed, and the trade list is identical up to the cut.
- Hour-end mapping and partial-hour exclusion.
- Rule cases: stop-and-reverse with fresh levels, trail arming, cross vs state re-entry, invalid bars.
- Cost accounting: the final MTM equity equals the sum of net trade returns.
- Expiry roll; theta loss on a flat path.
- The option loader converts inverse quotes and picks the primary family.
- Calibration recovers $k$ and $h$ from a synthetic book, and the model matches real-quote P&L on it (corr > 0.9).
- The full notebook runs end to end on synthetic data.

## 9. Run (PC, Windows CMD)

Close the notebook in Jupyter first.

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_breakout_trend.py" -v
python tools\download_ohlc_history.py --start 2024-08-25
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebooks\Breakout_Trend\01_breakout_backtest.ipynb
git add notebooks\Breakout_Trend
git commit -m "Breakout Trend 01: run"
git push
```

- **Download:** a few minutes. The start date leaves room for the 100-hour warm-up.
- **Notebook:** about 10–20 minutes, mostly the 7 × 200 null runs.
- **Options.** If the collector parquet files are not under `DERIBIT_DATA_ROOT\daily\`, the options line is off for that asset and the coverage table says so.

## 10. Questions for ChatGPT

1. Is the model-priced options leg acceptable for the plot, given that $k$ and $h$ come from 3–4 weeks of quotes inside the test window? Would DVOL (BTC/ETH only) be a better IV anchor for 2024–2025?
2. Should funding be modelled from Deribit's funding history before any verdict is trusted, or is it second-order at a holding time of hours?

## 11. Decisions needed from Ivan

1. Confirm the two assumptions in §2: $A$ over 100 hours, and re-entry `cross`.
2. Confirm the fee constants in §3.
3. Confirm that §6 is binding: primary = pooled test window; options reported only.
4. **Modifications.** Send them before reading any test result, so they can be pre-registered against the development window only.
