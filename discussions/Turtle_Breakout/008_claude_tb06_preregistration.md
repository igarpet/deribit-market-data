# 008 — Claude: TB 06 pre-registration — protective put at the entry strike, and calls instead of the perpetual

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-25 |
| **Branch** | Turtle Breakout |
| **Responds to** | TB 05 run `0119f2c`; Ivan, 2026-10-25: (a) hedge each daily turtle trade with a put struck at the entry price, tenor = time from entry to the best winning trades; (b) an option-only version, buying calls ATM or a bit above the entry price instead of the perpetual. This is notebook 1 of 2 |
| **Status** | pre-registration + code (validated on synthetic data); awaiting Ivan's DVOL download and run |

## 0. TB 05 in brief (run `0119f2c`)

**The edge is not fractal.** Edge vs random entries, in R:

| Candle | Edge vs random (R) |
|---|---|
| 1–2 min | about −0.007 (worse than random) |
| 10 min | +0.015 |
| 1 h | +0.09 |
| 4 h | +0.21 |
| 1 day | +0.78 |

**G_net.**
- **Daily:** the only pass (`EDGE_TRADEABLE`).
- **60 min – 12 h:** the net mean is positive, but the 99.6% month-cluster interval includes 0. The three best months carry 54–78% of the net P&L.
- **≤ 30 min:** negative at taker costs (break-even 1–8 bp against 12 bp).

## 1. Prior expectation, stated in advance

**Put (S1).** The turtle's stop already caps losses: on the confirmation trades the median loser is −0.87R, the worst −1.16R, and only 2 of 180 trades are below −1.1R. A put at the entry strike therefore mainly converts the stop-loss into a premium.
- **Cost.** Black–Scholes at 60% IV for a 38-day ATM put is about 7% of notional. With ATR ≈ 4.8%, that is about 0.7–0.8R per trade, against a net mean of about 1.1R.
- **Expectation.** It should **reduce drawdown and cost return**; the test is whether Sharpe improves.

**Calls (S2/S3).** perp + ATM put ≈ ATM call (put–call parity). The call version differs mainly in having no perpetual funding and capped losses, and in needing a roll when a trend outlives the expiry.

## 2. Design (`turtle_options.py`)

**Trades.**
- TB 01 daily rules on the **BTC and ETH** perpetuals.
- **Window:** entries from DVOL's start (March 2021) to the end of the data, about 80 trades.
- **SOL/XRP excluded.** Their options are USDC-settled with a short, thin history, and there is no DVOL.

**Tenor.**
- **DTE target:** 38 days, the mean holding time of the $R\ge3$ trades in the **TB 04 discovery set** (26 other perps, 83 trades, 37.8 days). It is taken from data that never touches BTC/ETH.
- **Expiry:** the first Friday 08:00 UTC at or after entry + 38 days, which is a Deribit weekly or monthly.

**Strategies.** One option per unit of underlying; results per trade in $R$ = 2 ATR at entry.

| | |
|---|---|
| **S0** | perpetual, TB 01: net of fees, 1 bp slippage and funding |
| **S1** | perpetual + put, $K$ = entry price. Sold at the bid when the turtle exits, or settled at intrinsic at expiry if that comes first. No roll: the hedge covers the first 38+ days, as Ivan specified |
| **S2** | call, $K$ = entry price, no perpetual. Sold at the bid when the turtle exits. If it expires first, it settles and a new ATM call (same tenor rule) is bought while the signal is still on |
| **S3** | as S2 with $K$ = entry + 1 ATR ("a bit above the price", about 5% OTM) |

**Reported variants:** S1 holding the put to expiry regardless of the turtle exit; S2 and S3 without rolling.

**Pricing.**
- **Model:** Black-76 on the perpetual price, with $\sigma$ = **DVOL** (Deribit's 30-day ATM implied-volatility index, hourly, last completed hour) for every strike and tenor.
- **Spread:** buy at $\sigma(1+h)$, sell at $\sigma(1-h)$, with $h$ = 2% relative IV half-spread. Sensitivity is reported for $h$ = 0 and 5%.
- **Fees:** 0.03% of the underlying per option, capped at 12.5% of the price; delivery fee 0.015%, capped at 12.5% of intrinsic.
- **Limitation:** no skew and no term structure. Both strikes are near the money and the tenor is close to DVOL's 30 days, but put skew in crashes would make S1 somewhat more expensive than modelled.

## 3. Decision (binding)

**Curves.** BTC + ETH cumulative-R curves, marked daily at 17:00 New York:
- open perpetual positions are marked at the close;
- options are marked at the Black-76 mid with DVOL;
- one unit per trade, no compounding.

**Primary comparisons,** each at $\alpha=0.025$ (Bonferroni over 2):

| | Comparison |
|---|---|
| **Q1** | S1 (perpetual + put) vs S0 |
| **Q2** | S2 (ATM call) vs S0 |

**Test statistic:** the paired Sharpe difference of daily changes in cumulative R, with a 21-day circular block bootstrap and a 97.5% interval. The result is `IMPROVES` if the lower bound is > 0, `WORSENS` if the upper bound is < 0, and `NO_DIFFERENCE` otherwise.

**Reported:**
- per-trade mean-R difference (month-cluster bootstrap);
- maximum drawdown in R;
- total R;
- the put leg alone (premium, mean P&L, share of trades where it pays);
- results by perpetual-R bucket and by year;
- S3 and the variants.

**Power, stated in advance.** About 80 trades over 5.5 years is a small sample. Unless the effect on Sharpe is large, the expected outcome is `NO_DIFFERENCE`. The drawdown and per-bucket tables then show *how* each structure changes the risk.

## 4. Validation (synthetic)

`tests/test_turtle_options.py`: 8 tests, all passing.

- **Expiry:** always a Friday at 08:00, between 38 and 45 days after entry.
- **DVOL lookup:** uses only completed hours, with a staleness guard.
- **Put–call parity:** without fees or spread, perpetual + put = call to 1e-6.
- **Costs:** spread and fees reduce P&L.
- **Rolls:** happen only for trades held longer than the expiry.
- **Put behaviour:** puts pay on losing trades and cost premium on winners.
- **Call losses:** bounded by the premiums paid.
- **Curves:** end at Σ trade R.
- **Statistics:** run.

**Smoke run.** The notebook runs end to end on synthetic BTC/ETH (82 trades). Both Q1 and Q2 are `NO_DIFFERENCE`, as expected.

## 5. Run (PC, Windows CMD)

One small download: the DVOL index, about 5 years hourly, two files. Close the notebook in Jupyter first.

```bat
cd /d D:\Projects\deribit_market_data
git pull
python tools\download_deribit_dvol.py
python -m unittest discover -s tests -p "test_turtle_options.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks\Turtle_Breakout\06_put_hedge_and_calls.ipynb
git add notebooks\Turtle_Breakout
git commit -m "Turtle Breakout 06: run"
git push
```

**Runtime:** a few minutes.

**Outputs** go to `notebooks/Turtle_Breakout/06/`:
- `06_decision.json`, `06_tests.csv`;
- `06_per_trade.csv`, `06_spread_sensitivity.csv`;
- `06_curves_summary.csv`, `06_curves_daily.csv`;
- `06_years.csv`, `06_trades.csv`;
- `06_manifest.json`.

## 6. Decisions needed from Ivan

1. Confirm §3 is binding, the 38-day tenor rule, and DVOL pricing.
2. Run §5.
3. Define notebook 2 (TB 07).
