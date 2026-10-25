# 009 — Claude: TB 07 pre-registration — what happens after a losing turtle trade, and does stop-and-reverse pay?

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-25 |
| **Branch** | Turtle Breakout |
| **Responds to** | TB 06 run `2cca94c`; Ivan, 2026-10-25: profit from the negative-R trades and keep idle capital working; "test what happens to losing trades, the stop and reverse idea is the cheapest and fastest option right now" |
| **Status** | pre-registration + code (validated on synthetic data); awaiting Ivan's run |

## 0. TB 06 in brief (run `2cca94c`, BTC/ETH, 86 trades)

| Strategy | Sharpe (daily R) | Total R | Max DD (R) | vs perpetual |
|---|---|---|---|---|
| S0 perpetual | 0.76 | 40.4 | −10.3 | — |
| S1 perpetual + put at entry strike | 0.53 | 24.2 | −9.9 | **`WORSENS`** (Sharpe diff −0.23, 97.5% CI [−0.41, −0.05]) |
| S2 ATM call | 0.68 | 30.1 | −10.6 | `NO_DIFFERENCE` |
| S3 call +1 ATR | 0.64 | 27.2 | −10.5 | `NO_DIFFERENCE` (reported) |

- **Put hedge:** it costs about 0.19R per trade net and barely moves drawdown (−10.3 → −9.9R). The stop already caps losses, so the put mainly pays twice for the same protection.
- **Calls:** cheaper than expected (no funding), but not better. No reason to move off the perpetual.
- TB 06 notebook 2 is left undefined; this turn takes its slot.

## 1. Question and prior

Of the 180 confirmation trades (TB 04), **107 are losers** (mean −0.74R), all closed by a stop (140 trailing, 40 fixed across all trades).

**Hypothesis (Ivan):** a failed breakout is information. If price keeps falling after a losing exit, a short opened at the stop recovers part of the loss and uses capital that is otherwise idle.

**Prior, stated in advance: probably not.**
- TB 01 short-only turtle lost about 9% CAGR. Crypto drifts up and shorts pay funding drag only when funding is negative.
- TB 04: losers below −0.5R after 5 days end with a 0–7% win rate, but that describes the *long's* path before the exit, not what comes after it.
- A losing trailing exit is a 10-day low. Short-term reversal (crypto bounces after breakdowns) is about as plausible as continuation.

## 2. Design (`turtle_reverse.py`)

**Long trades.** TB 01 daily rules, unchanged (`turtle_trade_features.run_trades`). Universe as TB 04:
- **Confirmation:** BTC/ETH/SOL/XRP from 2020-10-15 (binding).
- **Discovery:** the 26 other USDT perpetuals, full history (reported). TON stops at 2026-08-07 (flat-series artefact).

**Losing exit.** A long closed by a stop (`trail`, `fixed_stop`, `stop_same_period`) with net R < 0. Known at the exit fill; no look-ahead.

**Reverse short (S-R).** Opened at the exit fill, unit notional (results in R, not equity).

| | |
|---|---|
| fixed stop | reverse price + 2 ATR (ATR known at the start of the exit day); never moves |
| trailing stop | highest high of the last 10 completed days, recalculated daily |
| exit | min(fixed, trailing) traded through, fill max(stop, bar open), checked from the bar after the reverse; also closed at the next long breakout (in practice never binding: the 10-day high is below the 20-day high, so the stop is hit first) |
| costs | taker 0.05% + 1 bp per side on the short (the long's exit cost is already in the long); funding at Binance rates (a short receives positive funding) |
| R_rev | net return per notional / (2 ATR / reverse price) |

**Null.** Shorts opened at the **open of random flat days** (no long open, same window, same asset), same exits and costs, the same number per asset as the real reversals; 2,000 runs. This asks whether the *timing* (right after a failed breakout) beats shorting at any idle moment.

**Event study (reported).** Forward move at +1, +3, +5, +10 days after (a) losing exits, (b) winning stop exits, (c) random flat-day opens: in ATR (of the start day) and bp, share of down moves, and the net R of a short held exactly h days. Month-cluster 95% intervals.

## 3. Decision (binding, confirmation set)

| Gate | Test |
|---|---|
| **G1** pays on its own | mean R_rev > 0 and month-cluster bootstrap 95% lower bound > 0 |
| **G2** beats random shorts | mean R_rev above the random flat-day null, empirical one-sided p < 0.05 |
| **Verdict** | `REVERSE_PAYS` (G1 & G2) / `BEATS_RANDOM_SHORTS_ONLY` (G2 only) / `NO_EDGE` |

**Reported:** discovery-set statistics; breakdowns by long exit reason, short exit reason, year, asset and long-R bucket; share of reverse R from the top 3 trades; cumulative-R curves long-only vs long + reverse (total R, max drawdown, daily-R Sharpe).

**Power.** About 100 reversals on the confirmation set; R of trend-style shorts is right-skewed. A true mean of about +0.3R would be needed to clear G1 reliably. A `NO_EDGE` result does not prove zero; the discovery set (several hundred reversals) and the event study show the size and sign.

## 4. Validation (synthetic)

`tests/test_turtle_reverse.py`: 10 tests, all passing.
- One reverse per losing stop exit; shorts exit only by stop or next long; flat days exclude open longs.
- Positive funding is received by the short; curves end at Σ R.
- **Martingale:** real reversals ≈ random shorts. Calibration over 40 four-asset martingale runs: mean R_rev −0.075 vs null −0.070; p mean 0.50; G2 false-positive rate 5.0%, G1 2.5%.
- **Downtrend:** reversals pay.
- Notebook smoke run (8 synthetic assets): `NO_EDGE` on both sets, as expected.

A first version used a z-test for G2; on skewed short R it passed 2 of 6 martingale seeds, so G2 uses the empirical p instead (z is still reported).

## 5. Run (PC, Windows CMD)

No download: it uses the TB 04 files in `D:\MarketData\Binance\long`. Close the notebook in Jupyter first.

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_turtle_reverse.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebooks\Turtle_Breakout\07_stop_and_reverse.ipynb
git add notebooks\Turtle_Breakout
git commit -m "Turtle Breakout 07: run"
git push
```

**Runtime:** similar to TB 04's data loading (30 perps), roughly 20–40 minutes.

**Outputs** go to `notebooks/Turtle_Breakout/07/`: `07_decision.json`, `07_event_study.csv`, `07_reversals.csv`, `07_by_reason.csv`, `07_by_year.csv`, `07_by_asset.csv`, `07_by_long_R.csv`, `07_curves_summary.csv`, `07_curves_daily.csv`, `07_coverage.csv`, `07_manifest.json`.

## 6. Decisions needed from Ivan

1. Confirm §3 is binding.
2. Run §5.
