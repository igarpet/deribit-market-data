# 011 — Claude: TB 08 readout and TB 09 note — the same risk sweep on long + stop-and-reverse (descriptive)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-26 |
| **Branch** | Turtle Breakout |
| **Responds to** | TB 08 run `1e23d58`; Ivan, 2026-10-26: run the same risk analysis on the stop + reversal variant "just to see how drawdowns and equity curves compare"; not for implementation |
| **Status** | code (validated on synthetic data); awaiting Ivan's run. **Descriptive only: no gate** |

## 0. TB 08 readout (run `1e23d58`, BTC 2020-01-31 → 2026-10-24, 54 trades)

**Verdict `2PCT_NOT_BEATEN`, in all three sweeps.** None of the 99 levels grows significantly faster or slower than 2% after Holm. The best unadjusted p is about 0.05.

| f | P: CAGR | P: max DD | P: Sharpe | C3: CAGR / max DD | EQ: CAGR / max DD |
|---|---|---|---|---|---|
| 1% | 6.0% | −10.8% | 0.97 | same as P | 5.7% / −10.4% |
| **2%** | **12.3%** | **−20.8%** | **1.00** | same as P | 11.2% / −19.2% |
| 5% | 32.5% | −46.8% | 1.05 | same as P | 26.1% / −39.4% |
| 10% | 70.9% | −82.6% | 1.07 | 51.7% / −78.1% | 45.9% / −61.3% |
| 12% | 87.8% | −95.5% | 1.02 | 46.8% / −85.8% | 52.0% / −67.6% |
| 13% | **ruin** | −100% | — | 35.4% / −90.7% | 54.6% / −70.4% |
| 20% | ruin | | | 21.7% / −96.4% | 65.5% / −84.6% |
| 100% | ruin | | | 21.3% / −97.1% | ruin (from 80%) |

- **Growth rises steadily up to the cliff, then falls off it.** On the primary sweep the best level in hindsight is 12% (CAGR 88%, drawdown −95.5%), and 13% is wiped out. HWM sizing is what makes the cliff sharp: after a −50% drawdown, 12% of the peak is 24% of the account.
- **Sharpe hardly moves** (0.97–1.07 below the cliff), as expected: the same trades, scaled.
- **Kelly on the trades** (equity sizing): $f^*$ = 23%, half-Kelly 11.5%; the trade bootstrap's 5–95% range is 6%–40%. Only 2.4% of the draws put $f^*$ at or below 2%, so 2% is conservative, not optimal. The data cannot say how much more is safe.
- **The 99 comparisons are what make it insignificant.** Growth from 3% to 7% has unadjusted p of 0.07–0.11. The run does not show that higher risk is no better; it shows the sample cannot prove it.

## 1. TB 09 design (`turtle_sizing.py` 0.2.0)

- **Trade lists:** (a) TB 01 longs on BTC; (b) the same longs plus the TB 07 reverse short at every losing stop exit. Positions never overlap: the short opens at the long's exit bar and fill, and closes before the next long.
- **Sizing:** every trade, long or short, gets the same rule, $u = \min\!\big(f\cdot \text{base}/(2\,\mathrm{ATR}),\ \text{cap}\cdot E/\text{price}\big)$. The short uses the ATR of its own day.
- **Engine:** `equity_from_trades` replays a trade list with daily marks. On the long-only list it reproduces `turtle_breakout.simulate` exactly at every tested f, cap and sizing base (unit test), so the TB 08 sweep is the long-only column.
- **Sweeps:** P, C3, EQ × 2 trade lists × 100 levels = 600 backtests.
- **Reported:** side-by-side CAGR, Sharpe, max DD, Calmar, ruin, worst trade and leverage per level; best f and first ruin per sweep; the highest f with max DD above −50%; paired block-bootstrap growth difference (long + reverse − long only) per f, without multiplicity control; equity and drawdown curves at 1/2/5/10%.
- **The one open reverse short at the end of the data is dropped** (TB 07 only keeps closed reversals); this is at most one trade.

**Expectation.** BTC was the best confirmation coin for reversals (+0.30R, 30 trades), so the combined curve will probably end higher at the same f. The reversals add about 30 more stop-loss trades, though, so drawdowns at the same f should be at least as deep. If so, the cliff moves to a lower f.

## 2. Validation

`tests/test_turtle_sizing.py`: 15 tests, all passing (11 from TB 08 + 4 new).
- The engine reproduces `simulate`.
- Reverse shorts open exactly at the long's exit bar and price, with no overlap.
- The reverse count matches TB 07, and reverse P&L scales with f.

The smoke run on synthetic data (a sample where reversals lose) works end to end.

## 3. Run (PC, Windows CMD)

No download. A few minutes.

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_turtle_sizing.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks\Turtle_Breakout\09_risk_per_trade_btc_reverse.ipynb
git add notebooks\Turtle_Breakout
git commit -m "Turtle Breakout 09: run"
git push
```

**Outputs** go to `notebooks/Turtle_Breakout/09/`: `09_sweep_{P,C3,EQ}_{long,long_rev}.csv`, `09_side_by_side.csv`, `09_summary.csv`, `09_pair_P.csv`, `09_curves_P_selected.csv`, `09_manifest.json`.
