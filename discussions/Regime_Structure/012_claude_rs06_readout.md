# Regime Structure 012 — RS 06 readout: hedging makes the long straddle worse

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-14 |
| Branch | Regime Structure |
| Responds to | RS 011; Ivan's run, commit 5ebfd37 |
| Status | Result. The primary verdict is binding. |

## 1. Primary (H = 15 min, taker 0.05% + half-spread)

1,610 straddles over 22 sessions (2026-09-21 → 10-13).

**R_hedged = −0.410 [−0.484, −0.337]. Verdict `NO_LONG_EDGE`. 0 of 22 sessions are positive.**

| Session mean R | 5 min | 15 min | 30 min | 60 min |
|---|---|---|---|---|
| Hedged, taker | −0.53 | **−0.41** | −0.38 | −0.34 |
| Hedged, 0.02% fee | −0.33 | −0.29 | −0.29 | −0.27 |
| Hedged, zero cost | −0.19 | −0.20 | −0.23 | −0.22 |
| Hedge cost / premium | 35% | 21% | 15% | 12% |
| Hedge P&L / premium | +9% | +8% | +5% | +6% |

Unhedged on the same entries: −0.28 [−0.42, −0.12]. Hedged minus unhedged at 15 min: −0.13 [−0.25, −0.01].

## 2. Reading

- **Hedging does capture the path.** Hedge P&L is positive (+5% to +9% of premium), and faster hedging captures more, as the RS 05 reversion predicts. Realised variance at the hedge interval averages 1.10–1.20 × implied, though the median is 0.885 at 15 min.
- **Costs dominate.** Perpetual taker hedging costs 12–35% of a 0DTE premium. The gain from the path is a fraction of that.
- **Even free hedging loses about 20%**, because the option is bought at the ask. The only route to a positive result would need near-zero option *and* hedge costs, which VRP 03 showed are not available passively.
- **By asset:**
  - BTC and ETH: realised/implied at 15 min of 1.36 (path variance clearly above implied), but they are the costliest to hedge (23–30% of premium).
  - SOL and XRP: below 1 (0.91–0.94).

## 3. Noise vs reversion (Binance 2 years, variance ratio by sampling interval)

| Test window | 5 min | 15 | 30 | 60 |
|---|---|---|---|---|
| BTC | 0.86 | 0.87 | 0.86 | 0.86 |
| ETH | 0.93 | 0.95 | 0.94 | 0.93 |
| SOL | 0.78 | 0.81 | 0.83 | 0.85 |
| XRP | 0.93 | 0.99 | 0.97 | 0.97 |

- BTC and SOL keep a ratio of about 0.85 at 60-minute sampling, so their intraday reversion is real at hedgeable horizons, not only noise.
- For ETH and XRP the ratio is close to 1 beyond 5 minutes; that part was mostly noise.
- The development window agrees.
- RS 05's pooled 0.75 combined both effects.

## 4. Conclusion

Gamma scalping 0DTE options on Deribit does not pay at retail costs, with or without hedging fees. With RS 02 and RS 04 this closes the long-option line on this data:
- entry timing fails (RS 02);
- the predictor is already priced (RS 04);
- hedging fails (RS 06).

The intraday reversion in BTC and SOL is real. Exploiting it would need an instrument without the option spread, for example a perpetual mean-reversion strategy, and it would face the same 0.05% taker fee. That would be a new, separate question for Ivan to decide on.

## 5. Decision needed from Ivan

Accept the result. The Regime Structure branch's option tests are complete; see the strategic review for next steps.
