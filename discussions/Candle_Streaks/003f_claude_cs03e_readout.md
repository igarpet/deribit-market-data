# Candle Streaks 003f — CS03e readout: passive execution makes the signal negative before fees

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-15 |
| Branch | Candle Streaks |
| Responds to | CS 003e pre-registration; Ivan's run 7c6f426 |
| Status | Result. The pre-registered verdict is binding. |

## 1. Verdict: `NO_PASSIVE_EDGE`

BTC/ETH, 92 days (2026-07-01 → 09-30), about 3,100 signal orders per asset, no missing trade days. At the gate cost (8 bp), net P&L is −108 to −265 bp/day at every offset. Even the optimistic 4 bp line loses 55–140 bp/day.

## 2. The key table (H = 5, zero cost, bp per trade)

| | BTC taker | BTC passive k=0 / 1 / 2 / 5 | ETH taker | ETH passive k=0 / 1 / 2 / 5 |
|---|---|---|---|---|
| Signal | **+0.45** | −0.22 / −0.46 / −0.41 / −0.23 | **+0.21** | −0.51 / −0.63 / −0.46 / −0.65 |
| Random side | −0.01 | −0.61 / −0.85 / −0.75 / −0.60 | +0.05 | −0.67 / −0.86 / −0.86 / −0.80 |

- **The taker signal is reproduced.** The CS03c gross edge is replicated on actual trade prices (BTC +15 bp/day, CI [+1, +27]).
- **Passive entry turns the same signal negative before any fee.** Even with fills assumed on trade-through only (an upper bound), every offset is below zero.
- **Random-side passive orders lose about 0.6–0.9 bp per fill** at zero cost. That is the adverse-selection cost of resting liquidity on these books.
- **The signal reduces that cost by only about 0.2–0.4 bp.** It does not overcome it.

## 3. Mechanism (fill selection, BTC signal arm)

Taker outcome of the same orders, split by whether the limit would have filled:

| k (bp) | If filled | If not filled |
|---|---|---|
| 0 | −0.22 | +6.51 |
| 1 | −1.46 | +5.84 |
| 2 | −2.41 | +5.13 |
| 5 | −5.23 | +4.06 |

- **Profitable signals don't fill.** All the profit sits in the orders that never fill, because price ran away in the forecast direction. Fills are concentrated where the forecast failed.
- **Markouts confirm it.** They are negative 60 s after every fill: −0.3 to −1.0 bp. ETH is the same.
- **Markouts can't be summed with the P&L.** They are measured over a different window from the trade's P&L.

## 4. Conclusion

- Passive execution does not rescue CS03c. It removes the gross edge before costs.
- This is an upper bound on fills: queue position and latency would make it worse.
- **The flow signal is closed as a retail strategy at 5-minute horizons**, taker or maker.
- **The only remaining use is market making:** quoting both sides and using the signal to lean the quotes. That requires queue-aware L2 data, low latency and maker fees at or below 0. It is outside this project's data and infrastructure.

## 5. Decision needed from Ivan

Accept. The directional research on 5-minute perp/spot bars (CS, RS, BT, VWAP) is complete and negative at retail costs.
