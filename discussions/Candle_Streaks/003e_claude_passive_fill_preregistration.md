# Candle Streaks 003e — Passive execution of the CS03c signal (pre-registration)

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-15 |
| Branch | Candle Streaks |
| Responds to | CS 003d (CS03c readout); Ivan's choice of option A |
| Status | Pre-registration. Written before any aggTrades data has been downloaded or seen. |

## 1. Question

CS03c: signed flow predicts the perp's own next 5 minutes, but the gross edge is about 0.5 bp per trade against 12 bp taker round-trip costs.

Does entering with a **limit order** (maker fee, no spread crossing) recover it? Or does adverse selection — fills happening mainly when the forecast is wrong — remove the edge?

## 2. Data

- **Execution:** Binance USD-M `aggTrades` for BTCUSDT and ETHUSDT, 2026-07-01 → 2026-09-30 (92 days, inside the CS03c TEST window of 2026-05-17 → 2026-10-10).
- **Signal:** CS03c reproduced exactly:
  - same complete-day intersection and 60/20/20 split;
  - `price_vol_signed` fitted on TRAIN;
  - trade when $|P_{up}-P_{down}|\ge$ TRAIN q90, side $=\operatorname{sign}(P_{up}-P_{down})$.

## 3. Execution model (frozen)

At signal origin $t$ (end of a 5m bar), $P_0$ is the last trade at or before $t$, and the limit is

$$L = P_0\,(1 - s\,k/10^4),\qquad k\in\{0,1,2,5\}\ \text{bp},$$

where $s=+1$ for a buy and $-1$ for a sell.

- **Fill window:** the order lives for $(t, t+5\text{m}]$ only.
- **Fill rule:** filled only on a **strict trade-through**. A buy needs a trade at a price $< L$. Trades exactly at $L$ are assumed to be queued ahead of us, so this does not model queue position. Latency is ignored. Both choices make the result an **upper bound on fills**.
- **Exit:** at the first trade at or after $t+H$; $H=5$ min is the primary horizon, $H=15$ min is reported.
- **Gross P&L:** $s(\log P_{exit}-\log L)$. Unfilled orders contribute 0.
- **Overlap:** orders are independent units. Consecutive signals are not netted, which is conservative on costs.

| Costs (round trip on a filled order) | bp | Role |
|---|---|---|
| maker in (0.02%) + taker out (0.05% + 1 bp) | 8 | **Gate** |
| maker in + maker out at the mark | 4 | Optimistic, report only |
| zero | 0 | Gross / adverse-selection measure |
| taker benchmark (both sides) | 12 | Reference (CS03c) |

## 4. Controls and diagnostics

- **Random-side arm:** a passive order with a random side at **every** TEST origin in the window. This gives the adverse selection a passive order suffers with no information. The signal's value is signal minus random.
- **Markouts:** signed returns from $L$ to the price 60 s and 300 s after the fill.
- **Selection table:** the taker outcome of the same orders, split by whether the passive order filled. This directly shows that fills happen mainly when the trade goes wrong.

## 5. Gate

There must exist **one** offset $k$ such that, for **both** BTC and ETH:
- at $H=5$,
- at the 8 bp cost,

the signal arm's net bp/day has a day-bootstrap lower bound above 0, at Bonferroni $\alpha=0.05/4$.

| Gate | Verdict |
|---|---|
| Pass | `PASSIVE_CANDIDATE` (then prospective test only, with real queue data) |
| Fail | `NO_PASSIVE_EDGE` |

## 6. Validation done before the real run

- **Unit tests:**
  - a trade at $L$ does not fill;
  - a trade-through fills;
  - offset and side signs;
  - the order expires after the fill window;
  - price running away never fills;
  - day and cost arithmetic;
  - downloader parsing with and without a header;
  - datetime-resolution-independent ms conversion.
- **Synthetic random walk:**
  - no significant positive gross;
  - the selection table shows the expected pattern: the taker outcome is about −1 to −5 bp when the order fills at $k=1$–5, and about +0.5 to +2.6 bp when it does not.

## 7. Expectation (stated in advance)

`NO_PASSIVE_EDGE`.
- The 8 bp gate cost alone is 16× the CS03c gross edge.
- The selection effect should make gross P&L on filled orders **lower** than the taker gross.
- Only the 4 bp optimistic line has any chance, and it would still need queue-aware live data to be believed.
