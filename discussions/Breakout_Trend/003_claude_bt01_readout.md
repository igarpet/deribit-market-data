# BT 003 — Claude: BT 01 readout — no edge, not even before costs

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-14 |
| Branch | Breakout Trend |
| Responds to | BT 001, BT 002; Ivan's run, commit e2cb44e |
| Status | Result. The pre-registered decision is binding. |

## 1. Decision (pooled test window, 2026-01-01 → 2026-10-12)

| | Value |
|---|---|
| Net perp P&L (sum over 4 assets, 1 unit of notional each) | **−135%** |
| Sharpe | −1.66 |
| Coin-flip null: median Sharpe / 95th percentile | −0.88 / +0.26 |
| Share of coin-flip runs doing at least as well | **0.895** |
| **Verdict** | **NO_EDGE**. The strategy is worse than random entries with the same exits. |

Every asset in every window is NO_EDGE. In the test window all four lose net:
- BTC −48%
- ETH −41%
- SOL −46%
- XRP −1%

## 2. Before costs (from `01_trades.csv.gz`)

| Window | Trades | Gross mean per trade | SE | Net sum |
|---|---|---|---|---|
| Development | 1,741 | +6.8 bp | 6.0 bp | −0.90 |
| Test | 1,150 | −0.2 bp | 6.2 bp | −1.40 |

- **There is no gross edge.** The development-window +6.8 bp is within one standard error and vanishes in test.
- **Costs are the entire loss.** Each round trip pays 12 bp (taker 0.05% + 1 bp slippage per side). At 330–390 trades a year per asset, that is about 40–45% of notional a year per asset.
- Breakout entries: +6.1 bp (development), −3.9 bp (test). Reversals: +7.9 / +4.5 bp. Trailing exits win about 105–130 bp; stop-and-reverse exits lose about 135–175 bp. That is the expected shape of a trend system; the wins do not pay for the losses.

## 3. Options leg (reported, no decision)

- Worse than the perp leg in every asset and window.
- The real-quote check covers only 2–4 trade pieces per asset in the collector window. Model and real prices agree (correlation 0.999; mean difference −0.05 to −0.16 pp).

## 4. Reading

- An hourly breakout (trend-following) system has no edge on crypto perps over these 2 years. It loses to random entry with the same exits.
- This is consistent with RS 05/06: intraday paths mean-revert (variance ratio about 0.85 for BTC and SOL at 60 minutes), which works against intraday breakouts.
- The general lesson for the planned move to perps: **at taker cost, a strategy trading about once a day per asset needs a gross edge above 12 bp per round trip.** Nothing in the repo has shown a directional signal of that size.

## 5. Decision needed from Ivan

Accept NO_EDGE. The branch is closed.
