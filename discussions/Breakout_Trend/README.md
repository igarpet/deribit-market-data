# Breakout Trend discussions

Does a third-party hourly breakout strategy (originally built on US index futures and reviewed on YouTube) make money after Deribit costs on crypto perpetuals, does its entry beat a coin flip, and how does buying ATM options on the same signals compare? Notebooks: `../../notebooks/Breakout_Trend/`.

Independent branch: it uses only the shared 5-minute OHLC history and the collector's option quotes, no other branch's outputs.

## History

| Turn | Content |
|---|---|
| `001_claude_branch_preregistration.md` | Rules (with Ivan's clarifications and two flagged assumptions), costs, model-priced and real-quote options leg, dev/test split, coin-flip null, pooled test-window decision |
| `002_claude_source_change_perp.md` | Amendment before any run: Binance USD-M perp 1m → 5m OHLC, BTC/ETH/SOL/XRP, from 2024-10-01; everything else unchanged |
| `003_claude_bt01_readout.md` | BT 01: NO_EDGE (pooled test −135%, Sharpe −1.66 vs coin-flip median −0.88; 89.5% of random runs as good); gross +6.8 bp (dev, n.s.) / −0.2 bp (test) per trade vs 12 bp round-trip cost; branch closed |

Next turn in this branch: `004`.

Procedure and numbering rules: `../README.md`.
