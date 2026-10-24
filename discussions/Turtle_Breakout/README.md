# Turtle Breakout discussions

Does the turtle breakout (Richard Dennis, in Rallio Trading's simplified long-only version) make money on crypto perpetuals after fees, slippage and funding, and does it beat random entries with the same exits (luck) and buy-and-hold at the same leverage (doing nothing)? Notebooks: `../../notebooks/Turtle_Breakout/`.

Independent branch: Binance USD-M 1m klines and funding history only.

## History

| Turn | Content |
|---|---|
| `001_claude_tb01_preregistration.md` | TB 01: exact rules, costs incl. funding, primary daily long-only 4-asset portfolio from 2020-10-15; G1 random-entry portfolios (N2), G2 Sharpe vs matched-leverage buy-and-hold; N1 trade twins, variants (no funding, short, long+short, 1h) reported |

| `002_claude_tb01_readout.md` | TB 01 result: `TREND_EDGE` as registered — CAGR 74.5%, Sharpe 1.57, 0/1,000 random portfolios as good; G2 not credible (never-rebalanced benchmark went below zero via funding on SOL's notional); concentration and survivorship open |
| `003_claude_tb02_preregistration.md` | TB 02: constant-mix benchmarks B1 perp monthly, B2 perp daily, B3 spot proxy monthly; strategy must beat all three on Sharpe (21d block bootstrap) |

Next turn in this branch: `004`.

Procedure and numbering rules: `../README.md`.
