# Turtle Breakout discussions

Does the turtle breakout (Richard Dennis, in Rallio Trading's simplified long-only version) make money on crypto perpetuals after fees, slippage and funding, and does it beat random entries with the same exits (luck) and buy-and-hold at the same leverage (doing nothing)? Notebooks: `../../notebooks/Turtle_Breakout/`.

Independent branch: Binance USD-M 1m klines and funding history only.

## History

| Turn | Content |
|---|---|
| `001_claude_tb01_preregistration.md` | TB 01: exact rules, costs incl. funding, primary daily long-only 4-asset portfolio from 2020-10-15; G1 random-entry portfolios (N2), G2 Sharpe vs matched-leverage buy-and-hold; N1 trade twins, variants (no funding, short, long+short, 1h) reported |

| `002_claude_tb01_readout.md` | TB 01 result: `TREND_EDGE` as registered — CAGR 74.5%, Sharpe 1.57, 0/1,000 random portfolios as good; G2 not credible (never-rebalanced benchmark went below zero via funding on SOL's notional); concentration and survivorship open |
| `003_claude_tb02_preregistration.md` | TB 02: constant-mix benchmarks B1 perp monthly, B2 perp daily, B3 spot proxy monthly; strategy must beat all three on Sharpe (21d block bootstrap) |

| `004_claude_tb02_readout.md` | TB 02 result: `FAILS_B1_B2_B3` — Sharpe 1.57 vs 0.94–1.21, all differences positive but every 95% interval includes 0 |
| `005_claude_tb03_preregistration.md` | TB 03: 9 strategies = daily / 4h candles / last 4 hours × ATR / engine sizing / engine sizing + stop; per-strategy G1 (luck, net CAGR > 0) and G2 (TB 02 benchmarks), Bonferroni over 9; contrasts C1–C8 |

| `006_claude_tb04_preregistration.md` | TB 03 result in brief (no strategy beats the benchmarks; engine sizing and intraday windows do not help) and TB 04: winners vs losers of the original strategy — 41 entry features (price, flow/CVD, funding, engine, OI/positioning, basis, market), discovery on 26 other perps, confirmation on BTC/ETH/SOL/XRP; path features descriptive |

| `007_claude_tb05_preregistration.md` | TB 04 in brief (model-only confirmation, TON data artefact, no persistent regimes) and TB 05: original rules on 13 candle sizes 1 min → 1 day; fractal edge (R vs random entries, z-test) and tradeable edge (net bp, taker), Bonferroni over 13; bar-close fills |

| `008_claude_tb06_preregistration.md` | TB 05 in brief (not fractal; daily the only tradeable timeframe) and TB 06: daily turtle on BTC/ETH with a put struck at entry (38-day tenor from TB 04 discovery) and option-only versions (ATM call, call +1 ATR); DVOL-priced; Q1/Q2 Sharpe vs perpetual |


| `009_claude_tb07_preregistration.md` | TB 06 in brief (put hedge `WORSENS`, calls `NO_DIFFERENCE`) and TB 07: what price does after losing exits (event study vs winners and random flat days) and a stop-and-reverse short at each losing exit; G1 mean R > 0, G2 vs random flat-day shorts |


| `010_claude_tb08_preregistration.md` | TB 07 in brief (`BEATS_RANDOM_SHORTS_ONLY`: reversals +0.17R, beat random shorts p 0.03, G1 fails; bear-year concentration) and TB 08: risk per trade 1%–100% on BTC vs 2%; growth difference, Holm over 99; 3× cap and equity-sizing sweeps, Kelly bootstrap |


| `011_claude_tb09_note.md` | TB 08 result (`2PCT_NOT_BEATEN`; best in-sample 12% with −95% DD, ruin from 13% on HWM sizing; Kelly 23%, bootstrap 6–40%) and TB 09 (descriptive): the same 1%–100% sweep on long + stop-and-reverse vs long only |


| `012_claude_tb10_preregistration.md` | TB 09 in brief (BTC: long + reverse shallower DD at every f, 16.1% vs 12.3% CAGR at 2%; not significant) and TB 10: the same comparison on all 30 coins (test set excl. BTC; sign tests + pooled sleeves); risk-portfolio design options R0–R4 |

Next turn in this branch: `013`.

Procedure and numbering rules: `../README.md`.
