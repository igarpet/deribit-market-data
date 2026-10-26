# 015 — Claude: TB 12 readout and TB 13 pre-registration — the frozen specification on untouched data

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-26 |
| **Branch** | Turtle Breakout |
| **Responds to** | TB 12 run `e1db734`; Ivan, 2026-10-26: test now on (1) coins never used and (2) 2017–2019 spot data, before a forward paper test |
| **Status** | pre-registration + code (validated on synthetic data); awaiting Ivan's data download and run |

## 0. TB 12 in brief (run `e1db734`): `NO_RULE_BEATS_SAME_EXPOSURE`, no run ruined

**Rules vs same exposure.** At $f = 0.5\%$ (equity sizing) in the 2024 → end evaluation, every rule is at or below R0 run at the same mean open risk; none is significant.

| Calmar vs R0m | R1 | R2 | R3 |
|---|---|---|---|
| Long | −0.62 | −0.35 | −0.54 |
| Long + reverse | −0.07 | −0.49 | +0.08 |

- **Why:** clustered breakouts carry the winners. For example, long R1 skipped 459 trades and made CAGR 0.7%, against 11.1% for R0m.
- **Discovery again picked tight caps** that then failed.

**Base risk dominates.** R0 long:

| f | CAGR | max DD | Calmar |
|---|---|---|---|
| 0.25% | 16.9% | −25% | 0.67 |
| 0.5% | 26.9% | −42% | 0.64 |
| 1% | 31.9% | −65% | 0.49 |

**Long + reverse beats long** under every rule and every $f$ (15/15).
- R0 Calmar: 0.94 vs 0.67, 0.96 vs 0.64, 0.81 vs 0.49; drawdowns 4–7 points shallower.
- It holds at matched exposure: long + reverse at 0.25% (mean open risk 3.7%) scores 0.94, against long at 0.5% (3.9%) at 0.64.
- p ≈ 0.33, and the window was seen.
- Full period at 0.5%: long 41.2% / −41.9%; long + reverse 46.6% / −36.5%.

## 1. Frozen specification (no further changes)

**Rules.**
- One account.
- TB 01 daily longs (20-day high entry, max of the 2 ATR fixed stop and the 10-day-low trail).
- TB 07 reverse short at every losing stop exit (mirrored exits).
- No portfolio rule.
- Risk per trade $f \times$ current equity, primary $f = 0.5\%$ (0.25% and 1% reported).
- Notional ≤ 3× equity per position.

**Costs and data.**
- Costs: taker 0.05% + 1 bp per side; funding where it exists.
- 5m fills from 1m klines; TB 01–07 code unchanged.

## 2. Data no earlier decision used, chosen by rule

Universe selection is mechanical and runs before any strategy output (`tools/download_tb13_data.py screen`). The resulting `13/13_universe.json` is committed before the notebook runs.

**NEW: Binance USD-M USDT perpetuals outside the 30 coins of TB 04–12.**
- **Source:** the whole archive, **delisted contracts included** (survivorship).
- **Excluded:** stablecoins, index contracts (BTCDOM, DEFI, …), dated futures; `1000PEPE`-style names are mapped to their coin.
- **Data:** cut at the first gap > 7 days (a delisting; a relisted ticker may be another coin).
- **Eligible:** ≥ 400 days, first day ≤ 2025-06-30, median daily quote volume ≥ $10M.
- **Sample:** 60 drawn at random (seed 20261026) if more are eligible.
- **Window:** each coin's history after 30 days of warm-up.
- **Limits:** same calendar period as TB 01–12, and alts are correlated with the 30, so this tests the cross-section, not time. Smaller coins have worse real liquidity than the 1 bp slippage assumes.

**SPOT: Binance spot USDT pairs, 2017-08-17 → 2019-12-31.**
- This is before the perpetual era that every earlier test used, and it includes the 2018 bear market.
- **Eligible:** first day ≤ 2018-12-31, median daily quote volume ≥ $1M in the window.
- **Excluded:** stablecoins and leveraged tokens (UP/DOWN/BULL/BEAR); delisted pairs are included.
- **Funding:** 0, since spot has none. A real short needs margin borrowing, which is not modelled.
- **Sensitivity:** spot fees of the time (0.1% + 1 bp per side), and an assumed funding of +0.01% per 8 h (longs pay, shorts receive).

## 3. Decision (binding)

Per set, Bonferroni over 4 tests, $\alpha = 0.0125$ each, one-sided, 21-day circular block bootstrap (4,000 draws):

| | Test |
|---|---|
| **A** the strategy grows | long + reverse at $f = 0.5\%$: mean daily log growth, lower bound > 0 |
| **B** the reversal helps | Calmar(long + reverse) − Calmar(long) at $f = 0.5\%$, lower bound > 0 |
| **Verdict per set** | `HOLDS` (A and B) / `EDGE_ONLY` (A) / `REVERSAL_ONLY` (B) / `FAILS` |

**Reported:**
- $f = 0.25\%$ and $1\%$;
- equal-weight daily-rebalanced buy and hold of each set;
- long only at the mean open risk of long + reverse (the reversal effect net of exposure);
- calendar-year returns;
- per-coin trade-R sums;
- SPOT sensitivities;
- equity and drawdown curves.

**Prior.**
- **A on NEW:** TB 04's discovery set (26 non-core perps) averaged +0.44R per trade, so A is likely on NEW.
- **A on SPOT:** 2017 and early 2018 were the strongest trends in crypto history, and the turtle should catch them; 2018–19 was a long bear market where longs mostly stop out.
- **B** is the open question. Its mechanism (clusters of failed breakouts become clusters of shorts) predicts it should show most clearly in the 2018 bear market on SPOT.
- **Power is limited:** SPOT has about 2.3 years and fewer coins. A `FAILS` on B would not prove the reversal useless, but a pass would be the first evidence on data it was not tuned on.

## 4. Validation

`tests/test_turtle_oos.py`: 11 tests, all passing.
- Universe filters: candidates, stablecoins, index contracts, dated futures, leveraged tokens, `1000x` mapping.
- Gap cut.
- Screen thresholds.
- Seeded order-independent draw.
- Growth gate passes on drift and fails on noise.
- Calmar gate on identical curves.
- Verdict logic.
- Equal-weight hold.
- Archive listing pagination and kline parsing (ms/µs timestamps, with or without header), with network calls mocked.

**Not testable from Claude's environment:** it cannot reach data.binance.vision, so the first real run of the tool is Ivan's.

**Smoke run** (synthetic driftless coins): both sets `FAILS`; the notebook runs end to end.

## 5. Run (PC, Windows CMD)

**Step 1: choose the universes by rule, then commit them before any strategy output.**

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_turtle_oos.py" -v
python tools\download_tb13_data.py list
python tools\download_tb13_data.py screen
git add notebooks\Turtle_Breakout\13\13_universe.json
git commit -m "Turtle Breakout 13: universe by rule"
git push
```

**Step 2: download and run.**

```bat
python tools\download_tb13_data.py download
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 notebooks\Turtle_Breakout\13_frozen_spec_untouched_data.ipynb
git add notebooks\Turtle_Breakout
git commit -m "Turtle Breakout 13: run"
git push
```

- **Screening:** daily klines for several hundred symbols, roughly 10–30 minutes.
- **Download:** 1m data for about 60 perpetuals plus the spot pairs, a few GB; up to 1–2 hours.
- **All steps resume:** existing files are skipped.

**Outputs** go to `notebooks/Turtle_Breakout/13/`: `13_universe.json`, `13_decision.json`, `13_results.csv`, `13_benchmarks.csv`, `13_yearly.csv`, `13_coverage_{new,spot}.csv`, `13_curves.csv.gz`, `13_manifest.json`.
