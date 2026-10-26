# 013 — Claude: TB 10 readout and TB 11 pre-registration — one account across 30 coins, risk rules R0–R4, long only and long + reverse

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-26 |
| **Branch** | Turtle Breakout |
| **Responds to** | TB 10 run `4a84770`; Ivan, 2026-10-26: keep the reversal; test both strategies with all proposed rules R0–R4 |
| **Status** | pre-registration + code (validated on synthetic data); awaiting Ivan's run |

## 0. TB 10 in brief (run `4a84770`): `NO_IMPROVEMENT`

Test set: the 29 coins other than BTC, at f = 2%, P sweep.

| | Result | p | Gate |
|---|---|---|---|
| H1: shallower max DD with reverse | 8 / 29 coins | 0.996 | FAIL; deeper on 21 |
| H2: higher growth with reverse | 19 / 29 coins | 0.068 | FAIL |
| H3: pooled sleeves growth | +0.45%/yr, Bonferroni lower bound −2.2% | — | FAIL |

**Findings.**
- **BTC was not typical:** it had the second-largest drawdown improvement of the 30 coins.
- **Median CAGR rises from 4.5% to 6.1%**, which is consistent with TB 07's +0.04R per reversal.
- **Above 5% risk the reversal ruins more coins:** 8 vs 4 at 7%, 13 vs 10 at 9%.
- **Per-coin ceiling:** the median coin (long only) is ruined at 10.5%; the highest f keeping DD above −50% is about 4.5%.
- **Pooled equal-weight sleeves at 2%, long only:** growth 8.2%/yr, max DD −10.1%, against −22% for the median single coin.

## 1. Question and prior

**Question.** In one account, TB 01's 2% per trade on 30 coins lets open risk stack when breakouts cluster. Does a portfolio rule allocate that risk better than simply trading the same exposure without the rule?

**Prior, stated in advance.**
- **Any rule that cuts exposure improves Calmar mechanically** when the account is above its growth-optimal risk. With 30 coins at 2%, open risk at a common rally can exceed the per-coin ruin levels of TB 10. The fair comparison is therefore against **R0m**: R0 with $f$ scaled to the rule's mean open risk.
- **Against R0m**, a rule only wins if *where* it puts risk matters, for example by skipping the tenth correlated breakout rather than shrinking all of them.
- **Expected:** R2 (correlation scaling) and R1 (heat cap) are the most likely to beat R0m, because they act exactly when breakouts cluster. R3 is coarser. R4 depends on per-coin trade Sharpe persisting, which TB 04 found weak.
- **Power:** the evaluation window is about 2.8 years; Calmar differences of a few tenths will not be significant.

## 2. Design (`turtle_portfolio.py`)

**Trades.** Per coin, the TB 01 daily longs and (variant `long_rev`) the TB 07 reverse short at each losing stop exit (`turtle_sizing.trade_list`).
- **Universe:** the TB 04 universe of 30 USDT perpetuals.
- **Start:** each coin's first bar + 30 days. TON is truncated at 2026-08-07.
- **Rules only size or skip trades;** entries and exits are unchanged.
- **A reverse short is taken even if its long was skipped,** since the signal is price, not the position.

**One account.**
- Equity is marked at every 17:00 New York close; sizing uses the last mark, as TB 01.
- Base risk $r = f\cdot\mathrm{HWM}$ with $f = 2\%$.
- Units $= r/(2\,\mathrm{ATR})$; notional ≤ 3× equity per position.
- **Open risk** = the sum of the risk at entry of the open positions (conservative: it does not shrink as trailing stops rise).

| Rule | Size of a new trade |
|---|---|
| **R0** | $r$ |
| **R1** heat cap | $\min(r,\ H\cdot\mathrm{HWM} - \text{open risk})$; skipped if below $0.25r$. $H$ from {4, 6, 8, 10, 15, 20, 30}% |
| **R2** correlation | $r / \big(1 + \sum_{j\in\text{open}} \max(0, d_i d_j \rho_{ij})\big)$, with $\rho$ the trailing 60-day correlation of daily log returns (≥ 30 obs; else the cross-sectional median; else 0.7). Shorts against correlated longs are not penalised |
| **R3** clusters | $K=4$ clusters, average linkage on $1-\rho$ (90 days, refitted monthly; coins with < 60 obs join BTC's cluster). Per cluster, $\min(r,\ C\cdot\mathrm{HWM} - \text{cluster open risk})$; skipped if below $0.25r$. $C$ from {2, 4, 6, 8}% |
| **R4** walk-forward weights | $r\cdot m_c$, with $m_c = \mathrm{clip}(s_c/\mathrm{median}(s), 0, 3)$ and $s_c$ = mean/sd of the coin's trade R over trades closed before the quarter (≥ 10 trades, else 1; all 1 if the median ≤ 0). Same variant's trades |

**Windows.**
- **Discovery:** 2020-01-01 → 2023-12-31. $H$ and $C$ maximise Calmar, per variant; ties go to the smaller value, and ruined runs are never chosen.
- **Evaluation:** 2024-01-01 → end. A fresh account with only trades entered in the window; positions open at the end are closed at the last close.
- **Caveat:** discovery-period data has been seen in TB 01–10, so the evaluation window is the only clean one.

## 3. Decision (binding, evaluation window)

For each rule in {R1(H*), R2, R3(C*)} × variant in {long, long_rev}, 6 comparisons in all:
- Calmar(rule) − Calmar(R0m), where R0m = R0 with $f' = f\cdot \overline{\text{open risk}}(\text{rule}) / \overline{\text{open risk}}(\text{R0})$ over the same window.
- Paired 21-day circular block bootstrap of the daily log returns (2,000 draws); Calmar is recomputed on each resampled path; one-sided $p = P(\Delta \le 0)$.
- Holm over 6 at $\alpha = 0.05$ → `IMPROVES` / `NO_DIFFERENCE`.
- **Verdict:** `RULE_IMPROVES: <rule/variant, …>` or `NO_RULE_BEATS_SAME_EXPOSURE`.

**Reported:**
- each rule against plain R0;
- R4;
- long + reverse minus long under each rule;
- equal-weight sleeves (TB 10 pool) in both windows;
- the full period 2020 → end;
- the 4-coin portfolio (BTC/ETH/SOL/XRP from 2020-10-15);
- CAGR, max DD, Calmar, Sharpe, ruin, max and mean open risk, max gross leverage, entries, skipped and scaled;
- evaluation equity, drawdown and open-risk curves.

## 4. Validation (synthetic)

`tests/test_turtle_portfolio.py`: 13 tests, all passing.
- **Single coin:** R0 reproduces `turtle_sizing.equity_from_trades` exactly.
- **Slack rules reduce to R0:** R1/R3 with a huge cap, R2 on one coin, and R4 without history are all identical to R0.
- **Rules act as designed:**
  - the heat cap is never exceeded and skips trades when binding;
  - correlation scaling lowers mean open risk without skipping;
  - tight cluster caps skip.
- **Mechanics:**
  - average linkage separates two correlated blocks;
  - R4 multipliers are bounded, and all 1 without history;
  - window filtering drops reverses whose long started before the window;
  - Calmar of a path, an identical-curve Calmar difference of 0, the selection tie rule, and Holm.

**Smoke run** on 10 correlated synthetic coins with shared trend blocks: the verdict is `NO_RULE_BEATS_SAME_EXPOSURE`. In that run the rules beat plain R0 on Calmar mostly by cutting exposure, and the gap shrinks against R0m. This is the reason R0m is the benchmark.

## 5. Run (PC, Windows CMD)

No download. Loading 30 coins takes about as long as TB 10's loading; the simulations take a few minutes.

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_turtle_portfolio.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 notebooks\Turtle_Breakout\11_risk_portfolio.ipynb
git add notebooks\Turtle_Breakout
git commit -m "Turtle Breakout 11: run"
git push
```

**Outputs** go to `notebooks/Turtle_Breakout/11/`: `11_decision.json`, `11_discovery.csv`, `11_evaluation.csv`, `11_tests.csv`, `11_reverse_vs_long.csv`, `11_sleeves.csv`, `11_full_and_4coins.csv`, `11_eval_curves.csv`, `11_coverage.csv`, `11_manifest.json`.

## 6. Decisions needed from Ivan

1. Confirm §3 is binding, including R0m as the benchmark.
2. Run §5.
