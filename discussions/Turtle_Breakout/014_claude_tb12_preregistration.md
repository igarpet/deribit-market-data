# 014 — Claude: TB 11 readout and TB 12 pre-registration — the risk portfolio at a survivable base risk

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-26 |
| **Branch** | Turtle Breakout |
| **Responds to** | TB 11 run `b36a318`; Ivan, 2026-10-26: fix the design error even though the evaluation window has been seen |
| **Status** | pre-registration + code (validated on synthetic data); awaiting Ivan's run |

## 0. TB 11 readout (run `b36a318`): `NO_RULE_BEATS_SAME_EXPOSURE`. The test was degenerate, and the error is mine

**The error.**
- TB 11 risked 2% of the HWM per trade on each of 30 coins in one account. That is about 30× the risk of the TB 10 equal-weight sleeves, whose worst evaluation day was −4.7%.
- HWM sizing compounds it: in a drawdown, each new trade still risks 2% of the old peak.
- TB 08–10 had already shown per-coin ruin at about 10–13%; stacking 30 coins at 2% goes far past that. The base risk should have been set relative to the number of coins before pre-registering.

**Evaluation 2024 → end:**

| | Long | Long + reverse |
|---|---|---|
| R0 | ruined 2024-07 | ruined 2024-10 |
| R1 (H = 20% / 30%) | ruined 2026-02 | ruined 2025-11 |
| R2 | CAGR 14.0%, DD −52% | CAGR 14.7%, DD −37% |
| R3 (C = 8% / 4%) | ruined 2026-02 | CAGR 26.1%, DD −50% |
| R4 | ruined 2024-08 | ruined 2024-11 |
| Sleeves (TB 10) | 4.5%, DD −9.3%, Calmar 0.48 | 5.3%, DD −8.1%, Calmar 0.65 |

**What it means.**
- **The tests compare ruin with ruin:** 5 of 6 R0m benchmarks were ruined, so their Calmar differences are 0. Long R2 vs R0m is +1.26, raw p 0.03, Holm 0.19.
- **R2 is the only rule that survived every run.** It cuts size exactly when correlated breakouts stack.
- **Discovery chose the loosest caps** (bull market 2020–23), and they went bust after 2024.
- **The sleeves beat every surviving one-account rule on Calmar** in evaluation: exposure dominated allocation.
- **4-coin portfolio:** the rules never bind (max open risk 8%). R0 long 70.5% CAGR, −36%; long + reverse 81.9%, −34%.

## 1. TB 12: what changes, and what does not

**Two changes only:**
1. **Sizing on current equity.** Risk per trade, the R1 cap and the R3 caps are all measured on the last marked equity, not the HWM. TB 08 EQ survived to far higher risk than HWM sizing.
2. **Base risk per trade** $f \in \{0.25\%, 0.5\%, 1\%\}$, **primary $f = 0.5\%$**, about $2\%/\sqrt{16}$: with ~16 coins trading on average, that is the per-coin risk whose sum matches one 2% position under independence. The grids scale with $f$:

$$H \in \{2, 3, 4, 5, 7.5, 10, 15\}\cdot f,\qquad C \in \{1, 2, 3, 4\}\cdot f,$$

the same number of positions as TB 11's grids (TB 11's H = 4%…30% was 2…15 positions of 2%).

**Unchanged from TB 11 (`013` §2–3):**
- books (30 coins, long and long + reverse);
- rules R0–R4 and their constants;
- windows: discovery 2020–2023, evaluation 2024-01-01 → end;
- selection of H and C by discovery Calmar;
- the test: Calmar(rule) − Calmar(R0m), R0m = R0 at the rule's mean open risk, paired 21-day block bootstrap, Holm over 6.

**Decision (binding):** at $f = 0.5\%$, the 6 comparisons (R1, R2, R3 × long, long_rev), Holm at $\alpha = 0.05$ → `RULE_IMPROVES: …` / `NO_RULE_BEATS_SAME_EXPOSURE`.

**Reported:**
- $f = 0.25\%$ and $1\%$, with the same tests (Holm within each $f$);
- each rule against plain R0, and R4;
- long + reverse minus long for every rule and $f$;
- sleeves;
- full period and 4-coin portfolio at 0.5%;
- curves, and Calmar against $f$.

**Caveat, stated plainly:** the evaluation window was seen in TB 11. TB 12 is a repaired test, not a fresh one; its p-values overstate the evidence by an amount that cannot be quantified. A clean confirmation needs data after 2026-10-26.

**Prior.**
- At 0.5% on equity, R0's open risk should peak at roughly 10–15% of equity; ruin is unlikely.
- R2's TB 11 advantage could have been only survival. Here every run should survive, so this is the first fair look.
- Expected result: R2 ahead of R0m on Calmar, but not significant over 2.8 years.

## 2. Validation

`tests/test_turtle_portfolio.py`: 15 tests, all passing (13 from TB 11 + 2 new):
- single-coin equity sizing reproduces `turtle_sizing.equity_from_trades` (EQ) exactly;
- the R1 cap binds on the equity base.

TB 11 defaults (`size_base = "hwm"`) are unchanged.

**Smoke run** on 10 synthetic coins: no run is ruined at any $f$. The verdict is `NO_RULE_BEATS_SAME_EXPOSURE`; R2 is ahead of R0m by about +0.25–0.30 Calmar, not significant.

## 3. Run (PC, Windows CMD)

No download; similar duration to TB 11 (three values of $f$).

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_turtle_portfolio.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 notebooks\Turtle_Breakout\12_risk_portfolio_equity_sizing.ipynb
git add notebooks\Turtle_Breakout
git commit -m "Turtle Breakout 12: run"
git push
```

**Outputs** go to `notebooks/Turtle_Breakout/12/`: `12_decision.json`, `12_discovery.csv`, `12_evaluation.csv`, `12_tests.csv`, `12_reverse_vs_long.csv`, `12_sleeves.csv`, `12_full_and_4coins.csv`, `12_eval_curves.csv.gz`, `12_coverage.csv`, `12_manifest.json`.
