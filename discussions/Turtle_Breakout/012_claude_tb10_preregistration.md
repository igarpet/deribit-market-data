# 012 — Claude: TB 09 readout and TB 10 pre-registration — long vs long + stop-and-reverse on all 30 coins; the risk-portfolio idea

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-26 |
| **Branch** | Turtle Breakout |
| **Responds to** | TB 09 run `73f8631`; Ivan, 2026-10-26: "Lets test on all", plus an idea: build a risk portfolio that allocates account risk efficiently across coins, because most rallies happen at the same time |
| **Status** | TB 10 pre-registration + code (validated on synthetic data); risk portfolio: design options, awaiting Ivan's choice |

## 0. TB 09 in brief (run `73f8631`, BTC, 54 longs + 33 reverse shorts, descriptive)

| At f = 2%, P sweep | Long only | Long + reverse |
|---|---|---|
| CAGR | 12.3% | 16.1% |
| Max DD | −20.8% | −16.4% |
| Sharpe | 1.00 | 1.12 |
| Calmar | 0.59 | 0.99 |
| First ruin (P) | 13% | 19% |
| Highest f with max DD above −50% (P / EQ) | 5% / 7% | 7% / 10% |

- **Drawdown is shallower at every f in all three sweeps.** This was the opposite of my expectation (TB 011 §1).
- **Mechanism:** the reversals paid in the long's bad years. 2022: long −9.7%, combined +4.3%. BTC reversals by year: 2021 +4.1R, 2022 +4.5R, 2026 +3.5R; 2023–25 −3.1R.
- **Not evidence:**
  - The paired growth-difference 95% interval includes 0 at every f below the long-only ruin point; at 2% it is +3.4%/yr, with an interval of [−1.9%, +8.9%].
  - The top 5 reversals give 12.7R of the 8.95R total.
  - BTC was the best confirmation coin for reversals (+0.30R vs +0.06 to +0.15R for the others).
  - TB 07's G1 failed.

## 1. TB 10: does the BTC picture hold on the other coins?

**Design.** TB 09 on every coin of the TB 04 universe (30 USDT perpetuals), each coin a separate account.
- **Window:** each coin's data after a 30-day warm-up. TON is truncated at 2026-08-07 (flat-series artefact).
- **Trade lists:** TB 01 longs, and longs + TB 07 reverse shorts.
- **Sizing:** $f = 1\%,\dots,100\%$ of the HWM per trade.
- **Sweeps:** P / C3 / EQ, as TB 08–09. In total 30 × 2 × 3 × 100 = 18,000 backtests.

**Prior.**
- TB 07 put the mean reversal at +0.04R on the 26 discovery coins and between +0.06R and +0.15R on ETH/SOL/XRP, against +0.30R on BTC.
- With reversal R near zero, extra stopped-out trades should mostly add noise. Expect a mix: some coins shallower, some deeper, and no clear growth gain.
- The drawdown hedge could still show up where the reversals come in bear years, as on BTC.

**Decision (binding).**
- **Test set:** the 29 coins other than BTC. BTC chose the experiment, so it is excluded.
- **Setting:** $f = 2\%$, P sweep. Bonferroni over 3 tests, so $\alpha = 0.05/3$ each.

| | Test |
|---|---|
| **H1** drawdown | number of coins with a shallower max DD with reverse; one-sided sign test |
| **H2** growth | number of coins with higher log growth; one-sided sign test |
| **H3** pooled growth | equal-weight sleeves (each coin its own account; the pool's daily log return is the mean over coins trading that day): paired growth difference, 21-day block bootstrap, lower bound at $\alpha/3$ > 0 |
| **Verdict** | `REVERSE_HELPS` (H1, H2, H3) / `HEDGE_ONLY` (H1 only) / `NO_IMPROVEMENT` |

**Reported:**
- per-coin table at 2% (CAGR, DD, Sharpe, Calmar, ruin);
- confirmation (ETH/SOL/XRP) and discovery subsets;
- for every f: count of coins with a shallower DD and with higher growth, median DD and CAGR, coins ruined;
- per coin and sweep: first ruin, best f, highest f with DD above −50%;
- pooled sleeves at 2/5/10% (growth, DD, Sharpe);
- scatter of ΔDD vs Δgrowth per coin.

**Validation.** `tests/test_turtle_sizing.py`, 19 tests (4 new):
- sign-test values;
- pooled comparison of identical curves = 0, and a known drift difference is detected;
- late-entering coins are pooled correctly;
- decision logic.

The 8-coin synthetic smoke run (no reversal edge) gives `NO_IMPROVEMENT`, with deeper drawdowns with reverse on most coins, as it should.

## 2. The risk-portfolio idea

**The premise is right, and it matters more than the reversal.**
- Breakouts across crypto cluster: when BTC breaks a 20-day high, most alts do within days.
- At 2% per position on 30 coins, the open risk at a common rally can reach several tens of percent of the HWM. A failed rally then stops all of them out together.
- That cluster is the portfolio's real drawdown, and per-coin $f$ does not see it.
- TB 01's 4-coin portfolio already showed this: its drawdowns were driven by simultaneous stops.

**Where I disagree: "maximize returns and minimize drawdowns" needs one objective, and optimized weights will overfit.**
- **Objective.** Pick one, for example maximum growth subject to max DD ≤ X, or maximum Calmar.
- **Sample size.** With about 25–50 trades per coin over 3–6 years, any optimized weight per coin, such as mean-variance on the coin sleeves, is fitted mostly to noise. The weights must be estimated walk-forward (only past data) and compared out of sample against simple rules.

**Candidate rules, from robust to fitted:**

| | Rule | Fitted parameters |
|---|---|---|
| R0 | equal risk per trade, no portfolio limit (TB 01 rules on 30 coins) | none, baseline |
| R1 | **heat cap**: total open risk ≤ H% of the HWM; new entries are skipped or scaled when the cap binds | H |
| R2 | **correlation-scaled entries**: a new position's risk is divided by the effective number of independent bets among the open positions (trailing 60-day correlations) | none (window fixed) |
| R3 | **cluster caps**: coins grouped by trailing correlation (e.g. BTC-beta buckets), each cluster limited to a share of total risk | number of clusters, cap |
| R4 | **walk-forward optimized weights**: per-coin risk weights refitted quarterly on past sleeve returns (min CVaR or max growth with a DD constraint), applied out of sample | weights |

**What it needs.** A joint multi-coin simulator with the reverse shorts included. `turtle_breakout.simulate` already runs a joint long-only portfolio; the reversal leg and the rules above have to be added.

**Proposal.** After TB 10's verdict, decide long-only or long + reverse as the base. Then:
- TB 11 = R0–R3 on the 26 discovery coins (rules and H chosen there);
- confirmed on BTC/ETH/SOL/XRP;
- R4 reported as walk-forward only.

## 3. Run TB 10 (PC, Windows CMD)

No download. About 30–60 minutes (loading 30 coins plus 18,000 backtests).

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_turtle_sizing.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 notebooks\Turtle_Breakout\10_reverse_all_coins.ipynb
git add notebooks\Turtle_Breakout
git commit -m "Turtle Breakout 10: run"
git push
```

**Outputs** go to `notebooks/Turtle_Breakout/10/`: `10_decision.json`, `10_per_coin_2pct.csv`, `10_by_f.csv`, `10_ruin.csv`, `10_sweeps_all.csv.gz`, `10_curves_P.csv.gz`, `10_coverage.csv`, `10_manifest.json`.

## 4. Decisions needed from Ivan

1. Confirm §1 is binding.
2. Run §3.
3. Choose the scope of the risk portfolio (§2).
