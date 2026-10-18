# Funding Carry 003 — FC 02 Stage A pre-registration: predictability of funding, and the volatility engine

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-18 |
| **Branch** | Funding Carry |
| **Responds to** | FC 002 (carry return = funding; basis ≈ 0); Ivan's request: "can it be predicted and by how much", plus whether AM-T-HAR-Blend is a useful volatility engine here |
| **Status** | Pre-registration. Written before any real-data run. Code: `src/data_engineering/modeling/funding_predict.py` v1.0.0 |

## 1. Questions (binding answers, set before results)

- **Q1:** Is next-7-day funding predictable out of sample, and by how much?
- **Q2:** Which drivers add information beyond persistence?
- **Q3:** Is the volatility engine useful here?
  - **Q3a:** Does it forecast next-24h / next-7d realized variance on these 30 assets better than naive and daily-HAR benchmarks?
  - **Q3b:** Does it add information about future funding beyond naive realized volatility?

Stage A is **descriptive and statistical only**: it contains no trading rule. A timing rule built from it (Stage B) will be pre-registered separately and judged only on data after 2026-10-18.

## 2. Panel

- **Rows:** one per asset and day, at origin $b$ = 00:05 UTC. Features use data with timestamps $\le b$; targets use $(b, b+h]$.
- **Assets:** the FC 01 universe (30).
- **Data:** Binance spot 1m, USD-M perpetual 1m, funding prints, and USD-M **metrics** (5-minute open interest and long/short ratios). Metrics come from the new `tools/download_binance_um_metrics.py`.

**Targets:** annualized funding, the sum of prints × 365/days.
- $y_7$ is the primary target;
- $y_1$ and $y_{30}$ are reported.

**Feature blocks**, added in this fixed order. Every model, including the benchmark, has asset fixed effects.

| Block | Features |
|---|---|
| P persistence | trailing annualized funding $f_1, f_7, f_{30}$ |
| PR premium | $\log(F/S)$ at $b$, and its 24h mean on the 5m grid (bp) |
| M momentum | 7d and 30d spot log return (%) |
| OI positioning | 1d / 7d log change of OI value; 24h means of log account, top-trader and taker long/short ratios |
| B breadth | cross-asset median $f_7$; share of assets with $f_7 \ge 10\%$ |
| V naive volatility | log trailing spot realized variance, 1d and 7d |
| E **engine** | log AM-T-HAR-Blend variance forecast for the next 24h and 7d |

**Engine definition.** The engine's variance level is the VD18 HAR_4, which VD19 and VD20 used as "ENGINE". Its Markov part only shapes the mixture, with blend weight ≈ 0.98 on HAR.
- Refitted with VD18's own `fit_har` / `predict_har`: log-RV lags 30/60/240/1440 min, asset effects, time of day, ridge 0.02, TRAIN net-return scale.
- Data: Binance **spot** 5m returns of the 30 assets.
- Fitted at $h$ = 1440 and 10080 min, on TRAIN bars only (target window ends before the split).
- **Deviation from VD18:** fitted on every 3rd TRAIN bar to fit the PC's memory. A ridge fit on about 1.2M bars instead of 3.6M changes the estimate negligibly.
- **Disclosure:** the engine's longest lag is 1 day, so its 7-day forecast extrapolates short memory. That is the engine as it is, and it is part of what Q3a tests.

**OI block rule:** included only if at least 80% of rows complete on all other blocks also have OI. Otherwise it is dropped and reported.

## 3. Split and estimation

- **TRAIN:** origins before 2025-10-01 00:05 whose target window ends by then. **TEST:** origins from 2025-10-01 00:05 to the end of funding data.
- **Method:** pooled OLS with asset fixed effects. Features are winsorized at TRAIN 1%/99%; the target at TRAIN 0.5%/99.5% (the same fixed bounds are applied in TEST).
- **Common rows:** all nested models are compared on the same rows (complete on the full model).
- **$R^2_{OOS}$** = $1 - \text{SSE}_\text{model}/\text{SSE}_\text{FE}$ on TEST, where FE (asset means) is the benchmark.
- **Inference:** mean squared-error difference, with a 14-day date-block bootstrap ($B = 2000$, 95%), because targets overlap.

## 4. Binding answers

| Q | Answer "yes" if |
|---|---|
| Q1 predictable | P beats FE on TEST: bootstrap lower bound of $\Delta$MSE $>0$. "By how much" is reported as $R^2_{OOS}$ of P and of the full model, plus TEST quintiles of predicted $y_7$ against realized $y_7$. |
| Q2 block adds | in the nested sequence, the block's $\Delta$MSE lower bound $>0$. Unique contribution (full minus block) is reported. |
| Q3a engine forecasts volatility better | QLIKE of ENGINE below **both** NAIVE (trailing RV, same horizon) **and** HAR_D (daily HAR on 1d/7d/30d log RV, fitted on TRAIN), lower bound $>0$, at 24h and 7d separately |
| Q3b engine helps predict funding | base (all non-volatility blocks) + V + E beats base + V, lower bound $>0$ |

"Engine useful for carry" = Q3b. A better volatility forecaster that adds nothing about funding is not useful *for this strategy*.

## 5. Validation before this turn (`tests/test_funding_predict.py`, 4 tests pass; synthetic smoke run of the notebook)

- **Exact accounting:** funding windows, annualization and the premium are computed exactly. Targets past the data end are NaN.
- **Causality:** changing prices or funding after $b$ leaves every feature at $b$ unchanged.
- **Metrics features:** match planted values.
- **Engine scale:** the engine refit runs, and its Mincer-Zarnowitz slope on synthetic data is about 1, so it is calibrated in scale.
- **Recovery:** planted persistence ($R^2_{OOS} > 0.5$) and planted premium information are both recovered by the nested test.

## 6. My prediction (opinion, before the run)

- **Q1:** yes, strongly. Persistence alone should give $R^2_{OOS}$ around 0.3–0.6 at 7 days, lower at 30 days.
- **Q2:** the premium adds (funding is computed from it); breadth and OI might add a little; momentum adds little beyond them.
- **Q3a:** likely yes at 24h (VD20 found 0.200 vs 0.296 QLIKE against daily HAR); uncertain at 7d.
- **Q3b:** **no**. Volatility should carry little information about funding once funding's own history and the premium are known.

## 7. Running it (PC, Windows CMD)

First download the metrics, once. Expect roughly 30 assets × 840 days of small files, run in parallel; re-runs skip days already present.

```bat
cd /d D:\Projects\deribit_market_data
git pull
python tools\download_binance_um_metrics.py --start 2024-07-01
python -m unittest tests.test_funding_predict -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 notebooks\Funding_Carry\02_funding_predictability.ipynb
git add notebooks\Funding_Carry
git commit -m "FC 02 run"
git push
```

## Questions for ChatGPT

1. Should Q3b condition on PR as well? It does, because "base" includes every non-volatility block. Is that the right bar for the engine, or too high?
2. Are the 4h-funding assets (TON, WIF, TIA, ENA) handled correctly by summing prints, or do you see a reason to treat them separately?

## Decisions needed from Ivan

1. Approve this pre-registration (or amend it before running).
2. Run the metrics download first. If Binance has no metrics files for some assets or dates, the OI block may be dropped by the 80% rule. That is reported, not worked around.
