# Multi-Scale HMM 001 — Branch pre-registration: momentum / reversal states on 1h, 4h and 1D bars

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-17 |
| **Branch** | Multi-Scale HMM |
| **Responds to** | Ivan's request (2026-10-17): let a model find momentum / reversal states by itself on bars longer than 5 minutes, with the bars built from 1-minute data so the short-term structure is kept |
| **Status** | Pre-registration. Written and pushed before any real-data run. Code: `src/data_engineering/modeling/multiscale_hmm.py` v1.0.0 |

## 1. Question and why this is not a repeat

Do latent states, discovered by an HMM on 1h, 4h or 1D bars of BTC, ETH, SOL and XRP perpetuals, predict the **sign and size of the next bar's return** well enough to trade net of costs?

What the project has **not** tested before:
- **Regime Structure RS01** tested Ivan's *own* label definitions on 15m–4h windows. Nothing was discovered by a model.
- **Return State Discovery / Regime Discovery** used unsupervised models at intraday resolution only.
- **Breakout Trend** was one fixed hourly rule.

The horizon argument: with hit rate $p$, the gross edge per trade is about $(2p-1)\,\mathbb{E}|r_T| - c$, where $\mathbb{E}|r_T|\propto\sigma\sqrt{T}$ grows with the horizon and the cost $c$ does not. At 5 minutes, break-even needs an impossible $p$; at one day it needs $p\gtrsim0.53$.

## 2. Data

- **Prices and flow:** Binance USD-M perpetual 1m klines (`<ASSET>USDT_um_1m.parquet`, `tools/download_binance_um_klines.py`) for BTC, ETH, SOL and XRP. Bars are built **only** from these minutes.
- **Funding:** `basis/funding/<ASSET>USDT_funding.parquet` (`tools/download_binance_basis_data.py`).
- **Proxy disclosure (§8 of the discussions README):** this is a Binance execution proxy, not Deribit executable prices. A pass here would still need a Deribit perpetual check before anything else.

**Windows (bar end times, UTC):**

| Window | Range | Use |
|---|---|---|
| TRAIN | 2024-10-01 → 2025-09-01 | HMM fit for model selection |
| VALID | 2025-09-01 → 2026-02-01 | choose $K$ and $\nu$ |
| TEST | 2026-02-01 → 2026-10-18 | binding verdicts S and E |
| PROSPECTIVE | from 2026-10-18 | verdict P, later notebook |

The notebook truncates the minute data at the end of TEST, so prospective data is never touched.

Disclosure: other branches (BT, PSH, RS) have already looked at 2026 data with *different* models. That is why the prospective verdict P exists.

## 3. Bars and intrabar emissions

- **Bars:** UTC-aligned 60, 240 and 1440 minutes. A bar is valid with at least 95% of its minutes present.
- **Returns:** minute log returns $r_i$, with the first minute of a bar measured from the previous bar's close, so $\sum_i r_i = \log C_t - \log C_{t-1}$.
- **Volatility:** $\sigma_t^2$ is an EWMA of $RV$ with a half-life of 24 bars (1h), 18 bars (4h) or 10 bars (1D).

| Emission | Definition | Captures |
|---|---|---|
| `z` | $r_t/\sigma_{t-1}$ | standardized bar return |
| `lrv` | $\log(RV_t/\sigma_{t-1}^2)$, $RV=\sum r_i^2$ | volatility surprise |
| `lsig` | $\log\sigma_t$ | volatility level (persistent regimes) |
| `semi` | $(RS^+-RS^-)/RV$ | up vs down semivariance |
| `er` | $r_t/\sum\lvert r_i\rvert$ | signed efficiency: trend vs chop inside the bar |
| `clv` | $(2C-H-L)/(H-L)$ | close location (wicks) |
| `wick` | $1-\lvert C-O\rvert/(H-L)$ | wick share of the range |
| `jump` | $\max(0,1-BV/RV)$, $BV=\tfrac{\pi}{2}\sum\lvert r_i\rvert\lvert r_{i-1}\rvert$ | jump share of variance |
| `flow` | $\sum(2\,TBQ-QV)/\sum QV$ | signed taker imbalance |
| `lvol` | $\log(QV_t/\overline{QV}_{t-1})$ | relative activity |

- **Standardization:** each emission is z-scored per asset with the fit window's mean and sd, then clipped at ±8.
- **Warm-up:** the first $3\times$half-life bars per asset are discarded.
- **Amendment before any run:** `lsig` was added after the synthetic check showed that dividing by $\sigma_{t-1}$ alone hides persistent volatility regimes.

## 4. Model

- **Emissions:** multivariate Student-t HMM, full covariance (ridge $10^{-3}$), one shared $\nu$, EM with the scale-mixture weights.
- **Pooling:** one model per scale, pooled over the four assets (one sequence per contiguous run of bars).
- **Selection:** $K\in\{1,\dots,6\}$ for 1h and 4h, $\{1,\dots,4\}$ for 1D; $\nu\in\{4,7,15,50\}$. Fitted on TRAIN (6 restarts); the pair with the highest one-step predictive log-likelihood per bar on VALID wins. If the choice lands on the grid edge, it is reported, not extended.
- **Frozen model:** refitted on TRAIN+VALID and saved to `01/01_frozen_model_<scale>.json`.

## 5. Forecast: momentum, reversal and drift per state

- $\pi_{t,k}$ = filtered state probability at the close of bar $t$ (causal forward filter).
- Per state, weighted least squares on TRAIN+VALID with weights $\pi_{t,k}$:

$$\frac{r_{t+1}}{\sigma_t} = a_k + b_k\,z_t + \varepsilon,\qquad y\ \text{winsorized at}\ \pm6\ \text{for fitting}$$

  - $b_k>0$: **momentum** state. $b_k<0$: **reversal** state. $a_k$: drift.
- Forecast:

$$\hat m_{t+1} = \sigma_t\sum_k \pi_{t,k}\,(a_k + b_k z_t)$$

- **Baselines (reported, not gates):**
  - $K=1$: the same regression without states.
  - Time-series momentum: the sign of the trailing 1-day (1h), 5-day (4h) or 20-day (1D) return.

## 6. Trading rules on TEST

Equal weight across the four assets, one unit of notional each, not compounded.

- **Always-in:** $s_t=\operatorname{sign}(\hat m_{t+1})$. This gives the gross statistic for verdict S.
- **Net rule:** $pos_t=\operatorname{sign}(\hat m_{t+1})\cdot\mathbf 1\{\lvert\hat m_{t+1}\rvert\ge\tau\}$, with $\tau = 12$ bp (the round-trip cost).
  - Cost: 6 bp per side (5 bp taker + 1 bp slippage, as in BT 01) per unit of $\lvert pos_t-pos_{t-1}\rvert$.
  - Positions are closed at data gaps.
  - A long position pays the actual Binance funding at every funding time in $(t,t+1]$.

## 7. Null

Every 1-minute return gets a random sign, **shared across assets at the same minute**. This is the RS v1.1 lesson: crypto assets co-move, so independent signs would make the null too narrow.
- Minute open/high/low are mirrored with the sign; taker buy volume becomes $QV-TBQ$.
- This keeps every absolute move and all volatility clustering, and destroys all directional information.
- Bars, emissions, the HMM fit ($K$ and $\nu$ fixed at the real-data choice, 3 restarts), the regressions and the forecasts are all recomputed for every draw.
- Draws: 99 (1h), 199 (4h), 199 (1D).
- Known bias: $K$ and $\nu$ are not re-selected per draw. This slightly favours the real data. I accept it for runtime and state it here.

## 8. Verdicts per scale (binding, set before results)

| Verdict | Pass condition |
|---|---|
| **S** statistical | always-in gross bp/bar (pooled) $>0$, sign-flip one-sided $p\le 0.05/3$, gross $>0$ on ≥ 3 of 4 assets, **and** above the drift-only position (each asset held in the sign of its TRAIN+VALID mean return throughout TEST) |
| **E** economic | net rule: day-block bootstrap (1-day blocks; 7-day blocks for 1D; $B=2000$) lower 2.5% bound of mean daily net P&L $>0$, net total $>0$ on ≥ 3 of 4 assets, funding available |
| **States add value** | same net rule, HMM minus $K=1$: bootstrap lower bound $>0$ (reported; not required for S or E) |
| **P** prospective | only for scales passing S and E: frozen model, data after 2026-10-18; after at least 60 (1h), 90 (4h) or 180 (1D) days, mean daily net $>0$ |

- A scale is `CANDIDATE_FOR_PROSPECTIVE` only if S **and** E pass. Otherwise the scale verdict is `NO_TRADEABLE_MOMENTUM_STATES`.
- If no scale passes, the branch verdict is `NO_TRADEABLE_MOMENTUM_STATES` and the branch closes.
- Kelly sizing with the volatility engine is **not** part of 01. It is only built for a scale that later passes P.

**Power warning:** 1D TEST has about 258 bars per asset. A real daily effect of moderate size can fail E for lack of data. The prospective window cannot fix that quickly (180 days).

## 9. Validation done before this turn (synthetic data, `tests/test_multiscale_hmm.py`, 8 tests pass)

- **Forward-backward:** matches brute-force enumeration (error $2\times10^{-16}$).
- **Bars:** bar return equals the sum of minute returns; RV is exact; coverage rule enforced.
- **Sign flip:** $\lvert r\rvert$ preserved, signs shared across assets, OHLC remains consistent, taker flow flipped.
- **Causality:** changing data after a cut leaves every earlier emission unchanged.
- **Student-t HMM:** recovers planted means and persistence.
- **P&L accounting:** costs at flips, at gaps and at the end of data; funding window correct.
- **End-to-end:** with hourly momentum planted in a high-volatility state, the model labels that state "momentum" and the other "no-autocorrelation". S ≈ +13 bp/bar against nulls below 1 bp, and states add value over $K=1$. With nothing planted, S ≈ 0 and the net rule does not trade.

## 10. My prediction (opinion, recorded before the run)

- **1h and 4h:** `NO_SKILL`. Everything at 5m–4h so far points to direction being efficient within costs. I expect the HMM to find volatility states with $b_k\approx0$.
- **1D:** uncertain. Time-series momentum is documented at daily-to-weekly horizons, but the TEST window is short. My guess is a positive but non-significant S and a failed E.

## 11. Running it (PC, Windows CMD)

Close the notebook in Jupyter first. Expected runtime: under an hour on 3 worker processes.

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest tests.test_multiscale_hmm -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 notebooks\Multi_Scale_HMM\01_multiscale_hmm.ipynb
git add notebooks\Multi_Scale_HMM
git commit -m "MSH 01 run"
git push
```

If the funding files are missing, run `python tools\download_binance_basis_data.py --assets BTC ETH SOL XRP --months 26` first.

## Questions for ChatGPT

1. The shared-sign null destroys long-run drift, so a model that is simply long in a rising market could beat it. I added a drift-only benchmark that S must also beat (§8). Is that sufficient, or would you de-mean returns for the null at 1D? Propose it before the run if so.
2. Do you see a leakage path in standardizing with fit-window statistics, then filtering through TEST with frozen parameters?

## Decisions needed from Ivan

1. Approve this pre-registration as binding, or amend it **before** running.
2. Confirm the four assets and that the funding files exist under `Binance\basis\funding\`.
