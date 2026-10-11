# 003 — Claude: notebook 02 — roughness of price and volatility (descriptive)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-11 |
| **Branch** | Hurst Regimes |
| **Responds to** | Ivan, 2026-10-11: "forget about straddles … analyse pure roughness and H exponent first" |
| **Status** | Built (`hurst_roughness.py` v0.1.0), tests 6/6, verified on synthetic rough-volatility data with known $H$. Not yet run on real data. **Descriptive: no gate, no trading claim.** |

## 1. Questions

1. **Price.** How does $H$ of the log price change with horizon: 5–30 min (microstructure), 1–8 h, 1–7 days?
2. **Volatility.** How rough is $\log\sigma$, at 1h, 4h and 1d resolution? Is it monofractal ($H_q=\zeta(q)/q$ constant in $q$), as in Gatheral, Jaisson & Rosenbaum (2018)?
3. **Mean reversion.** Does the autocorrelation of $\log\sigma$ decay as a power law (rough / long memory) or exponentially (OU)? What does the fOU of the thesis give?
4. **Stability.** Are these numbers stable across half-years?

## 2. Method

- **Structure functions.** $m_q(\Delta)=\overline{|x_{t+\Delta}-x_t|^q}$ for $q\in\{0.5,1,1.5,2,3\}$; $\zeta(q)$ is the log-log slope, and $H=\zeta(2)/2$.
- **$\log\sigma=\tfrac12\log RV$ per block.** Blocks need ≥ 90% of bars. Sub-daily blocks are computed on returns deseasonalized by a 288-slot clock.
- **Bias, and why calibration is needed.** Measured on synthetic rough-volatility paths (fOU $\log\sigma$ at 5m steps, about 770 days):

  | True $H$ | 1d raw | 1h raw | 1h noise-corrected |
  |---|---|---|---|
  | 0.05 | 0.16 | 0.02 | 0.09 |
  | 0.10 | 0.20 | 0.03 | 0.09 |
  | 0.30 | 0.34 | 0.08 | 0.30 |
  | 0.50 | 0.50 | 0.13 | 0.46 |

  Two effects pull in opposite directions:
  - **Daily RV integrates** volatility over the day, which biases $H$ **up** when $H$ is small.
  - **1h RV is noisy** (12 returns), which biases $H$ **down**.

  So every estimate is mapped back through a calibration: rough-volatility paths with known $H$ (0.03–0.5), the asset's own vol-of-vol and the same sample length, with 3 replicates. On synthetic data with true $H$ = 0.10 / 0.30, the calibrated noise-corrected estimates were:

  | True $H$ | 1h | 4h | 1d |
  |---|---|---|---|
  | 0.10 | 0.06 | 0.15 | 0.12 |
  | 0.30 | 0.32 | 0.33 | 0.32 |

  The thesis COF estimator on daily log RV is strongly biased (0.44 for true 0.10) and is reported for comparison only.
- **Noise-corrected fit:** $m_2(\Delta)=A\Delta^{2H}+2\eta^2$, by grid search over $H$ with non-negative least squares.
- **Mean reversion:**
  - ACF of $\log\sigma$ up to 168 h / 60 days, with power-law vs exponential fits on log $\rho$;
  - fOU via the thesis §5.5 identities using the calibrated $H$, plus the raw-COF version.

  **Caveat:** $\hat\kappa=[\hat\sigma^2\Gamma(2H+1)/(2\widehat{\operatorname{Var}})]^{1/(2H)}$ has exponent $1/(2H)$, i.e. 5 at $H=0.1$. Half-lives are therefore order-of-magnitude only (synthetic truth 69 days; estimates 56 days at 4h, unstable at 1d). The ACF is the model-free read.

## 3. Expected reading (stated before the run)

- **Price:** $H<0.5$ at 5–30 min (bid-ask bounce on spot closes), and about 0.5 from 1h up, consistent with HR 002.
- **Volatility:** rough, with calibrated $H$ around 0.05–0.2 and power-law ACF, if crypto behaves like the literature reports for equities.

If the run says otherwise, that is the finding.

## 4. Decisions needed from Ivan

1. Run 02 (commands in `notebooks/Hurst_Regimes/README.md`; about 1–2 minutes).
