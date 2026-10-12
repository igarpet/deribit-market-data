# 005 — Claude: notebook 03 — HR 004 review fixes (seasonality, mean reversion, SMM)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-12 |
| **Branch** | Hurst Regimes |
| **Responds to** | Review of HR 004 relayed by Ivan, 2026-10-12 (fixes 1–3 before using $H$ downstream) |
| **Status** | Built (`rough_vol_smm.py` v0.1.0), tests 7/7, verified on synthetic data. Not yet run on real data. Descriptive: no gate. |

## 1. Position on the three fixes

| Fix | Position | Implemented as |
|---|---|---|
| 1. dow × tod deseasonalization; drop/flag 24h and 168h lags | Agree | 7 × 288 RMS profile (smoothed over 7 slots); every fit excludes multiples of 24h (1h, 4h series) and 7d (daily). Before/after ACF at 24h and 168h is reported. |
| 2. Short pre-declared lag ranges, or fOU + noise structure function | Agree on both; the parametric form is **secondary** | Short ranges: 1h lags 1–12, 4h 1–5, 1d 1–5. Parametric: $m_2(k)=2V[1-\rho_{fOU}(k;H,\kappa)]+2\eta^2$ on hourly $\log\sigma$, with $\rho$ exact for AR(1)-filtered fGn. |
| 3. Calibration with matched $\kappa$, fat tails, seasonality, grid below 0.03, ≥ 10 reps; or a dedicated estimator | Agree; went to a joint estimator | Simulated method of moments (§2) |

**Why the parametric fit (fix 2) is secondary.** The formula is for point values of $\log\sigma$. Hourly log RV integrates variance over the hour, and that integration is what broke the daily calibration in HR 004. On synthetic data with known parameters, the parametric fit gets $H$ roughly right but $\kappa$ wrong:

| Synthetic truth | Parametric fit |
|---|---|
| $H$ = 0.05, $\kappa$ = 0.05/day | $H$ = 0.035, $\kappa$ at the grid edge (0.0024/day) |
| (second synthetic draw) | $H$ = 0.10, $\kappa$ = 0.076/day |

It is reported for comparison only.

**Why SMM and not Bolko et al. / Fukasawa et al.** Their GMM and Whittle estimators need closed-form moments of integrated variance under the model, which I cannot verify here. SMM addresses the same three issues (integration, measurement noise, mean reversion) by simulating them, and can be checked by recovering known parameters. If the SMM results matter downstream, porting the published GMM would be a useful cross-check.

## 2. SMM design

| Item | Value |
|---|---|
| Model | $\log\sigma_t$ = fOU$(H,\kappa,\nu)$ at 5m steps (exact fGn by circulant embedding, AR(1) filter); $r_t=s_{dow,tod}\,\sigma_t\,\varepsilon_t$, $\varepsilon\sim t(\text{df})$ with unit variance |
| Simulation layout | The data's own timestamps, gaps and seasonal profile; 140-day burn-in |
| Pipeline | Identical for data and simulations: seasonal profile re-estimated, 1h/4h/1d log RV, logs |
| Moments (18) | log $m_2$ at 1h lags 1,2,3,4,6,12; 4h lags 1,2,3,5; 1d lags 1,2,3,5,10,20; log Var of daily $\log\sigma$; log kurtosis of 5m returns standardized by their own hour |
| Weights | Diagonal, from the moment sampling variance at a pilot point |
| Optimization | Grid $H\in\{0.005,\ldots,0.45\}$ × $\kappa\in\{0.003,\ldots,1\}$/day × 3 $\nu$ values (3 reps); Nelder–Mead from the two best grid cells (10 reps, common random numbers), then polish |
| Inference | Sandwich SEs with $\Omega$ from 30 independent replicates; $\kappa$ CI by delta method on $\log\kappa$; `kappa_identified` = CI ratio < 100 |

## 3. Recovery on synthetic data

| Sample | Truth $(H,\kappa,\text{df})$ | Estimate $H$ [95% CI] | Estimate $\kappa$/day [95% CI] | df |
|---|---|---|---|---|
| 770 d, v1 optimizer | 0.05, 0.05, 5 | 0.052 [0.011, 0.094] | 0.047 [0.017, 0.130] | 4.9 |
| 770 d, v1 | 0.02, 0.20, 4 | 0.023 [0.008, 0.039] | 0.156 [0.064, 0.381] | 3.7 |
| 770 d, v1 | 0.12, 0.01, 8 | 0.077 [0.045, 0.109] ✗ | 0.012 [0.001, 0.13] | 11.3 |
| 770 d, final | 0.05, 0.05, 5 | 0.102 [0.024, 0.179] | 0.101 [0.050, 0.203] ✗ (edge) | 4.4 |
| 770 d, final | 0.12, 0.01, 8 | 0.142 [0.098, 0.187] | 0.013 [0.004, 0.048] | 7.7 |
| 400 d, final | 0.05, 0.05, 5 | 0.174 [0.054, 0.295] ✗ (edge) | 0.095 [0.032, 0.284] | 4.1 |

Reading:

- At full length, truth is inside the $H$ interval in 4 of 5 runs. The one miss was a single-start optimizer stuck on the $(H,\kappa)$ ridge, which the final two-start version fixed.
- **Precision is about ±0.04 SE for $H$ on 770 days.** The estimator can separate $H\approx0.05$ from $H\approx0.3$, but not 0.05 from 0.1.
- There is an upward tendency in shorter samples (400 days: 0.17 for truth 0.05, with the CI just missing).
- Notebook 03 repeats the check on the real data layout before the real fits.

The unit test checks the optimizer, not the statistics: on a short sample, the fitted point must match the moments at least as well as the true parameters.

## 4. Runtime

About 5–7 minutes per asset on the build machine, and about the same per recovery check. A full run (6 assets + 2 checks) takes roughly 1–1.5 h on the PC. `HR03_RECOVERY=0` skips the checks; `HR_ASSETS=BTC,ETH` limits assets.

## 5. Questions for ChatGPT

1. Is a 140-day burn-in adequate? For $\kappa$ near 0.003/day the fOU is not stationary within it. Data and simulation share the convention, so this changes what $\kappa$ means at the slow end, not the fit.
2. Would you add jumps (compound Poisson) to the return model on top of the $t$ innovations, or is the kurtosis moment enough for the purpose here (roughness of $\sigma$, not returns)?

## 6. Decisions needed from Ivan

1. Run 03 (commands in `notebooks/Hurst_Regimes/README.md`).
