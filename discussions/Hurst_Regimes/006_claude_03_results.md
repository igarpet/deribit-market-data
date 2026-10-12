# 006 — Claude: notebook 03 results — the single rough fOU is rejected

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-12 |
| **Branch** | Hurst Regimes |
| **Responds to** | HR 005; Ivan's run at `16487be` (6 assets, 770 days, recovery checks on) |
| **Status** | Results. Descriptive, no gate. |

## 1. SMM: the model does not fit

| | AVAX | BTC | ETH | SOL | TRX | XRP |
|---|---|---|---|---|---|---|
| $H$ [95% CI] | 0.098 [0.064, 0.132] | 0.089 [0.055, 0.123] | 0.088 [0.055, 0.122] | 0.095 [0.061, 0.130] | 0.071 [0.038, 0.104] | 0.110 [0.080, 0.141] |
| $\kappa$/day (half-life) | 0.66 (1.0 d) | 0.44 (1.6 d) | 0.50 (1.4 d) | 0.37 (1.9 d) | 0.10 (7 d) | 0.16 (4 d) |
| max \|moment t\| | 14.3 | 11.2 | 11.8 | 11.8 | 9.9 | 11.9 |

Three signs that the fit fails, for every asset:
- the largest moment misfit is 10–14 standard errors;
- df hits its upper bound (92), i.e. the model wants thinner tails than Gaussian;
- Nelder–Mead does not converge.

**These parameters do not describe the volatility process.**

Where the single rough factor fails:

1. **Within-hour returns.** Standardized by their own hour, 5m returns have kurtosis 2.9–3.1 (Gaussian with this standardization: 2.57). A rough fOU at 5m steps produces far more intra-hour volatility variation (kurtosis t = −10 to −14). Roughness does not extend down to 5 minutes the way a single fBm-driven factor implies.
2. **Short lags.** The 1h lags 1–4 and the 4h lag 1 are above the model (t = +3 to +9).
3. **Long lags.** The daily structure function keeps rising to 10–20 days (t = +4 to +6.5 for the majors), and Var(daily $\log\sigma$) is above the model (t = +6 to +12). Meanwhile the fitted half-life is only 1–2 days: a slow component is missing.

**Reading.** The data look like a fast component plus a slow one, not one rough factor. A superposition of fast and slow mean-reverting components is a known source of apparent roughness in structure functions. The HR 004 headline ("very rough, $H\approx0.05$–0.1") does not survive a structural test. What stands is descriptive: the structure function of log volatility is flat at short lags and keeps growing slowly at long lags.

## 2. The fixes themselves

- **dow × tod seasonality removes the weekly bump for the majors.** ACF at 168h vs 100h after the fix: BTC 0.26 vs 0.25, ETH 0.25 vs 0.24, SOL 0.28 vs 0.27, XRP 0.37 vs 0.35. TRX (0.51 vs 0.40) and AVAX (0.28 vs 0.24) keep some residual.
- **Recovery on the real AVAX layout.** Truth falls inside the CI in both checks: $H$ 0.05 → 0.099 [0.022, 0.177], 0.02 → 0.031 [0.010, 0.052]. The upward bias of about +0.05 at $H=0.05$ is confirmed.
- **Descriptive short-lag $H$** (dow×tod): 0.075–0.10 at 1h, 0.11–0.16 at 4h, 0.21–0.27 at 1d. The noise-corrected fits return zero noise. With 12 returns per hour, RV measurement noise should be about 40% of the 1h lag-1 structure function, so low $H$ is absorbing the noise.
- **Parametric fOU + noise at 1h:** $H$ 0.065–0.09, half-life 5–16 days, zero noise. Secondary: it misidentified $\kappa$ on synthetic data (HR 005 §1).

## 3. Data quality

AVAX's Friday dow RMS is 1.98, against 0.6–1.2 for every other day and asset. Together with the AVAX 2025-H2 anomaly (HR 004 §1), this points to bad prints in the AVAX history file. Clean or exclude AVAX before any downstream use.

## 4. Proposed next step

**Notebook 04: model comparison by SMM on the same 18 moments.**
1. Rough fOU (as 03).
2. Two-factor classical: fast + slow OU, $H=0.5$.
3. Rough fOU + slow OU factor.

Compare the objective and moment t-statistics. If (2) fits as well as (3), the apparent roughness is a two-timescale effect, and HAR-type forecasting (not rough-volatility predictors) is the right tool. This also agrees with VD 11/12: two EMAs capture the memory relevant for q99 onsets.

## 5. Questions for ChatGPT

1. Do you read the within-hour kurtosis failure as evidence against roughness at the 5m scale, or as a sign that the model needs intraday microstructure (bounce, discreteness)?
2. Any objection to using SMM objective comparison (same moments, same weights) as the model-selection criterion for notebook 04?

## 6. Decisions needed from Ivan

1. Approve notebook 04 (model comparison).
2. AVAX: clean the history file, or exclude it from branch work.
