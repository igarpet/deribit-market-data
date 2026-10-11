# 004 — Claude: notebook 02 results — price is Brownian, volatility is very rough

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-11 |
| **Branch** | Hurst Regimes |
| **Responds to** | HR 003; Ivan's run at `7aaef84` (Binance spot 5m, 2024-09-01 → 2026-10-11, 770 days; AVAX BTC ETH SOL TRX XRP; HYPE excluded, 16 days) |
| **Status** | Descriptive results. No gate. |

## 1. Price: Brownian at every horizon tested

$H=\zeta(2)/2$ of the log price:

| | 5–30 min | 1–8 h | 1–7 days |
|---|---|---|---|
| AVAX | 0.35 | 0.48 | 0.47 |
| BTC | 0.49 | 0.49 | 0.50 |
| ETH | 0.50 | 0.51 | 0.50 |
| SOL | 0.49 | 0.49 | 0.47 |
| TRX | 0.53 | 0.54 | 0.44 |
| XRP | 0.49 | 0.48 | 0.53 |

- The majors sit at 0.48–0.51 at every horizon, and half-year windows stay within 0.46–0.53. Spot closes show no visible bid-ask-bounce anti-persistence at 5 minutes.
- **AVAX at 5–30 min (0.35) comes from one window.** 2025-H2 reads 0.18; the other windows read 0.48–0.50. That points to a data problem in the AVAX file for that half-year (stale or repeated closes), not market behaviour. It should be checked before AVAX is used at 5m.
- TRX is mildly persistent at short horizons (0.53–0.55), consistent with its tick structure (HR 002 §2).

This confirms HR 002 from a different angle: price paths scale like Brownian motion between 5 minutes and a week.

## 2. Volatility: rough, monofractal, $H \approx 0.05$–$0.1$

$H$ of $\log\sigma$:

| | 1h raw | 1h calibrated | 4h raw | 4h calibrated | 1d raw |
|---|---|---|---|---|---|
| AVAX | 0.027 | 0.030 | 0.092 | 0.041 | 0.094 |
| BTC | 0.089 | 0.078 | 0.123 | 0.042 | 0.066 |
| ETH | 0.075 | 0.071 | 0.111 | 0.044 | 0.092 |
| SOL | 0.076 | 0.078 | 0.115 | 0.057 | 0.111 |
| TRX | 0.046 | 0.048 | 0.079 | 0.041 | 0.137 |
| XRP | 0.089 | 0.092 | 0.139 | 0.073 | 0.142 |

("Calibrated" is the noise-corrected estimate mapped through the simulation calibration.)

- **Very rough.** Every estimate is far below 0.5. The calibrated 1h and 4h values put $H$ around **0.03–0.09**, rougher than the 0.1–0.15 typically reported for equity indices.
- **Monofractal.** $H_q=\zeta(q)/q$ is flat across $q=0.5\ldots3$ for every asset and block. For example, BTC 1h runs from 0.090 to 0.086, and ETH 4h from 0.112 to 0.108. This is the rough-volatility signature (Gatheral, Jaisson & Rosenbaum 2018), not multifractal cascades.
- **Stable.** The 4h noise-corrected $H$ stays within 0.075–0.16 in every half-year window for every asset.
- **Daily calibration failed, and that is informative.** The observed 1d estimates (0.07–0.14) lie *below* what the simulated rough-volatility model produces even at $H=0.03$ (0.16). The model's within-day integration of volatility does not match the data. Real intraday volatility evidently varies less, relative to its day-to-day variation, than a single fOU with that vol-of-vol implies. Daily $H$ is therefore reported raw only. The BTC 1d noise-corrected fit is degenerate (0.99, noise share 1.0) and is discarded.

## 3. Mean reversion of $\log\sigma$

Autocorrelation:

| Lag | AVAX | BTC | ETH | SOL | TRX | XRP |
|---|---|---|---|---|---|---|
| 1h | 0.56 | 0.71 | 0.66 | 0.68 | 0.70 | 0.75 |
| 10h | 0.35 | 0.46 | 0.44 | 0.46 | 0.55 | 0.57 |
| 60h | 0.20 | 0.16 | 0.21 | 0.28 | 0.44 | 0.38 |
| 168h | 0.37 | 0.41 | 0.34 | 0.34 | 0.49 | 0.39 |
| 1d | 0.59 | 0.55 | 0.58 | 0.63 | 0.81 | 0.71 |
| 10d | 0.19 | 0.08 | 0.11 | 0.19 | 0.57 | 0.33 |
| 30d | 0.18 | 0.01 | 0.03 | 0.13 | 0.44 | 0.14 |

- **Hourly: power law, not exponential, for all six assets.** Exponents $\alpha$ = 0.10–0.32, and the RSS is 35–70% lower than the exponential fit. Shocks to volatility fade fast at first, then a long tail persists for days.
- **Daily: mixed.** Power law fits better for BTC, ETH and TRX; exponential for AVAX, SOL and XRP. For the majors, a volatility shock has largely decayed within 1–2 weeks (ACF about 0.1 at 10 days) and is gone by a month. TRX is the exception, with structural persistence.
- **Weekly cycle (post hoc).** The hourly ACF rises again at 168h (0.34–0.49, above its 60–100h values). That is the weekday/weekend volatility cycle, which notebook 02 did not remove (only time of day was deseasonalized). It inflates the long-lag ACF and probably biases daily $H$. A weekday-adjusted re-run would clean it.
- **fOU $\kappa$ (thesis estimator) is not identified here, as warned in HR 003 §2.** With calibrated $H\approx0.04$, the exponent $1/(2H)\approx12$ blows $\hat\kappa$ up (half-lives of $10^5$ blocks and more). With COF $H$ it gives daily half-lives of 1.5–5 days. Neither number should be used. The ACF is the reliable description.

## 4. What this means

1. **Answer to Ivan's question.**
   - Price: not rough, essentially Brownian from 5 minutes to a week.
   - Volatility: very rough ($H\approx0.05$–0.1), monofractal and stable over two years. Its mean reversion is power-law (long memory), not a single half-life.
2. **The fOU with a constant $\kappa$ is the wrong summary for crypto volatility.** At this roughness, the "speed of mean reversion" is horizon-dependent. That is why the thesis estimator breaks here.
3. **Consistent with earlier project results.** HAR forecasting worked (HSMM 06), and single-exponential GARCH memory was very short (HSMM 016). HAR approximates power-law memory; GARCH does not.

## 5. Possible next steps (Ivan to choose)

1. **Weekday-adjusted re-run** of 02 (clock by day of week × time of day). Small code change; it cleans the 168h bump and the daily $H$.
2. **Rough-volatility forecasting test.** The RFSV predictor (Gatheral et al. 2018: $E[\log\sigma_{t+\Delta}]$ as a power-law-weighted average of past $\log\sigma$, with weights set by $H$) against HAR and a GARCH-type benchmark. Target: next 4h / 1d log RV, out of sample, on the two-year history. This turns "volatility is rough" into a measurable forecasting claim.
3. **Check the AVAX 2025-H2 data** (5m closes) for stale or repeated prices.

## 6. Questions for ChatGPT

1. Do you accept the calibration-failure reading at 1d (model mismatch in intraday integration), or see a coding issue?
2. For the RFSV comparison, should HAR be fitted per asset or pooled?
