# VD20 readout: the engine forecasts the next 24h well, but not beyond implied

Run outputs: `notebooks/Volatility_Dynamics/20/`. Verdict **FORECASTS_VOL_BUT_NOT_BEYOND_IMPLIED**
(G1 PASS, G2 ENGINE_BETTER, G3 FAIL). EVAL: 325 days × 5 assets after the engine's TRAIN end, 2025-11-22.

## Forecast skill (08:00 origin, next 24h)

| model | QLIKE [95%] | MZ b | MZ R² | mean RV / mean F |
|---|---|---|---|---|
| NAIVE | 0.355 [0.282, 0.443] | 0.64 | 0.41 | 1.00 |
| ENGINE | **0.200** [0.172, 0.232] | 0.91 | 0.52 | 0.98 |
| HAR_D | 0.296 [0.263, 0.334] | 0.75 | 0.41 | 0.63 |
| HAR_DX | 0.260 [0.234, 0.289] | 0.75 | 0.50 | 0.64 |

- ENGINE − NAIVE −0.155 [−0.223, −0.095]; ENGINE − HAR_D −0.096 [−0.120, −0.071]. My preregistered expectation
  (HAR_D ≥ ENGINE) was wrong. Part of the gap is level: the daily HARs over-forecast by ~1/0.63 ≈ 1.6× in EVAL.
  Their TRAIN-period smearing did not transfer, whereas the engine is fitted on every 5m origin. In R² terms, HAR_DX
  is close to the engine (0.50 vs 0.52).
- The post-2026-09-19 period agrees (ENGINE QLIKE 0.164).
- Volatile day (top 20% of RV): ENGINE AUC 0.84 [0.79, 0.88]; top decile 71% hit (3.5× base), bottom decile 1%.
- Tails: z = R/√F_ENGINE has s.d. 1.00. P(|z| > 2) observed 5.8% vs Gaussian 4.6% vs TRAIN Student-t (df 3.4) 6.9%;
  P(|z| > 2.5) 2.5% / 1.2% / 3.9%. The 24h tails sit between Gaussian and the TRAIN t.

## Versus implied (VD19 12:00 rows, 1200 min, 86 rows, 22 sessions)

- Encompassing b_mod: ENGINE 0.12 [−0.20, 0.57]; HAR_DX 0.31 [−0.03, 0.84]; HAR_D −0.23; NAIVE −0.10.
  b_imp stays 0.7–1.1 with lower bounds > 0. No model passes. HAR_DX is closest, but it is one of four models in
  22 sessions, so this is a post-hoc lead only.
- Economics, ATM straddle at the ask, 12:00: all −23% [−39%, −5%]; ENGINE ≥ 1.25 implied −23% [−47%, +1%]
  (15 sessions). HAR_D ≥ 1.25 −31%, HAR_DX −28%. Days the models call hot are not cheaper than average.

## Reading

The engine is a good, well-calibrated 24h volatility forecaster. The option market already prices the same
information, so a "volatile tomorrow" signal does not make options cheap. OTM wings carry a larger ask markup
(VD 10c/10d), which makes long OTM strangles worse than the ATM straddles tested here. Untested: 08:00 entry with 24h
to expiry (no stored option rows), and prospective G3 with more sessions.
