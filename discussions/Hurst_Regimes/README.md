# Hurst Regimes discussions

Does the Hurst exponent of 5-minute price paths vary between sessions, persist, and predict the next session's terminal move per unit of path variance? These are the prerequisites for an HMM on $H$ (Ivan, 2026-10-11) and for any use of $H$ as a conditioner of the VRP 02 long straddle. Notebooks: `../../notebooks/Hurst_Regimes/`.

Independent branch: it uses only the shared 5-minute history files, no other branch's outputs. The estimator comes from Ivan's thesis (§5.1).

## History

| Turn | Content |
|---|---|
| `001_claude_branch_preregistration.md` | Branch rationale, three gates (variation, persistence, relevance), volatility-modulated fGn null, development/confirmation split |
| `002_claude_01_results.md` | 01 results: STOP. H ≈ 0.5 within noise (except TRX, structural); lag-1-only echo; no prediction of next-session terminal move |
| `003_claude_02_roughness_design.md` | Notebook 02 design: descriptive price and volatility roughness, estimator bias and simulation calibration |
| `004_claude_02_results.md` | 02 results: price Brownian 5m–7d; volatility very rough (H ≈ 0.05–0.1), monofractal, stable; power-law memory; weekly cycle; next-step options |
| `005_claude_03_review_fixes.md` | Notebook 03: review fixes 1–3; SMM over (H, κ, ν, df), recovery table, parametric fit secondary |
| `006_claude_03_results.md` | 03 results: single rough fOU rejected (max \|t\| 10–14, df at bound); fast + slow structure likely; seasonality fix works; AVAX data issue; proposes 04 model comparison |

Next turn in this branch: `007`.

Procedure and numbering rules: `../README.md`.
