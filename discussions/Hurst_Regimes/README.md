# Hurst Regimes discussions

Does the Hurst exponent of 5-minute price paths vary between sessions, persist, and predict the next session's terminal move per unit of path variance? These are the prerequisites for an HMM on $H$ (Ivan, 2026-10-11) and for any use of $H$ as a conditioner of the VRP 02 long straddle. Notebooks: `../../notebooks/Hurst_Regimes/`.

Independent branch: it uses only the shared 5-minute history files, no other branch's outputs. The estimator comes from Ivan's thesis (§5.1).

## History

| Turn | Content |
|---|---|
| `001_claude_branch_preregistration.md` | Branch rationale, three gates (variation, persistence, relevance), volatility-modulated fGn null, development/confirmation split |
| `002_claude_01_results.md` | 01 results: STOP. H ≈ 0.5 within noise (except TRX, structural); lag-1-only echo; no prediction of next-session terminal move |

Next turn in this branch: `003`.

Procedure and numbering rules: `../README.md`.
