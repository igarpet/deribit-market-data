# 018 — Claude: Module 08 results — jumps arrive as a homogeneous Poisson process

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-10 |
| **Branch** | Volatility Dynamics |
| **Responds to** | VD 017; Ivan's run at `39f4368` (data to 2026-10-10 07:55; 20 scored days) |
| **Status** | results against the rules pre-registered in VD 017 §4 |

## 1. Verdicts (primary c = 4)

| Rule | Result | Evidence |
|---|---|---|
| Self-excitation (c) vs (b), blocked CV | **FAIL.** Δ logLik/row = −0.000162, 95% day-bootstrap [−0.000316, −0.000015]. The Hawkes model is significantly worse out of fold | `08/08_primary_cv_boot.csv` |
| Inhomogeneity (b) vs (a), blocked CV | **FAIL.** Δ = −0.000190, [−0.00066, +0.00028] | same |
| Secondary holdout (6 days, 62 jumps) | Both FAIL | `08_primary_holdout_boot.csv` |
| Time-rescaling | Homogeneous Poisson: KS D = 0.051, p = 0.55 (238 intervals); (b) p = 0.39; (c) p = 0.43 | `08_primary_gof.csv` |

## 2. Estimates

- **Jumps.** 245 over 20 days and 7 assets. Rates per asset-day: AVAX 1.62, BTC 2.12, ETH 1.85, HYPE 0.84, SOL 1.51, **TRX 4.19**, XRP 1.57 (`08_jumps_per_asset.csv`).
- **Daily count dispersion.** Variance/mean is 0.49–1.56, near the Poisson value of 1.
- **Branching ratio.** $\rho=0.047$ (full sample) and 0.00–0.07 across CV folds. On synthetic data, the no-excitation null gave $\rho\le0.08$, and true $\rho=0.37$ was recovered at 0.23–0.43. With $\rho$ estimated this small and an out-of-fold effect that is significantly negative, a large branching ratio is unlikely, not merely undetected.
- **Half-life.** It sits at the 5-min grid edge, which is consistent with one event spilling over two bars rather than triggering.
- **Volatility level.** In-sample, $\gamma=-0.59$ (SE 0.14): standardized jumps are somewhat rarer when local volatility is high. This does not improve out-of-fold likelihood.
- **c sensitivity.** All of $c\in\{3,\dots,5\}$ FAIL self-excitation, with $\rho$ 0.036–0.081 (`08_c_sensitivity.csv`).
- **Co-jumps.** Of 170 bars containing a jump, 34 (20%) have 2–6 assets jumping in the same bar (`08_cojump_counts.csv`). These are simultaneous market-wide jumps; they are not excitation.

## 3. Secondary: raw q99 onsets

- **Inhomogeneity: strong PASS.** Δ = +0.0067, CI [0.0046, 0.0090]; $\gamma_{\text{localvol}}=+2.36$.
- **Self-excitation:** blocked CV FAIL, holdout PASS. $\rho=0.11$, all from cross-asset terms.

This is consistent with the whole branch: raw-variance extremes cluster because volatility clusters.

## 4. Reading

Once returns are measured relative to local volatility, the remaining jumps behave like a **homogeneous Poisson process** per asset, at roughly 1–2 per asset per day at 4σ, plus a market-wide co-jump component. This is the first positive, usable structure in the branch. It does not mean jumps are predictable. It means their arrival rate is stable and their timing is close to random.

## 5. Caveats

1. **TRX (4.2/day, about 2× the others)** may be index tick discreteness: a one-tick move against a tiny local volatility. Jump sizes should be checked in bps before TRX's rate is used.
2. **Twenty days.** The rates should be re-estimated as data accumulate. Module 08 reruns unchanged.
3. **Explanatory, not causal.** Local volatility is two-sided (VD 017 §3). The rate estimates are valid; the jump classification itself cannot be done in real time.

## 6. Proposed next steps

1. **Module 09: compound-Poisson jump model.**
   - Jump-size distribution (in bps and in local-σ units, by sign).
   - A common-jump plus idiosyncratic decomposition of arrivals.
   - The implied probability of at least one jump before expiry, $1-e^{-\lambda\tau}$, with uncertainty.
   - A TRX tick-discreteness check.
2. **Cross-branch (needs Ivan's approval).** Compare the measured jump intensity and size distribution with what 0DTE wing IV prices (VRP branch).

## 7. Questions for ChatGPT

1. Do you read anything in Module 08 as evidence of self-excitation that §1–2 does not account for?
2. For the co-jump component, would you model it as a separate market-wide Poisson process or as cross-sectional dependence in marks?

## 8. Decisions needed from Ivan

1. Accept the Module 08 verdicts.
2. Approve Module 09 (compound-Poisson jumps) within this branch.
3. Decide whether to open the cross-branch comparison with option-implied jump pricing.
