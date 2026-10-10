# 013 — Claude: 06c Student-t HMM results

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-10 |
| **Branch** | Volatility Dynamics |
| **Responds to** | VD 012; Ivan's run at commit `06fe3b2` (data to 2026-10-10 07:55; test from 2026-10-04) |
| **Status** | results against the rules pre-registered in VD 012 §4 |

## 1. Verdicts

| Rule (VD 012 §4) | Result | Evidence |
|---|---|---|
| 3. Student-t fits better | **Yes.** Validation LL/row is higher at every K; at the selected K = 10, −8.869 (t) vs −9.094 (Gaussian) | `06c/06c_k_selection_t_vs_gaussian.csv` |
| 2. Interior K | **No.** Selected K = 10 = grid edge. Validation LL/row rises monotonically for both emissions: t −10.03 → −8.87 over K = 2…10 | same |
| 1. Forecast beyond benchmark (primary) | **FAIL.** At $h=60$, $\Delta$logloss(M3−M1) = **+0.0082**, 95% day-bootstrap [+0.0002, +0.0142]. Adding the states significantly *worsens* the forecast. HMM-only logloss 0.2090 vs base rate 0.2014 | `06c/06c_decision.json`, `06c/06c_benchmark_bootstrap.csv` |

Note: K = 10, not 9. The state labels run 0–9.

## 2. Interpretation

- **The tails are real** (per-state $\nu$ between 2.1 and 12.5). **But they are not why K grows.** On synthetic data with known K the Student-t stops at the true K (VD 012 §3). Here it does not.
- **The states encode heterogeneity that a single pooled model cannot otherwise absorb** (`06c/06c_filtered_test.csv`):

  | State | Composition |
  |---|---|
  | 6 | 65% TRX, 25% XRP, $\nu=2.3$: per-asset microstructure |
  | 8 | 88% of rows on Sun 2026-10-04: weekend / date |
  | 9 | 62% on 2026-10-09, ATM IV deviation $z=-2.7$: the IV-crush day (06b state 5) |
  | 4 | 7.1× q99 lift but 136 rows, 6 event timestamps, 47% on 2026-10-08: a single episode |

- A diagonal emission also spends states on the strong correlation between range, volume and trades.
- With ~23k pooled training rows, validation likelihood rewards each added state. Likelihood-based K here measures density-fitting capacity, not a regime count.

## 3. Recommendation

The latent-state line in Volatility Dynamics has now failed the pre-registered forecast test twice (06b Gaussian, 06c Student-t) on these emissions. Opinion: a further HMM/HSMM variant on the same seven emissions would be specification search. I recommend closing the line here. Any new state model should be justified by new information, not by a new emission family:

- **Displacement from VWAP** (the R hypothesis) belongs to the HSMM branch under its own specification.
- **Asset-specific or covariance-aware models** would first need a reason to expect q99 information that the volatility benchmark lacks.

What stands in this branch: Module 02 clustering, an hour-level clock (02b), and a four-variable volatility benchmark with held-out AUC 0.77 at 60 min. The −95m motif (05b) remains pending on data.

## 4. Questions for ChatGPT

1. Do you see any outcome of 06b/06c that supports continuing the latent-state line in this branch?
2. If you want one more state model, state in advance what information it adds that the volatility benchmark does not have.

## 5. Decisions needed from Ivan

1. Accept the 06c verdicts (rules fixed in VD 012 §4).
2. Decide whether to close the Volatility Dynamics latent-state line (my recommendation) and keep only the 05b reruns open.
