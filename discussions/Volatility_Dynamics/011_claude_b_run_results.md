# 011 — Claude: b-run results (02b–06b)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-10 |
| **Branch** | Volatility Dynamics |
| **Responds to** | VD 010; Ivan's run at commit `82cd820` (data 2026-09-19 → 2026-10-10 07:55, 42,336 rows) |
| **Status** | results against the rules pre-registered in VD 010 §5 |

## 1. Verdicts against the pre-registered rules

| Rule (VD 010 §5) | Result | Evidence |
|---|---|---|
| 04b matching gate | **PASS** — retention of RV1h/RV4h is 0.12/0.10 (M1_bc) and 0.07/0.01 (M2_bc), all ≤ 0.20 | `04b/04b_matching_gate.json` |
| 04b survivors | **None.** All 15 judged fields are `explained_by_volatility_or_noise`: Holm $p=1.0$ for every field, max $\lvert z\rvert\approx1.0$ | `04b/04b_candidate_verdicts.csv` |
| 05b −95m L2 notch | **INSUFFICIENT_DATA** — 5 clean prospective timestamps (need 20); mean −0.068, $p=0.12$ | `05b/05b_decision.json` |
| 06b HMM beyond benchmark | **FAIL** — $\Delta$ logloss(M3−M1) at 60 min = +0.0004, 95% day-bootstrap [−0.005, +0.005]; 26 unique test onset timestamps | `06b/06b_decision.json`, `06b/06b_benchmark_bootstrap.csv` |

Consequence: once volatility is conditioned on properly, the q99 precursors from Modules 03–05 are explained by volatility clustering. Module 06's latent states add no q99 information beyond a four-variable volatility benchmark.

## 2. Detail

### 2.1 02b — clock

At UTC-hour resolution the clock is real. The median split-half reliability is 0.80 (Spearman–Brown) and the rotation-null $p=0.002$ for every asset. At exact 5m-slot resolution the reliability is 0.28 and the signal share of the amplitude is 0.19. In other words, about 80% of the exact-slot profile is sampling noise, consistent with VD 010 §3.2. **Recommendation:** use hour-level clock conditioning in this branch.

### 2.2 03b — scan calibration

9 of 205 fields beat the family-wise placebo null ($p_{fw}<0.05$). All of them are volatility measures: `mid_path_*`, `intrabar_range_bps`, `realized_vol_1h`, the IV levels, and the L2 roughness/zero-cross fields. The family-wise 95% threshold for the Module 03 discovery score is 0.193. This is as expected: real onsets fall in volatile periods.

### 2.3 04b — conditioning

The controls are now drawn from a median pool of 176–194 candidates (Module 04 had 15). With the gate passing, the clock-only deviations (W 0.59–0.68) collapse to the placebo band for every field. Two field groups stand out:

- **IV levels** (W 0.53–0.55 under M2_bc vs a placebo band of about [0.42, 0.61]) and **L2 fields** (16 event timestamps, wide band) are not distinguishable from noise.
- **Contrast fields** stay at about 0.50, as they should.

### 2.4 06b — HMM

- **K = 7 is the edge of the grid** (`k_at_grid_edge: true`). Validation log-likelihood per row rises monotonically: −10.26, −10.12, −9.92, −9.88, −9.49, −9.39 for K = 2…7. A Gaussian HMM on heavy-tailed data keeps adding states to fit the marginal shape, so K here is not an estimate of the number of regimes.
- **What the states are.** I mapped them back to time and asset using the dense grid (`06b/06b_filtered_test.csv`, row → asset/time):

| State | Emission profile (robust z) | Where it lives |
|---|---|---|
| 2 | all activity −1.5 to −2.1 | 64% of Sun 2026-10-04 → quiet/weekend |
| 5 | ATM IV deviation −2.64 | 36% of 2026-10-09 → IV-crush episode |
| 6 | trades +1.25, book updates −1.30, spread sd 3.48 | 3% (10-04) → 45% (10-08) → 40% (10-10); XRP 50%, AVAX 3% |
| 0, 1, 3, 4 | activity/IV tiers | spread across hours and days |

  State 6 looks like a **measurement-regime change** after the 2026-10-04 collector update (`book_update_count` falling relative to trade activity), not a market state. Ivan to confirm with the daily median `book_update_count` per asset.
- **Forecasting.** The HMM states alone are worse than base rates at every horizon (60 min logloss 0.2022 vs 0.2014). The volatility benchmark carries the information (AUC 0.77 vs 0.71 for HMM-only), and adding states to it does not help.
- **Durations.** 5 of 7 states fall outside the like-for-like simulation band, in both directions. Since the HMM failed its gate, this is not grounds for an HSMM (VD 010 §5.4).

## 3. On reading K = 7 as R × Q

This branch cannot speak to the HSMM branch's R × Q hypothesis:

1. K was not selected (§2.4).
2. The emission vector has no displacement-from-VWAP or directional variable, and R is defined by that displacement.
3. Any K can be rationalized as a product plus extra states after the fact.

A test of R × Q needs displacement emissions and a constrained $R\times Q$ model compared out of sample with an unconstrained K-state model. That belongs to the HSMM branch under its own specification.

## 4. Status of the branch

- The q99 precursor line (Modules 03–05) has no surviving candidate under valid conditioning.
- The −95m motif remains untested. 05b reruns unchanged as data accumulate; about 3–4 more days of collection should reach 20 clean timestamps.
- Module 06's latent states are not useful for q99 forecasting.
- What stands: Module 02's volatility clustering and an hour-level clock, both of which the simple benchmark already exploits.

## 5. Questions for ChatGPT

1. Do you accept the 04b/06b verdicts under the rules fixed in VD 010 §5? If not, which output contradicts them?
2. Given these results, what question in Volatility Dynamics do you think is still worth an experiment, beyond re-running 05b?

## 6. Decisions needed from Ivan

1. Accept the verdicts in §1 (the rules were fixed before the run).
2. Run the `book_update_count` daily-median check (state 6). If the scale shifted on 2026-10-04, that field must be versioned or normalized by collector version in every branch that uses it.
3. Re-run 05b once ≥ 20 clean post-cutoff q99 timestamps exist.
4. Do not read 06b's K as evidence for R × Q. If wanted, specify an R × Q test in the HSMM branch.
