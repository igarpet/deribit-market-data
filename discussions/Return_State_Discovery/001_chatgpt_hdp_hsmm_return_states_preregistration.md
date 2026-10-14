# RSD 001 — Bayesian nonparametric return-state discovery over AM-T-HAR-Blend

**2026-10-14, research proposal and source implementation before first real-market RSD run.** Exploratory, not an independently pristine test and not authorization to trade. Cross-branch reuse of the Volatility Dynamics 18 price-only baseline is explicitly requested by the user; its historical fitted results are not imported into model fitting.

## Question

Do non-overlapping 30-minute **price-return path behaviors**, observed relative to the *ex ante* variance of AM-T-HAR-Blend, contain a persistent latent-state structure that provides a better **causal density forecast of the next 30-minute signed return** than both (a) unchanged AM-T-HAR-Blend and (b) an adaptive-free Student-t calibration trained on exactly the state-discovery training period? Do the number and characteristics of states look unusual compared with a matched, regime-free, heavy-tailed conditional-return simulation?

This is explicitly *not* an attempt to discover high/low volatility, impose two states, or call a state 'momentum', 'mean reversion' or 'shock' in advance.

## Data provenance and chronology

- Use the shared Binance spot price_5m history via VD14 → VD18 code on **BTC and ETH first**. Historical sample before 2026-09-19 UTC; original approximately 60/20/20 training/validation/test days. No Deribit collector/database/tick changes, no options strategy.
- Refit a **30-minute** VD18 AM-T-HAR-Blend baseline using original TRAIN + VALID only; the HAR family is chosen on later VALID and the blend fit on earlier VALID. Student-t native df selection uses VALID as in VD15; this is before the historical TEST.
- Make 30-minute non-overlapping blocks at exactly 00 and 30 minutes UTC from six *subsequent* contiguous five-minute returns. AM variance forecast at the *start* of the block; the block's realized features become known only at its *end*. Check sum of six returns equals the corresponding VD14 R_30 and do not bridge missing bars.
- Within the already-inspected ~150-day original TEST, split by chronological complete UTC dates: first 60% (~90 days) for HSMM fit, next 20% (~30 days) for diagnosis and final 20% (~30 days) for descriptive forward-filter evaluation. The original TEST is **not untouched for project-level selection**; these are not prospectively confirmatory splits.
- All feature normalizers and null Student-t parameters are fit only on the HSMM training segment. All state filters use only previously completed blocks; Viterbi/full smoothing are descriptive only, never forecast inputs.

## Price-state observations

For each six-return block r=(r1,...,r6) and VD18 ex-ante net-return variance v>0:

1. `return_surprise = sum(r) / sqrt(v)` (signed, not absolute).
2. `signed_path_efficiency = atanh(clip(sum(r)/sum(abs(r)),[-.999,.999])) / 2` (directional vs choppy movement within the block).
3. `within_block_dependence = atanh(clip(sum(r_j*r_{j+1}) / sum(r_j^2),[-.95,.95]))` (short-lag same-direction vs alternating moves).

All are computed only when the entire 30-minute block has completed. Train-only robust centering/scaling; no future-return features at the prediction origin. These features are *statistical summaries* of realized paths; they are not independent and do not by themselves establish momentum alpha. Volatility level is omitted from the primary emission to prevent trivially stratifying by realized variance. Historical volatility enters only through ex-ante variance normalization.

## Model: finite weak-limit hierarchical Dirichlet process HSMM

- **Approach B, not a K-grid:** global `beta ~ Dirichlet(gamma / Kcap)`, row-specific transitions `pi_i,not-i ~ Dirichlet(alpha * beta_not-i / (1 - beta_i))`, zero self transitions at **episode** boundaries. `Kcap = 12` is a numerical truncation, not a chosen market-state count. Posterior occupied K is reported as a histogram and 5–95% interval. If mass at Kcap or Kcap−1 exceeds 10%, do not interpret the state count; rerun at Kcap=18 or 24 after assessing computation.
- **Persistence:** explicit shifted negative-binomial episode durations (two successes), `p_k ~ Beta(2,8)`, `dmax=96` half-hour blocks (48h) as a **maximum age-memory representation**. Beyond 48h the hazard remains at its last estimated value; no state switch is forced at the cap. HSMM provides duration preferences; no diagonal sticky self-transition because it would double-count/obscure explicit episode durations. This is the principled semi-Markov analogue of a 'sticky HDP' intuition, **not literally the Fox et al sticky HDP-HMM**, and not an infinite-state exact sampler.
- Emissions: state-dependent three-dimensional Student-t with **common df=6** and state-specific mean and full covariance. Gamma latent scale plus Normal-Inverse-Wishart Gibbs updates. The heavy-tailed emission is vital to avoid splitting tails into fake Gaussian states.
- Posterior inference: blocked explicit-duration forward-filter/backward-sampling in log space, Gibbs updates for emissions and transition rows, Metropolis updates for global beta with simplex Jacobian; right-censored duration survival incorporated by Metropolis updates. At least two independently seeded chains on real BTC/ETH, initial 140 iterations with 80 burn/4 thinning. Compare posterior occupied K, transition and duration diagnostics across chains. These are short exploratory chains, not convergence proof.
- The Final episode per contiguous segment is right-censored through its negative-binomial survival likelihood. Initial episodes are assumed to start at each segment boundary, not fully left-censored; this is a finite-window approximation. Report age-cap fractions and refrain from structural duration claims until sensitivity tests and left-censor handling. **A single state can persist past the age cap** (no forced state alternation).

## How to avoid false regime discovery

1. **Regime-free heavy-tail negative control:** use the actual series of ex-ante predicted variances and simulate independent standardized Student-t(6) five-minute innovations, reconstructing the same three path features; the null is conditional on the empirical variance path, not a self-consistent refit of HAR, so regard it as a *first* negative control, not a complete generative null.
2. Fit the same HDP-HSMM under at least two null seeds, report the same posterior K and sequential return-density gain over a state-free Student-t forecast. *Many occupied states under the null are expected to invalidate any naive 'K>1 means regimes' claim.*
3. Compare causal **next-block net-return** NLL on exactly common episodes: frozen AM-T-HAR-Blend mixture, an updated single Student-t residual forecast fit on the discovery training sample, and HDP-HSMM's posterior predictive return marginal. The HSMM may use the completed block's full path features **only after** its return is scored, not for that block's forecast.
4. Also report joint engineered-observation log score (not numerically comparable with return NLL), per-session paired NLL differences, state occupancy, duration caps, chain disagreement, and negative-control gain. Condition a candidate conclusion on genuine heldout gain beyond the Student-t calibration and beyond simulated null behavior. A simple Bayesian mixture might capture nonnormal path-shape features without meaningful temporal regimes.
5. Do not name states economically, proclaim 'number of regimes', or build options strategies based only on posterior K. Test gamma/duration/truncation sensitivity and eventually run a prospective frozen split.

## Artifact contract

New **Return_State_Discovery** branch (independent code files; explicitly references reused VD18 implementation):

- `src/data_engineering/modeling/hdp_hsmm_return_states.py`
- `tests/test_hdp_hsmm_return_states.py`
- `notebooks/Return_State_Discovery/01_hdp_hsmm_return_states.ipynb`
- `discussions/Return_State_Discovery/001_chatgpt_hdp_hsmm_return_states_preregistration.md`
- `notebooks/Return_State_Discovery/README.md`

Historical run outputs to `notebooks/Return_State_Discovery/01/`: compact CSV/JSON only; no full origin-level posterior arrays or raw price data. Synthetic tests don't need historical files. Historical results require user's local DERIBIT_DATA_ROOT and have **not** been run during implementation.
