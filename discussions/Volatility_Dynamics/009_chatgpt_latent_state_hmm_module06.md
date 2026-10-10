# 009 — ChatGPT: Module 06 latent market-state HMM

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-10 |
| **Branch** | Volatility Dynamics |
| **Responds to** | Ivan's instruction to build the first HMM after Modules 03–05 revealed coherent pre-q99 observable structure |
| **Status** | Implemented; awaiting real-data run |

## 1. Why an HMM is now justified

Modules 03–05 produced evidence that q99 volatility onsets are not preceded by one isolated feature alone.

Broad-history evidence showed elevated:

- native range;
- trading activity;
- option-IV level;
- spread/book activity.

The sparse L2 subset additionally suggested a local zero-crossing/instability shape around the -105 to -85 minute region.

The research question has therefore changed from:

> Does feature X move before q99?

to:

> Do these observables form recurrent latent market configurations whose current state or transition is associated with future q99 hazard?

This is the first point in this branch where a hidden-state model is introduced.

## 2. HMM first, not HSMM

The initial model is deliberately a standard first-order HMM.

No explicit duration distribution is fitted.

The HMM implies geometric state duration through:

[
P(S_{t+1}=k\mid S_t=k)=A_{kk}
]

and constant exit hazard:

[
h_k=1-A_{kk}.
]

Module 06 then measures empirical hard-filtered state durations and age-specific exit hazard on held-out observations.

Only if those diagnostics materially contradict the HMM duration structure should an HSMM be tested.

## 3. No economic state labels are imposed

The hidden state is:

[
S_t\in\{1,\ldots,K\}.
]

The model is not told that any state is:

- Mean Reversion;
- Momentum;
- VOL;
- Bridge;
- Clean;
- Noisy.

State meaning is assigned only after inspecting fitted emission profiles, transitions, persistence and q99 hazard.

This is intentionally narrower and more empirical than importing the architecture of the separate HSMM research branch.

## 4. Core emission vector

The main HMM uses:

[
X_t=
[
Range,
Volume,
TradeCount,
ATMIV,
Put25IV,
Call25IV,
Spread,
BookUpdates
].
]

These fields were motivated inside Volatility Dynamics itself by Modules 03–05.

`return_5m` is excluded from the emission vector.

q99 is therefore not mechanically embedded in the HMM state definition.

## 5. Sparse-L2 enhancement is separate

The following L2 features cover materially less history:

- `book_obi_20_raw_zero_cross_count`;
- `book_obi_20_transition_count`;
- `book_obi_20_raw_zero_cross_rate`;
- `book_obi_20_path_roughness_per_min`.

They are not allowed to shorten the core HMM sample.

Instead Module 06 optionally fits a separate L2-enhanced HMM using:

[
X_t^{L2}=X_t^{core}+L2_t.
]

The L2 model is explicitly exploratory because of its smaller sample.

## 6. Training-only deterministic normalization

The HMM should not simply rediscover that:

- BTC has different scales from TRX;
- one UTC clock slot is naturally more active than another.

For each asset, feature and exact five-minute UTC slot, robust location and scale are estimated using **training data only**.

Exact-slot estimates are shrunk toward the asset-wide training distribution when the slot sample is small.

Every test observation is transformed using those frozen training parameters.

No test observation influences normalization.

## 7. Chronological split

Module 06 creates a chronological split by UTC date.

The default training fraction is 70%.

All HMM parameters are fitted on the first portion only.

Held-out observations are used for:

- test log likelihood;
- filtered state inference;
- q99 hazard;
- state-transition hazard;
- event alignment;
- duration diagnostics.

## 8. Sequence boundaries

The market is continuous across UTC midnight, so date changes do not reset the HMM.

A sequence is broken only by:

- asset change;
- a missing required emission;
- a timestamp gap that is not exactly five minutes.

No transition probability is estimated across those discontinuities.

## 9. HMM implementation

Module 06 uses a branch-local NumPy implementation of a diagonal-Gaussian HMM:

- log-space forward/backward;
- Baum-Welch EM;
- multiple random initializations;
- Viterbi support;
- causal filtered posterior probabilities.

No new external HMM dependency is required.

The core candidate state counts are:

[
K\in\{2,3,4,5\}.
]

The L2-enhanced model defaults to:

[
K\in\{2,3,4\}.
]

## 10. State-count selection

For each K the notebook records:

- best training log likelihood;
- held-out test log likelihood;
- training BIC;
- minimum/maximum training state occupancy;
- convergence behavior across initializations.

The selected K used for downstream q99 evaluation is chosen from **training BIC**, subject to a minimum occupancy guard.

Held-out q99 information does not choose K.

## 11. q99 remains external

Per-asset q99 thresholds are estimated from the training-period distribution of:

[
r_t^2.
]

Those frozen thresholds are then applied to the full chronology.

The HMM never receives q99 labels.

For held-out timestamp t and horizon h:

[
Y_{t,h}
=
1\{\text{q99 onset occurs after }t\text{ within }h\}.
]

The current bar itself is excluded.

Horizons are:

[
30,60,90,120\text{ minutes}.
]

## 12. Filtered-state q99 hazard

For each state k:

[
\hat P(q99_{t+h}\mid S_t=k)
=
\frac{
\sum_tP(S_t=k\mid X_{1:t})Y_{t,h}
}{
\sum_tP(S_t=k\mid X_{1:t})
}.
]

Only filtered probabilities are used.

No future observation can affect the state probability used in this held-out calculation.

Hard argmax-state hazard is also reported as a secondary diagnostic.

## 13. State-transition q99 hazard

Module 06 also asks whether the transition itself is more informative than the destination state.

For exact contiguous five-minute test observations, hard filtered transitions:

[
i\rightarrow j
]

are associated with future q99 incidence.

This can reveal whether an empirical state transition resembles the pre-q99 structural change seen in Module 05.

Rare transitions must not be overinterpreted; counts are reported explicitly.

## 14. q99 event alignment

Held-out q99 events are aligned from:

[
-120m\rightarrow+60m.
]

For each relative bar, Module 06 summarizes filtered state probabilities.

Simultaneous multi-asset q99 events are collapsed by exact timestamp before pooled averaging.

This directly tests whether the -95m motif is the average footprint of a latent state transition rather than a literal fixed-time deterministic signal.

## 15. HMM versus future HSMM

For every state, Module 06 compares:

- empirical mean/median/q90 hard-filtered run duration;
- HMM self-transition probability;
- geometric implied mean duration;
- empirical age-specific exit hazard;
- HMM constant exit hazard.

An HSMM becomes scientifically justified only if:

1. state duration/exit hazard is materially age-dependent or systematically misfit by the HMM; and
2. an HSMM later improves chronological held-out performance.

## 16. Active files

Notebook:

`notebooks/Volatility_Dynamics/06_latent_market_state_hmm.ipynb`

Helper:

`src/data_engineering/modeling/volatility_hmm_states.py`

Expected core outputs:

- `06/06_run_metadata.json`
- `06/06_feature_contract.csv`
- `06/06_clock_normalizer_params.csv`
- `06/06_q99_train_thresholds.csv`
- `06/06_q99_counts.csv`
- `06/06_core_sequence_metadata.csv`
- `06/06_core_model_comparison.csv`
- `06/06_core_initial_probabilities.csv`
- `06/06_core_emissions.csv`
- `06/06_core_transitions.csv`
- `06/06_core_filtered_test.csv`
- `06/06_core_state_q99_hazard.csv`
- `06/06_core_transition_q99_hazard.csv`
- `06/06_core_q99_alignment.csv`
- `06/06_core_duration_summary.csv`
- `06/06_core_duration_hazard.csv`

If the L2 model has enough complete-case data, equivalent `06_l2_*.csv` outputs are also written.

## 17. Interpretation boundary

A useful Module 06 result could show that:

- recurrent latent states exist;
- one or more filtered states have elevated future q99 hazard;
- a specific state transition has higher q99 hazard than either state alone;
- q99-aligned state probabilities change systematically around the -95m region.

None of those findings alone proves an HSMM, Bridge state, or trading strategy.

Those layers must be earned separately.
