# Volatility Dynamics — ChatGPT / Claude research dialogue

Research branch for volatility dynamics developed independently from every other research branch.

- **001**: `001_chatgpt_module01_design.md` — historical first design; **superseded** because it reused prior-branch HAR/volatility logic.
- **002**: `002_chatgpt_all_available_data.md` — historical data-window decision; superseded as part of the full reset.
- **003**: `003_chatgpt_full_independence_reset.md` — authoritative reset: return-only Module 01, no imported research outputs or assumptions.
- **004**: `004_chatgpt_timestamp_unit_bugfix.md` — Pandas datetime-unit bugfix for the standalone engine.
- **005**: `005_chatgpt_native_5m_module02.md` — native five-minute squared-return analysis; no rolling 30m block.
- **006**: `006_chatgpt_q99_precursor_module03.md` — full-schema q99 precursor discovery with strict pre-event information boundary and matched controls.
- **007**: `007_chatgpt_module03_2h_and_conditional_module04.md` — extends Module 03 anatomy to ±120m and adds native-volatility-conditioned q99 precursor Module 04.
- **008**: `008_chatgpt_minus95m_motif_module05.md` — dedicated five-candle -95m motif study with prior-q99 contamination exclusion and stability checks.
- **009**: `009_chatgpt_latent_state_hmm_module06.md` — first latent-state HMM: training-only normalization, chronological held-out filtering, external q99 hazard, and duration diagnostics before any HSMM.
- **010**: `010_claude_review_and_b_run.md` — Claude review of Modules 01–06 (Module 04 matching failed its own attenuation check; −95m motif on 13 timestamps; HMM hazard on 5 events / 3 timestamps, no benchmark) and the parallel **b-run** 02b–06b with pre-registered gates.
- **011**: `011_claude_b_run_results.md` — b-run results: 04b gate passed and no field survives volatility conditioning; 05b insufficient data (5 timestamps); 06b HMM fails vs the volatility benchmark (K=7 is the grid edge, not a regime count); hourly clock real, exact-slot mostly noise.
- **012**: `012_claude_student_t_hmm_06c.md` — Student-t emission HMM (06c): tests whether 06b's grid-edge K was inflated by heavy tails; same pre-registered forecast decision as 06b.
- **013**: `013_claude_06c_results.md` — 06c: Student-t fits better but K still climbs to the grid edge (10); states encode assets/days; adding states significantly worsens the 60m q99 forecast. Recommendation: close the latent-state line in this branch.
- **014**: `014_chatgpt_q99_preshock_hmm_taxonomy_module07.md` — conditions on q99 shocks and clusters their strictly pre-shock 24-bar HMM trajectories into candidate shock types.
- **015**: `015_claude_module07_review.md` — Module 07 review: SyntaxError fixed (literal \n in `adjusted_rand_index`); on a synthetic single-factor world Module 07 finds 4 "types" passing all VD 014 §13 criteria (volatility-level bins). Proposed 07b with volatility-matched non-shock controls, nulls and numeric thresholds.
- **016**: `016_claude_module07_results.md` — Module 07 results: 58 L2-covered timestamps; 2 'types' with silhouette 0.24 (below the synthetic no-types reference), split by IV level/activity, validation ~one day, unstable under collector sensitivity. No evidence of shock types; recommend pausing the line.
- **Next turn:** `017`.

Module 01 remains historical baseline work. **Module 02 is authoritative for primitive volatility persistence/duration questions:** `notebooks/Volatility_Dynamics/02_native_5m_volatility.ipynb`.

Follow `../README.md`: one immutable discussion file per turn; evidence before interpretation.
