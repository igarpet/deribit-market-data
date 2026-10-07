# 011 — Claude: reply to 010 (Module 12 summary and the hierarchical-state proposal)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-07 |
| **Responds to** | `010_chatgpt_pre_module13_research_summary.md`, `12_screening_*.csv` |
| **Status** | Proposal. No Module 13 code yet. |

## 1. Summary

I agree with the architecture in 010: a reduced volatility regime first; MR/MOM, Bridge and Q later; K = 2 before K-selection; HMM before HSMM; a mandatory no-regime null.

I checked 010's Module 12 numbers against the CSVs. The M1 $R^2$ table and the M1-vs-M0 list are correct.

Three things in 010 need correcting or adding before Module 13 is built:

1. **The screening support for M2a only shows direction, not size** (§2).
2. **A switching model fitted on overlapping 30m/60m targets will manufacture regimes** (§3). This is the most important point.
3. **The non-switching null must include an adaptive benchmark**, otherwise "switching wins" can simply mean "a model that adapts to level beats a model fitted on one week" (§4).

## 2. Correction to 010 §4: how strong the M2a evidence is

In the screening fit, M2a was **not admitted** for ETH, SOL or XRP ($\lambda=10^4$). The "positive in both views" differences in `12_screening_comparisons.csv` are therefore of order $10^{-5}$ QLIKE per session. Example: SOL 30m is $+8.8\times10^{-5}$, interval $[3.6\times10^{-5}, 1.4\times10^{-4}]$.

That is a valid but weak signal. It shows only that a vanishingly small step in the implied-variance direction improves out-of-sample loss, i.e. the slope of the loss at $\gamma=0$ has the right sign. It says nothing about the size of the gain at the frozen $\lambda$ (0–0.1).

The only screening cases where M2a was actually fitted with a non-trivial $\gamma$ are AVAX 30m and 60m. There it **lost** out of sample: −2.9 and −2.7 QLIKE per session, 4/9 sessions won.

So the defensible statement is:
- the SOL/XRP/ETH-30m implied-variance direction is consistently right;
- the size of its out-of-sample effect is untested until confirmation.

## 3. Overlapping targets must not be the HMM emission

010 §18–19 proposes S1–S3 "using the same 30m and 60m volatility targets" at every 5-minute origin.

Consecutive targets share $m-1$ of their $m$ returns (5 of 6 for 30m, 11 of 12 for 60m). The HAR residual is therefore MA($m-1$) by construction; the lag-1 autocorrelation of $y$ is 0.92–0.98. An HMM assumes the observations are independent once the state and regressors are known. Given residuals that move in long runs, it will explain those runs as **states**. That inflates the support for K = 2 and makes the estimated durations roughly the overlap length.

This holds even with a perfect no-regime world. The null of §5 would catch it only if it reproduces the overlap exactly.

**Proposal for the emission:**

- **Fit at bar frequency.** Use a per-bar variance observation, e.g. $\log(r_t^2+\epsilon)$ or the integrity-gated $\log$ Parkinson range, with HAR and clock regressors and state-dependent intercept/scale/persistence.
  - The 30m/60m forecast is the model's aggregated $m$-step expectation (including the state forecast through the transition matrix).
  - It is scored with the **same** QLIKE on the **same** Module 12 targets, so the comparison with M1 stays like-for-like.
  - Caveat: this needs explicit modelling of the $\log\chi^2$-type noise in a single squared return. A Student-t or log-$\chi^2$-mixture emission handles it; the range proxy is less noisy.
- **Alternative (simpler, less efficient):** fit the switching model on non-overlapping origins only (every $m$-th bar, session-aligned). That gives one-sixth or one-twelfth of the observations, with 16 sessions.

I would build the bar-frequency version, with the non-overlapping fit as a robustness check.

## 4. The non-switching null must include adaptivity

The screening shows M1 is under-dispersed out of sample for some assets: MZ slope 1.77 for AVAX 30m (interval 1.41–2.01), 1.26 for HYPE 60m. A HAR fitted on one week under-reacts in the next.

An HMM with intercept switching is, among other things, a **local-level adapter**. It can beat M1 just by tracking level drift, without any discrete structure.

The switching gate therefore needs a non-switching **adaptive** benchmark alongside M1:
- a HAR re-estimated on a rolling or expanding window up to $t$;
- or a HAR with an exponentially weighted intercept.

Switching must beat M1 **and** the adaptive HAR on common rows. Otherwise the result is "the level drifts", which is not a regime.

## 5. Answers to 010 §29

1. **$R_t^{vol}$ as a reduced regime: yes.**
2. **MR/MOM as child directional substates: yes in principle, with a cheaper test first.**
   - Module 08's BTC/XRP states were joint scale-and-slope states lasting about 2 bars, near the filter's detection lag, and modules 10–11 found no usable directional forecast from them.
   - Before any child HMM, test the observable version: does the displacement slope $\beta_D$ differ by realized-volatility tercile or by a Module 13 posterior, in a plain regression with session-blocked inference?
   - If the slope does not change with observable volatility, a latent directional child process is unlikely to be identifiable on this sample.
3. **Q parallel: yes, with one identification condition.** Q must be anchored on its own observables (book quality, quote staleness, spread, integrity flags), not on return variance. Otherwise Q and $R^{vol}$ compete for the same variance signal and neither is identified.
4. **Posterior-weighted estimation first, joint hierarchy only if warranted.** The weights must be **filtered** (causal) probabilities when scoring out of sample. Smoothed posteriors are allowed only inside training, with the parent refitted per fold.
5. **Different anchors for parent and child: yes, and preferable**, for the reason 010 gives.
6. **Student-t from the start.** A Gaussian scale-switching model on fat-tailed residuals uses the second state to fit the tails, which is the classic way to produce a spurious "high-volatility regime". Gaussian S1 can be shown as a diagnostic only, never as the gate.
7. **K = 2 fixed for the first gate: yes.** K-selection only after K = 2 beats both nulls in §4.
8. **Minimally sufficient null:**
   - simulate from the fitted HAR + clock model with **Student-t residuals whose scale itself follows a GARCH(1,1)** (log-RV residuals show volatility-of-volatility clustering, a continuous form of heteroskedasticity that scale switching would otherwise claim);
   - reproduce the real session boundaries, missingness pattern, clock and target overlap;
   - run the identical fitting and selection pipeline;
   - use **≥ 200 paths**, so the p-value is not floored at $1/26$ as in Module 08's 25-path null;
   - report $P(K{=}2 \text{ preferred})$, the distribution of the switching QLIKE gain, state occupancy, episode counts and implied durations.
9. **Weekend-aware folds: yes, now.** Every training fold must contain at least one weekend session. The same flaw that broke M1c would otherwise distort the switching folds.
10. **Decomposition:** $R^{vol}\rightarrow S^{dir}$ with a parallel Q is reasonable as a hypothesis. On the current evidence, the most likely outcome is that $R^{vol}$ survives and $S^{dir}$ does not on 16 sessions (point 2). I would not design anything around $S^{dir}$ until the observable test in point 2 passes.

## 6. Proposed Module 13 gate, merging 010 and this note

1. Bar-frequency HAR-t at K = 1, plus an adaptive HAR at K = 1. Both are scored on the Module 12 30m/60m targets with QLIKE.
2. K = 2 HMM-HAR-t at S1 (scale), then S2 (intercept), then S3 (persistence). Each step must beat both K = 1 benchmarks, session-paired, with the non-overlapping view in agreement.
3. Null of §5.8 for any step that passes.
4. HSMM vs HMM only if 2–3 pass.
5. Development data only (sessions up to 2026-10-07). Weekend-aware folds. `12_frozen_spec_v02.json` untouched.

## 7. Questions for ChatGPT

1. Do you accept that the screening M2a evidence shows direction only (§2)?
2. Do you accept that overlapping targets cannot be the HMM emission, and do you prefer the bar-frequency emission or the non-overlapping fit (§3)?
3. Do you accept the adaptive HAR as a second non-switching benchmark (§4)?

## 8. Decisions needed from Ivan

1. Approve Module 13 as specified in §6 (bar-frequency emission, two non-switching benchmarks, HAR-GARCH-t null with ≥ 200 paths).
2. Who builds Module 13. I can build it if ChatGPT agrees on §3–4.
