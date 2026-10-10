# 021 — Claude: Module 10b, a model-free physical benchmark for 0DTE wings

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-11 |
| **Branch** | Volatility Dynamics (inherits Module 10's approved cross-branch reuse) |
| **Responds to** | VD 020 §6.1; Ivan's approval of the Module 09/10 verdicts, the P2 correction and Module 10b |
| **Status** | code — built and validated on synthetic data; awaiting Ivan's real-data run |

## 1. Question

Module 10's RICH verdict compares wings with a model that omits variance uncertainty and multi-bar moves. Are the wings still rich when compared with the **realized** distribution of moves to settlement?

## 2. Method

- **Outcomes.** One outcome is computed for every asset and every 5-minute origin $t$ with 60–1200 minutes to the next 08:00 expiry:

$$
u_t=\frac{\ln(S_T/I_t)}{\sqrt{D_t}},\qquad k_t=\frac{\sum_j w_j^2r_j^2}{D_t}.
$$

  - $D_t$ is exactly Module 10's ex-ante diffusive variance.
  - $S_T$ is the approximate settlement price from Module 10.
  - $k$ is the realized-to-ex-ante variance ratio.

  Quotes and outcomes are standardized by the same $D$, so a level bias in $D$ (for example 5-minute noise inflation) **cancels**. Only the shape of the physical distribution relative to $D$ is tested.

- **Empirical price.**

$$
P^{emp}=\sum_i w_i\,\text{payoff}\!\big(F e^{\sqrt D\,u_i}/\bar m,\;K\big),\qquad \bar m=\sum_i w_i e^{\sqrt D\,u_i}
$$

  - The pool of $u_i$ is **leave-one-session-out** and pooled across assets and TTEs.
  - Every raw outcome in the outer 1% on each side is kept; the centre is compressed into 500 equal-count bin means.
  - My first version used bin means everywhere. That capped the tail at the mean of the top bin and priced every strike beyond about 3σ at zero, so it was replaced before any real data were seen.
  - Dividing by $\bar m$ enforces the martingale condition for each quote: sample drift is removed and shape is kept.

- **Variance mixture (reported only).**

$$
P^{mix}=\sum_i w_i\,\text{B76}(F,K,k_iD).
$$

  RV already contains jumps, so I do **not** add Module 09 jumps. VD 020 §6.1 proposed adding them, which would double count. This turn corrects that.

## 3. Pre-registered (fixed before any real-data Module 10b output)

1. **Rows.** Module 10's primary sample, unchanged: non-floor, $z\in[2,5)$, from `10/10_priced_quotes.csv.gz`.
2. **Statistic.** $M$ = median over sessions of the session median of $\ln(\text{mid}/P^{emp})$.
3. **Uncertainty: joint session bootstrap, B = 1000.** Each replicate resamples sessions and rebuilds **both** the quote set and the LOSO outcome pool. The uncertainty of the physical tail itself therefore enters the confidence interval. A quote-only bootstrap would understate it badly, because the tail rests on about 133 session-asset outcomes.
4. **Verdict.**
   - **RICH** if the 2.5% bootstrap quantile is > 0.
   - **CHEAP** if the 97.5% quantile is < 0.
   - **CONSISTENT** otherwise.
5. **Zero-price rows.** Rows with $P^{emp}=0$ have a strike beyond every observed outcome. They are dropped from the point estimate and from every replicate. This is conservative against RICH: a positive market price for an unobserved event is not evidence of richness.
   - If more than 25% of primary rows are dropped, the verdict is **INSUFFICIENT_TAIL**.
6. **Reported, no decision.**
   - The variance mixture, with a joint bootstrap of B = 300.
   - The TTE-conditioned pool.
   - Per asset.
   - All $z\times$TTE buckets.
   - Tail frequencies $P(|u|>x)$ against the normal distribution, the number of distinct session-asset outcomes beyond each $x$, and the distribution of $\log k$.

**Interpretation is fixed in advance.**
- **RICH:** the wings carry a premium over realized outcomes. This is the precondition for researching a wing-selling strategy; it is not yet a strategy.
- **CONSISTENT:** with about 19 sessions, a premium cannot be told apart from fair tail pricing, and Module 10's RICH is explained by the model's thin tails.
- **INSUFFICIENT_TAIL:** the sample cannot price the primary bucket yet. Rerun as sessions accumulate, up to the 2026-11-07 boundary (about 48 sessions).

## 4. Validation (synthetic only)

- **Self-test.**
  - A Gaussian $u$ reproduces Black-76.
  - $k\equiv1$ reproduces Black-76.
  - In a $t_3$ world, the correct benchmark gives CONSISTENT and a Gaussian benchmark gives RICH.
  - LOSO excludes the session's own outcomes.
  - A constant-volatility Gaussian panel gives sd($u$) ≈ 1 and median $k$ ≈ 1.
- **Synthetic research DB.** That DB has strongly stochastic volatility (sd of $\log k$ is 1.3). On it, Module 10 says RICH with $M=3.98$, while 10b gives **CONSISTENT**: $M=-0.36$, CI [−0.98, 0.82]. This is exactly the confound 10b is designed to detect.

## 5. Files and run

- `src/data_engineering/modeling/volatility_wings_empirical.py` (0.1.0).
- `notebooks/Volatility_Dynamics/10b_model_free_wing_benchmark.ipynb`, with outputs in `10b/`. It needs Module 10's outputs to be present.
- `tests/test_volatility_dynamics_b.py`: 13 tests.
- Runtime is roughly 10 minutes for the primary and 15–20 minutes for the mixture, extrapolated from the synthetic run.

**PC (Windows CMD):**

```
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_volatility_dynamics_b.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=10800 notebooks\Volatility_Dynamics\10b_model_free_wing_benchmark.ipynb
git add notebooks\Volatility_Dynamics\10b_model_free_wing_benchmark.ipynb notebooks\Volatility_Dynamics\10b
git commit -m "Run Volatility Dynamics 10b"
git push
```

## 6. Questions for ChatGPT

1. Is dropping zero-price rows, combined with the 25% INSUFFICIENT_TAIL rule, the right conservative treatment? The alternative would be extrapolating a fitted tail, which adds a model back in.
2. Should the pool be per asset rather than pooled? Pooling assumes the shape of $u$ relative to $D$ is common across assets. The per-asset table reports how far that holds.

## 7. Decisions needed from Ivan

1. Run Module 10b as in §5.
2. Confirm that §3 is binding.
