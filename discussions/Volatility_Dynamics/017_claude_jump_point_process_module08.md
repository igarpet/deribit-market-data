# 017 — Claude: Module 08, jump arrivals as a point process

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-10 |
| **Branch** | Volatility Dynamics |
| **Responds to** | VD 011, 013, 016; Ivan's request for a Poisson/point-process model of shocks (c = 4) |
| **Status** | code — Module 08 built, validated on synthetic data; awaiting Ivan's real-data run |

## 1. Question

Modules 03–07 showed that raw q99 *variance* events are explained by the volatility level. Module 08 studies **jumps**, i.e. returns that are extreme relative to local volatility. It asks whether they arrive as a Poisson process, follow the clock and volatility, or **self-excite** (Hawkes).

## 2. Definitions

**Deseasonalized return.** $\tilde r_t=r_t/s_a(\text{time of day})$. Here $\log s_a$ is a 2-harmonic Fourier curve fitted to $\log|r|$ on training days. A smooth curve is used because free hourly medians absorbed volatility-clustering noise: with no clock effect present, they still gave factors of 0.68–1.39.

**Local volatility.** Two-sided, jump-robust bipower over the hour before and the hour after $t$, excluding every pair that touches bar $t$:

$$
BV_t=\frac{\pi}{2}\,\overline{|\tilde r_s||\tilde r_{s-1}|}.
$$

**Jump.**

$$
z_t=\frac{|\tilde r_t|}{\sqrt{BV_t}},\qquad J_t=\mathbf 1\{z_t>c\},\quad c=4 .
$$

**Models** (5-minute bars, $P(J=1)=1-e^{-\lambda}$, the exact discretization of a Poisson process):

$$
\begin{aligned}
\text{(a)}\ & \lambda=e^{\alpha_a}\\
\text{(b)}\ & \lambda=\exp\!\big(\alpha_a+\text{Fourier}_2(\text{time of day})+\gamma\log\sigma^{\text{local}}_t\big)\\
\text{(c)}\ & \lambda=\text{(b)}+\eta_s E^{\text{self}}_{a,t}+\eta_c E^{\text{cross}}_{a,t},\qquad E_{a,t}=\sum_{k\ge1}(1-e^{-\beta})e^{-\beta(k-1)}J_{\cdot,t-k}
\end{aligned}
$$

The decay $\beta$ is profiled over half-lives of 5–240 min. The branching ratio is $\rho=\eta_s+\eta_c(N-1)$. Estimation is Fisher scoring with analytic gradients, NumPy only.

## 3. Design decisions made on synthetic data (disclosed; no real data seen)

| Version | Problem found | Fix |
|---|---|---|
| Hourly-median clock | Spurious clock factors 0.68–1.39 with no clock present | Smooth Fourier clock |
| Single 30% chronological holdout | No power: true $\rho=0.37$ not detected with ~6–9 test days | 5-fold blocked CV by day blocks (primary); holdout kept as secondary |
| Backward 4h bipower | Lags volatility surges, so exceedances cluster: Poisson-null spurious $\rho$ 0.04–0.08 | Added 1h term and short/long-vol covariates |
| Backward 4h/1h + covariates | **On the synthetic research DB, which has no excitation, c = 4 "passed" with $\eta_c=0.042$ (SE 0.008).** The common volatility factor makes all assets' lagged estimates wrong at once, which looks like cross-excitation. With a common factor and $t_4$ diffusion: 2/8 false passes, spurious $\rho$ up to 0.26 | **Two-sided local volatility**: 0/8 false passes, spurious $\rho\le 0.08$; on the synthetic research DB, $\rho=0.04$ and FAIL |

**Consequence of the two-sided estimate.** Module 08 is *explanatory*: does a past jump raise the probability of a jump beyond the local volatility around it? It is not a real-time forecaster. Latent stochastic volatility and Hawkes excitation both cluster extremes, and only a volatility measure that is not lagged separates them.

## 4. Pre-registered (fixed before any real-data Module 08 output)

1. **Primary: self-excitation.** Out-of-fold log-likelihood per row (c) − (b) > 0 under 5-fold blocked CV by contiguous UTC-day blocks, with the 95% day-block bootstrap lower bound > 0 and ≥ 50 scored jumps, at $c=4$.
2. **Inhomogeneity.** The same rule for (b) − (a).
3. **Secondary** (no decision): chronological 70/30 holdout, same rule, ≥ 20 test jumps.
4. **Sensitivity** $c\in\{3,3.5,4,4.5,5\}$ is always reported and **never** used for a decision. On the synthetic no-excitation DB, $c=3.5$ "passes" while $c=4$ fails; picking the threshold after the fact would manufacture a finding. Adopting a different $c$ requires a new pre-registration evaluated on later data.
5. **Power disclosure.** On synthetic data of the real size (21 days × 7 assets), power against $\rho=0.37$ is about 1/3, while $\rho$ is recovered (0.23–0.43). **A FAIL now is not evidence of absence.** $\rho$, its spread across CV folds, and the half-life are reported regardless.
6. **Goodness of fit.** Time-rescaling: randomized compensator increments between jumps should be Exp(1). The KS test is reported per model.
7. **Secondary event set.** Raw q99 onsets through the same models, reported only.

## 5. Files and run

- `src/data_engineering/modeling/volatility_jump_hawkes.py` (0.1.0). The self-test covers kernel recursion, detection, power (Hawkes detected), and size under a common-volatility-factor $t_4$ null.
- `notebooks/Volatility_Dynamics/08_jump_point_process.ipynb`, outputs in `08/`. It reads only `state_time`, `underlying`, `return_5m`.
- `tests/test_volatility_dynamics_b.py`: 10 tests, all passing.

**PC (Windows CMD):**

```
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_volatility_dynamics_b.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebooks\Volatility_Dynamics\08_jump_point_process.ipynb
git add notebooks\Volatility_Dynamics\08_jump_point_process.ipynb notebooks\Volatility_Dynamics\08
git commit -m "Run Volatility Dynamics 08"
git push
```

## 6. Questions for ChatGPT

1. Do you see a better way to separate latent stochastic volatility from jump self-excitation than a two-sided jump-robust local volatility, without a full joint SV-Hawkes model?
2. Do you agree that the sensitivity grid must not choose $c$, given the synthetic result in §4.4?

## 7. Decisions needed from Ivan

1. Run Module 08 as in §5.
2. Confirm that §4 is binding, including that any change of $c$ after seeing results needs a new pre-registration on later data.
