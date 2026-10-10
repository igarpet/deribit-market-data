# 012 — Claude: Student-t emission HMM (06c)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-10 |
| **Branch** | Volatility Dynamics |
| **Responds to** | VD 011 §2.4; Ivan's request for a Student-t variant of 06b |
| **Status** | code — 06c built, tested on synthetic data; awaiting Ivan's real-data run |

## 1. Why

In 06b the Gaussian HMM's validation log-likelihood per row rose at every K up to the edge of the grid (K = 7, `06b/06b_k_selection.csv`). The leading explanation is emission misspecification: with heavy-tailed fields, a Gaussian HMM adds states to approximate the tails. If so, a heavy-tailed emission should stop the climb at an interior K. 06c tests that.

## 2. Model

For state $k$:

$$
x_t\mid S_t=k\ \sim\ t_{\nu_k}\!\left(\mu_k,\ \operatorname{diag}(s_k^2)\right)
\quad\Longleftrightarrow\quad
x_t\mid u_t,k \sim \mathcal N\!\left(\mu_k,\ \operatorname{diag}(s_k^2)/u_t\right),\ \ u_t\sim\Gamma\!\left(\tfrac{\nu_k}{2},\tfrac{\nu_k}{2}\right).
$$

- **One latent scale per bar.** It is shared by all fields, so an activity burst inflates range, volume and trades together.
- **Per-state $\nu_k$,** estimated rather than fixed.
- **Missing fields are marginalized exactly.** The marginal of a multivariate $t$ on the $p_t$ observed coordinates is $t_{\nu_k}$ with the same $\nu_k$.
- **Estimation:** Baum–Welch with ECME (Liu & Rubin 1995). The E-step weight is $E[u_t\mid x_t,k]=(\nu_k+p_t)/(\nu_k+\delta_{tk})$. $\nu_k$ solves the score equation by bisection on $[2.05, 300]$; the upper bound means the state is effectively Gaussian.
- **Everything else is the 06b pipeline unchanged:** emissions, split, hour normalization, sequences, benchmark, decision rule and duration check, imported from `volatility_hmm_states_b`.

Code: `src/data_engineering/modeling/volatility_hmm_states_c.py` (0.1.0), notebook `06c_student_t_hmm.ipynb`, outputs in `06c/`. Self-test and unit test are added to `tests/test_volatility_dynamics_b.py` (9 tests, all passing).

## 3. Does the test discriminate? (synthetic, known truth)

The test data are 3 states, $t_3$ emissions, 5 fields, with 5% missing. Validation LL/row on identical sequences:

| K | Gaussian | Student-t |
|---|---|---|
| 2 | −6.837 | −5.561 |
| **3** | −5.834 | **−5.035** |
| 4 | −5.434 | −5.042 |
| 5 | −5.320 | −5.038 |
| 6 | −5.202 | −5.046 |
| 7 | **−5.134** | −5.046 |

The Gaussian reproduces 06b's pattern (monotone, K = 7 at the grid edge). The Student-t peaks at the true K = 3 and recovers $\nu\approx(2.9, 2.9, 3.0)$. So 06c can separate "heavy tails inflate K" from "there really are many states".

## 4. Pre-registered (fixed before any real-data 06c output exists)

1. **Primary decision** is identical to 06b (VD 010 §5.4): at $h=60$ min, $\text{logloss}(M3)-\text{logloss}(M1)<0$ with 95% day-bootstrap upper bound $<0$ and ≥ 20 unique held-out onset timestamps.
2. **K question.** The grid is K = 2–10. *Interior K* iff the selected K < 10 **and** validation LL/row at K = 10 is not above the selected K's.
3. **Emission question.** The Student-t fits better iff its validation LL/row at its selected K exceeds the Gaussian's at the Gaussian's selected K. Both are fitted in the same run on identical sequences; the Gaussian uses one initialization per K to save time, and this is stated as such.

How results will be read:
- **Interior K with the t fitting better** supports "06b's K was inflated by tails". The number of states is then estimable, but that says nothing yet about whether they are useful.
- **Usefulness is decided only by item 1.**
- **No outcome of 06c bears on R × Q.** The emissions contain no displacement-from-VWAP variable, so no state can be a displacement regime (VD 011 §3).

## 5. Run (Ivan)

**PC (Windows CMD)**, with the notebook closed in Jupyter:

```
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_volatility_dynamics_b.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 notebooks\Volatility_Dynamics\06c_student_t_hmm.ipynb
git add notebooks\Volatility_Dynamics\06c_student_t_hmm.ipynb notebooks\Volatility_Dynamics\06c
git commit -m "Run Volatility Dynamics 06c"
git push
```

Runtime is about 1 h on a desktop CPU: 9 values of K × (3 Student-t + 1 Gaussian) fits. To halve it, run `set VD6C_N_INIT=2` before `jupyter nbconvert`; that changes only the number of random restarts. Do not change `VD6C_K`, since the K question is pre-registered on 2–10.

## 6. Questions for ChatGPT

1. Is a shared per-bar latent scale (multivariate $t$) the right choice over independent univariate $t$ per field, given how activity fields co-move in bursts?
2. If 06c gives an interior K and still fails item 1, do you agree the latent-state line in this branch should stop until new emission variables are justified?

## 7. Decisions needed from Ivan

1. Run 06c as in §5.
2. Confirm that the three items in §4 are binding.
