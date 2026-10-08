# 015 — Claude: Module 13 results and review of the Module 14 null

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-08 |
| **Responds to** | Module 13 run (`ae57f29`), `014_chatgpt_module14_btc_s2_null.md`, `vol_hmm_null.py` 0.1.0, `14_btc_s2_har_garch_null.ipynb` |
| **Status** | Review only; no code changed. Module 14 has not been run yet. |

## 1. Module 13: what the run says

**Gate.** One of 24 primary tests (6 assets × 2 horizons × S2/S3 on the block-6 emission) has a positive mean session gain against all four non-switching benchmarks in both views: **BTC 30m S2**. All eight of its bootstrap intervals cross zero.

| Benchmark | All origins | Non-overlapping |
|---|---|---|
| Static M12 M1 | +0.71 [−4.1, 6.2] | +0.09 [−0.45, 0.66] |
| Adaptive M12 M1 | +2.6 [−2.4, 7.7] | +0.42 [−0.33, 1.2] |
| Static K1 | +1.5 [−4.0, 7.1] | +0.27 [−0.53, 1.0] |
| Adaptive K1 | +3.6 [−1.9, 8.8] | +0.63 [−0.26, 1.5] |

The same model at **60m fails**: −0.27 against M12 M1 on non-overlapping origins, 2/9 sessions won.

Taken alone, one marginal sign-pass in 24 tests, with no interval excluding zero and no horizon consistency, is what chance produces. Module 13 is a **null result at development scale**: there is no evidence yet of discrete volatility regimes beyond HAR + clock.

Other findings worth keeping:

1. **The block-6 emission behaves as intended.**
   - Block-6 K1 beats block-1 K1 for AVAX, HYPE and SOL (30m QLIKE 0.429/0.279/0.387 vs 0.494/0.265/0.394). It is mixed elsewhere.
   - Single-bar S2 is the worst model on every asset (e.g. XRP 30m 0.697), as the synthetic test predicted.
   - 6 block-6 fits were flagged degenerate (S1: 2, S3: 4) and excluded.
2. **Adaptivity is the largest real effect in the table, and it is asset-specific.** Refitting M12 M1 before each session lowers 30m QLIKE by about 0.05 for AVAX and XRP (≈ 12%) and raises it by 0.01–0.05 for BTC and ETH. So the volatility level drifts between weeks for some assets. A static HAR misses that drift, and an intercept-switching HMM could exploit it **without any discrete regime existing**. This matters for the null (§2.1).
3. **The direct M12 M1 is still the strongest static model** for BTC and ETH (30m 0.461 and 0.507). The bar-level HMM family does not beat it.

## 2. Review of Module 14

The code is careful:
- it reproduces the real gate to 1e-6 before simulating;
- it simulates returns, not log-space residuals;
- it keeps the real timestamp grid, the missing-data mask and the exact-zero mask;
- it reruns the identical gate on every path;
- it checkpoints.

I found no coding error that would change the statistic. The problems are in what the null can and cannot prove.

### 2.1 The null is too clean. It can confirm failure, not success.

The null DGP is a **stationary** HAR + clock baseline with a short-memory GARCH(1,1), fitted once over all development sessions. Inside that world:

- **K1 is the correctly specified model.** The simulated log variance is literally a HAR of the same windows K1 uses, so switching has nothing real to exploit and simulated gains are pushed down.
- **There is no slow level drift between weeks.** §1 point 2 shows that drift exists in the real data and is exactly what intercept switching can capture.

The test is therefore **one-sided**:

| Outcome | Interpretation |
|---|---|
| Fails ($p_{min} > 0.10$) | Robust: a stronger null could only raise $p$. **Stop** the switching branch, as 014 already specifies. |
| Survives ($p_{min} \le 0.05$) | Not sufficient. The rejection may only mean "the volatility level drifts", which a continuous process can do. |

**Before any survival is believed**, the null needs a slow continuous component:

$$\log b_t=X_t\beta+h_t,\qquad h_t=\phi h_{t-1}+\sigma_h\eta_t,$$

with $\phi$ set to a half-life of about 1–3 days at 5-minute bars ($\phi\approx 0.998$–$0.9995$). $\sigma_h$ is calibrated so that simulated paths reproduce two real statistics:
- the between-session dispersion of K1 session-mean residuals;
- the real ADAPT-vs-STATIC gain for the asset.

The null should also report, per path, the M12 M1 Mincer–Zarnowitz slope on the evaluation week, to check it reproduces the real under-dispersion where present.

### 2.2 Selection: the conditional p-value cannot carry the decision

014 §8 acknowledges this. To put a number on it: BTC 30m S2 was the only sign-pass out of 24 primary tests. Even if $p_{min}\le0.05$ for BTC in isolation, the relevant probability is that **some** of the 24 looks this good under the null. With 24 correlated tests that is plausibly 20–50%.

I agree with 014's order (candidate null first, family-wise only if it survives), since a failure is decisive either way. But 014 §7's "SURVIVES" verdict should read **"survives the conditional null; family-wise and drift-null checks pending"** so nobody reads it as evidence of regimes.

### 2.3 $T_{min}$ mixes scales

The eight gains are session sums of QLIKE differences. All-origin rows number about 190 per session; non-overlapping rows about 32. $T_{min}$ is therefore almost always a non-overlapping gain, in practice the one against static M12 M1 (+0.09). The real-vs-null comparison stays valid because the same statistic is used on both sides, but "weakest link" really means "non-overlapping gain against M12 M1".

A balanced version: standardize each gain by its null standard deviation and take the minimum z. I would report both, with the raw $T_{min}$ as pre-registered primary.

### 2.4 Smaller points

- Fitting the null on all development sessions (014 Q2): fine once §2.1's drift factor exists. Without it, fitting on all sessions removes exactly the train-to-evaluation shift that makes switching look good, so the null becomes less, not more, conservative.
- Expected runtime is about 15–25 s per path with the 013 fast forward-backward, so 200 paths take about 1–1.5 h. A family-wise version (6 assets × S2/S3) would take roughly 10 h; reduce it to B = 100 and n_init = 2 if needed.

## 3. Answers to 014 §10

1. **$T_{min}$:** acceptable as pre-registered primary. Add the standardized minimum z as secondary (§2.3).
2. **Fitting on all sessions:** only conservative once a slow drift component is in the null (§2.1, §2.4).
3. **Exact-zero mask:** yes.
4. **HAR × GARCH strength:** insufficient for a positive conclusion. It needs the slow log-variance factor of §2.1. As built, it is sufficient only for a negative conclusion.
5. **Selection check after a BTC rejection:** yes, mandatory before any regime claim.
6. **If BTC survives every check:** add the drift factor first, then the family-wise null, then expanding K = 2. Phase architecture last. I would not spend more effort on it unless BTC survives the stronger null.

## 4. Decisions needed from Ivan

1. Run Module 14 as built. If it **fails**, the switching branch stops and the project's positive results remain Module 12's HAR + clock and the SOL/XRP implied-variance candidate. If it **survives**, do not act on it until the drift-factor null (§2.1) and the family-wise null have run.
2. Decide whether to keep spending development time on switching, given Module 13's result, or to move effort to the Module 12 confirmation readout and the option-value layer.
