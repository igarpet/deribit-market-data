# Signed Jumps 001 — SJ 01 preregistration: volatility engine + signed Hawkes jump layer (1-minute spot)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-16 |
| **Branch** | Signed Jumps (new) |
| **Responds to** | Ivan's design discussion of 2026-10-16 (chat); prior evidence VD 008 / VD 018 |
| **Status** | Preregistration. Fixed before any real-data output. Code `signed_jumps.py` (SJ01-0.1.0), tests `tests/test_signed_jumps.py` (11 pass) |

## 1. Question

Do 1-minute price jumps **self-excite**, does the **sign** of a past jump change the sign of the next one, and does **asymmetry in the continuous flow** (jump-truncated skewness) predict the sign of the next jump? The output is a signed jump-intensity forecast meant to choose between a single option wing, a straddle, or no trade (SJ 02). This module forecasts no mean returns.

**Cross-branch reuse (approved by Ivan, 2026-10-16).** The continuous component re-implements the Volatility Dynamics AM-T-HAR-Blend structure ("volatility engine"): HAR variance plus a fast/slow EWMA state (L, F). It is refitted here on jump-robust bipower variation at 1 minute. No VD fitted parameters, thresholds or conclusions are imported.

## 2. Prior evidence (stated, not used in the decision)

**VD 008 / VD 018** tested unsigned jumps on 5-minute Deribit index data: 20 days, 245 jumps at c = 4, with a two-sided local volatility estimate.
- Result: homogeneous Poisson. ρ ≈ 0.05, and the Hawkes model was significantly *worse* out of fold.
- That test had low power and was explanatory, not real-time.
- This module has about 26 months of 1-minute data, a real-time (backward) volatility forecast, and signs.
- **Prior: against self-excitation.**

## 3. Data

- **Source:** Binance spot 1m klines, `<BINANCE_UM_DIR>/basis/spot/<A>USDT_spot_1m.parquet`. These are the files downloaded for Pairs fOU 02.
- **Assets:** BTC, ETH, SOL, XRP (Ivan, 2026-10-16). AVAX has no spot file. TRX and HYPE are excluded from future tests.
- **Bars:** indexed by bar-end time; returns are log close-to-close, NaN across gaps.
- **Split:** the first 70% of calendar days are DEV, the last 30% TEST. Every parameter (seasonality, HAR, standardisations, Hawkes coefficients, kernel half-life) is fitted on DEV only. TEST is scored once.

## 4. Model

**Continuous engine (all quantities known at the close of bar $t-1$):**

$$\tilde r_t=\frac{r_t}{s(\text{slot},\text{weekend})},\qquad bp_t=\tfrac{\pi}{2}|\tilde r_t||\tilde r_{t-1}|$$

- $\log s$ is a 4-harmonic Fourier curve plus a weekend dummy, fitted to $\log|r|$ on DEV.
- HAR on log bipower: $\log \overline{bp}_{[t,t+59]} \sim 1+\log\overline{bp}_{30}+\log\overline{bp}_{60}+\log\overline{bp}_{240}+\log\overline{bp}_{1440}$, with backward windows ending at $t-1$. The forecast is $\hat\sigma^2_{c,t}=\exp(\hat y+s^2/2)$.
- Engine state: EWMAs of $\log bp$ with half-lives 30 and 240 minutes. $L$ = slow − DEV mean, $F$ = fast − slow.

**Jumps:**

$$Z_t=\tilde r_t/\hat\sigma_{c,t},\quad J^+_t=1\{Z_t\ge c,\ r_t\ge 5\text{ bp}\},\quad J^-_t=1\{Z_t\le -c,\ r_t\le -5\text{ bp}\},\quad c=4.$$

**Covariates, all strictly through $t-1$:**
- **Variance surprise** $TV_w$: $\log$ mean $Z^2$ over the last $w\in\{10,60\}$ bars with $|Z|<c$. This was added after the synthetic null in §6; it is the realised-to-forecast continuous-variance ratio.
- **Truncated skewness** TSK: skewness of $Z$ over the last 60 bars with $|Z|<c$ (at least 45 valid), standardised on DEV.

**Signed discrete Hawkes, per sign $s$:**

$$P(J^s_t=1\mid\mathcal F_{t-1})=1-e^{-\lambda^s_t},\qquad \lambda^s_t=e^{X^s_t b_s}+\sum_{s'}\eta_{s s'}E^{s'}_t,\qquad E^{s'}_t=\sum_{k\ge1}(1-q)q^{k-1}J^{s'}_{t-k}$$

- The kernel sums to 1, so $\eta$ is the branching matrix.
- $q=2^{-1/h}$, with the half-life $h\in\{1,3,10,30,90\}$ minutes profiled on DEV.

**Ladder (each nests the previous; baselines are sign-specific):**

| Model | Adds | Meaning |
|---|---|---|
| (a) | asset intercepts | homogeneous Poisson |
| (b) | Fourier(4), weekend, $L$, $F$, $TV_{10}$, $TV_{60}$ | engine-modulated Poisson (**null**) |
| (c) | $\eta_{s+}=\eta_{s-}=\eta_s$ | excitation, parent sign irrelevant |
| (d) | $\eta_{ss'}$ free | parent sign matters (conditional sign) |
| (e) | $\theta_s\,\text{TSK}$ | pre-jump asymmetry |

The unconditional size asymmetry ("down jumps are larger") is reported descriptively in `01_jump_sizes.csv`. It is not a gate.

## 5. Gates (TEST; per-row log-likelihood summed over both signs)

Bootstrap: 2,000 calendar-day blocks, with days shared across assets. Every gate also requires at least 200 TEST jumps per sign.

| Gate | Comparison | PASS iff |
|---|---|---|
| **G1 excitation** | (c) vs (b) | Δ logLik/row > 0 and q2.5 > 0 |
| **G2a parent sign** | (d) vs (c) | Δ > 0 and q1.25 > 0 (Bonferroni, 2 tests) |
| **G2b truncated skew** | (e) vs (d) | Δ > 0 and q1.25 > 0 |
| **G2** | — | G2a or G2b |
| **G3 horizon** | Selected model vs (b): signed occurrence log-loss of P(≥1 jump of each sign in the next 15 min) | Δ > 0 and q2.5 > 0 |

**Notes on G3:**
- Origins are every 15 minutes on TEST. Secondary horizons are 5 and 60 minutes.
- Covariates are frozen at the origin, with the true future clock. (b) is computed exactly.
- Excitation models use a conditional Monte Carlo estimator with 200 paths. On {no target-sign jump yet}, the target's excitation only decays, so $P(\text{none in }h)=E_Q[\prod_j e^{-\lambda^{\text{target}}_j}]$, with only the other sign simulated. It is unbiased and needs no smoothing.
- Plain hit-counting with Laplace smoothing was rejected on synthetic data: it added about 0.5/(M+1) ≈ 0.0025 to probabilities near 0.01 at 5 minutes, biasing excitation models upward.
- The selected model is (e) if G2b passes, else (d) if G2a passes, else (c) if G1 passes, else (d) for the record.

**SJ 02 (economics)** is built only if G2 and G3 both pass. It would be BTC/ETH Deribit 0DTE, single wing chosen by the predicted up-share versus straddle, at ask, entered only after the signal exists. Its decision rule, $P(L\mid x)(2\pi-1)E|J|$ net of costs, will be preregistered separately.

**Reported, not gated:**
- η matrix with day-clustered sandwich SEs (DEV);
- branching ratio;
- continuation index $(\eta_{++}+\eta_{--})-(\eta_{+-}+\eta_{-+})$;
- up-share calibration;
- c ∈ {3.5, 4.5, 5} sensitivity of (b), (c), (d);
- jump sizes by sign.

## 6. Synthetic validation (before any real data; `tests/test_signed_jumps.py`)

1. **Planted signed excitation** (η₊₊ = 0.35, η₋₋ = 0.45, cross terms 0, half-life 10, thin-tailed diffusion):
   - recovered 0.32 / 0.27 with cross terms 0 and half-life 10;
   - G1 and G2a pass;
   - G3 beats (b) at all horizons.
2. **VD 08 trap:** common stochastic volatility with fast bursts (half-life ≈ 70 min), Student-t(4), no excitation.
   - **Without** $TV$ in the baseline: 1 of 4 seeds falsely passed G1, with ρ up to 0.11.
   - **With** $TV$: 0 of 4, ρ ≤ 0.056.
   - Slower volatility (persistence 0.9995): 0 of 8, ρ ≤ 0.034.
   - G2a never false-passed: the artefact is sign-symmetric.
   - This is why $TV_{10}$ and $TV_{60}$ are in the null.
3. **Planted pre-jump skew:** θ recovered at ±1.07 and G2b passes. With no skew, G2b fails.
4. **Causality:** perturbing returns after bar $k$ leaves $Z$, $\hat\sigma_c$, $L$, $F$, TSK, $TV$ and jumps at bars ≤ $k$ unchanged.
5. The kernel recursion matches the explicit sum, and the analytic gradient matches finite differences.
6. **Horizon estimator:** with η = 0 it reproduces the exact Poisson probabilities. With planted cross-excitation it matches a 20,000-path brute-force simulation: pooled bias below 0.002 at P ≈ 0.34.

## 7. Known limitations

1. **Attenuation.** With Student-t(4) diffusion at c = 4, roughly half of detected "jumps" were diffusive tail draws, which shrank planted η of 0.35–0.45 to 0.04–0.09. Real η estimates are therefore lower bounds, and a G1 FAIL is weaker evidence than a PASS. The c sensitivity shows the direction of the effect.
2. **Null floor.** Under the synthetic nulls, ρ reached about 0.06 with no true excitation. Branching ratios below about 0.06 are not interpretable as excitation.
3. **Univariate per asset.** There is no cross-asset excitation, and same-minute co-jumps across assets are not modelled. A cross-asset model is a later step only if G2 passes.
4. **No mark dependence.** Jump size does not scale excitation in this version.
5. **Bar-close timing.** A jump is known at the close of its minute. Anything acting on it trades at the next bar or later; SJ 02 must enforce this.

## 8. Expectation (stated in advance)

- **G1:** uncertain, leaning FAIL given VD 008, the attenuation and the $TV$ control. If it passes, I expect ρ ≈ 0.05–0.15 with a short half-life.
- **G2a:** likely FAIL.
- **G2b:** most likely FAIL; third moments of small 1-minute returns are noisy.
- **G3:** passes only if G1 does.
- I expect no SJ 02.

## Questions for ChatGPT

1. Is the $TV_{10}$/$TV_{60}$ control too strong, i.e. could genuine short-lived excitation of *small* returns after a jump be absorbed? (Jumps are excluded from $TV$, but the bars right after a jump are not.)
2. Any objection to profiling the half-life on DEV instead of fixing it?

## Decisions needed from Ivan

1. Approve the protocol, or change it before the run. Nothing here is changed after results.
2. Runtime is about 30–60 minutes on the PC, with c sensitivity included.
