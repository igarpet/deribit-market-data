# Signed Jumps 002 — SJ 01 readout: jumps self-excite and the sign is predictable, but the effect lives in the first 1–3 minutes

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-16 |
| **Branch** | Signed Jumps |
| **Responds to** | Signed Jumps 001; Ivan's run `8a62395` |
| **Status** | Results against the rules preregistered in SJ 001 §5. **All gates PASS.** SJ 02 is unlocked by rule. |

## 1. Verdicts

TEST covers 2026-02-15 → 2026-10-14: 242 days, 1.39 M rows, 3,546 up and 3,212 down jumps. Bootstrap is over calendar-day blocks.

| Gate | Δ logLik/row | Lower bound | Result |
|---|---|---|---|
| G1 excitation, (c) vs (b) | +0.00083 | q2.5 +0.00066 | **PASS** |
| G2a parent sign, (d) vs (c) | +0.000046 | q1.25 +0.000023 | **PASS** |
| G2b truncated skew, (e) vs (d) | +0.000069 | q1.25 +0.000043 | **PASS** |
| G3 15-minute signed forecast, (e) vs (b) | +0.0019 | q2.5 +0.0014 | **PASS** (5 min and 60 min also pass) |

**Sensitivity to the jump threshold c** (`01_c_sensitivity.csv`): G1 and G2a pass at every c ∈ {3.5, 4, 4.5, 5}. ρ falls from 0.19 to 0.15 as c rises, which is the direction the attenuation argument in SJ 001 §7 predicts.

**Branching ratio:** ρ = 0.155, well above the synthetic null floor of about 0.06. **This overturns VD 008's Poisson conclusion.** That test was underpowered: 245 jumps at 5 minutes, against about 13,000 jumps at 1 minute here.

## 2. What the model found

**Excitation matrix** (model (d), DEV, day-clustered SEs; `01_param_se.csv`):

| Child \ parent | After an up jump | After a down jump |
|---|---|---|
| **Up** | $\eta_{++}=0.092$ (t = 15) | $\eta_{+-}=0.073$ (t = 13) |
| **Down** | $\eta_{-+}=0.037$ (t = 10) | $\eta_{--}=0.112$ (t = 14) |

**Half-life: 1 minute,** the shortest value on the grid.

- **After an up jump:** 0.13 offspring jumps. **71%** of them are up (continuation).
- **After a down jump:** 0.19 offspring. **60%** are down, and the 40% up is a bounce, i.e. reversal is substantial.
- **The effect is concentrated.** In the minute after an up jump, P(up jump) ≈ $\eta_{++}(1-q)=0.046$, against a baseline of about 0.0007. That is ~65× higher. It decays almost entirely within 2–3 minutes.

**Truncated skewness (model (e)):**
- θ₊ = +0.159 (t = 9.0), θ₋ = −0.150 (t = −7.9). Positive skew in recent non-jump returns raises up-jump intensity and lowers down-jump intensity, symmetrically.
- At ±2 s.d. of TSK, the up/down odds shift by a factor of about $e^{0.6}\approx1.8$.

**Out-of-sample sign calibration** (`01_sign_calibration.csv`): predicted up-share among 5-minute windows with at least one jump, in quintiles.

| Predicted up-share | Realised up-only | Realised down-only |
|---|---|---|
| 0.39 | 0.40 | 0.56 |
| 0.66 | 0.62 | 0.36 |

At 5 and 15 minutes this is close to the diagonal. At 60 minutes the realised spread shrinks (0.41 → 0.52), consistent with the 1-minute half-life.

**Baseline (descriptive):**
- **Weekend:** +0.23 to +0.28, i.e. more jumps per unit of volatility. Thin liquidity is the likely cause.
- **L:** −0.21. Standardised jumps are rarer when volatility is already high, which matches VD 008's γ = −0.59.
- **TV₁₀:** +0.75 to +0.84, by far the strongest baseline term. The variance-surprise control added after the synthetic null is doing real work, and G1 passed *with* it in the null.

**Jump sizes:** up and down are symmetric. BTC 26.4 vs 26.3 bp mean; ETH, SOL and XRP within 1–2 bp of each other (`01_jump_sizes.csv`). **The "down moves are larger" premise does not hold for 1-minute jumps in this sample.**

## 3. Caveats

1. **The half-life is at the grid edge (1 bar).** Part of the diagonal could be one move straddling a minute boundary and being counted twice. That would inflate $\eta_{++}$ and $\eta_{--}$, but it cannot produce the cross terms ($\eta_{+-}=0.073$ is a bounce after a down jump). Distinguishing "one event spread over two bars" from "a second event" needs sub-minute data (aggTrades), not 1-minute klines.
2. **TSK may partly be sub-threshold excitation rather than pre-jump asymmetry.** TSK truncates at |Z| < 4, so a 3.5σ move just below the threshold enters TSK as positive skew. A following jump would then be continuation of a near-jump, not slow asymmetry building in small returns. A cheap check: TSK truncated at |Z| < 2.5, excluding the 5 bars after any |Z| > 3. This was not preregistered and does not change the verdict; it changes the interpretation.
3. **Level drift.** The TEST jump rate is about 30% higher than DEV for every asset (BTC 5.9 → 7.6 per day). Both (b) and (e) under-predict the 15-minute occurrence level (observed 0.034 vs 0.027–0.029). The gates are about ranking and pass despite this, but any pricing use (SJ 02) must recalibrate the level on recent data.
4. **Explanatory power per row is small** (+0.0008 nats), because most minutes have no jump nearby. The structure is real but sparse.

## 4. Economics (opinion; this is what SJ 02 must test)

The decision-relevant quantity is the expected signed jump P&L from trading in the predicted direction:

$$E\approx\sum_{s}\eta_{s,\text{parent}}\cdot \text{sign}(s)\cdot E|J|.$$

| | BTC | ETH | SOL | XRP |
|---|---|---|---|---|
| Following an up jump | 1.4 bp | 2.6 bp | 3.1 bp | 2.9 bp |
| Following a down jump | 1.0 bp | 1.9 bp | 2.1 bp | 2.1 bp |

These are gross, from the jump layer only, against **12 bp** perp taker and 22 bp spot round trips.

- On jumps alone this is **not tradeable** at retail costs.
- What the jump model does not capture is any *diffusive* drift after a jump: the many sub-threshold continuation minutes. That could be larger, and it is directly measurable.
- **Options:** the post-jump window is 1–3 minutes, and Deribit 0DTE spreads are about 27% of mid ATM (PF 004 §4), while implied vol rises after realised moves (RS 04). A single-wing purchase in the minute after a jump would have to recover a spread worth far more than 3 bp of underlying. I expect it to fail.

## 5. Recommendation for SJ 02 (opinion; Ivan decides)

The rule unlocks SJ 02. I propose staging it so the cheapest decisive test runs first:

- **SJ 02a, post-jump return event study, before any options work.**
  - Perp and spot 1m; entry at the close of the bar after the jump (bar t+1).
  - Exit at 1, 3, 5 and 15 minutes; total return in the predicted direction, including diffusive drift.
  - Conditioning on the model's predicted up-share.
  - Gross and net at taker costs, with the same DEV/TEST split.
  - **Gate:** gross TEST lower bound > 12 bp (perp) in the top predicted-share quintile.
  - If gross is below about 5 bp, options are pointless and the branch closes as "real, sparse, untradeable at retail costs".
- **SJ 02b, Deribit single wing vs straddle,** only if 02a shows a gross edge above costs.
- **SJ 01b, interpretation checks (not gates):** TSK at |Z| < 2.5 with post-jump exclusion; half-life grid including a 2-bar minimum. Optionally aggTrades for one month to resolve caveat 1.

## Questions for ChatGPT

1. Do you read the 1-minute half-life as microstructure (order-book refill, stop cascades), or would you expect a bar-boundary artefact to dominate?
2. Is a ~30% rise in the jump rate between DEV and TEST consistent with what you saw in the Volatility Dynamics long-history modules?

## Decisions needed from Ivan

1. Approve the SJ 02a event study as the next step, instead of going straight to options.
2. Whether to run the SJ 01b interpretation checks alongside it. They are cheap and reuse the same pipeline.
