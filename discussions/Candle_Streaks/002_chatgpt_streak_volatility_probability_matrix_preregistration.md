# CS02 — Volatility-conditioned signed candle streaks and directional probability matrix

**2026-10-14 · Pre-registered before executing real Binance CS02 data · HISTORICAL_EXPLORATORY.**

## Hypothesis and objective

CS01 found that full-history continuation frequencies generally fall below 50% as same-sign streaks lengthen, and that long streaks are rarer than iid-sign permutations suggest. CS01's sign-length models **did not** show dependable out-of-sample gains over sign-only Markov on BTC/ETH. The new hypothesis asks whether *causally observable volatility conditions* change next-bar probabilities given current streak sign and length. Research target: a probability matrix `P(next bar = down / zero / up | sign_t, reached_length_t, vol_level_t, vol_phase_t)`, where every condition is known at the close of bar `t` before bar `t+1`.

**Do not assume** that adding volatility yields predictive gains or authorization to alter AM-T-HAR-Blend. No fixed arbitrary look-ahead candle window, no maximum streak length, no ex-post volatility at the future bar, and no conditional-on-next-nonzero forecasting gate.

## Data, quality and chronology

- Source: **existing Binance close-to-close 5-minute log returns**, using VD14 `load_returns`, with `end=2026-10-14T00:00Z`, for BTC and ETH primary, SOL and XRP secondary. This source may contain fewer dates than the announced cutoff. Per-asset **complete UTC days** require 288 correctly gridded, finite 5-minute returns spanning 00:00 to 23:55. Incomplete beginning/end and invalid-data days are excluded from model fitting and scoring (but earlier observations can warm up rolling volatility). In particular, the CS01 source's nearly empty final day MUST NOT get equal bootstrap weight as a full day.
- Current streak sign/length is formed continuously from observed `r5` in the same direction, preserving full actual run lengths without capping or resetting at a research-specific window. A zero-return bar breaks streaks. A gap/right-boundary censors rather than reverses. For prediction eligibility current signed streak must exist and the immediately next contiguous five-minute return must be valid; next sign is in `{-1,0,+1}`; zero is a **legitimate scored target**.
- Full UTC-day chronological **60% TRAIN / 20% VALID / 20% TEST** split by each asset's complete observed dates. The next bar must lie on a complete date **in the same split**. TRAIN alone fits all quantile boundaries and conditional probabilities. VALID is a stability audit only; TEST drives the fixed decision. Streak length itself may span split boundaries because it is known at each forecast origin, but no next-bar label crosses a split boundary.
- Prior research has repeatedly inspected these dates; even the labeled TEST is historical-exploratory, not a pristine prospective test.

## Causal volatility information

`RVfast_t = mean_{j=0..11}(r^2_{t-j})` (past **one hour** of observed returns, 12 bars), `RVslow_t = mean_{j=0..287}(r^2_{t-j})` (past **24 hours**, 288 bars), computed on **strictly contiguous finite** sequences. Both include the current completed candle, which is known at the forecast decision time. A gap or invalid return invalidates volatility features until all needed contiguous history has reaccumulated. Do not use RV of bar `t+1`.

- Level feature: `log(RVslow_t)`.
- Expansion feature: `log(RVfast_t/RVslow_t)`.
- Two TRAIN-only empirical tertile boundaries for each feature; frozen thereafter. Level labels low/mid/high; phase labels compression/neutral/expansion (relative to TRAIN quantiles, not physical RV< or > a fixed threshold). Crossing the two creates **nine volatility classes**. This is an explicit low-complexity categorical volatility representation, NOT inferred latent volatility states.

These are *HAR-style realized-variance memory features*, **not the fitted AM-T-HAR-Blend predicted variance**. This isolates the empirical hypothesis without changing/indirectly refitting the existing complex variance model. If successful, a separate future module must test true causally generated AM-T-HAR-Blend forecast features and jointly calibrate the resulting return density at a matched five-minute horizon.

## Registered three-class probability models

Each model predicts all three next-bar outcomes (`down, zero, up`) and is evaluated with proper **multinomial log loss (primary)** and **three-class Brier (secondary)** on the exact same origin grid:

1. `iid_3class`: empirical unconditional TRAIN class distribution, Jeffreys Dirichlet smoothing.
2. `sign_3class`: empirical current-sign conditional, pseudo-count 20 toward unconditional (first-order sign baseline).
3. `streak_3class`: exact **signed reached length** `(sign,k)` with pseudo-count 80 toward `sign_3class` (CS01 length-only analog but with zero outcome).
4. `sign_vol_3class`: `(sign, level, phase)` with pseudo-count 100 toward `sign_3class`. Captures volatility effects without using streak age.
5. `additive_3class`: the arithmetic mean of models 3 and 4. Tests whether the full joint cell adds value over additive combination of separate streak and volatility factors.
6. **`streak_vol_matrix_3class`**: `(sign,k,level,phase)` frequencies with pseudo-count 125 toward that same **fixed 50/50 additive prior**. For unseen exact lengths/cells, fall back to the additive prior, never a fabricated or manually capped class.

Every probability matrix row reports full three-class probabilities, implied continuation/reversal/zero probabilities, TRAIN cell count and current sign, k, volatility labels. Include every actually observed full-history `(sign,k,level,phase)` key even if TEST observes a cell unseen in TRAIN. A separate retrospective raw-count matrix is explicitly **descriptive**, never a deployable predictive model.

## Primary falsification gate

For **each of BTC and ETH**, require primary matrix to beat all of `streak_3class`, `sign_vol_3class`, and `additive_3class` on held-out TEST **multinomial log loss**, and all three paired complete-UTC-day **1000-bootstrap 95% upper bounds** must be below zero. Score Brier and sign/IID comparisons, with full VALID/TEST provenance. SOL/XRP are secondary and cannot rescue a failed BTC/ETH gate.

- `matrix - streak`: does adding volatility improve direction-based streak probabilities?
- `matrix - signvol`: does exact streak length add beyond sign and volatility?
- `matrix - additive`: does the **interaction** outperform simply blending the independent streak and volatility effects?
- `signvol - sign`: does volatility alone add beyond previous sign?
- `streak - sign`: does length alone add beyond previous sign?

Use paired UTC-day bootstrap (complete dates only), report both equal-day and bar-weighted differences and per-day event counts; no post-hoc reweighting or subgroup selection. Descriptive plots and monthly table may aggregate long ages only for visualization, never for model fitting and full output matrix.

## Null checks, stability, caveats

- Diagnostic volatility label randomization (100 reps): within held-out current-sign and coarse length groups, permute the **joint** `(level,phase)` bin pair across observed origins, leaving trained parameters and outcomes frozen. Compare primary matrix-minus-streak TEST logloss with the shuffled-label reference. This tests contemporaneous conditional alignment only, not a self-consistent price/HAR simulation, and is not a trading significance test.
- Monthly descriptive monitoring: signed ages `(1,2,3,4-5,6-8,9+)`, nine volatility bins, with minimum 50 eligible immediate next bars; report raw sample counts. These coarse groups are for **display only**, not modeling.
- Forecasts condition on a *currently nonzero* signed streak. Bars with a current zero return are not scored in this module. The *next* zero outcome is always scored. An eventual live integration should specify how to forecast immediately after a current zero; this module must not suggest its forecasts cover all bars.
- Volatility is clustered and sign streak observations within days are dependent; Wilson/binomial intervals or iid permutation tests alone are inadequate for performance claims. Chronological splits and paired day bootstrap are still exploratory and do not prove economic profit. No options strategy, AM return-density modification, or alpha claim is approved.

## Output and change contract

Add a new modeling module, tests, notebook `notebooks/Candle_Streaks/02_streak_volatility_matrix.ipynb`, discussion protocol, and append notebook branch README. The notebook writes aggregate outputs only to `notebooks/Candle_Streaks/02/` (matrix, bin thresholds, scores, bootstrap, monthly stability, quality, permutation diagnostic, manifest and decision); no tick data, no raw prices, no origin-level predictions, no collector/database edits. Leave CS01, Return State Discovery and AM-T-HAR-Blend unchanged.
