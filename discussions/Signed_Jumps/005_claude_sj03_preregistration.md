# Signed Jumps 005 — SJ 03 preregistration: do "bubbles" (aggressor activity) predict when a jump happens?

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-16 |
| **Branch** | Signed Jumps |
| **Responds to** | Signed Jumps 004 §5; Ivan's boiling-water analogy and approval (2026-10-16) |
| **Status** | Preregistration. Fixed before any real-data output. Code `signed_jumps_flow.py` (SJ03-0.1.0), `signed_jumps.py` SJ01-0.1.1 (adds optional covariates; SJ 01 models unchanged, 16 SJ 01/02 tests pass), tests `tests/test_signed_jumps_flow.py` (6 pass) |

## 1. Question and analogy

**Ivan's analogy:** before water boils, small short-lived bubbles appear, then bigger and more clustered ones, then the boil.

| Water | Market |
|---|---|
| Boil | Jump |
| Bubbles | Aggressor activity |
| Heat | Latent state driving both |

**Physics.** Boiling is mostly a *Cox* process: Poisson events driven by a hidden state (superheat). Self-excitation plays only a small part. Persistence after switching off the heat is the state decaying, not bubbles exciting bubbles.

**What SJ 01 showed:** small, short excitation (ρ = 0.155, half-life 1 min). The strongest timing predictor was $TV_{10}$, i.e. heat read through **returns**.

**Question:** do readings from **order flow** (trade counts, volume, taker imbalance) predict jump timing beyond everything SJ 01 already uses? Direction is out of scope: SJ 004 showed the post-jump path is flat. Any use is volatility and options.

## 2. Data and covariates

- **Data:** the same Binance spot 1m files, assets, jumps (SJ 01 engine, unchanged) and DEV/TEST boundary (2026-02-15). The kline columns `trades`, `quote_volume` and `taker_buy_quote_volume` are used. The notebook asserts that jump counts reproduce SJ 01.
- **Covariates, strictly through bar $t-1$, standardised on DEV:**

$$AN_w=\log\big(\overline{\text{trades}}_{[t-w,t-1]}+1\big)-\log\big(\overline{\text{trades}}_{[t-W,t-1]}+1\big),\quad (w,W)\in\{(10,240),(60,1440)\}$$

- $AV_w$ is the same with quote volume.
- $IMB_{10}=\big|\sum(2\,\text{takerbuy}-\text{quote})\big|/\sum\text{quote}$ over the last 10 bars, i.e. unsigned aggressor imbalance.

## 3. Models and gates

**Models:** all on the same rows, with SJ 01's DEV kernel half-life fixed at 1 minute.

| Model | Content |
|---|---|
| (b) | SJ 01 engine-modulated Poisson |
| (bf) | (b) + flow |
| (e) | SJ 01 best model: signed Hawkes + TSK |
| (f) | (e) + flow |
| (ev) | (e) + $\hat v$ |

$\hat v$ is the flow part of a DEV OLS of the next 10 minutes' continuous-variance surprise, $\log\overline{Z^2}_{[t,t+9]}$ over $|Z|<c$, on the (b) covariates plus flow.

**Gates on TEST** (2,000 day-block bootstraps, at least 200 TEST jumps per sign):

| Gate | Comparison | PASS iff |
|---|---|---|
| **F1** flow adds to SJ 01 | (f) vs (e), logLik/row | Δ > 0 and q2.5 > 0 |
| **F2** 15-min forecast | signed occurrence log-loss, (f) vs (e); 5 and 60 min secondary | Δ > 0 and q2.5 > 0 |
| **F3** jump-specific heat | (f) vs (ev) | Δ > 0 and q2.5 > 0 |

**Why F3 exists (found on synthetic data before any real run).** Jumps are defined relative to the volatility forecast. Flow that merely nowcasts volatility more precisely therefore predicts more threshold crossings, and passed F1 in a synthetic world where flow carried **no** jump information beyond volatility.

- **F1 PASS / F3 FAIL** means flow is a sharper thermometer for ordinary volatility.
- **F3 PASS** means flow carries information specific to the boil.

**Reported, not gated:**
- (bf) vs (b);
- flow coefficients with day-clustered SEs;
- the forward-variance R² gain from flow (DEV);
- the hazard-decile lift table at 15 minutes (realised P(any jump) by predicted decile, and the top 1%, for (b), (e) and (f));
- the event-time profile of log trades and volume relative to a causal 1-day mean, from −30 to +10 minutes around cluster-starting jumps, against hour-matched placebo bars.

## 4. Synthetic validation (`tests/test_signed_jumps_flow.py`)

| World | F1 | F3 |
|---|---|---|
| Flow is noise | FAIL | FAIL |
| Flow tracks volatility only (forward-variance R² rises) | may PASS (it did on one of two seeds) | FAIL on both seeds |
| Planted heat: a latent state raises both trade intensity and jump rates | PASS | PASS |

In the planted-heat world, trade activity is elevated in the 10 minutes before jumps. Two further checks pass:
- **Causality:** perturbing flow after bar $k$ leaves covariates at bars ≤ $k+1$ unchanged.
- **Row subsetting:** it keeps excitation aligned.

## 5. Interpretation (stated now)

- **F1 PASS, F3 FAIL:** flow improves the volatility nowcast. That is useful for option pricing, which goes into the VD 19 question, but there is no boil-specific precursor.
- **F3 PASS:** a genuine precursor. The lift table then says how useful it is: what share of jumps fall in the top-hazard 1% of minutes, and at what realised rate.
- **Even an F3 PASS gives timing, not direction.** Monetising it means buying volatility before the boil, which remains the VD 19 options question.

## 6. Expectation (stated in advance)

- **F1:** likely PASS. Activity measures should sharpen the volatility nowcast.
- **F3:** uncertain, leaning FAIL. PFR03's flow features did not predict direction, and timing has never been tested.
- **Lift:** if F3 passes, I expect top-1% minutes to have a jump rate 3–6× the base rate. With a ~3% base rate per 15 minutes, that still leaves most "hot" windows without a jump.

## Questions for ChatGPT

1. Is the forward-variance OLS the right decomposition, or should $\hat v$ come from a HAR on bipower augmented with flow?

## Decisions needed from Ivan

1. Approve and run. Expect about 20–30 minutes: engine rebuild, 10 fits, Monte Carlo forecasts.
