# Regime Structure 009 — RS 04 and RS 05 pre-registration; correction to RS 006

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-13 |
| Branch | Regime Structure |
| Responds to | RS 008 §7; Ivan (chat): "do all items on your list" |
| Status | Pre-registration, before any real-data RS 04 or RS 05 output |

## 1. Correction to RS 006 §4

RS 006 named execution near the mid as "the only lever left" and said this branch had not tested it. **VRP 03 already did** (VRP 011, imported here as a conclusion with Ivan's general approval of VRP reuse):
- A passive limit near the mid does not recover the spread on the buy side: ΔR −0.15 pp [−0.95, +0.80].
- On the sell side it hurts, because of adverse selection.

The execution study is therefore dropped. Signal straddles must clear their hurdle at taker prices, and RS 02 shows they do not.

## 2. RS 04 — does 0DTE implied vol underreact to yesterday's realised vol?

**Why.** RS 03 confirmed one predictor of big days: $x=\log(\text{RV}_{\text{yesterday}}/\hat\sigma_7)$, known at 08:00. It only helps an option buyer if implied vol does not already move with it.

**Design (`big_move_followups v1.0`, notebook RS 04):**
- Predictor from Binance 1m, exactly RS 03's `pre_rv_prev_vs_fcst`.
- One entry per session-asset: the ATM 0DTE straddle of the primary family at **12:00 UTC** (20 h to expiry), via VRP 02 code.
  - 12:00 is the first hourly entry inside VRP's 120–1200-minute range after the predictor is known.
- Outcome $y=\log(\lvert\log(S_T/F)\rvert/(\text{IV}_{\text{straddle}}\sqrt{T}\sqrt{2/\pi}))$.
- **Decision:** OLS slope $\beta$ of $y$ on $x$, 95% bootstrap over sessions (4,000 draws).
  - CI > 0: `UNDERREACTION`, i.e. after a high-RV day the realised move rises more than implied.
  - CI < 0: `OVERREACTION`.
  - Otherwise: `NO_EVIDENCE`.
- **Reported:**
  - slopes of $\log(\text{realised}/\hat\sigma)$ and $\log(\text{implied}/\hat\sigma)$ on $x$ (their difference is $\beta$);
  - straddle $R_{\text{net}}$ by predictor tercile, with cutoffs from RS 03's development window.
- **Limits:**
  - About 20–23 sessions × 4 assets.
  - The VRP confirmation window (sessions from 2026-11-07) is never loaded.
  - $\beta$ is a regression on noisy single-day outcomes, so its CI will be wide.

## 3. RS 05 — is the big-day profile more than a big net move produces?

**Why.** RS 008 described big days as trend-like (high efficiency ratio, low giveback) and mildly clustered. Part of that is mechanical: selecting days with a large net move selects efficient paths.

**Null.** Every 5-minute return is multiplied by a random sign, one sign per (session, bar) shared by all assets.
- It keeps |r|, so $\hat\sigma$, RV, shocks and volume are unchanged.
- It destroys direction.
- BIG uses the frozen RS 03 thresholds in the real data and in each draw.

**Statistics** (test window, pooled; 399 draws; two-sided permutation $p\le0.05/5=0.01$):

| | Statistic | Real above null means |
|---|---|---|
| S1 | mean $(\text{net}/\text{RV})^2$ over all sessions (daily variance ratio) | intraday trending; below = mean-reverting |
| S2 | efficiency ratio on BIG days | big days trend beyond chance |
| S3 | giveback on BIG days | more giveback |
| S4 | top-hour share on BIG days | more concentrated |
| S5 | P(BIG tomorrow \| BIG) / P(BIG) | clustering beyond volatility |

S1 is also the cleanest available test of Ivan's mean-reversion vs displacement picture at the daily scale. Below 1 relative to the null means 5-minute moves reverse within the day; above 1 means they continue.

## 4. Synthetic validation

`tests/test_big_move_followups.py`, 4 tests:
- the slope bootstrap recovers +0.8, 0 and −0.8 with the right verdicts;
- RS 04 outcome units;
- null signs are shared across assets;
- a pure random walk passes no RS 05 statistic, while planted AR(1) trending (φ = 0.3) makes S1 significant with ratio > 1.3.

Both notebooks smoke-run end to end on synthetic data.

## 5. Decision needed from Ivan

Run RS 04 and RS 05 after the data resync and database rebuild. Both runs are binding.
