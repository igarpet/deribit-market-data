# Liquidity Absorption 028 — W01 amendment: bar-by-bar 1-minute CVD

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-20 |
| **Branch** | Liquidity_Absorption |
| **Responds to** | Ivan (2026-10-20): he reads CVD on 1-minute bars; W01 (LA 027) tested CVD summed over the first 5/15/30/60 minutes and 3-bar bursts, not each 1-minute bar on its own |
| **Status** | Amendment, written **after** the W01 results (run a5c0a68: H1 not supported) and **before** any bar-by-bar output exists. Code `liquidity_absorption_w01.py` (`minute_study`, LA-W01-M-0.1.0); notebook 07 §4; outputs `07/W01M_*` |

## 1. Unit of analysis

**Rows.** Every completed 1-minute bar $t$ of every closed A03b 1:1 trade while the trade is still open, for $t = 0, \dots, 239$ minutes after entry. Population and splits are as in LA 027:
- MINING: signals before 2025-07-01;
- CONFIRM: from 2025-07-01 through 2026-06-30, and from 2026-10-01;
- INSAMPLE: Jul–Sep 2026, reported separately.

**Per bar**, signed toward the target ($s=\pm1$ is the trade direction, $F_{60}$ is the mean traded USD per minute over the hour before entry):

$$d1_t=\frac{s\,(\text{buy}_t-\text{sell}_t)}{F_{60}}, \qquad p1_t=\text{px}_t-\text{px}_{t-1}\ \ (\text{R}), \qquad \text{rem}_t=R_{\text{final}}-\text{px}_t$$

- $d1_t$ is the bar's delta, in minutes of normal volume.
- $p1_t$ is the bar's price change, in R.
- $\text{rem}_t$ is the rest of the trade from the bar close. It is the value of holding versus exiting there, since both choices pay the same costs.

**Bar thresholds (frozen from MINING).** The 5th and 95th percentiles of $d1$ over the MINING pre-entry context minutes $[-240,-61]$.

| Event | Rule |
|---|---|
| STRONG_TOWARD | $d1 \ge q_{95}$ |
| STRONG_AGAINST | $d1 \le q_{05}$ |
| ABSORB_FOR | $d1 \le q_{05}$ and $p1 \ge 0$: the other side hit hard, price did not give way. This is Ivan's absorption read, applied after entry. |
| ABSORB_AGAINST | $d1 \ge q_{95}$ and $p1 \le 0$: our side hit hard, price did not follow. |

## 2. Gates (binding)

**M1 — information.** The partial rank correlation of $d1_t$ with $\text{rem}_t$, controlling for $\text{px}_t$ and $p1_t$, over all open bars.
- **Supported** if the MINING estimate is $>0$ and the CONFIRM day-bootstrap one-sided 95% lower bound is $>0$.
- **Why control for $p1$:** a 1-minute delta and the same bar's price change move together. M1 asks whether the delta adds anything to what the candle already shows.

**M2 — money.** A new 1:1-sized unit is added at the close of each ABSORB_FOR bar, and separately at each STRONG_TOWARD bar. Its result is

$$\text{net}=\text{rem}_t-\text{cost}_R$$

where $\text{cost}_R$ is the trade's own Deribit cost in R.
- **Pass** if the CONFIRM mean net is $>0$ and its one-sided **97.5%** day-bootstrap lower bound is $>0$. That is a Bonferroni correction over the two events.

## 3. Descriptive (not gated)

- **Forward moves:** the gross market move toward the target over the next 1, 5 and 15 minutes after each event. The partial correlation of $d1$ with the forward move, given $p1$, is also reported.
- **Comparison to cost:** about 10 bp round trip on Deribit before slippage.
- **Time since entry:** events by bucket (0–4, 5–14, 15–59, 60+ min).
- **Deciles:** $\text{rem}$, forward moves and $p1$ by decile of $d1$.

## 4. Limits

- **Overlapping bars.** Bars within a trade, and trades within a day, overlap and are dependent. All bounds therefore resample signal days, with 500 bootstrap draws.
- **Bar 0** includes prints from the entry minute before the fill.
- **Threshold choice.** The thresholds are per-bar extremes relative to the trade's own pre-entry volume. They are not Ivan's discretionary eye. If they miss what he looks at, he should describe the bar he acts on (size, relative to what, inside which candle) before any re-run.

## Decisions for Ivan
1. If M1 and M2 both fail, close the CVD line for this setup: entry filter (V2), 15-minute sums (W01 H1), bursts (W01) and single 1-minute bars (this amendment).
