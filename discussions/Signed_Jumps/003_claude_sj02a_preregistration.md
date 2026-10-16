# Signed Jumps 003 — SJ 02a preregistration: the price path after a jump, on the tradeable perp

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-16 |
| **Branch** | Signed Jumps |
| **Responds to** | Signed Jumps 002 §5; Ivan's approval of 02a (2026-10-16) |
| **Status** | Preregistration. Fixed before any real-data output. Code `signed_jumps_events.py` (SJ02a-0.1.0), tests `tests/test_signed_jumps_events.py` (5 pass) |

## 1. Question

**Ivan's question:** once a jump happens, how far and for how long does price keep going in that direction, and is that worth trading?

SJ 01 measured only follow-on *jumps*: 1–3 bp gross. This study measures the full path, including ordinary returns, on the instrument we would trade.

## 2. Events and signal (nothing refitted)

- **Jumps:** the SJ 01 engine unchanged (Binance spot 1m, c = 4, 5 bp floor; BTC, ETH, SOL, XRP; same DEV/TEST boundary, 2026-02-15). The notebook asserts that jump counts reproduce `01_data_quality.csv`.
- **Primary events: cluster starters.** A jump at bar $t$ with no jump of either sign on that asset in bars $t-5\ldots t-1$. This is the trigger a live system would act on. All jumps are reported as secondary.
- **Signal:** SJ 01 model (e) with the DEV parameters read from `01/01_fits.csv`. At bar $t+1$ (covariates through $t$, excitation including the jump at $t$), the predicted continuation share is

$$\pi_c=\frac{\lambda^{\text{same}}_{t+1}}{\lambda^{\text{same}}_{t+1}+\lambda^{\text{opp}}_{t+1}}.$$

## 3. Path

**Instrument:** Binance USD-M perpetual 1m closes (`<BINANCE_UM_DIR>/<A>USDT_um_1m.parquet`), aligned by bar-end time. $T$ is the close of the jump bar.

$$G(k)=s\cdot\left(\log P^{\text{perp}}_{T_e+k}-\log P^{\text{perp}}_{T_e}\right)\cdot10^4\ \text{bp},\qquad k\in\{1,2,3,5,10,15,30,60\}\ \text{min}$$

**Entry timing:**
- **Primary: immediate entry** ($T_e=T$, the perp close of the jump minute). This is optimistic: it assumes a fill at the bar close in the second the jump bar closes.
- **Secondary: delayed entry** ($T_e=T+1$ min), reported with the same selected rule.

**Excursions:** maximum favourable and adverse signed move over 15 minutes (MFE / MAE).

## 4. Rule selection (DEV) and verdict (TEST)

**Grid (48 cells):** horizon $k$ × direction (continuation $+s$, reversal $-s$) × filter (all, top $\pi_c$ quintile, bottom $\pi_c$ quintile). Quintile cut points come from DEV cluster starters.

**Selection:** the cell with the highest **DEV 95% day-bootstrap lower bound of net bp**, at the primary cost, with n ≥ 30.
- On synthetic data, selecting on the mean picked a 60-minute top-quintile cell with only 81 DEV events, which then had 26 TEST events.
- The lower bound picks the robust cell. This change was made before any real data.

**Costs (round trip, perp):**

| Role | Cost | Basis |
|---|---|---|
| Primary | 12 bp | project standard: Binance taker 5 bp + 1 bp slippage per side |
| Reported | 7 bp | Deribit-perp base taker (unverified, SJ chat 2026-10-16) |
| Reported | 14 bp | jump-time slippage stress |

**Verdict on TEST (2026-02-15 → 2026-10-14), selected rule, immediate entry; 2,000 day-block bootstrap; at least 200 TEST events:**

| Verdict | Condition |
|---|---|
| `NET_EDGE` | net (12 bp) 95% lower bound > 0 |
| `GROSS_EDGE_ONLY` | gross lower bound > 0, net not |
| `NO_EDGE` | otherwise |

**Reported, not gated:**
- the full DEV and TEST surface (both entries);
- CAR curves by parent sign × $\pi_c$ quintile;
- MFE / MAE;
- the selected rule's TEST net at 7, 12 and 14 bp;
- all-jump events.

**SJ 02b (Deribit options)** is built only on `NET_EDGE`, or on `GROSS_EDGE_ONLY` with gross ≥ 7 bp *and* Ivan's approval.

## 5. Synthetic validation

1. **No drift, no excitation:** the TEST path is flat at 5, 15 and 60 minutes, and the verdict is never `NET_EDGE`.
2. **Planted drift:** +3 bp/min for 10 minutes after each jump is recovered at 25–35 bp at 10 minutes and stops afterwards. The selected rule is continuation, with a positive gross lower bound.
3. **Timing:** horizon returns start at the jump bar's close, so the jump itself is never counted. Delayed entry starts one bar later.
4. **No refit:** intensities rebuilt from the saved fit table equal the fitted ones exactly. $\pi_c>0.6$ under diagonal excitation.

## 6. Known limitations

1. **Immediate entry is optimistic.** Bar-close fills during a jump are unlikely; delayed entry is the realistic bound.
2. **Spot detection, perp execution.** Basis is 1–6 bp s.d. (PF 004), small relative to jump moves.
3. **Overlap.** Events within 60 minutes of each other overlap. The day-block bootstrap handles the dependence; the means are unaffected.
4. **Level drift.** The TEST jump rate is about 30% above DEV (SJ 002 §3). $\pi_c$ is a *share*, so the level drift largely cancels.

## 7. Expectation (stated in advance)

- Mean continuation of 2–6 bp over 1–5 minutes, then flat or slightly reverting by 30–60 minutes.
- Larger in the top $\pi_c$ quintile, but below 12 bp.
- **`GROSS_EDGE_ONLY` or `NO_EDGE`.**
- Delayed entry roughly halves whatever immediate entry shows.

## Questions for ChatGPT

1. Is cluster-starter gating (5 bars) the right trigger, or should events require a minimum jump size in bp?

## Decisions needed from Ivan

1. Approve and run. Expect about 15–25 minutes: the engine rebuild plus the path computation, with no Hawkes refit.
