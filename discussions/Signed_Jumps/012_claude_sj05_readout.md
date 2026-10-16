# Signed Jumps 012 — SJ 05 readout: the jump hazard adds nothing to VD 19 at 2–20 hours

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-16 |
| **Branch** | Signed Jumps |
| **Responds to** | Signed Jumps 011; Ivan's run `f3cee6f` |
| **Status** | Result. **`NO_EVIDENCE`** (as expected in SJ 011 §5). |

## 1. Verdict

**Primary** (04:00, Deribit-index RV, 77 rows, 21 sessions):

| Coefficient | Estimate | 95% interval |
|---|---|---|
| γ (jump hazard) | +0.10 | [−0.13, +0.63] |
| β (VD 19 model) | 1.50 | [0.49, 2.45] |

β is unchanged from VD 19's 1.45 when $x_J$ is added.

| Entry | γ ($y_{rv,index}$) | γ ($x_J$ alone) |
|---|---|---|
| 04:00 (240 min) | +0.10 [−0.13, 0.63] | +0.05 [−0.25, 0.65] |
| 06:00 (120 min) | −0.08 [−0.38, 0.28] | −0.06 [−0.35, 0.25] |
| 12:00 (1200 min) | −0.12 [−0.35, 0.09] | −0.12 [−0.33, 0.08] |

Spot RV and settlement-move outcomes give the same picture. Every interval spans zero.

## 2. Why

- **Excitation contributes nothing at these horizons.** Λ and Λ_baseline are identical to three decimals in every summary statistic: with a 1-minute half-life, the effect of any jump before the entry is gone long before expiry. $x_J$ is therefore the clock and level part of the SJ 01 baseline.
- **That part is not collinear with VD 19's $x$** (corr −0.16, 0.01, −0.03), but it does not predict realised over implied. The option market and HAR already carry the level information that matters at 2–20 hours.
- **The economics split is noise.** At 04:00, VD 19-selected straddles with high $x_J$ returned −2% [−54, +57] (9 rows) against −41% [−77, −3] (10 rows) for low $x_J$; at 06:00 the order reverses. At these sample sizes it is not evidence, and it was not gated.

## 3. Status

- **SJ 05 historical: `NO_EVIDENCE`.** The regression will be re-scored, unchanged, on VD 19's prospective sessions when VD 19 is re-scored (about mid-November).
- **Signed Jumps branch: closed for trading purposes.** Jumps carry minutes-scale volatility and hazard information and no direction (SJ 010). At the horizons where options are sampled (2–20 h), that information has decayed, and it adds nothing beyond HAR and implied vol.
- **The one untested route** is option entries minutes before expiry (07:30–07:50 UTC), where the SJ information is still alive. Two cautions:
  - 0DTE options in the last 30 minutes have tiny premiums, so relative spreads and Deribit's minimum tick dominate;
  - the collector would need to have stored chains at those times.

  I would not prioritise it.

## Decisions needed from Ivan

1. Confirm branch closure (SJ 05's prospective re-score stays scheduled with VD 19).
