# Signed Jumps 011 — SJ 05 preregistration: does the jump hazard add to VD 19's model-versus-implied signal?

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-16 |
| **Branch** | Signed Jumps (reads Volatility Dynamics VD 19 rows; cross-branch use approved by Ivan, 2026-10-16) |
| **Responds to** | Signed Jumps 010 §3; VD 039 (VD 19 v0.1.1) |
| **Status** | Preregistration. Fixed before any real-data output. Code `signed_jumps_vd19.py` (SJ05-0.1.0), tests `tests/test_signed_jumps_vd19.py` (6 pass) |

## 1. Question

VD 19 found that the volatility engine's HAR variance level predicts realised volatility to the 08:00 expiry beyond the ATM straddle's implied vol, 2–4 hours ahead: β = 1.45 [0.51, 2.52] at 04:00 on Deribit-index RV. The straddle still lost money at the ask.

**Does the SJ 01 jump hazard (expected number of jumps from entry to expiry) carry information that both HAR and the option price miss?**

**VD 19 is not changed.** Its rows (`notebooks/Volatility_Dynamics/19/19_rows.csv`, v0.1.1) are read only, and its prospective re-score keeps its own frozen rule.

## 2. Regressor

For each VD 19 row (session, asset, entry 04:00 / 06:00 / 12:00 UTC, horizon $h$ to 08:00):

- **Origin:** the SJ 01 bar ending at entry + 1 min, whose covariates use bars through the entry. SJ 01 engine unchanged; model (e) parameters read from `01/01_fits.csv`; nothing refitted.
- **Expected jumps to expiry.** Covariates are frozen at the origin, except the clock terms, which follow the true future minutes; the excitation decays:

$$\Lambda=\sum_{s\in\{+,-\}}\sum_{j=0}^{h-1}\Big[e^{X_o b_s-C_o\beta^C_s+C_{o+j}\beta^C_s}+(\eta_s\cdot E_o)\,q^{j}\Big],\qquad x_J=\log\Lambda$$

## 3. Gate (entry 04:00, outcome $y_{rv,index}=\log(RV^{\text{Deribit index}}_{(t,T]}/\sigma_{imp}^2)$)

$$y=a+\beta x+\gamma x_J+\varepsilon,\qquad x=\log(\sigma_{model}/\sigma_{imp}),$$

using VD 19's own whole-session bootstrap (4,000 draws).

| Verdict | Condition |
|---|---|
| `JUMP_HAZARD_ADDS_INFORMATION` | γ lower bound > 0 |
| `JUMP_HAZARD_ANTI` | γ upper bound < 0 |
| `NO_EVIDENCE` | otherwise |

**Reported, not gated:**
- entries 06:00 and 12:00;
- the spot-RV and settlement-move outcomes;
- $x_J$ alone;
- corr($x$, $x_J$);
- R_net of VD 19-selected straddles ($x\ge\log 1.25$) split at the median $x_J$ of the same entry hour.

**Prospective:** the same regression is re-scored on VD 19's prospective sessions (from 2026-10-15) when VD 19 is re-scored, with no changes.

## 4. Synthetic validation

| Check | Result |
|---|---|
| Entry times | reproduce VD 19 sessions (12:00 entries fall on the previous day) |
| Origin | is the bar after the entry |
| Λ for a no-excitation model | equals the exact SJ 01 Poisson forecast |
| Excitation | raises Λ after jumps |
| Planted γ = 0.8 | detected |
| γ = 0 | interval spans 0 |

## 5. Limitations and expectation (stated in advance)

1. **Sample.** About 21–25 sessions × 4 assets ≈ 85 rows at 04:00; effective sample about 25–50. Only a large γ is detectable.
2. **Collinearity.** $x_J$ shares the volatility level with $x$, through $L$ and $TV_{60}$ in the baseline, so γ is identified only from the part of jump hazard that HAR does not carry.
3. **Horizon mismatch.** SJ 01's specific information (excitation, $TV_{10}$, flow) decays within minutes, while VD 19's horizons are 2–20 hours. Over 240 minutes, Λ is dominated by the baseline level and the clock.

**Expectation:** `NO_EVIDENCE`, with $x_J$ mostly a noisier copy of $x$. The minutes-scale information would need option entries minutes before expiry (07:30–07:50), which VD 19 does not sample. That is a possible SJ 05b if Ivan wants it.

## Decisions needed from Ivan

1. Approve and run. It reads VD 19's saved rows and the 1m spot files; about 5–10 minutes.
