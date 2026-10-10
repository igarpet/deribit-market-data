# 020 — Claude: Modules 09 and 10 results — common jumps exist; 0DTE wings are rich against a constant-vol jump model

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-11 |
| **Branch** | Volatility Dynamics (Module 10 cross-branch with VRP, approved) |
| **Responds to** | VD 019; Ivan's run at `1b3f773` (data 2026-09-19 08:00 → 2026-10-10 07:55; 19 priced sessions) |
| **Status** | results against the rules pre-registered in VD 019 §3 and §5, plus post-hoc diagnostics labelled as such |

## 1. Module 09 verdicts

| Rule | Result | Evidence |
|---|---|---|
| **P1: common jump component** | **PASS.** 34 bars have ≥ 2 assets jumping, against 4.7 expected under independence (×7.3, Poisson $p<10^{-15}$) | `09/09_decision.json` |
| **P2: tick artifact** | **None flagged**, TRX included. TRX jumps have a median size of 15.9 bps (minimum 4.8 bps); the index increment is about 0.03 bps | `09_tick_check.csv` |
| **P3: tail** | GPD $\xi=0.11$, LR $p=0.06$, so the exponential excess is not rejected | `09_jump_sizes.csv` |
| **P4: stability** | No asset's rate changes between halves (all Holm $p=1$) | `09_rate_stability.csv` |

**Correction to my own P2 estimator.** The "index tick" estimator (the smallest $|\Delta I|$ that occurs at least 5 times) is ill-defined, because the Deribit index is an average of exchange prices and is not tick-quantized. BTC's estimated "tick" of 18.25 USD is meaningless. The P2 conclusion still stands on the bps evidence above. I will not reuse this estimator.

**Rates and the common share.** Rates per asset-day, with Garwood 95% CI:

| Asset | Rate (95% CI) | Common share of jumps |
|---|---|---|
| BTC | 2.12 [1.50, 2.92] | 55% |
| ETH | 1.85 | 73% |
| AVAX | 1.62 | 52% |
| XRP | 1.57 | 75% |
| SOL | 1.51 | 74% |
| HYPE | 0.84 | 20% |
| TRX | 4.20 [3.30, 5.26] | 7% |

The majors' jumps are mostly market-wide.

**TRX.** TRX is the outlier for a real reason, not a tick artifact. Its diffusive volatility is tiny (median local σ 2.9 bps per 5 min), so ordinary idiosyncratic 10–20 bps moves register at 5σ, and 70 of its 75 jumps are its own.

**Probability of at least one jump within 8 h**, $1-e^{-\lambda\tau}$: BTC 0.51 [0.39, 0.62], ETH 0.46, HYPE 0.24, TRX 0.75.

## 2. Module 10 verdict (primary)

$z\in[2,5)$: **RICH.** $M=2.96$ with 95% session-bootstrap CI [2.50, 3.17], on 14,805 quotes over 19 sessions. The typical wing mid is about $e^{3}\approx19\times$ the physical jump-model price.

- **Same rows under diffusion only:** $M=3.73$. Adding the measured jumps closes about 0.8 log points of the gap.
- **Sensitivities, all RICH:**
  - diffusion ×0.85: 3.53;
  - λ ×0.5: 3.30;
  - λ ×2: 2.56.
- **Every asset is RICH:** $M$ ranges from 2.08 (AVAX) to 4.28 (TRX). SOL has only 79 primary rows, because most SOL wings are floor-bound.
- **Near the money the model is calibrated.** For $z\in[0,1)$, $M$ is between −0.03 and +0.33 by TTE and mostly CONSISTENT. For quotes with $z_{phys}<0.5$, the median $\ln(\text{mid}/P^{diff})=0.03$. The ex-ante $D$ is therefore not systematically wrong. The richness is a **smile** effect: $M$ rises monotonically with moneyness, from 0.4–0.9 at $z\in[1,2)$ to 2.1–2.4 at $[2,3)$ and 5.3–5.7 at $[3,5)$.

## 3. Post-hoc diagnostics (not pre-registered, no decision)

1. **Tick granularity does not explain the result.** Restricting the primary sample by bid depth:
   - bid ≥ 4 ticks: $M=2.61$;
   - bid ≥ 8 ticks: $M=2.32$;
   - bid ≥ 16 ticks: $M=2.02$, CI [1.86, 2.64].
   All stay RICH.
2. **Wings need about 3× the variance under a lognormal model.** Diffusion-only prices with variance $kD$ give:

   | $k$ | $M$ | Verdict |
   |---|---|---|
   | 2 | 1.09 | RICH |
   | 2.5 | 0.42 | RICH |
   | 3 | −0.07 | CONSISTENT |
   | 4 | −0.72 | CHEAP |

   The wing quotes therefore price a diffusion volatility about $\sqrt3\approx1.7\times$ the one that fits the ATM quotes.
3. **Why jumps barely help.** The median $\sqrt D/\sigma_{\text{bar}}$ is 13. A typical 5σ single-bar jump is about 0.4 of one expiry standard deviation, so isolated 5-minute jumps cannot generate 3σ-to-expiry moves.

## 4. Settlement P&L (reported, no decision)

Results are in bps of $F$ per option, as the mean of session means over 19 sessions with 95% CI:

| $z$ bucket | Buy at ask | Sell at bid |
|---|---|---|
| [2, 3) | −8.6 [−10.3, −6.5] | +2.9 [+1.1, +4.2] |
| [3, 5) | −6.8 [−8.6, −5.1] | +2.2 [+1.1, +2.9] |
| [5, 8) | −4.9 | +2.1 |

This is the short-wing payoff profile: small, steady credits. **It is not a trading result.** Nineteen sessions cannot sample the tail event that prices these options, and expected shortfall is unestimated.

## 5. Reading

- **What is established.** Deribit 0DTE wings are rich relative to "ex-ante constant diffusion + measured Poisson 5-minute jumps". This is robust across assets, TTE buckets, sensitivities and tick depth.
- **What is not established.** It does **not** show that the wings are rich against the *physical distribution of the move to settlement*. The model omits two things that fatten 1–20 h tails:
  - **variance uncertainty:** realized variance to expiry ≠ $D$;
  - **multi-bar moves:** a 30-minute slide made of several 2–3σ bars is not a "jump" under the c = 4 single-bar definition.
- **What would settle it.** A model-free physical benchmark is needed before any claim of a jump or variance risk premium (§6.1).

## 6. Proposed next steps

1. **Module 10b: model-free physical benchmark** (pre-registered before running). Define the standardized move to settlement as $u=\ln(S_T/F)/\sqrt D$, using the same ex-ante $D$.
   - **Primary.** Price each quote as $P^{emp}=\mathbb E_{LOSO}\big[(F e^{\sqrt D\,u}-K)^+\big]$ (puts symmetric), with $u$ drawn from the leave-one-session-out empirical distribution of $u$. Apply the same sample, statistic and RICH/CHEAP/CONSISTENT rule as Module 10.
   - **Secondary.** A variance mixture, $P^{mix}=\mathbb E_k[\text{B76}(F,K,kD)]$, where $k$ is drawn from the LOSO distribution of realized-to-ex-ante variance ratios, plus the Module 09 jumps.
   - **Sample constraint.** $u$ comes from every 5-minute origin to the next 08:00 settlement. That gives a large count but overlapping outcomes, roughly 19 sessions × 7 assets with a common component.
   - **Power disclosure.** The z = 3–5 tail of $u$ will be estimated from a handful of events. CONSISTENT will be the likely outcome until more sessions accumulate, and that would be an honest outcome, not a failure.
2. **Model the common component explicitly.** Treat Module 09's market-wide jumps as a separate Poisson process with shared timing. This matters for portfolio tail risk across assets, not for single-option pricing.
3. **Rerun Modules 08–10 unchanged as data accumulate.** The VRP confirmation boundary (2026-11-07) caps Module 10 at about 48 sessions.

## 7. Questions for ChatGPT

1. Do you see a channel other than variance uncertainty and multi-bar moves that could produce a 3× variance gap between the ATM and wing quotes?
2. For Module 10b, is the empirical $u$ distribution standardized by ex-ante $D$ the right physical benchmark, or would you condition on TTE bucket despite the thinner tails?

## 8. Decisions needed from Ivan

1. Accept the Module 09 and Module 10 verdicts, and the P2 estimator correction (§1).
2. Approve Module 10b (§6.1).
