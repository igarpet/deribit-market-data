# Signed Jumps 006 — SJ 03 readout: the bubbles are real, but they are the volatility thermometer, not a separate boil signal

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-16 |
| **Branch** | Signed Jumps |
| **Responds to** | Signed Jumps 005; Ivan's run `11f8369` |
| **Status** | Result. **F1 PASS, F2 FAIL, F3 FAIL.** This is the "sharper volatility thermometer" outcome stated in SJ 005 §5. |

## 1. Verdicts

TEST: 6,758 jumps, 242 days. Flow data has no missing rows, so the panel is identical to SJ 01's.

| Gate | Δ | 95% interval | Result |
|---|---|---|---|
| F1 (f) vs (e) | +0.000147 /row | [+0.000078, +0.000215] | **PASS** |
| F3 (f) vs (e + flow-implied variance) | −0.000006 /row | [−0.000025, +0.000012] | **FAIL** |
| F2 15-min forecast, (f) vs (e) | −0.00125 | [−0.0022, −0.0003] | **FAIL** (worse) |

F2 at the other horizons: +0.0003 at 5 minutes (n.s.), −0.0137 at 60 minutes (worse).

**Flow nowcasts volatility.** Adding flow raises the DEV R² of the next 10 minutes' continuous-variance surprise from 0.040 to 0.060. Once that volatility information is in the jump model, flow adds nothing: F3's Δ is zero with a tight interval.

## 2. The bubbles exist

**Event-time profile** (`03_event_profile.csv`): log activity relative to a causal 1-day mean, cluster-starting jumps minus hour-matched placebo minutes, TEST, all assets.

| Minute vs jump | −30 | −10 | −5 | −2 | −1 | 0 | +1 | +5 | +10 |
|---|---|---|---|---|---|---|---|---|---|
| Quote-volume excess (log) | 0.06 | 0.12 | 0.24 | 0.34 | **0.56** | 2.40 | 1.58 | 0.73 | 0.54 |
| Trade-count excess (log) | 0.08 | 0.10 | 0.19 | 0.28 | **0.46** | 1.86 | 1.40 | 0.71 | 0.53 |

- **The build-up is visible but late.** Activity is roughly flat to −10 minutes, rises over the last 5, and is about +75% in volume one minute before the jump. It is about 10× at the jump minute itself, and the decay is the "keeps boiling after the heat is off" phase.
- **The fitted coefficients agree.** In (f), AV₁₀ has t = 11.3 (up) and 7.3 (down), AN₁₀ t = 3.7 and 7.6, IMB₁₀ t = 3.7 and 5.4.
- **Activity and volatility rise together**, so the activity carries no information that volatility does not (F3).

## 3. How much timing information exists (15 minutes, TEST; `03_lift_15min.csv`)

Base rate: P(≥1 jump in 15 minutes) = 6.1%.

| Predicted-hazard bucket | (b) engine | (e) SJ 01 | (f) + flow |
|---|---|---|---|
| Deciles 1–7 | 4.7–6.4% | 4.7–6.2% | 4.6–6.9% |
| Decile 10 | 10.2% | 10.8% | 11.3% |
| Top 1% | 16.1% (2.6×) | 17.2% (2.8×) | **20.1% (3.3×)** |

- Timing information sits only in the hottest ~10% of minutes.
- Even in the top 1%, **80% of 15-minute windows have no jump.**
- The models are miscalibrated at the extremes: the top 1% is predicted at 25–37% but realised at 16–20%.

**Why F2 fails.** The horizon forecast freezes covariates at the origin (SJ 001 §5). Flow accelerations are transient (the profile decays within minutes), so freezing a high reading for 15–60 minutes over-predicts. That is a limitation of the preregistered forecast design, not evidence against the one-step result (F1). It still means flow gives no usable 15-minute *forecast* improvement in its current form.

## 4. Reading

- **Ivan's analogy holds descriptively.** Activity builds in the minutes before a jump and decays slowly afterwards.
- **What the bubbles measure is the heat itself:** rising volatility. Price-based volatility (TV₁₀) and flow-based activity are two thermometers on the same pot. Flow is the sharper one, so it adds to F1, but there is no separate boil signal.
- **The resolution is minutes.** The build-up lives in the last ~5 minutes. Seconds-resolution data (aggTrades) is the only way to see it earlier or more sharply; coarser bars would wash it out.

## 5. Branch status

| Study | Result |
|---|---|
| SJ 01 | Jumps cluster, with sign-dependent excitation |
| SJ 02a | Post-jump perp path flat; `NO_EDGE` |
| SJ 03 | Flow precursors exist but are volatility information |

**What survives across the branch:** volatility and jump-hazard information with a horizon of minutes, and no direction.

**Its only route to value** is the options question (VD 19): does a better short-horizon volatility/jump forecast disagree with Deribit implied vol in a way that pays at ask? For 0DTE this is hard, because the information decays within about 5–15 minutes while option spreads are about 27% of mid.

## Questions for ChatGPT

1. Would a state-space ("latent heat") filter beat frozen covariates for the horizon forecast, or is the 15-minute information simply this small?

## Decisions needed from Ivan

Choose one:
1. Close Signed Jumps.
2. Test the jump/volatility nowcast inside VD 19's model-versus-implied framework.
3. Pull aggTrades for 1–2 months to look at the seconds-scale build-up. This is descriptive and costs data, not much code.
