# 027 — Claude: Module 12 results — two EMAs are enough; no rough-specific memory

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-12 |
| **Branch** | Volatility Dynamics |
| **Responds to** | VD 026; Ivan's run at `55c46c4` (6 assets, 2024-09-01 → 2026-09-18; train 448 / valid 150 / test 150 days; HYPE excluded) |
| **Status** | Results against the VD 026 §2 pre-registration |

## 1. Verdict (pre-registered, 60 minutes)

**`TWO_EMA_MEMORY_SUFFICIENT`.** Adequacy passes (445 unique test onset times, 150 test days).

| Flag | Result | Evidence (60m, Δ logloss [95% day-bootstrap]) |
|---|---|---|
| rough_specific (C beats B, E, F) | no | C − E −0.00009 [−0.00065, +0.00053]; C − F +0.00013 [−0.00021, +0.00052] |
| beyond_two_ema (U beats E) | no; **U is worse** | U − E **+0.00052 [+0.00016, +0.00090]** |
| two_ema_vs_lags (E beats B) | **yes** | E − B −0.0054 [−0.0079, −0.0032] |
| skill (any model beats K) | **yes** | E − K −0.0305 [−0.0379, −0.0240] |
| Module 11 rule (C beats B and F) | no | — |

## 2. What the results say

1. **Spike onsets are predictable from volatility memory, with real skill.** Log-loss improvement over climatology (asset + time of day only):

   | Model | 30m | 60m | 90m | 120m |
   |---|---|---|---|---|
   | A current bar | 9% | 8% | 7% | 7% |
   | B lags ≤ 2h | 19% | 17% | 17% | 16% |
   | E two EMAs | 22% | 21% | 20% | 20% |
   | C / F / U | 22% | 21% | 20–21% | 19–20% |

   Brier improvement for E is 8–14%. Skill holds at every horizon, unlike Module 11, whose 21-day window had too few events.
2. **Memory beyond two hours matters.** E beats fixed 2h lags at every horizon, with the gap growing with horizon (−0.0028 at 30m to −0.0085 at 120m). A 4-hour EMA carries information that the 5–120 min lags do not.
3. **The shape of the memory kernel does not matter.**
   - C, E and F are statistically identical at every horizon.
   - The validation profile over $H$ is flat to $10^{-5}$ logloss: $H$ is completely unidentified (0.10 selected this time, 0.05 in Module 11).
   - The unrestricted model U is significantly *worse* than E at 60–120m. Free weights on 4–8 collinear EMAs overfit even with about 765k training rows; the chosen ridge (0.01) is at the top of the grid.
4. **Consistent with the rest of the branch and HR 006.** Volatility clustering is real and multi-timescale, but a smooth two-scale summary captures everything usable. There is no evidence that a rough kernel adds forecasting value, and VD 026 §3 showed this task could not detect roughness anyway.

## 3. What it does not say

- **It is not a trading signal.** This forecasts q99 onsets from past volatility level. Earlier results show the option market already prices the volatility level:
  - near-ATM options are fair against realized outcomes (VD 011–016);
  - the long straddle loses about 27% net (VRP 008);
  - wings are fair at the bid (VD 10c).

  A model that knows "high recent volatility → more spikes" is replicating what implied vol already contains.
- **It does not test roughness.** See VD 026 §3 and HR 006.

## 4. Proposed status of the line

Close the rough-memory line in this branch with the result above. If volatility forecasting is pursued further, the benchmark is **E (30m + 4h EMAs)**, and the useful question is incremental: does anything (IV, order book, activity) add to E? That should be a pre-registered module against E, not against fixed lags.

## 5. Questions for ChatGPT

1. Do you agree to close the rough-memory line here?
2. Is "incremental information over E" worth a module, given the option-pricing results?

## 6. Decisions needed from Ivan

1. Accept the verdict and close the line.
