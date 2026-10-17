# Order Book Imbalance 001 — branch closed as covered by other branches

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-17 |
| **Branch** | Order Book Imbalance |
| **Responds to** | OBI 000 (handoff); Ivan's decision (2026-10-17): close as covered |
| **Status** | Closed. No module built. |

The three hypotheses in OBI 000 §2 were already tested elsewhere, on the same collector features, or are untestable on the stored data:

| OBI 000 hypothesis | Where it was tested | Result |
|---|---|---|
| **H1 direction:** OBI 5/10/20, ΔOBI, microprice, trade-flow imbalance → returns at 5–60 min | Pairwise Feature Regression PFR03 (PFR 008): 81 catalog features including all of these, discovery/confirmation split | `NO_DIRECTIONAL_FEATURE`. `book_twa_obi_20_change_5m` (BTC, 60 min): discovery IC +0.020, confirmation +0.011 [−0.005, +0.029] |
| **H2 magnitude, short horizon:** depth, spread, L2 instability → next-5-minute volatility | Volatility Regression 03 (VR 010 §3.3): L2 block B1→B2 | No consistent gain under either target |
| **H2 magnitude, to expiry:** book state → RV to 08:00 above ATM IV | Not tested | About 20–25 sessions available (VD 19's sample); only a very large effect would be detectable |
| **H3 tails:** book fragility → jump intensity | Signed Jumps SJ 03 (SJ 006), with flow rather than book precursors | Precursors carry volatility information only (F3 FAIL) |

**The decision in OBI 000 §7.2 (finer book data in the collector) is not taken.** The collector still stores only 5-minute time-weighted aggregates, so seconds-scale OBI effects remain untested and cannot be backfilled. If the question is ever reopened, starting that collection is the first step.
