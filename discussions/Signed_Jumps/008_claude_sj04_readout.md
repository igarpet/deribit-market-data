# Signed Jumps 008 — SJ 04 readout: the build-up starts ~2 minutes ahead, the excitation is real, and P3 is invalid (look-ahead in my design)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-16 |
| **Branch** | Signed Jumps |
| **Responds to** | Signed Jumps 007; Ivan's run `342ea73` |
| **Status** | Result. **P1 PASS, P2 `MOSTLY_DISTINCT_EVENTS`.** **P3 PASS by rule, but withdrawn:** the preregistered statistic is look-ahead biased (§4). |

**Sample:** 2,186 SJ 01 jump minutes in Jul–Sep 2026 (1,868 cluster starters), BTC, ETH, SOL and XRP, all with an onset found.
- Median onset is second 27–29 of the jump minute.
- The largest single second carries a median 30–33% of the minute's move.

## 1. P1 — the build-up starts about 2 minutes before onset (PASS)

Event minus hour-matched placebo log activity over [onset−120, onset−31] s: **+0.31 [+0.25, +0.37]**. That is 1,868 events against 6,558 placebos.

**Profile, 10 s bins** (`04_profile_10s.csv`):

| Seconds vs onset | −600 | −300 | −120 | −60 | −30 | −20 | −10 | 0 | +10 | +60 | +290 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Excess (log) | 0.10 | 0.19 | 0.22 | 0.34 | 0.49 | 0.69 | 1.17 | 2.43 | 1.82 | 1.16 | 0.51 |

- Activity is about 1.2× normal ten minutes out and about 1.4× two minutes out.
- It then accelerates sharply in the last 20 seconds, reaching about 11× at onset.
- The decay afterwards is slow (≈1.7× after 5 minutes).

This is Ivan's boiling-water picture, now measured. But SJ 006 already showed that this activity information is volatility information, and that it gives only modest unconditional timing lift (top-1% minutes: a 20% jump chance in 15 minutes).

## 2. P2 — SJ 01's excitation is mostly real, not a bar artefact

- **Same-sign consecutive jump minutes:** 104 pairs. Only **15%** are split events (peak seconds ≤ 30 s apart); the median gap between peaks is 52 s.
- **Opposite-sign pairs:** 55, with 40% split. These are mostly spikes and their immediate reversal: a wick.

SJ 01's diagonal η₊₊/η₋₋ is therefore mostly genuine second jumps about a minute later, inflated by at most ~15%.

## 3. Within-jump structure

- **Half the minute's move happens by second ~28.** The largest second holds about a third of it.
- **So a 1-minute "jump" is typically a burst lasting tens of seconds, not a single print.**

## 4. P3 — withdrawn: the preregistered statistic selects on the future

**Preregistered result:** signed spot return from onset + 5 s to onset + 65 s, gross **+8.2 bp [6.7, 9.7]**, 73% positive. That is PASS by rule; net −3.8 bp at 12 bp.

**Why it is invalid.**
- An event is a minute whose close-to-close move reached 4σ, which is known only at the minute's **close** $T$. The onset is defined from the **full** minute's path (the halfway second).
- Any window starting before $T$ is conditioned on the rest of the minute having continued. Minutes that stalled after their halfway point did not qualify as jumps.
- With median onset at second ~28, a window from onset + 5 s covers about 27 seconds of selected path.
- This is my design error. The synthetic tests planted drift *after* jumps and could not detect selection on the jump minute itself.

**The grid shows the bias directly** (`04_post_onset_grid.csv`, gross bp):

| Entry delay after onset | +10 s hold | +60 s hold | +300 s hold |
|---|---|---|---|
| 1 s | 5.5 | 10.5 | 11.3 |
| 5 s | 4.3 | 8.2 | 8.9 |
| 10 s | 2.6 | 5.3 | 6.4 |
| 30 s (≈ minute close) | 1.0 | 1.1 [0.1, …] | 2.5 [−0.0, …] |

- Returns shrink as entry moves toward the close, and **nearly all the "continuation" sits inside the jump minute.**
- From the close onwards, it is +1 to +2.5 bp with lower bounds at or near zero. That is the same flat path as SJ 02a.

**What would be valid:** a real-time trigger. At each second, compute the trailing-60-second signed return in units of the engine's $\hat\sigma$ (scaled to 60 s). Trigger the first time it crosses a threshold, without knowing how the minute ends, then measure returns from trigger + δ. That is a seconds-scale momentum test, and it is the honest version of what P3 tried to measure.

## 5. Status

- **P1, P2:** valid results (descriptive; TEST-era sample).
- **P3:** withdrawn. Proposed replacement: **SJ 04b, real-time trigger** (above). Expectation, stated now: positive gross over the next 10–30 s (bursts last tens of seconds), shrinking quickly with latency, and below 12 bp at any latency a retail setup can achieve.

## Questions for ChatGPT

1. Is there any other selection-on-future path in SJ 01–03? SJ 01 and SJ 03 condition only on information through $t-1$ and score bar $t$; SJ 02a enters at the jump bar's close. I believe they are clean.

## Decisions needed from Ivan

1. Build SJ 04b (real-time seconds trigger), or close the branch here.
