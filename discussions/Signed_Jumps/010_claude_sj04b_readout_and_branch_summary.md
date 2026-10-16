# Signed Jumps 010 — SJ 04b readout: no continuation after a real-time trigger, at any latency; branch summary

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-16 |
| **Branch** | Signed Jumps |
| **Responds to** | Signed Jumps 009; Ivan's run `6733112` |
| **Status** | Result. **`NO_CONTINUATION`.** Proposed branch closure (§3). |

## 1. Verdict

**Primary** ($c=4$, latency 2 s, hold 60 s; 4,913 triggers, Jul–Sep 2026): gross **+0.25 bp [−0.46, +1.03]**.

**Typical trigger** (`05_triggers_c4.csv.gz`): $|z|$ median 4.2, trailing-60 s move median 23 bp (10–46 bp, 10th–90th percentile).

**The whole grid is zero** (`05_grid.csv`):
- every cell at every threshold $c\in\{3,4,5\}$, latency δ ∈ {1, 2, 5, 10, 30} s and hold H ∈ {10 s … 5 min} lies between −0.7 and +0.5 bp;
- no lower bound at $c=4$ is above zero;
- the share of triggers that continue is **0.41–0.45 at every cell**: the median trigger reverts slightly.

**Splits:** every asset (BTC +0.25, ETH +0.59, SOL −0.44, XRP +0.57) and both directions are within ±0.6 bp, with intervals spanning zero.

## 2. What this settles

- **SJ 04 P3 (+8.2 bp) was entirely look-ahead.** A trigger that only sees the past fires once the burst has already moved, and from that second on the expected path is flat, **even at 1 s latency**.
- **The withdrawal in SJ 008 §4 was correct.** The size of the artefact (+8–10 bp) is a useful warning for any future event definition that uses the event's own completion.
- **Combined with SJ 02a**, which was flat at 1–60 minutes from the jump-minute close: price after a detected burst is a martingale at every horizon from 1 second to 1 hour, at every threshold tested, for BTC, ETH, SOL and XRP.

## 3. Branch summary (opinion; Ivan decides)

| Study | Result | Status |
|---|---|---|
| SJ 01 | Jumps self-excite (ρ = 0.155, half-life 1 min); sign-dependent excitation; TSK predicts the next jump's sign | Valid statistical description |
| SJ 02a | Perp path after a jump flat at 1–60 min | `NO_EDGE` |
| SJ 03 | Pre-jump activity is volatility information, not a separate signal; top-1% minutes give a 20% jump chance in 15 min | F1 PASS / F3 FAIL |
| SJ 04 | Build-up visible ~2 min before onset (×1.4 → ×11); excitation mostly genuine (15% split); P3 withdrawn | P1, P2 valid |
| SJ 04b | Real-time trigger: zero continuation at 1–30 s latency | `NO_CONTINUATION` |

**What survives:** jumps and their precursors carry **volatility and jump-hazard information** over minutes, and **no directional information** in the price path at any horizon tested.

**The only remaining route to value** is volatility-priced instruments: VD 19's model-versus-implied question. The information decays within about 5–15 minutes, while 0DTE option spreads are about 27% of mid. I do not expect that route to be tradeable, but it is the only one this branch leaves open.

**Proposal:** close Signed Jumps for direction and timing trading. Keep SJ 01's jump-intensity model as a candidate input if VD 19 is revisited prospectively.

## Decisions needed from Ivan

1. Close the branch, or carry the jump-intensity forecast into VD 19.
