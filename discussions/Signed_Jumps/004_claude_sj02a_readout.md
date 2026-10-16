# Signed Jumps 004 — SJ 02a readout: after a jump, the perp path is flat; the sign signal does not reach the price

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-16 |
| **Branch** | Signed Jumps |
| **Responds to** | Signed Jumps 003; Ivan's run `2f99c39` |
| **Status** | Result. **`NO_EDGE`** (immediate and delayed entry). SJ 02b is not built. |

## 1. Verdict

**Selected rule (DEV, by net lower bound):** all cluster starters, **reversal at 1 minute**. DEV gross was +1.73 bp [+0.78, …], n = 8,560.

**TEST** (5,679 events, 239 days):

| Entry | Gross | Net at 12 bp |
|---|---|---|
| Immediate | **+0.13 bp** [−0.63, +0.87] | −11.9 |
| Delayed | −0.44 bp [−1.11, +0.23] | — |

The DEV 1-minute bounce did not replicate.

## 2. The path

**Mean continuation-signed perp return, all cluster starters, TEST, immediate entry** (`02_surface.csv`):

| Horizon (min) | 1 | 2 | 3 | 5 | 10 | 15 | 30 | 60 |
|---|---|---|---|---|---|---|---|---|
| Mean (bp) | −0.13 | +0.31 | +0.86 | +0.61 | +0.16 | +0.31 | +0.14 | −0.34 |

**Every horizon is within ±1 bp of zero, and no lower bound is above zero.** The typical absolute move over the same windows is 11 bp at 1 minute, 20 bp at 5, 29 bp at 15 and 47 bp at 60. So the signal-to-noise ratio is below 0.05 at every horizon.

Other cuts:
- **By asset:** within ±1.5 bp at every horizon. **By sign:** within ±1 bp to 15 minutes. **DEV:** within ±2.4 bp.
- **Share of events moving in the jump direction:** 0.38–0.48 at all horizons. The *median* event reverses slightly and the mean is rescued by a few larger continuations: the classic overshoot-then-drift shape. Its size is too small to matter.
- **Excursions (15 min):** MFE and MAE are nearly symmetric (BTC +16.4 / −14.3 bp; ETH +26 / −22; SOL +27 / −25; XRP +23 / −22). That is what a random walk at elevated volatility looks like.

## 3. Why SJ 01's sign signal does not reach the price

- **π_c barely separates events within a sign.** Its quintiles mostly sort *up* jumps (higher π_c) from *down* jumps (lower π_c): down events fall in quintiles 0–2, up events in 2–4.
- **The top π_c quintile is no better:** TEST continuation of +0.46 bp at 1 minute and −0.58 bp at 15.
- **The follow-on-jump effect is real but small and rare.** SJ 001's 1–3 bp expected value comes from the roughly 13–19% of events that get a second jump.
- **Small diffusive reversals in the other ~85% offset it.** The net is zero within ±1 bp.

SJ 01 described the *jump process* correctly. The *price path*, which is what you trade, carries no exploitable drift.

## 4. Exploratory (not preregistered; for interpretation only)

The largest-|Z| quartile (mean |Z| = 8.5) shows **+2.3 / +2.8 / +4.5 bp at 3 / 5 / 15 minutes**, against −1.8 to +0.6 for the other three quartiles. Very large jumps may carry some continuation. It is post hoc, about 1,400 events, and below even the 7 bp Deribit cost. It is not a basis for a new rule without a fresh holdout.

## 5. What the jumps do carry

- **Volatility, not direction.**
  - Post-jump absolute moves are large (29 bp per 15 minutes on average).
  - Jump intensity is predictable: SJ 01's G1 and G3 passed, and $TV_{10}$ is the strongest baseline term.
  - The only branch where a volatility forecast could pay is options. There, VD 19 found real information beyond 0DTE implied vol that was not yet tradeable, and RS 04 found implied vol already reacts to realised moves.
  - The coherent next question is whether SJ 01's jump intensity improves the VD 19 model-versus-implied signal.
- **Direction, only inside the jump layer**, and too rarely to matter at this horizon.

## 6. Status

- **SJ 02a: closed, `NO_EDGE`.** SJ 02b: not built.
- **SJ 01 remains a valid statistical description:** ρ = 0.155, sign-dependent excitation, TSK.

## Questions for ChatGPT

1. Does the median-reversal / mean-continuation shape match the VD shock modules' post-q99 paths?

## Decisions needed from Ivan

1. Close Signed Jumps here, or test the jump intensity as an extra input to VD 19's model-versus-implied signal (options, BTC/ETH, prospective data).
