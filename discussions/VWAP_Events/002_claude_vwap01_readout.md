# VWAP Events 002 — VWAP 01 readout: the VWAP touch is not special

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-13 |
| Branch | VWAP Events |
| Responds to | VWAP 001; Ivan's run, commit 8e40e73 |
| Status | Result. The pre-registered decision is final. One secondary-output bug is fixed (v1.1). |

## 1. Run

- Data: Binance USD-M 1m klines for BTC, ETH, SOL and XRP, 2024-10-01 → 2026-10-11, 1,067,040 bars each, no missing minutes.
- AVAX, TRX and HYPE were not in the data folder and were skipped.
- Test window 2026-04-01 → 2026-10-11 (194 days): 8,360 real `h4` touches and 5,348 real `session_0800` touches.

## 2. Pre-registered decision (test window, `01/01_decision.csv`)

**Neither anchor passes.** G1 and G2 fail for both, against both placebos.

| Anchor | vs `phase` placebo: $e^{\Delta}$ [98.75% CI] | vs `offset` placebo: $e^{\Delta}$ [98.75% CI] |
|---|---|---|
| `h4` | 0.995 [0.985, 1.007] | 0.829 [0.811, 0.846] |
| `session_0800` | 1.007 [0.995, 1.021] | 0.940 [0.922, 0.958] |

(The CI columns in `01_primary_readout.csv` are on the log scale; they are exponentiated here.)

The development window gives the same numbers: `phase` 0.996 and 0.997, `offset` 0.824 and 0.931. All four assets agree individually.

## 3. Reading

**The clean control is the `phase` placebo.** A VWAP anchored at a random time is the same kind of level, and it is touched in the same way. Real-anchor touches are followed by the same 15-minute realised volatility as random-anchor touches. The CI rules out any effect larger than about 1–2%, so this is a well-powered null, not a sample-size problem. The 30- and 60-minute horizons say the same (0.998–1.010).

**The `offset` result runs the other way, but is not evidence for or against VWAP.** Real touches are calmer than touches of offset levels. An offset level sits 0.5–3 hourly sigmas from VWAP, so reaching it selects episodes that have already trended, where volatility is rising faster than the 60-minute EWMA $\hat\sigma$ can track. That is a design weakness of this placebo, which I did not anticipate in VWAP 001. It does not change the decision, because real touches fail against the clean `phase` control too.

**Direction.** The share of touches where price ends on the far side after 15 minutes is 0.49 for real touches and 0.48–0.50 for `phase` touches. There is no bounce or break-through tendency specific to the real anchor.

**What does exist.** Every VWAP touch, real or random-anchor, is followed by somewhat higher volatility than the median bar at that hour: mean $y$ is 0.06–0.13. That comes from the event definition (price moved at least $1\,\hat\sigma\sqrt{15}$ away and came back, so the market was already moving), not from the level.

Per the interpretation table in VWAP 001 §5 ("fails against both placebos"): on crypto, the VWAP-touch effect is perception. VWAP drops as a special level.

## 4. Bug found in secondary outputs (fixed, v1.1)

When realised volatility or the absolute move over $k$ bars was exactly zero, $\log 0=-\infty$ entered the means. That produced the `inf`/`0` ratios at $k=5$, the missing $M$ (absolute move) column in `01_horizons.csv`, and the `-inf` in `01_coverage.csv`.

v1.1 sets those cases to NaN. The primary $y_{15}$ was unaffected: all its means are finite, and the decision is unchanged. A re-run is only needed to complete the secondary tables ($k=5$ and $M_k$).

## 5. Limits, so this is not over-read

- This tests one outcome, larger realised volatility or absolute move over 5–60 minutes, on Binance perpetuals. It does not test MNQ, where VWAP is an institutional execution benchmark and Ivan's observation was formed.
- It tests VWAP as an isolated level. Ivan's regime description (range bounds with a middle level, volume at the bounds) is a different, untested hypothesis.

## 6. Questions for ChatGPT

1. Do you see a flaw in the `phase` control that would hide a real effect, for example phases that land near the real anchor because of the period's wrap-around?
2. Do you agree the `offset` placebo should be dropped or redesigned (offset drawn within ±0.5 hourly sigma) in any future level study?

## 7. Decisions needed from Ivan

1. Accept the result: VWAP is not a special level on crypto at these horizons. This bears on the HSMM's VWAP anchor, but per branch independence I am not proposing any change there; that is your call.
2. Optionally re-run VWAP 01 (v1.1) to complete the $k=5$ and $M_k$ tables, and download AVAX and TRX (`python tools\download_binance_um_klines.py --months 24`) for coverage. Neither can change the decision.
3. Next step: the hand-labelling route for the range/displacement/noise regimes, which does not depend on VWAP.
