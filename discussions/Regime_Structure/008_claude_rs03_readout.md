# Regime Structure 008 — RS 03 readout: big days are market-wide, US-session, volatility-persistence events

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-13 |
| Branch | Regime Structure |
| Responds to | RS 007; Ivan's run, commit 5a93dbf |
| Status | Result. The confirmation rule is binding. |

## 1. Sample

- BTC, ETH, SOL and XRP, 539 development sessions and 194 test sessions each.
- BIG = session |move| ≥ about 1.32–1.40 × the 7-session vol forecast.
- That gives 81 big days per asset in development and 25–30 in test (13–16%).

## 2. What precedes big days (binding)

**One feature is confirmed: `pre_rv_prev_vs_fcst`**, yesterday's realised vol relative to the 7-session forecast.
- AUC: development 0.557 [0.502, 0.608]; test 0.652 [0.533, 0.756].
- It holds for every asset in test (0.61–0.68).
- Reading: yesterday's vol above the weekly average predicts a move today above the weekly average. This is volatility persistence that the 7-session RMS forecast underweights (a short-memory version of the issue flagged in RS 007 §5.1). It says a forecast weighting yesterday more would be better. It is not a regime.

**Not confirmed:**
- `pre_rv7_vs_rv30`, last-4h return and the weekend flag were development candidates that failed in test.
- No EARLY feature was even a development candidate. The first-2h vol and volume reach AUC 0.59–0.61 in test only.
- My §5.2 expectation (intraday clustering) did not replicate.
- Yesterday's shocks (test 0.64) and last-4h volume (test 0.64) are strong in test but were not development candidates, so by the rule they are not findings. They are worth re-checking on later data.

## 3. What happens on big days (descriptive)

| | Big | Normal |
|---|---|---|
| Efficiency ratio (net / path) | 0.13 | 0.04 |
| Volume vs 30-session median | 1.29–1.44 | 0.91–0.93 |
| ≥3σ 5m bars | 6–9 | 2–3 |
| Taker flow aligned with the move | +0.03 | +0.01 |
| Share of the move in the top hour | 0.16–0.19 | 0.14–0.15 |
| Giveback from the max excursion (σ units) | 0.29–0.35 | 0.36–0.39 |

- **They are trend days, not one-burst days.** The move is spread over the session (the top hour holds less than a fifth of it) and little is given back. The high efficiency ratio is partly mechanical, because a large net move relative to σ raises it.
- **They are US-session days.** |Hourly return| is elevated all day (×1.3–1.7 versus normal days at the same hour) and peaks at 14:00–15:00 UTC, the US open. The largest hour is most often 14 or 15 UTC in development, and 22, 13 or 15 UTC in test.
- **Volume builds through the day.** Relative to normal it is ×1.1 in the European morning and ×1.5–1.75 from 19:00 UTC to the 08:00 close. The move draws participation as it goes.

## 4. Clustering (the regime question)

| | Development | Test |
|---|---|---|
| P(big tomorrow \| big today) / P(big) | **1.37 [1.02, 1.74]** | 1.48 [0.78, 2.20] |
| Sessions with all 4 assets big | **5.9%** (independent: 0.05%) | 5.7% (0.04%) |
| Big days that are big in ≥3 assets | **54%** | **64%** |

- **Big days are market-wide.** More than half are big in at least 3 of the 4 assets, and an all-4 big day is about 120× more frequent than under independence. They are common crypto-market events, not coin-specific.
- **Big days cluster mildly.** A big day raises the chance of another tomorrow by about 40% (significant in development, same point estimate in test with a wider CI). This is the same persistence §2 found.

## 5. Reading for the project

- The only regime-like structure in two years of crypto data at the daily scale is **a market-wide high-activity state**. It is persistent over days, driven by the US session, trends rather than reverses intraday, and is predictable mostly from recent realised vol.
- This is volatility clustering plus a common market factor. It matches HSMM 07/08 and VD 06–12: continuous, smooth, mostly captured by a better volatility forecast.
- **For an options buyer,** the market's implied vol at 08:00 also sees yesterday's vol. The confirmed predictor only helps if implied vol underreacts to it. That can be checked on the collector sessions (about 20 sessions × 4 assets): regress realised/implied straddle move on `pre_rv_prev_vs_fcst`. It will be underpowered.

## 6. Questions for ChatGPT

1. Is a 1.37 next-day lift for big days what a well-specified HAR would leave as residual, or is there something beyond a HAR?
2. Is the "trend day" profile (high efficiency ratio, low giveback) beyond what a random walk conditioned on a large |net| produces? A conditional-simulation check would answer it.

## 7. Decisions needed from Ivan

1. Accept: big days exist as market-wide, persistent, US-session volatility events; no directional regime and no early-warning feature beyond recent vol.
2. Next:
   - (a) the implied-underreaction check on collector sessions;
   - (b) the random-walk check on the trend-day profile;
   - (c) the strategic question in RS 006 / chat: whether this project should stay on crypto 0DTE options.
