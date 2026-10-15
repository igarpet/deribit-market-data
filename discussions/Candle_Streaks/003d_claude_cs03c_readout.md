# Candle Streaks 003d — CS03c readout: the flow signal is real on the perp, and about 20× too small to trade

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-15 |
| Branch | Candle Streaks |
| Responds to | CS 003c pre-registration; Ivan's run d3a9fb7 |
| Status | Result. The pre-registered verdict is binding. |

## 1. Verdict: `PERP_STATISTICAL_ONLY`

- 735 common complete days (2024-10-03 → 2026-10-10); the TEST window is 147 days.
- **Statistical gate (perp target):** passes for BTC and ETH on both metrics.
- **Economic gate (taker, 6 bp per side):** fails for both, by a wide margin.

## 2. Lead-lag: my concern was wrong

TEST Δ log-loss, `price_vol_signed` − `price_vol` (upper 95% bound in brackets):

| Asset | Binary, perp | Binary, spot | 3-class, perp | 3-class, spot |
|---|---|---|---|---|
| BTC | −0.00037 [−0.00013] | −0.00027 [−0.00007] | −0.00037 [−0.00011] | −0.00029 [−0.00006] |
| ETH | −0.00066 [−0.00039] | −0.00068 [−0.00040] | −0.00072 [−0.00039] | −0.00059 [−0.00028] |
| SOL | −0.00018 [0.00000] | −0.00018 [+0.00001] | −0.00012 [+0.00009] | −0.00026 [−0.00002] |
| XRP | −0.00021 [−0.00001] | −0.00024 [−0.00004] | +0.00002 [+0.00027] | −0.00027 [−0.00004] |

- The perp-target gain matches or exceeds the spot-target gain.
- CS03b was **not** spot catching up to the perp. Signed flow predicts the next 5 minutes in the futures market itself.
- My pre-registered expectation of a smaller perp effect was wrong.

## 3. Economics (TEST, `price_vol_signed`, top-decile rule)

| | BTC | ETH | SOL | XRP |
|---|---|---|---|---|
| Hit rate | 54% | 54% | 51% | 51% |
| Gross bp per active bar | 0.53 | 0.25 | 0.38 | 0.16 |
| Gross bp/day, zero cost [95% CI] | 17 [5, 29] | 8 [−7, 22] | 12 [−4, 28] | 5 [−12, 19] |
| Net bp/day at 3 bp per side | −96 | −99 | −100 | −94 |
| Net bp/day at 6 bp per side | −208 | −206 | −212 | −192 |

- **Gross edge vs cost:** about 0.5 bp per trade against a 12 bp round trip, i.e. roughly 20× short. This matches the $2e\,\mathbb{E}|r|$ estimate in 003c §1.
- **Trade count:** the top-decile rule trades about 33 bars per day, almost all of them single 5-minute holds, so cost scales with trade count.
- **Flow vs price alone:** BTC zero-cost gross rises from 5 bp/day (`price_vol`) to 17 bp/day with flow; that is the only gross-edge CI clearly above 0. ETH's is not significant, and SOL and XRP gain nothing from flow economically.
- **A zero-cost world** would mean passive maker fills with no adverse selection. That is not credible for a signal whose whole content is aggressive flow, so the zero-cost line is an upper bound, not a scenario.

## 4. Conclusion

- **Signed taker flow:** it carries real, perp-native information about the next 5 minutes.
- **Size:** about 0.5 bp per trade, which retail execution at 0.02–0.05% cannot capture, and no threshold or holding period in this design changes that order of magnitude.
- **Why holding longer won't fix it:** the forecast is for a single 5-minute bar, and longer horizons were already flat in RS 01 and BT 01.
- **Status of the line:** this closes the directional line at retail cost for 5-minute bars. CS03b/c remain valid as a description of microstructure, not as a strategy.

## 5. Decision needed from Ivan

Accept. Any further directional work would need an execution cost below about 1 bp per side (exchange-maker rebates plus queue position), which is a different project.
