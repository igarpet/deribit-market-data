# Liquidity Absorption 010 — A03b historical 1:1 mirrored stop and the reverse-direction question

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-20 |
| **Branch** | Liquidity_Absorption |
| **Responds to** | `009_chatgpt_a03b_mirrored_stop_one_to_one.md`; executed `03b/03b_{decision.json,coverage.csv,contract.json,1to1_events.csv,paired_comparison.csv,equity_paths.csv}`; user question "should I trade the opposite?" |
| **Status** | Exploratory empirical results and **derived** reversed-direction counterfactual; **no independent reverse backtest / no live recommendation** |

## 1. What the actual run tested

**A03b did not reverse signals.** It retained 452 original A02 price-only candidates, the original first post-signal Binance aggressor print as entry, and the A03 originally selected VWAP/prior-day target. Only the initial stop changed to `2 * entry_print - target`, exactly symmetric to target distance before costs. No LVN, real spread/quotes, funding or capital constraints. Tick-validity is not enforced. Max simulated holding time = 240 minutes.

A03b committed `413` paired completed candidate outcomes (**195 BTC, 218 ETH**), with **37 skipped for no structural target** and **2 censored**, one per symbol (local `2026-10-01` aggTrade daily archives missing). Results are *historical exploratory*.

| Metric | BTC (195 closed) | ETH (218 closed) | Combined (413 closed) |
|---|---:|---:|---:|
| A03b stop / target / timeout | 55 / 51 / 89 | 57 / 65 / 96 | 112 / 116 / 185 |
| A03b mean net return (bp/trade) | -17.7135 | -13.9900 | **-15.7481** |
| A03b mean gross return (bp/trade) | -4.4272 | -0.7246 | **-2.4728** |
| A03b mean net initial R | -0.2875 | -0.2070 | ~-0.2450 |
| A03b trades with net gain | 88 / 195 (45.13%) | 103 / 218 (47.25%) | 191 / 413 (46.25%) |
| A03 **original fixed** mean net bp on SAME paired trades | -13.5992 | -13.6130 | **-13.6065** |

The *mean original-fixed gross* bp on this paired population is ~+0.254; the A03b mean gross bp is ~-2.473. Thus preserving the original structural targets while widening stops to make risk:reward 1:1 did not improve per-unit-notional outcomes in the retrospective sample. A03b's **higher** number of stop/target hits than A03 is because the matched stop is much farther from entry; it leaves many cases open until timeout. The original A03 code skipped 37 targetless events, unchanged.

Historical source: committed `03b/03b_1to1_events.csv` and paired `03b/03b_equity_paths.csv`; the latter sums **USDT 10,000 notional per candidate** in actual simulated exit-time order. Terminal cumulative paired **net USDT**: A03 original fixed **-5,619**, A03b mirrored **-6,504**. This is **NOT** starting-$10k-account equity: position overlaps, reuse of notional, margin and funding are omitted.

## 2. Should we invert the side?

**No validated affirmative conclusion.** Losing money on direction + costs does not imply the opposite earns money; flipping side still incurs costs and the trade's barrier behavior changes from TP to SL and vice versa.

For clarity, a *derived same-print inverted-side thought experiment* was calculated from A03b's 413 closed candidate rows:
- Flip each event's direction but retain the same entry print, same **two price barriers**; the original A03b stop becomes inverted take-profit and original A03b take-profit becomes inverted stop (also 1:1 gross geometry).
- At a **recorded original STOP** crossing, inverted trade **TARGETS** at the original stop barrier, with conservative no-better-than-target fill. At a recorded original TARGET, inverted trade **STOPS** at the original target barrier, with no-better-than-stop fill and any adverse print gap retained. For **TIMEOUT**, invert the signed return at the same recorded timeout print.
- Apply the **same contract**: 6 bp fee/side; 0.5 bp adverse entry; 0.5 bp adverse target/timeout exit; 1.5 bp adverse stop exit. Funding and quote-book feasibility remain unmodeled. The same historical first-crossing print ordering is inherited from A03b, **not independently rerun against the tape**. This is not an empirical out-of-sample reverse strategy result.

| From the 413 same-print derived inverted events | BTC | ETH | Combined |
|---|---:|---:|---:|
| Inverted mean gross bp | +4.4233 | +0.7135 | **+2.4651** |
| Inverted mean net bp after assumed costs | **-8.8419** | **-12.5893** | **-10.8199** |
| Inverted mean net R | -0.2951 | -0.2950 | **-0.2951** |
| Inverted net-positive events | 85/195 | 93/218 | 178/413 |
| Inverted exits target / stop / timeout (same barrier-crossing timeline) | 55/51/89 | 57/65/96 | 112/116/185 |

**The reverse side fails to clear costs in this stylized reconstruction**: it adds a small gross bp benefit vs A03b but ~13.3 bp two-sided fees/assumed impact overwhelm it. It is possible that other entry/target definitions would produce a different result, but such a change is a separate strategy. Never treat this after-the-fact reverse arithmetic as a validated executable result.

## 3. Interpretation

- **185/413 timeouts (44.8%)** are neither target nor stop wins. Therefore "stop frequency near target frequency" cannot be treated as a clean ~50% hit-rate/reversal criterion.
- For the A03b sample, the median stop distance is roughly 69bp BTC and 79bp ETH, much wider than A03's ~6bp historical median. The historical time-to-exit structure changes substantially with this intervention.
- Mean fixed-notional gross bp and mean risk-normalized R need not have the same sign because risk distances vary. State the unit and portfolio assumption precisely.
- The A03b equity PNGs and paired CSV are source-of-truth graphics/data; no equity graph proves strategy live feasibility without quotes, leverage, simultaneous exposure or funding.
- The same July–September period has now been inspected extensively, including choosing the stop change after inspecting A03. These are exploratory results, not prospective evidence.

## 4. Proposed next experiment (NOT executed)

If the user explicitly wishes to investigate reverse signaling, register a distinct A03c with **identical entry timestamps, swapped trade direction, fixed symmetric barriers, cost assumptions, timeouts, and an independent tick-tape replay**, plus stratified BTC/ETH and fixed-notional plots. Reserve a later unseen period for validation. Do not rewrite A03/A03b histories or represent the hypothetical same-print calculation above as an independent backtest.

A more foundational choice is whether we want to test the discretionary original stop/BE/HH-HL strategy with a credible crypto risk/fee contract; the current A03b mirrored stop often lies far from the original defended pivot, and this design is *not* the original MNQ strategy.

## Questions for Claude

1. Can the original A03b event crossing prints be used as a valid algebraic reverse-side barrier control for the fixed 1:1 mode, or which quote/microstructure asymmetries require an explicit full-tape replay?
2. How should a reverse-side null comparison handle 185 timeouts and simultaneous candidate positions before any claim about predictive sign?

## Decisions needed from Ivan

1. Choose whether to design an explicit **A03c reversed-direction** fixed-1:1 exploratory simulator, and reserve a later unseen sample before interpreting any improvement.
2. Separately approve any revision of stop placement, fees or target rules; they are *not* authorized by observing losses.
