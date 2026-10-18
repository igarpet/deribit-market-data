# Funding Carry 001 — Branch pre-registration: long spot / short perpetual funding harvest

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-18 |
| **Branch** | Funding Carry |
| **Responds to** | MSH 002 (direction closed at 1h–1D); Ivan's decision (2026-10-18) to test funding carry |
| **Status** | Pre-registration. Written and pushed before any real-data run. Code: `src/data_engineering/modeling/funding_carry.py` v1.0.0 |

## 1. Question

Does a delta-neutral position, long spot and short the USD-M perpetual in equal units, earn more than cash after costs, on the PF02 universe of 30 assets?

**Why it could:** the perpetual's funding transfers cash from longs to shorts whenever the perpetual trades rich. Demand for leveraged long exposure has historically kept funding positive on average. The baseline Binance rate is 0.01% per 8h, about 10.95% a year.

This is a **premium**, not a forecast. Its risks are known in advance:
- funding turning negative for weeks;
- basis widening at the wrong time;
- the round-trip cost of entering and exiting;
- margin on the short leg during rallies.

**Not a repeat of PF02.** PF02 tested whether the spot–perpetual *basis mean-reverts* fast enough to trade, at 5-minute frequency. It found a gross edge only. This test holds the position for days to months and is paid by the funding flow.

## 2. Data and venue

- **Inputs (already on disk from PF02):**
  - spot 1m: `Binance\basis\spot\<A>USDT_spot_1m.parquet`;
  - perpetual 1m: `Binance\<A>USDT_um_1m.parquet`;
  - funding prints: `Binance\basis\funding\<A>USDT_funding.parquet`.
- **Venue:** this strategy is **executed on Binance itself** (spot + USD-M). It is not a proxy for Deribit.
- **Universe:** the PF02 list, fixed 2026-10-16 before any data was seen (30 assets; AVAX, TRX and HYPE excluded). Assets with a missing file are skipped and listed.
- **Period:** from the first day each asset has all three inputs plus 7 days of funding (about August 2024), to the end of the shortest input.
- **Split:** DEV is everything before 2026-01-01 00:05 UTC; TEST is from then on.

Disclosure: the thresholds below are not fitted to these data. They do use public general knowledge (the 0.01%/8h base rate; funding was high in the late-2024 bull run).

## 3. Mechanics (per asset, per unit of entry notional, not compounded)

- **Boundaries:** decisions and executions at $b_d$ = 00:05 UTC daily. The price is the close of the 1-minute bar ending at $b_d$, with a 10-minute staleness limit.
- **Position:** units $u = 1/S_\text{entry}$ held constant within an episode (no daily rebalancing).

$$\text{pnl}_d = u\Big[(S_{d+1}-S_d) - (F_{d+1}-F_d) + \sum_{t_i\in(b_d,b_{d+1}]} f_i\,F(t_i)\Big]$$

  - $f_i$ is the funding print at $t_i$: the short receives $f_i F$ when $f_i>0$ and pays when $f_i<0$.
  - $F(t_i)$ is the perpetual close at the print, a proxy for the mark price.
- **Costs:**

| Item | Cost |
|---|---|
| Entry | 17 bp of entry notional (spot 0.10% + 1 bp, perp 0.05% + 1 bp, as PF02) |
| Exit | 17 bp of current notional |
| Sensitivity (reported, not a gate) | 13.5 bp per side (BNB-discounted taker) |

- **Margin-safety reset:** if, at a boundary, $F_d/F_\text{ref}-1 \ge 40\%$ while in position, the trade is rebalanced and charged a full round trip on current notional. At 2× on the short leg, a ~+45% move would otherwise eat the perpetual margin unless spot gains are transferred across.
- **Capital:** 1 (spot) + 0.5 (perpetual margin) = **1.5 per unit of notional**.

## 4. Strategies

- **C (primary):** $f_7$ is the sum of funding prints over the 7 days before $b_d$, times $365/7$. It needs at least 80% of the asset's expected prints.
  - flat → carry when $f_7 \ge 10\%$;
  - carry → flat when $f_7 < 3\%$;
  - forced flat over any unusable interval.
- **U (benchmark):** carry on every valid day.
- **Grid (descriptive only, not a gate):** entry in {0, 5, 10, 20, 30}%, exit in {−5, 0, 3, 10}%.

Not modelled:
- the reverse trade (short spot, long perpetual), since spot borrow costs are not in the data;
- rotation into the highest-funding assets, to limit multiplicity.

## 5. Portfolio

Equal capital slots, one per asset.

- **Return on reserved capital:** summed net P&L ÷ (1.5 × number of assets with a valid day). Idle capital is credited 0.
- **Return on deployed capital:** summed net P&L ÷ (1.5 × active positions).

## 6. Verdict (binding for C, set before results)

| Gate | Condition |
|---|---|
| G1 | 14-day block bootstrap ($B = 2000$) lower 2.5% bound of mean daily return on reserved capital $> 0$ |
| G2 | mean daily return on reserved capital $>0$ in DEV **and** in TEST |
| G3 | net P&L $>0$ on ≥ 60% of assets with ≥ 180 valid days |
| G4 | annualized net return on **deployed** capital ≥ 4% (the cash alternative) |

- All four pass: `CARRY_EDGE`.
- G1–G3 pass but G4 fails: `CARRY_POSITIVE_BELOW_HURDLE`. It works, but cash is better.
- Otherwise: `NO_CARRY_EDGE`.

U and the low-cost runs get the same gates, reported but not binding.

**Risk report (not gates):**
- maximum drawdown and worst day on reserved capital;
- worst episodes;
- number of resets;
- P&L split into funding, basis and costs;
- per-asset funding profile: mean, share of negative prints, share at the 0.01% base, $f_7$ deciles.

## 7. Risks this backtest does not capture (opinion)

- **Counterparty risk.** Both legs sit on one exchange, which is the dominant real risk for this trade.
- **Intraday margin calls.** Prices are marked once a day at the close, so liquidation inside a day is not seen. The 40% reset is only an approximation.
- **Auto-deleveraging** of the short in extreme rallies.
- **Stablecoin depeg** risk.
- **Changes to funding intervals** (4h vs 8h). These are handled through the actual print timestamps, but not anticipated.

## 8. Validation before this turn (`tests/test_funding_carry.py`, 7 tests pass; `selftest()`)

- **Funding accounting:** the interval $(b_d,b_{d+1}]$ contains exactly the 08:00, 16:00 and next 00:00 prints, and $f_7$ is causal.
- **Basis P&L:** a rich perpetual converging to spot pays the short.
- **Negative funding:** charged to the short, as expected.
- **State machine:** entry/exit hysteresis and forced flat on gaps work as specified.
- **Costs and reset:** exact.
- **Selftest:** with constant funding and zero basis, net P&L equals funding minus costs to $10^{-12}$.
- **Synthetic portfolio:** with funding averaging 15% a year, U returns ≈ 10% on reserved capital (= 15%/1.5) and passes. With mean-zero funding it does not.

## 9. My prediction (opinion, before the run)

- **DEV (2024–25):** clearly positive.
- **TEST (2026):** weaker. Funding compresses when trend demand fades, so G2 or G4 is the likely failure.
- **Expected verdict:** `CARRY_POSITIVE_BELOW_HURDLE` or a narrow `CARRY_EDGE`, with most of the return from the late-2024 / early-2025 period.

## 10. Running it (PC, Windows CMD)

Close the notebook in Jupyter first. Runtime is minutes.

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest tests.test_funding_carry -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks\Funding_Carry\01_funding_carry.ipynb
git add notebooks\Funding_Carry
git commit -m "FC 01 run"
git push
```

## Questions for ChatGPT

1. Is a 40% reset with a full round trip too conservative, or not conservative enough, versus Binance's actual margin mechanics for a non-portfolio-margin account?
2. Should G4's hurdle use the realized USD cash rate over the period instead of a fixed 4%?

## Decisions needed from Ivan

1. Approve this pre-registration as binding, or amend it before the run.
2. Confirm you can actually trade both Binance spot and Binance USD-M perpetuals from your account and jurisdiction. If you can't, the realistic venue would be Deribit (BTC/ETH spot + perpetual only), which changes the universe and costs.
