# Funding Carry 002 — FC 01 readout: a real but small premium that does not beat cash

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-18 |
| **Branch** | Funding Carry |
| **Responds to** | FC 001 pre-registration; Ivan's run cc17447 |
| **Status** | Result. Registered run (30/30 assets, none missing). The verdict is binding. |

## 1. Verdict: C `NO_CARRY_EDGE` (fails G2); U `CARRY_POSITIVE_BELOW_HURDLE`

Period: 2024-08-08 → 2026-09-29 (2.14 years; ends where the funding files end). DEV runs until 2026-01-01.

| | C (primary) | U (always in) |
|---|---|---|
| Net, on reserved capital (ann.) | **+1.39%** [+0.46%, +2.85%] | +2.01% [+0.67%, +4.03%] |
| DEV / TEST, on reserved | +2.17% / **−0.08%** | +2.88% / +0.36% |
| Net, on deployed capital | 6.23% | 2.05% |
| Share of asset-days in position | 22% (2026: 6%) | 100% |
| Assets net positive | 28 / 30 | 24 / 30 |
| Max drawdown (reserved) | 0.14% | 0.74% |
| Gates G1 / G2 / G3 / G4 | ✔ / **✘** / ✔ / ✔ | ✔ / ✔ / ✔ / **✘** |

- **Low-cost sensitivity (13.5 bp per side):** the same verdicts (C 1.55%, U 2.15% on reserved).
- **Prediction recorded in FC 001 §9** was "positive in 2024–25, weaker in 2026; below-hurdle or narrow edge". The pattern came out as predicted. The binding primary failed on G2 rather than G4.

## 2. What the premium actually is

- **Gross funding for the always-in trade:** **4.1% a year per unit of notional**, averaged over 30 assets.
- **Basis P&L:** −0.02% a year, essentially zero. The hedge works.
- **Costs:** 1.0% a year, mostly the 70 margin resets during the Q4 2024 rally.
- **On 1.5 units of capital:** about 2% net.

**Funding sits below the 0.01%/8h base rate** (10.95% a year) on average:
- mean annualized funding per asset ranges from −9% (SEI, ENA) to +6%;
- 16–57% of prints are negative.

**Concentrated in time.** By quarter, U on reserved capital (annualized):

| Quarter | Return |
|---|---|
| Q4 2024 | +9.8% (the bull run) |
| Q3 2025 | +6.6% |
| Q4 2025 – Q2 2026 | −1.1% to −0.1% |

**C's TEST failure is mechanical.** Trailing funding was below the 10% entry threshold for almost all of 2026, so C held positions on only 6% of asset-days and earned nothing. The few 2026 Q3 entries lost slightly to costs.

## 3. Why G4 "passing" for C is not a real pass (opinion)

The 6.2% on deployed capital is earned only 22% of the time. If the idle 78% earns the same 4% cash rate the hurdle assumes, C's total is about **4.5%** against **4.0%** for cash alone. That is a 0.5%-a-year excess for taking full exchange counterparty risk on both legs. Even if TEST had passed, this would not be a worthwhile trade.

**Threshold grid (descriptive):** no entry/exit pair has a meaningful TEST return; the best is +0.26% a year on reserved capital. Higher entry thresholds raise the deployed return (up to 12.7%), but deploy capital 2–5% of the time.

## 4. Could better account mechanics rescue it? (approximate, opinion)

The most favourable realistic case:
- portfolio/unified margin, so capital is about 1.1 per unit notional;
- no resets.

That gives about 4.1% − 0.2% (entry/exit amortised) ≈ **3.5% on capital** for the always-in trade over 2024–26. That is still at or below cash, and it is concentrated in bull phases. Funding carry here is a **bull-market beta premium**, not an all-weather edge.

## 5. Consequence (opinion)

Funding carry was the last untested structural premium on the data we hold. It is positive, as expected, and small, which was the risk:
- It does not beat cash after costs and capital.
- 2026 so far has paid nothing.

Together with MSH 002, the research has no strategy with positive expectancy over cash for a retail account at taker costs.

What remains is not a new signal. It is a different cost or structure regime:
1. lower fees (VIP/maker tiers);
2. a different product, such as weekly options VRP with defined risk, where spreads take a smaller share of the premium;
3. treating funding carry as an opportunistic, discretionary bull-phase trade. For example, enter when $f_7 \ge 20\text{–}30\%$. That is a judgement call, not a validated edge.

## Questions for ChatGPT

1. Do you see an accounting error that would understate carry? For example, funding valued at the perpetual close instead of the mark price, or the 4h-interval assets (TON, WIF, TIA, ENA).
2. Is there a carry variant with a structural reason to differ that is worth pre-registering, such as cross-sectional rotation or the reverse trade with spot borrow? Or should the branch close?

## Decisions needed from Ivan

1. Accept FC 01 as binding and close the branch? (Recommended.)
2. Choose the project's next step:
   - **(a)** weekly-expiry options VRP with defined risk (Deribit vs Binance costs);
   - **(b)** pause new research and run the frozen prospective checks already scheduled (VD19/SJ05 mid-November);
   - **(c)** stop.
