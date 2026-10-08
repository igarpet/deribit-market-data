# Volatility Risk Premium research branch

Physical-volatility forecasts versus option-implied variance, and ultimately executable delta-hedged option P&L (discussions 017–018).

## Sequence

| Notebook | Purpose | Status |
|---|---|---|
| `15_vrp_pricing_audit.ipynb` | Step 1: Black-76 repricing, exchange IV time basis, ATM IV feature vs straddle IV, decomposition of Module 12's terminal/30m realized-vs-implied sign flip, index RV noise check | built, not yet run on real data |

Planned: unconditional premium (hold-to-settlement ATM straddle, executable prices) → conditional forecast-price wedge.

## Samples

Development uses sessions up to the Module 12 binding readout (`VRP_END`, default 2026-11-06). The branch's confirmation window is the sessions after it (018 §2.4).
