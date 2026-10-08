# Volatility Risk Premium research branch

Physical-volatility forecasts versus option-implied variance, and ultimately executable delta-hedged option P&L. Discussions: `../../discussions/Volatility_Risk_Premium/`.

## Sequence

| Notebook | Former | Content | Status |
|---|---|---|---|
| `01_vrp_pricing_audit.ipynb` | 15 | Step 1: Black-76 repricing, exchange IV time basis, ATM IV feature vs straddle IV, terminal/30m wedge decomposition, index RV noise | Run; A pass, B no offset (BTC/SOL narrow fail), C fail, D/E reported (VRP 006) |

Outputs: `01_*.csv` beside the notebook. Code: `src/data_engineering/modeling/option_pricing.py`, `vrp_audit.py`.

Planned: VRP 02, the unconditional hold-to-settlement ATM straddle at executable prices (5m and 30m perp hedges), then the conditional forecast-vs-price wedge.

## Samples

Development uses sessions up to the HSMM 06 binding readout (`VRP_END`, default 2026-11-06). The branch's confirmation window is the sessions after it (VRP 002 §2.4).
