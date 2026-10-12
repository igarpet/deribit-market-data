# Volatility Risk Premium research branch

Physical-volatility forecasts versus option-implied variance, and ultimately executable delta-hedged option P&L. Discussions: `../../discussions/Volatility_Risk_Premium/`.

## Sequence

| Notebook | Former | Content | Status |
|---|---|---|---|
| `01_vrp_pricing_audit.ipynb` | 15 | Step 1: Black-76 repricing, exchange IV time basis, ATM IV feature vs straddle IV, terminal/30m wedge decomposition, index RV noise | Run; A pass, B no offset (BTC/SOL narrow fail), C fail, D/E reported (VRP 006) |
| `02_long_straddle_settlement.ipynb` | — | Step 2 (long only, VRP 007): ATM straddle bought at the ask + taker fee every hour (TTE 2–20h), held to the exchange delivery price, no hedge. Pre-registered verdict on the session-bootstrap CI of mean return on premium | Run `8fc765d`: **NO_LONG_EDGE**, M = −0.27 [−0.39, −0.12], 20 sessions (VRP 008) |
| `03_passive_execution.ipynb` | Step 3 (VRP 009): passive-then-cross execution of the same straddles, both sides; 5m fill proxy (touch crosses or trade prints at/through the limit); primary q = 0.5, W = 15 min; paired ΔR vs aggressive | Built; not yet run on real data |
| `04_conditional_short.ipynb` | Step 4 (VRP 009): short straddle and defined-risk iron fly conditioned on the implied ÷ causal-forecast variance wedge; development descriptive; freeze after 2026-11-06; binding confirmation on sessions ≥ 2026-11-07 (≥ 20) | Built; not yet run on real data |

Outputs: `01/01_*.csv`, `02/02_*`. Code: `src/data_engineering/modeling/option_pricing.py`, `vrp_audit.py`, `vrp_long_straddle.py`.

Ivan trades long only (VRP 007 §1). Delta-hedged and short-side variants are reported mirrors, not decisions. 02 returned NO_LONG_EDGE. Proposed next (VRP 008 §4): passive-execution study before any conditional long test.

## Running 02 (PC, Windows CMD)

Close the notebook in Jupyter first. The PC must reach `www.deribit.com` (delivery prices, cached to `02/02_delivery_prices.csv`).

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_vrp_long_straddle.py" -v
jupyter nbconvert --to notebook --execute --inplace notebooks\Volatility_Risk_Premium\02_long_straddle_settlement.ipynb
git add notebooks\Volatility_Risk_Premium
git commit -m "VRP 02: long straddle run"
git push
```

Optional: `set VRP02_OFFLINE=1` (use the cached delivery prices only), `set VRP02_ASSETS=BTC,ETH`.

## Samples

Development uses sessions up to the HSMM 06 binding readout (`VRP_END`, default 2026-11-06). The branch's confirmation window is the sessions after it (VRP 002 §2.4).

## Running 03 and 04 (PC, Windows CMD)

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_vrp_execution_conditional.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks\Volatility_Risk_Premium\03_passive_execution.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks\Volatility_Risk_Premium\04_conditional_short.ipynb
git add notebooks\Volatility_Risk_Premium
git commit -m "VRP 03/04: execution and conditional short (development)"
git push
```

VRP 04 later: after 2026-11-06 run once with `set VRP04_FREEZE=1`; from 20 sessions after 2026-11-07 run with `set VRP04_CONFIRM=1` (binding).
