# Funding Carry research branch

**Question:** does long spot / short perpetual funding harvesting on Binance beat cash after costs? Discussions: `../../discussions/Funding_Carry/`.

| Notebook | Content | Status |
|---|---|---|
| `01_funding_carry.ipynb` | PF02 universe; daily frames from 1m spot/perp + funding prints; C (trailing-7d funding hysteresis) and U (always in); base and low-cost; per-asset, episodes, threshold grid (descriptive); verdict G1–G4 | Run cc17447 (FC 002): **NO_CARRY_EDGE** (C); U positive but below the cash hurdle |

Code: `src/data_engineering/modeling/funding_carry.py`. Tests: `tests/test_funding_carry.py`. Outputs: `01/01_*`.

Data: `<parent of DERIBIT_DATA_ROOT>\Binance` (or `BINANCE_UM_DIR`): `<A>USDT_um_1m.parquet`, `basis\spot\<A>USDT_spot_1m.parquet`, `basis\funding\<A>USDT_funding.parquet` (`tools\download_binance_basis_data.py`).

## Running 01 (PC, Windows CMD)

Close the notebook in Jupyter first.

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest tests.test_funding_carry -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks\Funding_Carry\01_funding_carry.ipynb
git add notebooks\Funding_Carry
git commit -m "FC 01 run"
git push
```
