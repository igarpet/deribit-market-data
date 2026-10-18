# Multi-Scale HMM research branch

**Question:** can an HMM find momentum / reversal states by itself on 1h, 4h and 1D bars built from 1-minute perpetual data, and do they predict the next bar's return net of costs? Discussions: `../../discussions/Multi_Scale_HMM/`.

| Notebook | Content | Status |
|---|---|---|
| `01_multiscale_hmm.ipynb` | Bars + intrabar emissions from Binance USD-M 1m (BTC/ETH/SOL/XRP); pooled Student-t HMM per scale; per-state $y_{t+1}=a_k+b_k z_t$; TEST 2026-02-01 → 2026-10-18; shared-sign minute null (full refit); verdicts S, E, states-add-value | Run 60ff884 (MSH 002): **NO_TRADEABLE_MOMENTUM_STATES** at all three scales |

Code: `src/data_engineering/modeling/multiscale_hmm.py`. Tests: `tests/test_multiscale_hmm.py`. Outputs: `01/01_*` (selection, states, per-asset, null, summary, decision, frozen models, equity plot).

Data: `<parent of DERIBIT_DATA_ROOT>\Binance\<ASSET>USDT_um_1m.parquet` (or `BINANCE_UM_DIR`) and `Binance\basis\funding\<ASSET>USDT_funding.parquet`.

## Running 01 (PC, Windows CMD)

Close the notebook in Jupyter first.

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest tests.test_multiscale_hmm -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 notebooks\Multi_Scale_HMM\01_multiscale_hmm.ipynb
git add notebooks\Multi_Scale_HMM
git commit -m "MSH 01 run"
git push
```

Optional: `set MSH01_JOBS=3` (null worker processes), `set MSH01_SCALES=1D` (one scale). `MSH01_N_NULL` is for quick looks only; the decision file marks such runs as not registered.
