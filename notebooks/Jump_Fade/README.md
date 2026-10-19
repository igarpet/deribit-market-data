# Jump Fade research branch

**Question:** does a passive ±1.5σ bracket fade with a take-profit at the pre-move price have positive expectancy on perpetuals? Discussions: `../../discussions/Jump_Fade/`.

| Notebook | Content | Status |
|---|---|---|
| `01_jump_fade.ipynb` | Binance USD-M 1m OHLC BTC/ETH/SOL/XRP from 2024-10; bracket at every 5m boundary; TP at P0, 3σ stop, 60-min time stop; splits all / burst / recent_jump; Deribit and Binance costs; sign-flip null (199); grid | Pre-registered (JF 001), awaiting run |

Code: `src/data_engineering/modeling/jump_fade.py`. Tests: `tests/test_jump_fade.py`. Outputs: `01/01_*`.

## Running 01 (PC, Windows CMD)

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest tests.test_jump_fade -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 notebooks\Jump_Fade\01_jump_fade.ipynb
git add notebooks\Jump_Fade
git commit -m "JF 01 run"
git push
```
