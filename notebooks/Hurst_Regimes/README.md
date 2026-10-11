# Hurst Regimes research branch

**Question:** is there a persistent, predictable variation in the Hurst exponent $H$ of 5-minute price paths that changes the terminal move a 0DTE straddle pays on? Discussions: `../../discussions/Hurst_Regimes/`.

| Notebook | Content | Status |
|---|---|---|
| `01_hurst_variation_gates.ipynb` | Session $H$ (change-of-frequency estimator) on non-overlapping 08:00-to-08:00 blocks; gates G1 variation, G2 persistence, G3 relevance against a volatility-modulated constant-$H$ fGn null; development ≤ 2026-03-31, confirmation from 2026-04-01 (G3, run once) | Run `1216fc8`: **STOP**. G1 fail (excess SD 0.013), G2 pass (lag-1 echo only), G3 fail (ρ₃ ≈ 0) (HR 002) |
| `02_roughness.ipynb` | Descriptive (no gate): price $H$ by horizon (5m–30m, 1–8h, 1–7d); volatility roughness of $\log\sigma$ at 1h/4h/1d with noise correction and simulation calibration; monofractal check $\zeta(q)$; ACF power-law vs exponential; fOU fit (thesis estimators); half-year stability | Built; not yet run on real data (HR 003) |

Code: `src/data_engineering/modeling/hurst_regimes.py`, `hurst_roughness.py`. Tests: `tests/test_hurst_regimes.py`, `test_hurst_roughness.py`. Outputs: `01/01_*`, `02/02_*`.

Data: `DERIBIT_DATA_ROOT/history/price_5m/<ASSET>.csv.gz` (from `tools/download_price_history.py`, already downloaded for VD 10c).

## Running 01 (PC, Windows CMD)

Close the notebook in Jupyter first.

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_hurst_regimes.py" -v
jupyter nbconvert --to notebook --execute --inplace notebooks\Hurst_Regimes\01_hurst_variation_gates.ipynb
git add notebooks\Hurst_Regimes
git commit -m "Hurst Regimes 01: development run"
git push
```

Confirmation (only after the development readout is accepted): `set HR_CONFIRM=1` before the `jupyter` line. Optional: `set HR_ASSETS=BTC,ETH`, `set HR_N_REP=400`.

## Running 02 (PC, Windows CMD)

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_hurst_roughness.py" -v
jupyter nbconvert --to notebook --execute --inplace notebooks\Hurst_Regimes\02_roughness.ipynb
git add notebooks\Hurst_Regimes
git commit -m "Hurst Regimes 02: roughness run"
git push
```
