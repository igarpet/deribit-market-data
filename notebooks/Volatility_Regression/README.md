# Volatility Regression — independent research branch

**Question:** How much of the next five-minute interval's **volatility** is predictable using the current five-minute Deribit market state? Which contemporaneous explanatory features retain incremental linear forecast value?

Discussion: [../../discussions/Volatility_Regression](../../discussions/Volatility_Regression/).

| Notebook | Purpose | Status |
|---|---|---|
| `01_full_feature_5m_ols.ipynb` | Next-bar Parkinson volatility; full causal feature-catalog OLS, missingness/rank inventory, chronological train/selection/test, greedy backward elimination, comparison to HAR/clock baseline | Built; **not executed on the local research DB** |

Reusable code: `src/data_engineering/modeling/vol_ols_5m.py` (`VOL_OLS_5M_VERSION`). Synthetic checks: `tests/test_vol_ols_5m.py`.

## Scientific boundaries

- **Y is volatility**, not returns or squared returns. The preliminary observable 5m proxy is next-bar **trade-high/low Parkinson volatility**; `log(Y + train-only epsilon)` is the estimated response.
- The target window is exactly one subsequent five-minute bar, not 30/60 minutes. Do not bridge missing futures bars or 08:00 UTC option-session changes.
- Trade-range bid/ask bounce and incomplete OHLC coverage may degrade the proxy. This must be audited before making trading claims.
- Start with the full established causal feature catalog and transparently distinguish low-coverage/exact-alias exclusion from predictive-value elimination.
- Final held-out test is scored once; model selection uses only earlier train/selection sessions. Baseline and full/reduced OLS must be evaluated on common rows.
- Development data stop at expiry 2026-10-07 to avoid touching the pre-registered HSMM 06 prospective confirmation observations. HSMM frozen specs remain unchanged.
- Current results are **research diagnostics**, not a validated trading signal.

## Run on PC

From the repo root in the `deribit-data` environment:

```bat
python -m unittest discover -s tests -p "test_vol_ols_5m.py" -v
jupyter nbconvert --to notebook --execute --inplace notebooks/Volatility_Regression/01_full_feature_5m_ols.ipynb
```

This runs against `DERIBIT_DATA_ROOT/database/research.duckdb` **read-only**. It writes `01_*.csv` and `01_run_metadata.json` beside the notebook; commit those and the executed notebook for review. Close JupyterLab's open copy before pulling remote notebook changes.
