# Volatility Regression — independent research branch

**Question:** How much of the next five-minute interval's **volatility** is predictable using the current five-minute Deribit market state? Which contemporaneous explanatory features retain incremental linear forecast value?

Discussion: [../../discussions/Volatility_Regression](../../discussions/Volatility_Regression/).

| Notebook | Purpose | Status |
|---|---|---|
| `01_full_feature_5m_ols.ipynb` | v0.1.2: next-bar volatility, two targets (trade-range Parkinson, index \|r\|) on the same rows; target-quality audit; full and backward-reduced OLS vs parameter-free references (train mean, persistence); OOS R² vs train mean; clock-hour coverage | **Closed (VR 008).** Diagnostic only: n/p≈8, capped deletion, no 05:00–08:59 UTC rows; coefficients not interpretable |
| `02_feature_provenance_audit.ipynb` | Read-only futures-collector/schema, raw-to-feature, quality-flag and regression-coverage diagnostics | v0.1.0 executed (`73767f5`, VR 006). **v0.2.0 built, not yet executed:** all expiries; unmatched states by expiry/hour; NULL index returns across assets; data-derived collector vintages and feature shift at each boundary |
| `03_nested_block_5m_ols.ipynb` | Nested blocks B0 returns/vol/clock → B1 flow/basis → B2 L2 book → B3 ATM options; shape/fit/wing features excluded by design; ≥80% dev coverage admission; n ≥ 20p; forward-chained folds + one final test; fold-majority deletion in B2; residuals by collector vintage | Built; **not executed**. Run 02 first (uses `02_vintage_boundaries.csv`) |

Reusable code: `src/data_engineering/modeling/vol_ols_5m.py` (01), `feature_provenance_audit.py` (02), `vol_ols_5m_blocks.py` (03). Synthetic checks: `tests/test_vol_ols_5m.py`, `tests/test_feature_provenance_audit.py`, `tests/test_vol_ols_5m_blocks.py`.

## Scientific boundaries

- **Y is volatility**, not directional returns or squared returns: next-bar trade-range Parkinson volatility and, as a measurement check, next-bar index |r|/√(2/π); `log(σ + train-only ε)` is the response.
- **Paired current/next eligibility (v0.1.2):** comparison rows require both estimators at t and t+5; notebook fails if complete-case/scored masks or retained candidate features differ across targets. Exclusions are tracked, not silently ignored.
- **Pure OLS.** No HAR study in this branch (Ivan, 2026-10-08); models are compared with parameter-free references only.
- The target window is exactly one subsequent five-minute bar, not 30/60 minutes. Do not bridge missing futures bars or 08:00 UTC option-session changes.
- Trade-range bid/ask bounce and incomplete OHLC coverage may degrade the proxy. This must be audited before making trading claims.
- Start with the full established causal feature catalog and transparently distinguish low-coverage/exact-alias exclusion from predictive-value elimination.
- Final held-out test is scored once; model selection uses only earlier train/selection sessions. Baseline and full/reduced OLS must be evaluated on common rows.
- **No development cutoff from 02 v0.2.0 / 03 on.** Research branches are independent; HSMM's holdout does not constrain this branch (Ivan, 2026-10-09). 01 keeps its original 2026-10-07 cutoff as a historical record.
- Current results are **research diagnostics**, not a validated trading signal.

## Run on PC

From the repo root in the `deribit-data` environment:

```bat
python -m unittest discover -s tests -p "test_feature_provenance_audit.py" -v
python -m unittest discover -s tests -p "test_vol_ols_5m_blocks.py" -v
jupyter nbconvert --to notebook --execute --inplace notebooks/Volatility_Regression/02_feature_provenance_audit.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/Volatility_Regression/03_nested_block_5m_ols.ipynb
```

02 must run before 03.

This runs against `DERIBIT_DATA_ROOT/database/research.duckdb` **read-only**. It writes `01_*.csv` and `01_run_metadata.json` beside the notebook; commit those and the executed notebook for review. Close JupyterLab's open copy before pulling remote notebook changes.

## Volatility Regression 02 — feature provenance

Uses `src/data_engineering/modeling/feature_provenance_audit.py` and `tests/test_feature_provenance_audit.py`. Reads futures five-minute Parquet and research DuckDB without modification. v0.2.0 reads all expiries; the section reproducing 01's complete-case loss keeps 01's cutoff. Reports `02_*.csv` outputs on collector field vintages, readiness flags, loss along data-engineering stages, and full-OLS complete-case exclusion.

Run from repository root with the `deribit-data` environment:

```bat
python -m unittest discover -s tests -p "test_feature_provenance_audit.py" -v
jupyter nbconvert --to notebook --execute --inplace notebooks/Volatility_Regression/02_feature_provenance_audit.ipynb
```

Commit executed notebook, `02_*.csv` and `02_run_metadata.json` for interpretation.
