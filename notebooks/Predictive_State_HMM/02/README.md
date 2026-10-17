# PSH02 — Volatility × signed-flow interaction, perpetual forecasts

**Preregistered and implemented 2026-10-17; awaiting owner-run historical results.** A research-only next experiment that leaves PSH01 and every existing frozen HSMM/VD19 specification intact.

## Main question

Does causal five-minute aggressive taker flow forecast the **next completed perpetual 5-minute direction** beyond continuous price/volatility features, and does either (a) a smooth flow×volatility interaction or (b) a filtered PSH01 HMM-state×flow interaction add independent skill? The 30-minute perpetual horizon is separately reported as secondary. Prior PSH01 showed **NO_STABLE_HMM_INCREMENT** against a nonlinear volatility null and no robust perp trading edge; no prior state label is taken as proven.

This branch uses only feature families with demonstrated historical out-of-sample information: VD12's 30/240-minute volatility memories and CS03c's signed aggressive flow. The present historical interval has been repeatedly studied and is **not pristine prospective confirmation**.

## Files

- Protocol: `../../discussions/Predictive_State_HMM/002_chatgpt_psh02_preregistration.md`.
- Engine: `../../src/data_engineering/modeling/predictive_state_flow.py`.
- Tests: `../../tests/test_predictive_state_flow.py`.
- Notebook: `02_volatility_signed_flow.ipynb`.
- Written artifacts after execution: `02/02_forecast_scores.csv`, `02/02_paired_day_comparisons.csv`, `02/02_perp_cost_scenarios.csv`, `02/02_coverage.csv`, `02/02_decision.json`, `02/02_manifest.json`.

## Run on Windows CMD

```bat
cd /d D:\Projects\deribit_market_data
git pull
conda activate deribit-data
python -m pytest -q tests\test_predictive_state_flow.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks\Predictive_State_HMM\02_volatility_signed_flow.ipynb
```

No downloads. Sources are local `DERIBIT_DATA_ROOT/history/price_5m/<ASSET>.csv.gz` (spot 5m closes) and `BINANCE_UM_DIR/<ASSET>USDT_um_1m.parquet` (Binance USD-M futures one-minute OHLCV and signed taker quote volumes), or the established sibling `Binance/` directory when `BINANCE_UM_DIR` is unset. **PSH02 uses real perpetual return labels rather than using spot returns as fictitious perp execution.**

After the run, commit the *small aggregate files* under `02/` and the executed notebook. Do not upload per-origin data, future labels, or raw market data.

## Inference boundaries

The primary binary score excludes next-return zero bars identically across all models; its estimate is conditional on a nonzero next five-minute return. Bonferroni controls the ten BTC/ETH primary comparisons within this specified experiment, not all earlier research. A superior historical HMM model is not prospective confirmation. Perpetual cost scenarios start at the observed completed 5-minute close and assume 12bp round-trip, so they are **not executable fill backtests**. Funding/fees/latency/spread/queue/liquidation are not modeled. Options P&L is not part of PSH02. No live trading is authorized.
