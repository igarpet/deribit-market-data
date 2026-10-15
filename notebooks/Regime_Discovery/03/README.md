# RD03 — Frozen AM-T-HAR-Blend return-density benchmark vs observed and latent regimes

RD03 reuses the **actual archived VD18 `AM_T_HAR_BLEND` fitted mixture-variance parameters** to test whether market microstructure–derived conditional return-location shifts improve next 5/15/30/60-minute raw signed-return likelihood. It compares a sign-only Ridge benchmark, a common continuous Ridge predictor, a CART observable-regime response, and frozen-spec latent static / HMM / duration-overlay response predictions. The primary horizon is **30 minutes**. BTC and ETH are primary; SOL and XRP are secondary. All methods have **identical conditional variance particles**, preserving the existing forecast scale and mixing shape; there is no new volatility fit.

**Prerequisites:** execute CS04-A and retain local Parquet feature stores; keep the existing archived `notebooks/Volatility_Dynamics/18/18_manifest.json` intact. The archived baseline is reconstructed with VD14's **original historical TRAIN clock normalizers**; later dates are only used for RD03 return-response training or evaluation. Input paths derive from `DERIBIT_DATA_ROOT` in `.env`.

Run from Windows CMD:

```bat
cd /d D:\Projects\deribit_market_data
git pull origin main
conda activate deribit-data
python -m pytest -q tests\test_regime_density_bridge.py
jupyter lab
```

Run `notebooks/Regime_Discovery/03_return_density_bridge.ipynb`. Normal mode reads local spot price histories, the four CS04-A per-candle feature stores, and archived VD18 serialized fitted parameters. It writes aggregate-only outputs to the nearby `03/` directory:

- `03_scores.csv` — asset × horizon × model signed return NLL, UP Brier and forecast mean.
- `03_paired_day_bootstrap.csv` — nominal whole-day comparisons versus archived VD18, exploratory only.
- `03_comparison_summary.csv` — incremental return-density scores versus direct continuous predictor and static state-ablation.
- `03_model_diagnostics.csv` — compact structure/response fit summaries, no per-origin state probabilities.
- `03_manifest.json` — provenance, exact chronological boundaries and coverage.
- `03_decision.json` — non-confirmatory screening interpretation.

Use `RD03_SYNTHETIC=1` for an isolated aggregate-scoring smoke test with no real market inputs. The smoke test does **not** exercise archived VD18 forward-variance numerical reconstruction, which requires the real repository data and parameters.

**Constraints:** RD02 historical screen was negative. Current Sep–Oct 2026 evaluation dates have been inspected elsewhere, so any positive RD03 results remain exploratory. This experiment tests *conditional location only*; it does not fit a full regime-specific heavy-tailed return mixture or calibrate conditional skewness. No AM-T-HAR-Blend code, collectors, option prices or strategies are modified. The fitted two-stage duration hazard is **not** a jointly estimated HSMM.

[Pre-registered protocol](../../../discussions/Regime_Discovery/003_chatgpt_regime_density_bridge_preregistration.md)
