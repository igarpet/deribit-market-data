# VD20: how well does the volatility engine forecast the next 24 hours? (preregistration)

Fixed before any real-data output. Code: `src/data_engineering/modeling/daily_vol_forecast.py` (VD20-0.1.0).
Notebook: `notebooks/Volatility_Dynamics/20_next_24h_volatility.ipynb`.

## Question

1. **Skill:** does the volatility engine (AM-T-HAR-Blend, whose variance level is VD18 HAR_4) forecast realised
   variance over the next 24 hours better than naive persistence, and better than a standard daily HAR?
2. **Use:** does its "volatile day" call add information that the option price does not already have? A long OTM
   strangle bought on predicted-volatile days only pays if realised volatility beats **implied**, not if it beats
   yesterday's volatility.

## Prior evidence (stated before the run)

- VD19, 12:00 entry (1200 min to expiry): β = 0.06 [−0.52, 0.79]; implied dominates the encompassing regression.
- VD 10d: calm-regime OTM wings at the ask ≈ 1.7× fair, net of fees. RS04: implied already reacts to realised volatility.
- The engine's longest lag is 1440 min. It cannot see weekly or monthly volatility memory, which matters most at a 24h horizon.
  Expectation: daily HAR ≥ engine at 24h; every model ≫ chance at ranking volatile days; nothing beats implied.

## Design

- **Assets:** AVAX, BTC, ETH, SOL, XRP (HYPE and TRX excluded). Option diagnostic: BTC, ETH, SOL, XRP.
- **Data:** Binance 5m spot returns (`history/price_5m`) through `forecast_end`.
- **Origins:** one per UTC day at 08:00, h = 1440 min (PRIMARY: the 24h until the next daily expiry). Also 12:00,
  h = 1200, used only to line up with VD19's option rows.
- **Target:** RV = Σ r²(5m) over (t, t+h], strict complete window. Features use only bars ending ≤ t.
- **Split:** the engine is used exactly as VD19 refits it: VD18 HAR_4, VD14 frame, VD14 TRAIN split, data
  < 2026-09-19. All daily competitors are fitted on origins whose target ends inside the same TRAIN dates.
  EVAL = every origin after the TRAIN end (the VD14 VALID+TEST periods plus the post-sample period to `forecast_end`).

## Models (variance forecast F)

| name | forecast |
|---|---|
| NAIVE | last-24h RV × per-asset TRAIN scale |
| ENGINE | VD18 HAR_4 net-return variance at h (VD19 `fit_har_models`/`model_sigma`) |
| HAR_D | log RV_next = a_asset + b₁ log RV_1d + b₅ log RV_5d + b₂₂ log RV_22d; per-asset TRAIN smearing |
| HAR_DX | HAR_D + log RV over the last 30/60/240 min + the negative part of the standardised 24h return |

## Metrics (EVAL, day-block bootstrap, 4000 draws)

- QLIKE L = RV/F − log(RV/F) − 1; MSE of log; Mincer–Zarnowitz log RV = a + b log F (R², b).
- **Volatile day:** RV_next above the asset's 80th percentile of RV_next over EVAL (a fixed per-asset label; amended
  before any real-data run: a trailing threshold gave AUC 0.55 on constant-volatility synthetic data because its own
  sampling noise predicts the label). Metrics: AUC of log F − log threshold; hit rate in the top and bottom forecast
  deciles vs the 20% base rate.
- **Tails for strangles:** z = R_next/√F. Observed P(|z| > k) for k ∈ {1, 1.5, 2, 2.5}, against Gaussian and against
  a Student-t (df and scale fitted on TRAIN z).

## Gates

- **G1 skill:** best of ENGINE and HAR_D beats NAIVE on QLIKE, with the bootstrap 97.5% bound of the difference below 0.
- **G2 engine adequacy:** ENGINE − HAR_D QLIKE difference. Reported; if HAR_D is significantly better, the 24h engine
  should be a daily HAR, not HAR_4.
- **G3 beyond implied (decides the option use):** on VD19's 12:00 rows (≈ 22 sessions), the encompassing regression
  log RV_path = a + b_imp log σ²_imp + b_mod log F_model, whole-session bootstrap. Pass needs b_mod with lower bound > 0
  for the best model.
  Economics: VD19 ATM-straddle R_net when F_model ≥ 1.25² σ²_imp, reported only.

## Verdict

- `BUY_OTM_ON_FORECAST_SUPPORTED_PROSPECTIVE`: G1 and G3 pass. This still needs a prospective OTM strangle at the ask
  (no OTM 24h economics in the stored outputs).
- `FORECASTS_VOL_BUT_NOT_BEYOND_IMPLIED`: G1 passes and G3 fails or is underpowered.
- `NO_SKILL`: G1 fails.

Status: HISTORICAL_EXPLORATORY. The pre-2026-09-19 data were inspected by earlier modules; G3 uses ≈ 86 rows in ≈ 22 sessions.
