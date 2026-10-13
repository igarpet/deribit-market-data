# Regime Structure 007 — RS 03 pre-registration: big-move days vs normal days

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-13 |
| Branch | Regime Structure |
| Responds to | RS 006; Ivan (chat): "what happens on big move days compared to normal days … maybe that way we can find regimes" |
| Status | Pre-registration, before any real-data RS 03 output |

## 1. Why not just study the winning sessions

- In VRP 02, session straddle P&L has a correlation of 0.98 with realised move / implied move.
- Only 3 of 20 sessions were clearly positive (09-28, 10-07, 09-29). They won because price moved 44–72% more than implied.
- Studying 3 outcome-selected days would find chance differences in whichever features one looks at.
- So the question is asked on about 2 years of Binance data, with about 300 big-move days per asset, and an explicit dev/test split.

## 2. Definitions (`big_move_days v1.0`)

- **Session:** the 24 h ending 08:00 UTC (Deribit daily expiry), labelled by its end date. 5-minute bars from Binance USD-M 1m: BTC, ETH, SOL, XRP.
- **Forecast:** $\hat\sigma$ = RMS of the previous 7 sessions' 5m realised vol (causal).
- **BIG:** $z=\lvert\Delta\log P_{\text{session}}\rvert/\hat\sigma\ge q_{85}$ of $z$ on the development window, per asset. $z$ is what a straddle held to expiry is paid on.
- **BIG_REST:** the same on the move from 10:00 to 08:00, scaled by $\hat\sigma\sqrt{22/24}$. It is the target for EARLY features, so the early move itself is not counted.
- **Windows:** development up to 2026-03-31; **test from 2026-04-01**.

## 3. Feature groups

| Group | When known | Features | Target | Use |
|---|---|---|---|---|
| DURING | after the session | efficiency ratio; volume and trades vs 30-session median; taker flow, and flow aligned with the move; number of ≥3σ 5m bars; max excursion; giveback; share of the move in the top hour; which hour; hourly profiles | BIG | **Descriptive only.** Many of these are consequences of a big move. |
| PRE | 08:00 | previous session: RV vs forecast, volume, trades, signed and absolute return, efficiency, flow, shocks; its last 4 h: RV, volume, return; 7- vs 30-session vol; distance to 7-day high and low; weekend; Monday | BIG | Screen and confirm |
| EARLY | 10:00 | first-2h RV, volume, absolute return, flow | BIG_REST | Screen and confirm |

## 4. Rule

For each feature, AUC = P(feature on a big day > feature on a normal day), pooled over assets, with a bootstrap over sessions (all assets of a session resampled together, 2,000 draws).

1. **Candidate:** development-window 95% CI excludes 0.5.
2. **Confirmed:** test-window CI at level $1-0.05/k$, where $k$ is the number of candidates in its group, excludes 0.5 on the same side.

Only confirmed features are findings. Everything else is description.

**Clustering (the regime question):**
- lift $P(\text{BIG}_{s+1}\mid\text{BIG}_s)/P(\text{BIG})$, with a session-bootstrap CI;
- the share of sessions where all 4 assets are big (against independence);
- the share of big days that are big in at least 3 assets;
- the share of big days that are up-moves.

## 5. Results I expect to be mechanical (flagged before the run)

1. **`pre_rv7_vs_rv30` low before big days.** When the last 7 sessions were calmer than the last 30, the 7-session forecast is too low, so $z$ is large. This is volatility mean reversion and a too-short forecast, not a regime. If confirmed, the right conclusion is "use a longer-memory forecast". The synthetic smoke run, which has no such effect planted, already confirms it for exactly this reason.
2. **EARLY volume and RV high before a big rest of session.** Intraday volatility clustering. Real, but the option market sees it too, so it only matters if implied vol lags it.
3. **DURING:** big days will show higher volume, more shocks, and more aligned flow by construction.

A finding beyond these three is what would be new.

## 6. Synthetic validation

`tests/test_big_move_days.py`, 6 tests:
- sessions start at 08:00 and are complete;
- session return equals the price change;
- the forecast uses only prior sessions;
- AUC matches brute force, including weights;
- EARLY target excludes the first 2 h;
- a planted precursor (quiet last 4 h before big days) is confirmed, while without the plant at most one PRE feature passes. That one is `pre_rv7_vs_rv30`, per §5.1.

## 7. Decision needed from Ivan

Run RS 03. Binding for which PRE and EARLY features count as confirmed.
