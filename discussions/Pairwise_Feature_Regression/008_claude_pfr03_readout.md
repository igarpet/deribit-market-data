# Pairwise Feature Regression 008 — PFR 03 readout: no catalog feature predicts direction

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-15 |
| Branch | Pairwise Feature Regression |
| Responds to | PFR 007 pre-registration; Ivan's run bdf2ba3 |
| Status | Result. The pre-registered verdict is binding. |

## 1. Verdict: `NO_DIRECTIONAL_FEATURE`

- **Sample:** 24 complete days per asset. Discovery runs 2026-09-19 → 10-02; confirmation runs 10-03 → 10-12.
- **Scope:** 81 features × 3 horizons × 4 assets = 972 cells. 860 have enough data; all nine L2 instability features are short because of their collector start date.
- **Outcome:** 5 cells pass discovery and **0 confirm**.

| Discovery pass | Discovery partial IC | Confirmation partial IC (one-sided 95% band) |
|---|---|---|
| BTC `tfi` / `buy_share` 60m | −0.036 | −0.007 [−0.025, +0.015] |
| BTC `obi_20_change` 60m | +0.020 | +0.011 [−0.005, +0.029] |
| XRP `log_spread_bps` 5m | −0.027 | **+0.035** (sign flips) |
| XRP `ret_15m` 15m | −0.050 | **+0.054** (sign flips) |

(`tfi` and `buy_share` are the same variable, as noted in PFR 003.)

## 2. A test that involves no selection

Across all eligible cells, how well does the discovery IC predict the confirmation IC?

| Horizon | Cells | corr(discovery IC, confirmation IC) | Same sign in both halves |
|---|---|---|---|
| 5m | 288 | 0.15 | 51% |
| 15m | 286 | 0.05 | 53% |
| 60m | 286 | 0.08 | 50% |

Sign agreement is a coin flip. On this sample, the direction of a feature's relationship with future returns does not persist from one fortnight to the next.

## 3. Large-looking confirmation ICs are noise

- The largest confirmation ICs (0.15–0.24) all sit at 60m on the options or depth features:
  - ETH `atm_iv_d60` at −0.24;
  - XRP `fit_skew_rel` at +0.19;
  - SOL `curvature_025` at +0.17.
- All have discovery ICs near 0.
- With about 10 confirmation days and overlapping 60-minute windows, the effective sample is roughly a few hundred independent observations. An IC of 0.15–0.2 is within sampling noise, and picking these cells after seeing them would be selection on noise.

## 4. The only consistent pattern

**Volatility features** (`log_range_bps`, `log_rv_1h`, `log_rv_4h`) have a negative partial IC in all four assets in both halves: about −0.02 to −0.05.

This is the familiar link between high volatility and falling prices, and here it rests on a 24-day window. None of these cells is individually significant, and the decile long-short is negative or about 1 bp. I record it as an observation, not a signal.

## 5. Power, stated honestly

- With 24 days, this screen can detect ICs of roughly 0.05 or more at 5 minutes, and larger at 60.
- The CS03c flow effect is real but about 0.02 in IC terms. That is below this screen's resolution, and needed two years of Binance data to establish.
- So PFR 03 does **not** prove that the collector features carry no information. It shows **no large or stable directional effect** in the catalog. A small one could exist, and CS03c/CS03e showed what an effect that small is worth: nothing after costs.

## 6. Conclusion

- **The collector's feature catalog has no tradeable directional predictor:** not order-book imbalance, not trade flow, not options skew or IV changes, at 5–60 minutes.
- **This matches the result on the Binance spot/perp history:** direction is efficient to within costs.
- **Rerunning later won't change the picture.** A rerun with several months of collector data would raise power, but the strongest catalog effects here are below 0.05 in IC, and the economics of such ICs are already known.

## 7. Decision needed from Ivan

Accept. The direction search is exhausted across price, flow, order book and options features.
