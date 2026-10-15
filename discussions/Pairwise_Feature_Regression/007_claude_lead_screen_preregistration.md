# Pairwise Feature Regression 007 — PFR 03 lead screen: which features predict the perp return after t? (pre-registration)

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-15 |
| Branch | Pairwise Feature Regression (new module 03; PFR 01/02 files untouched) |
| Responds to | Ivan: "test the features that were regressed to price to see which are worth looking at for directionality" |
| Status | Pre-registration. Written before any real-data run. |

## 1. Why a new module

PFR 01 regressed every feature on **same-bar** variables. Its strongest return links are mechanical, as PFR 003 already explained:

| Link with `return_5m` | r (BTC) | Why it is mechanical |
|---|---|---|
| `perp_ret_5m` | 0.995 | The same price move |
| `microprice_ref_bps` | −0.85 | Its denominator is the bar's end price |
| `close_vs_vwap_bps` | 0.81 | Contains the bar's own return |
| `ret_15m` | 0.57 | Contains the bar's own return |

Direction requires $X_t$ to forecast returns **after** $t$. That has not been tested on the collector feature catalog: Volatility Regression targeted volatility, and the Order-Book branch never started.

## 2. Design (frozen)

- **Data:** `time_series_feature_state` (research DuckDB), BTC/ETH/SOL/XRP, complete UTC days only. Features: the PFR 01 catalog via `model_feature_frame` (same transforms and readiness masks), all except the control.
- **Target:** the Deribit perpetual mid forward return,
  $$y_{t,h}=10^4\sum_{j=1}^{h/5} r^{perp}_{t+5j}\ \text{(bp)},\qquad h\in\{5,15,60\}\ \text{min}.$$
  It uses the integrity-gated `perp_ret_5m`, and every future bar must be present.
- **Statistic:** partial IC, i.e. the Pearson correlation of $X_t$ and $y_{t,h}$ after linearly removing the current perp return $r_t$ from both. This removes bounce and same-bar construction links. Raw IC is also reported.
- **Fitted on DISCOVERY only:** winsorisation of $X$ at the 0.5/99.5% quantiles, residualisation coefficients, and the decile cut-offs.
- **Split:** DISCOVERY is the first 60% of UTC days; CONFIRMATION is the last 40%.
- **Inference:** whole-UTC-day bootstrap (2000 draws) of correlations, built from per-day moment sums.

## 3. Rules

- **Discovery pass:** the two-sided bootstrap interval of the partial IC excludes 0 at Bonferroni $\alpha=0.05/(F\times3)$, per asset.
- **Confirmed:** discovery pass, plus a confirmation partial IC with the **same sign** whose one-sided 95% bound excludes 0. A cell needs at least 500 observations and 6 days in each split.
- **Economics (confirmation):** half the difference in mean forward return between the DISCOVERY top and bottom deciles of $X$, signed by the discovery sign (bp). **`covers_cost`** means confirmed and greater than 12 bp.
- **Verdicts:**

  | Verdict | Condition |
  |---|---|
  | `DIRECTIONAL_CANDIDATES_TRADEABLE` | A cell covers cost, and some feature is confirmed with the same sign in ≥2 assets including BTC or ETH |
  | `DIRECTIONAL_CANDIDATES_BELOW_COST` | The cross-asset condition holds, but no cell covers cost |
  | `SINGLE_ASSET_ONLY` | Confirmed cells exist, but none meets the cross-asset condition |
  | `NO_DIRECTIONAL_FEATURE` | No confirmed cells |

## 4. Known limits

- **Sample:** about 3–4 weeks of collector data (from 2026-09-19). Roughly 10 confirmation days, so the bootstrap is coarse.
- **Detectable effect size:** with about 7k bars per asset, the minimum detectable IC is about 0.03–0.04 at 5 min and much larger at 60 min, where windows overlap.
- **Data versions:** L2 instability fields exist only from about Oct 1–2, and the collector's L2 update count changed around Oct 4. Those features have fewer days and may fail the minimum.
- **Overlapping origins:** the forward windows at 15 and 60 min overlap. Day blocks handle within-day dependence only.
- **Status:** an exploratory screen. A confirmed feature would still need a prospective test and a cost-aware strategy test (CS03c style).

## 5. Synthetic validation (tests pass)

- A planted lead feature (IC about 0.10) is confirmed; noise features are not.
- A feature tied only to the current return, in a world with bid-ask bounce, has raw IC of about −0.07 but is not confirmed under partial IC.
- Forward-return gaps, the moment-based correlations against numpy, and the residualisation are all checked.
- **Smoke run:** a lead feature with IC 0.10 gives a long–short of only about 1.2 bp at 5 min, which illustrates how far IC is from cost.

## 6. Expectation (stated in advance)

`DIRECTIONAL_CANDIDATES_BELOW_COST` at best:
- **Microstructure features** (trade-flow imbalance, OBI, microprice) may confirm at 5 min with IC of about 0.03–0.08. That is worth about 1 bp, the same order as CS03c.
- **Options features** (risk reversal, skew changes, ATM IV changes) at 60 min are unlikely to pass with this little data.
