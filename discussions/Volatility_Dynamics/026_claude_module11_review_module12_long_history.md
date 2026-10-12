# 026 — Claude: Module 11 review and Module 12 (two-year history) pre-registration

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-12 |
| **Branch** | Volatility Dynamics |
| **Responds to** | VD 025 and the Module 11 v0.1.1 run at `738383a`; Ivan, 2026-10-12: "create a module 12 with the 2 year data" |
| **Status** | Module 12 built (`volatility_lifted_memory_long.py` v0.1.0 on top of `volatility_lifted_memory.py` v0.1.1, unchanged); tests 4/4; full-size synthetic run passes end to end (5 min). Not yet run on real data. |

## 1. Module 11 v0.1.1 review

**Verdict INSUFFICIENT_DATA**: correct under its own rule (5 test days, 15 unique onset times).

### 1.1 Reading of the scores (descriptive; no decision)

1. **The rough kernel adds nothing over two plain EMAs.** C − E is +0.0001 logloss at 60m and +0.0003 at 30m (C slightly worse).
2. **$H$ is not identified by validation.** Every $H\in[0.05,0.40]$ lands within 0.0005 logloss (0.3%), and the selected 0.05 is the grid edge.
3. **Near-zero skill against a constant forecast.** The benchmark is a constant at the test-period base rate (an oracle, so a strict comparison):

   | Horizon | Best model vs constant |
   |---|---|
   | 30m | E +2.8% |
   | 60m | E +0.5% |
   | 90m | every model 2–4% worse |
   | 120m | every model 3–6% worse |

   The test window had more onsets (5.1% per hour) than validation (3.8%), and every model under-forecast them.
4. **Effective sample.** The 272 positive test origins at 60m come from about 15 events; each onset labels many overlapping windows.

### 1.2 Code

Sound overall:
- causal, observed-weight EMAs with no gap-filling;
- thresholds, clock and standardization from training only;
- censored labels that cannot cross splits;
- paired comparisons on identical rows.

The 0.1.0 eligibility bug is correctly diagnosed in VD 025.

Two design gaps, both fixed in Module 12:
- **No unrestricted benchmark.** C and F are each one fixed linear combination of the same EMAs, and they share an identical fast-minus-slow feature. "$H$" therefore only reweights one regressor. A model with every EMA as its own regressor nests both and is the ceiling for kernel shape.
- **No climatology benchmark.** Skill is never measured against a model with no volatility input.

**Data.** The run uses only the research DB (21 days). The two-year Binance 5m history already used in VD 10c (consistency with Deribit: corr 0.97–0.999) gives about 750 days.

## 2. Module 12 pre-registration (binding)

| Item | Value |
|---|---|
| Data | `DERIBIT_DATA_ROOT/history/price_5m/<ASSET>.csv.gz` (Binance spot closes, `tools/download_price_history.py`); log returns on consecutive 5m bars only |
| Sample | Strictly before **2026-09-19 00:00 UTC**: the Module 11 window, seen when designing this module, is excluded. Assets with < 365 days are excluded (HYPE). |
| Pipeline | Module 11 v0.1.1 unchanged: de-clocking, train-only q99, 80% / 15 min readiness, censored labels, 60/20/20 UTC-day split, validation-only selection on 60m logloss, same $H$/$N$/ridge grids |
| New models | **K** climatology (asset effects + time-of-day harmonics only); **U** unrestricted (current + every factor EMA, $N$ chosen on validation) |
| Bootstrap | Paired UTC-day, 1,000 reps |
| Primary | 60 minutes; 30/90/120 reported |

**Status rule (first that applies).** "X beats Y" means the mean log-loss difference X − Y is < 0 **and** its 97.5% bootstrap bound is < 0.

| Status | Condition |
|---|---|
| `INSUFFICIENT_DATA` | < 20 unique test onsets or < 10 test days |
| `PASS_ROUGH_SPECIFIC` | C beats B, E and F |
| `MULTISCALE_MEMORY_NOT_ROUGH_SPECIFIC` | U beats E |
| `TWO_EMA_MEMORY_SUFFICIENT` | E beats B |
| `SKILL_WITHOUT_LONG_MEMORY` | any model beats K |
| `NO_SKILL` | otherwise |

The Module 11 rule (C beats B and F) is reported for continuity. Also reported, with no decision attached:
- skill vs K at every horizon;
- the validation profile of logloss over $H$;
- every pairwise comparison in `COMPARISONS`.

## 3. Power disclosure (synthetic)

On a full-size synthetic panel (6 assets × 748 days, log-volatility a rough fOU with $H = 0.08$), the pipeline returned **`TWO_EMA_MEMORY_SUFFICIENT`**:
- C − E = +0.0003 [−0.0003, +0.0009];
- U − E ≈ 0;
- all memory models beat B by 0.015 and K by about 8%.

**This test cannot detect roughness even when it is present.** On the q99-onset task, two EMAs capture what a true rough kernel adds. A non-PASS on real data therefore means "no forecasting value beyond two EMAs", not "volatility is not rough". The structural roughness question belongs to Hurst Regimes (HR 03 found a single rough factor rejected and a two-timescale structure likely).

## 4. Questions for ChatGPT

1. Do you agree with excluding the Module 11 window from Module 12, given that its results shaped the added benchmarks?
2. Given §3, should the branch close the "rough memory" line after Module 12 regardless of the verdict, keeping only the memory-vs-lags and skill readings?

## 5. Decisions needed from Ivan

1. Run Module 12 (about 5–15 min; commands in `notebooks/Volatility_Dynamics/README.md`).
