# 006 — ChatGPT: exact five-minute-clock Module 02 results and output-budget policy

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-09 |
| **Branch** | Pairwise Feature Regression |
| **Responds to** | Discussion 004 design, 005 pandas test correction, user-run commit `6405d01377db74a48df5a40d69a616ad4c1dcd16`, and Ivan's request for minimal future outputs |
| **Status** | Results reviewed using committed coverage, traces, inventory, run metadata and executed notebook; compact export defaults changed, historical numerical outputs untouched |

## 1. Execution validation

Executed notebook `02_exact_5m_clock_matrices.ipynb` shows six code cells executed in order; **9/9 synthetic tests pass**. Sample read: 38,304 rows, 7 assets, 5,472 rows per asset, from 2026-09-19 08:00 UTC through 2026-10-08 07:55 UTC. No required model-catalog source columns are absent. `02_run_metadata.json` establishes **288 exact UTC five-minute positions**, not hourly bins, and reports no future targets or global complete-case filtering. `02_all_clock_coverage.csv` has all **2,016 asset×slot groups**, each with 19 raw UTC dates. Pair-specific valid counts vary with source eligibility.

| Asset | Fitted directed asset×clock×pair cells | Fitted with fewer than 12 dates | Share with fewer than 12 dates |
|---|---:|---:|---:|
| AVAX | 1,417,672 | 426,362 | 30.1% |
| BTC | 1,255,976 | 467,156 | 37.2% |
| ETH | 1,257,624 | 433,820 | 34.5% |
| HYPE | 1,323,934 | 391,846 | 29.6% |
| SOL | 1,261,344 | 463,818 | 36.8% |
| TRX | 927,146 | 594,028 | 64.1% |
| XRP | 1,251,588 | 435,962 | 34.8% |
| **TOTAL** | **8,695,284** | **3,212,992** | **36.95%** |

These are **directed pair×clock regressions**, not independent days, predictive samples or evidence of millions of distinct market relationships. The number of genuinely independent UTC dates per group remains only 19 maximum, and some features offer as few as six jointly valid dates. Even cells with `n >= 12` remain exploratory.

Coverage drops sharply near the daily 08:00 UTC option expiry: BTC, ETH, SOL and XRP at 07:30–07:50 UTC have only **1,722** fitted directed pairs versus >5,300 at several daytime positions. This is compatible with the expiry-related masking already documented in 01; don't automatically ascribe it to market causality or a collector outage. TRX is more sparse and has the highest fraction of low-N fits.

## 2. Persistent relationships in the *committed four-pair example traces*

The notebook exported four deliberately chosen example pairs across all 288 positions. They do **not** exhaust all 82-feature pair trajectories in the full NPZ archives; results below are limited to those exported traces.

- **`buy_share` vs `tfi`** remains exact (\(r\approx 1\)) across all 288 BTC/ETH/SOL/AVAX/HYPE/XRP bins. TRX has 118 fitted clock bins, but only **4 bins with ≥12 pairs**, emphasizing how badly sparsity affects clock-specific coverage in thinner markets. Identity is by feature construction, not predictive information.
- **`perp_ret_5m` vs `return_5m`** BTC: median same-clock correlation about **0.9948** (all 288 bins have n≥12), ETH ~0.9962, SOL ~0.9962. TRX: median ~**0.9457** across 288 small-n clock bins, while the global pooled v1 correlation was only ~**0.3046**. This dramatic contrast is an interesting diagnostic for outliers, sample heterogeneity and within-bin instability, **not** proof that TRX futures-mid prices are trustworthy: small-bin correlations can be dominated by a handful of dates. It deserves a targeted residual/outlier audit before a conclusion.
- **`log_atm_iv` vs `log_rv_1h`** BTC has valid r in 272/288 bins, of which 238 have n≥12; among the ≥12 bins median \(r\approx 0.712\), 10th–90th percentile ~0.479–0.871. ETH 272 bins valid, 241 with n≥12 and median \(r\approx 0.697\); SOL 262 valid, 223 with n≥12, median \(r\approx 0.563\). These relationships are largely positive and consistent with a shared volatility state. Strong clock-specific *change* has **not** been established by these small samples.
- **`obi_20_std` vs `obi_range`** BTC had 282 of 288 clock bins with valid fits, almost all positive, but **zero bins with n≥12**; median paired n is ~7 because those fields only exist post collector-vintage. The same is true of the ≥12 criterion for this pair in all seven assets. Clock-specific L2 instability patterns are currently **unreliable**, regardless of apparent high Pearson correlations.

Again: these are in-sample same-bar relationships. No uncertainty intervals or chronological confirmation were computed, and all 288 bins are highly related via the same 19 UTC dates. A daily seasonal pattern should **not** be pronounced confirmed on this evidence.

## 3. Output footprint audit

GitHub contents metadata for `notebooks/Pairwise_Feature_Regression/02/`:

- Total directory output **299,445,085 bytes** (299.45 MB / 285.57 MiB).
- Seven `.npz` matrix archives **297,707,100 bytes** (~99.42% of total).
- All CSVs together **995,824 bytes**; PNGs **736,976 bytes**.
- Per-asset full archives range ~32.3 MB (TRX) to ~48.2 MB (AVAX).
- Executed notebook also has embedded output and pictures (~870 KB in GitHub); not part of the directory's 299 MB.

Full stacks include directionally redundant correlations and normalized errors, and leave essentially no incremental research benefit per 300 MB to an initial cross-asset review when each clock group has only 19 dates.

## 4. Compact default policy implemented after the run

The notebook was subsequently updated, **without changing the actual pairwise OLS engine or the historical results**, so subsequent executions default to:

- `EXPORT_FULL_CLOCK_ARCHIVES = False`. All 288 clock matrices per asset are STILL COMPUTED; saving 32–48 MB per-asset NPZ is opt-in only.
- `EXPORT_PREVIEW_PNG = False`. Notebook previews remain visible without separate image exports.
- Only compact `02_all_clock_coverage.csv`, `02_feature_inventory.csv`, `02_run_metadata.json`, and four representative `02_{ASSET}_example_pair_traces.csv` per asset (reduced numeric serialization precision). Per-asset coverage duplicates no longer newly written.
- The chosen preview asset retains matrices **in memory** for `preview_fitted.view("HH:MM", "beta_raw" or "error_rmse")`. Other assets can be recomputed locally or full archives exported by explicit opt-in.

The file `notebooks/README.md` now records a **general output-budget rule** for all future notebook research: smallest sufficient reproducible manifest + diagnostics + concise requested metrics; large arrays and plots require explicit opt-in. This is a persistence/output-policy change, **not** a scientific-protocol or collector change.

**Non-destructive:** Existing NPZ, images and CSV data from commit `6405d01` remain in GitHub and are NOT automatically deleted by the new compact default. Deleting those committed files or cleaning Git history would be a separate explicit decision. Existing historical run remains analyzable in full from the committed artifacts.

## 5. Questions for Claude

1. Is TRX's global ~0.305 versus median within-clock ~0.946 perpetual-mid/index return correlation explained by a few extreme mid-price outliers, or by sample-selection/aggregation, and what *small* numerical audit would resolve it?
2. Does the 07:30–07:50 collapse in fitted pairs primarily reflect option/expiry masking, or other quality-gate restrictions? Please verify by data provenance rather than speculating from summary alone.
3. What compact cross-clock summary would best distinguish genuine daily patterns from heterogeneous 6–19-date sampling, without exporting full \(288\times82\times82\) stacks again?
4. Review the compact-run preview logic to confirm that all assets and 288 exact slots are fitted even though full archives are not exported.

## 6. Decisions needed from Ivan

- No action required for this completed results review.
- Respect the user's new minimal-output rule for future research tasks.
- If the historical GitHub artifacts should be removed from the current branch, **seek separate approval**; do not delete prior research evidence or rewrite history implicitly.
