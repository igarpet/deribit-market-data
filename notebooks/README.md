# Notebooks

One folder per research branch, matching `discussions/`. Notebooks are numbered per folder from `01`; each notebook writes its results (CSV/JSON) into a subfolder named after its number (`06_volatility_har.ipynb` → `06/06_*.csv`), so the branch folder holds only notebooks and the README.

## Research-branch independence

Research branches are independent experiments. They may share the common data-engineering/database layer, but **must not silently reuse another branch's research outputs, fitted models, frozen specs, feature selections, thresholds, state definitions, or empirical conclusions as assumptions**. Any cross-branch import requires Ivan's explicit approval and must be documented. A new branch otherwise starts from the shared base data and establishes its own assumptions internally.

| Folder | Content |
|---|---|
| [initial_research](./initial_research/) | Foundational analysis used by all branches: feature inventory, market/surface distributions, microstructure, cross-feature relationships, clock/expiry effects, displacement tests. |
| [HSMM](./HSMM/) | Regime feature contract, exploratory HSMM/HMM, volatility HAR benchmark and frozen spec, volatility-state HMM, regime null test. |
| [Volatility_Risk_Premium](./Volatility_Risk_Premium/) | Option pricing audit, forecast vs implied variance, executable option P&L. |
| [Volatility_Regression](./Volatility_Regression/) | Native-frequency next-5m volatility OLS with broad market/option features and backward elimination. |
| [Pairwise_Feature_Regression](./Pairwise_Feature_Regression/) | v1 descriptive directed single-predictor OLS matrices: raw beta, in-sample residual RMSE, pairwise coverage and heatmaps by asset. |
| [Volatility_Dynamics](./Volatility_Dynamics/) | Independent return-only volatility research; Module 02 studies native 5m squared returns directly, including tails, clock structure, persistence, transitions, run duration and cross-asset synchronization. Modules 02b–06b re-run the q99 precursor / HMM line with nulls, repaired volatility conditioning and pre-registered gates (VD 010). Modules 08–10c: jump point process, compound-Poisson jumps, physical jump model, model-free and long-history benchmarks vs 0DTE wings (VD 017–023). |
| [Hurst_Regimes](./Hurst_Regimes/) | Session Hurst exponent of 5m paths: variation, persistence and relevance gates against a volatility-modulated fGn null, before any HMM on $H$ (HR 001). |
| [Breakout_Trend](./Breakout_Trend/) | Third-party hourly breakout strategy re-implemented exactly; perpetual leg net of Deribit costs vs long ATM options on the same signals; dev/test split, coin-flip null, per-asset equity curves (BT 001). |
| [VWAP_Events](./VWAP_Events/) | Does a touch of the 08:00 UTC session or 4-hour VWAP precede a violent move beyond offset and random-phase placebo touches? Binance USD-M 1m klines, time-of-day-normalised realised vol, Bonferroni gates on the test window (VWAP 001). **Concluded (VWAP 002): VWAP touches not special.** |
| [VWAP_Drift](./VWAP_Drift/) | Conditional drift toward the session/4H VWAP with displacement in volatility-engine units (frozen VD18 HAR_4), by vol regime and recent jumps, at 15/60/240 min and to the 08:00 expiry; placebo anchors, DEV selection, TEST gates; prerequisite for a directional option toward VWAP (VWD 001). |
| [Regime_Structure](./Regime_Structure/) | Do range / displacement / noise / shock states persist beyond volatility clustering (sign-randomised null), and do Ivan's cycle transitions appear? Hourly labels on Binance 1m, no VWAP (RS 001). |
| [Signed_Jumps](./Signed_Jumps/) | Do 1-minute jumps self-excite, does the parent jump's sign or pre-jump truncated skewness predict the next jump's sign? Volatility engine on bipower + signed Hawkes, Binance spot 1m BTC/ETH/SOL/XRP (SJ 001). |
| [Multi_Scale_HMM](./Multi_Scale_HMM/) | Can a Student-t HMM on 1h/4h/1D bars built from 1-minute perpetual data (intrabar emissions: volatility, semivariance, efficiency, wicks, jumps, flow) find momentum/reversal states that predict the next bar net of costs? Shared-sign minute null, drift benchmark (MSH 001). |
| [Funding_Carry](./Funding_Carry/) | Delta-neutral long spot / short USD-M perpetual funding harvest on Binance (PF02 universe); trailing-funding hysteresis vs always-in; net of costs vs a 4% cash hurdle (FC 001). |
| [Jump_Fade](./Jump_Fade/) | Passive ±1.5σ bracket at each 5m boundary, take-profit at the pre-move price, 3σ stop, 60-min time stop; 1m perp simulation; sign-flip null; Deribit costs (JF 001). |

## Output-size policy (from 2026-10-09)

Keep research results reproducible **without routinely committing massive exploratory intermediates**. Default notebooks to the smallest reviewable outputs: a short run manifest, essential coverage/quality diagnostics, and compact tables of the comparisons needed for the research question. Avoid repeated per-asset copies of an aggregate table, full n×n×time coefficient stacks, redundant correlation/R²/normalized-error matrices, and separate PNG exports if the notebook can display them inline. Use float precision appropriate to the analysis and compressed binary files only when dense full-resolution data are scientifically indispensable.

**Full matrix arrays, large diagnostic dumps, and high-resolution plots require explicit opt-in.** Generate on demand locally, or export compact selected relationships. Document output-size expectations and retention before future research modules. Do **not** silently delete previously committed evidence, frozen specifications, or historical research results; request approval before repository-history cleanup or deletion.

The first execution of Pairwise Feature Regression 02 saved ~299 MB, including ~298 MB of optional full matrix archives; the notebook has since been switched to compact exports by default. This changes *persistence only*, not the 288-by-asset computation.

## Mapping from the old global numbers (renumbered 2026-10-08)

| Old | New |
|---|---|
| `01`–`06` (root) | `initial_research/01`–`06` (same names) |
| `HSMM/06b_displacement_tests` | `initial_research/07_displacement_tests` |
| `07_regime_feature_contract` | `HSMM/01_regime_feature_contract` |
| `08_hsmm_exploratory` | `HSMM/02_hsmm_exploratory` |
| `09_factor_augmented_hsmm` | `HSMM/03_factor_augmented_hsmm` |
| `10_joint_sdf_hmm` | `HSMM/04_joint_sdf_hmm` |
| `11_full_information_price_model` | `HSMM/05_full_information_price_model` |
| `12_volatility_har` + `12_*` outputs (Module 12) | `HSMM/06_volatility_har` + `06_*` (incl. `06_frozen_spec_v02.json`) |
| `13_volatility_hmm_prototype` + `13_*` (Module 13) | `HSMM/07_volatility_hmm_prototype` + `07_*` |
| `14_btc_s2_har_garch_null` + `14_*` (Module 14) | `HSMM/08_btc_s2_har_garch_null` + `08_*` |
| `15_vrp_pricing_audit` + `15_*` | `Volatility_Risk_Premium/01_vrp_pricing_audit` + `01_*` |

Only file names and the paths in notebook code changed. Notebook prose, saved cell outputs, `src/` docstrings and discussion files may still use the old numbers. A note at the top of each renumbered notebook gives its former number.
