# PSH 002 — Preregistration: volatility × signed-flow interactions and perpetual forecasts

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-17 |
| **Branch** | Predictive_State_HMM |
| **Responds to** | PSH 001 completed results; owner's instruction to proceed with PSH02 |
| **Status** | Frozen before any PSH02 historical run; exploratory because existing historical TEST has already been inspected |

## 1. Scientific question and scope

PSH01 assigned BTC/ETH/SOL/XRP four persistent latent volatility bands, but did **not** beat a nonlinear continuous-volatility model for 60-minute volatility shock prediction. The PSH01 30-minute mean-return perp simulation almost never traded. Separately, CS03c found statistically significant but economically sub-cost **next-five-minute perpetual direction** information in aggressive signed trade flow for BTC/ETH. This is an explicit, owner-approved **cross-branch** experiment to test whether the signed-flow effect varies with causal volatility context, and whether PSH01 HMM filtering adds information above continuously modeled context. A separate perp edge may qualify independently of options. This does not authorize execution or change any prior model.

**No assumption of distinct structural regimes.** The HMM's prior descriptive states need not help prediction; this experiment is a falsification, not a promotion path by default.

## 2. Data and known-time contract

- BTC/ETH primary; SOL/XRP secondary diagnostics. Two-year Binance spot 5-minute closes supply PSH01 emissions; the matched Binance USD-M *same underlying's* one-minute perpetual klines supply signed aggressive flow and subsequent **perpetual** price-return labels.
- Use PSH01's existing 60/20/20 chronological day splits, historical close strictly before **2026-10-17 00:00 UTC**, TRAIN-only de-clocking and variance floor, exact UTC 30-minute origins, 24h memory-readiness. No new download, Deribit live requests, lookahead, or dataset modification.
- Signed flow at origin t is (2 × taker-buy quote volume − total quote volume) / total quote volume, aggregated over the **five complete minutes ending at t**. Incomplete, absent or zero-volume five-minute flow buckets are ineligible.
- Price target at horizon h is log(perp close[t+h] / perp close[t]), built from all complete intervening five-minute perpetual buckets. Reject any missing intermediate bucket or target crossing a chronological split. The 5-minute label is primary; 30-minute label is secondary. Both use perp returns, not spot as an execution proxy.
- **Execution limitation:** a completed t candle cannot generally be used for an instantaneous fill at t's recorded close. Return and P&L calculations here use that price as an optimistic research reference; execution delay, spread, slippage, funding and liquidation margin are not measured. Any apparent positive net P&L requires a separate latency-and-fill validation before a trading claim.

## 3. Prior evidence–gated features

Only three feature signals are admitted:
- **EMA30 and EMA240**: VD12 two-memory feature-family success for future volatility-shock probability (not proof of individual feature statistical independence).
- **Aggressive five-minute signed trade flow**: CS03c next-five-minute BTC/ETH perp directional log-loss gain. The standalone effect was too small to pay 12bp round-trip taker costs, and CS03e found adverse selection for passive fills.

Continuous controls: current close-to-close past spot volatility surprise (`now`), those two memories, fixed sine/cosine UTC clock, and signed flow. Interaction terms are fully predetermined: TRAIN-z-scored flow × TRAIN-z-scored slow EMA, and TRAIN-z-scored flow × (TRAIN-z-scored short EMA − TRAIN-z-scored slow EMA). All scales and outcome coefficients fit on TRAIN. No automatic inclusion of OBI, IV shape, Hurst, basis, jump intensity or hindsight-selected CS04A catalog combinations.

The **PSH01 K=4** Student-t HMM is re-fitted using the same TRAIN-only procedure and seed 2617, with no K re-selection or hindsight state labels; only **filtered** state probabilities are used. No HSMM or explicit-duration overlay is fitted.

## 4. Model ladder and mandatory falsifications

The same fixed origins and targets for every arm:

| ID | Model | Increment |
|---|---|---|
| B0 | `price_vol` | Continuous price/volatility/clock controls |
| B1 | `plus_flow` | B0 + signed flow |
| B2 | `continuous_interaction` | B1 + the two frozen flow×volatility interaction terms |
| B3 | `hmm_additive` | B2 + three HMM filtered state probabilities |
| B4 | `hmm_flow_interaction` | B3 + flow×three filtered HMM probabilities |
| N | `nonlinear_continuous` | Fixed small histogram gradient-boosted classifier on B1's continuous inputs only; no HMM |

TRAIN-only feature standardization and fixed LogisticRegression C=0.25 for B0–B4; fixed HistGradientBoostingClassifier (max 7 leaves, 80 iterations, learning rate 0.05, L2=10) for N.

**Primary target and score:** next-five-minute **perpetual UP conditional on its return being nonzero**, assessed by TEST binary log loss; zero-return origins are excluded identically for all sign models. This is a statistical direction classifier, not an actionable three-way up/down/flat probability. For each comparison define Δ = loss(left) − loss(right); negative is better:

1. B1 − B0: signed flow survives the reduced, matched PSH02 origin set;
2. B2 − B1: volatility × flow improves over a linear flow predictor;
3. B4 − B2: HMM state information improves beyond continuous interactions;
4. B4 − B3: HMM-specific flow interactions improve beyond additive states;
5. B4 − N: HMM improves beyond a flexible nonlinear continuous null.

BTC and ETH × five comparisons = **10 predeclared primary tests**. Use matched whole-UTC-day paired bootstrap (2,000 samples, fixed seed 2618) per asset and comparison. A candidate requires negative Δ and bootstrap upper **99.5th percentile < 0** for every one of the five checks in each **primary asset**; this is a one-sided Bonferroni 5% familywise policy. Also report nominal 95%-level descriptive intervals and exact sample coverage (at least 30 TEST days). **No relaxed criterion** if flow B1 fails; subsequent comparisons remain descriptive. SOL/XRP, the 30-minute horizon and trading-cost scenarios cannot rescue a failed primary gate. Repeated historical reuse makes even a gate pass exploratory, never prospective confirmation.

## 5. Perpetual economics separate from sign score

On each eligible origin and each horizon, independently fit TRAIN-only Ridge to **actual future signed perpetual return in basis points**, using matched B0–B4 features. Fixed alpha=100. Record (a) no position unless |predicted gross bp| > assumed **12 bp round-trip cost** and (b) a separate TRAIN-defined absolute-prediction 90th-percentile activity scenario. For each arm/horizon report number of trades, gross signed bp and net signed bp after the assumption. Fixed-origin decisions are nonoverlapping at 30-minute anchors, although the primary 5m holding window is shorter. Scenarios are research-only estimates (starting at the already completed origin close); **no measured spreads, actual futures fills or funding**. Report zero trades honestly; a one-trade success is not an edge.

No options P&L is tested in PSH02; a separate point-in-time bid/ask quote analysis is required. The project's approved research mandate is both perpetual and options opportunities, each with distinct out-of-sample and cost requirements.

## 6. Deliverables and integrity

Read-only engine: `src/data_engineering/modeling/predictive_state_flow.py`; tests: `tests/test_predictive_state_flow.py`; notebook `notebooks/Predictive_State_HMM/02_volatility_signed_flow.ipynb`. Numbered outputs `notebooks/Predictive_State_HMM/02/`: `02_forecast_scores.csv`, `02_paired_day_comparisons.csv`, `02_perp_cost_scenarios.csv`, `02_coverage.csv`, `02_decision.json`, `02_manifest.json`. No origin-level market data exported. Do not edit PSH01 assets, existing HSMM/VD19 frozen specs, collectors or production DB.

## Questions for Claude

1. Do the B2 vs B1 and B4 vs B2 comparisons sufficiently isolate structural-state information from deterministic nonlinear variance responses?
2. Are q99 onset context (PSH01) and signed-return context (PSH02) being separated correctly, or does event-conditioned selection inadvertently reappear?
3. Before any live perp claim, which timestamped maker/taker fee, latency, spread and funding fields should be joined to the historical fill simulation?

## Decisions needed from Ivan

PSH02 is authorized to implement and execute as exploratory research. After results, decide whether to terminate this branch, continue to a genuinely new prospective sample, or design a more execution-realistic independent perp evaluation. Neither success nor failure here alters the frozen HSMM and VD19 confirmations.
