# Liquidity Absorption 032 — LA10 unselected observable-reference population

**Date:** 2026-10-21  
**Status:** Protocol and synthetic-tested implementation; real original historical replay pending local Binance USD-M archive.  
**Module:** `notebooks/Liquidity_Absorption/10_unselected_level_universe.ipynb`  
**Version:** `LA-LEVEL-UNIVERSE-10-0.1.0`

## Motivation

Module 09 correctly reconstructed the causal sequence **level source → confirmed availability → first touch → second touch → subsequent market path** for all 452 original A02 events. However, those events **were selected because they had a valid second touch**. Its conditional TARGET/STOP/TIMEOUT summaries cannot estimate the frequency with which a new observable level gets no valid rejection, only one rejection or a valid two-touch signal. That missing denominator is the explicit target of LA10.

This study **does not** optimize the signal, execution or volatility overlay. It builds the full population of **reference episodes** that the exact original A02 engine would actually see: four streams per symbol (`prior_day` support/resistance, `pivot_5m` support/resistance). A last-confirmed-pivot reference may be replaced or age out (720 minutes). LA10 counts a new level episode when the stream's `(source_time, price)` changes after confirmation. This is the observable *active* A02 reference population, **not every conceivable latent support/resistance line nor all swing extrema simultaneously**.

## Causal clock and baseline identity

- BTCUSDT and ETHUSDT Binance USD-M **one-minute completed candles**, July 1–Sep 30, 2026 plus original 3-day warmup. Original `02_feature_contract.json` is used directly to instantiate the unchanged A02 configuration. One-minute raw Parquet is read once per asset.
- Pivot source candle is not known as a confirmed pivot until **10 minutes later** (two right five-minute bars); previous UTC-day high/low becomes available **next UTC midnight**. The price/level cannot be consulted earlier than availability. First-touch closing exactly on availability is flagged as intraminute-ambiguous.
- `replay_level_episodes` keeps original A02 state machine semantics for wick size, rejection distance, invalidation, required intervening departure, minimum 3-minute separation, 90-minute maximum retest gap, 30-minute post-signal cooldown, and reference replacement. Its original second-touch detections must reproduce **every frozen A02 452-event signal** using symbol, side, level family/source/price, first-touch close, second-touch close, and signal close. **Abort without publishing results on ANY missing/extra event.** No A03b/A03d result labels enter the first/second-touch incidence study.
- Every observed reference receives `NO_VALID_TOUCH`, `FIRST_TOUCH_NO_SIGNAL` or `SECOND_TOUCH_SIGNAL` classification, plus separately reported invalidations, old-reference replacements, visible exposure, and censoring. References can generate multiple touch spells or signals. The unit of analysis for these classes is the **reference episode**, not the trade.

## Fixed event clocks and denominators

Two clearly distinct clock origins:

1. **From level availability to first touch:** fixed 5, 15, 30, 60, 90-minute horizons. A reference episode replaced or expired before an uneventful horizon is censored for this simple descriptive fraction, not forcibly labeled as a clean no-touch. If a first touch has occurred by that horizon it remains a known occurrence even when the reference later expires. Report both original episode denominator and *assessable* count by horizon. This is **not a nonparametric competing-risk incidence estimator** and should not be presented as such.
2. **From a first valid touch to a second-touch A02 confirmation:** an individual *first-touch spell* starts after a candidate first-rejection candle closes. It may end in an A02 signal, invalidation, 90-minute retest expiration, reference replacement, or right censor at the archive boundary. Replacement/expiry/invalidation are observed **competing ends**; only archive-end-open spells are removed from an otherwise assessable fraction. The full spell-level file permits a later proper survival/hazard model and exact risk-set analysis. Same-close-as-pivot-confirmation first touches are flagged, and only strictly later-than-availability first touches contribute to the simple second-horizon incidence table.

**Controls here are unselected level episodes, NOT executed contrarian trades.** A level never touched cannot be assigned a P&L, entry time, or profit target. Retrospective classification is used only to describe the population, not as a feature available at level confirmation.

## Deliverables

- `src/data_engineering/modeling/liquidity_absorption_stage_a10.py` — no changes to original A02.
- `tests/test_liquidity_absorption_stage_a10.py` and `tests/test_liquidity_absorption_stage_a10_notebook.py` — detector replay equivalence, invalidation/cooldown/expiry, temporal release, censoring, empty cases and notebook syntax.
- `notebooks/Liquidity_Absorption/10_unselected_level_universe.ipynb` — executable `deribit-data` notebook.
- Local results written to `notebooks/Liquidity_Absorption/10/`: `10_decision.json`, `10_level_episodes.csv`, `10_episodes_by_type.csv`, `10_first_touch_spells.csv`, `10_first_touch_incidence.csv`, `10_second_touch_incidence.csv`, `10_replayed_A02_signals.csv`, `10_episode_classes.png`, and `10_encounter_incidence.png`.

## Interpretation gates

- The available population is mechanically defined by the exact A02 algorithm; signal incidence is not the universe of all possible levels in financial markets. “No valid touch” means **no touch meeting A02's rejection-wick criteria**, not that the market never traded near that price.
- A reference episode can be short because it is replaced by a newer pivot. Per-horizon proportions have differing *assessable* denominators and cannot simply be compared as comparable probabilities of survival. This diagnostic precedes formal competing-risks estimation; no confidence interval or alpha claim yet.
- Every absolute UTC close is causal at candle completion, but intraminute wick/touch order is unknown. The 16 original 09 cases with first touch closing at the exact confirmation timestamp must be flagged/excluded from any strict acting-after-confirmation subset.
- Historical 2026 original data and Claude's extended Deribit 2024–2026 research were previously inspected; July–September findings are not a fresh holdout. Binance A02 levels cannot be treated as identical to Deribit execution-book levels. This notebook does not use the new Deribit archive, because its sole objective is **an exact same-venue A02 reference denominator**.
- No order-book depth, queue fills, funding, strategy edge, stop/target adjustment or portfolio capital assumptions are introduced.

## Next planned comparison, conditioned on LA10 audit

Once the observed population and exact frozen-A02 replay are confirmed, define a **time-at-risk, availability-centered comparison** with symbol, level type, direction, volatility, session and age/time exposure controlled. Compare `first-touch risk`, then conditional `second-touch after first`, keeping invalidation/replacement competing outcomes. Only after preregistration consider whether observable first-touch features separate valid second encounters from non-encounters on a genuinely unseen later period. Pair this with frozen stop/target geometry; do not optimize a filter on the repeatedly inspected sample.

## Execution

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_liquidity_absorption_stage_a10*.py" -v
jupyter lab
```

Open `notebooks/Liquidity_Absorption/10_unselected_level_universe.ipynb` and **Restart & Run All**, then commit the executed notebook plus lightweight `10/` outputs to GitHub `main`. **An exact replay mismatch means stop, investigate, and do not interpret levels or incidences.**
