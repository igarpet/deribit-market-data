# Liquidity Absorption 033 — LA10 unselected observable-level population results review

**Date:** 2026-10-21  
**Scope:** executed `10_unselected_level_universe.ipynb`, `10/{10_decision.json,10_episodes_by_type.csv,10_first_touch_spells.csv,10_first_touch_incidence.csv,10_second_touch_incidence.csv,10_replayed_A02_signals.csv}`, plus comparison to LA09 `09_decision.json` and review 031.  
**Status:** retrospective descriptive process analysis; **no independently validated signal, causal absorption explanation, or strategy edge**.  
**Changes:** this review document only; frozen A02 detector, A03b/A03d simulations, local archive, and A09/A10 notebooks untouched.

## 1. Successful completion and exact identity replay

The executed notebook records **16 LA10 engine tests + 1 notebook test, all passing**, with no saved notebook error. `10_decision.json` reports **452 original A02 signals exactly reproduced (multiset identity, no missing/extra)**, **15,045** level-episode rows including warmup, and **14,577** newly observable primary-period episodes. This is the same original BTCUSDT/ETHUSDT Binance USD-M, July–September 2026 two-touch selection logic; it does not use the new Deribit tick download.

The reference-episode universe is one last-active A02 level stream per symbol/direction/family: `pivot_5m` support/resistance and `prior_day` support/resistance. The unit is a **reference episode**. The exact replay can produce multiple signals from one episode.

| Primary episode class | Count | Original denominator |
|---|---:|---:|
| `NO_VALID_TOUCH` | 11,137 | 76.40% |
| `FIRST_TOUCH_NO_SIGNAL` | 3,015 | 20.68% |
| `SECOND_TOUCH_SIGNAL` | 425 | 2.92% |
| **Total newly observable episodes** | **14,577** | **100.00%** |

The 425 signal-containing episodes produce **452** A02 signals. A `NO_VALID_TOUCH` episode means no wick/rejection candle passing exact A02 criteria, NOT that price never visited the level.

Family-specific episode counts: pivot_5m **14,209** total (**10,894** no valid touch, **2,942** first-touch-only, **373** produced a signal); prior_day **368** total (**243**, **73**, **52** respectively). **Do not compare raw pivot and prior-day signal percentages as evidence of superiority:** typical observable durations are ~30–35 minutes for pivots vs 1,440 minutes for prior-day levels, and a prior-day level can support multiple touch spells.

## 2. Strict first-touch-spell outcomes

`10_first_touch_spells.csv` contains **3,967** rows including warmup. For spells first touching during the primary date range and **strictly after level availability**, n=**3,744**. End reasons:

| Strict first-touch spell terminal event | Count |
|---|---:|
| Replaced by newer reference | 1,811 |
| Invalidated | 1,435 |
| Exact A02 second-touch signal | 436 |
| Retest-gap expired | 61 |
| Right-censored at archive end | 1 |
| **Total** | **3,744** |

Thus **436/3,743 assessable strict spells = 11.65%** recorded an exact second-touch signal by the fixed 90-minute retest horizon. This is an observed detector-defined incidence among first-touch spells, not a future trade success rate. There is **one** observed end-of-archive right censor. Replacements, invalidations and retest expiries are competing observed ends, not randomly missing data.

The distinction between **452** original signals and **436** second-touch strict-primary spells is not unexplained missing events: timestamp equality of the level becoming available and the first-touch candle close is ambiguous at the intraminute level. LA09 flagged **16** same-close first touches among the original 452; LA10's broader primary *episode* universe flags **108** such first-touch records. Do not conflate the 108 across all levels with 108 excluded original selected signals.

## 3. The long-horizon first-touch percentage is a conditional risk-set statistic

The `10_first_touch_incidence.csv` metric is calculated as observed early first touches divided by first touches **plus episodes remaining visible to the horizon**. Episodes replaced or expired without a first touch before that horizon are excluded, so the assessable denominator becomes strongly selective as the horizon increases.

Aggregated **pivot_5m** episodes across both assets and both sides:

| Availability horizon | New original episodes | First touched by h | Count counted as assessable | Replaced/ended early without first touch | Reported assessed ratio |
|---|---:|---:|---:|---:|---:|
| 15 min | 14,209 | 1,898 | 13,040 | 1,169 | 14.56% |
| 90 min | 14,209 | 3,304 | 3,542 | 10,667 | **93.28%** |

That **93.28% IS NOT P(first touch within 90 minutes | new level)**. The observed touch count over all new pivot episodes is **3,304/14,209 = 23.25%**. The vast majority of no-touch episodes have left the observable reference state by minute 90. Replacements and expiries are competing terminations, not independent random censoring. Future analysis should estimate properly normalized cause-specific/competing transition probabilities and report the full original episode denominator, remaining-at-risk counts, and competing-end incidences. Do not train thresholds on the misleading 93% figure.

## 4. Relationship to LA09 full lifecycle anatomy

LA09 (`LA-LIFECYCLE-09-0.1.0`) reconstructed four chronological milestones: level source, confirmed availability, first touch and second touch/signal; 452 selected signal candidates yielded **3,616** stage records, **83,168** minute landmark records, no missing landmark minute bars, and 16 same-close availability/first-touch cautions.

For the **original A03b fixed 1:1 outcome cohort** (116 TARGET, 112 STOP, 185 TIMEOUT, 37 without structural target and 2 censored), the median `FIRST_TO_SECOND` level-overlap share was approximately **0.423** for eventual targets vs **0.298** for stops, and directional aggressive quote imbalance **-0.028** vs **+0.025**, with median first-to-second elapsed times both **7 minutes**. These are selected-cohort retrospective differences under the original detector and 1m OHLC proxy; they cannot prove passive absorption. LA10 now supplies the **unselected first-touch and newly available level denominators** needed to decide whether those anatomy patterns are discriminative beyond selection.

Do not combine LA09 A03b outcome rates with LA04/A05a A03d volatility-protected outcomes (119 TARGET, 202 STOP, 93 TIMEOUT). These exit policies differ.

## 5. Important restrictions and next work

- **Causal clock:** 5m pivot becomes observable only after 2 right 5m confirmation bars (10-minute delay); prior-day reference at the next UTC midnight. Never apply a future-confirmed pivot retroactively as if tradable. First touch closing simultaneously with pivot availability is intraminute ambiguous.
- **Event unit:** level episode and first-touch spell are not equivalent to independently placed trades. Different level references and signals can overlap; cluster uncertainty by time and asset/reference.
- **Competing ends:** reference replacement, invalidation and expiry carry information. Show cause-specific event counts and remaining risk sets; avoid incorrectly independent-censoring reference replacements.
- **Venue:** LA09/LA10 use Binance USDT-M 1m candles, not the Deribit futures trade history. This protects the frozen same-venue A02 identity replay but does NOT constitute millisecond-level absorption or queue evidence.
- **Repeatedly inspected sample:** July–September 2026 is not a fresh OOS validation. No P&L or prospective post-cost improvement follows from these incidence tables.
- **Predeclared next comparison:** at a strictly observable first-touch candle close, compare spells which later become valid second-touch signals against spells that invalidate, expire or see a reference replacement. Use only `available_at`, pre- and first-touch candle information, past quote flow/volatility, level age, existing reference family and price geometry. Avoid any information after first touch. Separate pivot versus prior-day, BTC versus ETH, direction, and genuine exposure duration. Use a cause-specific or multi-state time-to-event design with proper competing-end handling. Freeze hypotheses before touching later genuinely unseen data.

**Verdict:** Module 10 is technically complete and matches all 452 frozen A02 signals, filling the major denominator gap from LA09. The 425/14,577 episode classification and 436/3,743 strict first-spell signal incidence are useful descriptions of how uncommon the complete two-touch process is. The next defensible advance is an **ex-ante first-touch quality / transition-hazard comparison against unselected controls**, rather than further outcome-only medians or same-sample trading optimizations.
