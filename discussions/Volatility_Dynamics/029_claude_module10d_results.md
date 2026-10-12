# 029 — Claude: Module 10d results — the wing line closes

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-12 |
| **Branch** | Volatility Dynamics |
| **Responds to** | VD 028 (pre-registration); Ivan's run (commit 563cd42) |
| **Status** | Binding: `NOT_CHEAP` / `NOT_RICH`. Under VD 028 §2, the wing line is **closed** in this branch. |

## 1. Run

| Item | Value |
|---|---|
| Quotes | 14,493 |
| Pool origins | 1.03 M (two-year pool, last session 2026-09-19) |
| B | 1,000 |
| Consistency gate | Passes on all 7 assets (corr ≥ 0.974; SD ratio 0.96–1.03) |
| Low tercile | 6,375 quotes, 19 sessions |
| High tercile | 2,187 quotes, 15 sessions |

## 2. Pre-registered decisions

| Decision | $M$ (log quote ÷ fair) | 95% CI | one-sided p | Holm | Verdict |
|---|---|---|---|---|---|
| D_long: calm tercile, ask net of fee | **+0.54** | [+0.23, +1.21] | 0.999 | no | `NOT_CHEAP` |
| D_short: volatile tercile, bid net of fee | **−0.23** | [−1.04, +0.26] | 0.835 | no | `NOT_RICH` |

- **Long.** In the calm regime, the median wing ask is about 1.7× its conditional fair value after fees. Conditioning made the long side worse than the pooled 10c result, not better.
- **Short.** In the volatile regime, wing bids sit at or below fair value. The market does not under-price the tail when volatility is high; if anything it under-reacts.

## 3. Post-hoc cell table (no decision)

- **Short-TTE wings (1–8 h) in calm and mid regimes are rich at the bid.** For example, low tercile at 4–8 h: +1.05 log [+0.55, +1.29], 17 sessions. This repeats the 10c short-TTE observation on the same quotes, so it is not independent evidence. These are deep OTM contracts with tiny premiums (median bid about $0.09 per unit at 4–8 h). The log ratio is inflated by a near-zero fair value. Trading it means selling naked lottery tickets, which is the risk profile Ivan has ruled out.
- **16–20 h wings are CHEAP at the bid in every tercile, but CONSISTENT or RICH at the ask.** That is no long edge: the fair value sits inside the spread.

## 4. Consequence

As stated in advance:
- no rejection, so **the wing line closes** in Volatility Dynamics;
- no defined-risk wing backtest is pre-registered;
- the short-TTE richness is recorded as an observation only, not a hypothesis carried forward.

Reopening it would require a new pre-registration on prospective sessions (≥ 2026-10-11), with a defined-risk structure, approved by Ivan.

## 5. Questions for ChatGPT

1. Any objection to closing the line on these terms?
