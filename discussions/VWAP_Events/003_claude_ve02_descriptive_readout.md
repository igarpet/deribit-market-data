# VWAP Events 003 — VE02 descriptive readout: touches, crosses and return times

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-17 |
| **Branch** | VWAP Events |
| **Responds to** | Ivan's request (2026-10-17); run `d31464d` |
| **Status** | Descriptive (no gates) |

## Data and definitions

- **Data:** BTC and ETH Binance USD-M perpetual 1m, 741 days. Session VWAP anchored 08:00 UTC; 4H VWAP anchored 00/04/…/20 UTC.
- **Touch:** the bar range contains the VWAP known at the bar's open.
- **Cross:** the close switches side.
- **Placebos:** VWAPs anchored at unused times.

## Counts (real anchors)

| | Touch episodes / window | Crosses / window | Crosses / day | Up / down |
|---|---|---|---|---|
| BTC session | 34.7 | 41.2 | 41.3 | 15,302 / 15,292 |
| BTC 4H | 13.4 | 15.7 | 93.9 | 34,854 / 34,708 |
| ETH session | 36.5 | 42.5 | 42.6 | 15,782 / 15,753 |
| ETH 4H | 14.0 | 16.2 | 97.3 | 36,144 / 35,956 |

## After a cross

| | Back before reset | Median: next touch | Median: cross back | Median: return after detaching | Back within 5 min / 15 min / 60 min (after detaching) |
|---|---|---|---|---|---|
| BTC session | 97% | 2 min | 3 min | 5 min | 53% / 72% / 88% |
| BTC 4H | 93% | 2 min | 3 min | 4 min | 53% / 74% / 89% |
| ETH session | 97% | 2 min | 3 min | 4 min | 54% / 73% / 88% |
| ETH 4H | 93% | 1 min | 3 min | 4 min | 55% / 75% / 90% |

- **Means are much larger than medians** (session 26 min, 4H 11–12 min after detaching): a long tail of crosses that run away. The 90th percentile is about 60 min (session) and 30 min (4H).
- **Up and down crosses are indistinguishable.**

## Real vs placebo

Every statistic matches the placebo anchors to within about 1–3%:
- touch episodes and crosses per window;
- share returning;
- median and Kaplan–Meier times;
- share within 5 min.

The one outlier is the 14:00 session placebo, which has fewer touches and crosses (≈ −8%). That is the US-open anchor, which starts in high volatility, so price leaves early; it is a property of that anchor time, not of the real VWAP.

## Reading

**VWAP crossings are near-continuous chop.** About 41 crosses per session day, and half are reversed within ~3 minutes. That is what a running average sitting close to a random-walk price produces. The real 08:00 and 4H anchors behave exactly like arbitrary anchors.

This is consistent with VWAP 002: no special behaviour at the real VWAP levels.
