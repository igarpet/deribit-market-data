# 004 — ChatGPT: Pandas timestamp-unit bugfix

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-10 |
| **Branch** | Volatility Dynamics |
| **Status** | Bug fixed |

## Problem

The standalone v0.2.0 engine used `DatetimeIndex.asi8` directly and assumed the returned integer unit was always nanoseconds.

That assumption is unsafe in Pandas 3, where datetime arrays can be backed by microseconds. The engine then divided microsecond integers by a nanosecond conversion factor, causing a true five-minute interval to be inferred as approximately 0.005 minutes. Exact-grid checks against `Timedelta.value` could also fail for the same reason.

## Fix

Version bumped to `0.2.1`.

All integer timestamp arithmetic now normalizes explicitly:

```python
t.as_unit("ns").asi8
```

before differences or comparisons.

The synthetic self-test now explicitly converts its timestamps to microsecond backing with `as_unit("us")` and verifies that:

- five-minute spacing is still inferred as 5.0 minutes;
- backward 30m realized-variance windows still match the expected calculation.

No research assumption, threshold, feature, or other branch was introduced by this fix.
