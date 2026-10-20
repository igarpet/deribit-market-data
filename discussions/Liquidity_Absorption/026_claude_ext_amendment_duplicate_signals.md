# Liquidity Absorption 026 — EXT amendment before results: duplicate signals from coinciding level streams

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-20 |
| **Branch** | Liquidity_Absorption |
| **Responds to** | Ivan's first EXT run (commit 74334cf): stopped at A02b with `Repeated A02 candidate identity` |
| **Status** | Amendment to LA 024. No execution result existed when it was made. |

**What happened.**
- **A02 ran:** 2,566 signals, BTC 1,204 / ETH 1,362, 2024-01-02 → 2026-10-19.
- **A02b refused:** 2 pairs of rows share symbol, side, signal minute and level price. In both, a prior-day extreme and a 5m pivot sat at the same price, so two of the detector's four streams fired the same signal:

| Row | Symbol | Side | Signal | Level |
|---|---|---|---|---|
| 1 | ETH | short | 2025-03-19 00:25 | 1935.55 |
| 2 | ETH | long | 2026-04-09 00:49 | 2179.50 |

**Fix.**
- These are one trade, so the first row is kept (the prior-day stream, which is first in detector order) and the duplicate is dropped. This happens in both A02 files, keeping row alignment.
- The dropped rows are logged in `EXT/02/02_duplicates_dropped.csv`.
- `stage_a02b` applies the same dedupe to an A02 output that already exists, so the re-run reuses the finished venue inputs and A02.
- The test `test_dedupe_same_signal_two_streams` was added.

**Effect.** 2,564 signals, gate and everything else unchanged. V2 (LA 025) uses the same deduplicated signals.
