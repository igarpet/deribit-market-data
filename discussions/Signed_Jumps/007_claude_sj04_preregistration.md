# Signed Jumps 007 — SJ 04 preregistration: jumps at 1-second resolution

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-16 |
| **Branch** | Signed Jumps |
| **Responds to** | Signed Jumps 006 §4–5 (option 3); Ivan downloaded spot 1s klines (2026-07 → 2026-09) |
| **Status** | Preregistration. Fixed before any real-data output. Code `signed_jumps_seconds.py` (SJ04-0.1.0), tests `tests/test_signed_jumps_seconds.py` (5 pass), downloader `tools/download_binance_spot_1s.py` |

## 1. Questions

SJ 006 left three questions that 1-minute bars cannot answer:

1. **P1:** does the activity build-up start more than 30 seconds before the move? At 1 minute it appeared only in the last ~5 minutes.
2. **P2:** is SJ 01's same-sign excitation within a minute or two partly one move straddling a minute boundary?
3. **P3:** does a system that reacts within seconds see continuation that the 1-minute close (SJ 02a) missed?

## 2. Data and events

- **Jumps:** the SJ 01 engine on the full 1m spot history, unchanged. The notebook asserts that jump counts reproduce SJ 01. Only jump minutes covered by the 1s files (BTC, ETH, SOL, XRP; 2026-07 → 2026-09, all inside SJ 01's TEST period) are used. Nothing is fitted in SJ 04.
- **Seconds:** Binance spot 1s klines, bar-end timestamps; seconds with no kline row count as zero trades.
- **Onset:** inside the jump minute $(T-60\,\text{s},T]$, the first second at which the signed cumulative return reaches half of the minute's move.

## 3. Tests

**P1 (gate), early build-up.** Activity is $a_t=\log(\text{trades}_t+1)-\log(\overline{\text{trades}}_{[t-3600,\,t-1]}+1)$.
- **Event statistic:** mean $a$ over $[\text{onset}-120,\ \text{onset}-31]$ s for cluster-starting jumps.
- **Placebo:** 3 random seconds per event, in minutes with no jump within ±10 minutes, matched by asset and UTC hour.
- **PASS** iff event minus placebo > 0 with day-block bootstrap q2.5 > 0.

**P2 (stated threshold), split events.**
- **Sample:** consecutive jump minutes on the same asset.
- **Statistic:** the gap between their largest signed 1s returns.
- **Split:** a gap of 30 s or less.
- **Rule:** a split share above 50% among same-sign pairs means SJ 01's diagonal excitation is mostly a bar-boundary artefact (`MOSTLY_ARTEFACT`); otherwise `MOSTLY_DISTINCT_EVENTS`. Opposite-sign pairs are reported.

**P3 (gate), seconds-scale continuation.**
- **Statistic:** signed spot return from onset + 5 s to onset + 65 s, cluster starters.
- **PASS** iff gross q2.5 > 0. Net at 12 bp (perp-equivalent) and 22 bp (spot taker) is reported.
- The grid $\delta\in\{1,5,10,30\}$ s × $H\in\{10,30,60,300\}$ s is reported only.

**Also reported:**
- the 1s and 10s-binned activity profile from −600 to +300 s;
- the distribution of onset seconds within the minute;
- the share of the minute's move in its largest second.

## 4. Synthetic validation

| Check | Result |
|---|---|
| No precursor, no drift | P1 and P3 FAIL |
| Planted precursor (trade rate × e^0.7 for 120 s) | P1 PASS, excess +0.58; profile rises from −120 s |
| Planted 20 bp drift over 60 s | P3 PASS, +18 bp |
| Jumps straddling minute boundaries | `MOSTLY_ARTEFACT` |
| Genuine second jumps 90 s later | `MOSTLY_DISTINCT_EVENTS` |
| Activity baseline | strictly causal |

## 5. Limitations

1. **Sample.** Three months, spot only, TEST-era. This is descriptive confirmation, not a new holdout for trading rules.
2. **Onset in a quiet minute** can be pulled by noise. The share of the move in the peak second is reported to show how concentrated jumps are.
3. **P3 measures spot.** Perp moves track spot within 1–6 bp (PF 004).

## 6. Expectation (stated in advance)

- **P1:** PASS, but small. The build-up is volatility and activity rising together (SJ 006), and seconds data should show it starting 1–3 minutes ahead.
- **P2:** `MOSTLY_DISTINCT_EVENTS`, but with a split share of 20–40%, so SJ 01's η₊₊/η₋₋ is partly inflated.
- **P3:** gross 0–3 bp; FAIL or a PASS far below costs.

## Decisions needed from Ivan

1. Approve and run. Expect about 10–20 minutes: the 1m engine rebuild plus seconds processing.
