# Liquidity Absorption 022 — A05b results review and where the Deribit history fits

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-20 |
| **Branch** | Liquidity_Absorption |
| **Responds to** | 018 (A05a), 019–021 (validator, A05b, audits); executed `05b/` outputs (commit 07fb14f) |
| **Status** | Review + proposal. No code changed. |

## 1. A05b ran cleanly — but it is uninformative

**Integrity (Jul–Sep 2026 only).** 180/180 currency-days `PASS_WITH_LIMITATIONS`; `millisecond_page_limit_buckets = 0`, `possible_missing_trade_seq_span = 0`; max prints in one millisecond = 182 (far below the 1,000-row page that triggered the old downloader skip). For this quarter the crowded-millisecond defect in 020 did not bite. Anchors: 845/904 `PRESENT`, 56 stale reference (39 of them ETH), 3 empty windows.

**Density.** Median prints per ±60 s window: BTC-PERPETUAL **112**, ETH-PERPETUAL **49** (≈1 and 0.4 prints/s). Median prints in [−10,−1) s and [1,10) s ≈ **3–4**; in [−1,0) and [0,1) s the median is **0**. The 1/10/100 ms bucket grid is therefore almost entirely empty, and the median aligned count imbalances of exactly ±1.00 in the phase table are single-trade artefacts, not flow.

**Separation (my re-test of `05b_event_phases.csv`, PRESENT & complete days, TARGET vs STOP, Mann–Whitney).** 16 comparisons (2 anchors × 4 phases × price/amount imbalance): only SIGNAL [−60,−10) signed price change differs (TARGET −0.81 bp vs STOP −2.20 bp, p = 0.001); SIGNAL [−10,−1) p = 0.04; everything else p > 0.15. The one surviving difference lies **inside the second-touch candle itself** — it is the rejection candle's own adverse body, already visible on the Binance 1m bar, re-measured on another venue. It adds no Deribit-specific information and is in-sample.

**Structural reason.** The anchors are Binance candle closes, not touch instants, and Deribit perp is a satellite venue of Binance price discovery at ~1 print/s. Millisecond microstructure on this tape cannot observe absorption at a Binance level.

**Recommendation:** close A05b as descriptive/null. No A05c millisecond follow-up.

## 2. What the branch actually lacks: sample size and a holdout

Every LA result sits on **one quarter (452 signals, Jul–Sep 2026)**, inspected repeatedly from A02 to A05b. Gross returns per closed trade are already tightly bracketed around zero:

| Variant | n closed | Mean gross bp | SE | Mean net bp |
|---|---:|---:|---:|---:|
| A03b forward 1:1 taker | 413 | −2.47 | 2.97 | −15.7 |
| A03c reverse 1:1 taker | 413 | +2.47 | 2.97 | −10.8 |
| A03c maker touch 3rd | 362 | −1.13 | 3.17 | −9.9 |
| A03c maker through 3rd | 361 | −1.23 | 3.18 | −10.0 |

Forward and reverse are exact mirrors around zero gross: this is what barrier first-passage on a near-martingale looks like. A gross edge large enough to pay ~10 bp round-trip taker costs is already ~3–4 SE away. Honest prior for any re-run: **the edge will not appear.** The value of the Deribit history is to make that verdict decisive on untouched data, not to rescue the strategy.

## 3. Where the Deribit futures history is useful

**(a) Out-of-sample replay of the frozen strategy — primary use.** The A02 detector is price-only on Binance UM 1m klines (`liquidity_absorption_stage_a02.Config`), which exist on disk for ~2 years. Only the execution layer needed Binance aggTrades (92 days; downloading 2+ years of Binance aggTrades would exceed the 100 GB budget). Deribit BTC/ETH-PERPETUAL prints (~2.6 MB/day/currency, ~5 GB for 2024–2026) can replace that tape:

- Signals: frozen A02 `Config` on Binance 1m, period = Binance kline start → **2026-06-30** (never inspected by LA). Jul–Sep 2026 is excluded from the holdout.
- Execution: A03b/A03c target and stop translated to **bp distances from the entry reference**, applied to the Deribit perp print path (print-ordered first passage, 240 m timeout). Cross-venue basis (~1–3 bp) is small against 60–80 bp median risk; measure it explicitly.
- Fees: Deribit perp schedule (maker/taker to be confirmed from Deribit's fee page before freezing).
- Bridge check on Jul–Sep 2026: re-run the Deribit-execution engine on the original 452 signals and compare outcome labels to A03b. Agreement rate is the transfer-validity gate; if < ~85% label agreement, stop and diagnose before the holdout.

Expected size ≈ 7× the current sample → SE ≈ 1 bp on gross mean.

**(b) Venue-native robustness arm (secondary).** Build 1m OHLCV + taker-buy volume from Deribit perp prints and run the same frozen detector natively on Deribit (2024-01 →). Answers whether the pattern is Binance-specific. Run only if (a) passes.

**(c) Dated futures.** Basis/term structure as a context filter is a new feature search on a strategy with zero gross edge, and funding/basis carry was already rejected in FC 02. **Not recommended** inside LA.

## 4. Pre-registration (to freeze before any holdout run)

1. Detector `Config`, zone width, target rules, 1:1 stop, 240 m hold, 1 s latency — unchanged from A02/A03b/A03c.
2. Primary variant: A03b forward 1:1 taker. Secondary: A03c reverse taker and maker-touch (maker fills remain unverified).
3. Pass gate: mean **net** bp per closed trade > 0 with one-sided 95% CI lower bound > 0, in **both** BTC and ETH, and in both halves of the period. Otherwise LA is closed.
4. No parameter, filter or variant added after seeing holdout output.

## 5. Gate 0 — full-history validator

The QC so far covers only Jul–Sep 2026. Before (a), run the validator on 2024-01-01 → 2026-10-19. Any day with `millisecond_page_limit_buckets > 0` was downloaded by the pre-fix pager and must be re-downloaded with the patched downloader before use; sequence-span warnings get listed and excluded from the holdout rather than silently kept.

## Questions for ChatGPT

1. Do you accept closing A05b as null on the density and anchor-resolution arguments in §1?
2. Any objection to bp-distance translation of Binance-derived barriers onto the Deribit print path, versus the venue-native arm?
3. What should the label-agreement threshold on the Jul–Sep bridge be?

## Decisions needed from Ivan

1. Approve (a) as the final LA test with the §4 kill gate, or close LA now on the existing gross ≈ 0 evidence.
2. Confirm the Binance UM 1m kline start date on disk (sets the holdout start).
3. Run Gate 0 (full-history validator) and push its `deribit_daily_qc.csv`.
