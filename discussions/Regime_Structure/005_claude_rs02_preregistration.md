# Regime Structure 005 — RS 02 pre-registration: does the C3 signal pay for long 0DTE options?

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-13 |
| Branch | Regime Structure |
| Responds to | RS 004 §5 (Ivan chose option (a)); Ivan approved reuse of VRP code (chat, 2026-10-13) |
| Status | Pre-registration, before any real-data RS 02 output |

## 1. Question

RS 004 found one robust effect. After a 15-minute window with an upward, flow-confirmed displacement (C3 signal), the next 15 minutes contain a ≥3σ 5-minute bar 6.7% of the time, against 4.6% unconditionally.

Ivan only buys options. Does this signal make buying 0DTE options profitable, or at least clearly less unprofitable than buying at any time?

**Expectation, stated before the run.** VRP 02 found the unconditional long ATM 0DTE straddle loses about 27% of premium (20 sessions). Half of that is the spread. A 2-percentage-point rise in short-horizon tail probability is small against that. I expect "less bad", not profitable.

## 2. Cross-branch reuse (approved by Ivan, 2026-10-13)

Imported from `vrp_long_straddle.py` (VRP 02), code only:
- ATM straddle selection;
- fee schedule;
- settlement (exchange delivery prices, TWAP fallback);
- coin-premium conversion;
- session bootstrap and verdict labels.

No VRP result, threshold or decision is used as an assumption. Sessions on or after the VRP confirmation start (2026-11-07) are never loaded, which protects VRP's holdout.

## 3. Signal

`regime_structure v1.2`, scale 15 minutes, development-window thresholds (identical to RS 01):
- a window labelled DISP with net > 0;
- signal time $t_s$ = window end;
- Binance USD-M 1m, BTC/ETH/SOL/XRP.

## 4. Part A — Binance, 2 years (does the signal raise the move an option is paid on?)

$z_h=\lvert\log P(t_s+h)-\log P(t_s)\rvert/(\hat\sigma_5\sqrt{h/5\,\text{min}})$, with $\hat\sigma_5$ the RS EWMA at the window end.

- $h$ = 60 min, and $h$ = to the next 08:00 UTC, the latter only for 120–1200 minutes to 08:00, the same entry range as Part B.
- Ratio = mean $z$ over signals ÷ hour-of-day-matched mean $z$ over all other eligible 15-minute window ends.
- 97.5% CIs (Bonferroni over the 2 horizons) from a bootstrap over settlement days. **Test window from 2026-04-01.**
- Reading:
  - The 60-minute ratio should exceed 1 if C3 is real.
  - The settlement ratio shows how much of that survives to settlement, which is what a held straddle is paid on.
- No trading decision is taken from Part A.

## 5. Part B — collector quotes (executable)

- **Entries:** every 5-minute collector snapshot, 120–1200 minutes to the 08:00 UTC expiry, ATM straddle of the primary family (VRP selection).
- **Signal entries:** the first snapshot at or after a signal time, within 10 minutes.
- Only entries up to the last Binance signal time are used.
- **Primary:** $M_{\text{signal}}$ = mean over sessions of the session mean of $R_{\text{net}}$ (payoff at settlement / (ask + taker fee) − 1) over signal entries, with a 95% session bootstrap (4,000 draws).
  - CI > 0: `LONG_EDGE_CANDIDATE` (must then hold on later sessions).
  - CI < 0: `NO_LONG_EDGE`.
  - Otherwise: `INCONCLUSIVE`.
- **Secondary (reported, no decision):**
  - signal minus all entries ($\Delta$, sessions with a signal);
  - ATM call only, held to settlement (60% of the shocks went up);
  - straddle sold at the bid 60 minutes after entry, with fees (USD values at each time; the coin-premium FX effect over one hour is ignored);
  - gross, mid, and move/implied.

## 6. Limits

- About 20 sessions: the primary CI will be wide.
- Signal entries cluster within sessions, so the session is the unit and effective sample size is the session count.
- Binance price is the signal source; Deribit's index is the settlement source. They differ by a few bps.

## 7. Synthetic validation

`tests/test_regime_c3_options.py`, 6 tests:
- signal frame alignment and price path;
- settlement-horizon eligibility;
- the matched ratio finds a planted 1.5× move;
- TTE filter;
- signal tagging (first snapshot within tolerance, nothing beyond it);
- call and exit variants.

A notebook smoke run on synthetic chains and klines completes end to end.

## 8. Decision needed from Ivan

Run RS 02. That run is binding for the primary statistic.
