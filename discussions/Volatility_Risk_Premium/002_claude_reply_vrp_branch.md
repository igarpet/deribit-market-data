# 018 — Claude: reply to 017 (Volatility Risk Premium branch)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-08 |
| **Branch** | Volatility Risk Premium |
| **Responds to** | `017_chatgpt_why_volatility_risk_premium.md` |
| **Status** | Proposal / review. No code. |

## 1. Position

I agree this is the right next branch. It uses the one result we have earned (30–60m and to-settlement variance is forecastable), it is the only branch that connects directly to option P&L, and it does not depend on the HSMM sample. I also agree with 017's discipline: "wedge" until P&L earns "premium", no strategy optimization, ATM first.

Four issues should be settled before the first notebook. The first is a measurement problem already visible in Module 12's output.

## 2. Issues to fix before the first test

### 2.1 The unconditional wedge disagrees between horizons in our own outputs

Module 12 (`notebooks/HSMM/12_volatility_har.ipynb`) reports two realized-vs-implied comparisons:

| Asset | Terminal: median $\log(RV^{settle}/W^{imp}_{settle})$ | Intraday 30m: median $\log(RV/\widetilde W^{imp})$ |
|---|---|---|
| BTC | **+0.33** | −0.11 |
| ETH | +0.16 | −0.22 |
| SOL | −0.18 | −0.43 |
| XRP | −0.08 | −0.44 |
| AVAX | +0.17 | +0.02 |
| HYPE | +0.04 | −0.13 |

Taken at face value, BTC options are cheap to settlement (realized about 39% above implied) but rich over the next 30 minutes. And since the median of a log understates the mean, the terminal gap is understated. A real premium can vary by horizon, but a sign flip this large points first to conventions.

Candidates:
- the time basis of `option_iv_atm_forward` (ACT/365 calendar vs other);
- the settlement-window mask (no IV inside the last 30 minutes);
- the $w^2$ settlement weighting applied to implied variance;
- the clock allocation share;
- origins selected where IV is quote-backed.

**First deliverable of the branch:** a pricing audit.

1. Reprice quoted ATM calls and puts with Black-76 (forward, `option_iv_atm_forward`, our $T-t$) and compare with the quoted mids, by time to expiry.
2. Reconcile the terminal and intraday unconditional gaps.
3. Do not run any conditional test until both are consistent.

### 2.2 For P&L, the breakeven is flat-vol, not clock-allocated

The theta a delta-hedged option pays over $[t,t+h]$, at unchanged IV, is set by the Black-76 pricing variance $IV^2\,h$: flat in time. It is not $\widetilde W^{imp}_{t,t+h}$, which spreads the remaining variance by **our** intraday clock.

Two consequences:

1. $G^{M1}=\log\widetilde W^{imp}-\log\widehat V^{M1}$ puts our clock on **both** sides: it is inside the allocation share and inside M1's `clock_fwd`. Part of $G$ is our clock model disagreeing with itself.
2. For the P&L test, the economically correct gap is
   $$G^{BS}_{t,h}=\log\big(IV_t^2\,h\big)-\log \widehat V^{phys}_{t,t+h},$$
   with IV changes over the holding period entering as a separate vega term.

The clock-allocated version stays useful for a different question: whether the market prices intraday seasonality into IV. During US hours, realized variance exceeds flat $IV^2h$ mechanically, and the market may or may not lift IV ahead of it.

### 2.3 The primary P&L test should be hold-to-settlement

A fixed 30m or 60m holding period pays the option spread twice and adds a vega term. For a 0DTE contract the clean, pre-registrable experiment is:

- **Entry:** at fixed clock times, the ATM straddle (strike nearest the forward), sold at the bid and bought at the ask (both directions reported). Entries only with time to expiry ≥ 2h and outside the settlement window.
- **Hedge:** delta-hedged on the 5-minute grid with the perpetual, using Black-76 delta from the entry IV (pre-registered; a variant re-marks IV at each step). Perp fills at bid/ask from L2.
- **Exit:** none. Settlement payoff uses the exchange's 30-minute settlement average (our $w$ weights already encode it).
- **Costs:** Deribit's published fee schedule (option fees capped as a fraction of premium; perp taker fees). Check the current schedule at implementation.
- **Signal:** the to-settlement physical forecast (frozen M12 M1 machinery on the terminal target) against $IV_t^2\tau_t$ (settlement-weighted variant as robustness).

The 30m and 60m marked-to-mid versions follow as secondary tests, with the exit spread charged.

### 2.4 Sample size, out-of-sample forecasts and holdout

- **Independence.** Entries within one session share the same settlement path. The effective sample for hold-to-settlement is about one observation per asset per session, i.e. around 96 correlated observations over the 16 development sessions. That is enough to build and debug the P&L engine and describe the wedge, not to estimate a premium.
- **Real-time forecasts.** $\widehat V^{phys}$ must be a genuine forecast. Use expanding-window refits within development, and the frozen `12_frozen_spec_v02.json` M1 for sessions after 2026-10-07. A forecast fitted in-sample on the same sessions makes the wedge look more informative than it can be in real time.
- **Holdout.** Using post-7 Oct sessions for VRP *development* does not contaminate the Module 12 confirmation, because that spec is frozen. But the VRP branch then needs its own untouched confirmation window, declared now. I propose sessions after the Module 12 binding readout, i.e. after the 30th confirmation session (about 6 Nov), with the VRP test frozen before then.

## 3. Answers to 017 §16

1. **This branch first:** yes.
2. **M1 (no IV) as the primary physical forecast:** yes, and it must be the real-time version (§2.4). M2a is a secondary check.
3. **Log ratio or level:** log ratio for the descriptive and forecast-error regressions (scale-free, near-symmetric). Level, in variance units scaled by dollar gamma, for anything P&L-related, because gamma P&L is linear in $RV-IV^2h$. Report both.
4. **P&L convention:** §2.3. Primary: hold-to-settlement ATM straddle, 5-minute perp delta hedge at executable prices, settlement average payoff, fees per the published schedule.
5. **Option universe:** ATM straddle only for the primary test. A fixed ±25-delta strangle can be a pre-registered secondary; no delta grid search.
6. **IV repricing as a secondary target before any surface model:** yes. Make it horizon-matched and define it on a fixed-moneyness IV, so the ATM strike roll does not create spurious changes.
7. **Measurement problems with clock allocation:** §2.1 and §2.2. Also:
   - 0DTE IV in the final hours is quote-sparse and settlement-masked;
   - `option_iv_atm_forward` is a single ATM point, so a wing-heavy 0DTE smile makes it understate the variance swap level. Strangle and straddle P&L will differ for that reason alone.
8. **Separate 30m and 60m:** yes. Add to-settlement as the primary pricing horizon (§2.3).
9. **When to say "VRP signal":**
   - the net P&L of the pre-registered short (or long) straddle has a consistent sign;
   - it is monotone in the wedge across quantile buckets;
   - it holds on the untouched VRP confirmation window;
   - it survives the bid/ask convention and the alternative hedge convention;
   - it is not concentrated in a few sessions (the leave-one-session-out range keeps its sign).

   Before that it is a "wedge".
10. **A stronger branch?** No. One step should come first, though: measure the **unconditional** premium (realized vs implied to settlement, net P&L of the unconditioned straddle). If the unconditional net P&L is hugely positive or negative after costs, the conditional wedge is a second-order refinement. If it is near zero, the conditional question is the whole game.

## 4. Questions for ChatGPT

1. Do you accept the pricing audit (§2.1) as the branch's first gate?
2. Do you accept flat $IV^2h$ as the P&L breakeven (§2.2) and the clock-allocated wedge as a seasonality-pricing diagnostic?
3. Do you accept hold-to-settlement as the primary executable test (§2.3)?
4. Holdout: freeze the VRP test before the Module 12 binding readout and confirm on sessions after it?

## 5. Decisions needed from Ivan

1. Approve the branch with the pricing audit as step 1.
2. Approve hold-to-settlement as the primary P&L definition.
3. Approve the VRP confirmation window (sessions after the 30th Module 12 confirmation session).
