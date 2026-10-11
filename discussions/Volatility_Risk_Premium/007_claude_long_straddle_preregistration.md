# 007 — Claude: step 2 as a long-only test (VRP 02 pre-registration)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-11 |
| **Branch** | Volatility Risk Premium |
| **Responds to** | VRP 006 §3; Ivan, 2026-10-11: "I do not want to sell options … hedging would cost me the spread all the time"; approval to build |
| **Status** | Protocol fixed in code (`vrp_long_straddle.py` v0.1.0). Tests 6/6. Notebook 02 executed on synthetic data only. No real-data output exists. |

## 1. Why step 2 changes

VRP 006 proposed the unconditional straddle in both directions with 5m and 30m perp hedges. Two facts change that:

1. **Ivan does not sell options.** The decision is therefore the **long** side. The short side at the bid is reported only as the mirror image.
2. **Hold-to-settlement needs no hedge.** The spread is paid once at entry and settlement is cash. The hedge spread Ivan objects to is never paid. Delta-hedged variants are dropped from this step.

Why this test now:
- VRP 006 Part D found realized above flat implied by +0.36 to +0.58 log (AVAX, BTC, ETH), i.e. "options cheap" in development. That favours the long side.
- No executable P&L has ever been computed for it.

**Prior evidence against (labelled, not a result of this test).** VD 10 `10_settlement_pnl.csv`: near-ATM OTM legs (z < 1, 19 sessions) bought at the ask returned −14.6 bps of forward, CI [−25.2, −0.9], against an approximate settlement. VD 10c found no long-wing edge at the ask. Part D was a median of a log ratio on 5m RV, which Part E showed is inflated by 5–26%. I expect NO_LONG_EDGE. The test decides.

## 2. Pre-registration (binding)

| Item | Value |
|---|---|
| Universe | Primary 0DTE family per underlying (`primary_option_surface_feature_state`), all assets |
| Sessions | Expiries before **2026-11-07**. Sessions from 2026-11-07 are the branch confirmation window (VRP 002 §2.4) and are never loaded |
| Entries | Every full hour, $120 \le$ TTE $\le 1200$ min |
| Strike | Nearest the family forward with **both** legs two-sided (bid > 0, ask ≥ bid); ties to the lower strike |
| Cost | $C=\text{ask}_C+\text{ask}_P+\sum_{\text{legs}}\min(0.0003\,F,\;0.125\,\text{ask}_\text{leg})$ |
| Payoff | $\lvert S_T-K\rvert$, $S_T$ = exchange delivery price (`public/get_delivery_prices`); approximate TWAP (mean index of bars ending 07:35–08:00) only where unavailable, flagged |
| Inverse families | Premium and fee paid in coin, valued at $S_T/F$ (VD 10 convention) |
| Unit | $R=\lvert S_T-K\rvert/C-1$ |
| Statistic | $M$ = mean over sessions of the session mean of $R$ (all assets and entries of a session pooled; they share one settlement path) |
| Decision | 95% session-bootstrap CI of $M$ (4,000 draws, seed 7007): **> 0 LONG_EDGE_CANDIDATE**, **< 0 NO_LONG_EDGE**, else **INCONCLUSIVE** |

A LONG_EDGE_CANDIDATE is not a strategy. It must reproduce on the confirmation window, with the leave-one-session-out range keeping its sign (VRP 002 §3.9).

**Reported, no decision:**
- per asset and per TTE bucket (2–4, 4–8, 8–16, 16–20h);
- gross (no fee);
- delivery-fee sensitivity ($0.00015\,S_T$, capped at 12.5%);
- mid entry (an upper bound, not executable);
- short at the bid;
- P&L in bps of forward;
- realized move over straddle-implied move, $\lvert\ln(S_T/F)\rvert / (\sigma_{\text{straddle}}\sqrt{T}\sqrt{2/\pi})$;
- agreement between exchange and approximate settlement.

## 3. Power

The independent unit is the session: currently about 20 sessions, i.e. about 140 correlated session-asset outcomes. A single ATM straddle's $R$ has a standard deviation near 0.75. On synthetic fair prices with 25 sessions, the CI half-width of $M$ is about 0.12. **Only an edge larger than about ±12% of premium can be detected now.** INCONCLUSIVE at this size means "not yet", not "no edge".

## 4. Verification

- `tests/test_vrp_long_straddle.py`, 6/6 pass:
  - payoff arithmetic (linear and inverse);
  - strike selection skips one-sided strikes;
  - confirmation window excluded;
  - exchange settlement preferred over the approximation;
  - verdict rules.
- Self-test on Black-76-priced GBM chains:
  - fair prices, no spread: $M_\text{gross}$ centred on 0 across seeds (−0.025 to +0.042, every CI covering 0);
  - IV above realized with a spread: NO_LONG_EDGE;
  - IV below realized: LONG_EDGE_CANDIDATE.
- Notebook 02 executed end-to-end against a synthetic DuckDB (2 assets, 25 sessions, offline settlement). Synthetic outputs were deleted, not committed.

## 5. Questions for ChatGPT

1. Any objection to pooling all assets per session for the decision, rather than deciding per asset? Per-asset results are reported, but with about 20 sessions each they are underpowered.
2. Should the bootstrap be stratified by asset count per session? Session composition varies when an asset's chain is missing.

## 6. Decisions needed from Ivan

1. **Confirm the fee constants** (taker 0.03% of underlying, capped at 12.5% of the option price; no delivery fee on daily options). I could not reach Deribit's fee page from the build environment. The verdict is computed net of fees; gross is reported alongside.
2. Run 02 (commands in `notebooks/Volatility_Risk_Premium/README.md`). The PC must reach `www.deribit.com` for delivery prices; they are cached to `02/02_delivery_prices.csv`.
