# Regime Structure 010 — RS 04 and RS 05 readout: implied already reacts; intraday paths mean-revert; big days are spike-and-giveback days

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-13 |
| Branch | Regime Structure |
| Responds to | RS 009; Ivan's run, commit 9e6db93 |
| Status | Result. Both decisions are binding. **Corrects RS 008 §3** ("trend days"). |

## 1. RS 04 — implied vol does not underreact to yesterday's realised vol

79 session-assets, 20 sessions (2026-09-22 → 10-11), all settled at exchange delivery prices.

| Slope on $x=\log(\text{RV}_{\text{yday}}/\hat\sigma_7)$ | β | 95% CI |
|---|---|---|
| **log(realised / implied move), primary** | **−0.59** | [−1.49, +0.21] |
| log(implied move / $\hat\sigma$) | **+0.24** | **[+0.04, +0.40]** |
| log(realised move / $\hat\sigma$) | −0.35 | [−1.29, +0.54] |

**Verdict: `NO_EVIDENCE`.**
- The option market raises implied vol after a high-RV day, significantly.
- In these 20 sessions the realised move did not follow. The point estimate leans toward *over*reaction (3 of 4 assets negative).
- Straddle R_net by predictor tercile: low −9% [−44%, +29%], mid −29% [−48%, −7%], high −19% [−43%, +8%]. No tercile is positive.

RS 03's only confirmed predictor is already priced into 0DTE options. The underreaction route is closed.

## 2. RS 05 — real paths vs shared-sign null (399 draws, test window; dev agrees)

| | Real | Null median | Real / null | p | |
|---|---|---|---|---|---|
| **S1 daily variance ratio (net/RV)², all sessions** | 0.746 | 0.996 | **0.75** | 0.005 | **significant** |
| S2 efficiency ratio, BIG days | 0.136 | 0.145 | 0.94 | 0.12 | — |
| **S3 giveback, BIG days** | 0.495 | 0.282 | **1.76** | 0.005 | **significant** |
| **S4 top-hour share, BIG days** | 0.200 | 0.164 | **1.22** | 0.005 | **significant** |
| S5 next-day BIG lift | 1.48 | 1.20 | 1.24 | 0.32 | — |

Development window: S1 0.79, S3 1.78, S4 1.07 (all p = 0.005); S2 and S5 not significant. Every asset agrees on S1 (0.73–0.85) and S3 (1.52–2.16).

### Reading

1. **Intraday 5-minute moves partially reverse within the session.** The session net move has about 25% less variance than the sum of its 5-minute variances. Under the null it would be equal. This is the first robust *directional* structure in the branch, and it is mean reversion, not continuation.
2. **Big days are spike-and-giveback days, not trend days.** Against a null with the same volatility path, they:
   - concentrate more of their movement in one hour (1.22×);
   - give back 1.76× more of their maximum excursion;
   - are no more efficient than chance.

   RS 008 §3's "trend-like" reading was the mechanical selection effect it warned about. **RS 008 §3 is corrected accordingly.**
3. **Big-day clustering is volatility clustering.** The real lift (1.48) is within the null (1.20, 95% range up to about 1.7), because the null keeps |r|.

### Caveat on S1, not resolved here

Part of a variance ratio below 1 at 5-minute sampling can be microstructure noise (bid-ask bounce, wick prints) rather than tradable reversion. VRP 006 Part E found 5m RV inflated 5–26% by noise. The two can be separated by repeating S1 with 15-minute and 30-minute bars:
- noise fades with coarser sampling;
- genuine reversion persists.

Proposed as RS 06 if Ivan wants it.

## 3. What this means

- **For long options held to settlement, the structure works against the buyer.** Intraday paths revert, so the settlement move is smaller than the path's realised variance. That is consistent with realised/implied ≈ 0.83 in VRP 02 and RS 02.
- **The side that would harvest it** is a long option delta-hedged during the day (gamma scalping). It earns path variance rather than the net move. That requires the frequent hedging Ivan has ruled out because of its spread and fee cost, and it is only worth it if S1 is not mostly noise (§2 caveat).
- **Ivan's "mean reversion" intuition has partial support** as a property of the whole session: moves tend to reverse. Range was never supported as a persistent *state* (RS 01).

## 4. Questions for ChatGPT

1. Does a coarser-sampling variance-ratio test (15m, 30m, 60m) cleanly separate noise from reversion here, or is a Hasbrouck/Roll-type noise estimate needed?
2. Is spike-and-giveback on big days consistent with liquidation cascades? If so, is it predictable from open-interest or funding data (Binance `metrics`)?

## 5. Decisions needed from Ivan

1. Accept RS 04 (`NO_EVIDENCE`; the predictor is priced in) and RS 05 (intraday mean reversion; big days are spike-and-giveback).
2. Choose:
   - (a) RS 06, the variance ratio at coarser sampling (noise vs reversion);
   - (b) revisit the no-hedging constraint for a delta-hedged long-option test;
   - (c) close the branch and decide on the strategic review.
