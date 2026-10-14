# VD 036 — Claude: review of Modules 13–18 (AM-T-HAR-Blend)

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-14 |
| Branch | Volatility Dynamics |
| Responds to | VD 030–035; Ivan's runs up to c8add99; ChatGPT's AM-T-HAR-Blend summary (relayed in chat) |
| Status | Review. Evidence from `notebooks/Volatility_Dynamics/18/*.csv`; opinions labelled. |

## 1. What I agree with

- The summary's framing is right, and so are its own caveats:
  - volatility is predictable, direction is not;
  - discrete regimes are not established;
  - the TEST period has been inspected six times (VD13–18), so all ranking is exploratory;
  - the horizon-calibrated overlay is not a coherent path model.
- It also matches the Regime Structure branch (RS 001–010), which reached the same picture independently from price and flow labels.
- The next step ChatGPT names, **HAR variance + a direct Student-t without the Markov integration**, is the right one. I would make it the only next step (§4).

## 2. What the numbers actually show

**2.1 At the primary horizon the gain over M2 is negligible, and it is concentrated in the assets with data issues.**

60m NLL, HAR-Blend − M2 (VD18 per-asset):

| AVAX | BTC | ETH | SOL | TRX | XRP | Pooled |
|---|---|---|---|---|---|---|
| +0.0002 | −0.0009 | −0.0002 | −0.0039 | **−0.0058** | −0.0035 | −0.0023 [−0.0050, +0.0003] |

- For BTC and ETH the two models are identical to within 0.001 nats.
- The largest gain is TRX, which VD18 flags for frequent exact-zero 5-minute returns.
- 0.002 nats per observation is a likelihood ratio of about 1.002.

The honest summary is: **AM-T-HAR-Blend ≈ M2 ≈ AM-T-HAR-Direct at 30–240 minutes.** What these models have in common, heavy tails and a good variance level, is what works. Nothing establishes that the Markov machinery adds anything.

**2.2 The 5–15-minute "wins" are against a broken benchmark.** At 5 minutes M2 misses its 95% interval 23% of the time (target 5%). Its QLIKE is −9.39 against −12.51, and it loses by 0.35 nats; TRX alone loses by 1.11 nats. At 10 minutes it still misses 9%. M2 was designed as a horizon-specific mapping and is evidently mis-specified at the shortest horizons. **Claims that HAR-Blend beats M2 at 10, 15 or 30 minutes should not be cited** until the comparison is against a short-horizon benchmark that is itself calibrated.

**2.3 Tick discreteness can inflate continuous-density likelihoods.**
- TRX and AVAX have many exact-zero 5-minute returns. A Gaussian-mixture density with small-variance components can place very high density at 0 and collect "free" likelihood on those bars.
- The TRX numbers fit this:
  - AM_T_BOTH misses its TRX interval 8.8% of the time, against about 5% elsewhere;
  - the TRX NLL gaps between models are the largest at every horizon.
- Until this is controlled, TRX and AVAX should be excluded from any pooled ranking. Alternatives are a discretisation-aware likelihood (probability of the observed tick bin) or adding jitter of ±½ tick.

## 3. Relevance to options (from the Regime Structure branch, imported as conclusions with Ivan's general approval)

- **RS 04:** after a high-realised-vol day, 0DTE implied vol rises significantly (slope +0.24 [+0.04, +0.40]), and the realised move does not follow. The option market already uses recent realised volatility.
- **RS 06:** a delta-hedged long 0DTE straddle loses more than an unhedged one (−41% vs −28% of premium at H = 15 min, taker). Even with zero hedge cost it loses −20%. The perpetual taker fee (0.05%) costs about 21% of premium at 15-minute rebalancing.

A better physical density only creates value if it disagrees with implied vol in a way that predicts outcomes. That is the single decision-relevant use of this model for the project, and it has not been tested.

## 4. Proposal (opinion)

1. **One ablation, then freeze.** Fit HAR-4 variance plus a direct Student-t on the same origins (`HAR_T`):
   - Score it with TRX and AVAX excluded (the pooled-4 ranking is primary), and against a recalibrated short-horizon benchmark.
   - If `HAR_T` ≈ HAR-Blend, the Markov integration is dropped for terminal-return forecasting.
2. **No further TEST-period specification search.** Freeze the winner and score it prospectively on data after 2026-10-14, as HSMM 016 proposed for its own model.
3. **Model vs market (the only trading-relevant test):** at each 12:00 UTC collector entry, regress $\log(\text{realised}/\text{implied move})$ on $\log(\hat\sigma_{\text{model}}/\sigma_{\text{implied}})$, session bootstrap.
   - A slope above 0 means the model knows something the market does not.
   - This reuses RS 04's code with the model forecast in place of yesterday's RV.
   - About 22 sessions, so underpowered, but cheap.
4. Defer path coherence (a HAR-driven stochastic process). Its use case, path-dependent hedging, just failed on costs in RS 06.

## 5. Questions for ChatGPT

1. Do you agree that M2 at 5–10 minutes is mis-specified (23% and 9% interval misses), and that short-horizon "wins" against it should be withdrawn until there is a calibrated benchmark?
2. Is the TRX advantage a discreteness artefact? Would you accept the tick-bin likelihood or jitter fix before any pooled ranking?

## 6. Decisions needed from Ivan

1. Approve the `HAR_T` ablation with TRX and AVAX excluded as primary, then a freeze.
2. Approve the model-vs-implied test (§4.3) as the only bridge from this branch to options.
