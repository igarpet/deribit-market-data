# VD 037 — VD19: AM-T-HAR-Blend variance vs 0DTE implied volatility (pre-registration)

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-16 |
| Branch | Volatility Dynamics |
| Responds to | VD 036 §4.3; Ivan approved the test on 2026-10-15 |
| Reuse | VD14/VD18 code unchanged (same branch); VRP 02 straddle code and the RS 04 settlement flow (reuse previously approved by Ivan) |
| Status | Pre-registration. Written before any real-data run. |

## 1. Question

Does the volatility model know something the 0DTE option price does not? If the model's forecast of the move to expiry is above implied, does the realised move come out above implied?

This is the only route from the volatility model to a long-only trade.

## 2. Model side (frozen)

- **What represents the model.** The **level** of the AM-T-HAR-Blend forecast is its HAR_4 net-return variance: the blend weight is about 0.98 at 60–240 minutes, and the Markov part only shapes the mixture's tails. A straddle's value depends on variance level, so the HAR_4 level stands in for the model.
- **Fit.** VD18 `fit_har` with HAR_4 lookbacks (30/60/240/1440 min), clock harmonics and asset effects. It is fitted on VD18's own TRAIN split of Binance spot 5m data from before 2026-09-19 (all seven VD18 assets), with VD18's TRAIN net-return scale.
- **Out-of-sample by construction.** Every option entry falls after the fit sample.
- **Forecasts.** The origin is the 5m bar ending at the entry time (`price_5m` timestamps are bar ends), so there is no lookahead.

| Entry (UTC) | Horizon to 08:00 expiry | Role |
|---|---|---|
| **04:00** | 240 min | **Primary**; a native VD18 horizon |
| 06:00 | 120 min | Native VD18 horizon |
| 12:00 | 1200 min | HAR_4 refitted at 1200 min; an extension, secondary |

## 3. Option side

- The ATM straddle of the primary family at the entry (VRP 02 `select_straddles`).
- Implied s.d. of the log move: $\sigma_{imp}=IV_{mid}\sqrt{T}$.
- Settlement from the Deribit delivery price, with the 07:35–08:00 index average as a fallback.
- $R_{net}$ = payoff / (ask + fee) − 1.

## 4. Statistics

$$x=\log\frac{\sigma_{model}}{\sigma_{imp}},\qquad y_{rv}=\log\frac{RV_{(t,T]}}{\sigma_{imp}^2}\ (\text{Binance 5m}),\qquad y_{rv}=a+\beta x+\varepsilon.$$

- **Primary:** $\beta$ at the 04:00 entry, with a 95% session bootstrap (4000 draws).
- **Why realised path variance rather than the settlement move:** a single $|\log(S_T/F)|$ has noise of about 1.1 in log terms. In the synthetic test, the settlement-move outcome's interval is more than twice as wide.
- **Also reported:**
  - $y$, the settlement move over implied, for every entry;
  - the encompassing regression $\log RV=a+b_{imp}\log\sigma_{imp}+b_{mod}\log\sigma_{model}$;
  - mean $x$, i.e. the model level versus implied.
- **Economics:** mean session $R_{net}$ of straddles bought only when $\sigma_{model}\ge1.25\,\sigma_{imp}$. The 1.25 threshold is fixed in advance: VD 020–022 put the break-even near 25–30% realised over implied. Its 95% session bootstrap is reported for all entries.

## 5. Verdicts (primary entry)

| Condition | Verdict |
|---|---|
| β lower bound > 0, and selected-trade $R_{net}$ lower bound > 0 | `MODEL_EDGE_CANDIDATE` (then the prospective window only) |
| β lower bound > 0 | `MODEL_ADDS_INFORMATION` |
| β upper bound < 0 | `MODEL_ANTI_INFORMATIVE` |
| Otherwise | `NO_EVIDENCE` |

## 6. Power and limits (stated in advance)

- **Sample:** about 25 sessions × 4 assets ≈ 100 rows per entry. Assets on the same session are correlated, so the effective sample is closer to 25–50.
- **What the test can detect:** in the synthetic test the realised-variance outcome detects a strongly informative model. A weak one would come out `NO_EVIDENCE`. "No evidence" is not "no effect".
- **The economics are fragile:** the selected-trade subset may contain only a handful of sessions.
- **Market mismatch:** the model is fitted on Binance **spot** and the option is on the Deribit index. 5m spot realised variance carries microstructure noise (variance ratio below 1). That adds a constant offset to $y_{rv}$, which the intercept absorbs, not β.
- **Overlap with VRP:** VRP's confirmation period, from 2026-11-07, is excluded by the VRP code.

## 7. Expectation (stated in advance)

`NO_EVIDENCE`. RS 04 showed implied already reacts to yesterday's realised volatility. HAR is a richer version of the same information, so the market most likely prices it too. A positive β would be the first evidence in the project that the option market under-uses public price history.
