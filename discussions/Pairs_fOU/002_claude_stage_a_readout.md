# Pairs fOU 002 — Stage A readout: crypto majors are not cointegrated at 5 minutes; the pipeline barely trades

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-16 |
| Branch | Pairs fOU |
| Responds to | Pairs fOU 001; Ivan's run ed43ec9 |
| Status | Result. The pre-registered verdict is `INCONCLUSIVE`; the substantive reading is a failure at the formation stage. Stage B is not unlocked. |

## 1. Formation funnel: the strategy fails before trading

Over 734 trade days × 6 pairs = 4,404 pair-days:

| | Share of pair-days |
|---|---|
| Integration screen passed | ~85% |
| **Engle–Granger p ≤ 1%** | **1.1–2.7%** per pair (4,314 of 4,404 rejected) |
| fOU-eligible (also H < 0.5, structural T70) | 64 pair-days (1.5%) |

- **The cointegration pass rate is at the test's own false-positive level (1%).** On rolling 7-day windows of 5-minute data, no pair of BTC/ETH/SOL/XRP shows cointegration beyond chance.
- **Where it does pass, the fit makes little sense:**
  - median H on eligible days is 0.44–0.49, i.e. near 0.5;
  - half-lives are 45–76 bars, close to the 48-bar cap;
  - structural T70 is 107–152 bars, well beyond the trading horizon.

## 2. Trades

| Mode, TEST | Trades | Days with trades | Gross bp/trade [95%] | Net bp/trade [95%] |
|---|---|---|---|---|
| fOU (thesis) | 25 | **4** | 11.8 [−1.7, 16.4] | −0.2 [−13.7, 4.4] |
| OU (H = 0.5) | 35 | 5 | 6.4 [3.4, 11.3] | −5.6 [−8.6, −0.7] |
| z-score | 226 | 51 | −1.8 [−11.2, 6.4] | −13.8 [−23.2, −5.7] |

- **The 25 fOU trades are one episode, not evidence.** All 25 come from 2025-10-12 → 10-15, the days right after the 10 October 2025 crash. That crash sat inside the 7-day formation window and produced spurious cointegration on BTC-XRP and ETH-XRP.
- **DEV produced no fOU trades.**
- **The pre-registered label, `INCONCLUSIVE`, is technically right, but four days of trades cannot support any inference.**
- **The z-score rule:** it trades every cointegrated pair-day and loses net. Gross is about 0.

## 3. Calibration repeats the thesis failure, but worse

| Mode | Admitted forecasts | Mean predicted P(converge by T70) | Realised |
|---|---|---|---|
| fOU | 25 | 70.9% | **0%** |
| OU | 35 | 71.0% | 2.9% |

- **Thesis comparison:** 70.3% predicted against 33.1% realised.
- **Here:** every fOU trade ended at "expiry" (T70) without the spread touching the formation mean. The spread's level shifts out of sample faster than the fitted mean reversion can pull it back. This is the same equilibrium-instability problem as thesis §12, and at 5 minutes it is total.

## 4. Conclusion

- **The framework does not transfer to 5-minute crypto majors:**
  - the cointegration premise fails at the formation stage;
  - the few admitted forecasts are badly over-confident;
  - the simpler z-score rule loses after costs.
- **Stage B (options) stays locked.** It would add two option spreads to a signal with no gross edge.
- **Not proposed:** loosening the 1% cointegration threshold, lengthening the formation window, or changing the horizon after seeing these results. That would be specification search on a result that already failed its premise.

## 5. Decision needed from Ivan

Accept and close the Pairs fOU branch. A genuinely different universe — e.g. the perpetual vs its own quarterly future (basis), or spot vs perp on the same asset, which *are* cointegrated by construction — would be a new, separately pre-registered question.
