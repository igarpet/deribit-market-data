# Pairs fOU 004 — PF02 readout: the basis mean-reverts reliably and is about 100× too small to trade

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-16 |
| Branch | Pairs fOU |
| Responds to | Pairs fOU 003; Ivan's run c13eb48 |
| Status | Result. SP `GROSS_EDGE_ONLY`, PQ `NO_EDGE`. Economically decisive in both cases. |

## 1. The premise now holds

- **Construction-tied spreads are stationary:**
  - the ADF screen passes on 88% of SP pair-days (23,965 pair-days, 30 assets), against 1–3% for the Stage A coin pairs;
  - 81% are fOU-eligible;
  - about 59% of fOU trades exit by convergence.
- **The thesis machinery finally has a stable equilibrium to trade.**
- **Reporting bug:** the calibration table is empty. The notebook dropped the boolean `complete`/`converged` columns when checkpointing. This doesn't affect any P&L number; fixed in a later version if the branch continues.

## 2. The size is the problem

**Median rolling 7-day s.d. of the perp − spot basis:**

| Asset group | Median 7-day s.d. |
|---|---|
| BTC, ETH | 0.8–0.9 bp |
| SOL, XRP, DOGE | 1.4–1.5 bp |
| Most altcoins | 2–5 bp |
| APT, OP, WIF (the widest) | 5.6–6.4 bp |
| PQ (BTC and ETH quarterly − perp) | 6.6 bp |

A 1.5σ entry is therefore worth 1–10 bp of basis, i.e. half that per gross notional, against 17 bp (SP) and 12 bp (PQ) of costs.

**TEST, fOU, all assets** (bp per gross notional, 95% day-bootstrap):

| Family | Trades | Gross | Net |
|---|---|---|---|
| SP | 202,343 | **+0.16 [0.14, 0.18]** | −16.8 [−16.85, −16.81] |
| PQ | 2,633 | +0.02 [−0.09, 0.14] | −12.0 [−12.1, −11.9] |

- **The SP gross edge is real.** It is statistically significant, positive for 28 of 30 assets, and the same sign in DEV.
- **It is about 1% of costs.** Even the best asset (WIF) makes 0.54 bp gross per trade. Only 0.24% of trades make more than 17 bp gross, and the 99th percentile is 9.4 bp.
- **Funding adds about 0.01 bp per trade.** Holds average 3 bars, so they rarely cross a funding time.
- **Neither side changes it:** short perp/long spot and long perp/short spot are the same.
- **OU and z-score benchmarks give the same picture:** gross 0.1–0.3 bp, net about −17 bp.

## 3. Reading

- **Cost needed:** for SP to pay, the all-in round trip would need to be below about 0.3 bp per gross notional. That is the regime of market makers with fee rebates.
- **The fOU fits confirm how small the swings are:** median H on SP formation windows is about 0.04. The basis is almost white noise around its mean (bid-ask bounce and timing noise between the two venues' 5m closes), not a slow fractional mean reversion.
- **The thesis question is answered:**
  - with a stable equilibrium, the fOU first-passage logic does identify convergence (59% converge within T70);
  - the deviations it trades are economically empty at retail costs.

## 4. Status

- **SP and PQ: closed.**
- **Options (PF03, put-call parity on Deribit):** still unbuilt. Given these results I expect executable parity violations at bid/ask to be rare. Deribit option spreads are wide, about 27% of mid ATM 0DTE, far wider than any parity mispricing. I recommend not building it unless Ivan specifically wants the diagnostic.

## 5. Decision needed from Ivan

Accept and close Pairs fOU (01, 02). Decide on PF03: build the diagnostic, or skip it.
