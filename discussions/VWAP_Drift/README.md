# VWAP Drift discussions

Is there conditional drift of price **toward** the anchored VWAP (session 08:00 UTC or 4H), with displacement measured in volatility-engine units and conditioned on the engine's volatility regime and on recent jumps? This is the prerequisite for Ivan's thesis-style trade: one long option in the direction of VWAP, with VWAP as the mean. Notebooks: `../../notebooks/VWAP_Drift/`.

Independent branch: Binance USD-M 1m klines and the frozen VD18 HAR_4 engine artifact (read only). Builds on, but does not reuse conclusions from, VWAP Events (closed, VE04 `NO_EDGE`).

## History

| Turn | Content |
|---|---|
| `001_claude_vwd01_preregistration.md` | VWD 01: drift $y_h=-\operatorname{sign}(D)\,\Delta\log C$ at 15/60/240 min and to 08:00; 144 cells (anchor × horizon × \|Z\| × vol tercile × jump); DEV selection, TEST gates G1 (vs placebo anchors) and G2 (> 7 bp); options gate; S1 high-vol hypothesis |

Next turn in this branch: `002`.

Procedure and numbering rules: `../README.md`.
