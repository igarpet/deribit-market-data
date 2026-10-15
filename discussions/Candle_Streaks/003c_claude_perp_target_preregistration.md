# Candle Streaks 003c — CS03b with the perpetual's own next return as target (pre-registration)

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-15 |
| Branch | Candle Streaks |
| Responds to | CS03b readout (`notebooks/Candle_Streaks/03b/`); Ivan's request "do the perpetual target in CS03b" |
| Status | Pre-registration. Written before any real-data run of CS03c. |

## 1. Why

CS03b is the only directional pass in the project: USD-M signed taker flow improves next-candle forecasts for **Binance spot** (BTC TEST binary log-loss 0.693150 → 0.692820). Two problems:

1. **Lead-lag.** The perpetual leads spot. Perp flow at bar $t$ is partly a perp price move that spot has not yet completed. Predicting spot from it can be spot catching up, which a perp trader cannot capture.
2. **Size.** A binary log-loss gain $\Delta$ corresponds to an edge $e \approx \sqrt{\Delta/2}$. For $\Delta \approx 3\text{–}5\times10^{-4}$ that is $e \approx 1.3\text{–}1.6\%$, i.e. a 51.3–51.6% hit rate. With $\mathbb{E}|r_{5m}| \approx 10$ bp, the gross gain is $2e\,\mathbb{E}|r| \approx 0.3$ bp per trade against a 12 bp round trip.

CS03c asks the question a perp trader needs answered: does flow predict the **perp's own** next 5-minute return, and does it survive costs?

## 2. Data and construction (frozen)

- Assets: BTC, ETH (primary); SOL, XRP (secondary). Cutoff 2026-10-14 00:00 UTC.
- Perp: existing `BINANCE_UM_DIR/<ASSET>USDT_um_1m.parquet`. The 5-minute bar ending at $t$ needs minutes $t-5,\dots,t-1$. $r_t = \log C_{t-1\text{min}} - \log C_{t-6\text{min}}$, defined only if the previous bar is complete.
- Sign, streak, volatility and target all come from perp returns. Flow is the CS03 construction, unchanged.
- **Spot reference:** CS03b's spot-target panel refitted on the same common complete days, so the perp vs spot gap is measured like for like.
- Panel, complete-day rule, 60/20/20 day split, features, multinomial logit (newton-cholesky, $C=0.25$), losses and paired UTC-day bootstrap (1000) are identical to CS03b (`ab.*`, `tf.*`, unmodified).

## 3. Statistical gate (as CS03b)

On TEST, for **both** BTC and ETH, perp target: `price_vol_signed` minus `price_vol`, upper 95% bound $< 0$ on both three-class and binary log-loss.

## 4. Economic gate (new)

Edge $e_t = P_t(\text{UP}) - P_t(\text{DOWN})$ from `price_vol_signed`. Threshold $\tau$ = TRAIN 90th percentile of $|e|$.

$$\text{pos}_t = \operatorname{sign}(e_t)\,\mathbf{1}\{|e_t|\ge\tau\},\qquad \text{PnL}_d=\sum_{t\in d}\text{pos}_t\,r_{t+1} - c\sum_{t\in d}|\text{pos}_t-\text{pos}_{t-1}|$$

The position is flat at the start and end of each UTC day and across any gap.

| Cost scenario | Per side | Role |
|---|---|---|
| taker | 6 bp (0.05% + 1 bp slippage) | **Gate** |
| low_fee | 3 bp (0.02% + 1 bp) | Report |
| zero | 0 | Report only (no adverse selection modelled) |

**Gate:** TEST mean net bp/day for the taker scenario, with the lower 95% day-bootstrap bound $> 0$, for both BTC and ETH.

Also reported: the every-bar rule ($\tau = 0$) at zero cost, as a pure gross-edge measure, and the `price_vol` and `price_vol_both` models.

## 5. Verdicts

| Statistical gate | Economic gate | Verdict |
|---|---|---|
| Pass | Pass | `PERP_TRADEABLE_CANDIDATE`, then a prospective test only |
| Pass | Fail | `PERP_STATISTICAL_ONLY` |
| Fail | Pass | `ECONOMIC_WITHOUT_STAT_GATE`, treated as noise |
| Fail | Fail | `NO_PERP_SIGNAL` |

The CS03 TEST window has been inspected before (CS03, CS03b, CS04a). All results are historical-exploratory.

## 6. Validation done before the real run

The synthetic tests include:
- a pure perp→spot lead-lag world: the spot target is detected (Δ ≈ −0.048), the perp target is not (Δ ≈ +0.0003);
- a planted perp effect, which is detected;
- the null, which is not detected;
- the P&L and turnover arithmetic.

## 7. Expectation (stated in advance)

`NO_PERP_SIGNAL` or `PERP_STATISTICAL_ONLY`. I expect the perp-target gain to be smaller than the spot-target gain, and the taker-cost net P&L to be clearly negative. A positive economic result at taker cost would contradict every directional test so far, and would need a prospective test before anything else.
