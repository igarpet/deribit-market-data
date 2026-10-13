# Regime Structure 011 — RS 06 pre-registration: delta-hedged long 0DTE straddle (gamma scalping)

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-13 |
| Branch | Regime Structure |
| Responds to | RS 010 §5; Ivan (chat): "I want to test the gamma scalping hedge, we'll remove the limitation for this test" |
| Status | Pre-registration, before any real-data RS 06 output |

## 1. Why

RS 05 found intraday paths mean-revert. The session's net move has about 25% less variance than its 5-minute path. A straddle held to settlement is paid on the net move, while a delta-hedged straddle earns the *path*:

$$\text{P\&L}\approx\tfrac12\sum_k\Gamma_k S_k^2\left(r_k^2-\sigma_{\text{impl}}^2\,\Delta t\right)$$

If the path variance at the hedge interval exceeds implied variance by more than the hedge and option costs, the hedged straddle profits where the unhedged one does not. **Ivan lifted the no-hedging constraint for this test only.**

## 2. Design (`gamma_scalp v1.0`, notebook RS 06)

**Option leg (VRP 02 code, approved):**
- long ATM 0DTE straddle of the primary family at the ask + option taker fee;
- every full-hour entry with 120–1200 minutes to the 08:00 UTC expiry;
- settled at the exchange delivery price, with coin-premium conversion as VRP 02.

**Hedge leg (Deribit perpetual, 5-minute collector grid, `time_series_feature_state`):**
- **Price:** perpetual tradable mid. Fallback: Deribit index, with the share reported. Forward-filled up to 15 min. An entry is skipped if more than 10% of its hedge times are missing.
- **Delta:** Black-76 straddle delta at the hedge price, strike $K$, remaining $T$, with the **entry straddle IV held fixed**.
- **Rebalance:** to $-\Delta$ every $H$ minutes from entry. The position is closed at the last bar before 08:00. Basis between that price and the delivery price is ignored.
- **Cost per side:** |position change| × price × (half of `futures_spread_bps` + fee). Fee is taker 0.05% (as BT 001). Default half-spread 1 bp where missing.

$$R_{\text{hedged}}=\frac{\text{payoff}+\text{hedge P\&L}-\text{hedge cost}}{(\text{ask}+\text{option fee})\cdot\text{conv}}-1$$

**Primary:** $H$ = 15 min, taker. Statistic and verdict exactly as VRP 02: mean over sessions of the session-mean $R$, 95% session bootstrap.
- CI > 0: `LONG_EDGE_CANDIDATE` (must then hold on later sessions).
- CI < 0: `NO_LONG_EDGE`.
- Otherwise: `INCONCLUSIVE`.

**Reported (no decision):**
- $H\in\{5,15,30,60\}$;
- fee scenarios: taker 0.05% + half-spread / 0.02% + half-spread / zero hedge cost;
- hedged minus unhedged;
- realised variance at $H$ over implied variance;
- hedge cost and hedge P&L as a share of premium;
- by asset and TTE bucket.

**Section 4, descriptive, answers RS 010 option (a).** Binance 2-year variance ratio $\sum\text{net}^2/\sum\text{RV}_k^2$ at 5, 15, 30 and 60-minute sampling.
- If it rises toward 1 as sampling coarsens, the 5-minute reversion is mostly noise, which a hedger cannot capture.
- If it stays below 1 at 30–60 minutes, the reversion is real at hedgeable horizons.

## 3. Expectation, stated before the run

The hurdle is high:
- Option spread plus fee is about 13% of premium.
- Implied has run about 15% above the realised net move.
- Taker hedging every 15 minutes costs on the order of 10% of premium (synthetic smoke run at BTC-like parameters).

Hedging only wins if realised variance at 15 minutes exceeds implied by about 25–30%. RS 05's variance ratio of 0.75 implies 5-minute path variance about 33% above net variance. That is about the size needed, *before* hedge costs. I expect `NO_LONG_EDGE` or `INCONCLUSIVE` at taker cost. The zero-cost scenario shows the gross structure.

## 4. Fee constants (Ivan to confirm)

Perpetual taker 0.05% per side, as in BT 001. If Ivan's account tier differs, the low-fee scenario (0.02%) brackets it.

## 5. Synthetic validation

`tests/test_gamma_scalp.py`, 5 tests:
- a fair-priced straddle hedged at zero cost has mean $R\approx0$ and much lower variance than unhedged;
- realised vol above implied pays when hedged;
- the cost gap equals turnover × cost;
- planted 5-minute reversion pays a 5-minute hedger more than a 60-minute one;
- the variance ratio by sampling detects planted reversion.

The notebook smoke-runs end to end.

## 6. Decision needed from Ivan

Run RS 06. Binding for the primary verdict.
