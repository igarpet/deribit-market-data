# 000 — Claude: Order-Book Imbalance branch handoff

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-11 |
| **Branch** | Order Book Imbalance (new; not yet started) |
| **Responds to** | Ivan's request after VD 022: explore order-book imbalance as a source of a long-only options edge |
| **Status** | handoff, the starting brief for a new chat |

## 1. Why this branch exists

The Volatility Dynamics branch (VD 001–022) closed the price-only routes to a **long** options edge:

- **Shock timing from price and volatility precursors (VD 011–016).** Precursors reduce to the volatility level, which the market already prices. Near-the-money 0DTE options are fairly priced against realized outcomes.
- **Jump timing (VD 018).** Jumps measured relative to local volatility arrive as a **homogeneous Poisson process**. There is no self-excitation and no clock or volatility dependence.
- **Unconditional long wings (VD 020, 022).** OTM 0DTE wings trade at about 3.8× their realized-outcome value at the ask. Even with hindsight-perfect exits, buyers got back about 0.76× their cost.

**A long-only edge therefore needs information the option price does not already contain:** a predictor of direction, of realized volatility above ATM IV, or of jump intensity. The perpetual's order book is the obvious unused source. All of the information above came from prices.

## 2. Intuition

The limit order book shows **pending supply and demand** before it trades.

- **OBI (order-book imbalance)** at depth $k$, as defined in the collector (`futures_state_cache.py`):

$$
\text{OBI}_k=\frac{\text{BidDepth}_k-\text{AskDepth}_k}{\text{BidDepth}_k+\text{AskDepth}_k}\in[-1,1].
$$

  When OBI > 0, more size is resting on the bid side. Market-microstructure work finds that OBI and order-flow imbalance predict the **next** mid-price moves. The effect is strongest over seconds to a few minutes and decays quickly.
- **Thin or unstable books** (low depth, wide spread, OBI flipping sign rapidly) are less able to absorb an aggressive order. Plausibly they raise the probability of a **large** move: volatility or jumps rather than direction.

Three channels follow, and each maps to a different long-option trade:

| Hypothesis | Book signal → outcome | Long trade if true |
|---|---|---|
| **H1 direction** | OBI / microprice / trade-flow imbalance → sign and size of $r_{t\to t+h}$ | Near-ATM call or put |
| **H2 magnitude** | Depth thinness, spread, OBI instability → realized vol to expiry **above ATM IV** | Near-ATM straddle or strangle |
| **H3 tails** | Book fragility → jump intensity $\lambda$ (Module 08 framework) | Wings, only if $\lambda$ multiplies a lot (see §5) |

## 3. Data already collected

5-minute bars from the **perpetual** book of each underlying (AVAX, BTC, ETH, HYPE, SOL, TRX, XRP), in `time_series_feature_state` (research DuckDB). The pipeline is in `src/data_engineering/features/market_time_series.py`.

| Group | Fields |
|---|---|
| Imbalance | `book_twa_obi_5`, `book_twa_obi_10`, `book_twa_obi_20`, `book_min_obi_20`, `book_max_obi_20`, `book_twa_obi_20_change_5m` |
| Depth and price | `book_twa_bid_depth_20`, `book_twa_ask_depth_20`, `book_twa_spread_bps`, `book_twa_microprice`, `book_twa_microprice_vs_reference_bps` |
| OBI instability (**only from about Oct 1–2**) | `book_twa_abs_obi_20`, `book_obi_20_std`, `book_obi_20_range`, `book_obi_20_path_roughness_per_min`, `book_obi_20_coherence_noise`, `book_obi_20_transition_count`, `book_obi_20_raw_zero_cross_count`/`_rate`, `book_obi_20_reversal_strength_mean`/`_max`, histograms |
| Flow | `trade_flow_imbalance` |
| Quality | `book_valid`, `book_stats_complete`, `book_stats_coverage_ms`, `book_update_count`, `book_out_of_order_update_count`, `book_max_update_gap_ms` |
| Options (for §5) | `normalized_option_chain` (bid/ask, forward, TTE), `primary_option_surface_feature_state` (ATM IV and the surface) |

"TWA" means time-weighted over the 5-minute bar. **Raw L2 snapshots are not stored.** If the effect lives at the seconds scale, 5-minute TWA aggregates will blur it, and the collector would need a finer field (see §7).

## 4. Known traps, from earlier branches

1. **Construction leakage.** `book_twa_microprice_vs_reference_bps` correlates −0.75 to −0.86 with the **same-bar** return, by construction: the numerator is a within-bar average and the denominator is the end-of-bar reference (VR 007). Every test must predict **strictly after** the bar, from $t+5$ onward.
2. **Collector vintages.**
   - The L2 instability fields exist only from about Oct 1–2.
   - An out-of-order-update fix and a `book_update_count` scale shift happened around **Oct 4**.
   - Split or test by vintage before reading any coefficient (VR 007, VD 011).
3. **The volatility benchmark.** Raw extremes cluster because volatility clusters (VD 011). Any H2/H3 claim must beat a benchmark of **local vol + clock + ATM IV**, not zero.
4. **Sample size.** About 3 weeks means about 20 independent days. Use blocked CV by day, a day-block bootstrap and Holm across fields. Pre-register thresholds before seeing output, and never relax gates afterwards.
5. **Multiple horizons and fields.** Choose a small, pre-declared grid. VD 017 showed that picking a threshold after the fact manufactures findings on null synthetic data.
6. **Cross-asset differences.** Perp book liquidity differs hugely across the assets: BTC/ETH versus TRX/HYPE. Report results per asset; pool only when the per-asset results agree.

## 5. The economic bar a signal must clear

Option spreads are the binding cost (VD 020–022): about 27% of mid near the money and 77–100% in the wings. In addition, the fee is $\min(0.03\%\,F,\ 12.5\%\ \text{of premium})$ per side.

- **H1 (near-ATM directional).** An ATM option is worth about $0.4\,\sigma\sqrt{T}F$, and a predicted drift $\mu$ adds about $\Delta\mu F\approx0.5\mu F$. A round trip costs about one full spread, roughly $0.27\times0.4\,\sigma\sqrt T F$. Break-even therefore needs

$$
\mu \gtrsim 0.2\,\sigma\sqrt{T},
$$

  i.e. the signal must shift the expected move by about **0.2 standard deviations of the holding-period move**, net. That is a strong signal. Most published OBI effects are far smaller at horizons beyond a minute.
- **H2 (near-ATM straddle).** The predicted realized vol must exceed ATM IV by roughly **25–30%** on the selected trades.
- **H3 (wings).** Wings cost about 3.8× realized value at the ask, so the conditional jump intensity would need to be about **4× the unconditional rate**, and on identifiable bars. Treat H3 as a long shot. A more realistic use of H3 is as a **risk filter for the short side**: when *not* to be short wings, which links to VD 022.

## 6. Suggested module sequence

1. **01 — data audit.**
   - Field definitions.
   - Coverage by asset and day.
   - Vintage breaks: shifts around Oct 1–2 and Oct 4.
   - Construction checks against the same-bar return.
   - Distributions, and autocorrelation of OBI.
2. **02 — H1 direction.**
   - Targets: future returns over $h\in\{5,15,30,60\}$ minutes, on the index and the perp.
   - Predictors: OBI$_{5,10,20}$, ΔOBI, trade-flow imbalance, microprice displacement, all strictly lagged.
   - Benchmarks: zero and AR(1).
   - Validation: blocked CV by day.
   - Metrics: out-of-sample $R^2$, information coefficient, and the **§5 bar** (net drift vs $0.2\sigma\sqrt T$).
3. **03 — H2 magnitude.**
   - Target: realized variance to the next 08:00 settlement, relative to ATM IV² and relative to ex-ante $D$ (as in VD 021).
   - Predictors: depth, spread, OBI instability.
   - Benchmark: local vol + clock + ATM IV.
4. **04 — H3 jump intensity.** Add book covariates to the Module 08 inhomogeneous Poisson model. Reusing VD code needs Ivan's approval.
5. **05 — economic translation.** Only if 02–04 pass. Simulate the long trades at **real bid/ask** from `normalized_option_chain`, with fees, walk-forward, and pre-registered entry rules. The gate is the session-bootstrap lower bound of net P&L > 0, plus maximum-drawdown disclosure.

Each module should get a synthetic self-test first: a planted signal is detected, and a null gives no false pass. That has caught a design error in every VD module.

## 7. Open decisions for Ivan

1. Open this as a new branch (`discussions/Order_Book_Imbalance/`, `notebooks/Order_Book_Imbalance/`) under the usual rules:
   - one immutable file per turn;
   - "Questions for ChatGPT" / "Decisions needed from Ivan" at the end of each turn;
   - independence from other branches unless reuse is approved.
2. Decide whether to extend the collector with finer book data (e.g. 1-second OBI snapshots or event-level OBI changes) **now**. Without it, H1 can only be tested at 5-minute resolution, where the literature expects the effect to be mostly gone. Data collected later cannot be backfilled.
3. Decide whether Modules 04–05 may reuse Volatility Dynamics code (Module 08 Poisson framework, Module 10 quote preparation).
