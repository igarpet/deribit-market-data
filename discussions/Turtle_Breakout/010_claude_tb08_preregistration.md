# 010 — Claude: TB 08 pre-registration — risk per trade from 1% to 100% on BTC

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-25 |
| **Branch** | Turtle Breakout |
| **Responds to** | TB 07 run `9293f61`; Ivan, 2026-10-25: "opening a reversal trade does in fact generate more edge"; test the 2% allocation against 1%–100% (every integer), BTC only |
| **Status** | pre-registration + code (validated on synthetic data); awaiting Ivan's run |

## 0. TB 07 in brief (run `9293f61`): the verdict is `BEATS_RANDOM_SHORTS_ONLY`, not `REVERSE_PAYS`

| | Confirmation (BTC/ETH/SOL/XRP) | Discovery (26 perps) |
|---|---|---|
| Reversals | 105 | 710 |
| Mean $R_{rev}$ (net) | **+0.17** | +0.04 |
| 95% month-cluster interval | [−0.14, +0.53] → **G1 FAIL** | [−0.16, +0.25] |
| Random flat-day shorts | −0.01 | −0.03 |
| p vs random shorts | **0.029 → G2 PASS** | 0.010 |
| Win rate | 41% | 41% |

**What this does and does not show.**
- **It does show timing information.** Shorting right after a failed breakout beats shorting at a random idle moment, and the result repeats on the discovery set.
- **It does not show that the reverse trade makes money.** Confirmation +0.17R per trade cannot be told apart from 0; discovery is +0.04R.
- **It is concentrated in bear phases.** On confirmation, 2022 (+9.3R) and 2026 (+14.6R) carry it; 2023–2025 sum −9.2R. On discovery, 2022 (+24.6R) and 2026 (+38.1R) carry it; 2020 is −22.3R.
- **Curves:** long + reverse gives total 214.6R vs 196.8R, max DD −12.4R vs −12.8R, daily-R Sharpe 1.19 vs 1.11. This is not tested, and the gain is the bear-year concentration above.
- **Sub-results** (reported, not tested): longs that lost the least (−0.25…0R) give the best reversals (+0.96R confirmation, +0.22R discovery); reversals after the fixed stop are about 0. These are post-hoc and need their own pre-registration before use.

So "more edge" is right in the narrow sense of G2 and not established in the sense that matters for capital (G1).

## 1. Question and prior

**Question.** TB 01 risks 2% of the high-water mark per trade: $u = f\cdot \mathrm{HWM}/(2\,\mathrm{ATR})$, $f=0.02$. What happens to growth, drawdown and ruin for $f = 1\%, 2\%, \dots, 100\%$ on BTC?

**Prior, stated in advance.**
- **Sharpe is almost invariant to $f$:** the same trades are scaled. Differences come only from compounding and from the cap.
- **Growth rises with $f$ up to roughly the Kelly fraction, then falls, then ruin.** For trade outcomes $R_i$ (in units of the 2 ATR stop), equity-based sizing gives
$$G(f)=\sum_i \log(1+f R_i),\qquad f^*=\arg\max_f G(f).$$
TB 01's BTC trades (54, win rate 39%, payoff about 2.4) suggest $f^*$ of several tens of percent in-sample, with a very wide uncertainty.
- **HWM sizing ruins earlier than Kelly suggests.** After a drawdown $D$, the actual risk is $f/(1-D)$ of equity, so drawdowns accelerate.
- **In-sample choice of $f$ is the most overfit parameter there is.** The best $f$ in-sample sits just below the ruin point, with drawdowns near 90%+. The test therefore asks whether any $f$ *significantly* out-grows 2%, after correcting for 99 comparisons.
- **Expected outcome: `2PCT_NOT_BEATEN`.** On synthetic data with BTC-like Sharpe (about 1–1.5), no $f$ clears Holm over 99. Detection needs a Sharpe of about 4 (§4).

## 2. Design (`turtle_sizing.py`, `turtle_breakout.py` 0.3.0)

- **Strategy:** TB 01 daily long-only on BTCUSDT perpetual, 5m fills, taker 0.05% + 1 bp, funding. Window: all BTC data after a 30-day warm-up, as TB 01's single-asset table (2020-01-31 → end). Only $f$ changes.
- **Sweeps:**

| | Cap on notional | Sizing base | Role |
|---|---|---|---|
| **P** | 125× equity (Binance BTCUSDT maximum) | HWM (TB 01) | primary |
| **C3** | 3× equity (TB 01) | HWM | reported; saturates once the cap binds |
| **EQ** | 125× | current equity | reported; the Kelly-consistent version |

- **Ruin:** equity ≤ 0 at a daily mark; absorbing; floored at $10^{-6}$ of start for log returns.
- **Check:** $f=2\%$, cap 3× reproduces TB 01's BTC single-asset run exactly (unit test on synthetic; real comparison printed in §1 of the notebook).
- `turtle_breakout.py` 0.3.0 adds `size_base` (`"hwm"` default); all earlier results are unchanged.

## 3. Decision (binding, P sweep)

For each $f \ne 2\%$: $\Delta g(f) = 365.25\cdot\overline{(\ell_t(f)-\ell_t(2\%))}$, the paired difference of mean daily log returns (geometric growth per year).
- 21-day circular block bootstrap, 2,000 draws.
- Two-sided p from the bootstrap standard error (normal approximation: a percentile p cannot reach the Holm threshold $0.05/99$ with 2,000 draws).
- Holm over the 99 comparisons, $\alpha = 0.05$ → `HIGHER_GROWTH` / `LOWER_GROWTH` / `NO_DIFFERENCE` per $f$.
- **Verdict:** `LARGER_RISK_GROWS_FASTER` if any $f$ is `HIGHER_GROWTH`; otherwise `2PCT_NOT_BEATEN`.

**Reported:** for every $f$ in every sweep: CAGR, log growth, Sharpe, max DD, Calmar, ruin, worst trade as % of equity, mean/max leverage, capped entries; Sharpe differences (same bootstrap, Holm); C3 and EQ decisions; Kelly curve on the 2% trades with a 5,000-run trade bootstrap of $f^*$ (5–95% range, share of draws with $f^* \le 2\%$); equity curves for 1%, 2%, 5%, 10%, ½ Kelly, Kelly.

## 4. Validation (synthetic)

`tests/test_turtle_sizing.py`: 11 tests, all passing.
- Defaults reproduce TB 01 exactly; entries are identical across $f$ and units scale with $f$.
- Leverage and drawdown grow with $f$; ruin is absorbing and floored; the 3× cap binds at high $f$; equity sizing differs from HWM sizing.
- Holm, circular blocks and the Kelly optimum (analytic case $f^*=0.2$) check out.
- **No-edge data:** no $f$ is `HIGHER_GROWTH`.
- **Power:** at synthetic Sharpe ≈ 1.5, 0 of 99 pass; at Sharpe ≈ 4.2, 38 pass; at ≈ 6.3, 98 pass.
- **Notebook smoke run** (synthetic, Sharpe 1.5): in-sample best $f$ = 24% (CAGR 309%, max DD −99%), ruin from 25%; verdict `2PCT_NOT_BEATEN`.

## 5. Run (PC, Windows CMD)

No download. Close the notebook in Jupyter first.

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_turtle_sizing.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks\Turtle_Breakout\08_risk_per_trade_btc.ipynb
git add notebooks\Turtle_Breakout
git commit -m "Turtle Breakout 08: run"
git push
```

**Runtime:** a few minutes (300 simulations of one asset).

**Outputs** go to `notebooks/Turtle_Breakout/08/`: `08_decision.json`, `08_sweep_{P,C3,EQ}.csv`, `08_compare_{P,C3,EQ}.csv`, `08_curves_P_selected.csv`, `08_manifest.json`.

## 6. Decisions needed from Ivan

1. Confirm §3 is binding (growth, not Sharpe, is the criterion; P = 125× cap and HWM sizing is the primary sweep).
2. Run §5.
