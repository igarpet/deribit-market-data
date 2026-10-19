# Jump Fade 001 — Branch pre-registration: passive ±1.5σ bracket fade with take-profit at the pre-move price

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-19 |
| **Branch** | Jump Fade |
| **Responds to** | Ivan's idea (2026-10-19): resting limits at ±1.5σ around $P_{t-1}$, cancel the unfilled side, passive take-profit back at $P_{t-1}$; Ivan's choices: time stop + price stop, every bar with a forecast-vol split |
| **Status** | Pre-registration. Written before any real-data run. Code: `src/data_engineering/modeling/jump_fade.py` v1.0.0 |

## 1. Hypothesis and prior evidence

Moves of 1.5σ inside a 5-minute window partly revert. A passive fade, with entry and take-profit both as maker orders (zero fee on Deribit), then earns more than the stops and time exits lose.

**Prior evidence, with honest framing:**
- **SJ01:** the *next jump's* sign continues (71% after up jumps, 60% after down). That points against a fade.
- **SJ02a:** the post-jump perpetual path is flat on average (|mean| < 1 bp at 1–60 min), but **the median event reverses slightly**.

A bracket with a take-profit converts "frequent small reversals plus occasional large continuations" into its P&L. That is exactly what this test measures.

**Built-in risk: adverse selection.** Orders fill *because* price runs through them. The worst fills are the real jumps (CS03e showed the same mechanism).

## 2. Rule (binding)

- **Data:** Binance USD-M perpetual **1-minute** OHLC (`<A>USDT_um_1m.parquet`) for BTC, ETH, SOL, XRP, from 2024-10-01. The first 7 days per asset are warm-up for σ.
- **Price path vs cost venue:** Binance prices are the path. Costs are charged at Deribit (primary) and Binance VIP0.
- **Why not 5-minute candles:** they cannot tell whether the fill or the take-profit came first inside a bar. Decisions stay on the 5-minute grid; execution is simulated minute by minute.

**Decision**, at every 5-minute boundary $t$ while flat:
- $P_0$ = close at $t$.
- $\sigma_t = \sqrt{\mathrm{EWMA}_{hl=288}(r_5^2)}$, the standard deviation of 5-minute log returns over a 1-day half-life, causal.
- Resting buy at $P_0 e^{-1.5\sigma_t}$ and sell at $P_0 e^{+1.5\sigma_t}$, live for minutes $(t, t+5]$.

**Fill:** strict trade-through (low below the buy / high above the sell). The other order is cancelled at once. If both sides cross in the same minute, both single-side outcomes are simulated and the **worse** one is booked.

**Exits:**

| Exit | Rule | Order type |
|---|---|---|
| Take-profit | $P_0$, filled on a strict trade-through | maker |
| Price stop | $3\sigma_t$ from $P_0$ (1.5σ beyond the entry); a gap through the stop fills at the minute's open | taker |
| Time stop | 60 minutes after the fill minute, at that minute's close | taker |

**Conservative ordering:** in the fill minute only the stop can trigger; in later minutes the stop is checked before the take-profit.

**Positions:** one at a time per asset. The next bracket goes at the first 5-minute boundary at or after the exit.

**Costs per side:**

| Venue | Maker | Taker |
|---|---|---|
| Deribit (primary) | 0 bp | 5 bp + 1 bp slippage |
| Binance VIP0 | 2 bp | 5 bp + 1 bp slippage |

**Splits** (Ivan's choice: every bar, split by forecast volatility):

| Split | Definition |
|---|---|
| **all** (primary) | every boundary |
| **burst** | $\sigma_{1h}/\sigma_t$ in its DEV top decile, per asset. A volatility-burst / jump-hazard proxy: jumps cluster within minutes (SJ01), and HAR-type engines track the same short-memory variance. |
| **recent_jump** | a 1-minute move $>4\sigma_t/\sqrt5$ in the last 15 minutes. This is SJ01's clustering mechanism directly. |

## 3. Null

Every 1-minute log return gets a random sign, shared across assets by timestamp. Minute OHLC is mirrored.
- This keeps every absolute move, the volatility clustering and the jump sizes, and destroys any reversion or continuation.
- σ, the splits and the full simulation are recomputed for each draw. 199 draws.
- **Why the null and not zero:** on a martingale this bracket's gross P&L is negative by construction, about −3 bp per trade on synthetic data, because of the conservative fill model. The null measures that bias on the real volatility structure.

## 4. Verdicts (TEST = 2026-01-01 → end of data; DEV = 2024-10 → 2025-12, reported)

| Gate | Pass condition |
|---|---|
| **G1 reversion** | pooled gross bp/trade above the null: one-sided $p \le 0.05$ for *all*, $\le 0.025$ for each secondary split; gross $>0$ on ≥ 3 of 4 assets **and** above zero in absolute terms |
| **G2 net edge** | Deribit costs: daily equal-weight net P&L, bootstrap 95% lower bound $>0$; net total $>0$ on ≥ 3 of 4 assets |

- G1 and G2 both pass: `FADE_EDGE`.
- G1 only: `REVERSION_NOT_TRADEABLE`.
- Otherwise: `NO_REVERSION`.

**Grid (descriptive only, not a gate):** $K\in\{1,1.5,2,3\}$, $M\in\{2,3,4\}$, $H\in\{15,30,60,120\}$ min.

## 5. Validation before this turn (`tests/test_jump_fade.py`, 8 tests pass; `selftest()`)

- **Exit types:** a hand-built take-profit, short-side stop, gap-through stop at the open, 60-minute time stop and no-fill case all resolve correctly.
- **Ambiguous minutes:** the worse side is booked.
- **Costs:** correct per venue.
- **Sign flip:** keeps |returns| and valid OHLC.
- **End to end:** with planted reversion, gross P&L is ≥ 3 bp above every null draw. On a martingale, gross is below zero.

## 6. Fill-model caveats (they can only flatter the backtest)

- **Queue position.** A trade-through by one tick is counted as a fill. Real fill probability is lower, and the fills you miss are mostly the profitable "touch and reverse" ones.
- **Latency.** Placement and cancellation are assumed instant.
- **Order book.** Binance's book stands in for Deribit's.

A pass would therefore need a 1-second / aggTrades check (BTC/ETH data exist) before anything else.

## 7. My prediction (opinion, before the run)

`NO_REVERSION` or `REVERSION_NOT_TRADEABLE`:
- gross close to the null's −3 bp, at best a couple of bp above it;
- net at Deribit costs negative, because stops and time exits pay the taker fee while take-profits earn only 1.5σ (≈ 10–25 bp);
- *recent_jump* worse than *all*, since clustering means continuation.

## 8. Running it (PC, Windows CMD)

Close the notebook in Jupyter first.

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest tests.test_jump_fade -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 notebooks\Jump_Fade\01_jump_fade.ipynb
git add notebooks\Jump_Fade
git commit -m "JF 01 run"
git push
```

## Questions for ChatGPT

1. Is "worse outcome" for both-sides-in-one-minute too pessimistic? It affects only a small share of trades.
2. Should the null also randomize the *timing* of the bracket? I think the sign flip is sufficient, because σ is recomputed per draw.

## Decisions needed from Ivan

1. Approve this pre-registration (or amend it before running).
