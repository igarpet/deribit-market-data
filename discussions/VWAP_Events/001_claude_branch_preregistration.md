# VWAP Events 001 — branch pre-registration

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-13 |
| Branch | VWAP Events |
| Responds to | Ivan, chat of 2026-10-13 (regime definitions from discretionary MNQ trading) |
| Status | Pre-registration, before any real-data run |

## 1. Question

Ivan's observation from trading MNQ: when price touches the VWAP (session and 4-hour), a violent move follows. This branch tests that claim on crypto before it is used as an anchor or regime feature anywhere.

Two weaknesses the test must remove:

1. **Salience.** Violent moves after a touch are memorable; quiet touches are not.
2. **Many levels.** Session VWAP, 4h VWAP and range levels together put a level near price most of the time, so most moves "start at a level".

Hence the comparison is not "touch vs. no touch" but **real VWAP touch vs. touches of placebo levels**, built and detected identically.

## 2. Data (independent branch)

Binance USD-M perpetual 1-minute klines, about 24 months, downloaded with `tools/download_binance_um_klines.py` (free, checksum-verified). Reason: Deribit perpetual flow is a small share of BTC volume, so a Deribit-only VWAP is a local VWAP. Klines give the VWAP exactly: $V_t=\sum \text{quote vol}/\sum \text{vol}$ since the anchor.

No other branch's outputs, thresholds or state definitions are used.

## 3. Definitions (fixed in `src/data_engineering/modeling/vwap_events.py`, `vwap_events v1.0`)

**Anchors.**
- `session_0800`: daily VWAP anchored at 08:00 UTC, the Deribit daily expiry. **Assumption for Ivan to confirm or replace**: crypto has no cash session, so this is my choice of "session". Warm-up 60 min.
- `h4`: 4-hour VWAP anchored at 00/04/08/12/16/20 UTC. Warm-up 30 min.

**Volatility.** $\hat\sigma_t$ = EWMA of squared 1m log returns, half-life 60 min, causal.

**Touch at bar $t$.**
- $\text{low}_t\le V_{t-1}\le \text{high}_t$: the bar touches the level known at its open.
- The previous $m=15$ closes all lie on one side of the level, and at least one was $\ge 1.0\,\hat\sigma_{t-1}\sqrt{15}$ away in log distance.
- At least the warm-up has elapsed since the anchor.
- A 15-bar cooldown follows each event.

**Outcomes** from the close of bar $t$, for $k\in\{5,15,30,60\}$ minutes:

$$R_k=\frac{\sqrt{\sum_{i=1}^{k} r_{t+i}^2}}{\hat\sigma_{t-1}\sqrt{k}},\qquad M_k=\frac{\lvert \log C_{t+k}-\log C_t\rvert}{\hat\sigma_{t-1}\sqrt{k}}$$

- $\text{cross}_k$: price ends on the opposite side of the touched level from the approach side.
- Primary: $y=\log R_{15}-\operatorname{median}_{\text{UTC hour}}\log R_{15}$. Normalising by time of day stops session opens from producing a false effect.

**Placebos** (20 draws each, same detector and outcomes):
- `offset`: the real VWAP times $e^{\delta}$, with $\delta$ constant within each anchor window. $\lvert\delta\rvert\sim U(0.5,3)\cdot\hat\sigma_{\text{anchor}}\sqrt{60}$, random sign. Same smoothness as VWAP, but nobody watches it.
- `phase`: a VWAP of the same period anchored at a random phase. For `session_0800` the phase is 1–23 h after 08:00; for `h4` it is 30–210 min after the grid. It is a real VWAP, but anchored at a time nobody uses.

## 4. Windows and decision rule

- Development: start of data → 2026-03-31. Used for display and robustness only.
- **Test: 2026-04-01 → end of data. Gates are evaluated here only.**
- Statistic: $\Delta=\bar y_{\text{real}}-\bar y_{\text{placebo}}$, pooled over assets, with a bootstrap over UTC days (2,000 resamples).
- Comparisons: 2 anchors × 2 placebo types = 4. Bonferroni at family $\alpha=0.05$, so each CI is 98.75%.
- **G1:** the CI lower bound of $\Delta$ is above 0.
- **G2:** $e^{\Delta}\ge 1.10$, meaning post-touch realised volatility is at least 10% above placebo. Significance alone at this sample size could be economically trivial.
- **Pass per anchor:** G1 and G2 against **both** placebo types.

Secondary results are descriptive, with no gates: other horizons, $M_k$, per-asset readouts, and cross-versus-bounce probability.

## 5. Interpretation, written before results

| Result | Reading |
|---|---|
| Fails against both placebos | The VWAP-touch effect is perception on crypto. VWAP drops as a special level; the range/structure route (hand labels) goes ahead without it. |
| Passes against `offset`, fails against `phase` | Any VWAP works, the anchor does not matter: a "smooth volume-weighted level" effect. |
| Passes both, direction ≈ 50/50 | A volatility signal, not a directional one. It is relevant to long options later, but straddles are on hold by Ivan's decision. |
| Passes both, cross or bounce skewed | A directional signal, for a separate pre-registration. |

## 6. Synthetic validation

`tests/test_vwap_events.py`, 9 tests:
- exact VWAP and its reset at the anchor;
- grid reindexing;
- the detector uses only the level known at the bar's open;
- cooldown;
- the offset placebo is constant within each window;
- the bootstrap finds a known 25% effect;
- a null table does not pass;
- an end-to-end null run.

A full notebook run on 180 synthetic days with no VWAP effect gives ratios 0.97–1.02 and no gate passes.

## 7. Questions for ChatGPT

1. Is the 08:00 UTC session anchor defensible, or should UTC midnight replace it as primary? I kept two anchors to limit multiple testing.
2. Is a phase placebo with the same period the right control for the "anchor matters" question, or should the phase also be drawn within ±30 min of the real anchor?

## 8. Decisions needed from Ivan

1. Confirm or replace the 08:00 UTC session anchor.
2. Approve the windows and gates above before the first real run.
3. Later, and separately: whether to repeat this on the Deribit collector data (Deribit-only VWAP) for comparison.
