# Regime Structure 001 — branch pre-registration

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-13 |
| Branch | Regime Structure |
| Responds to | Ivan, chat of 2026-10-13: approved the label definitions (shock changed to 3σ), assets BTC/ETH/SOL/XRP, no VWAP |
| Status | Pre-registration, before any real-data run |

## 1. Question

Before any feature search or HSMM: **do Ivan's regimes (range / mean reversion, displacement, noise, shock) exist on crypto as persistent states, beyond what volatility clustering alone produces?** And do the transitions he describes from discretionary trading appear?

The cycle he describes:
- range ends in displacement;
- displacement fades into noise;
- shocks come "out of thin air", especially after a big upward displacement.

VWAP is not used. VWAP 002 found VWAP touches are not special on crypto.

## 2. Data (independent branch)

- Binance USD-M perpetual 1-minute klines, BTC, ETH, SOL and XRP, 2024-10-01 → 2026-10-11 (`tools/download_binance_um_klines.py`).
- Aggregated to 5-minute bars; labels are computed on clock-hour windows of $n=12$ bars.
- CVD comes from `taker_buy_volume`: per-minute delta $=2\,V^{\text{buy}}-V$.

No other branch's outputs, thresholds or state definitions are used.

## 3. Labels (fixed in `src/data_engineering/modeling/regime_structure.py`, `regime_structure v1.0`)

Per clock hour $h$, with $r$ the 5-minute log returns in the hour:

$$ER=\frac{\lvert\sum r\rvert}{\sum\lvert r\rvert},\qquad s=\hat\sigma_{\text{pre}}\sqrt{12},\qquad \text{shock}=\frac{\max\lvert r\rvert}{\hat\sigma_{\text{pre}}}$$

- $\hat\sigma_{\text{pre}}$ is an EWMA of $r^2$ (half-life 24 bars) measured at the bar *before* the hour, so it is causal.
- Relative volume and relative realised volatility are each divided by the development-window median for the same UTC hour.

Labels, first match wins:

| Label | Rule |
|---|---|
| SHOCK | shock $\ge 3$ (Ivan: 3σ) |
| DISP | $ER\ge q_{67}(ER)$ and $\lvert\text{net}\rvert\ge 1.0\,s$ and sign(window CVD change) = sign(net) |
| NOISE | relative volume $\le q_{25}$ and relative realised vol $\le q_{33}$ |
| RANGE | $ER\le q_{33}(ER)$ and $\lvert\text{net}\rvert<1.0\,s$ |
| OTHER | otherwise |

Quantiles are per asset, taken on the development window of the real data, then frozen and applied unchanged to the null.

**Operationalisation note.** "Price contained between the rolling high and low" is implemented as low efficiency plus a net move under one hourly sigma: price moved a lot but went nowhere. Volume concentrating at the range bounds is not part of the label in this module.

## 4. Null

Every 1-minute log return, and its taker delta, is multiplied by an independent random sign. This keeps exactly:
- 1-minute absolute returns (volatility clustering),
- volume (activity clustering),
- time-of-day structure.

It destroys all directional dependence: trend, reversion, and persistence of CVD agreement with price. Persistence that survives the null is volatility or activity in disguise. Persistence beyond the null is directional structure, which is what range and displacement claim to be.

Limits:
- NOISE depends mostly on volume and volatility, so its own persistence is largely preserved by construction. That is why it is not tested by itself, only as the target of C2.
- The null also removes genuine 1-minute microstructure (for example bid-ask bounce). That mainly shifts base rates, not hour-to-hour persistence.

## 5. Tests, windows and decision rule

- Development: data start → 2026-03-31, for thresholds and display. **Test: 2026-04-01 → end; gates are evaluated here only.**
- Lift for transitions between consecutive, non-overlapping hours: $L(X\to Y)=P(Y_{h+1}\mid X_h)/P(Y_{h+1})$.
- A lift needs at least 30 "from" hours. A null draw below that counts as $L=1$, meaning no evidence of persistence.
- Pooled over assets as the **geometric mean of per-asset lifts**. Summing raw counts across assets with different base rates creates spurious lift; the synthetic smoke run showed this, and `TestPooling` covers it.

| ID | Transition | Ivan's claim |
|---|---|---|
| E1 | RANGE → RANGE | ranges persist |
| E2 | DISP → DISP | displacements persist |
| C1 | RANGE → DISP | a range ends in displacement |
| C2 | DISP → NOISE | displacement fades into noise |
| C3 | up-DISP → SHOCK | shocks follow a big upward displacement |

- 199 null draws per asset; the null is pooled the same way, draw by draw.
- One-sided permutation p-value $p=(1+\#\{L_{\text{null}}\ge L_{\text{real}}\})/200$.
- **Pass:** $p\le 0.05/5=0.01$ **and** $L_{\text{real}}/\operatorname{median}L_{\text{null}}\ge 1.10$.

Descriptive outputs, with no gates: per-asset lifts, full 5×5 transition matrices for real and null, run lengths, base rates.

## 6. Interpretation, written before results

| Result | Reading |
|---|---|
| E1 and E2 fail | Range and displacement are not directional states on crypto at the 1-hour scale; what persists is volatility. The HSMM would only be recovering volatility states, which is consistent with the history of this project. |
| E1 and/or E2 pass, C1–C3 fail | The states exist, but the cycle Ivan describes does not hold here; transitions are not structured the way discretionary experience suggests. |
| E1, E2 and some Cs pass | The regime picture has empirical support. The next module asks which ex-ante features identify the current state earliest (feature AUC against these labels). |

## 7. Synthetic validation

`tests/test_regime_structure.py`, 6 tests:
- hour alignment and sums;
- the null preserves volume and changes direction;
- label priority, and that disagreeing CVD prevents DISP;
- planted hourly trend/range states make E1 and E2 pass;
- no planted structure means nothing passes;
- geometric-mean pooling is unbiased under heterogeneous base rates.

The notebook smoke run on synthetic data, with regimes planted in 2 of 4 assets, gives E1 and E2 passing at $p=0.01$ and every C test failing, as designed.

## 8. Questions for ChatGPT

1. Is a per-minute sign flip the right null, or should the flip be at 5-minute blocks? A 1-minute flip also destroys within-5-minute reversion.
2. One-hour windows: should a 2-hour or 4-hour window be run as a robustness check (descriptive only) after the readout?

## 9. Decisions needed from Ivan

1. Approve the operationalisation of RANGE in §3, which as written ignores volume at the bounds.
2. Approve the windows and gates in §5 before the run.
