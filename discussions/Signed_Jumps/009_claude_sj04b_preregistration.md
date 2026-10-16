# Signed Jumps 009 — SJ 04b preregistration: a real-time seconds trigger

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-16 |
| **Branch** | Signed Jumps |
| **Responds to** | Signed Jumps 008 §4 (P3 withdrawn for look-ahead); Ivan's approval |
| **Status** | Preregistration. Fixed before any real-data output. Code `signed_jumps_trigger.py` (SJ04b-0.1.0), tests `tests/test_signed_jumps_trigger.py` (5 pass) |

## 1. Question

If a system detects a burst **as it happens**, using only past seconds, and enters within a few seconds, does price keep going? This replaces SJ 04 P3, which conditioned on the jump minute's completion.

## 2. Trigger (strictly real time)

**Volatility scale.** $\sigma_{\min}(T)$ is the SJ 01 engine's 1-minute return s.d. for the minute ending at $T$, i.e. seasonal factor × $\sqrt{\hat\sigma^2_c}$. It is forecast with data through $T-1$ min and fitted on SJ 01's DEV, unchanged.

**Statistic.** For a second $\tau$ inside that minute:
$$R_{60}(\tau)=\log P_\tau-\log P_{\tau-60\,\text{s}},\qquad z(\tau)=R_{60}(\tau)/\sigma_{\min}(T)$$

**Firing rule:**
- fire at the first $\tau$ with $|z|\ge c_{\text{trig}}$ and $|R_{60}|\ge 5$ bp;
- then disarm for 300 s;
- direction $d=\operatorname{sign}R_{60}(\tau)$.

No information after $\tau$ is used. A test checks this: perturbing future prices leaves earlier triggers unchanged.

## 3. Outcome and gate

$$G(\delta,H)=d\left(\log P_{\tau+\delta+H}-\log P_{\tau+\delta}\right)\cdot10^4\ \text{bp}$$

Spot 1s closes; Binance spot 1s, BTC/ETH/SOL/XRP, 2026-07 → 2026-09.

**Primary:** $c_{\text{trig}}=4$, latency $\delta=2$ s, hold $H=60$ s.

| Verdict | Condition |
|---|---|
| `CONTINUATION` | gross day-block bootstrap q2.5 > 0 |
| `NO_CONTINUATION` | otherwise |

**Economic flags:** gross ≥ 7 bp (cheapest perp taker, Deribit base tier) and gross ≥ 12 bp (project standard).

**Reported only:**
- the grid $c\in\{3,4,5\}$ × $\delta\in\{1,2,5,10,30\}$ s × $H\in\{10,30,60,300\}$ s;
- share of positive outcomes;
- per-asset and per-direction splits.

**Why δ = 2 s.** It is a realistic end-to-end latency for a retail system: feed, decision and order. From Europe to Binance (Tokyo) the round trip alone is about 0.2–0.3 s, and a Tokyo VM would be a few ms.

## 4. Known limitations

1. **Fill price.** The 1s close is the last trade in that second. During bursts spreads widen and the book thins, so real taker fills are worse. Gross results are an **upper bound**.
2. **Spot prices; a trade would be on perps.** Perps track spot within 1–6 bp (PF 004).
3. **Sample.** Three months, all inside SJ 01's TEST period. The trigger has no fitted parameters of its own; σ comes from the SJ 01 engine fitted on DEV.

## 5. Synthetic validation

| Check | Result |
|---|---|
| Triggers fire on planted jumps | ~1 per jump; all with $\lvert z\rvert\ge4$ and $\lvert R_{60}\rvert\ge5$ bp |
| No post-jump drift | `NO_CONTINUATION` |
| Planted 30 bp drift over 120 s | `CONTINUATION`, > 8 bp; entering at 30 s captures less than at 2 s |
| Causality | perturbing future prices leaves earlier triggers unchanged |
| σ scale | matches the planted 1s volatility |

## 6. Expectation (stated in advance)

- **Verdict:** `CONTINUATION` likely, because bursts last tens of seconds (SJ 008 §3) and a 4σ trigger usually fires mid-burst.
- **Size at δ = 2 s, H = 60 s:** 2–6 bp gross, falling toward 0 by δ = 10–30 s.
- **Economics:** gross below 7 bp, i.e. not tradeable at taker costs before even accounting for burst slippage.

## Decisions needed from Ivan

1. Approve and run. Expect about 5–10 minutes.
