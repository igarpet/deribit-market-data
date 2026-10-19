# Jump Fade 002 — JF 01 readout: no reversion beyond the null; the mirrored (breakout) trade is no better

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-19 |
| **Branch** | Jump Fade |
| **Responds to** | JF 001; Ivan's run 85a989b |
| **Status** | Result. Registered run (199 nulls). The verdict in §1 is binding. §3 is an exploratory mirror test that Ivan asked for, knowing it is post hoc. |

## 1. Verdict: `NO_REVERSION` (all three splits)

Data: BTC, ETH, SOL and XRP perpetuals, 2024-10-01 → 2026-10-11.

TEST (2026) has 42,779 trades on *all*, about 150 a day across 4 assets. Costs are Deribit's.

| Split | Gross bp/trade | Null median (p95) | p | Take-profit / stop / time | Net bp/trade | Net bp/day [95% CI] |
|---|---|---|---|---|---|---|
| all (primary) | −0.91 | −0.95 (−0.65) | 0.40 | 47% / 51% / 2% | −4.11 | −154 [−171, −137] |
| burst | −0.84 | −1.23 (−0.69) | 0.125 | 48% / 52% / 0% | −3.96 | −44 [−55, −33] |
| recent_jump | −0.69 | −1.47 (−0.83) | **0.02** | 48% / 52% / 0% | −3.83 | −33 [−41, −25] |

- **Per asset:** gross is negative on 4/4 assets in every split and period, and net is negative everywhere. DEV is the same.
- **Shape:** an average win of about +25 bp against an average loss of about −25 bp (TEST), with a 47–48% take-profit rate. That is a coin flip minus the fill-model bias.
- **Near-miss:** *recent_jump* beats its null at p = 0.02, just under its 0.025 bar. It still fails G1, because gross is negative on every asset and below zero overall. The +0.8 bp relative to the null is a small, real tilt toward reversion just after jumps. It is about 5× too small to pay the taker exits.
- **Prediction recorded in JF 001 §7** was `NO_REVERSION` / `REVERSION_NOT_TRADEABLE`. The result came out as predicted, except that *recent_jump* was not worse than *all*.

## 2. Why it loses (mechanics)

Ignoring costs, on a martingale the take-profit and the stop sit symmetrically (1.5σ each side of the entry), so hit rates are about 50/50. The conservative fill rules then cost about 1 bp (the null).

Costs are asymmetric. Take-profits are maker (0 bp on Deribit); stops and time exits are taker (6 bp). About 52% of trades pay 6 bp, so the average cost is about 3.1 bp per trade. That is the whole −4.1 bp net.

## 3. Exploratory (post hoc, Ivan's request): the mirrored trade

**Construction:** take the other side of every fade trade.
- Entry is a stop order (taker) at ±1.5σ, in the direction of the move.
- Your stop is at $P_0$ (taker); your take-profit is at 3σ (maker); the time exit is taker.
- Computed from the JF 01 trade file. Mirror gross = −fade gross.

| | DEV | TEST |
|---|---|---|
| Mirror gross bp/trade | +0.96 | +0.92 |
| Mirrored null median (p95) | +1.06 (+1.28) | +0.95 (+1.25) |
| p vs null | 0.70 | 0.61 |
| Net, Deribit (bp/trade) | −7.98 | −8.01 |
| Net, Binance (bp/trade) | −9.00 | −9.04 |

The positive gross is entirely the fill-model bias reversed: the null shows the same +0.95 bp. There is no momentum either.

Net is worse than the fade, because the mirror pays taker on entry *and* on most exits. TEST net is about −300 bp/day of notional.

Two further points:
- The mirror gross *overstates* reality. A real stop-entry fills beyond the trigger, so slippage is ≥ 1 bp, and the conservative ordering of the fade becomes favourable ordering for the mirror.
- This test is uninformative by construction. The fade's gross was indistinguishable from the null, so its mirror must be too. It costs nothing to show, and confirms there is no hidden momentum edge.

## 4. Consequence (opinion)

Short-horizon fades and breakouts around ±1.5σ on these perpetuals are a coin flip at 1-minute resolution. Volatility, bursts and recent jumps do not change that by more than about 1 bp.

This agrees with SJ02a–04, VE03/04 and CS03e. I recommend closing the branch. The only directional tilt seen, about +0.8 bp more reversion just after jumps, is too small to monetize at any fee level that includes a taker leg.

## Decisions needed from Ivan

1. Accept JF 01 and close the branch. (Recommended.)
