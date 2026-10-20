# Liquidity Absorption 027 — W01 pre-registration: anatomy of the 1:1 trades, "CVD explodes toward the TP"

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-20 |
| **Branch** | Liquidity_Absorption |
| **Responds to** | Ivan (2026-10-20): study the A03b 1:1 variant (most trades, highest target rate) with every data layer available; his observation: *in winners the CVD explodes toward the take-profit*; he trades without L2 |
| **Status** | Pre-registration, committed before any W01 output exists. Code `liquidity_absorption_w01.py` (LA-W01-0.1.0); notebook `07_winner_anatomy_1to1.ipynb`; outputs `07/` |

## 1. Population and splits

**Population:** every closed EXT A03b forward 1:1 taker trade (TARGET / STOP / TIMEOUT), Deribit BTC- and ETH-PERPETUAL, 2024-01-01 → latest day. That is 2,280 trades, 632 of which hit the target. Cost = the trade's Deribit-repriced cost (fees + the engine's adverse fills).

**Splits (frozen):**

| Split | Signals | Use |
|---|---|---|
| MINING | before 2025-07-01 | the only split where patterns are searched |
| CONFIRM | 2025-07-01 → 2026-06-30, and from 2026-10-01 | confirmation only |
| INSAMPLE | Jul–Sep 2026 (inspected since A02) | reported separately, never used to confirm |

**Baseline fact to keep in view.** The 1:1 variant's target rate among resolved trades is $632/(632+670)=48.5\%$, i.e. a coin toss. Most of the "higher win rate" relative to A03 comes from the wider 1:1 stop, not from selection.

## 2. Layers (per trade, all signed toward the target, R = the trade's initial risk in bp)

| Layer | Content |
|---|---|
| Trade record and A02 setup | level type, target source, zone width and touch features, VD18 $\sigma_{60}$, $R/\sigma_{60}$, hour (New York), weekday |
| Session volume profile | from the last 18:00 New York to entry, using the V2 code. Fields: POC position in R, profile mass between entry and target, thinnest bin on the path (LVN), HVN on the path, target outside the session range, value-area width |
| Perpetual tape, per minute | price, high/low, taker flow, volume, large-print flow (≥ the day's 99th-percentile print), print count |
| Deribit context, per minute (A06) | liquidations with / against, premium, blocks, nearest-future basis, ATM IV, skew, option delta flow and volume |

**Normalisation.** $F_{60}$ is the mean traded USD per minute over the hour before entry. Per-minute flow is

$$\text{flow}_t=\frac{s\,(\text{buy}_t-\text{sell}_t)}{F_{60}}$$

i.e. net aggressive volume in minutes of normal volume, with $s=\pm1$ the trade direction. The 3-minute burst statistic is

$$f3_t=\tfrac13\sum_{k=t-2}^{t}\text{flow}_k$$

## 3. The mechanical problem with "CVD explodes toward the TP in winners"

A 1:1 target is about 50–100 bp away. On a perpetual, price moves that far mostly through aggressive orders in that direction. So, looking back, winners showing CVD toward the target is close to true by construction, and it says nothing about what is predictable.

The only usable form of the observation is causal: **at a time $L$ after entry, among trades still open, does the CVD so far predict the rest of the trade, beyond what price progress already says?**

At landmark $L$ the choice between holding and exiting is worth

$$\text{remaining}_L = R_{\text{final}} - \text{prog}_L$$

gross. Both choices pay the same entry and exit costs, so this is also the net value of holding.

## 4. Hypotheses and gates

**H1 (binding, primary $L=15$).** CVD toward the target over $[0,15)$ min has a **positive partial rank correlation** with $\text{remaining}_{15}$, given $\text{prog}_{15}$.
- **Supported** if the MINING estimate is $>0$ and the CONFIRM day-bootstrap one-sided 95% lower bound is $>0$.
- $L\in\{5,30,60\}$ are reported, not gated.

**Management rules (frozen thresholds; economic test).** Exit at $L$ if:
- (a) $\text{cvd}_L$ is in the MINING bottom tercile;
- (b) Ivan's rule: the trade is losing at $L$.

Reported per split: net R per trade, its lower bound, and the paired improvement over hold-to-exit. The strategy-level bar stays the LA 024 bar: CONFIRM net mean $>0$ with lower bound $>0$.

**Burst study (secondary).**
- **Burst threshold:** the 95th percentile of $f3$ over MINING pre-entry context minutes (frozen from MINING).
- **Reported:** for the first burst toward the target while the trade is open — share of trades that have one, onset time, price progress at onset (R), time to exit.
- **Add-on unit:** the net R of a new 1:1-sized unit added at burst onset, $R_{\text{final}}-\text{prog}_{\text{onset}}-\text{cost}_R$, with its lower bound.
- **Reading:** if winners' bursts arrive at about $+0.8R$, the burst *is* the move and is not actionable.

**Screen (exploratory).**
- **Feature sets:** all entry-time features, and separately all $L=15$ features.
- **MINING test:** TARGET vs STOP Mann–Whitney, BH-FDR at $q\le0.10$ within each set.
- **Confirmation:** survivors must keep their sign on CONFIRM with Holm-adjusted one-sided $p<0.05$.
- Anything not confirmed is not a pattern.

## 5. Known limits

- **Minute 0** includes prints from that minute just before entry.
- **No order book:** resting liquidity at LVN/HVN is not observed, only inferred.
- **The 1:1 geometry is A03b's mirrored stop**, i.e. risk = target distance (median about 95 bp), not Ivan's discretionary stop.
- **Timeouts** (43% of trades) are kept in all R statistics and excluded only from the TARGET vs STOP contrasts.

## Questions for ChatGPT
1. Is partial rank correlation on $\text{remaining}_L$ given $\text{prog}_L$ the right control? Or should MFE-to-date also be partialled out?
2. Does the 95%-of-context threshold for a burst seem reasonable, or should it be a level relative to $\sigma_{60}$?

## Decisions for Ivan
1. Accept the MINING/CONFIRM split and H1 as the binding hypothesis before the run.
2. If H1 fails, the CVD-burst observation is retired as a management signal for this setup.
