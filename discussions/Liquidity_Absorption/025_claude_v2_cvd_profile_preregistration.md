# Liquidity Absorption 025 — V2 pre-registration: CVD absorption, 18:00-ET volume profile targets, engine stop

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-20 |
| **Branch** | Liquidity_Absorption |
| **Responds to** | Ivan (2026-10-20): the strategy as tested omits his CVD and volume-profile rules; his choices: CVD divergence between touches, profile reset at 18:00 ET, TP at the far side of the LVN (next HVN), his MNQ stop/BE/trail rules crypto-scaled with the engine, +15 min losing cut as a separate test |
| **Status** | Pre-registration. Committed **before the EXT results (commit 74334cf) were opened** and before any V2 output exists. Code: `liquidity_absorption_v2.py` (LA-V2-0.1.0); notebook `V2_cvd_profile.ipynb`; outputs `V2/` |

## 1. What changes from LA 024 (everything else is identical)

**Signals:** the EXT A02 second-touch candidates (frozen detector) on Deribit perpetual 1m bars.

| Element | LA 024 (A03b primary) | **V2** |
|---|---|---|
| Absorption | not used | **CVD divergence between touches** (below) |
| Take-profit | session VWAP if ≥ 2R, else opposite prior-day extreme | **far side of the first LVN, i.e. near edge of the next HVN**, from the 18:00 ET session profile |
| Stop | mirrored to the target distance (1:1) | **farthest of** beyond the zone edge, beyond the 2nd-touch wick, and 1 × VD18 60-min σ from entry (A03d `risk_stop`) |
| Management | FIXED | **TRAIL_1M** (A03 rules): stop to entry at +1R, then 2 ticks beyond each confirmed 1m higher low (long), never loosened; 240-min time stop |

## 2. Rules (exact)

**CVD divergence (absorption).**
- **Flow:** signed quote flow = Σ (taker-buy USD − taker-sell USD) over the 1m bars that close **after the first-touch candle, up to and including the second-touch candle**.
- **Long:** flow < 0 (sellers kept hitting) **and** second-touch low ≥ first-touch low − 1 zone width (price held).
- **Short:** flow > 0 **and** second-touch high ≤ first-touch high + 1 zone width.

**Session profile.**
- **Window:** prints from the most recent **18:00 America/New_York** (DST-aware: 22:00 UTC in summer, 23:00 UTC in winter) to the signal close, strictly causal.
- **Binning:** traded USD volume in log-price bins of **5 bp**, smoothed with a 3-bin centred moving average. POC = the largest bin.
- **Young session:** if it is under **120 min** old at the signal, the result is `NO_PROFILE_SESSION_TOO_YOUNG`. No trade, reported.

**Target.**
1. Start at the signal-candle close, at least **10 bp** beyond it in the trade direction.
2. Walk outward to the first **LVN**: a run of bins below **20% of the POC**.
3. Continue to the first bin at or above **50% of the POC** — the next **HVN**.
4. **TP = the near edge of that HVN bin.**
5. Leaving the profile range first gives `NO_PROFILE_TARGET`. No trade, reported.

**Entry, fills, fees.**
- **Entry:** the first print after the signal close within 60 s, as in A03.
- **Adverse fills:** as in A03 (entry 0.5, target 0.5, stop 1.5, timeout 0.5 bp).
- **Fees:** Deribit, taker 5 bp per side. Funding is not modelled.
- **Exit check:** stop before target on each print, as in A03.

## 3. Variants reported

- **Management:** FIXED / BREAKEVEN / TRAIL_1M.
- **CUT15:** each of the above with the cut — at the first print ≥ 15 min after entry, exit at the market if losing.
- **CVD filter:** every variant with the filter on and off.

That is 12 variants. Only one is binding.

## 4. Gate (binding, primary only)

**Primary:** CVD on, TRAIL_1M, no CUT15.

- **Pass:** mean net bp per closed trade $>0$ **and** its one-sided 95% day-bootstrap lower bound $>0$, in **BTC-H1, BTC-H2, ETH-H1 and ETH-H2**, plus the holdout lower bound $>0$ per symbol.
- **Periods:** H1 = signals before 2025-04-01. H2 = from 2025-04-01, **excluding** the in-sample Jul–Sep 2026 quarter, which is reported separately.

**CUT15 and the other variants are secondary hypotheses.** Because CUT15 comes from Ivan's observation on Jul–Sep 2026 (A04), its holdout result is a legitimate confirmation test. It is still not promoted unless it would pass the same gate on its own.

## 5. Kelly (descriptive)

- **Per trade:** net R = net bp / initial risk bp.
- **Per variant and cell:** $f^* = \arg\max_f \overline{\log(1 + fR)}$ (fraction of equity at risk per trade) and the log growth.
- **Reading:** $f^* > 0$ only if the mean net R is positive. The engine enters through the stop width (the risk unit), not as a $\mu/\sigma^2$ denominator.
- **Use:** sizing is only meaningful for a variant that passes.

## 6. Rule for "patterns in how moves happen"

Anything discovered from EXT/V2 trade outcomes — features at +15 min, at entry, or in the −240/+240 context — is mined on **2024-01-01 → 2025-06-30** only. It is confirmed, unchanged, on **2025-07-01 → 2026-06-30 plus 2026-10-01 → latest**. Jul–Sep 2026 stays out of both. A pattern that does not confirm is dropped.

## 7. Expected issues (stated in advance)

- **Sparse ETH profile:** the ETH perpetual's print density (about 0.4/s) gives a coarse profile. Many 5 bp bins near the session range edges will be thin, which creates "LVNs" that are thin data rather than thin interest.
- **Many skipped signals:** `NO_PROFILE_TARGET` and young sessions may remove a large share of signals. Shares are reported, not tuned.
- **HVN as TP:** if Ivan's reading is right (resting orders at the LVN/HVN boundary), target hits should exceed what a random barrier at the same distance gives.

## 8. My prediction (opinion)

`FAIL`. The CVD filter selects a subset with no different outcome distribution, because 1m taker flow at the level carried little information in A02b/A06-type checks. Targets further away (median R:R about 2) lower the hit rate roughly in proportion.

## 9. Running it (PC, Windows CMD), after the EXT run

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest tests.test_liquidity_absorption_v2 -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=86400 notebooks\Liquidity_Absorption\V2_cvd_profile.ipynb
git add notebooks\Liquidity_Absorption
git commit -m "LA V2 run"
git push
```

## Decisions needed from Ivan

1. Approve this pre-registration, including the 5 bp bins, 20% / 50% POC thresholds, 120-min minimum session age and 10 bp minimum target. These are my translations of "little activity" and "high-volume node". Change them **now**, before any V2 output exists, not after.
