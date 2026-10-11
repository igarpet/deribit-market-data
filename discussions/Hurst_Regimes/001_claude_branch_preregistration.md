# 001 — Claude: Hurst Regimes branch — pre-registration of notebook 01

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-11 |
| **Branch** | Hurst Regimes (new) |
| **Responds to** | Ivan, 2026-10-11: "If we apply an HMM to the Hurst exponent, wouldn't we be able to identify/predict anti-persistence and persistence?"; approval to create the branch |
| **Status** | Protocol fixed in code (`hurst_regimes.py` v0.1.0). Tests 7/7. Notebook 01 executed on synthetic data only. No real-data output exists. |

## 1. Question and motivation

A 0DTE straddle pays on the terminal move. For a fractional process, the variance of the move over $n$ bars scales as $n^{2H}$, not $n$:
- $H<0.5$ (anti-persistent): the terminal move is smaller than path variance implies, so straddles are rich;
- $H>0.5$ (persistent): it is larger, so straddles are cheap.

VRP 008 found a long straddle loses about 27% of premium unconditionally and breaks even only if the realized move beats the implied move by about 1.35×. Over 288 bars that requires $\Delta H\approx0.05$ ($288^{2\Delta H}\approx1.8$ in variance).

Ivan's proposal is an HMM on $H$ to identify and predict persistent sessions. Before building one, three prerequisites must hold. Otherwise an HMM will find "states" that are estimator noise:

| Gate | Question |
|---|---|
| G1 variation | Does session $H$ vary more than estimator noise and volatility bursts produce at constant $H$? |
| G2 persistence | Does today's $H$ predict tomorrow's? |
| G3 relevance | Does today's $H$ predict tomorrow's terminal move per unit of path variance? |

**Source of the method.** The estimator is the change-of-frequency statistic from Ivan's thesis (§5.1; Kubilius & Skorniakov 2017). Applying fractional dynamics to volatility follows Wang, Xiao & Yu (2023).

## 2. Why rolling $H$ cannot be used directly

1. **Overlapping windows create persistence by construction.** That is the HSMM 016 failure mode: two-state fits appeared under a regime-free null. The gates therefore use **non-overlapping** session blocks.
2. **Estimator noise.** On 288 bars of exact fGn, the session estimate has SD ≈ **0.078** (verified, unbiased at $H=0.3,0.5,0.7$). That is larger than the effect we need.
3. **Volatility bursts mimic $H$ variation.** In a synthetic test at constant $H=0.5$ with intraday volatility bursts, the SD of session $H$ rose from 0.078 to **0.13**. A naive test would call that regimes. The null below absorbs it ($p_1=0.73$).

## 3. Pre-registration (binding)

| Item | Value |
|---|---|
| Data | `DERIBIT_DATA_ROOT/history/price_5m/<ASSET>.csv.gz` (Binance spot closes, `tools/download_price_history.py`). Shared base data only, no other branch's outputs |
| Session | 288 bars ending in (d−1 08:00, d 08:00] UTC; valid if ≥ 280 finite returns (missing set to 0) |
| Deseasonalization | Per-asset 288-slot RMS clock, development blocks only, frozen for confirmation |
| Estimator | $H=\tfrac12\log_2\!\big(\sum D_2^2/\sum D_1^2\big)$ on $X=\sum r/s$, clipped to [0.01, 0.99]. $H_{10m}$ (every second bar) reported for bid-ask bounce, no decision |
| Outcome | $VR_d=(\sum r)^2/\sum r^2$ on deseasonalized returns of session $d$ |
| Null | fGn at the asset's development median $H_0$, multiplied by each block's centered 12-bar RMS volatility path; 400 replicates, seed 9001, same estimator |
| G1 | $p_1$ = MC upper-tail p of $\operatorname{Var}(H_d)$. Pass: Holm < 0.05 in ≥ half of eligible assets **and** median excess SD ≥ 0.02 |
| G2 | $\rho_2$ = Spearman$(H_d,H_{d+1})$ over consecutive sessions. Pass: Holm < 0.05 in ≥ half **and** median $(\rho_2-\bar\rho_2^{null})$ ≥ 0.10 |
| G3 | $\rho_3$ = Spearman$(H_d,\log VR_{d+1})$. Pass: Holm < 0.05 in ≥ half **and** median excess $\rho_3$ > 0 |
| Eligibility | ≥ 200 valid development blocks (≥ 100 in confirmation) |
| Development | Sessions ≤ 2026-03-31 |
| Confirmation | Sessions ≥ 2026-04-01; G3 only, development clock and $H_0$ frozen; run once (`HR_CONFIRM=1`) after Ivan accepts the development readout |
| Verdict | **PROCEED_TO_HMM** iff G1, G2, G3 pass on development **and** G3 passes on confirmation. Otherwise **STOP** |

Reported, no decision: per-asset $H$ mean and SD, null SD, $H_{10m}$, and mean $VR_{d+1}$ by tercile of $H_d$ (the economic size of G3).

**If PROCEED.** The HMM would not be fitted on rolling $H$. Its emissions would be fractional Gaussian noise with a state-specific $H$ and volatility, fitted on returns. It would then be tested as a conditioner of the VRP 02 long straddle, where the 1.35× hurdle applies.

## 4. Verification

- `tests/test_hurst_regimes.py`: 7/7 pass. They cover:
  - estimator unbiased and scale-invariant;
  - session alignment to 08:00;
  - missing-bar rule;
  - Holm;
  - decision rules.
- Self-test on synthetic data:

  | Case | Result |
  |---|---|
  | Constant $H$ | $p_1=0.87$ |
  | Constant $H$ + volatility bursts | $p_1>0.05$ |
  | Persistent regimes (0.35/0.65 in 20-day runs) | G1 $p_1=0.016$ (minimum at 60 reps), $\rho_2=0.66$, $\rho_3=0.57$, all detected |

- Notebook 01 ran end-to-end on a synthetic two-year random walk with gaps: development and confirmation both returned STOP, as they should. Full run time is about 6 s per asset at 400 replicates. Synthetic outputs were deleted.

## 5. Questions for ChatGPT

1. Is the volatility-modulated fGn null strong enough? Volatility that co-moves with the sign of returns (leverage) is not in it.
2. Should G3 also use an intraday outcome closer to the option horizon (entry 12:00 → 08:00), or is the 24h block adequate as a prerequisite?

## 6. Decisions needed from Ivan

1. Accept the pre-registration, or amend it **before** the first real run.
2. Run notebook 01 (commands in `notebooks/Hurst_Regimes/README.md`). It needs only the history files already downloaded for VD 10c.
