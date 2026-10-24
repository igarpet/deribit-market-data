# 001 — Claude: VWAP Drift 01 pre-registration — conditional drift toward VWAP in volatility-engine units

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-24 |
| **Branch** | VWAP Drift (new, independent) |
| **Responds to** | Ivan, 2026-10-24: thesis-style convergence trade with VWAP as the mean. Buy one option toward VWAP, Z-score from the volatility engine, conditioned on volatility regime and jumps, exit on Z dynamics. Approved: build and test at 4 horizons. |
| **Status** | pre-registration + code (validated on synthetic data); awaiting Ivan's run |

## 1. Why this test comes first

A long call or put toward VWAP is profitable only if price actually drifts toward VWAP by more than the premium plus the spread. Volatility, jumps and exit rules decide *when* to look; they cannot produce the drift.

- **Exit rules.** Under a martingale with fairly priced options, the optional stopping theorem says no exit rule can change expected P&L.
- **Volatility.** Volatility clustering predicts the size of a move, not its sign.

The prerequisite is therefore

$$
\mathbb E\big[\,y_h\;\big|\;\text{state}_t\big]>\text{costs},\qquad y_h=-\operatorname{sign}(D_t)\,\big(\log C_{\text{exit}}-\log C_{t+1}\big)\cdot10^4,\qquad D_t=\log C_t-\log \mathrm{VWAP}_t .
$$

It is measured on the perpetual, with no option involved.

**Relation to earlier work:**
- **VWAP Events VE03/VE04.** These faded displacement with fixed bp targets and stops, using an EWMA $\sigma$. They found no edge, and real anchors behaved like placebo anchors.
- **What is new here:**
  1. Displacement in engine units.
  2. Conditioning on the engine's volatility regime and on recent jumps (Ivan's hypothesis).
  3. Pure drift at fixed horizons, including to the 08:00 option expiry, instead of a target/stop trade.

## 2. Data and engine

- **Price.** Binance USD-M perpetual 1-minute klines, BTC, ETH, SOL and XRP (HYPE and TRX excluded per Ivan). Same files as VWAP Events (`BINANCE_UM_DIR/<A>USDT_um_1m.parquet`).
- **VWAP.** Anchored VWAP through bar $t$ is cumulative quote volume / volume since the anchor.
  - **Real anchors:** session (08:00 UTC, the Deribit expiry) and 4H (00/04/…/20 UTC).
  - **Placebo anchors:** same period, unused phases; session at 02/14/20 UTC, 4H at +1/+2/+3 h (`vwap_touch_stats.SPECS`).
- **Volatility engine.** The **frozen** VD18 HAR_4 coefficients (`notebooks/Volatility_Dynamics/18/18_manifest.json`), no refit. This is the variance level of AM-T-HAR-Blend, whose blend weight is ≈1 at 30–240 min.
  - **Inputs:** 5-minute closes built from the perpetual klines (bar END).
  - **Disclosed transfer:** VD18 was fitted on spot. The spot–perp basis s.d. is about 1 bp (Pairs fOU 004), so this is immaterial for variance.
  - **Fit overlap:** the engine was fitted on TRAIN, which overlaps this branch's DEV window. It contains no direction information, so it cannot leak a drift.

## 3. Origins, states, outcome

- **Origins:** every minute whose bar END is on the 5-minute grid, at least 30 min after its anchor.

**Definitions:**

| Symbol | Definition |
|---|---|
| $v_h$ | Engine forecast of net-return variance over $h\in\{15,60,240\}$ min, from the 5m bar ending at $t$ |
| EXP | Horizon to the next 08:00 UTC, $\tau\in[15,1440]$ min |
| $Z_h$ | $D_t/\sqrt{v_h}$ |
| zb | $\lvert Z\rvert$ bucket: `z0_1` (descriptive only), `z1_2`, `z2_3`, `z3p` ($\ge3$) |
| vol | Tercile of $\log v_{60}$; cut-offs from the asset's DEV origins only |
| jump | $\max\lvert r^{1m}\rvert$ over the last 60 min $\ge 4\sqrt{v_{15}/15}$ (the SJ 04b-style trigger) |

**Variance to the EXP horizon:** log-linear interpolation of the per-minute rates $v_h/h$ over $\log h$, flat beyond 240 min. This is an approximation and is used only to scale $Z$.

**Outcome:**
- **Entry** at the close of bar $t+1$ (one-minute latency, as in VWAP Events).
- **Exit** $h$ minutes later, or for EXP at the bar ending 08:00.
- $y_h$ is in bp; positive means price moved toward VWAP.

**Reported only:**
- **Hit:** price reached the decision-time VWAP level after entry, from 1m highs and lows.
- **Driftless benchmark:** $p_0=2\,[1-\Phi(\lvert\log C_{t+1}-\log \mathrm{VWAP}_t\rvert/\sigma_h)]$, together with the placebo hit rate.
- On synthetic martingales the hit rate is within about 0.02 of $p_0$ at 60 min (test tolerance 0.05).

## 4. Decision (binding)

- **Cells:** 144 real cells = anchor (2) × horizon (4) × zb (`z1_2`, `z2_3`, `z3p`) × vol (3) × jump (2).
- **Windows:** DEV is before 2026-04-01; TEST is from 2026-04-01 to the end of the data.
- **Bootstrap:** day-block, which keeps overlapping origins and all assets of a day together. 2,000 draws.

**Selection on DEV:** the real cell with the highest q2.5 of mean $y$, among cells with ≥ 30 days and ≥ 500 origins.

**Gates on TEST (selected cell only).** Fewer than 20 TEST days gives `INSUFFICIENT_TEST`.

| Gate | Rule |
|---|---|
| **G1, VWAP-specific** | real − placebo, matched within 6-hour blocks, q2.5 > 0 |
| **G2, tradeable on perp** | real mean $y$ q2.5 > 7 bp |

**Verdict:**

| Gates passed | Verdict |
|---|---|
| G1 and G2 | `VWAP_DRIFT_EDGE` |
| G1 only | `VWAP_SPECIFIC_BUT_UNTRADEABLE` |
| G2 only | `GENERIC_DRIFT_NOT_VWAP` |
| Neither | `NO_EDGE` |

**Options gate (fixed now).**
- An ATM option ($\Delta\approx0.5$) pays $2\times0.03\%$ of the underlying in fees per round trip, i.e. ≥ 12 bp per unit of delta before the spread.
- An options module opens only if **G1 and G2 pass** and the TEST mean exceeds 12 bp plus the measured half-spreads/$\Delta$ (measured from the collector data at that point).
- Otherwise no option work follows from this branch.

**Secondary S1 (Ivan's volatility-regime hypothesis).**
- **Cells:** vol = high, zb ∈ {`z2_3`, `z3p`}, both jump states.
- **Tests:** each anchor × horizon, 8 tests.
- **Reported:** TEST real mean and hour-block-matched real − placebo.
- **Intervals:** Bonferroni two-sided at $1-0.05/8$.
- **Use:** reported with pass flags; it does not override the primary verdict.

**Descriptive outputs (no decision):**
- tables by |Z| bucket, by volatility tercile (|Z| ≥ 2) and by jump state (|Z| ≥ 2);
- all cells, real and placebo, DEV and TEST;
- hit vs $p_0$ vs placebo hit.

## 5. Not in scope here

- **Exit on Z dynamics.** Ivan's rule (sell when Z grows back or its acceleration toward 0 fades) is deferred. It can only matter if a drift exists, and it adds parameters. If G1 and G2 pass, it gets its own pre-registration, with the fixed-horizon result as the benchmark.
- **Option prices.** Deferred to the options gate above.

## 6. Validation (synthetic)

`tests/test_vwap_drift.py`: 14 tests, all passing.

- **Engine prediction:** equals `markov_har_variance.predict_har` to $10^{-12}$.
- **Causality:**
  - HAR features and origin states are unchanged when future bars are removed;
  - the 5-minute grid is at bar END.
- **Mechanics:**
  - variance interpolation;
  - expiry mapping, so every EXP exit is the 08:00 bar and $\tau\in[15,1440]$;
  - outcome arithmetic;
  - aggregates equal the per-origin rows.
- **Benchmark:** under a martingale, hit ≈ $p_0$.
- **Decision rule:**
  - Martingale (κ = 0): `NO_EDGE`. Across 6 additional seeds with 300 days, every verdict was `NO_EDGE`.
  - Injected pull toward the session VWAP ($r_t \mathrel{+}= -\kappa D_{t-1}$, κ = 0.01/min): positive drift at all horizons, and real beats the placebo phases.
- **Notebook:** runs end to end on synthetic data.

**Power caveat.** A pull toward the real VWAP also shows up partly at placebo anchors, because all VWAPs are recent averages of the same price. G1 is therefore conservative. This is intended, because a generic pull toward any recent average is not a VWAP effect (`GENERIC_DRIFT_NOT_VWAP`).

## 7. Run (PC, Windows CMD)

Close the notebook in Jupyter first.

```bat
cd /d D:\Projects\deribit_market_data
git pull
python tools\download_binance_um_klines.py --months 25 --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT
python -m unittest discover -s tests -p "test_vwap_drift.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebooks\VWAP_Drift\01_vwap_drift.ipynb
git add notebooks\VWAP_Drift
git commit -m "VWAP Drift 01: run"
git push
```

- **Download.** This only extends the existing files through the current month. Skip it if they already reach late October.
- **Runtime.** About 1–2 minutes per asset to build the aggregates, then a few minutes of bootstraps.
- **Outputs** go to `notebooks/VWAP_Drift/01/`:
  - `01_decision.json`, `01_dev_selection.csv`, `01_secondary_s1.csv`;
  - `01_by_zbucket.csv`, `01_by_vol_regime.csv`, `01_by_jump.csv`, `01_all_cells.csv`;
  - `01_coverage.csv`, `01_manifest.json`.

## 8. Questions for ChatGPT

1. Is hour-block matching (6-hour blocks) sufficient for G1, or should the placebo be matched on minutes-since-anchor as well? A VWAP's distance distribution depends on its age.
2. The per-horizon $Z_h$ uses a different $\sigma$ for each horizon, so the same displacement falls in different buckets at 15 and 240 min. Is a single $Z_{60}$ preferable for interpretability?

## 9. Decisions needed from Ivan

1. Confirm that §4 is binding, including the options gate and the deferral of the Z-dynamics exit (§5).
2. Run §7.
