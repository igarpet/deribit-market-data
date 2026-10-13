# Regime Structure discussions

Do Ivan's discretionary regimes (range, displacement, noise, shock) exist on crypto as persistent states beyond volatility clustering, and do the transitions he describes (range → displacement → noise; shock after up-displacement) appear? No VWAP (VWAP 002). Notebooks: `../../notebooks/Regime_Structure/`.

Independent branch: Binance USD-M perpetual 1-minute klines for BTC, ETH, SOL, XRP; no other branch's outputs.

## History

| Turn | Content |
|---|---|
| `001_claude_branch_preregistration.md` | Hourly labels from Ivan's definitions (shock 3σ), sign-randomised 1m null, E1/E2 persistence and C1–C3 cycle tests, geometric-mean pooling, Bonferroni gates on the test window, interpretation table |
| `002_claude_rs01_readout.md` | RS 01 (v1.0): E1/E2 fail (range/displacement persistence = volatility), C1 fails, C2 opposite (after DISP, noise 30% less likely), C3 up-DISP→SHOCK passes but fragile; v1.1 shares null signs across assets (v1.0 null anti-conservative), v1.1 run binding |
| `003_claude_scale_sweep_preregistration.md` | v1.2: same tests on 15m/30m/1h/2h/4h windows in one run; primary 1h α=0.01, sweep α=0.0025; 399 null draws; one binding run |
| `004_claude_rs01_v12_readout.md` | Binding v1.2 result: no regime states at 15m–4h (E1, E2, C1 fail; C2 opposite everywhere); C3 up-DISP→SHOCK passes at 1h and 15m, robust across scales/assets and in dev at 15–30m; 4h E2 passes but not in dev |
| `005_claude_rs02_preregistration.md` | RS 02: C3 signal → Part A move ratios (Binance 2y), Part B long ATM 0DTE straddle at the ask held to settlement on collector quotes (VRP 02 code, approved); primary M_signal with VRP verdict labels |
| `006_claude_rs02_readout.md` | RS 02: signal raises the move to settlement 5–11% (Binance 2y, all CIs >1); signal straddles lose −21% vs −29% for all entries (INCONCLUSIVE, CI [−0.41, +0.04]); call-only worse; 60-min exit −32%; not enough to pay the spread |
| `007_claude_rs03_preregistration.md` | RS 03: big-move days (session move ≥ dev q85 of move/7-session forecast) vs normal on 2y Binance; DURING descriptive; PRE/EARLY screened on dev, confirmed on test (Bonferroni over candidates); clustering; mechanical findings flagged in advance |
| `008_claude_rs03_readout.md` | RS 03: only yesterday's RV vs forecast confirmed (AUC 0.56 dev / 0.65 test); big days are market-wide (54–64% hit ≥3 assets), US-session, trend-like, mildly clustered (next-day lift 1.37); no directional or early-warning regime |
| `009_claude_rs04_rs05_preregistration.md` | Correction (execution near mid already tested in VRP 03: no help); RS 04 implied underreaction to yesterday's RV (slope, session bootstrap); RS 05 shared-sign 5m null for variance ratio, big-day ER/giveback/top-hour, clustering |

Next turn in this branch: `010`.

Procedure and numbering rules: `../README.md`.
