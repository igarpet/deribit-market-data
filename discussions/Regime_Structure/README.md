# Regime Structure discussions

Do Ivan's discretionary regimes (range, displacement, noise, shock) exist on crypto as persistent states beyond volatility clustering, and do the transitions he describes (range → displacement → noise; shock after up-displacement) appear? No VWAP (VWAP 002). Notebooks: `../../notebooks/Regime_Structure/`.

Independent branch: Binance USD-M perpetual 1-minute klines for BTC, ETH, SOL, XRP; no other branch's outputs.

## History

| Turn | Content |
|---|---|
| `001_claude_branch_preregistration.md` | Hourly labels from Ivan's definitions (shock 3σ), sign-randomised 1m null, E1/E2 persistence and C1–C3 cycle tests, geometric-mean pooling, Bonferroni gates on the test window, interpretation table |
| `002_claude_rs01_readout.md` | RS 01 (v1.0): E1/E2 fail (range/displacement persistence = volatility), C1 fails, C2 opposite (after DISP, noise 30% less likely), C3 up-DISP→SHOCK passes but fragile; v1.1 shares null signs across assets (v1.0 null anti-conservative), v1.1 run binding |
| `003_claude_scale_sweep_preregistration.md` | v1.2: same tests on 15m/30m/1h/2h/4h windows in one run; primary 1h α=0.01, sweep α=0.0025; 399 null draws; one binding run |

Next turn in this branch: `004`.

Procedure and numbering rules: `../README.md`.
