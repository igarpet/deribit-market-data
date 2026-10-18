# Multi-Scale HMM discussions

Do latent states discovered by a Student-t HMM on 1h, 4h and 1D bars (built from 1-minute data with intrabar emissions) identify momentum or reversal regimes that predict the next bar's return on BTC/ETH/SOL/XRP perpetuals, net of costs? Notebooks: `../../notebooks/Multi_Scale_HMM/`.

Independent branch: Binance USD-M perpetual 1m klines and funding only; no other branch's outputs.

## History

| Turn | Content |
|---|---|
| `001_claude_branch_preregistration.md` | Bars and 10 intrabar emissions, pooled Student-t HMM (K, ν on VALID), per-state momentum/reversal regression, always-in and net rules, shared-sign minute null with full refit, verdicts S / E / states-add-value / P, drift-only benchmark |

| `002_claude_msh01_readout.md` | MSH 01 (60ff884): NO_TRADEABLE_MOMENTUM_STATES at 1h/4h/1D (S p 0.57 / 0.46 / 0.93; no net edge; states add nothing vs K=1); 16 states are volatility/bar-shape states, none with a significant momentum/reversal slope; 1D overfits; weak post-hoc K=1 daily slope (t 1.9) noted, not a finding |

Next turn in this branch: `003`.

Procedure and numbering rules: `../README.md`.
