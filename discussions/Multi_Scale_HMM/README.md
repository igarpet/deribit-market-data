# Multi-Scale HMM discussions

Do latent states discovered by a Student-t HMM on 1h, 4h and 1D bars (built from 1-minute data with intrabar emissions) identify momentum or reversal regimes that predict the next bar's return on BTC/ETH/SOL/XRP perpetuals, net of costs? Notebooks: `../../notebooks/Multi_Scale_HMM/`.

Independent branch: Binance USD-M perpetual 1m klines and funding only; no other branch's outputs.

## History

| Turn | Content |
|---|---|
| `001_claude_branch_preregistration.md` | Bars and 10 intrabar emissions, pooled Student-t HMM (K, ν on VALID), per-state momentum/reversal regression, always-in and net rules, shared-sign minute null with full refit, verdicts S / E / states-add-value / P, drift-only benchmark |

Next turn in this branch: `002`.

Procedure and numbering rules: `../README.md`.
