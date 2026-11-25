# Dealer Gamma discussions

Does option-dealer hedging leave a footprint in BTC and ETH? Dealer gamma is rebuilt from the taker side of every Deribit option print since 2024-01, then tested three ways: next-session volatility and autocorrelation (regime), signed perpetual taker flow after a move (hedge flow), and settlement relative to the dealer-long-gamma strike (pinning). Notebooks: `../../notebooks/Dealer_Gamma/`.

## History

| Turn | Content |
|---|---|
| `001_claude_branch_preregistration.md` | Positioning model (customer = taker), 5 sign variants, gate vs collector OI/volume, tests DG 03–05, periods (burn-in to 2024-06, discovery 2024-07 → 2025-06, holdout 2025-07 →), decision rule, run order |

Next turn in this branch: `002`.

Procedure and numbering rules: `../README.md`.
