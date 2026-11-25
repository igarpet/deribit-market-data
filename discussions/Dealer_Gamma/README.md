# Dealer Gamma discussions

Does option-dealer hedging leave a footprint in BTC and ETH? Dealer gamma is rebuilt from the taker side of every Deribit option print since 2024-01, then tested three ways: next-session volatility and autocorrelation (regime), signed perpetual taker flow after a move (hedge flow), and settlement relative to the dealer-long-gamma strike (pinning). Notebooks: `../../notebooks/Dealer_Gamma/`.

## History

| Turn | Content |
|---|---|
| `001_claude_branch_preregistration.md` | Positioning model (customer = taker), 5 sign variants, gate vs collector OI/volume, tests DG 03–05, periods (burn-in to 2024-06, discovery 2024-07 → 2025-06, holdout 2025-07 →), decision rule, run order |

| `002_claude_dg01_05_discovery_readout.md` | DG 01–05 (75c7105): gate data PASS (volume ratio 0.998/0.999) but positioning FAIL (5.6%/7.4% of rows \|net\| > OI); regime, hedge flow, pinning all **FAIL** in discovery; BTC hedge flow wrong sign without blocks (t −3.6); recommend **close** |
| `003_claude_branch_closed.md` | **Branch closed** by Ivan on DG 002; holdout untouched; module, cache and completeness check kept |

Branch closed (DG 003).

Procedure and numbering rules: `../README.md`.
