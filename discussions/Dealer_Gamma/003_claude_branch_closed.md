# Dealer Gamma 003 — Branch closed

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-25 |
| **Responds to** | Ivan (2026-11-25): "we'll close the branch" |
| **Status** | **Closed** on DG 002. Holdout (2025-07-01 →) never run and remains untouched |

Decision by Ivan on the DG 002 readout: regime, hedge-flow and pinning tests all failed in discovery for BTC and ETH, and dealer inventory cannot be recovered reliably from taker-signed public prints (gate violations, block sensitivity, wrong-sign BTC flow).

Kept for reuse (cross-branch use needs Ivan's approval, `../README.md` §2): the positioning/GEX module `dealer_gamma.py`, the cache under `DERIBIT_DATA_ROOT\derived\dealer_gamma`, and the trade-archive vs collector completeness check (volume ratio 0.998/0.999 over 2026-09-19 → 11-24). Still to do before another study relies on the 2024–2026 option prints: run `tools/validate_deribit_history_trades.py --kind option`.

Reopening requires a new pre-registration built on a better dealer-positioning source, not on further variants of this reconstruction.
