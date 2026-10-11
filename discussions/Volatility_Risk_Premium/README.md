# Volatility Risk Premium discussions

How the project's physical-volatility forecasts relate to option prices, implied variance and, ultimately, executable delta-hedged option P&L. Notebooks: `../../notebooks/Volatility_Risk_Premium/`.

The branch is separate from the HSMM line and can progress while the prospective HSMM/volatility-state sample accumulates.

## History

| Turn | Former number | Content |
|---|---|---|
| `001_chatgpt_why_volatility_risk_premium.md` | 017 | Branch proposal |
| `002_claude_reply_vrp_branch.md` | 018 | Reply: pricing audit first, flat breakeven, hold-to-settlement, holdout |
| `003_claude_vrp_step1_pricing_audit.md` | 019 | VRP 01 built (pricing audit) |
| `004_claude_pricing_audit_results.md` | 020 | First results, B/C fixes |
| `005_claude_audit_rerun.md` | 021 | Rerun: B ill-conditioning, C fails |
| `006_claude_audit_part_b_final.md` | 022 | Audit complete, proposal to start step 2 |
| `007_claude_long_straddle_preregistration.md` | — | Step 2 as a long-only test: VRP 02 pre-registration (unhedged ATM straddle at the ask, held to settlement) |

Their text uses the former numbers (e.g. "018 §2.4" = VRP 002 §2.4; "Module 12" = HSMM 06).

Next turn in this branch: `008`.

Procedure and numbering rules: `../README.md`.
