# Volatility Regression — Claude / ChatGPT discussions

Independent branch for five-minute volatility OLS research (not the HSMM duration branch and not VRP option P&L). Notebooks: `../../notebooks/Volatility_Regression/`.

- **001**: `001_chatgpt_first_5m_ols_design.md` — initial implementation and request for Claude's methodological/code review.
- **002**: `002_claude_review_01_design_and_code.md` — Claude's review of 01 v0.1.0: baseline, OOS R², target quality, coverage, selection protocol.
- **003**: `003_claude_01_v011_implemented.md` — v0.1.1 implemented (pure OLS, no HAR, per Ivan).
- **004**: `004_chatgpt_paired_samples_v012.md` — fix current-bar volatility eligibility, assert identical scored populations, add regression test; future target and selection ideas remain separate.
- **005**: `005_chatgpt_feature_provenance_audit.md` — read-only raw-to-model provenance audit, collector vintage, option/L2 masking, and complete-case coverage.
- **Next turn:** `006`, unless a newer numbered file is committed first.

Follow `../README.md`: new file per discussion turn, no edits to past turns, evidence before opinions, Claude and ChatGPT may propose, Ivan decides.
