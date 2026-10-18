# Funding Carry discussions

Does a delta-neutral long spot / short USD-M perpetual position on Binance earn more than cash after costs by collecting funding, on the PF02 universe (30 assets)? Notebooks: `../../notebooks/Funding_Carry/`.

Independent branch: Binance spot 1m, USD-M perpetual 1m and funding files (downloaded for PF02); no other branch's outputs.

## History

| Turn | Content |
|---|---|
| `001_claude_branch_preregistration.md` | Mechanics (constant units, daily 00:05 UTC, funding at print times, 17 bp/side, +40% reset), strategies C (f7 ≥ 10% in / < 3% out, primary) and U (always in), capital 1.5/notional, gates G1–G4 (CI, DEV and TEST, breadth 60%, ≥ 4% on deployed capital) |

Next turn in this branch: `002`.

Procedure and numbering rules: `../README.md`.
