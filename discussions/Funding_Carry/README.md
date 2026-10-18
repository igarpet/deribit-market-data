# Funding Carry discussions

Does a delta-neutral long spot / short USD-M perpetual position on Binance earn more than cash after costs by collecting funding, on the PF02 universe (30 assets)? Notebooks: `../../notebooks/Funding_Carry/`.

Independent branch: Binance spot 1m, USD-M perpetual 1m and funding files (downloaded for PF02); no other branch's outputs.

## History

| Turn | Content |
|---|---|
| `001_claude_branch_preregistration.md` | Mechanics (constant units, daily 00:05 UTC, funding at print times, 17 bp/side, +40% reset), strategies C (f7 ≥ 10% in / < 3% out, primary) and U (always in), capital 1.5/notional, gates G1–G4 (CI, DEV and TEST, breadth 60%, ≥ 4% on deployed capital) |

| `002_claude_fc01_readout.md` | FC 01 (cc17447): C NO_CARRY_EDGE (TEST −0.08%/yr; fails G2), U CARRY_POSITIVE_BELOW_HURDLE (+2.0%/yr on capital; gross funding 4.1%/yr per notional, concentrated in Q4 2024 and Q3 2025); not better than cash after costs and capital |

Next turn in this branch: `003`.

Procedure and numbering rules: `../README.md`.
