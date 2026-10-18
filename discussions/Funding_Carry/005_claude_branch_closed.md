# Funding Carry 005 — Branch closed

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-18 |
| **Branch** | Funding Carry |
| **Responds to** | FC 002, FC 004; Ivan's decision (2026-10-18) |
| **Status** | Closed by owner decision. |

Ivan ruled out carry trades (long and reverse).

**Evidence:**
- **FC 01:** long carry is positive but below cash: +2.0%/yr on capital always-in; C `NO_CARRY_EDGE`.
- **FC 02:** funding is predictable by persistence alone ($R^2_{OOS}$ 0.51 at 7d), yet the top forecast quintile realized only 4.1%/yr gross.

**Practical reason:** venue access. Binance has no MiCA licence since 2026-07-01, its Spanish registration has lapsed, and EU service continues only under a contested reverse-solicitation exemption.

**Reverse carry:** not tested (no borrow-rate data). It is closed with the branch.

**Kept from this branch:**
- `funding_predict.py`, including the VD18 HAR_4 refit on Binance spot.
- The finding that the volatility engine ties a calibrated daily HAR at 24h and loses at 7d. This is relevant to VD20 / VD 041.
