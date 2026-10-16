# Signed Jumps

**Question.** Do 1-minute jumps self-excite, does the sign of a past jump predict the sign of the next one, and does pre-jump asymmetry in the continuous flow (jump-truncated skewness) predict the jump's sign? The volatility engine (AM-T-HAR-Blend structure, refitted on bipower variation) is the continuous component; a signed Hawkes process is the jump layer.

**Data.** Binance spot 1m, BTC/ETH/SOL/XRP, about 26 months; DEV 70% / TEST 30% chronological.

| Turn | Content | Status |
|---|---|---|
| `001_claude_sj01_preregistration.md` | SJ 01: model ladder (a)–(e), gates G1–G3, synthetic validation, limitations | Preregistered; awaiting run |

Sequence: SJ 01 (statistics) → SJ 02 (single-wing options vs straddle) only if G2 and G3 pass.
