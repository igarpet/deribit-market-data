# Signed Jumps

**Question.** Do 1-minute jumps self-excite, does the sign of a past jump predict the sign of the next one, and does pre-jump asymmetry in the continuous flow (jump-truncated skewness) predict the jump's sign? The volatility engine (AM-T-HAR-Blend structure, refitted on bipower variation) is the continuous component; a signed Hawkes process is the jump layer.

**Data.** Binance spot 1m, BTC/ETH/SOL/XRP, about 26 months; DEV 70% / TEST 30% chronological.

| Turn | Content | Status |
|---|---|---|
| `001_claude_sj01_preregistration.md` | SJ 01: model ladder (a)–(e), gates G1–G3, synthetic validation, limitations | Preregistered |
| `002_claude_sj01_readout.md` | SJ 01 results: **all gates PASS**. ρ = 0.155, half-life 1 min; continuation 71% after up jumps, 60% after down; TSK predicts sign (t ≈ ±8); jump sizes symmetric. Economics ≈ 1–3 bp gross per jump vs 12 bp costs; proposes SJ 02a event study before options | Result |

Sequence: SJ 01 (statistics) → SJ 02 (single-wing options vs straddle) only if G2 and G3 pass.
