# Signed Jumps

**Question.** Do 1-minute jumps self-excite, does the sign of a past jump predict the sign of the next one, and does pre-jump asymmetry in the continuous flow (jump-truncated skewness) predict the jump's sign? The volatility engine (AM-T-HAR-Blend structure, refitted on bipower variation) is the continuous component; a signed Hawkes process is the jump layer.

**Data.** Binance spot 1m, BTC/ETH/SOL/XRP, about 26 months; DEV 70% / TEST 30% chronological.

| Turn | Content | Status |
|---|---|---|
| `001_claude_sj01_preregistration.md` | SJ 01: model ladder (a)–(e), gates G1–G3, synthetic validation, limitations | Preregistered |
| `002_claude_sj01_readout.md` | SJ 01 results: **all gates PASS**. ρ = 0.155, half-life 1 min; continuation 71% after up jumps, 60% after down; TSK predicts sign (t ≈ ±8); jump sizes symmetric. Economics ≈ 1–3 bp gross per jump vs 12 bp costs; proposes SJ 02a event study before options | Result |
| `003_claude_sj02a_preregistration.md` | SJ 02a: perp path after cluster-starting jumps (1–60 min), continuation vs reversal × π_c quintile, rule chosen on DEV by net lower bound, TEST verdict at 12 bp | Preregistered; awaiting run |



Sequence: SJ 01 (statistics) → SJ 02a (post-jump perp path, event study) → SJ 02b (Deribit single wing vs straddle) only on SJ 02a `NET_EDGE`.
