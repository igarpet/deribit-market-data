# Signed Jumps

**Question.** Do 1-minute jumps self-excite, does the sign of a past jump predict the sign of the next one, and does pre-jump asymmetry in the continuous flow (jump-truncated skewness) predict the jump's sign? The volatility engine (AM-T-HAR-Blend structure, refitted on bipower variation) is the continuous component; a signed Hawkes process is the jump layer.

**Data.** Binance spot 1m, BTC/ETH/SOL/XRP, about 26 months; DEV 70% / TEST 30% chronological.

| Turn | Content | Status |
|---|---|---|
| `001_claude_sj01_preregistration.md` | SJ 01: model ladder (a)–(e), gates G1–G3, synthetic validation, limitations | Preregistered |
| `002_claude_sj01_readout.md` | SJ 01 results: **all gates PASS**. ρ = 0.155, half-life 1 min; continuation 71% after up jumps, 60% after down; TSK predicts sign (t ≈ ±8); jump sizes symmetric. Economics ≈ 1–3 bp gross per jump vs 12 bp costs; proposes SJ 02a event study before options | Result |
| `003_claude_sj02a_preregistration.md` | SJ 02a: perp path after cluster-starting jumps (1–60 min), continuation vs reversal × π_c quintile, rule chosen on DEV by net lower bound, TEST verdict at 12 bp | Preregistered |
| `004_claude_sj02a_readout.md` | SJ 02a: **NO_EDGE**. Post-jump perp path flat at 1–60 min (|mean| < 1 bp, typical |move| 11–47 bp); median event reverses slightly; π_c does not separate paths. Jumps carry volatility, not direction | Result; SJ 02b not built |
| `005_claude_sj03_preregistration.md` | SJ 03 (boiling analogy): do trade-count / volume acceleration and \|taker imbalance\| predict jump timing beyond SJ 01? Gates F1 (f vs e), F2 (15-min forecast), F3 (beyond flow-implied volatility: boil-specific heat) | Preregistered |
| `006_claude_sj03_readout.md` | SJ 03: **F1 PASS, F2 FAIL, F3 FAIL**. Activity builds in the last ~5 min before jumps (+75% volume at −1 min) but carries only volatility information; top-1% hazard minutes: 20% jump rate in 15 min vs 6.1% base | Result |
| `007_claude_sj04_preregistration.md` | SJ 04: jumps at 1-second resolution (spot 1s, Jul–Sep 2026): P1 early activity build-up (>30 s before onset), P2 split events (bar-boundary artefact), P3 continuation onset+5 s → +65 s | Preregistered |
| `008_claude_sj04_readout.md` | SJ 04: **P1 PASS** (activity +1.4× ~2 min before onset, ~11× at onset); **P2 mostly distinct events** (15% split) → SJ 01 excitation real; **P3 withdrawn** (look-ahead: onset uses the full jump minute); post-close returns ≈ flat. Proposes SJ 04b real-time seconds trigger | Result |
| `009_claude_sj04b_preregistration.md` | SJ 04b: real-time trigger (trailing-60 s return ≥ 4 × engine 1-min σ, past data only); gate δ = 2 s latency, H = 60 s, gross q2.5 > 0; latency × hold grid; economic flags 7 / 12 bp | Preregistered |
| `010_claude_sj04b_readout_and_branch_summary.md` | SJ 04b: **NO_CONTINUATION** — +0.25 bp [−0.46, +1.03] at 2 s / 60 s; whole grid (c 3–5, latency 1–30 s, hold 10 s–5 min) within ±0.7 bp; confirms SJ 04 P3 was look-ahead. Branch summary: jumps carry volatility/hazard information, no direction; proposes closure | Result |



Sequence: SJ 01 (statistics) → SJ 02a (post-jump perp path, event study) → SJ 02b (Deribit single wing vs straddle) only on SJ 02a `NET_EDGE` (not built). SJ 03: flow precursors of jump timing.
