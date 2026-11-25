# Dealer Gamma 002 — Readout of DG 01–05 (discovery): no hedging footprint; branch closed

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-25 |
| **Responds to** | Ivan (2026-11-25): ran DG 01–05, outputs in `75c7105` |
| **Status** | Readout. **All three primary tests FAIL in discovery.** Holdout not run (nothing to confirm). Recommendation: close the branch |

## 1. Gate (DG 01) — data PASS, positioning model FAIL

| | BTC | ETH |
|---|---|---|
| reconstructed / collector 24h volume (pooled) | 0.998 | 0.999 |
| rows with volume within 5% | 99.5% | 99.6% |
| rows with \|net\| > OI·1.01 + 0.1 (limit 5%) | **5.6%** | **7.4%** |
| excess over OI, share of total OI | 0.4–3.7% by expiry type | 0.3–3.0% |
| Spearman(\|net\|, OI) | 0.92 | 0.91 |

The archive is complete over the collector overlap, 2026-09-19 → 11-24 (missing prints would make \|net\| smaller, not larger). **The archive validator was not run** (`01` found no `deribit_daily_qc.csv`), so 2024-01 → 2026-09 is checked only by file presence (no missing day). The violations come from the positioning model: "customer = taker" double-counts positions that change hands between passive customers. They are largest on big weekly/monthly expiries (e.g. ETH-23OCT26-2300-P: OI 12.6k, reconstructed 14.1k). Under 001 §5 this is a FAIL; Ivan ran 02–05 anyway, so everything below is conditional on a positioning proxy that is noisy (biases effects towards zero) but strongly correlated with OI.

## 2. What the reconstruction says (DG 02)

1,058 days per currency, no missing option or perpetual file (2024-01-02 → 2026-11-24).

| | BTC | ETH |
|---|---|---|
| sessions with dealers long gamma (`recon`) | 23% (2024: 1%, 2025: 37%, 2026: 32%) | 8% |
| median $G$ at 08:05, USD per 1% move | −15.6 m | −5.3 m |
| median $G$/ADV₃₀ (discovery) | −3.7% | −2.7% |
| corr(`recon`, `recon_noblock`) | 0.60 | 0.71 |
| corr($x$, log DVOL) | −0.48 | −0.14 |

The hedge implied for a 0.1% move is about 10% of an average 5-minute bar's perpetual volume, so the effect was not too small to see in principle. Block trades change the picture a lot (correlation only 0.6–0.7): the taker side of a block is a poor customer label, and blocks are where institutional positioning sits.

## 3. Regime (DG 03) — FAIL

Primary `recon`, discovery (365 sessions), coefficient per 1 SD of $x$, HAC:

| | log RV β (t) | AC1 β (t) |
|---|---|---|
| BTC | −0.003 (−0.20) | −0.005 (−1.39) |
| ETH | −0.022 (−1.49) | +0.003 (+0.67) |

Raw terciles do show lower realized vol in the long-gamma tercile (BTC 2.31% vs 2.57%, ETH 3.25% vs 3.62%), but it disappears once DVOL and lagged RV are controlled: dealer gamma is mechanically larger when implied vol is low (corr −0.48 in BTC). That is the confound 001 warned about, not hedging. Variants: 4 of 24 fits reach p < 0.05 in the hypothesised direction, scattered across variants and currencies (`call_long_put_short` BTC, `recon_clean` BTC AC1 and ETH RV) — consistent with multiple-testing noise; none holds in both currencies.

## 4. Hedge flow (DG 04) — FAIL, wrong sign in BTC

Primary: next-15-minute signed perpetual taker flow on $H_t=-100\,G_{t-1}r_t/\text{ADV5}$, controls $r_t$ and $G_{t-1}$, clustered by session (≈103k bars, 365 sessions):

| | b (t) | b, block/combo excluded (t) | b, `short_all` (t) |
|---|---|---|---|
| BTC | −0.019 (−1.98) | −0.054 (**−3.64**) | +0.002 (0.73) |
| ETH | −0.026 (−0.92) | +0.054 (1.71) | +0.034 (2.46) |

- The binned plot `04_binned.png` looks strongly positive, but it is **not evidence**: with dealers mostly short gamma, $H$ has the sign of the last return, and taker flow simply continues the last move (momentum in flow, `b_r_bp` > 0). Once $r_t$ is controlled, the gamma-dependent part is zero or negative.
- BTC goes the **wrong way** with blocks excluded (t = −3.6): after a move, flow continues *less* when the reconstruction says dealers are more short gamma. Read together with §1–2, the likeliest reading is that the reconstructed sign is often wrong, not that dealers trade against their hedge.
- The ETH `short_all` fit (p = 0.007) is one variant in one currency; under 001 §7 it cannot rescue the primary.

## 5. Pinning (DG 05) — FAIL

Primary `long_gamma`, all expiries: mean Δ = −3.7 bp (BTC, p 0.22), −10.3 bp (ETH, p 0.07); 52–54% of expiries settle closer to the pin strike than to the placebo. The pin strike sits a median 73–80 bp from spot at 05:00, and three hours of movement (≈ 80–100 bp) swamp any pull of a few bp. The `short_gamma` control behaves like the pin rules except in 8 BTC monthly expiries (+89 bp, p 0.008), which is too few to mean anything. Weekly/monthly/quarterly groups have 5–52 expiries and are underpowered as expected.

## 6. Verdict and recommendation

Per 001 §7: DG 03, 04 and 05 all FAIL in discovery for both currencies; the mechanism is **not established**. The holdout is not run — there is nothing to confirm, and running it now would only spend it.

Why it failed, in order of likelihood:
1. **Dealer inventory cannot be recovered from public prints.** The taker is not the customer often enough (gate violations, block sensitivity, wrong-sign BTC flow). Deribit does not publish who the market makers are, and positions move between passive participants.
2. **Hedging is passive, cross-venue or in dated futures/spot.** Taker flow on the Deribit perpetual cannot see that.
3. **The effect is real but too small** relative to BTC/ETH perpetual and spot volume (the hedge is ~3% of daily perp volume per 1% move, and Binance/CME volume is several times Deribit's).

Recommendation: **close Dealer Gamma.** A rescue would need a better inventory signal (e.g. a published dealer-positioning feed, or Deribit's `trades` with maker/taker account flags, which are not public), not more variants of this one. The module, the cache and the gate's completeness check stay useful for other branches (with Ivan's approval, per the branch-independence rule); the archive validator should still be run before any other study relies on the 2024–2026 option prints.

## 7. Decision for Ivan

1. Close the branch as recommended, or
2. Keep it open for one pre-registered follow-up only if a new positioning source becomes available.
