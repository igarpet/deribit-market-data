# Dealer Gamma 001 — Pre-registration: does option-dealer hedging leave a footprint in BTC/ETH?

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-11-25 |
| **Responds to** | Ivan (2026-11-25): test institutional/dealer hedging on the BTC and ETH data already on disk; run the hedge-flow, regime and pinning tests |
| **Status** | Pre-registration. Written and pushed before any real-data run. Code `src/data_engineering/modeling/dealer_gamma.py` (DG-0.1.0); tests `tests/test_dealer_gamma.py`; notebooks `notebooks/Dealer_Gamma/01`–`05` |

## 1. Question and mechanism

Option market makers on Deribit hedge to stay delta-neutral. A dealer **long gamma** re-hedges against the move (sells rallies, buys dips) and should dampen volatility and push returns towards reversion; a dealer **short gamma** re-hedges with the move and should amplify it. The direction of the hedge comes from the position, not from a view. If this is real and large enough relative to the perpetual market, three things follow, each tested separately:

1. **Regime:** dealer gamma at the start of a session predicts that session's realized volatility (lower when long) and 5-minute return autocorrelation (more negative when long).
2. **Hedge flow:** the hedge implied by gamma and the last move shows up in signed perpetual taker flow over the next 15 minutes.
3. **Pinning:** at expiry, the settlement price ends closer to the strike where dealers are most long gamma than to a mirrored placebo strike.

This branch is independent: it uses no frozen spec, threshold or result from another branch. It reuses the downloaded data and the trade-file layout only.

## 2. Data

| Source | Use | Coverage |
|---|---|---|
| `history/trades/{BTC,ETH}/option/` (history.deribit.com) | every option print, taker side, iv, index | 2024-01-02 → last downloaded day |
| `history/trades/{BTC,ETH}/future/`, `*-PERPETUAL` rows | 5-minute perpetual price, taker buy/sell USD, index (settlement proxy) | same |
| `daily/option_grid{,_1dte}/` (collector) | open interest and 24h volume per instrument, gate only | 0DTE from 2026-09-19, 1DTE from 2026-10-06 |
| `history/dvol/{BTC,ETH}_dvol_1h.csv` | implied-vol control | since 2021 |

Only inverse BTC and ETH options (the `BTC`/`ETH` currency files). USDC linear options are out of scope.

## 3. Positioning model (fixed)

- Customer = **taker** of each print. Net customer position per instrument = cumulative signed `amount` (coin) since the first print in the archive. Dealer position = −net customer.
- Gamma: Black-Scholes with r = q = 0, the Deribit index as S, the instrument's **last traded iv**, T to the 08:00 UTC expiry. Dealer dollar gamma per 1% move: $G = \sum_i \text{dealer}_i\,\Gamma_i\,S^2\,0.01$ (USD). $G>0$ = dealers long gamma.
- Grid: every 5 minutes (bar end), state after all prints before that time.
- **Variants** (all computed, one primary):

| Variant | Dealer position | Role |
|---|---|---|
| `recon` | −net customer, all prints | **primary** |
| `recon_noblock` | −net customer, block and combo prints excluded | robustness (block direction is less clearly the customer side) |
| `recon_clean` | `recon`, only instruments first seen on/after 2024-01-16 | robustness (unseen opening positions) |
| `short_all` | −\|net\| | "dealers short everything" comparison |
| `call_long_put_short` | +\|net\| calls, −\|net\| puts | equity-style convention comparison |

- Normalisation: $x = G_{08:05} / \text{ADV}_{30}$, where ADV₃₀ = mean perpetual USD volume of the previous 30 sessions. It reads as "hedge per 1% move as a share of a normal day's perpetual volume".

Known limits, stated before the run: (i) the taker is not always the customer; (ii) dealers may hedge passively, with dated futures, or on other venues, which the taker-flow test cannot see; (iii) positions opened before 2024-01-02 are unknown, hence a burn-in and the `recon_clean` variant; (iv) a missing trade day corrupts positions of instruments alive across it — days without an option file are excluded from every test, and the count is reported.

## 4. Periods

| Period | Sessions | Use |
|---|---|---|
| burn-in | 2024-01-02 → 2024-06-30 | positions accumulate; never tested |
| **discovery** | 2024-07-01 → 2025-06-30 | first verdict |
| **holdout** | 2025-07-01 → last day | confirmation; runs only with `DG_HOLDOUT=1`, after the discovery readout is posted |

Any change to this spec after seeing discovery results must be posted as a numbered amendment **before** the holdout run.

## 5. Stage gate (DG 01) — must pass before 02–05 are read

1. `tools/validate_deribit_history_trades.py --kind option` on BTC and ETH, 2024-01-01 → last day. Days `FAIL`, `UNREADABLE`, `MISSING_FILE`, or with `millisecond_page_limit_buckets > 0`, are listed; the latter must be re-downloaded with the patched downloader.
2. Over the collector overlap, for every 0DTE/1DTE instrument and snapshot: reconstructed trailing-24h traded amount vs the collector's 24h `volume`, and \|net customer\| vs `open_interest`.
3. **PASS** (each currency): pooled volume ratio in [0.95, 1.05] **and** share of rows with \|net\| > OI·1.01 + 0.1 at most 5%. Spearman(\|net\|, OI) and the median \|net\|/OI are reported, not gated.

FAIL stops the branch: the reconstructed positions are not trustworthy and 03–05 would measure noise.

## 6. Tests (fixed before the run)

**DG 03 — Regime.** One row per session d, (08:00 d, 08:00 d+1]. Predictor $x$ at 08:05 (after expiry). Targets: log realized vol $\tfrac12\log\sum r_5^2$ and $\text{AC}_1=\sum r_t r_{t-1}/\sum r_t^2$ from 5-minute perpetual log returns. OLS with controls log DVOL (07:00 bar), lagged log RV, weekday dummies; HAC (5 lags). Coefficient per 1 SD of $x$. Sessions need ≥ 250 valid bars. **H:** β < 0 for both targets.

**DG 04 — Hedge flow.** 5-minute bars. Predicted dealer hedge $H_t = -100\,G_{t-1}\,r_t/\text{ADV5}$, ADV5 = ADV₃₀/288. Target $f = \sum_{j=1}^{3}(\text{buy}-\text{sell})_{t+j}/\text{ADV5}$ (next 15 minutes). OLS $f = a + b\,H_t + c\,r_t + d\,G_{t-1}/\text{ADV5}$, SE clustered by session. Bars 07:50–08:15 UTC excluded. **H:** b > 0. Reported also: h = 1 bar, contemporaneous flow (descriptive only — mechanically entangled), and a 0.1%/99.9% winsorized fit.

**DG 05 — Pinning.** For each daily 08:00 expiry, strike table at 05:00 for that expiry, strikes within ±5% of S. Rules: `long_gamma` = strike with the largest positive dealer gamma (primary); `long_gamma_noblock`; `max_position` = largest \|net\| summed over call+put; `short_gamma` = most negative (control). Placebo $K' = 2S_{05}-K$. Settlement proxy X = mean of minute-mean index, 07:30–08:00. Statistic $\Delta = (|X-K|-|X-K'|)/S_{05}$ in bp. Groups: all, daily (Mon–Thu, Sat–Sun), weekly, monthly, quarterly, all Fridays. Bootstrap over expiries (4,000). **H:** mean Δ < 0 for `long_gamma` and `max_position`; `short_gamma` is the control (no pin or repulsion, Δ ≥ 0).

## 7. Decision rule

Primary variant `recon`, primary horizon 15 minutes, primary pinning group `all` with rule `long_gamma`.

- A test **PASSES in a period** only if the hypothesised sign has one-sided p < 0.05 (discovery) / p < 0.10 (holdout) in **both** BTC and ETH.
- **Mechanism established** only if the hedge-flow test (DG 04) passes discovery **and** holdout. Regime alone (DG 03) passing is read as a gamma–implied-vol effect, not hedging, unless DG 04 agrees.
- Variants and groups are reported; they cannot rescue a failed primary. A variant that disagrees in sign with the primary is flagged.
- No trading rule is built in this branch. If DG 04 passes both periods, a separate pre-registration would propose one.

## 8. Run order

1. Update the trade archive to the latest day (resumes):
   `python tools\download_deribit_history_trades.py --currencies BTC ETH --kinds option future --start 2024-01-01`
2. Validator (options and futures):
   `python tools\validate_deribit_history_trades.py --kind option --start 2024-01-01 --out notebooks\Dealer_Gamma\01\validator_option`
   `python tools\validate_deribit_history_trades.py --kind future --start 2024-01-01 --out notebooks\Dealer_Gamma\01\validator_future`
3. Copy the latest collector `daily/option_grid*` folders from the VM to the PC data folder (`tools\sync_deribit_data.py`).
4. Notebooks in order: `01_data_gate` → `02_build_gex` (slow, cached) → `03_regime` → `04_hedge_flow` → `05_pinning`. Commit the numbered output folders.
5. Holdout only after the discovery readout: `set DG_HOLDOUT=1`, rerun 03–05.

## 9. Questions for Ivan

1. Is 2025-07-01 an acceptable holdout start? It gives roughly 16 months of holdout against 12 months of discovery.
2. Should USDC linear options (SOL, XRP, …) be added later as a third/fourth asset if BTC/ETH pass?
