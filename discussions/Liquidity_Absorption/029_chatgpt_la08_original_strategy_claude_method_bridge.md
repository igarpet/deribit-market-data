# Liquidity Absorption 029 — LA-08 original A03b 1:1 with Claude methodology, controlled retrospective bridge

**Author:** ChatGPT  
**Date:** 2026-10-20  
**Version:** `LA-BRIDGE-08-0.1.0`  
**Status:** implemented and synthetic-tested; user's local historical EXT/V2 files required to execute; **no new out-of-sample performance claim**.  
**Changes to existing research:** none. A03b, A03d, V2, EXT, collection and raw archive inputs are read-only.

## Motivation and which strategy is being tested

The owner requested a *new notebook testing the strategy we originally specified, using Claude's implemented methodology*. Existing EXT already evaluates the frozen original A03b price-only 1:1 setup on **Deribit-native** BTC/ETH perpetuals over 2024–2026. Existing V2 already modifies CVD, target selection, stop and trailing simultaneously. Running either again adds no controlled insight.

LA-08 instead tests whether Claude's **signal-time** absorption confirmation and causal **18:00 America/New_York** traded-volume-profile availability have incremental value when applied to **our original A03b 1:1 payoff**, without silently replacing its original target/stop or resimulating different entries.

This is a **controlled historical filter replay**, not a new engine-level order-execution backtest. Static eligibility known at the second-touch signal close may be applied to the already replayed original fixed-strategy outcomes, because it cannot change how an otherwise accepted A03b trade was managed. Future work changing fills, level, target or trailing would need a genuinely new tick replay.

### Frozen original strategy

- Original EXT A02 Deribit **venue-native** 5m-pivot/prior-day second-touch signals; the candidate universe is 2,564 *if the local EXT/V2 results match the current repository*, not a hard-coded number.
- Original A03 first post-signal executed-print entry within 60 seconds, strictly after the completed signal candle.
- Original A03 already selected UTC-session VWAP or prior-day extreme structural target, without LVN target substitution.
- Owner-requested A03b mirrored stop `S=2E-T`, *gross 1:1*, with `T` untouched, 240m timeout, exact original A03b print-derived STOP/TARGET/TIMEOUT outcomes. No BE/trail on this policy, as +1R BE and a +1R target collide.
- The simulation's adverse entry/exit fill offsets remain frozen and are repriced using Claude's EXT Deribit **5 bp taker per side** convention (funding, queue, bid/ask not identified).

### Claude methods used in the bridge

All feature fields come from the saved local `V2/V2_signals.csv`, where they were calculated **before the outcome** using venue-native perpetual 1m bars and price/size prints:

- **`cvd_ok`:** after first touch through second-touch close, long requires net sell aggressor quote flow and a low that does not break the first-touch low by more than one original zone width; short is the mirror. No threshold fitted here.
- **`profile_available`:** the saved V2 profile provides a valid `PROFILE_LVN_TO_HVN` target, using the last DST-aware 18:00 New York session reset, 5 bp log bins, 3-bin smoothing, 120-minute minimum session history and 20%/50% POC LVN/HVN conditions. **This is only a yes/no profile feasibility gate for A03b: the resulting V2 target price is NOT substituted into A03b.**
- **Same original signal IDs:** EXT A03b events, V2 signal features and V2 no-CUT15 TRAIL_1M simulation must agree exactly on candidate ID, symbol, side and signal UTC timestamp. Duplicated/mismatched identities halt execution rather than giving a plausible but wrong comparison.
- **Claude V2 reference** (`TRAIL_1M`, CVD on and profile available): this fifth policy uses *already executed* Claude V2 events, which employ the different LVN→HVN target, VD18 sigma stop and structural trail. It is **not** a one-factor control for the A03b filters and must not be interpreted as such.

## Five predeclared policy panels

| Policy ID | Entry acceptance | Exit/payoff source |
|---|---|---|
| `A03B_ALL` | every original EXT candidate | frozen A03b |
| `A03B_CVD_ONLY` | `cvd_ok` | frozen A03b unchanged |
| `A03B_PROFILE_ONLY` | `v2_target_source == PROFILE_LVN_TO_HVN` | frozen A03b unchanged |
| `A03B_CVD_AND_PROFILE` | both signal-time gates | frozen A03b unchanged |
| `CLAUDE_V2_CVD_PROFILE_VOL_TRAIL` | both gates | **V2** `TRAIL_1M`, without CUT15; different TP/SL |

Each candidate appears exactly once per policy, even if ineligible (labeled `FILTERED_OUT`) or without a trade. No V2 post-entry CUT15 feature is used as an entry gate.

## Statistical and economic reporting

- Closed trades = `TARGET`, `STOP`, `TIMEOUT`. Show conditional mean assumed-cost net bps / **completed trade**, counts for targets/stops/timeouts, skipped, censored and filtered. Compare all BTC/ETH × H1/H2 and Jul–Sep-2026 separately.
- UTC-day cluster bootstrap (2,000 resamples, seed 20261020), one-sided 5th-percentile lower bound. Overlapping events share the day; the bootstrap still does **not** fix unobserved fill risks.
- Historical period labels reproduce Claude EXT: H1 before 2025-04-01, H2 from then *excluding* Jul–Sep-2026; that original quarter is `INSAMPLE_JUL_SEP_2026`. Any pooled `HOLDOUT_INSPECTED` is **not** a newly untouched holdout since Claude's results and our A04–A05a historical results are now observed.
- `realized_closed_net_bp_per_original_signal` = sum of **realized closed** net bp divided by all original signal IDs. This treats unselected/no-trade candidates as zero *contribution*, **not zero-return completed trades**. Censored outcomes are not imputed; this statistic is **not true strategy expectation** when censorship exists, and simultaneous signals are not capital-constrained.
- Paired realized-contribution differences against baseline are computed using the *same original ID*, but **exclude IDs censored by either the original or comparator**. Skipped trades may validly contribute zero. These are not a test of incremental information after multiple in-sample analyses.
- Profile and CVD selection can change target/stop composition and risk exposure. Do not treat a higher conditional hit rate from taking fewer trades as economic edge, and do not pick new thresholds from the already-inspected 2024–2026 data.

## Inputs / outputs

**Inputs — already created by the Claude notebooks:**

- `notebooks/Liquidity_Absorption/EXT/03b/03b_1to1_events.csv`
- `notebooks/Liquidity_Absorption/V2/V2_signals.csv`
- `notebooks/Liquidity_Absorption/V2/V2_events.csv`

These files may exceed the GitHub connector's preview size; the notebook loads them directly from the owner's local Git checkout. It does not require the raw futures archive at run time, does not download any files, and does not create new signal times, targets or trades.

**Outputs under `notebooks/Liquidity_Absorption/08/`:**

- `08_policy_candidate_panel.csv` — original candidate × policy, eligibility, skip/censor statuses, assumed-cost realized contribution.
- `08_policy_comparison.csv` — BTC/ETH × H1/H2/INSAMPLE/HOLDOUT_INSPECTED aggregate, original denominator and day-bootstrap lower bound.
- `08_paired_selection_effects.csv` — paired same-ID contribution differences and exclusion counts for either-side censored records.
- `08_decision.json` — exact provenance, policy list and no-confirmation status.
- `08_comparison_net_bp.png` — mean assumed-cost net bp per closed trade for the inspected holdout-era cells, grouped by BTC/ETH.

## Validation / blockers

The synthetic suite validates candidate cardinality and identity, strict gross 1:1 geometry, static filter logic, no-CUT15 V2 selection, no outcomes in eligibility, original filled outcome invariance, skip/censor accounting, day bootstrap and all notebook-cell Python syntax. **20 synthetic tests pass**. The complete historical run requires the local EXT/V2 CSVs and has not been performed by the assistant against the owner's private archive.

**Known negative prior evidence:** `EXT_decision.json` reports `FAIL_CLOSE_LA` for baseline Deribit A03b; `V2_decision.json` reports `FAIL` for Claude's CVD/profile/VD18/trail strategy. Thus, the new notebook is an interpretability / controlled-attribution exercise on inspected history, not a route to claim a fresh positive strategy. Further confirmation requires truly later unseen market data and independent cost/venue verification.

## How to execute

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_liquidity_absorption_bridge_08.py" -v
jupyter lab
```

Open `notebooks/Liquidity_Absorption/08_original_strategy_claude_method_bridge.ipynb` in the `deribit-data` kernel; select **Restart Kernel and Run All Cells**. Commit the executed notebook and the `08/` outputs to GitHub `main` for the next review. No modifications to EXT, V2, A03b or old research results are authorized by this run.
