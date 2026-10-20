# Liquidity Absorption 023 — A06 protocol: Deribit derivatives context around entries (−240 … +240 min)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-20 |
| **Branch** | Liquidity_Absorption |
| **Responds to** | Ivan (2026-10-20): use the downloaded Deribit history to see what happens at entries and ±240 min, and what distinguishes winners; 018 (A05a), 022 (A05b review) |
| **Status** | Protocol + code, written before any real-data run. Engine `liquidity_absorption_stage_a06.py` (LA-A06-0.1.0); notebook `06_deribit_entry_context.ipynb` |

## 1. Why this study and not a repeat of A05a/A05b

- **A05a** already covered −240…+240 minutes on Binance 1m price and taker flow. Winners and losers share an adverse approach; target geometry (target distance / σ60) separates them far more than any pre-signal flow.
- **A05b** showed the Deribit perpetual is too sparse for millisecond work (about 1 print/s).

A06 uses only what **Binance lacks and Deribit has**, at 1-minute resolution, anchored on the **entry**:

| Source | Features (all signed to the trade side; + = favourable) |
|---|---|
| Perpetual | aligned price path (bp from the last print before entry); taker imbalance; relative activity; **liquidations of positions on the trade side** (`liq_with`: forced sells into a long's support) and **against it** (`liq_against`); mark–index premium; block-trade amount |
| Options | short-dated (1–35 d) ATM IV (\|ln K/S\| ≤ 3%), change vs entry; put-minus-call IV (OTM 4–12%), change vs entry, signed so + = skew moving in the trade's favour; taker-signed option delta flow (call buys − call sells − put buys + put sells); option amount; option blocks |
| Dated futures | nearest (≥ 3 d) annualized basis vs index, change vs entry |

**Liquidation side rule:** Deribit's `liquidation` field marks `T` (taker side liquidated), `M` (maker side) or `MT`.
- A taker sell or a maker on a taker buy = a liquidated **long**.
- The mirror cases = a liquidated **short**.

**The absorption hypothesis in this language:** a long at support that works should see `liq_with` (longs being force-sold into the level) and no follow-through.

## 2. Cohort, groups, controls

- **Cohort:** the frozen A04 cohort, 415 entries with a time (TARGET 119 / STOP 202 / TIMEOUT 93 / CENSORED 1). No recomputation of A02–A04.
- **Groups:** TARGET vs STOP (primary contrast), split by geometry (near: target ≤ 1σ60; far: > 1σ60), because A05a showed geometry dominates. Also **WIN vs LOSS** by gross sign (Ivan's "trades that end up winning").
- **Controls:** **SIGNAL vs CONTROL** uses 3 controls per event:
  - same symbol, same UTC hour, a random day in Jul–Sep 2026;
  - at least 240 min from any signal of that symbol;
  - side copied from the event.

  This separates "what happens before *our* entries" from "what happens at any time of day".

## 3. Phases and tests

| Phase | Minutes | Pre-entry? |
|---|---|---|
| context | −240 … −61 | yes |
| approach | −60 … −16 | yes |
| formation | −15 … −1 | yes |
| entry | 0 … +5 | no |
| early | +6 … +60 | no |
| late | +61 … +240 | no |

**Per phase:**
- price change;
- sums of liquidations, blocks and option flow;
- means of imbalance, activity and premium;
- end-minus-start change of basis, ATM IV and skew.

**Tests:** Mann–Whitney two-sided, with rank-biserial effect size. BH-FDR runs **within each comparison** over pre-entry phase × feature (39 tests), and separately over everything.

## 4. How results may be used (binding)

- **Pre-entry** differences with q ≤ 0.10 are **candidate filters only**. They become usable only by passing the untouched 2024-01 → 2026-06 holdout proposed in LA 022 §3(a), pre-registered before that run.
- **Entry/post-entry** differences are anatomy. TARGET vs STOP is *defined* by the post-entry path, so these cannot be filters.
- **No in-sample cut-offs, rule additions or "refined" variants** are evaluated on Jul–Sep 2026. The winners in this quarter have been examined since A02. Selecting a rule from them and measuring it on the same trades guarantees an apparent improvement.

## 5. Validation before this turn (`tests/test_liquidity_absorption_stage_a06.py`, 5 tests pass; synthetic notebook smoke run)

- **Liquidation side mapping** (T/M × buy/sell) and **instrument parsing** checked.
- **BH** matches the reference.
- **Controls:** hour-matched and ≥ 240 min from any signal.
- **End to end:** planted on-side liquidations in the 10 minutes before TARGET entries are recovered in *formation* only (q < 0.01), not in *context*.
- **Reference:** price at minute −1 is exactly 0 bp, as required.

## 6. Running it (PC, Windows CMD)

Close the notebook in Jupyter first. It reads `DERIBIT_DATA_ROOT\history\trades\{BTC,ETH}\{future,option}\2026\*.parquet` for 2026-06-30 → 2026-10-01. If option files are missing, the option features come out NaN and the coverage file shows it.

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest tests.test_liquidity_absorption_stage_a06 -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebooks\Liquidity_Absorption\06_deribit_entry_context.ipynb
git add notebooks\Liquidity_Absorption
git commit -m "LA A06 run"
git push
```

`06_event_minute_panel.csv.gz` is about 1,800 anchors × 481 minutes. If it exceeds GitHub's limit, keep it local and push the rest.

## Questions for ChatGPT

1. Is copying the event's side onto controls the right alignment, or should controls be reported unsigned?
2. Is the 1–35 d / 4–12% OTM skew definition acceptable, given trade-based (not quote-based) IV?

## Decisions needed from Ivan

1. Run A06 as specified.
2. Accept §4: anything found here is tested only on the LA 022 holdout, not refined on Jul–Sep 2026.
