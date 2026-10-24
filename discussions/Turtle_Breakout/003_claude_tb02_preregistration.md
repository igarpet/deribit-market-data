# 003 — Claude: TB 02 pre-registration — the turtle portfolio against proper benchmarks

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-25 |
| **Branch** | Turtle Breakout |
| **Responds to** | TB 002 §2; Ivan, 2026-10-25: "first I want to see if it beats proper benchmarks" |
| **Status** | pre-registration + code (validated on synthetic data); awaiting Ivan's run |

## 1. Question

Does the TB 01 turtle portfolio beat **doing nothing**, when doing nothing is a sensible, rebalanced position in the same four markets?

## 2. Strategy (unchanged)

TB 01 primary, unchanged:

| Setting | Value |
|---|---|
| Module | `turtle_breakout.Params()` defaults |
| Bars and side | daily, long only |
| Assets | BTC/ETH/SOL/XRP |
| Sizing | 2% HWM |
| Costs | fees + slippage + funding |
| Window | 2020-10-15 → end of data |

- **Re-simulation.** The notebook re-simulates it deterministically and reports whether it reproduces TB 01's trades.
- **Data end.** The data now ends later than TB 01's run, so the window extends to the new end date.

## 3. Benchmarks

**Common construction** (`turtle_benchmarks.py`):
- **Weights:** equal across the four assets.
- **Size:** total notional $=L\times$ equity, with $L$ = the strategy's mean gross leverage (TB 01: 0.41).
- **Marking:** at the daily 17:00 New York close.
- **Rebalancing cost:** taker + slippage on every rebalance trade.
- **Cash:** idle cash earns nothing, as for the strategy.

**Benchmarks:**

| | Instrument | Rebalance | Funding | Role |
|---|---|---|---|---|
| **B1** | perpetuals | monthly (first NY day of each month) | paid | gated |
| **B2** | perpetuals | daily | paid | gated |
| **B3** | spot proxy: perpetual price path, no funding | monthly | none | gated |
| B4 | perpetuals, inverse 60-day volatility weights | monthly | paid | reported |
| B1-vol | B1 levered to the strategy's realised volatility | monthly | paid | reported |

- **B3 basis.** The spot–perpetual basis s.d. is about 1 bp (Pairs fOU 004).
- **Why $L$ matters little.** The Sharpe of a constant-mix portfolio is almost independent of $L$, so the gated tests do not depend on matching leverage. CAGR and drawdown do, which is why B1-vol is reported.

## 4. Decision (binding)

**Test.** For each of B1, B2 and B3:
$$\Delta SR = SR_{\text{strategy}} - SR_{B}$$
- **Returns:** daily.
- **Uncertainty:** 21-day circular block bootstrap, 2,000 draws, seed 20261025.
- **Pass:** q2.5 > 0.

**Verdict.**
- `BEATS_ALL_BENCHMARKS` only if all three pass.
- Otherwise `FAILS_<list>`.

**Multiplicity.** This is an intersection–union test. Requiring every comparison to pass at 95% needs no multiplicity correction, and it is deliberately strict because the benchmark is being chosen after TB 01's results.

**Reported only:**
- B4 and B1-vol Sharpe differences.
- Alpha and beta of strategy daily returns on B1 at $L=1$: OLS, Newey–West t with 5 lags.
- Calendar years.
- Halves split at 2023-07-01, with the three gated tests in each.
- Rolling 365-day Sharpe.
- Equity and drawdown curves.

## 5. Validation (synthetic)

`tests/test_turtle_benchmarks.py`: 11 tests, all passing. `test_turtle_breakout.py` still passes.

- **Panel:** closes are at 17:00 New York.
- **Constant mix:**
  - one asset, daily, no costs, equals the price path;
  - $L=0$ is flat;
  - funding makes the perpetual benchmark trail the spot proxy;
  - rebalancing costs reduce equity;
  - inverse-vol weights run.
- **Vol matching:** hits the target volatility within 0.5 pp.
- **Statistics:**
  - alpha and beta are recovered on a known model;
  - identical series give $\Delta SR=0$ and FAIL.
- **End to end:**
  - a trending synthetic market gives `BEATS_ALL_BENCHMARKS`;
  - a driftless random walk with positive funding does not;
  - the notebook runs end to end, giving `FAILS_B1_B2_B3` on the random walk.

## 6. Run (PC, Windows CMD)

No new download. It uses TB 01's `Binance\long` data. Close the notebook in Jupyter first.

```bat
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_turtle_benchmarks.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks\Turtle_Breakout\02_turtle_benchmarks.ipynb
git add notebooks\Turtle_Breakout
git commit -m "Turtle Breakout 02: run"
git push
```

**Runtime:** a few minutes.

**Outputs** go to `notebooks/Turtle_Breakout/02/`:
- `02_decision.json`, `02_sharpe_tests.csv`, `02_benchmarks.csv`;
- `02_yearly.csv`, `02_halves.csv`, `02_equity_daily.csv`;
- `02_manifest.json`.

## 7. Next (TB 03, not in scope here)

1. A point-in-time universe, including delisted perpetuals, to address survivorship.
2. Removing the largest trades, to check concentration.
3. Whatever else Ivan adds.

## 8. Decisions needed from Ivan

1. Confirm §4 is binding.
2. Run §6.
