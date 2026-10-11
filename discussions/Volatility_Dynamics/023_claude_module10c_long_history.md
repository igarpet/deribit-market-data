# 023 — Claude: Module 10c, a long-history physical tail for the 0DTE wings

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-11 |
| **Branch** | Volatility Dynamics (inherits Module 10's approved cross-branch reuse) |
| **Responds to** | VD 022 §5.1; Ivan's approval of the 10b verdict and of Module 10c |
| **Status** | code — downloader + Module 10c built and validated on synthetic data; awaiting Ivan's download and run |

## 1. Question

Module 10b found the wings RICH, but the realized-outcome pool covered only 18 sessions. Do the wings remain rich, **at the bid**, against about two years of out-of-sample outcomes, enough to include real crash days?

## 2. Data

`tools/download_price_history.py` downloads 5-minute closes from 2024-09-01 through the current date.
- **Primary source:** Binance spot `<ASSET>USDT`. These pairs are liquid constituents of the Deribit index and the best long proxy for it.
- **Fallback:** Deribit perpetual trade candles, used per asset where Binance spot has no pair (likely HYPE).
- Output goes to `DERIBIT_DATA_ROOT/history/price_5m/`, with a coverage manifest. Re-running resumes from the last bar.

## 3. Method

Each outcome is built **exactly** as in 10b:

$$
u_t=\ln(S_T/I_t)/\sqrt{D_t}.
$$

- $D_t$ uses the backward 4h bipower, settlement-TWAP weights, and **Module 10's own clock**, i.e. the Module 08 clock fitted on the research-DB index. Pool outcomes and quotes are therefore standardized identically.
- **Disclosed design change.** My first version refitted the clock on the long history. On the synthetic DB the gate then failed for 4 of 7 assets (sd ratio 0.78–0.83) **even though the prices were identical in the window**, because the two clocks differed. The clock is now inherited from Module 10, and the refitted clock is reported as a sensitivity check. This was fixed before any real data were seen.
- The **pool** is every session dated ≤ 2026-09-19, strictly before the first Module 10 quote. It is fully out of sample, so no LOSO is needed.
- **Pricing** is the 10b empirical price:

$$
P=\sum_i w_i\,\text{payoff}\big(Fe^{\sqrt D u_i}/\bar m,K\big),
$$

  using the quote's own $D$. The outer 0.1% of outcomes on each side are kept raw; the centre is compressed into 500 bin means.

## 4. Pre-registered (fixed before any real-data Module 10c output)

1. **Consistency gate.** On the quote window, the long-history $u$ is matched to 10b's index-based $u$ by (asset, session, minutes to expiry). An asset passes if corr ≥ 0.90 and the sd ratio is in [0.85, 1.15]. Failing assets are removed from the pool. If more than 2 assets fail, every verdict is **GATE_FAIL**.
2. **Rows.** Module 10's primary sample, unchanged.
3. **V_bid (the decision).** $M_{bid}$ is the median over sessions of the session median of $\ln(\text{bid}/P)$.
4. **V_mid (continuity with 10 and 10b).** The same statistic computed at the mid.
5. **Uncertainty.** A joint bootstrap with B = 1000 that resamples quote sessions and history dates independently. Resampling by history date keeps all assets of a date together, which preserves the common jump component (VD 020).
   - **RICH** if the 2.5% quantile > 0.
   - **CHEAP** if the 97.5% quantile < 0.
   - **CONSISTENT** otherwise.
   - Zero-price rows are dropped. If more than 25% of rows are dropped, the verdict is INSUFFICIENT_TAIL.
6. **Consequence, fixed now.**
   - **If V_bid = RICH:** the next module is a pre-registered **defined-risk** wing-selling backtest (spreads, with the long wing capping the loss), gated on expected shortfall rather than mean P&L.
   - **Otherwise:** no wing-selling strategy research from this branch until more quote sessions exist.
7. **Reported, no decision.**
   - The bid net of fees, $\min(0.03\%F,\,12.5\%\text{ premium})$; the current fee schedule needs verifying.
   - Per-asset results.
   - The own-asset pool.
   - The TTE-conditioned pool.
   - $z\times$TTE buckets.
   - Tail frequencies, overall and by volatility tercile.
   - Break-even disaster mass at ±6/8/10σ, together with the share of pool outcomes actually observed beyond each.
   - The refitted-clock sensitivity.

**How to read it.** A long pool answers the peso question directly. The break-even table shows how many unobserved disasters would still be needed, against how many the two-year history actually contains.

## 5. Validation (synthetic)

- **Self-test:**
  - returns are taken only between consecutive bars;
  - with the true-law pool the verdict is CONSISTENT, and with a thin-tailed pool it is RICH;
  - the gate passes identical outcomes and fails outcomes scaled by 1.3;
  - break-even is positive when the book is rich and 0 when it is cheap;
  - fees stay bounded.
- **Downloader:** tested offline against mocked APIs, covering resume, de-duplication, the Binance→Deribit fallback and the manifest. Its network calls were not run here, because this workspace cannot reach either API.
- **End-to-end on the synthetic DB**, with 2 years of synthetic history whose last 3 weeks equal the DB index:
  - the gate passes for all 7 assets (corr ≥ 0.998, sd ratio 1.03–1.10);
  - 1.2 million pool origins over 749 dates.

## 6. Run

**PC (Windows CMD):**

```
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_volatility_dynamics_b.py" -v
python tools\download_price_history.py --start 2024-09-01
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 notebooks\Volatility_Dynamics\10c_long_history_wing_benchmark.ipynb
git add notebooks\Volatility_Dynamics\10c_long_history_wing_benchmark.ipynb notebooks\Volatility_Dynamics\10c
git commit -m "Run Volatility Dynamics 10c"
git push
```

- **Download:** a few minutes, roughly 220 requests per asset to Binance. It needs internet access from the PC.
- **Notebook:** about 45–75 minutes. Of that, about 5 minutes builds 1.2 million outcomes twice, about 25–35 minutes is the joint bootstrap, and about 5 minutes is the break-even analysis.
- The history files live under the data root, not in the repo.

## 7. Questions for ChatGPT

1. Is Binance spot an acceptable proxy for the Deribit index in the tails, given that the basis widens in crashes? The gate tests this only on the calm quote window.
2. Should the pool be weighted toward recent history, i.e. regime-relevance, or is equal weighting over two years the right conservative choice?

## 8. Decisions needed from Ivan

1. Run the download and Module 10c as in §6.
2. Confirm that §4 is binding, including the consequence in §4.6.
