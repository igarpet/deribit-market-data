# 019 — Claude: Modules 09 and 10 — compound-Poisson jumps and the 0DTE wings

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-10 |
| **Branch** | Volatility Dynamics (Module 10 cross-branch with VRP, approved by Ivan 2026-10-10) |
| **Responds to** | VD 018 §6; Ivan's approval of Modules 09 and 10 and of the cross-branch reuse |
| **Status** | code — both modules built and validated on synthetic data; awaiting Ivan's real-data run |

## 1. Questions

1. **Module 09.** What do the Module 08 jumps look like as a compound-Poisson process? Specifically:
   - their sizes;
   - whether a market-wide component exists;
   - whether TRX's rate is an index-tick artifact;
   - $P(\ge1\text{ jump before expiry})$.
2. **Module 10.** Do OTM 0DTE options price jumps at the rate and size measured in the physical index?

## 2. Module 09 — definitions

Detection is Module 08's, unchanged: $c=4$, two-sided local volatility, Fourier clock.

**Sizes.** Sizes are measured in bps of index return and in signed local-σ units $z$.

**Tail.** The tail of $z-c$ is fitted as exponential vs GPD by profile likelihood ratio.

**Co-jumps.** Let $p_{a,t}$ be the per-bar jump probability of asset $a$. Under independence, the expected number of bars with ≥ 2 assets jumping is

$$
E_{\text{ind}}=\sum_t\Big[1-\prod_a(1-p_{a,t})-\sum_a p_{a,t}\prod_{b\ne a}(1-p_{b,t})\Big],
$$

where $p$ is each asset's full-sample rate on its valid bars.

**Rates and jump probabilities.** Rates come with a Garwood exact 95% CI, together with

$$
P(\ge1\text{ in }\tau)=1-e^{-\lambda\tau},\qquad \tau\in\{1,4,8,16,24\}\text{ h},
$$

where the CI is mapped through the same expression.

**Tick.** The tick is the smallest nonzero $|\Delta I|$ that occurs at least 5 times. For each jump, the absolute index change is expressed in ticks.

## 3. Module 09 — pre-registered

- **P1 (decision).** A common-jump component exists iff observed ≥ 2-asset bars exceed $E_{\text{ind}}$ with one-sided Poisson $p<0.01$.
- **P2 (decision, feeds Module 10).** An asset is a tick-discreteness artifact iff its median jump is ≤ 3 index ticks. Flagged assets are excluded from Module 10.
- **P3 (reported).** GPD vs exponential tail.
- **P4 (reported).** First- vs second-half rate stability: conditional binomial test, Holm across assets.

Synthetic DB result: P1 $p=0.014$, so no common component. That DB has a common volatility factor but no common jumps. No tick artifact was flagged. The self-test detects a planted common component and a planted tick artifact, and gives no false common component under independence.

## 4. Module 10 — model

For each quote, $P^{jump}$ is an undiscounted Black-76 mixture in USD:

$$
P^{jump}=\sum_{n=0}^{3}\frac{e^{-\Lambda}\Lambda^n}{n!}\,
\mathbb E\Big[\text{B76}\big(F e^{S_n-\Lambda(\bar m-1)},K,D\big)\Big],
\qquad S_n=\textstyle\sum_{i\le n}\sigma_{\text{bar}} z_i,\quad \bar m=\mathbb E[e^{\sigma_{\text{bar}}z}]
$$

- **Diffusion, ex-ante.**
  - $D=v_t\sum_j s_j^2w_j^2$.
  - $v_t$ is the **backward** 4h jump-robust bipower at the quote time, deseasonalized.
  - $s_j$ is the Module 08 clock.
  - $w_j$ is the settlement-TWAP weight of the bar: 1 before 07:30, $(m+2.5)/30$ inside the 07:30–08:00 window.
- **Jumps.**
  - $\Lambda=\lambda_a\sum_j w_j$.
  - $z_i$ are draws from the pooled signed-z distribution (40 quantile nodes).
  - $\sigma_{\text{bar}}$ is the rms clock-scaled local σ.
  - $\lambda_a$ and the $z$ distribution are **leave-one-session-out**, so a session's own jumps never price it.
  - $n=2,3$ use 64 fixed draws, and $P(n\ge4)$ is folded into $n=3$.
- **Pricing utilities.** Black-76 follows the VRP `option_pricing` conventions. Coin-quoted options are converted to USD, and puts are computed directly, not by parity, to keep tail precision. A NumPy erfc with relative error < 1.2e-7 matches `op.black76_usd`.
- **Physical moneyness.** $z=|\ln K/F|/\sqrt D$.

## 5. Module 10 — pre-registered (fixed before any real-data output)

- **Sample.**
  - Two-sided OTM quotes of the primary 0DTE family, 60 ≤ minutes to expiry ≤ 1200, on the 15-min grid.
  - Not floor-bound: bid > one native tick.
  - Excludes Module 09 P2 assets.
  - Excludes any session from 2026-11-07 onward (VRP confirmation boundary).
- **Primary.** $z\in[2,5)$. The statistic $M$ is the median over sessions of the session median of $\ln(\text{mid}/P^{jump})$.
  - **RICH** if the 95% session-bootstrap CI of $M$ is > 0.
  - **CHEAP** if it is < 0.
  - **CONSISTENT** otherwise.
- **Reported, no decision.**
  - Every $z\times$TTE bucket and each asset.
  - $\ln(\text{mid}/P^{diff})$ on the same rows, i.e. how much of the gap jumps close.
  - Sensitivities on the **same rows**: diffusion variance ×0.85 (VRP 006 Part E 5m-noise inflation) and λ ×0.5 / ×2.
  - The share of $P^{jump}$ outside [bid, ask].
  - Hold-to-settlement P&L in bps of $F$, buying at ask and selling at bid, against the approximate settlement price, with a session bootstrap.

Interpretation is fixed in advance:

- **RICH** means wings carry a premium over the physical jump model. That is consistent with a jump risk premium, or with model error in $D$; the sensitivities bound the latter.
- **CHEAP** would be surprising and would first be checked against the floor rule and the index basis.
- None of these verdicts is a trading result. Only the settlement P&L speaks to tradability, and it is reported, not decided.

## 6. Caveats

1. **Settlement price.** It is approximated by the mean index over the 07:35–08:00 bars, not Deribit's exact 30-minute TWAP of 1-second prints.
2. **Index basis.** The research `futures_index_price` (USD index) may differ from the index used for USDC-linear options by a small basis.
3. **5m bipower.** It is inflated by microstructure noise (VRP 006), so model prices are biased **up** and RICH becomes harder to reach. Diffusion ×0.85 is reported.
4. **Jump definition.** The jump catalogue uses two-sided local volatility (VD 017 §3). It is used here only for the jump rate and size distribution, which are estimated leave-one-session-out. Quote-time diffusion is strictly backward-looking.
5. **Sample size.** About 20 sessions. The session bootstrap is the honest unit of uncertainty.
6. **Synthetic sanity.** On the synthetic DB, quote IV (~0.6) exceeds realized volatility (0.2–0.55), and Module 10 returns RICH with $M=3.8$. The pipeline recovers the planted direction.

## 7. Files and run

- `src/data_engineering/modeling/volatility_jump_compound.py` (0.1.0) and `volatility_jump_option_wings.py` (0.1.0), each with a self-test.
- `notebooks/Volatility_Dynamics/09_compound_poisson_jumps.ipynb` writes to `09/`. `10_jump_model_vs_option_wings.ipynb` writes to `10/`. Run 09 before 10; 10 recomputes the P2 exclusion itself.
- `tests/test_volatility_dynamics_b.py`: 12 tests, all passing.

**PC (Windows CMD):**

```
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_volatility_dynamics_b.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebooks\Volatility_Dynamics\09_compound_poisson_jumps.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebooks\Volatility_Dynamics\10_jump_model_vs_option_wings.ipynb
git add notebooks\Volatility_Dynamics\09_compound_poisson_jumps.ipynb notebooks\Volatility_Dynamics\09 notebooks\Volatility_Dynamics\10_jump_model_vs_option_wings.ipynb notebooks\Volatility_Dynamics\10
git commit -m "Run Volatility Dynamics 09 and 10"
git push
```

`10/10_priced_quotes.csv.gz` may be large. If it exceeds about 50 MB, leave it out of the commit.

## 8. Questions for ChatGPT

1. Is a session-bootstrap median of $\ln(\text{mid}/P^{jump})$ in $z\in[2,5)$ the right single primary statistic, or would you pre-register a different bucket before any real-data output?
2. Do you agree that the 07:35–08:00 mean index is an acceptable settlement proxy for a reported, non-decision P&L?

## 9. Decisions needed from Ivan

1. Run Modules 09 and 10 as in §7.
2. Confirm that §3 and §5 are binding.
