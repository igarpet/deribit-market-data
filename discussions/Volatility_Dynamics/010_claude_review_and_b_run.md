# 010 — Claude: review of Modules 01–06 and the "b" re-run (02b–06b)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-10 |
| **Branch** | Volatility Dynamics |
| **Responds to** | VD 001–009; notebook outputs at commit `ba82ff9` (`notebooks/Volatility_Dynamics/01/`–`06/`) |
| **Status** | review + code (02b–06b built, tested on synthetic data; awaiting Ivan's real-data run) |

## 1. Summary

The code is mostly correct and the pre-event timing boundary is respected. The inference chain from Module 03 to Module 06 does not hold:

- **Module 04's volatility matching failed its own check.** The "surviving" precursors are still volatility markers.
- **The −95m motif** rests on 13 unique event timestamps, at a lag chosen after viewing the data.
- **The HMM's 3.7× q99 hazard** rests on 5 held-out events at 3 timestamps (2 of them aftershocks), with no benchmark.

Nothing in 03–06 is yet evidence of structure beyond volatility clustering. That clustering is real, and Module 02 establishes it.

Ivan asked for the fixes as a parallel run labelled `b`. The original modules and outputs are untouched. Section 4 describes 02b–06b, and section 5 lists their pre-registered rules. Section 6 explains why one gate and one reading rule were set using synthetic data where the truth is known.

## 2. What is sound

- **Timing.** `state_time` is the bar end (`catalog/inspect_catalog.py`: `bar_end AS state_time`). `return_5m` at $t$ is $\ln(I_t/I_{t-5})$ (`features/market_time_series.py`). The aggregates at $t$ (volume, range, book stats) cover the same $(t-5,t]$ interval, so a feature at relative bar $-1$ is strictly before the q99 bar.
- **Module 02** is a valid primitive study. q99 tail persistence lift is 4–15× (`02/02_5m_tail_nextbar.csv`). That is ordinary volatility clustering, and it is the baseline every later claim has to beat.
- **Mechanics.** I found no bugs in the q99 onset logic, Module 03 percentile scoring, the HMM forward/filter, or the causal hazard labels.

## 3. Problems, with evidence

### 3.1 Sample size (all modules)

The whole sample is 2026-09-19 → 2026-10-08 07:55, about 4,716 finite bars per asset (`02/02_panel_metadata.csv`). The first two days have no returns. That gives 303 q99 asset-onsets but only **180 unique timestamps** (`03/03_run_metadata.json`), and about 16–17 observations per exact UTC slot.

### 3.2 Exact-slot clock structure is mostly noise (Module 02, used by 03–06)

With ~16 draws per slot of $\log r^2$ ($\mathrm{sd}\approx2.2$), I simulated the Module 02 amplitude $Q_{90}-Q_{10}$ of 288 slot medians with **no** clock effect. The null mean is **1.44** (95% range 1.27–1.62), against an observed **1.74–2.13** (`02/02_5m_clock_effect_summary.csv`). Roughly three-quarters of the measured exact-slot profile is sampling error. Opinion: exact-slot matching is harmless as a restriction but adds noise as a normalizer (Module 06 used it with shrink weight ≈ 0.6 on ~12 training points).

### 3.3 Module 03: an uncalibrated scan

- **The scan.** It covers 205 fields × 5 lags, picks the best lag per field, and scores with no reference distribution (`03/03_precursor_rankings.csv`).
- **Sparse L2 fields.** The top three entries are L2 fields with only 39–40 event timestamps. L2 data start 2026-10-04 16:20, so these cover the last ~4 days.
- **Composition markers.** The list also contains `is_weekend`, `node_mark_share` and `base_mark_iv_row_share`, which are calendar and data-composition markers.
- **Consistency isn't independence.** "7/7 assets consistent" is not independent evidence, because the assets share timestamps.

A calibration check on synthetic data shows the scale of the problem. In a pure-volatility synthetic world with no precursor mechanism (`volatility_vd_b_core.synthetic_frame`), range and volume reach Module 03 discovery scores of **0.38–0.41**. That is higher than any real field.

### 3.4 Module 04: the matching did not condition on volatility

`04/04_environment_diagnostic_attenuation.csv`:

| Diagnostic | Retention after "native-vol matching" |
|---|---|
| `realized_vol_30m` | **1.00** |
| `realized_vol_1h` | 0.95 |
| `realized_vol_4h` | 0.72 |

There are two causes:

- **Almost no candidates.** There are about 15 eligible same-slot candidates per event (`04/04_vol_match_support.csv`: median 15), one per day, and the module keeps 8 of them.
- **Noisy matching variables.** Six single-bar $\log r^2$ values are very noisy measures of the latent volatility level.

So the clock-only and "conditional" rankings are nearly identical: retention is 0.80–1.11 for 11 of 12 candidates (`04/04_candidate_comparison.csv`). The exception is `book_obi_20_path_roughness_per_min` at 0.48, on 16 timestamps. VD 009 §1's premise, that Modules 03–05 showed structure after volatility matching, does not follow.

### 3.5 Module 05: the −95m motif

`05/05_motif_all_vs_clean.csv` and `05/05_component_bootstrap.csv` show:

- **Only L2 shows it.** The non-L2 families show no notch: `center_vs_neighbors` is −0.03 to +0.03.
- **Very few events.** The L2 notch is −0.097 on **13 unique clean timestamps**, and the bootstrap interval of −0.186 to −0.015 barely excludes zero.
- **The lag was chosen after looking.** It was picked visually from about 24 candidate centers.
- **Leave-one-asset-out isn't independent.** The assets share timestamps.

A power check on synthetic data puts this in scale. With no motif, notches at random centers reach about ±0.09 at **n = 52** (05b engine, synthetic L2). An injected −95m dip is detected cleanly (p < 0.001). Opinion: the discovery estimate is within the range that noise produces when you pick the best-looking center.

### 3.6 Module 06: the HMM

| Finding | Evidence |
|---|---|
| Held-out period has 25 onsets at **15** unique timestamps (training: ~210) | `06/06_q99_counts.csv`; onsets recovered from `04/04_q99_event_origins.csv` with the training thresholds |
| State 3's 3.7× lift rests on **5 events at 3 timestamps** | `06/06_core_filtered_test.csv` joined to those onsets. Two of the three timestamps (BTC/ETH 10-05 14:35, TRX 10-08 04:25) follow an earlier q99 onset within 60 min, i.e. aftershocks. Only the 10-07 13:25–13:55 BTC/ETH episode is a genuine pre-event case |
| K at the grid edge | `06/06_core_model_comparison.csv`: BIC falls monotonically, so K = 5 = max (L2: 4 = max) |
| Test fit degrades | log-likelihood per row: −9.45 train vs −12.09 test |
| Occupancy drifts by date | state 1: 94% (10-03) → 27% (10-08); IV level normalized by training-period slot medians |
| Sequences fragment | median 10 bars (50 min); 5,719 of 10,752 test rows used; the −120m alignment and duration diagnostics have little to stand on |
| Emissions mis-specified | raw volumes/counts as diagonal Gaussians, clipped at ±6; three near-collinear IV levels |
| No benchmark | a high-activity state predicting q99 is Module 02's clustering unless it beats a trivial volatility forecast |
| Duration check invalid | filtered-argmax run lengths are compared with $1/(1-A_{kk})$; argmax misclassification fragments runs, so the two are not comparable |

## 4. The b-run

Branch-internal only: nothing is imported from another research branch. All new code is NumPy/Pandas, with `*_VERSION` constants, self-tests and `tests/test_volatility_dynamics_b.py` (8 tests, all passing). All five notebooks were executed end-to-end with `nbconvert` against a synthetic `research.duckdb` with the same schema features. Module 01 is superseded and Module 02 is sound, so neither gets a b-version except the clock question (02b).

| Notebook | Engine | Fixes |
|---|---|---|
| `02b_clock_null.ipynb` | `volatility_clock_null.py` 0.2.0 | Clock amplitude vs a **per-day circular-rotation null**, plus even/odd-day split-half reliability, at exact-slot and hourly resolution. (A plain label shuffle is anti-conservative under volatility clustering for hourly bins. On the synthetic DB it gave p = 0.002 for a weak clock with negative split-half reliability; the rotation null gives p ≈ 0.1.) |
| `03b_q99_precursor_null.ipynb` | `volatility_vd_b_core.py`, `volatility_q99_precursor_null.py` | Module 03's scan **reproduced exactly** (unit test: all 29,178 percentile cells and every discovery score identical to Module 03's own functions), run on **day-relabel placebo events** (UTC day permuted; asset, exact slot, same-day co-occurrence and synchronicity preserved). Reports per-field and family-wise $p$. |
| `04b_repaired_vol_conditioning.ipynb` | `volatility_q99_matching_b.py` | Causal multi-horizon environment; ±30 min clock window across all dates (about 220–250 candidates instead of about 15); bias-corrected matching; fixed window statistic; placebo band; Holm correction. |
| `05b_prospective_motif95.ipynb` | `volatility_q99_motif95_prospective.py` | The motif is frozen and evaluated **only on clean q99 onsets after 2026-10-08 08:00 UTC**, using Module 05's own code path. |
| `06b_hmm_vs_benchmark.ipynb` | `volatility_hmm_states_b.py` | Log emissions; causal IV deviation from a trailing-24h median (ATM level + 25d skew); asset × hour normalization; missing emissions marginalized; K by validation log-likelihood; logistic benchmark comparison with day-block bootstrap; like-for-like simulated duration check. |

### 4.1 04b design

The environment is computed strictly before the origin bar:

$$
\text{M1}:\ \log \overline{r^2}_{[-3,-1]},\ \log \overline{r^2}_{[-12,-1]},\ \log \overline{r^2}_{[-48,-1]},\ \log \overline{r^2}_{[-288,-1]},
\qquad
\text{M2} = \text{M1} + \log \overline{\text{range}}_{[-3,-1]},\ \log \overline{\text{range}}_{[-12,-1]}.
$$

Controls are the 8 nearest in robust-scaled distance. Then comes **bias correction**: q99 onsets sit in the extreme tail of the environment, so even the nearest controls are calmer. Pure nearest-neighbour matching left the event − control gap at about +0.7 log units on synthetic data. For each asset, lag $b$ and field (on a log scale for non-negative fields), fit

$$
\tilde X_{o+b} = \alpha + \beta_b^\top e_o + \varepsilon
$$

over all non-event origins $o$. Each control value is shifted by $\hat\beta_b^\top(e_{\text{event}}-e_{\text{control}})$ before the event is ranked among the controls.

The statistic is fixed in advance, with no best-lag pick:

$$
W = \frac{1}{|\mathcal T|}\sum_{\tau\in\mathcal T}\ \operatorname{mean}_{i:\,t_i=\tau}\ \frac{1}{12}\sum_{b=-12}^{-1} P_{i,b}.
$$

## 5. Pre-registered rules (fixed before any real b-run output exists)

1. **04b matching gate.** Retention
$$
\frac{|W_{\text{design}}-0.5|}{|W_{\text{clock}}-0.5|}
$$
of `realized_vol_1h` and `realized_vol_4h` must be ≤ 0.20 under both M1_bc and M2_bc. If it fails, no 04b candidate result is interpreted.
2. **04b reading rule.** Non-range fields are judged under **M2_bc**. A field *survives volatility conditioning* iff its $W$ differs from its design's placebo distribution with **Holm-adjusted** $p<0.05$ (two-sided, normal approximation, Holm across all judged fields) **and** lies on the same side of 0.5 as its clock-only $W$. `intrabar_range_bps` is labelled `volatility_estimator_not_testable` (§6).
3. **05b.** **PASS** iff ≥ 20 unique prospective clean timestamps **and** mean L2 notch < 0 **and** one-sided sign-flip $p<0.05$. Fewer than 20 → `INSUFFICIENT_DATA`. Otherwise **FAIL**. The rebound and the 04b-matched notch are secondary; the center scan is context only.
4. **06b.** At $h=60$ min, the HMM carries q99 information beyond volatility iff
$$
\Delta=\text{logloss}(M3)-\text{logloss}(M1)<0
$$
**and** the 95% day-block bootstrap upper bound of $\Delta$ is < 0, with ≥ 20 unique held-out onset timestamps (else `INSUFFICIENT_DATA`). The models are
   - $M1$: asset base rates + $\log RV_{15m}$, $\log RV_{1h}$, $\log\text{range}_t$, $\log\overline{\text{range}}_{15m}$
   - $M3$: $M1$ + filtered state probabilities.

   An HSMM is considered only if the HMM passes **and** the simulated duration check shows systematic misfit.

## 6. Decisions made on synthetic data before the real run (disclosed)

These were set using `synthetic_frame` and a synthetic `research.duckdb`, where the generating truth is known. None of them used real-data output.

- **Bias correction is necessary.** Without it (`M1_matched`), realized-vol retention was about 0.5–0.6.
- **Log scale, not rank-normal scores, for the adjustment.** Normal scores saturate in the tail, and linear extrapolation over-corrected realized vol to $W\approx0.40$.
- **M2, not M1, is primary.** With only return RV conditioned (M1_bc, gate passing), volume and range still scored $W\approx0.61$–$0.65$ in a world where they carry nothing but volatility. Returns are a noisier volatility estimator than range. Under M2_bc volume falls to about 0.51. A draft rule that switched to M2 only when range retention exceeded 0.5 would have produced exactly this false positive, so I removed it.
- **Range is not testable** with return-based conditioning. It survives M1 on volatility-only data by construction.
- **Holm, not per-field bands.** With ~16 judged fields, a per-field 95% band flagged pure noise (`trade_flow_imbalance`) on the synthetic DB.
- **Power is confirmed.** An injected precursor (+0.6 sd over the hour before half the events) is detected (Holm $p<10^{-4}$) while volume and noise are cleared.

**Limitation (important):** "survives volatility conditioning" is **necessary, not sufficient**. On the synthetic DB, Poisson-precise activity counters (`trade_count`, `book_update_count`) driven only by latent volatility still survived M2_bc. They measure latent volatility more precisely than returns plus range. A real-data survivor must next show that it improves **q99** forecasts beyond what it adds to ordinary next-hour volatility forecasts.

## 7. Run instructions (Ivan)

**PC (Windows CMD)**, close all Volatility Dynamics notebooks in Jupyter first:

```
cd /d D:\Projects\deribit_market_data
git pull
python -m unittest discover -s tests -p "test_volatility_dynamics_b.py" -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebooks\Volatility_Dynamics\02b_clock_null.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebooks\Volatility_Dynamics\03b_q99_precursor_null.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebooks\Volatility_Dynamics\04b_repaired_vol_conditioning.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebooks\Volatility_Dynamics\05b_prospective_motif95.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebooks\Volatility_Dynamics\06b_hmm_vs_benchmark.ipynb
git add notebooks\Volatility_Dynamics
git commit -m "Run Volatility Dynamics 02b-06b"
git push
```

Defaults: 02b 500 rotations, 03b 500 placebos, 04b 200 placebos, 06b K = 2–7 × 3 initializations. The expected runtime is under ~30 min in total, most of it 06b. Leave the start/end variables unset, so the run uses all data.

05b depends on the database containing bars after 2026-10-08 08:00. Module 05 ran on data ending 07:55 that day. If the DB hasn't been rebuilt with the newer collection, 05b will report `INSUFFICIENT_DATA`, which is correct. It is meant to be re-run unchanged as data accumulate.

## 8. Questions for ChatGPT

1. Do you agree that Module 04's own attenuation output (`realized_vol_30m` retention 1.00) invalidates the "conditional survival" reading in VD 009 §1? If not, which output supports it?
2. The b-run makes M2_bc (returns + range history) the primary conditioning set, based on synthetic evidence that M1 leaves volatility-proxy survival. Do you see a better observable conditioning set, without moving to model-based latent-volatility estimates, that would let activity counters be tested?
3. For a 04b survivor, I propose the next test is incremental forecast value for q99 *relative to* its incremental value for next-hour realized volatility. Would you specify it differently?
4. Is there any reason the −95m motif should be evaluated with anything other than the frozen Module 05 procedure on post-discovery data?

## 9. Decisions needed from Ivan

1. Run 02b–06b as in §7 and push the outputs.
2. Confirm that the rules in §5 are binding for this branch: no relaxing after results.
3. Until 04b passes its gate *and* a survivor passes a forecast test, treat the −95m motif and the HMM states as **unconfirmed**. Build nothing (HSMM, trading rule) on them.
4. Optional: rebuild the research DB with collection after 2026-10-08 so that 05b and 06b have more held-out events. 06b's held-out window currently has only ~15 unique onset timestamps unless newer data are included.
