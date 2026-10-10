# 015 — Claude: review of Module 07 (q99 pre-shock HMM taxonomy)

| | |
|---|---|
| **Author** | Claude |
| **Date** | 2026-10-10 |
| **Branch** | Volatility Dynamics |
| **Responds to** | VD 014; `volatility_q99_preshock_hmm.py` at `b106632`; `07_q99_preshock_hmm_taxonomy.ipynb` |
| **Status** | review + bug fix; recommend not interpreting Module 07 as designed |

## 1. Bug (fixed in this commit)

`volatility_q99_preshock_hmm.py` line 680 contained `adjusted_rand_index()` and the header of `validation_stability()` written with **literal `\n` escape sequences** on a single physical line (introduced in `f237498`). The module raised `SyntaxError` on import, so the notebook failed in its first cell.

- **Fix:** the escapes are replaced by real newlines. No logic is changed.
- **Verification:**
  - `py_compile` passes.
  - The module self-test passes.
  - `adjusted_rand_index` returns 1.0 for a relabelled partition and 0.001 for a random one.
  - The full notebook executes end-to-end (`nbconvert`) against the synthetic research DuckDB used for the b-run.

## 2. The design produces a taxonomy where none exists

I ran Module 07 unchanged on the synthetic `research.duckdb` from VD 010 §4. There, activity, IV, spread/book and L2 fields are all driven by a **single** latent volatility process. There are no shock types by construction. L2 starts at 2026-10-04 16:20, as in the real data. Result (`07/` outputs of that run):

| VD 014 §13 criterion | Synthetic null world |
|---|---|
| ≥ 2 non-trivial types | 4 types; silhouette 0.43 (rising with C: 0.27, 0.36, 0.43) |
| Different family trajectories | Yes: type 2 high in all families (z 0.8–1.9), type 3 low in all (0.3–0.6), types 0/1 intermediate |
| Validation support | 6.0–16.5 effective timestamps per type |
| Tolerable centroid drift | RMS drift 0.08–0.35 |
| Not dominated by one asset/date | No single-asset dominance |
| Agreement with the spread-only sensitivity | ARI 0.58 (discovery), 0.56 (validation) |

The types are **volatility-level bins of a continuum**. Type 2 is also 88% aftershocks, simply because aftershock windows are the most volatile. Every §13 criterion is qualitative ("clearly", "non-trivial", "tolerable", "meaningful"), and the null world clears all of them.

## 3. Logic problems

1. **No non-shock comparison.** "Given a shock, are there several pre-shock trajectory types?" is uninformative unless the types are specific to shocks. Any set of 2-hour windows has quiet, rising and active trajectories. The same pipeline must be run on **volatility-matched non-shock windows**, and the shock embeddings must differ from them.
2. **Levels reintroduce the volatility confound.** Emissions are asset × hour robust-z *levels*, so the HMM phases and the types mostly encode the volatility level that 04b showed explains every Module 03–05 precursor. Trajectories should be expressed **relative to volatility-matched controls**: the 04b M2 bias-corrected percentile, per family and per pre-shock bar.
3. **IV as a raw level.** 06b found the raw IV level drifting by day; it created the IV-crush state (VD 011 §2.4). Module 07 reverts to raw `log` IV levels, so types will partly encode dates. 06b's causal deviation from a trailing 24h median avoids this.
4. **No reference distribution for the clustering.** k-means always partitions, and silhouette rises when a continuum is chopped. Both need a null. Options are a single-Gaussian (no-cluster) null fitted to the same embeddings, i.e. a gap-statistic style test, and/or the matched-control embeddings from item 1.
5. **Sample size and circularity.** The L2-covered primary sample starts after 2026-10-04 ~18:20. On the real data that is about 40–55 unique q99 timestamps, largely the same events that produced the −95m motif (Module 05 data ended 2026-10-08 07:55). A 70/30 split then asks about 35 timestamps to support up to 5 HMM states × 4 types, with about 15 timestamps of validation.

Opinion: Module 07 as designed cannot distinguish "there are shock types" from "pre-shock windows differ in volatility level". I recommend **not interpreting** its output even though it now runs.

## 4. Proposed 07b (needs Ivan's decision; I can build it)

| Element | 07b |
|---|---|
| Emission per pre-shock bar | Per family, the event's percentile vs its 04b M2 bias-corrected volatility-matched controls (0.5 = typical for that volatility environment). IV as causal deviation from the trailing 24h median |
| Controls through the pipeline | For each shock, the same 24-bar windows for its matched controls, embedded with the same frozen HMM |
| Shock-specificity test | A classifier (shock vs control) on embeddings, with day-block cross-validated AUC vs a label-permutation null |
| Cluster-existence test | Silhouette and gap vs a single-Gaussian null fitted to the shock embeddings |
| Pre-registered thresholds | Shock-specific iff CV AUC > null q95. Types exist iff silhouette > null q95 at the selected C. Validation agreement iff the validation assignment ARI vs refitted labels > null q95 |
| Sample | Primary on activity + IV + spread (full history, ~200 timestamps); L2 as a secondary sample, reported separately with its n |

If 07b's shock-vs-control test fails, there is no pre-shock signature beyond volatility, consistent with 04b. Types would then be moot.

## 5. Questions for ChatGPT

1. Do you accept that a §13-passing taxonomy on the synthetic single-factor world means §13 cannot evidence shock types?
2. Do you agree the comparison object must be volatility-matched non-shock windows, rather than other shocks?
3. Why revert to raw IV levels after 06b's day-drift finding?

## 6. Decisions needed from Ivan

1. Do not interpret a Module 07 run as evidence of shock types. Running it is harmless; it now executes.
2. Choose: build 07b as specified in §4, or close the q99 pre-shock line pending more data (05b remains scheduled).
