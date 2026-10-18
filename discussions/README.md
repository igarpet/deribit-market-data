# Discussions — how we work

Written research debate between **Claude** and **ChatGPT** on this repository. **Ivan** moderates, relays between the two, runs the notebooks and decides. This file is the operating procedure from 2026-10-08 on; read it before writing a turn.

## 1. Roles

| Who | Does |
|---|---|
| Ivan | Runs notebooks on his PC against the research DuckDB, commits the outputs, tells each model when there is something to read, makes every decision. |
| Claude / ChatGPT | Pull the repo, review results and each other's code and logic, write or fix code, post discussion turns. Both may push code. |

Proposals are not decisions. A spec or code change is authoritative only when it is implemented and committed, or Ivan decides it in writing.

## 2. Research branches

Each research line is a **branch**: one folder here and one under `notebooks/`, with the same name.

| Branch | Discussions | Notebooks |
|---|---|---|
| Initial research (foundational, shared by all branches) | — | `notebooks/initial_research/` |
| HSMM (latent regime / volatility state) | `HSMM/` | `notebooks/HSMM/` |
| Volatility Risk Premium | `Volatility_Risk_Premium/` | `notebooks/Volatility_Risk_Premium/` |
| Volatility Regression | `Volatility_Regression/` | `notebooks/Volatility_Regression/` |
| Pairwise Feature Regression | `Pairwise_Feature_Regression/` | `notebooks/Pairwise_Feature_Regression/` |
| Volatility Dynamics | `Volatility_Dynamics/` | `notebooks/Volatility_Dynamics/` |
| Order Book Imbalance (proposed 2026-10-11; **closed 2026-10-17 as covered**, OBI 001) | `Order_Book_Imbalance/` | — (no notebooks) |
| Hurst Regimes (2026-10-11) | `Hurst_Regimes/` | `notebooks/Hurst_Regimes/` |
| Breakout Trend (2026-10-13) | `Breakout_Trend/` | `notebooks/Breakout_Trend/` |
| VWAP Events (2026-10-13, concluded VWAP 002) | `VWAP_Events/` | `notebooks/VWAP_Events/` |
| Regime Structure (2026-10-13) | `Regime_Structure/` | `notebooks/Regime_Structure/` |
| Signed Jumps (2026-10-16) | `Signed_Jumps/` | `notebooks/Signed_Jumps/` |
| Predictive State HMM (2026-10-17) | `Predictive_State_HMM/` | `notebooks/Predictive_State_HMM/` |
| Multi-Scale HMM (2026-10-17) | `Multi_Scale_HMM/` | `notebooks/Multi_Scale_HMM/` |
| Funding Carry (2026-10-18) | `Funding_Carry/` | `notebooks/Funding_Carry/` |

A new branch gets both folders, each with a `README.md` that states its question, sequence and current status. Branches can run in parallel.

**Research branches are independent experiments.** They may share the common data-engineering/database layer, but they must not silently inherit another branch's research outputs, fitted models, frozen specifications, feature selections, thresholds, state definitions, or empirical conclusions. Cross-branch reuse requires Ivan's explicit approval and must be documented in that branch.

## 3. Numbering (per branch)

- **Discussion turns** are numbered per branch, three digits, from `001`: `NNN_<author>_<topic>.md`, e.g. `HSMM/017_chatgpt_....md`, `Volatility_Risk_Premium/007_claude_....md`.
- **Notebooks** are numbered per branch, two digits, from `01`: `NN_<topic>.ipynb`.
- **Generated outputs** (CSV/JSON) carry the number of the notebook that wrote them and live in that notebook's numbered subfolder: `notebooks/HSMM/06/06_screening_metrics.csv`. (Moved 2026-10-09; discussion turns written before then cite the old flat paths.)
- **References** always include the branch: `HSMM 06` for a notebook, `VRP 004 §2` for a discussion section, `HSMM/06_frozen_spec_v02.json` for a file. A bare number is ambiguous.
- Before writing a turn, `git pull` and take the next free number in that branch.

### Mapping from the old global numbers (before 2026-10-08)

Historical files were renamed, not edited, so their text still uses the old numbers.

| Old | New |
|---|---|
| Notebooks 01–06 (root), 06b | `initial_research/01`–`06`, `initial_research/07_displacement_tests` |
| Notebook/Module 07, 08, 09, 10, 11 | HSMM 01, 02, 03, 04, 05 |
| Module 12 (volatility HAR, frozen spec) | HSMM 06 (`06_frozen_spec_v02.json`) |
| Module 13 (volatility HMM prototype) | HSMM 07 |
| Module 14 (BTC S2 HAR-GARCH null) | HSMM 08 |
| Notebook 15 (VRP pricing audit) | VRP 01 |
| Discussions 001–016 | HSMM 001–016 (unchanged) |
| Discussions 017–022 | VRP 001–006 |

Environment variables keep their old names (`M13_BLOCKS`, `M14_NULL_SCOPE`, `VRP_*`).

## 4. Writing a turn

1. One file per turn. Never edit another participant's past file. To correct or answer it, write a new turn.
2. Start with the header table:

   ```
   | | |
   |---|---|
   | **Author** | Claude / ChatGPT |
   | **Date** | YYYY-MM-DD |
   | **Branch** | ... |
   | **Responds to** | file(s) or commit |
   | **Status** | proposal / review / results / ... |
   ```
3. Back claims with evidence: notebook outputs, CSVs, commits, references. Label opinion as opinion.
4. Formulas in LaTeX.
5. End with **Questions for <the other model>** and **Decisions needed from Ivan**.

## 5. Code and notebooks

- Reusable code lives in `src/data_engineering/modeling/` as versioned modules (`*_VERSION` constants). Notebooks import it; they do not hide core logic in cells.
- Every notebook writes its key results to CSV/JSON in its numbered subfolder (`NN/`), so results can be reviewed from the repo without saved cell outputs.
- New or changed code is tested before pushing (synthetic data where real data is not available in the reviewer's environment).
- Notebooks locate the repo by walking up to the folder containing `src/data_engineering`; data comes from `DERIBIT_DATA_ROOT` (`.env`).

## 6. The loop for one step

1. A model builds or fixes code, tests it, commits and pushes, and posts a turn saying what to run.
2. Ivan pulls, runs the notebook (`jupyter nbconvert --to notebook --execute --inplace ...`), commits the notebook and its outputs, pushes.
   - Close the notebook in Jupyter before pulling. Otherwise Jupyter saves its stale copy over the update.
3. Ivan tells the model(s) to check. They read the outputs, interpret them, and post the next turn.
4. Ivan decides.

## 7. Research discipline

- Gates and pass thresholds are set **before** results are seen and are not relaxed afterwards. A failed gate is reported as failed, with its consequences.
- Frozen specifications (e.g. `notebooks/HSMM/06_frozen_spec_v02.json`) and holdout windows are declared in advance and not touched.
- Agreement between the models is not evidence. Held-out results and null tests decide.
- The framework document (`SDF_HSMM_0DTE_Regime_Framework v3.2`) binds the HSMM price branch unless Ivan says otherwise.

## 8. Research markets — owner instruction, 2026-10-17

**Perpetual futures AND options are independent eligible research/trading markets** from now on. A validated, executable perpetual-futures opportunity does not require an associated options edge, and vice versa. This broadens the project mandate; it does not retroactively change prior branch protocols, undo negative findings, or authorize live orders.

For all NEW studies: assess forecast skill at the intended instrument/horizon, then instrument-specific economics using actual contract prices or executable quote/fill data, maker/taker fees, slippage, spreads, funding, latency and risk/position constraints where applicable. Keep **statistical significance**, **economic significance after costs**, and **prospective confirmation** as separate verdicts. Do not substitute Binance spot movements for Deribit perpetual executable returns or option premium returns. Cross-market research output reuse still needs the owner's approval and a written declaration under §2. The user retains final approval for any live deployment.

The new `Predictive_State_HMM` branch implements this mandate as an exploratory two-market research line; it does not modify any archived HSMM, VD19, signed-jump or option-strategy frozen study.
