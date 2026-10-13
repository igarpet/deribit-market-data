# Regime Structure 003 — pre-registration addendum: window-scale sweep (v1.2)

| | |
|---|---|
| Author | Claude |
| Date | 2026-10-13 |
| Branch | Regime Structure |
| Responds to | RS 002; Ivan (chat): "try all the windows you proposed, in the same module 1" |
| Status | Pre-registration, before the v1.1/v1.2 run. No result under the shared-sign null has been seen. |

## 1. Why

RS 01 (v1.0) labelled regimes on 1-hour windows. Ivan's ranges and displacements may live at a different scale. Changing the scale after a failure is a forking path. So every scale is tested in **one** run, under one family-wise correction, and nothing outside this run counts.

## 2. What changes (code `regime_structure v1.2`, notebook RS 01)

- Labels, null (shared sign per UTC minute, v1.1), tests E1–C3 and the effect-size gate are unchanged.
- **Windows:** clock-aligned windows of 3, 6, 12, 24 and 48 five-minute bars, i.e. **15 min, 30 min, 1 h, 2 h, 4 h**.
- $\hat\sigma_{\text{pre}}$ is measured before each window, and $s=\hat\sigma_{\text{pre}}\sqrt{n}$.
- Persistence and transition tests compare consecutive non-overlapping windows of the same scale.
- Quantile thresholds ($ER$ q33/q67, relative volume q25, relative realised vol q33) are computed **separately per scale** on the development window of the real data, then frozen.
- **399 null draws** (was 199). A 0.0025 threshold needs at least 399 draws to be reachable. The 1-hour family uses the same 399 draws; its threshold is unchanged.

## 3. Decision rule

| Family | Scales | Tests | α per test |
|---|---|---|---|
| Primary (RS 001, unchanged) | 1 h | E1, E2, C1, C2, C3 | 0.05 / 5 = 0.01 |
| Sweep (new) | 15 min, 30 min, 2 h, 4 h | E1, E2, C1, C2, C3 | 0.05 / 20 = 0.0025 |

A test passes iff, on the test window (2026-04-01 → end), $p\le\alpha$ **and** lift / null median $\ge 1.10$. **This run supersedes v1.0 for the 1-hour family**, as declared in RS 002.

## 4. Known property, flagged before results

SHOCK is "any 5-minute bar $\ge 3\hat\sigma$ in the window" and has first priority.
- Its base rate rises with window length: about 15% of hours in v1.0, and plausibly close to half of all 4-hour windows.
- At 4 h this crowds out DISP and RANGE labels, which weakens E1, E2 and C1 there.
- This follows from Ivan's definition and is kept as is. Base rates per scale are reported so it can be judged.

## 5. Synthetic validation

`tests/test_regime_structure.py`, 9 tests. Added:
- window alignment and sums at every scale;
- `run_asset` returns all scales;
- the sweep α is 0.0025.

A smoke run with regimes planted at 1 h shows E1 and E2 ratios of 1.09–1.72 at every scale, and no C test above 1.06.

## 6. Decisions needed from Ivan

Run RS 01 (v1.2). That one run is binding for both families.
