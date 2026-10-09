# 001 — ChatGPT: Volatility Dynamics Module 01 design

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-09 |
| **Branch** | Volatility Dynamics |
| **Responds to** | Ivan's request to continue volatility research while keeping MOM/MR/VS/Bridge as economic hypotheses |
| **Status** | Module 01 built; awaiting Ivan's real-data run |

## 1. Why this branch exists

The original HSMM intuition remains economically plausible:

- **MR:** price repeatedly traverses a bounded range and reverses;
- **MOM:** displacement persists into a directional run;
- **VS:** a panic/volatility event arrives unexpectedly;
- **Bridge:** large activity is absorbed with little immediate price response before a later transition.

The empirical work so far does not justify hard-coding those four labels as latent states.

What has earned substantially more support is the **volatility dimension**:

1. HSMM 06 showed useful 30m/60m physical-volatility predictability.
2. Volatility Regression 03 found past-volatility + clock B0 explains meaningful next-bar log Parkinson variance on liquid assets; added feature blocks are much less stable.
3. Pairwise Feature Regression 02 shows ATM IV and realized volatility remain strongly related after exact five-minute clock conditioning for BTC/ETH/SOL.

Therefore the next question is not yet "which of four states are we in?" It is:

> What observable forms does the volatility process take before, during and after high-volatility events?

## 2. Objects Module 01 separates

For each origin, build a causal M1 physical-volatility forecast:

[
widehat V_{t,30},qquad widehat V_{t,60}.
]

The forecast is HAR + deterministic clock only and excludes IV.

After the outcome becomes known, define the retrospective surprise:

[
U_{t,h}
=
lograc{RV_{t,t+h}+epsilon_h}
{widehat V_{t,t+h}+epsilon_h}.
]

This creates three descriptive event classes at the 90th-percentile thresholds, asset by asset:

### Shock candidate

[
RV_{30}ge Q_{.90}(RV_{30})
]

and

[
U_{30}ge Q_{.90}(U_{30}).
]

This is the cleanest current empirical analogue of the conceptual **Volatility Shock** idea: realized variance was not merely high; it was high relative to what the causal physical-volatility model expected.

### Expected high volatility

[
RV_{30}ge Q_{.90}(RV_{30}),
qquad
widehat V_{30}ge Q_{.90}(widehat V_{30}),
]

excluding shock candidates.

This distinguishes "high volatility that the market state already warned about" from true forecast surprise.

### Forecast-high miss

[
widehat V_{30}ge Q_{.90}(widehat V_{30}),
qquad
RV_{30}<Q_{.90}(RV_{30}).
]

This is useful because any proposed volatility state must also explain false alarms.

These labels are explicitly **ex post diagnostics**, not origin-time predictors or HMM emissions.

## 3. Causality and sample handling

For each asset and expiry session:

1. all nuisance quantities are fitted on prior sessions only;
2. M1 is fitted on prior sessions only;
3. the current session receives 30m/60m forecasts;
4. realized RV/surprise are attached only for retrospective scoring.

Minimum history defaults to seven prior sessions.

No IV enters M1, preserving the distinction between physical-volatility expectation and option-implied information.

## 4. Volatility dynamics exported

Module 01 constructs observable origin-time dynamics:

### Forecast acceleration

[
A^{15}_t
=
logwidehat V_{t,30}
-
logwidehat V_{t-15m,30}
]

and the analogous 30m change.

### Forecast term slope

The forecasts are converted to per-bar variance rates before comparison:

[
Slope^{60/30}_t
=
log(widehat V_{t,60}/12)
-
log(widehat V_{t,30}/6).
]

Positive means the 60m expected variance rate exceeds the nearer 30m rate; negative means the front of the volatility curve is hotter.

### Backward volatility slope

[
Slope^{30/4h}_t
=
log(RV^{back}_{30}/6)
-
log(RV^{back}_{4h}/48).
]

These are observable candidate descriptors. None is a state label yet.

## 5. De-overlapping the 30m target

Six adjacent five-minute origins share most of the same 30m future returns.

Therefore counting every threshold exceedance as a separate event would manufacture sample size.

Module 01 clusters adjacent candidate origins of the same event type within 30 minutes and keeps the strongest representative origin.

Event counts are episode counts after this declustering, not overlapping target rows.

## 6. Exact-clock-normalized diagnostics

The event study includes available contemporaneous diagnostics such as:

- absolute 5m return;
- volume and trade count;
- absolute TFI;
- signed volume;
- close-vs-VWAP displacement;
- perp basis;
- OI return;
- book spread;
- OBI;
- total L2 depth;
- microprice displacement;
- ATM IV and IV changes;
- option spread.

Each is normalized by asset and **exact five-minute UTC position** using median/MAD.

The reason is methodological: Pairwise Feature Regression 02 demonstrated that clock structure and expiry-related coverage are substantial. We should not call ordinary US-session or pre-expiry behavior a precursor to a volatility event.

The clock z-scores are descriptive normalizations fitted on the development sample; they are not used in the causal HAR forecast.

## 7. Event-study output

For each declustered event class the module reports:

- origin diagnostic values;
- trailing 15m diagnostic averages;
- paths from -60m to +120m;
- causal volatility forecast level;
- forecast acceleration;
- forecast term slope;
- backward realized-volatility level;
- exact-clock-normalized market diagnostics.

Paths do not cross expiry-session boundaries.

The central questions after Ivan runs it are:

1. Does forecast volatility tend to rise gradually before **expected-high** episodes?
2. Are **shock** episodes genuinely flat/unremarkable beforehand, or do other observables warn before M1 does?
3. Does IV move before physical-volatility shock realization?
4. Do spreads/depth/flow deteriorate before shocks versus expected-high periods?
5. How quickly does forecast/realized volatility decay after each event class?
6. Are forecast-high misses associated with a distinct liquidity/flow structure?

## 8. Why Bridge/MOM/MR are deferred

A possible Bridge mechanism is:

[
	ext{large activity}
+
	ext{abnormally small immediate price impact}
ightarrow
	ext{later volatility/directional transition}.
]

But Module 01 should first tell us what volatility expansion and surprise look like.

If the event paths support a buildup structure, Module 02 can define an **absorption residual** such as:

[
ImpactResidual_t
=
|Delta P_t|
-
E[|Delta P_t|mid Volume_t,Volatility_t,Liquidity_t],
]

and test whether strongly negative impact residuals precede later volatility expansion.

Only after that should Module 03 test:

[
E[r_{t+h}mid D_t,	ext{volatility environment}]
]

to see whether displacement mean-reverts in stable-volatility conditions but persists in expanding-volatility conditions.

That is the empirical route back to MOM/MR rather than imposing the labels on an HMM.

## 9. Protection of existing confirmation research

Volatility Regression 011 froze an untouched confirmation window beginning with 2026-10-09 expiry sessions.

Module 01 therefore defaults to:

[
2026	ext{-}09	ext{-}22
ightarrow
2026	ext{-}10	ext{-}08
]

and refuses to cross into 2026-10-09+ unless Ivan deliberately sets:

`VD_ALLOW_CONFIRMATION_CONTAMINATION=1`.

Do **not** use that override while the Volatility Regression confirmation is intended to remain untouched.

## 10. Files

- `src/data_engineering/modeling/volatility_dynamics.py`
- `notebooks/Volatility_Dynamics/01_volatility_state_panel.ipynb`
- `notebooks/Volatility_Dynamics/README.md`

Expected compact outputs:

- `01/01_run_metadata.json`
- `01/01_fit_status.csv`
- `01/01_forecast_quality.csv`
- `01/01_event_thresholds.csv`
- `01/01_event_origins.csv`
- `01/01_event_counts.csv`
- `01/01_event_feature_summary.csv`
- `01/01_event_paths.csv`

No full state-panel dump or PNG archive is written by default.

## 11. Questions for Claude

1. Is the expected-high / shock / forecast-high-miss split the right first descriptive decomposition?
2. Is 30-minute declustering sufficient for an overlapping 30m target, or would you require a 60-minute refractory rule?
3. Do you agree that exact-clock robust normalization is preferable to pooled z-scores for the event diagnostics?
4. Which event-path variable would most clearly distinguish volatility buildup from a genuinely unanticipated shock?
5. Before Module 02, would you add a no-event matched control sampled at the same clock/TTE, or is exact-clock z-normalization enough for this exploratory pass?
6. Any concern with using expanding M1 rather than the HSMM 06 frozen-development M1 for this descriptive branch?

## 12. Decisions needed from Ivan

Run `notebooks/Volatility_Dynamics/01_volatility_state_panel.ipynb` with the default protected end date and commit the `01/` outputs.

Do not enable the confirmation-contamination override.
