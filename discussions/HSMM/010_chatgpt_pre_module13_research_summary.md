# 010 — ChatGPT: pre-Module 13 research summary and hierarchical-state proposal

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-07 |
| **Purpose** | Canonical handoff before Module 13. Summarizes the corrected Module 12 evidence and the latent-state architecture discussion that followed. |
| **Evidence base** | `notebooks/12_volatility_har.ipynb`, `12_frozen_spec_v02.json`, `12_screening_metrics.csv`, `12_screening_comparisons.csv`, discussions 006–009 |
| **Status** | Research summary / proposal. No Module 13 code has been implemented from this note. |

## 1. Why this note exists

Modules 10–11 weakened the original five-minute directional-return thesis, while Module 12 showed that future 30–60 minute realized volatility has much stronger conditional structure.

Before starting Module 13, Ivan asked three architectural questions:

1. What latent states are we now trying to discover?
2. Can a second HMM be applied conditionally inside states found by a first HMM?
3. Do child HMMs need to use the same anchor / observable process as the parent HMM?

This note summarizes the empirical position after Module 12 v0.2 and proposes how those questions should reshape the latent-state architecture.

---

# Part I — What Module 12 v0.2 actually established

## 2. The return-prediction branch remains weak

Modules 10–11 did not establish robust five-minute directional alpha.

The strongest safe interpretation remains:

[
E[r_{t+1}mid X_t]
]

is difficult to forecast using the current causal market-state feature set.

That result does **not** imply the market state is useless. It motivates moving from the conditional mean of returns to the conditional variance / volatility process.

---

## 3. Future volatility is materially more predictable

Module 12 targets:

[
y_{t,h}
=
log(RV_{t,t+h}+epsilon),
qquad
hin{30m,60m}.
]

The corrected v0.2 screening results show substantial predictive structure.

M1 is:

[
HAR+Clock.
]

Development-screening (R^2) for M1:

| Asset | 30m | 60m |
|---|---:|---:|
| BTC | 0.486 | 0.541 |
| HYPE | 0.402 | 0.483 |
| ETH | 0.371 | 0.366 |
| AVAX | 0.235 | 0.266 |
| SOL | 0.203 | 0.246 |
| XRP | 0.113 | 0.105 |

These numbers are descriptive development results, not confirmation statistics.

The cleanest M1-vs-M0 improvements, positive in both all-origin and non-overlapping views, occur for:

- BTC 30m
- BTC 60m
- ETH 60m
- HYPE 30m
- HYPE 60m

This means the project now has a meaningful non-switching volatility process that future regime models must beat.

---

## 4. The strongest incremental development signal is horizon-matched implied variance

The corrected hierarchy is:

[
M1=HAR+Clock
]

[
M2a=M1+logwidetilde W^{imp}_{t,t+h}
]

[
M2b=M2a+Delta IV_{15m}+Delta IV_{30m}
]

[
M3=M2b+	ext{returns/flow/L2/basis/OI/D/ranges}
]

[
M4=M3+	ext{option-surface shape/dynamics/liquidity}.
]

After fixing the v0.1 comparison flaws, M2a shows the most coherent repeated signal.

### M2a vs M1: positive in both all-origin and non-overlapping screening views

- ETH 30m
- SOL 30m
- SOL 60m
- XRP 30m
- XRP 60m

The frozen v0.2 full-development specification admits exactly those five M2a cases.

Frozen standardized M2a coefficients:

[
ETH_{30m}: 0.165
]

[
SOL_{30m}: 0.216
]

[
SOL_{60m}: 0.148
]

[
XRP_{30m}: 0.251
]

[
XRP_{60m}: 0.273.
]

The cross-horizon consistency for SOL and XRP is currently the strongest repeated development signal in Module 12.

This remains a **candidate effect**, not a confirmed result. The binding test is the prospective 30-session confirmation sample after 2026-10-07.

---

## 5. Recent IV changes are less stable

M2b adds:

[
Delta IV_{15m},quad Delta IV_{30m}.
]

Development screening shows some positive cases, especially:

- ETH 60m
- SOL 60m

but the full-development frozen selection retains the block for ETH 60m and not for SOL 60m.

AVAX 30m is admitted in the frozen spec but lacks comparable cross-horizon stability.

Current interpretation:

[
oxed{
	ext{IV level / allocated implied variance is more stable than recent IV movement}
}
]

at this development stage.

---

## 6. M3 — L2, flow, basis/OI and VWAP displacement — is not yet stable

The M3 block includes:

- recent returns;
- volume / trade flow;
- L2 imbalance / depth / spread;
- book activity;
- basis and open interest;
- range estimators;
- signed VWAP displacement (D);
- (|D|);
- (Delta|D|).

The frozen v0.2 spec admits M3 only for BTC 30m.

However, the cleanest screening improvement appears at BTC 60m, not BTC 30m.

Therefore the location of the effect is unstable across the development split.

The correct interpretation is:

[
oxed{
	ext{M3 may contain useful information, but it has not yet shown a stable horizon/asset effect}
}
]

rather than either "M3 works" or "M3 is useless."

The variables may later prove more useful for:

- transition hazard;
- shock probability;
- observation-quality state;
- nonlinear effects;

than for the conditional mean of future log RV.

---

## 7. M4 — full option surface — is not yet a clean general signal

The frozen spec admits the new M4 block only for SOL 60m.

Some standalone M4 (R^2) values look very high, especially for ETH, but the corrected common-row nested comparison shows that these do **not** translate into robust M4-vs-M3 gains.

The main reason the standalone numbers can look attractive is population selection: the full option-surface model is available on a much smaller and more liquid subset of the session.

M4 coverage near expiry is particularly weak.

Therefore:

[
oxed{
	ext{M4 is currently a selective mid-session/full-surface model, not a general 0DTE model}
}
]

and no broad surface-alpha claim is justified yet.

---

## 8. M1c recalibration is not reliable enough to be a primary null

The training-only affine recalibration M1c behaves poorly in the short expanding-fold development sample.

Likely reason: early folds can train entirely on weekdays and then validate on weekend sessions, causing the weekend clock structure to shift abruptly.

M1c should remain a secondary diagnostic unless fold construction is fixed.

For Module 13, raw M1 should remain the principal non-switching null.

A corrected M1c can be included as a secondary benchmark.

---

# Part II — What states are we actually trying to find next?

## 9. Module 13 is not trying to rediscover the old four labels directly

The original conceptual taxonomy included:

- MR — mean-reversion-like
- MOM — momentum-like
- Volatility Shock
- Bridge / transition

Those labels remain economically interesting, but Module 13 should **not** force them into a four-state HMM.

Module 13 asks a narrower empirical question:

[
oxed{
	ext{Does the volatility process itself require a latent structural regime?}
}
]

The first latent variable is therefore a reduced volatility-regime process:

[
R_t^{vol}.
]

We should fit anonymous states:

[
R_1, R_2,dots
]

and characterize them **after fitting**.

For a first K=2 model, plausible post-fit descriptions might be:

- low / stable volatility;
- high / persistent volatility;

or perhaps:

- normal persistence;
- volatility expansion / shock.

But the model should not be told those labels in advance.

---

## 10. Module 13 is testing a reduced form of (R_t), not (Q_t)

The broader architecture distinguishes:

[
R_t=	ext{structural market regime}
]

from:

[
Q_t=	ext{fast market-quality / observation-quality state}.
]

(Q_t) is conceptually closer to:

- clean vs noisy;
- degraded book quality;
- transient observation instability;
- temporary dispersion scaling.

Module 13 should **not** estimate (Q_t).

Otherwise a high-variance state could ambiguously mean either:

- a true structural volatility regime; or
- merely a temporarily noisy observation environment.

First establish whether structural volatility switching exists.

Then revisit (Q_t) separately.

---

## 11. MR and MOM are not identifiable from a volatility-only anchor

The original MR/MOM distinction relies on directional dynamics, for example:

[
r_{t+1}
=
mu_R+eta_{D,R}D_t+epsilon_t.
]

A negative displacement slope can be MR-like:

[
eta_{D,R}<0,
]

while a positive slope can be continuation / MOM-like:

[
eta_{D,R}>0.
]

But Module 13 models future variance:

[
RV_{future}.
]

Variance has no sign.

Therefore a volatility-only HMM cannot tell us:

> mean reversion vs momentum

in the original sense.

A volatility-shock state can emerge naturally; MR/MOM require a later directional latent process or conditional dynamic model.

---

## 12. Bridge should not be forced either

A Bridge state was intended to capture something closer to:

[
	ext{structural deterioration / transition environment}
]

rather than simply high volatility.

Evidence for Bridge would probably require variables such as:

- L2 instability;
- liquidity deterioration;
- flow shifts;
- basis / OI changes;
- IV dynamics;
- transition timing.

Those variables are intentionally not part of the minimal Module 13 state anchor.

Therefore Bridge should remain a **post-fit / later-layer interpretation**, not an initial state label.

---

# Part III — Hierarchical HMM idea

## 13. The first HMM states do not need to be the final states

Ivan's key architectural observation is correct:

> a second latent model can be applied conditionally within the coarse regimes found by a first HMM.

Conceptually:

[
R_t^{(1)}
=
	ext{coarse regime}
]

and:

[
R_t^{(2)}
mid
R_t^{(1)}
=
	ext{conditional sub-regime}.
]

For example:

[
R_t^{vol}=	ext{Low-vol}
]

could later contain directional substates:

[
S_t^{dir}
in
{	ext{MR-like},	ext{MOM-like}}.
]

A high-volatility parent regime might instead contain something like:

[
S_t
in
{	ext{persistent expansion},	ext{normalization / transition}}.
]

This is only an architectural example. The child labels must still be earned empirically.

---

## 14. Do not implement the hierarchy by hard-labeling and blindly refitting

The simplest implementation would be:

1. fit HMM 1;
2. assign each bar a hard parent label;
3. split the data;
4. fit HMM 2 inside each split.

This is statistically fragile because parent-state classification errors become treated as truth.

A better development sequence is:

### Stage A — independent / conditional second-stage model

Retain parent posterior probabilities:

[
P(R_t^{(1)}=kmidmathcal F_t)
]

and use them as weights when fitting the child process.

This preserves uncertainty.

### Stage B — eventual joint hierarchical model

If both levels earn OOS value, estimate a joint hierarchical structure:

[
R_t^{coarse}
ightarrow
S_t^{sub}
ightarrow
Y_t.
]

That would be a hierarchical HMM / hierarchical HSMM style architecture.

It is more statistically coherent but substantially more complex, so it should come later.

---

# Part IV — Anchors do not need to match across layers

## 15. Parent and child HMMs can use different anchors

The child HMM does **not** need to use the same anchor as the parent HMM.

In fact, using a different anchor is desirable when the child process is intended to discover another economic dimension.

### Parent example

Module 13 parent state:

[
R_t^{vol}
]

anchored by volatility dynamics:

[
log RV_{future}
sim
HAR+Clock+R_t^{vol}.
]

Question:

> What volatility environment are we in?

### Child example

Later directional child state:

[
S_t^{dir}
]

anchored by directional dynamics:

[
r_{t+1}
=
mu_{S_t}
+
eta_{D,S_t}D_t
+
gamma_{S_t}'Flow_t
+
epsilon_t.
]

Question:

> Conditional on the volatility environment, what directional / displacement-response substate are we in?

This may eventually recover MR/MOM-type behavior in a statistically more defensible way.

---

## 16. When should the child use the same anchor?

If the goal is simply to subdivide the same phenomenon, reusing the anchor can make sense.

Example:

[
High volatility
ightarrow
egin{cases}
moderately high\
extreme high
end{cases}.
]

But if the goal is to recover another structural dimension, reusing mostly the same volatility inputs risks merely discretizing the same continuous volatility axis more finely.

For this project, the more interesting hierarchy is likely:

[
oxed{
	ext{coarse volatility structure first}
}
]

then:

[
oxed{
	ext{conditional directional / transition structure second}
}
]

rather than repeatedly subdividing realized volatility.

---

# Part V — Where (Q_t) fits in a hierarchical architecture

## 17. (Q_t) should probably remain parallel rather than automatically becoming a child of (R_t)

A useful eventual structure may be:

[
R_t^{vol}
ightarrow
S_t^{dir}
]

while separately:

[
Q_t^{quality}
]

runs as a faster process.

Then observations can depend on all three:

[
Y_t
mid
R_t^{vol},
S_t^{dir},
Q_t.
]

Conceptually:

- (R^{vol}): coarse persistent volatility environment;
- (S^{dir}): conditional directional / displacement-response structure;
- (Q): clean/noisy or market-quality modulation.

For example, (Q) could scale dispersion:

[
Y_tmid R=r,S=s,Q=q
sim
t_
u(mu_{r,s},c_qSigma_{r,s}).
]

This keeps "bad observations / temporary noisy market" from automatically becoming a structural economic regime.

---

# Part VI — Proposed Module 13 experiment

## 18. Module 13 should start with the smallest possible switching question

The first question is:

[
oxed{
K=1 HAR
quad vs quad
K=2 HMM	ext{-}HAR
}
]

using the same 30m and 60m volatility targets.

Do **not** begin with:

- four states;
- DSF;
- L2;
- IV;
- option surface;
- Q;
- transition features.

The point is to isolate the value of discrete switching itself.

---

## 19. State complexity should be introduced incrementally

Recommended sequence:

### S0 — non-switching HAR

[
y_t
=
alpha+phi'HAR_t+Clock_t+epsilon_t.
]

### S1 — switching scale only

[
epsilon_tmid R_t=k
sim
t_
u(0,sigma_k).
]

Question:

> Does the HAR residual process have discrete low/high conditional dispersion states?

### S2 — switching intercept + scale

[
y_t
=
alpha_{R_t}
+
phi'HAR_t
+
Clock_t
+
epsilon_{t,R_t}.
]

### S3 — switching HAR persistence

[
y_t
=
alpha_{R_t}
+
phi_{R_t}'HAR_t
+
Clock_t
+
epsilon_{t,R_t}.
]

Each layer must improve chronological OOS forecasting before the next is justified.

---

## 20. HMM comes before HSMM

HMM assumes geometric state durations.

Only if:

[
HMM	ext{-}HAR > HAR
]

should we test:

[
HSMM	ext{-}HAR.
]

The relevant duration comparison is:

[
oxed{
HSMM	ext{-}HAR
quad vs quad
HMM	ext{-}HAR
}
]

not merely HSMM versus HAR.

If HSMM does not beat HMM, explicit duration is unnecessary.

---

## 21. A no-regime simulation null is mandatory

Volatility is strongly persistent and long-memory-like.

A switching model can manufacture apparently meaningful low/high states even if the true process is continuous.

Therefore Module 13 should simulate from a fitted non-switching HAR-t process and rerun the exact switching-selection procedure.

Key quantities:

[
P(K>1mid	ext{no true regimes})
]

and:

[
P(
Delta QLIKE_{switch}
ge
Delta QLIKE_{observed}
mid
HAR null
).
]

State occupancy, episode counts, apparent coefficient separation and duration patterns should also be recorded under the null.

A visually convincing state decomposition is not enough.

---

## 22. Clock remains mandatory

The deterministic intraday volatility pattern must stay in every model.

Otherwise the HMM may merely rediscover:

- active hours;
- quiet hours;
- weekend effects.

Those are deterministic calendar effects, not latent structural regimes.

---

## 23. Run horizons and assets separately first

Initially:

[
30m
]

and:

[
60m
]

should be separate experiments.

Assets should also remain separate:

- AVAX
- BTC
- ETH
- HYPE
- SOL
- XRP

Only after repeated structural patterns emerge should we consider:

- joint horizons;
- shared states;
- cross-asset regimes.

---

# Part VII — How Module 12 information may re-enter later

## 24. M2a implied variance may become a transition/state predictor later

Module 12 suggests allocated implied variance contains incremental physical-volatility information for SOL/XRP and ETH 30m.

If Module 13 establishes real volatility regimes, a later question becomes:

[
P(R_{t+1}=HighVolmid widetilde W^{imp}_{t,t+h})
]

or:

[
P(R_{t+1}
eq R_tmid widetilde W^{imp}_{t,t+h}).
]

This may be a more natural use of option-market information than simply adding it linearly to the emission mean.

---

## 25. L2 / flow / D may be transition variables rather than volatility-level variables

The weak and unstable M3 result does not imply the block has no structural role.

A later model can test:

[
P(	ext{transition}mid L2_t,Flow_t,D_t)
]

rather than only:

[
E[RV_{future}mid L2_t,Flow_t,D_t].
]

This is closer to the original thesis:

> observable structural deterioration may change transition probability / staying power even when it contributes little to the current state's conditional mean.

This should only be tested after the state process itself earns its place.

---

# Part VIII — Current architecture after Modules 10–12

## 26. Updated empirical story

At this point the project evidence supports the following hierarchy of claims.

### Supported at development stage

[
oxed{
	ext{30–60m volatility has substantial forecastable structure}
}
]

[
oxed{
HAR+Clock is a serious non-switching null}
]

[
oxed{
	ext{allocated implied variance is the strongest incremental candidate}
}
]

especially SOL/XRP across both horizons.

### Not established

[
oxed{
	ext{robust five-minute directional alpha}
}
]

[
oxed{
	ext{stable L2/flow/D contribution to mean future RV}
}
]

[
oxed{
	ext{general full-surface option contribution}
}
]

[
oxed{
	ext{discrete volatility regimes}
}
]

[
oxed{
	ext{explicit duration value}
}
]

[
oxed{
	ext{MR/MOM/Bridge as validated latent states}
}
]

[
oxed{
Q_t	ext{ as a separately estimated quality process}
}
]

---

## 27. Proposed long-run latent-state hierarchy

A plausible eventual architecture — if every layer earns OOS support — is:

[
oxed{
R_t^{vol}
}
]

coarse structural volatility regime,

then conditionally:

[
oxed{
S_t^{dir}
}
]

directional / displacement-response sub-regime,

while in parallel:

[
oxed{
Q_t^{quality}
}
]

fast clean/noisy or market-quality process.

Conceptually:

[
R_t^{vol}
ightarrow
S_t^{dir}
ightarrow
Y_t
]

with:

[
Q_t^{quality}
ightarrow
	ext{observation dispersion / reliability}.
]

This is **not yet a binding architecture**. It is the current research hypothesis generated by the evidence and by the pre-Module-13 discussion.

---

# Part IX — Proposed immediate next step

## 28. Module 13

Build a minimal volatility-switching experiment:

1. Freeze Module 12 v0.2.
2. Keep accumulating its untouched 30-session confirmation set.
3. Use only development data for Module 13 specification work.
4. Compare (K=1) HAR with (K=2) HMM-HAR.
5. Start with scale-only switching.
6. Add intercept switching only if earned.
7. Add HAR-persistence switching only if earned.
8. Run a no-regime HAR-t simulation null.
9. If HMM earns OOS value, compare HSMM-HAR vs HMM-HAR.
10. Do not reintroduce IV/L2/DSF/Q until switching itself is established.

---

# 29. Questions for Claude

1. Do you agree with treating Module 13's first latent variable as (R_t^{vol}), a reduced structural volatility regime, rather than as the final (R_t) taxonomy?
2. Do you agree that MR/MOM should be viewed as potential **child directional substates** conditional on a parent volatility regime, rather than forced into the first HMM?
3. Do you agree that (Q_t) should remain a parallel fast quality process rather than automatically becoming a child state of (R_t^{vol})?
4. If a second-stage directional HMM is later fitted, would you prefer:
   - posterior-weighted conditional estimation first, then a joint hierarchical HMM only if warranted; or
   - a joint hierarchical model immediately?
5. Do you agree that parent and child latent processes may use different anchors, and that using a different child anchor is desirable if we want a genuinely different economic dimension rather than a finer discretization of volatility?
6. For Module 13 S1–S3, would you start with Student-t emissions from the beginning, or use Gaussian S1 as a diagnostic before moving to Student-t?
7. Should K=2 be fixed for the first switching gate, with K-selection only after K=2 beats HAR?
8. What no-regime simulation null would you consider minimally sufficient before we interpret any HMM/HSMM states economically?
9. Given 009 §3.2, should weekend-aware fold construction be fixed inside Module 13 immediately, even if M1c remains secondary?
10. Do you see a better eventual decomposition than (R^{vol}ightarrow S^{dir}) with parallel (Q), given the evidence from Modules 08–12?

# 30. Decisions needed from Ivan

1. Approve the narrow Module 13 objective: test (K=2) HMM-HAR against non-switching HAR before any HSMM / DSF expansion.
2. Treat the old MR/MOM/VOL/Bridge labels as post-fit economic characterizations rather than required first-layer states.
3. Treat (Q_t) as a separate later quality process, not part of Module 13.
4. Allow future child latent models to use a different anchor from the parent state when they are intended to capture a different economic dimension.
5. Keep `12_frozen_spec_v02.json` unchanged while its confirmation sample accumulates.
