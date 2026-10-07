"""
Module 13 prototype: bar-frequency volatility HMM.

Purpose
-------
Test whether a discrete latent volatility regime adds forecasting information
beyond a non-switching bar-frequency HAR-t model, without using overlapping
30m/60m realized-volatility targets as HMM emissions.

The HMM emission is a single-bar index variance observation:

    y_t = log(r_t^2 + eps_bar)

with predictors known before / at the bar close:

    log backward RV over 5m, 30m, 1h, 4h (ending at t-1),
    deterministic New York volatility clock,
    weekend indicator.

The latent model is a standard K-state Student-t HMM. It reuses the direct HMM
implementation in sdf_hmm with omega_f=0 and an empty factor block.

Structures
----------
K1 : all mean coefficients shared, one state.
S1 : mean shared, state-specific scale only.
S2 : state-specific intercept + scale; HAR/clock shared.
S3 : state-specific intercept + HAR persistence + scale; clock/weekend shared.

The bar model produces m-step expected integrated variance by propagating the
filtered state through the HMM transition matrix and recursively updating the
HAR history with expected future bar variance. Forecasts are scored against
the same strict 30m/60m index-RV targets used by Module 12.

Interpretation caveat (discussion 013): the emission is log(r^2 + eps), whose
noise is log-chi-square-like (left-skewed) and is NOT Student-t. Scale-only
switching (S1) therefore changes the dispersion / shape of log r^2, not the
volatility level, and can be driven by flat or stale index bars (a Q-like
observation effect). Only intercept switching (S2/S3) is a volatility-level
regime; state_emission_diagnostics reports the floor share per state.

Block emission (0.2.0, discussion 013): with block = b > 1 the emission is
the non-overlapping b-bar realized variance y_t = log(sum_{i<b} r_{t-i}^2 + eps)
with HAR windows ending at t - b. Each origin phase (minute of session mod b)
forms its own chain of non-overlapping blocks; the phases are pooled for
estimation (pseudo-likelihood) and every bar origin is filtered on its own
phase chain. b = 6 gives a 30m emission whose noise (log chi^2 with ~6 df) is
far less skewed than a single log r^2, so states cannot be built from the noise
of one squared return. block = 1 reproduces 0.1.x except that HAR lags across a grid gap are now
missing instead of silently spanning the gap.

This file is development-only. It does not implement the binding HAR-GARCH-t
simulation null yet; any switching result must pass that null before states are
interpreted economically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from data_engineering.modeling import sdf_hmm
from data_engineering.modeling import vol_targets as vt
from data_engineering.modeling import vol_har as vh
from data_engineering.modeling.clock_baseline import ClockBaselineSet
from data_engineering.modeling.hsmm.engine import HSMMData


VOL_HMM_VERSION = "0.2.0"

FEATURE_NAMES = ["const", "l5m", "l30m", "l1h", "l4h", "clock", "weekend"]
SWITCH_FEATURES = {
    "K1": [],
    "S1": [],
    "S2": ["const"],
    "S3": ["const", "l5m", "l30m", "l1h", "l4h"],
}
BACK_WINDOWS = dict(vt.BACK_WINDOWS)
BACK_MIN_COVERAGE = 0.90
EPS_QUANTILE = 0.01
# 0.1.3: degenerate-state guards. log(r^2 + eps) piles up at log(eps) on flat or
# stale index bars; without a floor a state can collapse onto those identical
# values (sigma -> 0, unbounded likelihood). The floor is a fraction of the
# training emission sd; fits with a state below MIN_OCCUPANCY or at the floor
# are flagged degenerate and cannot pass the switching gate.
SIGMA_FLOOR_FRAC = 0.25
MIN_OCCUPANCY = 0.02


def _positive_quantile(x: np.ndarray, q: float = EPS_QUANTILE) -> float:
    x = np.asarray(x, float)
    v = x[np.isfinite(x) & (x > 0)]
    if len(v) == 0:
        raise ValueError("No positive values available for epsilon.")
    return float(np.quantile(v, q))


def _backward_sum_coverage(
    x: np.ndarray,
    times: pd.Series,
    k: int,
    min_fraction: float = BACK_MIN_COVERAGE,
) -> np.ndarray:
    """Backward sum ending at the current array element.

    Missing values are allowed up to the pre-registered coverage threshold and
    the observed sum is rescaled by k / n_present. The timestamp grid itself
    must remain contiguous.
    """
    x = np.asarray(x, float)
    n = len(x)
    out = np.full(n, np.nan)
    if n < k:
        return out

    j = np.arange(k - 1, n)
    i = j - (k - 1)
    t = pd.to_datetime(times, utc=True).to_numpy()
    contiguous = (t[j] - t[i]) == (j - i) * vt.BAR.to_timedelta64()

    finite = np.isfinite(x)
    csum = np.r_[0.0, np.cumsum(np.where(finite, x, 0.0))]
    ccnt = np.r_[0, np.cumsum(finite.astype(int))]
    s = csum[j + 1] - csum[i]
    cnt = ccnt[j + 1] - ccnt[i]
    need = int(np.ceil(min_fraction * k))

    valid = contiguous & (cnt >= need) & (cnt > 0)
    scaled = np.where(cnt > 0, s * (k / np.maximum(cnt, 1)), np.nan)
    out[j] = np.where(valid, scaled, np.nan)
    return out


def _phase_order(raw_index: np.ndarray, times: np.ndarray, block: int) -> np.ndarray:
    """Reorder rows so each phase chain (minute-of-day / 5 mod block) is contiguous
    and time-ordered. block = 1 leaves the order unchanged."""
    raw_index = np.asarray(raw_index, int)
    if block == 1 or len(raw_index) == 0:
        return raw_index
    t_ns = pd.DatetimeIndex(pd.to_datetime(times[raw_index], utc=True)).as_unit("ns").asi8
    phase = (t_ns // int(pd.Timedelta(vt.BAR).as_unit("ns").value)) % block
    return raw_index[np.lexsort((t_ns, phase))]


def _segments(raw_index: np.ndarray, sessions: np.ndarray, times: np.ndarray, block: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Segment selected rows at session changes, missing rows, or grid gaps.
    Consecutive rows of one chain must be exactly `block` bars apart."""
    raw_index = np.asarray(raw_index, int)
    if len(raw_index) == 0:
        return np.array([], int), np.array([], int)

    r = raw_index
    # 0.1.3: force nanoseconds. With object Timestamps pandas 3 returns
    # microsecond asi8, which never equals BAR.value (ns): every row became its
    # own segment and the HMM lost all time dependence.
    t_ns = pd.DatetimeIndex(pd.to_datetime(times, utc=True)).as_unit("ns").asi8
    bar_ns = int(pd.Timedelta(vt.BAR).as_unit("ns").value)
    breaks = np.r_[
        True,
        (np.diff(r) != block)
        | (sessions[r[1:]] != sessions[r[:-1]])
        | ((t_ns[r[1:]] - t_ns[r[:-1]]) != block * bar_ns),
    ]
    starts = np.flatnonzero(breaks)
    ends = np.r_[starts[1:], len(r)]
    return starts.astype(int), ends.astype(int)


@dataclass
class BarVolPanel:
    asset: str
    times: pd.Series
    sessions: np.ndarray
    tte_min: np.ndarray
    r2: np.ndarray
    y: np.ndarray
    features: pd.DataFrame
    clock_log: np.ndarray
    weekend: np.ndarray
    eps_bar: float
    eps_back: dict
    score_eps: dict
    fwd_rv: dict
    train_sessions: list[str]
    clock: dict
    min_back_coverage: float = BACK_MIN_COVERAGE
    block: int = 1

    def rows_for_sessions(self, sessions: Iterable[str]) -> np.ndarray:
        return np.isin(self.sessions, sorted(set(sessions)))


@dataclass
class ModelData:
    sdf: sdf_hmm.SDFHMMData
    raw_index: np.ndarray
    structure: str
    names_s: list[str] = field(default_factory=list)
    names_c: list[str] = field(default_factory=list)


@dataclass
class VolHMMFit:
    structure: str
    K: int
    spec: sdf_hmm.SDFHMMSpec
    result: sdf_hmm.SDFHMMResult
    smear: np.ndarray
    names_s: list[str]
    names_c: list[str]


def build_panel(
    g: pd.DataFrame,
    train_sessions: set[str],
    *,
    score_eps: dict | None = None,
    min_back_coverage: float = BACK_MIN_COVERAGE,
    block: int = 1,
) -> BarVolPanel:
    """Build a causal bar-frequency volatility panel.

    Clock and epsilon nuisances are fitted only from train_sessions. Backward
    HAR windows end at t-1; the emission y_t uses the current bar return.
    """
    g = g.sort_values("state_time").reset_index(drop=True)
    times = pd.to_datetime(g["state_time"], utc=True).reset_index(drop=True)
    sessions = vt.expiry_time(times).sub(pd.Timedelta(hours=8)).dt.strftime("%Y-%m-%d").to_numpy()
    train = np.isin(sessions, sorted(train_sessions))

    block = int(block)
    if block < 1:
        raise ValueError("block must be >= 1")
    r = pd.to_numeric(g["return_5m"], errors="coerce").to_numpy(float)
    r2 = r ** 2
    # Emission: b-bar realized variance ending at t (strict: all b returns, contiguous grid).
    obs = r2 if block == 1 else _backward_sum_coverage(r2, times, block, 1.0)
    eps_bar = _positive_quantile(obs[train])

    # HAR information available before y_t: windows end at t-block.
    lag_r2 = np.r_[np.full(block, np.nan), r2[:-block]]
    lag_ok = np.r_[np.zeros(block, bool), (times.iloc[block:].to_numpy() - times.iloc[:-block].to_numpy())
                   == block * vt.BAR.to_timedelta64()] if len(times) > block else np.zeros(len(times), bool)
    lag_r2 = np.where(lag_ok, lag_r2, np.nan)
    back = {
        w: _backward_sum_coverage(lag_r2, times, k, min_back_coverage)
        for w, k in BACK_WINDOWS.items()
    }
    eps_back = {w: _positive_quantile(back[w][train]) for w in BACK_WINDOWS}

    frame = g[["state_id", "state_time", "underlying", "is_weekend", "return_5m"]].reset_index(drop=True)
    cbs = ClockBaselineSet.fit(frame, pd.Series(train), specs=(vh.CLOCK_SPEC,))
    if not cbs.fitted:
        raise RuntimeError(f"{g['underlying'].iloc[0]}: volatility clock could not be fitted ({cbs.skipped}).")
    tr = cbs.transform(frame)
    cf = pd.to_numeric(tr["return_5m__clock_vol_factor"], errors="coerce").to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if block == 1:
            clock_log = np.log(cf)
        else:
            # log f scale of the block: 0.5 * log mean f^2 over the block's bars (deterministic)
            f2 = pd.Series(cf ** 2).rolling(block, min_periods=block).mean().to_numpy()
            clock_log = 0.5 * np.log(f2)

    weekend = g["is_weekend"].fillna(False).astype(bool).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        features = pd.DataFrame({
            "const": 1.0,
            "l5m": np.log(back["5m"] + eps_back["5m"]),
            "l30m": np.log(back["30m"] + eps_back["30m"]),
            "l1h": np.log(back["1h"] + eps_back["1h"]),
            "l4h": np.log(back["4h"] + eps_back["4h"]),
            "clock": clock_log,
            "weekend": weekend.astype(float),
        })
        y = np.log(obs + eps_bar)

    base = vt.build_panel(g)
    if score_eps is None:
        score_eps = {
            h: _positive_quantile(base.fwd_rv[h][train])
            for h in vt.HORIZONS
        }

    return BarVolPanel(
        asset=str(g["underlying"].iloc[0]),
        times=times,
        sessions=sessions,
        tte_min=base.tte_min,
        r2=r2,
        y=y,
        features=features,
        clock_log=clock_log,
        weekend=weekend,
        eps_bar=float(eps_bar),
        eps_back=eps_back,
        score_eps={k: float(v) for k, v in score_eps.items()},
        fwd_rv=base.fwd_rv,
        train_sessions=sorted(train_sessions),
        clock=cbs.to_dict(),
        min_back_coverage=float(min_back_coverage),
        block=block,
    )


def make_model_data(panel: BarVolPanel, sessions: set[str], structure: str) -> ModelData:
    if structure not in SWITCH_FEATURES:
        raise ValueError(f"Unknown structure {structure!r}")

    switch = SWITCH_FEATURES[structure]
    shared = [c for c in FEATURE_NAMES if c not in switch]
    all_names = list(dict.fromkeys(switch + shared))

    rows = panel.rows_for_sessions(sessions)
    ok = rows & np.isfinite(panel.y)
    if all_names:
        ok &= np.isfinite(panel.features[all_names].to_numpy(float)).all(axis=1)

    raw = _phase_order(np.flatnonzero(ok), panel.times.to_numpy(), panel.block)
    if len(raw) < 40:
        raise ValueError(f"{panel.asset} {structure}: only {len(raw)} complete bar rows.")

    times_np = panel.times.to_numpy()
    starts, ends = _segments(raw, panel.sessions, times_np, panel.block)
    if len(raw) > 100 and np.median(ends - starts) < 3:
        raise RuntimeError(f"{panel.asset} {structure}: segments have median length {np.median(ends - starts)}; "
                           "time-gap detection is broken.")
    Xs = panel.features.loc[raw, switch].to_numpy(float) if switch else np.empty((len(raw), 0))
    Xc = panel.features.loc[raw, shared].to_numpy(float) if shared else np.empty((len(raw), 0))
    price = HSMMData(
        panel.y[raw],
        Xs,
        Xc,
        starts,
        ends,
        list(switch),
        list(shared),
    )
    F = np.empty((len(raw), 0), float)
    sdf = sdf_hmm.SDFHMMData(price, F, F.copy(), np.empty((len(raw), 0), bool), [])
    return ModelData(sdf=sdf, raw_index=raw, structure=structure, names_s=list(switch), names_c=list(shared))


def _state_smear(result: sdf_hmm.SDFHMMResult, data: ModelData) -> np.ndarray:
    p = result.params
    d = data.sdf.price
    shared = d.Xc @ p.beta_c if d.Xc.shape[1] else np.zeros(d.T)
    if d.Xs.shape[1]:
        mu = shared[:, None] + d.Xs @ p.beta_s.T
    else:
        mu = np.repeat(shared[:, None], p.K, axis=1)
    e = d.y[:, None] - mu
    w = result.gamma
    num = (w * np.exp(np.clip(e, -20.0, 20.0))).sum(axis=0)
    den = np.maximum(w.sum(axis=0), 1e-12)
    return np.maximum(num / den, 1e-8)


# --------------------------------------------------
# Fast forward-backward (same algorithm and outputs as sdf_hmm._fb)
# --------------------------------------------------
# sdf_hmm._fb calls scipy.special.logsumexp at every time step; on bar-frequency
# data that makes one EM fit take minutes. _fb_fast implements the identical
# recursions with an inlined log-sum-exp (numba-compiled when available). It is
# swapped in only for the duration of vol_hmm's own fits; fb_selftest checks
# that both give the same output.

try:
    from numba import njit as _njit
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False

    def _njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]


@_njit(cache=True)
def _lse(v):
    m = v.max()
    if not np.isfinite(m):
        return m
    return m + np.log(np.exp(v - m).sum())


@_njit(cache=True)
def _fb_kernel(ly, lfw, logP, logpi, pi, P, starts, ends):
    T, K = ly.shape
    lb = ly + lfw
    la = np.empty((T, K))
    filt = np.empty((T, K))
    scores = np.empty(T)
    lbe = np.zeros((T, K))
    xi = np.zeros((K, K))
    ll = 0.0
    tmp = np.empty(K)
    tmp2 = np.empty(K)
    for s in range(len(starts)):
        aa, zz = starts[s], ends[s]
        # forward (filtered on the price emission, factor term as prior tilt)
        for t in range(aa, zz):
            if t == aa:
                prior = pi.copy()
            else:
                prior = filt[t - 1] @ P
            for k in range(K):
                tmp[k] = np.log(max(prior[k], 1e-300)) + lfw[t, k]
            c = _lse(tmp)
            for k in range(K):
                tmp[k] -= c
                tmp2[k] = tmp[k] + ly[t, k]
            scores[t] = _lse(tmp2)
            c2 = _lse(tmp2)
            for k in range(K):
                la[t, k] = tmp2[k] - c2
                filt[t, k] = np.exp(la[t, k])
        # backward
        lbe[zz - 1, :] = 0.0
        for t in range(zz - 2, aa - 1, -1):
            for i in range(K):
                for j in range(K):
                    tmp[j] = logP[i, j] + lb[t + 1, j] + lbe[t + 1, j]
                lbe[t, i] = _lse(tmp)
            c = _lse(lbe[t])
            for i in range(K):
                lbe[t, i] -= c
        # xi
        lx = np.empty((K, K))
        for t in range(aa, zz - 1):
            for i in range(K):
                for j in range(K):
                    lx[i, j] = la[t, i] + logP[i, j] + lb[t + 1, j] + lbe[t + 1, j]
            c = _lse(lx.ravel())
            for i in range(K):
                for j in range(K):
                    xi[i, j] += np.exp(lx[i, j] - c)
        # likelihood
        prev = pi.copy()
        for t in range(aa, zz):
            if t == aa:
                prior = pi.copy()
            else:
                prior = prev @ P
            for k in range(K):
                tmp[k] = np.log(max(prior[k], 1e-300)) + lb[t, k]
            sc = _lse(tmp)
            ll += sc
            for k in range(K):
                prev[k] = np.exp(tmp[k] - sc)
    lg = la + lbe
    gamma = np.empty((T, K))
    for t in range(T):
        c = _lse(lg[t])
        for k in range(K):
            gamma[t, k] = np.exp(lg[t, k] - c)
    return gamma, filt, xi, ll, scores


def _fb_fast(p, d, spec):
    ly = sdf_hmm.price_logdens(p, d)
    if spec.omega_f != 0.0 and d.m:
        lfw = spec.omega_f * sdf_hmm.factor_logdens(p, d, spec.factor_nu)
    else:
        lfw = np.zeros_like(ly)
    P = np.ascontiguousarray(p.P, dtype=float)
    pi = np.ascontiguousarray(p.pi, dtype=float)
    gamma, filt, xi, ll, scores = _fb_kernel(
        np.ascontiguousarray(ly, dtype=float), np.ascontiguousarray(lfw, dtype=float),
        np.log(np.maximum(P, 1e-300)), np.log(np.maximum(pi, 1e-300)), pi, P,
        np.asarray(d.price.starts, np.int64), np.asarray(d.price.ends, np.int64),
    )
    return gamma, filt, xi, float(ll), scores


def fb_selftest(seed: int = 0) -> float:
    """Max abs difference between _fb_fast and sdf_hmm._fb on random data."""
    rng = np.random.default_rng(seed)
    T, K = 300, 2
    y = rng.standard_normal(T)
    Xs = np.ones((T, 1))
    Xc = rng.standard_normal((T, 2))
    price = HSMMData(y, Xs, Xc, np.array([0, 120]), np.array([120, T]), ["const"], ["a", "b"])
    F = np.empty((T, 0))
    d = sdf_hmm.SDFHMMData(price, F, F.copy(), np.empty((T, 0), bool), [])
    spec = sdf_hmm.SDFHMMSpec(K=K, omega_f=0.0, max_iter=3, n_init=1, seed=seed)
    p = sdf_hmm.fit(d, spec).params
    a = sdf_hmm._fb(p, d, spec)
    b = _fb_fast(p, d, spec)
    diffs = [np.max(np.abs(np.asarray(x) - np.asarray(z))) for x, z in zip(a[:3] + (a[4],), b[:3] + (b[4],))]
    return float(max(diffs + [abs(a[3] - b[3])]))


class _FastFB:
    """Context manager: use _fb_fast inside sdf_hmm.fit."""

    def __enter__(self):
        self._orig = sdf_hmm._fb
        sdf_hmm._fb = _fb_fast
        return self

    def __exit__(self, *exc):
        sdf_hmm._fb = self._orig
        return False


def fit_model(
    panel: BarVolPanel,
    train_sessions: set[str],
    structure: str,
    *,
    K: int | None = None,
    seed: int = 0,
    n_init: int = 4,
    max_iter: int = 150,
) -> tuple[VolHMMFit, ModelData]:
    if K is None:
        K = 1 if structure == "K1" else 2
    if structure == "K1" and K != 1:
        raise ValueError("K1 structure must use K=1.")

    md = make_model_data(panel, train_sessions, structure)
    min_sigma = float(SIGMA_FLOOR_FRAC * np.std(md.sdf.price.y))
    spec = sdf_hmm.SDFHMMSpec(
        K=int(K),
        omega_f=0.0,
        max_iter=int(max_iter),
        n_init=int(n_init),
        seed=int(seed),
        ridge=1e-5,
        min_sigma=min_sigma,
        trans_prior=1e-4,
        sticky_prior=0.0,
        init_block_mean=12,
        nu_grid=(3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 30.0),
    )
    with _FastFB():
        res = sdf_hmm.fit(md.sdf, spec)
    smear = _state_smear(res, md)
    return VolHMMFit(structure, int(K), spec, res, smear, md.names_s, md.names_c), md


def filter_model(fit: VolHMMFit, panel: BarVolPanel, sessions: set[str]) -> tuple[ModelData, np.ndarray]:
    md = make_model_data(panel, sessions, fit.structure)
    _, filt, _, _, _ = _fb_fast(fit.result.params, md.sdf, fit.spec)
    return md, filt


def _feature_vector(panel: BarVolPanel, raw_idx: int, history: list[float]) -> dict[str, float] | None:
    vals = np.asarray(history, float)
    out = {"const": 1.0, "clock": float(panel.clock_log[raw_idx]), "weekend": float(panel.weekend[raw_idx])}

    for name, k in BACK_WINDOWS.items():
        if len(vals) < k:
            return None
        w = vals[-k:]
        good = np.isfinite(w)
        need = int(np.ceil(panel.min_back_coverage * k))
        if good.sum() < need:
            return None
        s = float(np.nansum(w) * k / good.sum())
        out[{"5m": "l5m", "30m": "l30m", "1h": "l1h", "4h": "l4h"}[name]] = float(np.log(s + panel.eps_back[name]))
    return out


def _state_means(fit: VolHMMFit, feat: dict[str, float]) -> np.ndarray:
    p = fit.result.params
    shared = 0.0
    if fit.names_c:
        shared = float(np.asarray([feat[n] for n in fit.names_c]) @ p.beta_c)
    if fit.names_s:
        xs = np.asarray([feat[n] for n in fit.names_s], float)
        return shared + p.beta_s @ xs
    return np.full(p.K, shared, float)


def forecast_integrated_variance(
    fit: VolHMMFit,
    panel: BarVolPanel,
    eval_data: ModelData,
    filtered: np.ndarray,
    horizon_bars: int,
) -> np.ndarray:
    """Forecast sum of future bar variances for every filtered origin.

    Future latent probabilities are propagated through P without conditioning
    on unobserved future emissions. HAR windows are recursively updated with
    expected future bar variance. (0.1.3: same recursion as 0.1.2, vectorized
    over a fixed 48-bar history buffer instead of copying the full history.)
    """
    p = fit.result.params
    out = np.full(len(eval_data.raw_index), np.nan)
    times = panel.times.to_numpy()
    bar = vt.BAR.to_timedelta64()
    b = int(panel.block)
    if horizon_bars % b:
        raise ValueError(f"horizon {horizon_bars} bars is not a multiple of block {b}")
    steps = horizon_bars // b
    max_k = max(BACK_WINDOWS.values())
    win = [(n, k, int(np.ceil(panel.min_back_coverage * k))) for n, k in BACK_WINDOWS.items()]
    col = {"5m": "l5m", "30m": "l30m", "1h": "l1h", "4h": "l4h"}
    names_s, names_c = list(fit.names_s), list(fit.names_c)
    beta_s = np.asarray(p.beta_s, float)
    beta_c = np.asarray(p.beta_c, float)
    Pm = np.asarray(p.P, float)
    smear = np.asarray(fit.smear, float)
    eps_back = {n: panel.eps_back[n] for n, _ in BACK_WINDOWS.items()}

    for q, raw_idx in enumerate(eval_data.raw_index):
        if raw_idx < max_k - 1 or raw_idx + horizon_bars >= len(panel.r2):
            continue
        fut = np.arange(raw_idx + 1, raw_idx + horizon_bars + 1)
        if np.any(panel.sessions[fut] != panel.sessions[raw_idx]):
            continue
        if np.any(np.diff(times[np.r_[raw_idx, fut]]) != bar):
            continue

        hist = np.empty(max_k + horizon_bars)
        hist[:max_k] = panel.r2[raw_idx - max_k + 1: raw_idx + 1]
        n_hist = max_k
        prob = filtered[q].copy()
        total = 0.0
        good = True

        for j in range(1, steps + 1):
            rr = raw_idx + j * b
            feat = {"const": 1.0, "clock": float(panel.clock_log[rr]), "weekend": float(panel.weekend[rr])}
            for name, k, need in win:
                w = hist[n_hist - k: n_hist]
                fin = np.isfinite(w)
                cnt = int(fin.sum())
                if cnt < need:
                    good = False
                    break
                feat[col[name]] = float(np.log(w[fin].sum() * k / cnt + eps_back[name]))
            if not good or not np.all(np.isfinite(list(feat.values()))):
                good = False
                break

            prob = prob @ Pm
            shared = float(np.asarray([feat[n] for n in names_c]) @ beta_c) if names_c else 0.0
            if names_s:
                mu = shared + beta_s @ np.asarray([feat[n] for n in names_s], float)
            else:
                mu = np.full(Pm.shape[0], shared)
            level = np.maximum(np.exp(np.clip(mu, -50.0, 20.0)) * smear - panel.eps_bar, 0.0)
            v = float(prob @ level)
            if not np.isfinite(v):
                good = False
                break
            total += v
            hist[n_hist: n_hist + b] = v / b
            n_hist += b

        if good:
            out[q] = total
    return out


def score_forecasts(
    panel: BarVolPanel,
    eval_data: ModelData,
    forecast: np.ndarray,
    horizon: str,
    model: str,
) -> pd.DataFrame:
    raw = eval_data.raw_index
    target = panel.fwd_rv[horizon][raw]
    eps = panel.score_eps[horizon]
    F = np.asarray(forecast, float) + eps
    Y = np.asarray(target, float) + eps

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = Y / F
        qlike = ratio - np.log(ratio) - 1.0
        ly = np.log(Y)
        lf = np.log(F)
        sqerr = (ly - lf) ** 2

    ok = np.isfinite(target) & np.isfinite(F) & (F > 0)
    return pd.DataFrame({
        "raw_index": raw,
        "session": panel.sessions[raw],
        "tte_min": panel.tte_min[raw],
        "horizon": horizon,
        "model": model,
        "target_rv": target,
        "forecast_rv": np.where(ok, forecast, np.nan),
        "qlike": np.where(ok, qlike, np.nan),
        "sqerr": np.where(ok, sqerr, np.nan),
    })


def parameter_table(fit: VolHMMFit) -> pd.DataFrame:
    p = fit.result.params
    rows = []
    occ = fit.result.gamma.mean(axis=0)
    degenerate = bool(fit.K > 1 and (np.any(occ < MIN_OCCUPANCY) or np.any(p.sigma <= 1.01 * fit.spec.min_sigma)))
    for k in range(p.K):
        row = {
            "state": k,
            "occupancy": float(occ[k]),
            "sigma": float(p.sigma[k]),
            "nu": float(p.nu),
            "smear": float(fit.smear[k]),
            "p_stay": float(p.P[k, k]),
            "implied_mean_duration": float(1.0 / max(1.0 - p.P[k, k], 1e-12)),
            "sigma_floor": float(fit.spec.min_sigma),
            "degenerate_fit": degenerate,
        }
        for j, n in enumerate(fit.names_s):
            row[f"b_{n}"] = float(p.beta_s[k, j])
        for j, n in enumerate(fit.names_c):
            row[f"b_shared_{n}"] = float(p.beta_c[j])
        rows.append(row)
    return pd.DataFrame(rows)


def episode_table(fit: VolHMMFit, data: ModelData) -> pd.DataFrame:
    lab = fit.result.gamma.argmax(axis=1)
    out = []
    for seg, (a, z) in enumerate(zip(data.sdf.price.starts, data.sdf.price.ends)):
        t = a
        while t < z:
            k = int(lab[t])
            e = t + 1
            while e < z and int(lab[e]) == k:
                e += 1
            out.append({
                "segment": seg,
                "state": k,
                "start": int(t),
                "end": int(e),
                "length": int(e - t),
                "censored": bool(t == a or e == z),
            })
            t = e
    return pd.DataFrame(out)


def state_emission_diagnostics(fit: VolHMMFit, data: ModelData, panel: BarVolPanel) -> pd.DataFrame:
    """Per posterior-mode state: occupancy, emission moments and the share of
    bars at the epsilon floor (r^2 <= eps_bar). A state dominated by floor bars
    is describing flat / stale index prints, not a volatility regime."""
    lab = fit.result.gamma.argmax(axis=1)
    raw = data.raw_index
    y = data.sdf.price.y
    r2 = np.exp(y) - panel.eps_bar          # emission level (bar r^2, or block RV when block > 1)
    rows = []
    for k in range(fit.K):
        m = lab == k
        if not m.any():
            rows.append({"state": k, "bars": 0})
            continue
        rows.append({
            "state": k,
            "bars": int(m.sum()),
            "share": float(m.mean()),
            "mean_y": float(np.mean(y[m])),
            "sd_y": float(np.std(y[m])),
            "skew_y": float(pd.Series(y[m]).skew()),
            "floor_share": float(np.mean(r2[m] <= panel.eps_bar)),
            "median_r2_over_eps": float(np.median(r2[m]) / panel.eps_bar),
        })
    return pd.DataFrame(rows)
