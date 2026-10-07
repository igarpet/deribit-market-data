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


VOL_HMM_VERSION = "0.1.1"

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


def _segments(raw_index: np.ndarray, sessions: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Segment selected rows at session changes, missing rows, or grid gaps."""
    raw_index = np.asarray(raw_index, int)
    if len(raw_index) == 0:
        return np.array([], int), np.array([], int)

    r = raw_index
    t_ns = pd.DatetimeIndex(times).asi8
    bar_ns = int(vt.BAR.value)
    breaks = np.r_[
        True,
        (np.diff(r) != 1)
        | (sessions[r[1:]] != sessions[r[:-1]])
        | ((t_ns[r[1:]] - t_ns[r[:-1]]) != bar_ns),
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
) -> BarVolPanel:
    """Build a causal bar-frequency volatility panel.

    Clock and epsilon nuisances are fitted only from train_sessions. Backward
    HAR windows end at t-1; the emission y_t uses the current bar return.
    """
    g = g.sort_values("state_time").reset_index(drop=True)
    times = pd.to_datetime(g["state_time"], utc=True).reset_index(drop=True)
    sessions = vt.expiry_time(times).sub(pd.Timedelta(hours=8)).dt.strftime("%Y-%m-%d").to_numpy()
    train = np.isin(sessions, sorted(train_sessions))

    r = pd.to_numeric(g["return_5m"], errors="coerce").to_numpy(float)
    r2 = r ** 2
    eps_bar = _positive_quantile(r2[train])

    # HAR information available before y_t: windows end at t-1.
    lag_r2 = np.r_[np.nan, r2[:-1]]
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
        clock_log = np.log(cf)

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
        y = np.log(r2 + eps_bar)

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

    raw = np.flatnonzero(ok)
    if len(raw) < 40:
        raise ValueError(f"{panel.asset} {structure}: only {len(raw)} complete bar rows.")

    times_np = panel.times.to_numpy()
    starts, ends = _segments(raw, panel.sessions, times_np)
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
    spec = sdf_hmm.SDFHMMSpec(
        K=int(K),
        omega_f=0.0,
        max_iter=int(max_iter),
        n_init=int(n_init),
        seed=int(seed),
        ridge=1e-5,
        trans_prior=1e-4,
        sticky_prior=0.0,
        init_block_mean=12,
        nu_grid=(3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 30.0),
    )
    res = sdf_hmm.fit(md.sdf, spec)
    smear = _state_smear(res, md)
    return VolHMMFit(structure, int(K), spec, res, smear, md.names_s, md.names_c), md


def filter_model(fit: VolHMMFit, panel: BarVolPanel, sessions: set[str]) -> tuple[ModelData, np.ndarray]:
    md = make_model_data(panel, sessions, fit.structure)
    _, filt, _, _, _ = sdf_hmm._fb(fit.result.params, md.sdf, fit.spec)
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
    expected future bar variance.
    """
    p = fit.result.params
    out = np.full(len(eval_data.raw_index), np.nan)
    times = panel.times.to_numpy()

    for q, raw_idx in enumerate(eval_data.raw_index):
        if raw_idx < 47 or raw_idx + horizon_bars >= len(panel.r2):
            continue

        # The forecast must remain inside one expiry session on a contiguous grid.
        fut = np.arange(raw_idx + 1, raw_idx + horizon_bars + 1)
        if np.any(panel.sessions[fut] != panel.sessions[raw_idx]):
            continue
        if np.any(np.diff(times[np.r_[raw_idx, fut]]) != vt.BAR.to_timedelta64()):
            continue

        history = panel.r2[: raw_idx + 1].tolist()
        prob = filtered[q].copy()
        total = 0.0
        good = True

        for j in range(1, horizon_bars + 1):
            rr = raw_idx + j
            feat = _feature_vector(panel, rr, history)
            if feat is None or not np.isfinite(list(feat.values())).all():
                good = False
                break

            prob = prob @ p.P
            mu = _state_means(fit, feat)
            state_level = np.exp(np.clip(mu, -50.0, 20.0)) * fit.smear - panel.eps_bar
            state_level = np.maximum(state_level, 0.0)
            v = float(prob @ state_level)
            if not np.isfinite(v):
                good = False
                break
            total += v
            history.append(v)

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
    for k in range(p.K):
        row = {
            "state": k,
            "occupancy": float(occ[k]),
            "sigma": float(p.sigma[k]),
            "nu": float(p.nu),
            "smear": float(fit.smear[k]),
            "p_stay": float(p.P[k, k]),
            "implied_mean_duration": float(1.0 / max(1.0 - p.P[k, k], 1e-12)),
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
