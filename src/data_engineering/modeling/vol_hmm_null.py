"""
Module 14: continuous-volatility no-regime null for the Module 13 BTC S2 candidate.

The purpose is deliberately narrow. Module 13 found one primary candidate:
BTC, block-6 emission, S2 (state-specific intercept + scale), scored at 30m.
This module asks whether a continuous non-switching volatility process can
produce an equally strong apparent HMM advantage.

Null DGP
--------
A five-minute return process with no discrete state:

    log b_t = X_t beta                          (HAR + deterministic clock)
    q_t = omega + alpha u_{t-1}^2 + beta_g q_{t-1}
    r_t = sqrt(b_t * q_t) z_t
    z_t ~ standardized Student-t_nu

X_t uses the same causal lagged realized-variance windows as Module 13. The
baseline b_t is estimated from log(r_t^2 + eps) with an empirical smearing
correction; standardized returns are then fitted with a continuous GARCH(1,1)
Student-t residual process. The unconditional GARCH variance is normalized to
one (omega = 1 - alpha - beta_g).

Simulation preserves the real timestamp grid, missing-return mask, exact-zero
return mask (stale/flat observations), session/calendar pattern, and uses a
fixed real-data deterministic clock. Each simulated path then reruns the exact
BTC Module 13 candidate gate:

    S2@b6 vs STATIC_K1@b6
    S2@b6 vs ADAPT_K1@b6
    S2@b6 vs M12_M1
    S2@b6 vs M12_M1_ADAPT

in both all-origin and non-overlapping views.

Primary null statistic
-----------------------
The gate requires all eight mean-session QLIKE gains to be positive. Therefore
the scalar primary statistic is

    min_gain = min(the eight required gains).

This is stronger and less cherry-pickable than testing the largest gain.

This is a candidate-level parametric bootstrap, not yet a family-wise null over
all six assets and all Module 13 candidates. If BTC survives this null, a
selection-adjusted robustness test should be considered before HSMM work.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from data_engineering.modeling import vol_hmm as vhm
from data_engineering.modeling import vol_har as vh
from data_engineering.modeling import vol_targets as vt
from data_engineering.modeling.hsmm.session_eval import paired_sessions


VOL_HMM_NULL_VERSION = "0.1.0"
NU_GRID = (4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 30.0)
GARCH_CAP = 0.995
ZERO_TOL = 1e-15


@dataclass
class ContinuousVolNullFit:
    beta: list[float]
    feature_names: list[str]
    eps_bar: float
    eps_back: dict
    smear: float
    base_scale: float
    garch_alpha: float
    garch_beta: float
    garch_omega: float
    nu: float
    median_base_var: float
    train_eta_lo: float
    train_eta_hi: float
    clock_log: list[float]
    weekend: list[float]
    fit_sessions: list[str]
    n_fit: int

    def to_dict(self) -> dict:
        return asdict(self)


def _std_t_logpdf(z: np.ndarray, nu: float) -> np.ndarray:
    """Student-t log density standardized to variance one (nu > 2)."""
    z = np.asarray(z, float)
    s = np.sqrt((nu - 2.0) / nu)
    x = z / s
    c = (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * np.log(nu * np.pi)
        - np.log(s)
    )
    return c - 0.5 * (nu + 1.0) * np.log1p((x * x) / nu)


def _garch_nll(
    u: np.ndarray,
    times: pd.Series,
    alpha: float,
    beta: float,
    nu: float,
) -> float:
    if alpha < 0 or beta < 0 or alpha + beta >= GARCH_CAP:
        return 1e100

    omega = 1.0 - alpha - beta
    bar_ns = int(pd.Timedelta(vt.BAR).as_unit("ns").value)
    t_ns = pd.DatetimeIndex(pd.to_datetime(times, utc=True)).as_unit("ns").asi8

    ll = 0.0
    q = 1.0
    prev_u = np.nan
    prev_t = None
    n = 0
    for i, x in enumerate(np.asarray(u, float)):
        contiguous = prev_t is not None and (t_ns[i] - prev_t == bar_ns)
        if not contiguous:
            q = 1.0
            prev_u = np.nan
        elif np.isfinite(prev_u):
            q = omega + alpha * prev_u * prev_u + beta * q
        else:
            q = omega + beta * q

        q = float(np.clip(q, 1e-6, 1e6))
        if np.isfinite(x):
            z = x / np.sqrt(q)
            ll += float(_std_t_logpdf(np.array([z]), nu)[0] - 0.5 * np.log(q))
            prev_u = x
            n += 1
        else:
            prev_u = np.nan
        prev_t = t_ns[i]

    return -ll if n >= 50 else 1e100


def _fit_garch_t(u: np.ndarray, times: pd.Series) -> tuple[float, float, float, float]:
    best = (np.inf, 0.05, 0.90, 8.0)
    starts = ((0.03, 0.90), (0.08, 0.85), (0.15, 0.70), (0.03, 0.70))
    for nu in NU_GRID:
        for x0 in starts:
            res = minimize(
                lambda x: _garch_nll(u, times, float(x[0]), float(x[1]), float(nu)),
                np.asarray(x0, float),
                method="SLSQP",
                bounds=((1e-6, 0.50), (1e-6, 0.994)),
                constraints=({"type": "ineq", "fun": lambda x: GARCH_CAP - x[0] - x[1]},),
                options={"maxiter": 300, "ftol": 1e-9, "disp": False},
            )
            if res.success and np.isfinite(res.fun) and res.fun < best[0]:
                best = (float(res.fun), float(res.x[0]), float(res.x[1]), float(nu))

    if not np.isfinite(best[0]):
        raise RuntimeError("Continuous-vol null: GARCH-t fit failed.")
    _, alpha, beta, nu = best
    return alpha, beta, 1.0 - alpha - beta, nu


def fit_continuous_null(
    g: pd.DataFrame,
    fit_sessions: set[str],
) -> ContinuousVolNullFit:
    """Fit the continuous HAR-GARCH-t null on the supplied sessions.

    For a parametric bootstrap, fitting on the full development sample is a
    conservative default because the null is allowed to learn the continuous
    volatility dynamics that occurred during the evaluation period. The
    simulated paths still rerun the original train/evaluation split.
    """
    g = g.sort_values("state_time").reset_index(drop=True)
    panel = vhm.build_panel(g, fit_sessions, block=1)

    names = list(vhm.FEATURE_NAMES)
    rows = panel.rows_for_sessions(fit_sessions)
    X = panel.features[names].to_numpy(float)
    ok = rows & np.isfinite(panel.y) & np.isfinite(X).all(axis=1)
    if ok.sum() < 200:
        raise ValueError(f"Continuous-vol null: only {ok.sum()} complete fit rows.")

    beta, *_ = np.linalg.lstsq(X[ok], panel.y[ok], rcond=None)
    eta = X @ beta
    resid = panel.y[ok] - eta[ok]
    smear = float(np.mean(np.exp(np.clip(resid, -20.0, 20.0))))

    base = np.exp(np.clip(eta, -60.0, 20.0)) * smear - panel.eps_bar
    positive = base[ok & np.isfinite(base) & (base > 0)]
    if len(positive) < 100:
        raise RuntimeError("Continuous-vol null: baseline variance fit is invalid.")
    tiny = max(float(np.quantile(positive, 0.001)) * 0.1, 1e-16)
    base = np.maximum(base, tiny)

    r = pd.to_numeric(g["return_5m"], errors="coerce").to_numpy(float)
    u = np.where(np.isfinite(r) & np.isfinite(base) & (base > 0), r / np.sqrt(base), np.nan)

    # Normalize the continuous GARCH component to unconditional variance one.
    fit_u = u[rows & np.isfinite(u)]
    base_scale = float(np.mean(fit_u ** 2))
    if not np.isfinite(base_scale) or base_scale <= 0:
        raise RuntimeError("Continuous-vol null: standardized-return scale is invalid.")
    u = u / np.sqrt(base_scale)

    alpha, beta_g, omega, nu = _fit_garch_t(u[rows], panel.times[rows])

    eta_fit = eta[ok]
    return ContinuousVolNullFit(
        beta=np.asarray(beta, float).tolist(),
        feature_names=names,
        eps_bar=float(panel.eps_bar),
        eps_back={k: float(v) for k, v in panel.eps_back.items()},
        smear=smear,
        base_scale=base_scale,
        garch_alpha=alpha,
        garch_beta=beta_g,
        garch_omega=omega,
        nu=nu,
        median_base_var=float(np.median(positive) * base_scale),
        train_eta_lo=float(np.quantile(eta_fit, 0.001)),
        train_eta_hi=float(np.quantile(eta_fit, 0.999)),
        clock_log=np.asarray(panel.clock_log, float).tolist(),
        weekend=np.asarray(panel.weekend, float).tolist(),
        fit_sessions=sorted(fit_sessions),
        n_fit=int(ok.sum()),
    )


def _har_feature_from_history(
    history: np.ndarray,
    eps_back: dict,
    min_fraction: float = vhm.BACK_MIN_COVERAGE,
) -> dict[str, float] | None:
    out = {}
    col = {"5m": "l5m", "30m": "l30m", "1h": "l1h", "4h": "l4h"}
    for name, k in vhm.BACK_WINDOWS.items():
        if len(history) < k:
            return None
        w = np.asarray(history[-k:], float)
        fin = np.isfinite(w)
        need = int(np.ceil(min_fraction * k))
        cnt = int(fin.sum())
        if cnt < need:
            return None
        rv = float(w[fin].sum() * k / cnt)
        out[col[name]] = float(np.log(rv + eps_back[name]))
    return out


def _std_t_draw(rng: np.random.Generator, nu: float) -> float:
    s = np.sqrt((nu - 2.0) / nu)
    return float(rng.standard_t(nu) * s)


def simulate_returns(
    g: pd.DataFrame,
    fit: ContinuousVolNullFit,
    seed: int,
    *,
    preserve_zero_mask: bool = True,
) -> np.ndarray:
    """Simulate one regime-free return path on the real timestamp grid."""
    g = g.sort_values("state_time").reset_index(drop=True)
    n = len(g)
    if len(fit.clock_log) != n:
        raise ValueError("Null fit/grid length mismatch.")

    rng = np.random.default_rng(seed)
    times = pd.to_datetime(g["state_time"], utc=True).reset_index(drop=True)
    t_ns = pd.DatetimeIndex(times).as_unit("ns").asi8
    bar_ns = int(pd.Timedelta(vt.BAR).as_unit("ns").value)

    real_r = pd.to_numeric(g["return_5m"], errors="coerce").to_numpy(float)
    finite_mask = np.isfinite(real_r)
    zero_mask = finite_mask & (np.abs(real_r) <= ZERO_TOL)

    beta = np.asarray(fit.beta, float)
    clock = np.asarray(fit.clock_log, float)
    weekend = np.asarray(fit.weekend, float)
    max_k = max(vhm.BACK_WINDOWS.values())

    # Stationary warm-up history. It is hidden; only the simulated real grid is returned.
    hist = np.empty(max_k, float)
    for j in range(max_k):
        z = _std_t_draw(rng, fit.nu)
        hist[j] = max(fit.median_base_var * z * z, 0.0)

    out = np.full(n, np.nan)
    q = 1.0
    prev_u = np.nan
    prev_t = None

    for i in range(n):
        if prev_t is not None and (t_ns[i] - prev_t != bar_ns):
            # A true timestamp gap breaks the causal HAR history and GARCH recursion.
            q = 1.0
            prev_u = np.nan
            for j in range(max_k):
                z = _std_t_draw(rng, fit.nu)
                hist[j] = max(fit.median_base_var * z * z, 0.0)

        if np.isfinite(prev_u):
            q = fit.garch_omega + fit.garch_alpha * prev_u * prev_u + fit.garch_beta * q
        else:
            q = fit.garch_omega + fit.garch_beta * q
        q = float(np.clip(q, 1e-6, 1e6))

        har = _har_feature_from_history(hist, fit.eps_back)
        if har is None:
            base_var = fit.median_base_var
        else:
            feat = {
                "const": 1.0,
                **har,
                "clock": float(clock[i]),
                "weekend": float(weekend[i]),
            }
            x = np.asarray([feat[nm] for nm in fit.feature_names], float)
            eta = float(x @ beta)
            # Only a numerical guard against explosive recursion, wider than the
            # fitted 0.1%-99.9% range by two log-variance units on each side.
            eta = float(np.clip(eta, fit.train_eta_lo - 2.0, fit.train_eta_hi + 2.0))
            base_var = (np.exp(np.clip(eta, -60.0, 20.0)) * fit.smear - fit.eps_bar) * fit.base_scale
            base_var = max(float(base_var), 1e-16)

        if not finite_mask[i]:
            r = np.nan
            prev_u = np.nan
        elif preserve_zero_mask and zero_mask[i]:
            r = 0.0
            prev_u = 0.0
        else:
            z = _std_t_draw(rng, fit.nu)
            u = np.sqrt(q) * z
            r = np.sqrt(base_var) * u
            prev_u = float(u)

        out[i] = r
        hist = np.r_[hist[1:], r * r if np.isfinite(r) else np.nan]
        prev_t = t_ns[i]

    return out


def replace_returns(g: pd.DataFrame, returns: np.ndarray) -> pd.DataFrame:
    """Copy g, replace return_5m, and reconstruct a compatible index path.

    The Module 13/M12 forecast targets use returns directly; reconstructing the
    index price simply keeps the rest of vol_targets internally coherent.
    """
    h = g.sort_values("state_time").reset_index(drop=True).copy()
    r = np.asarray(returns, float)
    if len(r) != len(h):
        raise ValueError("replace_returns: length mismatch.")
    h["return_5m"] = r

    if "futures_index_price" in h:
        real_s = pd.to_numeric(h["futures_index_price"], errors="coerce").to_numpy(float)
        times = pd.to_datetime(h["state_time"], utc=True)
        t_ns = pd.DatetimeIndex(times).as_unit("ns").asi8
        bar_ns = int(pd.Timedelta(vt.BAR).as_unit("ns").value)
        s = np.full(len(h), np.nan)
        for i in range(len(h)):
            contig = i > 0 and (t_ns[i] - t_ns[i - 1] == bar_ns)
            if contig and np.isfinite(s[i - 1]) and np.isfinite(r[i]):
                s[i] = s[i - 1] * np.exp(r[i])
            elif np.isfinite(real_s[i]):
                s[i] = real_s[i]
        h["futures_index_price"] = s
    return h


def _m12_m1_scores(
    g: pd.DataFrame,
    hmm_panel: vhm.BarVolPanel,
    train_sessions: set[str],
    eval_sessions: set[str],
    name: str,
    score_eps: dict,
    horizon: str = "30m",
) -> pd.DataFrame:
    """Exact Module 13 direct-M1 benchmark on one path."""
    P = vt.build_panel(g)
    P.back_rv = {
        w: vhm._backward_sum_coverage(P.r ** 2, P.times, k)
        for w, k in vt.BACK_WINDOWS.items()
    }
    nuis = vh.fit_nuisance(g, P, train_sessions)
    nuis.eps_target = {h: float(score_eps[h]) for h in vt.HORIZONS}
    mat = vh.materialize(P, nuis, pd.DataFrame(index=np.arange(len(g))))
    tr = np.isin(mat.sessions, sorted(train_sessions))
    raw = np.flatnonzero(np.isin(mat.sessions, sorted(eval_sessions)))
    fv = vh.fit_vol(mat, horizon, "M1", [], tr)
    level = np.exp(vh.predict_log(fv, mat, horizon)) * fv.smear
    return vhm.score_forecasts(
        hmm_panel,
        type("_Rows", (), {"raw_index": raw})(),
        level[raw] - score_eps[horizon],
        horizon,
        name,
    )


def _mean_session_gain(
    scores_a: pd.DataFrame,
    scores_b: pd.DataFrame,
    view: str,
    horizon: str = "30m",
) -> dict:
    """A minus B QLIKE gain; positive means A is better."""
    keys = ["raw_index", "session", "horizon"]
    d = scores_a.merge(scores_b, on=keys, suffixes=("_a", "_b"))
    ok = np.isfinite(d["qlike_a"]) & np.isfinite(d["qlike_b"])
    if view == "non_overlapping":
        step = vt.HORIZONS[horizon] * vt.BAR_MIN
        ok &= np.isclose(np.mod(np.round(d["tte_min_a"].to_numpy()), step), 0.0)
    d = d.loc[ok]
    if d.empty:
        return {"gain": np.nan, "sessions_won": np.nan, "n_rows": 0, "n_sessions": 0}

    diff = d["qlike_b"].to_numpy(float) - d["qlike_a"].to_numpy(float)
    by = pd.Series(diff).groupby(d["session"].to_numpy()).sum()
    return {
        "gain": float(by.mean()),
        "sessions_won": float((by > 0).mean()),
        "n_rows": int(len(d)),
        "n_sessions": int(len(by)),
    }


def evaluate_btc_s2_gate(
    g: pd.DataFrame,
    train_sessions: set[str],
    eval_sessions: set[str],
    dev_sessions: Iterable[str],
    *,
    seed_base: int = 1300,
    n_init: int = 4,
    max_iter: int = 150,
) -> tuple[dict, pd.DataFrame]:
    """Rerun the exact BTC block-6 S2 30m Module 13 gate on one return path."""
    g = g.sort_values("state_time").reset_index(drop=True)
    dev_sessions = sorted(set(dev_sessions))

    # Module 13 BTC index in ASSETS=[AVAX,BTC,...] is 1.
    ai = 1
    b = 6
    horizon = "30m"

    base_panel = vhm.build_panel(g, train_sessions, block=1)
    score_eps = dict(base_panel.score_eps)
    panel = vhm.build_panel(g, train_sessions, score_eps=score_eps, block=b)

    fit_k1, _ = vhm.fit_model(
        panel, train_sessions, "K1", K=1,
        seed=seed_base + 100 * ai + 10 * b,
        n_init=1,
        max_iter=max_iter,
    )
    fit_s2, tr_s2 = vhm.fit_model(
        panel, train_sessions, "S2", K=2,
        seed=seed_base + 100 * ai + 10 * b + 2,
        n_init=n_init,
        max_iter=max_iter,
    )

    ev_k1, filt_k1 = vhm.filter_model(fit_k1, panel, eval_sessions)
    fk1 = vhm.forecast_integrated_variance(fit_k1, panel, ev_k1, filt_k1, vt.HORIZONS[horizon])
    sc_k1 = vhm.score_forecasts(panel, ev_k1, fk1, horizon, "STATIC_K1@b6")

    ev_s2, filt_s2 = vhm.filter_model(fit_s2, panel, eval_sessions)
    fs2 = vhm.forecast_integrated_variance(fit_s2, panel, ev_s2, filt_s2, vt.HORIZONS[horizon])
    sc_s2 = vhm.score_forecasts(panel, ev_s2, fs2, horizon, "S2@b6")

    adapt_frames = []
    for si, session in enumerate(sorted(eval_sessions)):
        prior = {s for s in dev_sessions if s < session}
        p_ad = vhm.build_panel(g, prior, score_eps=score_eps, block=b)
        fit_ad, _ = vhm.fit_model(
            p_ad, prior, "K1", K=1,
            seed=seed_base + 10000 + 100 * ai + 10 * b + si,
            n_init=1,
            max_iter=max_iter,
        )
        ev_ad, filt_ad = vhm.filter_model(fit_ad, p_ad, {session})
        f_ad = vhm.forecast_integrated_variance(fit_ad, p_ad, ev_ad, filt_ad, vt.HORIZONS[horizon])
        adapt_frames.append(vhm.score_forecasts(p_ad, ev_ad, f_ad, horizon, "ADAPT_K1@b6"))
    sc_adapt = pd.concat(adapt_frames, ignore_index=True)

    sc_m12 = _m12_m1_scores(g, base_panel, train_sessions, eval_sessions, "M12_M1", score_eps, horizon)
    m12_adapt = []
    for session in sorted(eval_sessions):
        prior = {s for s in dev_sessions if s < session}
        m12_adapt.append(
            _m12_m1_scores(g, base_panel, prior, {session}, "M12_M1_ADAPT", score_eps, horizon)
        )
    sc_m12_adapt = pd.concat(m12_adapt, ignore_index=True)

    bench = {
        "STATIC_K1@b6": sc_k1,
        "ADAPT_K1@b6": sc_adapt,
        "M12_M1": sc_m12,
        "M12_M1_ADAPT": sc_m12_adapt,
    }

    rows = []
    stats = {}
    gains = []
    for name, sc in bench.items():
        for view in ("all", "non_overlapping"):
            z = _mean_session_gain(sc_s2, sc, view, horizon)
            key = f"gain__{name}__{view}"
            stats[key] = z["gain"]
            stats[f"won__{name}__{view}"] = z["sessions_won"]
            rows.append({"model": "S2@b6", "vs": name, "view": view, **z})
            gains.append(z["gain"])

    p = fit_s2.result.params
    intercept = np.asarray(p.beta_s[:, fit_s2.names_s.index("const")], float)
    occ = fit_s2.result.gamma.mean(axis=0)
    degenerate = bool(
        np.any(occ < vhm.MIN_OCCUPANCY)
        or np.any(p.sigma <= 1.01 * fit_s2.spec.min_sigma)
    )
    diag = vhm.state_emission_diagnostics(fit_s2, tr_s2, panel)

    stats.update({
        "min_gain": float(np.nanmin(gains)),
        "gate_pass": bool(np.all(np.asarray(gains, float) > 0) and not degenerate),
        "degenerate_fit": degenerate,
        "intercept_gap": float(np.ptp(intercept)),
        "min_occupancy": float(np.min(occ)),
        "max_occupancy": float(np.max(occ)),
        "sigma_ratio": float(np.max(p.sigma) / np.min(p.sigma)),
        "max_p_stay": float(np.max(np.diag(p.P))),
        "min_p_stay": float(np.min(np.diag(p.P))),
        "max_floor_share": float(pd.to_numeric(diag["floor_share"], errors="coerce").max()),
        "train_loglik_gain_x2": float(2.0 * (fit_s2.result.loglik - fit_k1.result.loglik)),
    })
    return stats, pd.DataFrame(rows)


def empirical_p(real: float, null: np.ndarray, side: str = "greater") -> float:
    x = np.asarray(null, float)
    x = x[np.isfinite(x)]
    if not len(x) or not np.isfinite(real):
        return np.nan
    if side == "greater":
        return float((1 + np.sum(x >= real)) / (1 + len(x)))
    if side == "less":
        return float((1 + np.sum(x <= real)) / (1 + len(x)))
    raise ValueError(side)
