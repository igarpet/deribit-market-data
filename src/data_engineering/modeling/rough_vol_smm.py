"""
Hurst Regimes branch, notebook 03 (HR 005): volatility roughness estimated
with the HR 004 review fixes.

Fix 1  Seasonality: returns are deseasonalized by a day-of-week x time-of-day
       RMS profile (7 x 288 slots, smoothed along time of day). Lags at
       multiples of 24 h and 7 days are excluded from every fit.
Fix 2  Mean reversion gets its own parameter: discrete fOU + noise structure
       function on the hourly log-sigma series,
           m2(k) = 2 V [1 - rho_fOU(k; H, kappa)] + 2 eta^2,
       rho computed exactly for an AR(1)-filtered fGn at the hourly step.
       This treats hourly log RV as a point value of log sigma (it ignores
       integration over the hour) and is reported as an approximation.
Fix 3  Primary estimator: simulated method of moments (SMM). The model is
           log sigma_t  = fOU(H, kappa, nu) at 5-minute steps,
           r_t          = s(dow, tod) * sigma_t * eps_t,  eps ~ unit-variance t(df),
       simulated on the data's own timestamps, gaps and seasonal profile,
       and passed through exactly the data pipeline (seasonal re-estimation,
       5m -> 1h/4h/1d RV, logs, structure functions). Block integration,
       RV measurement noise, fat tails, seasonality and mean reversion are
       therefore inside the estimator rather than corrected afterwards.
       Moments (18): log m2 of log sigma at 1h lags 1,2,3,4,6,12; 4h lags
       1,2,3,5; 1d lags 1,2,3,5,10,20; log Var of daily log sigma; log
       kurtosis of 5m returns standardized by their own hour.
       Common random numbers across parameter values; R_FINAL = 10
       replicates in the objective; grid H from 0.005; sandwich standard
       errors with the moment covariance from R_COV = 30 replicates.

Not implemented: the exact GMM of Bolko, Christensen, Pakkanen & Veliyev
(2023) and the Whittle-type estimator of Fukasawa, Takabatake & Westphal.
SMM addresses the same issues by simulation and is verified by parameter
recovery (selftest, recovery_check).

Units: kappa per day, nu per day^H (fBm increment SD over one day), so the
5-minute step uses kappa/288 and nu * (1/288)^H.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import t as student_t

from data_engineering.modeling import hurst_roughness as rg


VERSION = "0.1.0"
BARS_DAY = 288
BLOCKS = {"1h": 12, "4h": 48, "1d": 288}
MOMENT_LAGS = {"1h": (1, 2, 3, 4, 6, 12), "4h": (1, 2, 3, 5), "1d": (1, 2, 3, 5, 10, 20)}
MOMENT_NAMES = ([f"log_m2_{b}_lag{l}" for b, ls in MOMENT_LAGS.items() for l in ls]
                + ["log_var_lv_1d", "log_kurt_5m"])
MIN_SHARE = 0.9
BURN_DAYS = 140                   # with the 770-day sample this keeps the fGn FFT at 2^19
H_GRID = (0.005, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.45)
KAPPA_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)          # per day
NU_MULT = (0.7, 1.0, 1.4)
DF_GRID = (3.0, 4.0, 6.0, 10.0, 30.0)
R_GRID = 3
R_FINAL = 10
R_COV = 30
SEED = 3103
H_BOUNDS = (0.002, 0.7)

# Fix 2: hourly lags, no multiple of 24 h; long lags carry the mean-reversion information
PARAM_LAGS_1H = (1, 2, 3, 4, 6, 8, 12, 36, 60, 84, 108, 156, 300, 500)
PARAM_H_GRID = tuple(np.round(np.r_[np.linspace(0.005, 0.1, 20), np.linspace(0.12, 0.5, 20)], 4))
PARAM_KAPPA_GRID_H = tuple(np.logspace(-4, -0.5, 15))    # per hour (phi = 1 - kappa > 0)
DESC_LAGS = {"1h": (1, 2, 3, 4, 6, 8, 12), "4h": (1, 2, 3, 4, 5), "1d": (1, 2, 3, 4, 5)}


# --------------------------------------------------------------------------
# Data pipeline (used identically for data and simulations)
# --------------------------------------------------------------------------

def day_grid(lp: pd.Series) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """5m log returns on whole UTC days (first bar ends 00:05, last ends 00:00)
    and the dow*288+tod slot of each bar (from the bar start)."""
    r = lp.diff()
    idx = r.index
    starts = idx[(idx.hour == 0) & (idx.minute == 5)]
    ends = idx[(idx.hour == 0) & (idx.minute == 0)]
    r = r.loc[starts[0]:ends[-1]]
    n = (len(r) // BARS_DAY) * BARS_DAY
    r = r.iloc[:n]
    t0 = r.index - pd.Timedelta(minutes=5)
    slot = np.asarray(t0.dayofweek * BARS_DAY + (t0.hour * 60 + t0.minute) // 5, int)
    return r.to_numpy(float), slot, r.index


def dow_tod_profile(r: np.ndarray, slot: np.ndarray) -> np.ndarray:
    """RMS by (day of week, 5m slot), smoothed over 7 slots, mean square 1."""
    sq = np.where(np.isfinite(r), r * r, np.nan)
    ms = pd.Series(sq).groupby(slot).mean().reindex(range(7 * BARS_DAY)).to_numpy()
    ms = np.where(np.isfinite(ms) & (ms > 0), ms, np.nanmean(ms))
    M = ms.reshape(7, BARS_DAY)
    ext = np.concatenate([M[:, -3:], M, M[:, :3]], axis=1)
    k = np.ones(7) / 7.0
    sm = np.stack([np.convolve(row, k, mode="valid") for row in ext]).reshape(-1)
    return np.sqrt(sm / sm.mean())


def tod_profile(r: np.ndarray, slot: np.ndarray) -> np.ndarray:
    """Time-of-day-only profile (HR 02 convention), on the same slot layout."""
    tod = slot % BARS_DAY
    sq = np.where(np.isfinite(r), r * r, np.nan)
    ms = pd.Series(sq).groupby(tod).mean().reindex(range(BARS_DAY)).to_numpy()
    ms = np.where(np.isfinite(ms) & (ms > 0), ms, np.nanmean(ms))
    ext = np.r_[ms[-3:], ms, ms[:3]]
    sm = np.convolve(ext, np.ones(7) / 7.0, mode="valid")
    return np.tile(np.sqrt(sm / sm.mean()), 7)


def log_vol_blocks(rt: np.ndarray, b: int) -> np.ndarray:
    X = rt.reshape(-1, b)
    fin = np.isfinite(X)
    cnt = fin.sum(1)
    s = np.where(fin, X * X, 0.0).sum(1)
    ok = (cnt >= MIN_SHARE * b) & (s > 0)
    lv = np.full(len(X), np.nan)
    lv[ok] = 0.5 * np.log(s[ok] * b / cnt[ok])
    return lv


def m2(x: np.ndarray, lag: int) -> float:
    d = x[lag:] - x[:-lag]
    d = d[np.isfinite(d)]
    return float(np.mean(d * d)) if len(d) > 20 else np.nan


def log_vol_all(r: np.ndarray, slot: np.ndarray, prof: np.ndarray | None = None) -> dict:
    if prof is None:
        prof = dow_tod_profile(r, slot)
    rt = r / prof[slot]
    return {"rt": rt, **{b: log_vol_blocks(rt, n) for b, n in BLOCKS.items()}}


def moments(r: np.ndarray, slot: np.ndarray, prof: np.ndarray | None = None) -> np.ndarray:
    lv = log_vol_all(r, slot, prof)
    out = [np.log(m2(lv[b], l)) for b in BLOCKS for l in MOMENT_LAGS[b]]
    out.append(np.log(np.nanvar(lv["1d"], ddof=1)))
    scale = np.repeat(np.exp(lv["1h"]) / np.sqrt(BLOCKS["1h"]), BLOCKS["1h"])
    z = lv["rt"] / scale
    z = z[np.isfinite(z)]
    out.append(np.log(np.mean(z ** 4) / np.mean(z ** 2) ** 2))
    return np.asarray(out, float)


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------

_LAM_CACHE: dict = {}


def fgn_fast(H: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Exact fGn by circulant embedding (Davies-Harte) on a power-of-two length;
    the eigenvalues are cached per (H, length). Draws depend only on the length."""
    m = 1 << int(np.ceil(np.log2(2 * n)))
    key = (round(float(H), 8), m)
    root = _LAM_CACHE.get(key)
    if root is None:
        half = m // 2
        k = np.arange(half + 1, dtype=float)
        g = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * k ** (2 * H) + np.abs(k - 1) ** (2 * H))
        c = np.r_[g, g[-2:0:-1]]
        lam = np.clip(np.fft.fft(c).real, 0, None)
        root = np.sqrt(lam / m)
        if len(_LAM_CACHE) > 64:
            _LAM_CACHE.clear()
        _LAM_CACHE[key] = root
    z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    return np.fft.fft(root * z).real[:n]


def t_innovations(u: np.ndarray, df: float) -> np.ndarray:
    """Unit-variance Student-t draws by inversion of fixed uniforms (smooth in df
    for common random numbers)."""
    return student_t.ppf(u, df) * np.sqrt((df - 2.0) / df)


def simulate(theta: Sequence[float], slot: np.ndarray, mask: np.ndarray, prof: np.ndarray,
             seed: int, burn_days: int = BURN_DAYS) -> np.ndarray:
    """5m returns under (H, kappa/day, nu/day^H, df) on the data's slots; NaN where mask.
    Random numbers depend only on (seed, length): common random numbers across theta."""
    H, kappa, nu, df = (float(v) for v in theta)
    n = len(slot)
    B = int(burn_days * BARS_DAY)
    rng = np.random.default_rng(seed)
    e = fgn_fast(H, n + B, rng)
    k5 = kappa / BARS_DAY
    y = lfilter([nu * (1.0 / BARS_DAY) ** H], [1.0, -(1.0 - k5)], e)[B:]
    eps = t_innovations(np.clip(rng.random(n), 1e-12, 1 - 1e-12), df)
    r = prof[slot] * np.exp(y) * eps
    r[mask] = np.nan
    return r


def mean_moments(theta, slot, mask, prof, R: int, seed0: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    M = np.stack([moments(simulate(theta, slot, mask, prof, seed0 + i), slot) for i in range(R)])
    return M.mean(0), M


# --------------------------------------------------------------------------
# SMM
# --------------------------------------------------------------------------

def _theta(p: np.ndarray) -> tuple:
    return (float(p[0]), float(np.exp(p[1])), float(np.exp(p[2])), 2.0 + float(np.exp(p[3])))


def _p(theta) -> np.ndarray:
    H, k, nu, df = theta
    return np.array([H, np.log(k), np.log(nu), np.log(df - 2.0)])


def nu_anchor(m_data: np.ndarray, H: float) -> float:
    """nu such that a pure fBm would match the 4h lag-1 structure function."""
    i = MOMENT_NAMES.index("log_m2_4h_lag1")
    return float(np.sqrt(np.exp(m_data[i])) / (4.0 / 24.0) ** H)


def fit(r: np.ndarray, slot: np.ndarray, *, h_grid=H_GRID, kappa_grid=KAPPA_GRID, r_grid: int = R_GRID,
        r_final: int = R_FINAL, r_cov: int = R_COV, maxfev: int = 240, verbose: bool = False) -> dict:
    mask = ~np.isfinite(r)
    prof = dow_tod_profile(r, slot)
    m_hat = moments(r, slot, prof)
    k_i = MOMENT_NAMES.index("log_kurt_5m")
    # Stage 0: df at a reference roughness (kurtosis moment only)
    ref = (0.08, 0.03, nu_anchor(m_hat, 0.08))
    dfs = [(abs(mean_moments((*ref, df), slot, mask, prof, 2)[0][k_i] - m_hat[k_i]), df) for df in DF_GRID]
    df0 = min(dfs)[1]
    # Stage 1: diagonal weights from the moment sampling variance at the reference point
    _, Mref = mean_moments((*ref, df0), slot, mask, prof, 8, seed0=SEED + 500)
    w = 1.0 / np.maximum(Mref.var(0, ddof=1), 1e-8)
    obj_calls = {"n": 0}

    def J(theta, R):
        obj_calls["n"] += 1
        mb, _ = mean_moments(theta, slot, mask, prof, R)
        d = m_hat - mb
        return float(np.sum(w * d * d)) if np.all(np.isfinite(d)) else 1e12

    # Stage 2: grid over (H, kappa), nu over multiples of its anchor
    grid = []
    for H in h_grid:
        for k in kappa_grid:
            for mlt in NU_MULT:
                th = (H, k, nu_anchor(m_hat, H) * mlt, df0)
                grid.append((J(th, r_grid), th))
    grid.sort(key=lambda g: g[0])
    start = grid[0][1]
    # second start: best grid point with a different (H, kappa) cell, to cross the (H, kappa) ridge
    start2 = next((g[1] for g in grid[1:] if (g[1][0], g[1][1]) != (start[0], start[1])), start)
    if verbose:
        print("grid best", start, grid[0][0], "second", start2)

    # Stage 3: Nelder-Mead in (H, log kappa, log nu, log(df-2)), CRN, R_FINAL replicates
    def Jp(p):
        if not (H_BOUNDS[0] <= p[0] <= H_BOUNDS[1]) or not (-9 < p[1] < 3) or not (-0.5 < p[3] < 4.5):
            return 1e12
        return J(_theta(p), r_final)

    def run_nm(th0, fev):
        p0 = _p(th0)
        init = np.array([p0, p0 + [max(0.5 * th0[0], 0.01), 0, 0, 0], p0 + [0, 0.7, 0, 0],
                         p0 + [0, 0, 0.25, 0], p0 + [0, 0, 0, 0.5]])
        return minimize(Jp, p0, method="Nelder-Mead",
                        options={"initial_simplex": init, "maxfev": fev, "xatol": 2e-3, "fatol": 1e-2})

    first = [run_nm(start, maxfev // 2)]
    if start2 is not start:
        first.append(run_nm(start2, maxfev // 2))
    best = min(first, key=lambda o: o.fun)
    res = run_nm(_theta(best.x), maxfev // 2)                  # polish from the better start
    if res.fun > best.fun:
        res = best
    res.nfev = sum(o.nfev for o in first) + res.nfev
    p_hat = res.x
    theta_hat = _theta(p_hat)

    # Stage 4: inference (sandwich), moment fit
    m_bar, _ = mean_moments(theta_hat, slot, mask, prof, r_final)
    _, Mcov = mean_moments(theta_hat, slot, mask, prof, r_cov, seed0=SEED + 10_000)
    Omega = np.cov(Mcov, rowvar=False)
    steps = np.array([min(0.01, 0.4 * p_hat[0]), 0.15, 0.05, 0.15])
    G = np.empty((len(m_hat), 4))
    for j in range(4):
        up, dn = p_hat.copy(), p_hat.copy()
        up[j] += steps[j]
        dn[j] -= steps[j]
        G[:, j] = (mean_moments(_theta(up), slot, mask, prof, r_final)[0]
                   - mean_moments(_theta(dn), slot, mask, prof, r_final)[0]) / (2 * steps[j])
    W = np.diag(w)
    A = np.linalg.pinv(G.T @ W @ G)
    V = (1.0 + 1.0 / r_final) * A @ G.T @ W @ Omega @ W @ G @ A
    se_p = np.sqrt(np.clip(np.diag(V), 0, None))
    H, k, nu, df = theta_hat
    se = {"H": se_p[0], "kappa": k * se_p[1], "nu": nu * se_p[2], "df": (df - 2.0) * se_p[3]}
    tstat = (m_hat - m_bar) / np.sqrt(np.diag(Omega) * (1.0 + 1.0 / r_final))
    lo_k, hi_k = k * np.exp(-1.96 * se_p[1]), k * np.exp(1.96 * se_p[1])
    out = {"H": H, "H_se": se["H"], "H_ci_low": max(H - 1.96 * se["H"], 0.0), "H_ci_high": H + 1.96 * se["H"],
           "kappa_per_day": k, "kappa_ci_low": lo_k, "kappa_ci_high": hi_k,
           "half_life_days": float(np.log(2) / k), "half_life_ci_low": float(np.log(2) / hi_k),
           "half_life_ci_high": float(np.log(2) / lo_k),
           "nu": nu, "nu_se": se["nu"], "df": df, "df_se": se["df"],
           "stat_var_log_sigma": float(nu ** 2 * _gamma(2 * H + 1) / (2 * k ** (2 * H))),
           "objective": float(res.fun), "nm_evals": int(res.nfev), "nm_converged": bool(res.success),
           "grid_best_H": start[0], "grid_best_kappa": start[1], "df_stage0": df0,
           "max_abs_moment_t": float(np.nanmax(np.abs(tstat))),
           "kappa_identified": bool(np.isfinite(se_p[1]) and hi_k / lo_k < 100.0)}
    fit_table = pd.DataFrame({"moment": MOMENT_NAMES, "data": m_hat, "model": m_bar, "t": tstat,
                              "weight": w})
    return {"summary": out, "moments": fit_table, "theta": theta_hat, "prof": prof, "G": G, "Omega": Omega,
            "w": w, "m_hat": m_hat, "mask": mask}


def objective(theta, fit_out: dict, slot: np.ndarray, R: int = R_FINAL) -> float:
    """The fit's objective (same weights, seeds and replicates) at any theta."""
    mb, _ = mean_moments(theta, slot, fit_out["mask"], fit_out["prof"], R)
    d = fit_out["m_hat"] - mb
    return float(np.sum(fit_out["w"] * d * d))


def _gamma(x: float) -> float:
    from math import gamma
    return gamma(x)


def recovery_check(theta0, slot, mask, prof, seed: int = 99_001, **kw) -> dict:
    """Simulate a data set at theta0 on the given layout (independent seed) and refit."""
    r = simulate(theta0, slot, mask, prof, seed)
    f = fit(r, slot, **kw)["summary"]
    return {"H_true": theta0[0], "kappa_true": theta0[1], "nu_true": theta0[2], "df_true": theta0[3],
            **{k: f[k] for k in ("H", "H_ci_low", "H_ci_high", "kappa_per_day", "kappa_ci_low", "kappa_ci_high",
                                 "nu", "df", "max_abs_moment_t")}}


# --------------------------------------------------------------------------
# Fix 2: parametric fOU + noise structure function on hourly log sigma
# --------------------------------------------------------------------------

def fou_one_minus_rho(H: float, kappa_step: float, lags: Sequence[int], nmax: int = 1_000_000) -> np.ndarray:
    """1 - rho(k) of y_t = (1-kappa) y_{t-1} + e_t, e unit fGn(H), at the given lags (exact up to truncation)."""
    phi = 1.0 - kappa_step
    N = int(min(nmax, np.ceil(60.0 / kappa_step) + max(lags) + 10))
    m = np.arange(-N, N + 1, dtype=float)
    am = np.abs(m)
    ge = 0.5 * (np.abs(m + 1) ** (2 * H) - 2 * am ** (2 * H) + np.abs(m - 1) ** (2 * H))
    lphi = np.log(phi)

    def g(k):
        return float(np.sum(ge * np.exp(lphi * np.abs(k - m))))

    g0 = g(0)
    return np.array([1.0 - g(k) / g0 for k in lags])


def param_fit_1h(lv1h: np.ndarray, lags: Sequence[int] = PARAM_LAGS_1H, h_grid=PARAM_H_GRID,
                 kappa_grid_h=PARAM_KAPPA_GRID_H) -> dict:
    """Fit m2(k) = A (1 - rho(k; H, kappa)) + c, A = 2V, c = 2 eta^2 >= 0, relative weights."""
    y = np.array([m2(lv1h, l) for l in lags])
    ok = np.isfinite(y) & (y > 0)
    lags_ok = np.asarray(lags)[ok]
    y = y[ok]
    wgt = 1.0 / y
    best = None
    for H in h_grid:
        for k in kappa_grid_h:
            x = fou_one_minus_rho(H, k, lags_ok)
            if not np.all(np.isfinite(x)):
                continue
            X = np.column_stack([x, np.ones_like(x)]) * wgt[:, None]
            coef = np.linalg.lstsq(X, y * wgt, rcond=None)[0]
            if coef[1] < 0:
                coef = np.array([float(np.sum(X[:, 0] * y * wgt) / np.sum(X[:, 0] ** 2)), 0.0])
            if coef[0] <= 0:
                continue
            rss = float(np.sum((y * wgt - X @ coef) ** 2))
            if best is None or rss < best[0]:
                best = (rss, H, k, coef)
    rss, H, k, (A, c) = best
    return {"H": float(H), "kappa_per_hour": float(k), "kappa_per_day": float(24 * k),
            "half_life_days": float(np.log(2) / (24 * k)), "stat_var_V": float(A / 2), "noise_eta2": float(c / 2),
            "noise_share_lag1": float(c / (A * fou_one_minus_rho(H, k, [1])[0] + c)), "rss": rss,
            "at_grid_edge": bool(H in (h_grid[0], h_grid[-1]) or k in (kappa_grid_h[0], kappa_grid_h[-1]))}


# --------------------------------------------------------------------------
# Self test (fast) ; slow recovery lives in tests/ and the notebook
# --------------------------------------------------------------------------

def _layout(days: int, start: str = "2025-01-06 00:05") -> tuple[np.ndarray, pd.DatetimeIndex]:
    t = pd.date_range(start, periods=days * BARS_DAY, freq="5min", tz="UTC")
    t0 = t - pd.Timedelta(minutes=5)
    return np.asarray(t0.dayofweek * BARS_DAY + (t0.hour * 60 + t0.minute) // 5, int), t


def selftest() -> dict:
    slot, t = _layout(28)
    # Profile recovers a dow x tod pattern
    true = (1.0 + 0.5 * np.sin(np.arange(7 * BARS_DAY) / BARS_DAY * 2 * np.pi)) * np.repeat([1, 1, 1, 1, 1, 0.6, 0.6], BARS_DAY)
    true = true / np.sqrt(np.mean(true ** 2))
    r = true[slot] * np.random.default_rng(1).standard_normal(len(slot)) * 0.001
    big = np.tile(slot, 40)
    rb = true[big] * np.random.default_rng(2).standard_normal(len(big)) * 0.001
    prof = dow_tod_profile(rb, big)
    assert np.corrcoef(prof, true)[0, 1] > 0.97
    # Simulator: CRN (same seed, different H -> correlated paths), t scaling, mask
    mask = np.zeros(len(slot), bool)
    mask[100:120] = True
    a = simulate((0.1, 0.05, 0.5, 5.0), slot, mask, true, 7)
    b = simulate((0.12, 0.05, 0.5, 5.0), slot, mask, true, 7)
    assert np.isnan(a[100:120]).all() and np.corrcoef(np.abs(a[~mask]), np.abs(b[~mask]))[0, 1] > 0.9
    # Moments are finite
    m = moments(a, slot)
    assert m.shape == (len(MOMENT_NAMES),) and np.all(np.isfinite(m[:-8]))
    # Fix 2: H = 0.5 reduces to AR(1): 1 - rho(k) = 1 - phi^k
    x = fou_one_minus_rho(0.5, 0.1, [1, 2, 5])
    assert np.allclose(x, 1 - 0.9 ** np.array([1, 2, 5]), atol=1e-6), x
    return {"version": VERSION, "profile_corr": round(float(np.corrcoef(prof, true)[0, 1]), 4), "ok": True}
