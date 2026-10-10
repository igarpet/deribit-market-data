"""
Volatility Dynamics Module 08: jump arrivals as a point process.

Question
--------
Modules 03-07 showed that raw q99 variance events are explained by the
volatility level. Module 08 studies a different object: JUMPS, i.e. returns
that are extreme RELATIVE TO the causal local volatility. Do jumps arrive as
a Poisson process, follow the clock / volatility level, or self-excite
(Hawkes)?

Jump definition (fixed before real-data results)
------------------------------------------------
1. Deseasonalize: r~_t = r_t / s_a(slot), log s_a a 2-harmonic Fourier curve
   in time of day fitted to log|r| on TRAINING days (smooth; free hourly
   medians absorb volatility-clustering noise).
2. Local volatility: two-sided jump-robust bipower over the hour before and
   the hour after t, excluding every pair that involves bar t:
       BV_t = (pi/2) * mean |r~_s||r~_{s-1}|,  >= 75% of the 24 pairs finite.
3. z_t = |r~_t| / sqrt(BV_t);   J_t = 1{ z_t > c },  c = 4  (primary).
   Sensitivity grid c in {3, 3.5, 4, 4.5, 5}: always reported, never used
   for the decision.

Why two-sided (design fixed on synthetic data before the real run)
------------------------------------------------------------------
Latent stochastic volatility and Hawkes excitation both cluster extremes.
With a backward-only local-vol estimate, volatility surges (in particular
market-wide surges from the common crypto volatility factor) produce
clustered exceedances that the Hawkes terms absorb: on synthetic data with a
common volatility factor, Student-t(4) diffusion and NO excitation, a
backward 4h/1h estimate gave 2/8 false passes with spurious branching up to
0.26. The two-sided estimate gave 0/8 and spurious branching <= 0.08.
Consequence: Module 08 answers an EXPLANATORY question -- does a past jump
raise the probability of a jump beyond the local volatility around it? --
and is not a real-time forecasting model.

Models (discrete time, 5-minute bars, asset a, bar t)
-----------------------------------------------------
P(J_{a,t} = 1 | history) = 1 - exp(-lambda_{a,t})   (exact discretization)

  (a) homogeneous Poisson:   lambda = exp(alpha_a)
  (b) inhomogeneous Poisson: lambda = exp(alpha_a + Fourier_2(time of day) + g * log localvol_t)
  (c) Hawkes:                lambda = (b) + eta_s * E^self_{a,t} + eta_c * E^cross_{a,t}
      E^self_{a,t}  = sum_{k>=1} g(k) J_{a,t-k},
      E^cross_{a,t} = sum_{b != a} sum_{k>=1} g(k) J_{b,t-k},
      g(k) = (1 - e^{-beta}) e^{-beta (k-1)}   (sums to 1; half-life ln2/beta bars).
  Branching ratio (expected direct offspring per jump, all assets):
      rho = eta_s + eta_c * (N_assets - 1).
  beta is profiled over a fixed half-life grid; eta >= 0.

Same-bar co-jumps across assets are not excitation (k >= 1); they are
reported descriptively.

Pre-registered decision (fixed before any real-data Module 08 output)
---------------------------------------------------------------------
Primary c = 4. PRIMARY test: 5-fold blocked cross-validation by contiguous
UTC-day blocks; each block is scored by models fitted on the other days.
- Self-excitation established iff out-of-fold logLik/row (c) - (b) > 0 with
  95% day-block bootstrap lower bound > 0, AND >= 50 scored jumps.
- Clock/volatility inhomogeneity: the same rule for (b) - (a).
SECONDARY (reported, no decision): chronological 70/30 holdout, same rule
with >= 20 test jumps.
Power (synthetic, 21 days x 7 assets, true branching 0.37): about 1/3 for the
primary rule; the branching ratio itself was recovered (0.23-0.43). A FAIL at
the current sample size is therefore NOT evidence of absence; the estimate of
rho with its spread across CV folds is reported in every case.
Fewer than the minimum jumps -> INSUFFICIENT_DATA.

Secondary event set (reported, no decision): raw q99 onsets (full-sample
per-asset thresholds, Modules 03-05 definition) through the same models.

NumPy/Pandas only. Branch-internal reuse: volatility_vd_b_core (grid).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, log, pi, sqrt
from typing import Sequence

import numpy as np
import pandas as pd

from data_engineering.modeling import volatility_vd_b_core as vb


VERSION = "0.1.0"
BARS_PER_DAY = 288
PRIMARY_C = 4.0
C_GRID = (3.0, 3.5, 4.0, 4.5, 5.0)
LOCAL_HALF_WINDOW = 12          # bars each side (1h), pairs touching t excluded
HALF_LIFE_GRID_BARS = (1, 2, 3, 4, 6, 9, 12, 18, 24, 36, 48)
TRAIN_FRACTION = 0.70
MIN_TEST_JUMPS = 20
N_BOOT = 2000


# --------------------------------------------------------------------------
# Panel
# --------------------------------------------------------------------------

@dataclass
class Panel:
    assets: list[str]
    bar: np.ndarray            # (T,) global bar numbers
    R: np.ndarray              # (T, N) returns, NaN missing
    time_ns: np.ndarray        # (T,)
    slot: np.ndarray           # (T,)
    day: np.ndarray            # (T,)

    @property
    def T(self) -> int:
        return len(self.bar)

    @property
    def N(self) -> int:
        return len(self.assets)


def build_panel(grid: vb.Grid) -> Panel:
    b0 = int(grid.bar.min())
    b1 = int(grid.bar.max())
    bars = np.arange(b0, b1 + 1, dtype=np.int64)
    R = np.full((len(bars), len(grid.assets)), np.nan)
    R[grid.bar - b0, grid.asset_code] = grid.r
    return Panel(
        assets=list(grid.assets), bar=bars, R=R,
        time_ns=bars * vb.BAR_NS, slot=(bars % BARS_PER_DAY).astype(int),
        day=(bars // BARS_PER_DAY).astype(np.int64),
    )


# --------------------------------------------------------------------------
# Jump detection
# --------------------------------------------------------------------------

def clock_factors(panel: Panel, train_mask: np.ndarray) -> np.ndarray:
    """(288, N) smooth multiplicative intraday scale of |r|, training rows only.

    OLS of log|r| on a 2-harmonic Fourier basis of the UTC slot, per asset.
    A smooth curve (5 parameters per asset) instead of 24 free hourly medians:
    free medians absorb volatility-clustering noise (factors 0.68-1.39 with no
    clock effect on synthetic data) and their steps at hour boundaries inflate
    the jump statistic.
    """
    slots = np.arange(BARS_PER_DAY)
    Fs = np.column_stack([np.ones(BARS_PER_DAY), fourier(slots)])
    S = np.ones((BARS_PER_DAY, panel.N))
    F = np.column_stack([np.ones(panel.T), fourier(panel.slot)])
    for a in range(panel.N):
        x = np.abs(panel.R[:, a])
        ok = np.isfinite(x) & (x > 0) & train_mask
        if ok.sum() < 500:
            continue
        coef, *_ = np.linalg.lstsq(F[ok], np.log(x[ok]), rcond=None)
        curve = Fs[:, 1:] @ coef[1:]
        S[:, a] = np.exp(curve - curve.mean())
    return S


def _bipower_two_sided(A: np.ndarray, m: int) -> np.ndarray:
    """(pi/2) * mean of |r~_s||r~_{s-1}| over pairs in the m bars before and the
    m bars after t, excluding every pair that involves bar t itself."""
    T, N = A.shape
    prod = np.vstack([np.full((1, N), np.nan), A[1:] * A[:-1]])   # pair (s, s-1) at row s
    fin = np.isfinite(prod)
    cs = np.vstack([np.zeros((1, N)), np.cumsum(np.where(fin, prod, 0.0), axis=0)])
    cc = np.vstack([np.zeros((1, N)), np.cumsum(fin, axis=0)])
    BV = np.full((T, N), np.nan)
    t = np.arange(m + 1, T - m - 2)
    # before: pair rows t-m .. t-1 ; after: pair rows t+2 .. t+m+1
    s_ = (cs[t] - cs[t - m]) + (cs[t + m + 2] - cs[t + 2])
    c_ = (cc[t] - cc[t - m]) + (cc[t + m + 2] - cc[t + 2])
    with np.errstate(invalid="ignore", divide="ignore"):
        v = (pi / 2.0) * s_ / c_
    v[c_ < 1.5 * m] = np.nan
    BV[t] = v
    return BV


def jump_statistics(panel: Panel, S: np.ndarray, *, half_window: int = LOCAL_HALF_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """z_t = |r~_t| / sqrt(BV_t) with two-sided jump-robust local bipower (±1h,
    pairs touching t excluded). Two-sided on purpose: with a backward-only
    estimate, volatility surges (especially market-wide ones) produce clustered
    exceedances that a Hawkes model mistakes for excitation (synthetic common-
    vol + t4 null: 2/8 false passes, spurious branching up to 0.26; two-sided:
    0/8, spurious branching <= 0.08). Consequence: Module 08 is explanatory
    (does a past jump raise the jump probability beyond local volatility?),
    not a real-time forecaster."""
    A = np.abs(panel.R / S[panel.slot, :])
    BV = _bipower_two_sided(A, int(half_window))
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = A / np.sqrt(BV)
    Z[~(np.isfinite(Z) & (BV > 0))] = np.nan
    return Z, BV


def baseline_covariates(panel: Panel, S: np.ndarray, BV: np.ndarray) -> np.ndarray:
    """(T, N, 1): log local volatility 0.5 * log BV_t (two-sided, jump-robust).

    Tests whether jump frequency depends on the volatility level once jumps
    are defined relative to it.
    """
    with np.errstate(all="ignore"):
        C = (0.5 * np.log(BV))[:, :, None]
    C[~np.isfinite(C)] = np.nan
    return C


def jump_indicator(Z: np.ndarray, c: float) -> tuple[np.ndarray, np.ndarray]:
    """J (0/1 with 0 where undefined) and the valid-row mask."""
    valid = np.isfinite(Z)
    J = np.zeros(Z.shape)
    J[valid] = (Z[valid] > float(c)).astype(float)
    return J, valid


def q99_indicator(panel: Panel) -> tuple[np.ndarray, np.ndarray]:
    """Secondary event set: full-sample per-asset q99 onsets of r^2."""
    V = panel.R ** 2
    J = np.zeros(V.shape)
    valid = np.isfinite(V)
    for a in range(panel.N):
        v = V[valid[:, a], a]
        thr = np.quantile(v, 0.99)
        tail = valid[:, a] & (V[:, a] >= thr)
        prev = np.concatenate([[False], tail[:-1]])
        J[:, a] = (tail & ~prev).astype(float)
    return J, valid


# --------------------------------------------------------------------------
# Design and excitation
# --------------------------------------------------------------------------

def fourier(slot: np.ndarray, K: int = 2) -> np.ndarray:
    u = 2 * np.pi * slot / BARS_PER_DAY
    cols = []
    for k in range(1, K + 1):
        cols += [np.sin(k * u), np.cos(k * u)]
    return np.column_stack(cols)


def excitation(J: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """E_self (T,N) and E_cross (T,N) with kernel g(k) = (1-e^-b) e^{-b(k-1)}, k>=1."""
    T, N = J.shape
    d = exp(-beta)
    E = np.zeros((T, N))
    for t in range(1, T):
        E[t] = d * E[t - 1] + (1 - d) * J[t - 1]
    tot = E.sum(axis=1, keepdims=True)
    return E, tot - E


@dataclass
class Fit:
    model: str
    theta: np.ndarray
    eta_self: float
    eta_cross: float
    beta: float
    loglik: float
    n_rows: int
    n_events: int
    names: list[str] = field(default_factory=list)
    se: np.ndarray | None = None

    @property
    def half_life_bars(self) -> float:
        return log(2) / self.beta if self.beta and np.isfinite(self.beta) else np.nan


def _design(panel: Panel, covars: np.ndarray, model: str, vol_center: np.ndarray):
    """Stack rows (t, a) into long design: returns X (rows, p), names.

    covars: (T, N, k) baseline covariates; vol_center: (N, k) training medians.
    """
    T, N = panel.T, panel.N
    blocks, names = [], []
    D = np.zeros((T * N, N))
    D[np.arange(T * N), np.tile(np.arange(N), T)] = 1.0
    blocks.append(D)
    names += [f"alpha_{a}" for a in panel.assets]
    if model in ("b", "c"):
        F = np.repeat(fourier(panel.slot), N, axis=0)
        blocks.append(F)
        names += ["sin1", "cos1", "sin2", "cos2"]
        k = covars.shape[2]
        cv = (covars - vol_center[None, :, :]).reshape(T * N, k)
        blocks.append(np.nan_to_num(cv))
        names += ["gamma_log_localvol"][:k]
    return np.hstack(blocks), names


def _loglik_terms(y, lam):
    lam = np.maximum(lam, 1e-12)
    return np.where(y > 0, np.log(-np.expm1(-lam)), -lam)


def fit_model(
    panel: Panel,
    J: np.ndarray,
    valid: np.ndarray,
    rows_mask: np.ndarray,
    logvol: np.ndarray,
    vol_center: np.ndarray,
    model: str,
    *,
    beta: float | None = None,
    max_iter: int = 100,
    ridge: float = 1e-6,
) -> Fit:
    """Fisher scoring for (a)/(b)/(c) at fixed beta. rows_mask: (T,) bars used."""
    T, N = panel.T, panel.N
    X, names = _design(panel, logvol, model, vol_center)
    use = (valid & rows_mask[:, None]).reshape(-1)
    if model == "b" or model == "c":
        use &= np.isfinite(logvol).all(axis=2).reshape(-1)
    y = J.reshape(-1)[use]
    Xu = X[use]
    if model == "c":
        Es, Ec = excitation(J, beta)
        H = np.column_stack([Es.reshape(-1)[use], Ec.reshape(-1)[use]])
    else:
        H = np.zeros((len(y), 0))
    p_mu = Xu.shape[1]
    # init: homogeneous rates per asset
    theta = np.zeros(p_mu)
    a_idx = np.tile(np.arange(N), T)[use]
    for a in range(N):
        m = a_idx == a
        rate = (y[m].sum() + 0.5) / (m.sum() + 1.0)
        theta[a] = log(-log(1 - min(rate, 0.5)))
    eta = np.full(H.shape[1], 0.05)

    def lam_of(th, et):
        return np.exp(np.clip(Xu @ th, -30, 5)) + (H @ et if H.shape[1] else 0.0)

    lam = lam_of(theta, eta)
    ll = float(_loglik_terms(y, lam).sum())
    for _ in range(int(max_iter)):
        mu = np.exp(np.clip(Xu @ theta, -30, 5))
        lam = mu + (H @ eta if H.shape[1] else 0.0)
        lam = np.maximum(lam, 1e-12)
        em = np.exp(-lam)
        u = np.where(y > 0, em / (-np.expm1(-lam)), -1.0)
        w = em / (-np.expm1(-lam))
        G = np.hstack([Xu * mu[:, None], H])
        grad = G.T @ u
        info = G.T @ (G * w[:, None]) + ridge * np.eye(G.shape[1])
        try:
            step = np.linalg.solve(info, grad)
        except np.linalg.LinAlgError:
            break
        s = 1.0
        improved = False
        for _h in range(30):
            th_new = theta + s * step[:p_mu]
            et_new = np.maximum(eta + s * step[p_mu:], 0.0)
            ll_new = float(_loglik_terms(y, lam_of(th_new, et_new)).sum())
            if ll_new >= ll - 1e-10:
                improved = True
                break
            s *= 0.5
        if not improved:
            break
        delta = ll_new - ll
        theta, eta, ll = th_new, et_new, ll_new
        if abs(delta) < 1e-8 * (1 + abs(ll)):
            break
    # approximate SEs from Fisher information at the optimum
    mu = np.exp(np.clip(Xu @ theta, -30, 5))
    lam = np.maximum(mu + (H @ eta if H.shape[1] else 0.0), 1e-12)
    w = np.exp(-lam) / (-np.expm1(-lam))
    G = np.hstack([Xu * mu[:, None], H])
    try:
        cov = np.linalg.inv(G.T @ (G * w[:, None]) + ridge * np.eye(G.shape[1]))
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = None
    return Fit(
        model=model, theta=theta,
        eta_self=float(eta[0]) if len(eta) else 0.0,
        eta_cross=float(eta[1]) if len(eta) > 1 else 0.0,
        beta=float(beta) if beta is not None else np.nan,
        loglik=ll, n_rows=int(len(y)), n_events=int(y.sum()),
        names=names + (["eta_self", "eta_cross"] if model == "c" else []), se=se,
    )


def intensity(panel: Panel, J: np.ndarray, logvol: np.ndarray, vol_center: np.ndarray, fit: Fit) -> np.ndarray:
    """(T, N) lambda under a fitted model (NaN where the model's covariates are missing)."""
    X, _ = _design(panel, logvol, fit.model, vol_center)
    lam = np.exp(np.clip(X @ fit.theta, -30, 5)).reshape(panel.T, panel.N)
    if fit.model in ("b", "c"):
        lam[~np.isfinite(logvol).all(axis=2)] = np.nan
    if fit.model == "c":
        Es, Ec = excitation(J, fit.beta)
        lam = lam + fit.eta_self * Es + fit.eta_cross * Ec
    return lam


def fit_all(panel, J, valid, train_rows, logvol, vol_center, *, half_lives=HALF_LIFE_GRID_BARS):
    """Fit (a), (b) and (c) with beta profiled over the half-life grid (train LL)."""
    common = valid & np.isfinite(logvol).all(axis=2)   # identical row set for all models
    fa = fit_model(panel, J, common, train_rows, logvol, vol_center, "a")
    fb = fit_model(panel, J, common, train_rows, logvol, vol_center, "b")
    prof = []
    best = None
    for hl in half_lives:
        beta = log(2) / float(hl)
        fc = fit_model(panel, J, common, train_rows, logvol, vol_center, "c", beta=beta)
        prof.append({"half_life_bars": hl, "half_life_min": 5 * hl, "train_loglik": fc.loglik,
                     "eta_self": fc.eta_self, "eta_cross": fc.eta_cross,
                     "branching_ratio": fc.eta_self + fc.eta_cross * (panel.N - 1)})
        if best is None or fc.loglik > best.loglik:
            best = fc
    return fa, fb, best, pd.DataFrame(prof), common


def test_comparison(panel, J, common, test_rows, lams: dict, *, n_boot=N_BOOT, seed=808):
    """Held-out logLik per row for each model, and day-block bootstrap deltas."""
    m = common & test_rows[:, None]
    y = J[m]
    days = np.broadcast_to(panel.day[:, None], J.shape)[m]
    terms = {k: _loglik_terms(y, np.maximum(v[m], 1e-12)) for k, v in lams.items()}
    rows = [{"model": k, "test_rows": int(len(y)), "test_events": int(y.sum()),
             "test_loglik_per_row": float(t.mean())} for k, t in terms.items()]
    rng = np.random.default_rng(int(seed))
    ud = np.unique(days)
    idx = {d: np.flatnonzero(days == d) for d in ud}
    boot = []
    for a, b in (("c_hawkes", "b_inhomogeneous"), ("b_inhomogeneous", "a_homogeneous"), ("c_hawkes", "a_homogeneous")):
        dl = terms[a] - terms[b]
        bs = np.empty(int(n_boot))
        for i in range(int(n_boot)):
            pick = rng.choice(ud, size=len(ud), replace=True)
            ii = np.concatenate([idx[d] for d in pick])
            bs[i] = dl[ii].mean()
        boot.append({"comparison": f"{a} - {b}", "delta_loglik_per_row": float(dl.mean()),
                     "boot_q025": float(np.quantile(bs, 0.025)), "boot_q975": float(np.quantile(bs, 0.975)),
                     "n_test_days": int(len(ud))})
    return pd.DataFrame(rows), pd.DataFrame(boot)


MIN_TOTAL_JUMPS = 50


def decide(boot: pd.DataFrame, n_events: int, *, min_events: int) -> dict:
    def one(comp):
        r = boot[boot.comparison == comp].iloc[0]
        if n_events < min_events:
            st = "INSUFFICIENT_DATA"
        elif r.delta_loglik_per_row > 0 and r.boot_q025 > 0:
            st = "PASS"
        else:
            st = "FAIL"
        return {"delta_loglik_per_row": float(r.delta_loglik_per_row), "boot_q025": float(r.boot_q025),
                "boot_q975": float(r.boot_q975), "status": st}
    return {
        "rule": f"logLik/row delta > 0 with day-block bootstrap q2.5 > 0 and >= {min_events} scored jumps",
        "n_scored_jumps": int(n_events),
        "self_excitation_c_vs_b": one("c_hawkes - b_inhomogeneous"),
        "inhomogeneity_b_vs_a": one("b_inhomogeneous - a_homogeneous"),
    }


# --------------------------------------------------------------------------
# Goodness of fit: time rescaling (randomized, discrete time)
# --------------------------------------------------------------------------

def _ks_pvalue(D: float, n: int) -> float:
    if n <= 0 or not np.isfinite(D):
        return np.nan
    x = (sqrt(n) + 0.12 + 0.11 / sqrt(n)) * D
    s = 0.0
    for k in range(1, 101):
        s += 2 * (-1) ** (k - 1) * exp(-2 * k * k * x * x)
    return float(min(max(s, 0.0), 1.0))


def time_rescaling(panel, J, common, test_rows, lam, *, seed=909) -> dict:
    """Randomized compensator increments between consecutive test events; Exp(1) under the model."""
    rng = np.random.default_rng(int(seed))
    taus = []
    for a in range(panel.N):
        m = common[:, a] & test_rows
        y = J[m, a]
        l = np.maximum(lam[m, a], 1e-12)
        acc = 0.0
        started = False
        for yi, li in zip(y, l):
            if yi > 0:
                if started:
                    u = rng.random()
                    taus.append(acc - log(1 - u * (-np.expm1(-li))))
                started = True
                acc = 0.0
            elif started:
                acc += li
    taus = np.sort(np.asarray(taus))
    n = len(taus)
    if n < 5:
        return {"n_intervals": n, "ks_D": np.nan, "ks_p": np.nan}
    F = 1 - np.exp(-taus)
    i = np.arange(1, n + 1)
    D = float(max(np.max(i / n - F), np.max(F - (i - 1) / n)))
    return {"n_intervals": int(n), "ks_D": D, "ks_p": _ks_pvalue(D, n),
            "mean_tau": float(taus.mean()), "cv_tau": float(taus.std() / taus.mean())}


# --------------------------------------------------------------------------
# Descriptives
# --------------------------------------------------------------------------

def jump_descriptives(panel, J, valid, Z, c) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from math import erfc
    p_gauss = erfc(c / sqrt(2))
    per_asset = []
    for a, name in enumerate(panel.assets):
        n = int(valid[:, a].sum())
        k = int(J[valid[:, a], a].sum())
        days = panel.day[valid[:, a]]
        counts = pd.Series(J[valid[:, a], a]).groupby(days).sum()
        per_asset.append({
            "underlying": name, "valid_bars": n, "jumps": k,
            "jump_rate_per_day": k / max(n / BARS_PER_DAY, 1e-9),
            "gaussian_expected_jumps": n * p_gauss, "excess_vs_gaussian": k / max(n * p_gauss, 1e-12),
            "daily_count_var_to_mean": float(counts.var() / counts.mean()) if counts.mean() > 0 else np.nan,
        })
    hour = panel.slot // 12
    by_hour = []
    for h in range(24):
        m = (hour == h)[:, None] & valid
        by_hour.append({"hour_utc": h, "valid_bar_assets": int(m.sum()), "jumps": int(J[m].sum()),
                        "jump_rate_per_1000_bars": 1000 * J[m].sum() / max(m.sum(), 1)})
    co = (J * valid).sum(axis=1)
    cj = pd.Series(co[co > 0]).value_counts().sort_index().rename_axis("assets_jumping_same_bar").rename("n_bars").reset_index()
    return pd.DataFrame(per_asset), pd.DataFrame(by_hour), cj


# --------------------------------------------------------------------------
# One full analysis at a given event set
# --------------------------------------------------------------------------

def analyse(panel, J, valid, covars, train_rows, test_rows, *, n_boot=N_BOOT, label="jumps_c4",
            n_folds=5):
    """PRIMARY: blocked-CV out-of-fold comparison (all days). SECONDARY:
    chronological holdout. Parameters: full-sample fit (descriptive)."""
    all_days = np.ones(panel.T, bool)
    vc_all = np.array([[np.nanmedian(covars[:, a, j]) for j in range(covars.shape[2])] for a in range(panel.N)])
    fa, fb, fc, prof, common = fit_all(panel, J, valid, all_days, covars, vc_all)
    cv_scores, cv_boot, cv_folds, n_cv = blocked_cv(panel, J, valid, covars, n_folds=n_folds, n_boot=n_boot)
    primary = decide(cv_boot, n_cv, min_events=MIN_TOTAL_JUMPS)

    vc_tr = np.array([[np.nanmedian(covars[train_rows, a, j]) for j in range(covars.shape[2])] for a in range(panel.N)])
    ha, hb_, hc, _, common_tr = fit_all(panel, J, valid, train_rows, covars, vc_tr)
    lams_ho = {"a_homogeneous": intensity(panel, J, covars, vc_tr, ha),
               "b_inhomogeneous": intensity(panel, J, covars, vc_tr, hb_),
               "c_hawkes": intensity(panel, J, covars, vc_tr, hc)}
    ho_scores, ho_boot = test_comparison(panel, J, common_tr, test_rows, lams_ho, n_boot=n_boot)
    n_test = int(J[common_tr & test_rows[:, None]].sum())
    secondary = decide(ho_boot, n_test, min_events=MIN_TEST_JUMPS)

    lams_full = {"a_homogeneous": intensity(panel, J, covars, vc_all, fa),
                 "b_inhomogeneous": intensity(panel, J, covars, vc_all, fb),
                 "c_hawkes": intensity(panel, J, covars, vc_all, fc)}
    gof = pd.DataFrame([{"model": k, "fit": "full_sample_in_sample",
                         **time_rescaling(panel, J, common, all_days, v)} for k, v in lams_full.items()])
    params = []
    for f in (fa, fb, fc):
        vals = list(f.theta) + ([f.eta_self, f.eta_cross] if f.model == "c" else [])
        ses = list(f.se) if f.se is not None else [np.nan] * len(vals)
        for nm, val, se in zip(f.names, vals, ses):
            params.append({"model": f.model, "parameter": nm, "estimate": float(val), "approx_se": float(se)})
    rho = fc.eta_self + fc.eta_cross * (panel.N - 1)
    se_map = dict(zip(fc.names, ses)) if fc.se is not None else {}
    summary = {
        "event_set": label,
        "n_events_full_sample": int(fc.n_events),
        "hawkes_half_life_min_full_sample": 5 * fc.half_life_bars,
        "eta_self": fc.eta_self, "eta_self_approx_se": float(se_map.get("eta_self", np.nan)),
        "eta_cross": fc.eta_cross, "eta_cross_approx_se": float(se_map.get("eta_cross", np.nan)),
        "branching_ratio": rho,
        "branching_ratio_across_cv_folds": [round(x, 4) for x in cv_folds["branching_ratio"].tolist()],
        "gamma_log_localvol_b": float(fb.theta[fb.names.index("gamma_log_localvol")]),
        "primary_blocked_cv": primary,
        "secondary_chronological_holdout": secondary,
    }
    return {"summary": summary, "cv_scores": cv_scores, "cv_boot": cv_boot, "cv_folds": cv_folds,
            "holdout_scores": ho_scores, "holdout_boot": ho_boot, "gof": gof,
            "params": pd.DataFrame(params), "profile": prof}


def blocked_cv(panel, J, valid, covars, *, n_folds=5, n_boot=N_BOOT, seed=818,
               half_lives=HALF_LIFE_GRID_BARS):
    """Out-of-fold logLik for (a), (b), (c) with contiguous day-block folds.

    Each fold's days are scored by models fitted on all other days (beta
    re-profiled per fold). Intensities use only past jumps (the excitation is
    causal); fitting on later days is ordinary cross-validation, not a
    forecasting claim. Out-of-fold per-row terms are pooled and compared with
    a day-block bootstrap.
    """
    days = [d for d in np.unique(panel.day) if np.isfinite(panel.R[panel.day == d]).mean() >= 0.5]
    folds = np.array_split(np.asarray(days), int(n_folds))
    terms = {k: [] for k in ("a_homogeneous", "b_inhomogeneous", "c_hawkes")}
    row_days, ys, fold_rows = [], [], []
    for fi, fd in enumerate(folds):
        test = np.isin(panel.day, fd)
        train = ~test & np.isin(panel.day, days)
        vc = np.array([[np.nanmedian(covars[train, a, j]) for j in range(covars.shape[2])]
                       for a in range(panel.N)])
        fa, fb, fc, prof, common = fit_all(panel, J, valid, train, covars, vc, half_lives=half_lives)
        m = common & test[:, None]
        y = J[m]
        for k, f in (("a_homogeneous", fa), ("b_inhomogeneous", fb), ("c_hawkes", fc)):
            lam = intensity(panel, J, covars, vc, f)
            terms[k].append(_loglik_terms(y, np.maximum(lam[m], 1e-12)))
        ys.append(y)
        row_days.append(np.broadcast_to(panel.day[:, None], J.shape)[m])
        fold_rows.append({"fold": fi, "first_day": int(fd[0]), "last_day": int(fd[-1]),
                          "test_rows": int(m.sum()), "test_events": int(y.sum()),
                          "half_life_min": 5 * fc.half_life_bars, "eta_self": fc.eta_self,
                          "eta_cross": fc.eta_cross,
                          "branching_ratio": fc.eta_self + fc.eta_cross * (panel.N - 1)})
    T_ = {k: np.concatenate(v) for k, v in terms.items()}
    y = np.concatenate(ys)
    dd = np.concatenate(row_days)
    rng = np.random.default_rng(int(seed))
    ud = np.unique(dd)
    idx = {d: np.flatnonzero(dd == d) for d in ud}
    boot = []
    for a_, b_ in (("c_hawkes", "b_inhomogeneous"), ("b_inhomogeneous", "a_homogeneous")):
        dl = T_[a_] - T_[b_]
        bs = np.empty(int(n_boot))
        for i in range(int(n_boot)):
            pick = rng.choice(ud, size=len(ud), replace=True)
            bs[i] = dl[np.concatenate([idx[d] for d in pick])].mean()
        boot.append({"comparison": f"{a_} - {b_}", "delta_loglik_per_row": float(dl.mean()),
                     "boot_q025": float(np.quantile(bs, 0.025)), "boot_q975": float(np.quantile(bs, 0.975)),
                     "n_days": int(len(ud))})
    scores = pd.DataFrame([{"model": k, "oof_rows": int(len(v)), "oof_events": int(y.sum()),
                            "oof_loglik_per_row": float(v.mean())} for k, v in T_.items()])
    return scores, pd.DataFrame(boot), pd.DataFrame(fold_rows), int(y.sum())


def chronological_rows(panel: Panel, frac: float = TRAIN_FRACTION) -> tuple[np.ndarray, np.ndarray, int]:
    fin_days = [d for d in np.unique(panel.day) if np.isfinite(panel.R[panel.day == d]).mean() >= 0.5]
    n_tr = max(3, min(int(np.floor(len(fin_days) * frac)), len(fin_days) - 2))
    cut = fin_days[n_tr]
    train = panel.day < cut
    return train, ~train, int(cut)


# --------------------------------------------------------------------------
# Synthetic data and self test
# --------------------------------------------------------------------------

def simulate(n_days=20, N=4, base=0.0015, eta_s=0.0, eta_c=0.0, half_life=6, seed=0,
             common_vol=0.0, t_df=None):
    """Returns with stochastic volatility + jumps arriving per a known (Hawkes) intensity.

    common_vol > 0 adds a market-wide log-volatility factor (crypto-like);
    t_df gives Student-t diffusive innovations (heavy tails without jumps).
    """
    rng = np.random.default_rng(seed)
    T = n_days * BARS_PER_DAY
    beta = log(2) / half_life
    d = exp(-beta)
    lv = np.zeros((T, N))
    cf = np.zeros(T)
    for t in range(1, T):
        lv[t] = 0.99 * lv[t - 1] + rng.normal(scale=0.08, size=N)
        cf[t] = 0.99 * cf[t - 1] + rng.normal(scale=common_vol)
    sig = 0.002 * np.exp(lv + cf[:, None])
    eps = rng.normal(size=(T, N)) if t_df is None else rng.standard_t(t_df, size=(T, N)) / np.sqrt(t_df / (t_df - 2))
    R = sig * eps
    E = np.zeros(N)
    J = np.zeros((T, N))
    for t in range(T):
        if t > 0:
            E = d * E + (1 - d) * J[t - 1]
        lam = base + eta_s * E + eta_c * (E.sum() - E)
        J[t] = rng.random(N) < -np.expm1(-lam)
    R = R + J * sig * rng.choice([-1, 1], size=(T, N)) * rng.uniform(8, 14, size=(T, N))
    times = pd.date_range("2026-03-01", periods=T, freq="5min", tz="UTC")
    frames = [pd.DataFrame({"state_time": times, "underlying": f"A{a}", "return_5m": R[:, a]}) for a in range(N)]
    return pd.concat(frames, ignore_index=True), J


def selftest() -> dict:
    # 1. excitation recursion equals the explicit kernel sum
    J = np.zeros((40, 2)); J[3, 0] = 1; J[10, 1] = 1
    Es, Ec = excitation(J, log(2) / 3)
    b = log(2) / 3
    g = lambda k: (1 - exp(-b)) * exp(-b * (k - 1))
    assert np.isclose(Es[12, 0], g(9)) and np.isclose(Ec[12, 0], g(2))
    # 2. detection: planted jumps are found; false rate bounded (Gaussian diffusion)
    f, Jtrue = simulate(n_days=16, N=3, base=0.002, seed=1)
    g_ = vb.build_grid(f)
    p = build_panel(g_)
    tr, te, _ = chronological_rows(p)
    S = clock_factors(p, tr)
    Z, BV = jump_statistics(p, S)
    Jh, valid = jump_indicator(Z, PRIMARY_C)
    hit = (Jh * Jtrue)[valid].sum() / max(Jtrue[valid].sum(), 1)
    false = (Jh * (1 - Jtrue))[valid].sum() / max((1 - Jtrue)[valid].sum(), 1)
    assert hit > 0.9 and false < 2e-3, (hit, false)
    # 3. Size and power at realistic size (21 days x 7 assets)
    def run(**kw):
        f, _ = simulate(n_days=21, N=7, base=0.0015, half_life=6, **kw)
        p = build_panel(vb.build_grid(f))
        tr, te, _ = chronological_rows(p)
        S = clock_factors(p, tr)
        Z, BV = jump_statistics(p, S)
        Jh, valid = jump_indicator(Z, PRIMARY_C)
        return analyse(p, Jh, valid, baseline_covariates(p, S, BV), tr, te, n_boot=300)["summary"]
    hawkes = run(eta_s=0.25, eta_c=0.04, seed=8)
    null = run(seed=11, common_vol=0.1, t_df=4)
    assert hawkes["primary_blocked_cv"]["self_excitation_c_vs_b"]["status"] == "PASS", hawkes
    assert null["primary_blocked_cv"]["self_excitation_c_vs_b"]["status"] != "PASS", null
    assert 0.2 < hawkes["branching_ratio"] < 0.55, hawkes["branching_ratio"]
    assert null["branching_ratio"] < 0.12, null["branching_ratio"]
    out = {"hawkes": hawkes}
    return {"kernel_recursion": True, "jump_detection": True,
            "hawkes_detected_when_present": True, "no_false_hawkes_common_vol_t4_null": True,
            "branching_ratio_recovered": round(out["hawkes"]["branching_ratio"], 3)}
