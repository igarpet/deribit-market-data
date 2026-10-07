"""
Joint Switching Dynamic Factor HMM for framework v3.2 (Module 10).

This is deliberately the HMM stage, not the final HSMM.  For each asset and
model row whose price target is z_{t+1}, the latent structural state R_t uses
information available at t:

    price:  z_{t+1} = x_t' beta_R + l_t' rho + sigma_R * eps
    factors: f_{j,t} = c_{R,j} + a_{R,j} f_{j,t-1} + s_{R,j} * eta

The initial factor transition is diagonal because Module 09 found that most
OOS factor predictability came from own persistence, not dense cross-factor
propagation.  State-specific (c_R, a_R) are shrunk toward the non-switching
factor dynamics. Missing factors are masked equation-by-equation and are never
mean-filled. Factor transition evidence is tempered by omega_F:

    log evidence_t(k) = log p_price,t(k) + omega_F * log p_factor,t(k).

For one-step price scoring, the factor transition at t is observed first and
updates P(R_t); the score for z_{t+1} is then computed before observing that
return. Thus omega_F can only earn its place by improving chronological OOS
price prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from data_engineering.modeling.hsmm.engine import HSMMData, t_logpdf


SDF_HMM_VERSION = "0.1.1"


@dataclass
class SDFHMMData:
    price: HSMMData
    F: np.ndarray
    F_lag: np.ndarray
    M: np.ndarray
    factor_names: list[str] = field(default_factory=list)

    def __post_init__(self):
        T = self.price.T
        self.F = np.asarray(self.F, float)
        self.F_lag = np.asarray(self.F_lag, float)
        self.M = np.asarray(self.M, bool)
        assert self.F.shape == self.F_lag.shape == self.M.shape
        assert self.F.shape[0] == T
        self.F = np.where(self.M, self.F, 0.0)
        self.F_lag = np.where(self.M, self.F_lag, 0.0)
        if not self.factor_names:
            self.factor_names = [f"F{j}" for j in range(self.F.shape[1])]
        assert len(self.factor_names) == self.F.shape[1]

    @property
    def T(self) -> int:
        return self.price.T

    @property
    def m(self) -> int:
        return self.F.shape[1]

    def subset(self, rows: np.ndarray) -> "SDFHMMData":
        rows = np.asarray(rows, int)
        if len(rows) == 0:
            raise ValueError("No SDF rows selected")
        seg = np.empty(self.T, int)
        for i, (a, b) in enumerate(zip(self.price.starts, self.price.ends)):
            seg[a:b] = i
        brk = np.r_[True, (np.diff(rows) != 1) | (seg[rows][1:] != seg[rows][:-1])]
        st = np.flatnonzero(brk)
        en = np.r_[st[1:], len(rows)]
        lengths = en - st
        new_st = np.r_[0, np.cumsum(lengths)[:-1]].astype(int)
        new_en = np.cumsum(lengths).astype(int)
        p = self.price
        pp = HSMMData(
            p.y[rows], p.Xs[rows], p.Xc[rows], new_st, new_en,
            list(p.names_s), list(p.names_c),
        )
        return SDFHMMData(pp, self.F[rows], self.F_lag[rows], self.M[rows], list(self.factor_names))


@dataclass
class SDFHMMSpec:
    K: int = 2
    omega_f: float = 0.25
    max_iter: int = 150
    tol: float = 1e-5
    n_init: int = 3
    seed: int = 0
    ridge: float = 1e-5
    min_sigma: float = 1e-3
    min_factor_sd: float = 0.05
    factor_nu: float = 5.0
    factor_shrink: float = 100.0
    factor_scale_shrink: float = 50.0
    trans_prior: float = 1e-6
    sticky_prior: float = 0.0
    init_block_mean: int = 12
    nu_grid: tuple = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 30.0)


@dataclass
class SDFHMMParams:
    pi: np.ndarray
    P: np.ndarray
    beta_s: np.ndarray
    beta_c: np.ndarray
    sigma: np.ndarray
    nu: float
    factor_c: np.ndarray
    factor_a: np.ndarray
    factor_sd: np.ndarray
    common_c: np.ndarray
    common_a: np.ndarray
    common_sd: np.ndarray

    @property
    def K(self) -> int:
        return len(self.pi)

    def permute(self, order: np.ndarray) -> "SDFHMMParams":
        o = np.asarray(order)
        return SDFHMMParams(
            self.pi[o], self.P[np.ix_(o, o)], self.beta_s[o], self.beta_c.copy(),
            self.sigma[o], self.nu, self.factor_c[o], self.factor_a[o], self.factor_sd[o],
            self.common_c.copy(), self.common_a.copy(), self.common_sd.copy(),
        )


@dataclass
class SDFHMMResult:
    params: SDFHMMParams
    loglik: float
    gamma: np.ndarray
    filtered: np.ndarray
    price_scores: np.ndarray
    iterations: int
    converged: bool
    init_logliks: list[float]
    history: list[float]


def build_sdf_data(design, factor_frame: pd.DataFrame, factor_names: list[str]) -> SDFHMMData:
    """Align factor transition at predictor time t with design price target z_{t+1}."""
    price = design.data(xs=["const", "D"] + list(design.extra_names), xc=["z1", "z2", "z3"])
    src = design.frame["source_row"].to_numpy(int)
    if len(src) != price.T:
        raise ValueError("SDF alignment expects the full design without an additional row mask")
    V = factor_frame[factor_names].to_numpy(float)
    p = src - 1
    q = src - 2
    F = np.full((len(src), len(factor_names)), np.nan)
    L = np.full_like(F, np.nan)
    ok = (p >= 0) & (q >= 0)
    F[ok] = V[p[ok]]
    L[ok] = V[q[ok]]
    M = np.isfinite(F) & np.isfinite(L)
    return SDFHMMData(price, F, L, M, list(factor_names))


def rows_from_sessions(design, sessions: set[str]) -> np.ndarray:
    return design.rows_of(design.frame["session_date"].isin(sessions).to_numpy())


def _price_resid(p: SDFHMMParams, d: SDFHMMData) -> np.ndarray:
    shared = d.price.Xc @ p.beta_c if d.price.Xc.shape[1] else 0.0
    return (d.price.y - shared)[:, None] - d.price.Xs @ p.beta_s.T


def price_logdens(p: SDFHMMParams, d: SDFHMMData) -> np.ndarray:
    e = _price_resid(p, d)
    return t_logpdf(e / p.sigma[None, :], p.nu) - np.log(p.sigma)[None, :]


def factor_logdens(p: SDFHMMParams, d: SDFHMMData, factor_nu: float) -> np.ndarray:
    out = np.zeros((d.T, p.K))
    for k in range(p.K):
        mu = p.factor_c[k][None, :] + d.F_lag * p.factor_a[k][None, :]
        z = (d.F - mu) / p.factor_sd[k][None, :]
        lp = t_logpdf(z, factor_nu) - np.log(p.factor_sd[k])[None, :]
        out[:, k] = np.where(d.M, lp, 0.0).sum(axis=1)
    return out


def fit_common_factors(d: SDFHMMData, ridge: float = 1e-3):
    c = np.zeros(d.m)
    a = np.zeros(d.m)
    sd = np.ones(d.m)
    for j in range(d.m):
        ok = d.M[:, j]
        if ok.sum() < 30:
            continue
        X = np.column_stack([np.ones(ok.sum()), d.F_lag[ok, j]])
        y = d.F[ok, j]
        G = X.T @ X + ridge * np.diag([0.0, 1.0])
        b = np.linalg.solve(G, X.T @ y)
        e = y - X @ b
        c[j], a[j] = b
        a[j] = float(np.clip(a[j], -0.995, 0.995))
        sd[j] = max(float(np.std(e)), 0.05)
    return c, a, sd


def _fb(p: SDFHMMParams, d: SDFHMMData, spec: SDFHMMSpec):
    ly = price_logdens(p, d)
    lf = factor_logdens(p, d, spec.factor_nu)
    lb = ly + spec.omega_f * lf
    T, K = lb.shape
    la = np.empty((T, K))
    filt = np.empty((T, K))
    price_scores = np.empty(T)

    for aa, zz in zip(d.price.starts, d.price.ends):
        prev = None
        for t in range(aa, zz):
            prior = p.pi if prev is None else prev @ p.P
            log_prior = np.log(np.maximum(prior, 1e-300))
            lq = log_prior + spec.omega_f * lf[t]
            lq -= logsumexp(lq)
            price_scores[t] = logsumexp(lq + ly[t])
            lpost = lq + ly[t]
            la[t] = lpost - logsumexp(lpost)
            filt[t] = np.exp(la[t])
            prev = filt[t]

    lbe = np.zeros((T, K))
    for aa, zz in zip(d.price.starts, d.price.ends):
        lbe[zz - 1] = 0.0
        for t in range(zz - 2, aa - 1, -1):
            nxt = np.log(np.maximum(p.P, 1e-300)) + lb[t + 1][None, :] + lbe[t + 1][None, :]
            lbe[t] = logsumexp(nxt, axis=1)
            lbe[t] -= logsumexp(lbe[t])

    lg = la + lbe
    lg -= logsumexp(lg, axis=1, keepdims=True)
    gamma = np.exp(lg)
    xi = np.zeros((K, K))
    for aa, zz in zip(d.price.starts, d.price.ends):
        for t in range(aa, zz - 1):
            lx = (la[t][:, None] + np.log(np.maximum(p.P, 1e-300))
                  + lb[t + 1][None, :] + lbe[t + 1][None, :])
            lx -= logsumexp(lx)
            xi += np.exp(lx)

    ll = 0.0
    for aa, zz in zip(d.price.starts, d.price.ends):
        prev = None
        for t in range(aa, zz):
            prior = p.pi if prev is None else prev @ p.P
            lv = np.log(np.maximum(prior, 1e-300)) + lb[t]
            sc = logsumexp(lv)
            ll += sc
            prev = np.exp(lv - sc)
    return gamma, filt, xi, float(ll), price_scores


def _wls_prior(X, y, w, ridge, prior=None, prior_weight=0.0):
    G = X.T @ (X * w[:, None]) + ridge * np.eye(X.shape[1])
    h = (X * w[:, None]).T @ y
    if prior is not None and prior_weight > 0:
        G += prior_weight * np.eye(X.shape[1])
        h += prior_weight * prior
    return np.linalg.solve(G, h)


def _m_step(p: SDFHMMParams, d: SDFHMMData, spec: SDFHMMSpec, gamma, xi) -> SDFHMMParams:
    K = spec.K
    pi = gamma[d.price.starts].sum(axis=0) + 1e-6
    pi /= pi.sum()
    if K == 1:
        P = np.ones((1, 1))
    else:
        P = xi + spec.trans_prior + spec.sticky_prior * np.eye(K)
        P /= P.sum(axis=1, keepdims=True)

    e0 = _price_resid(p, d)
    u = (p.nu + 1.0) / (p.nu + (e0 / p.sigma[None, :]) ** 2)
    W = gamma * u
    y_c = d.price.y - (d.price.Xc @ p.beta_c if d.price.Xc.shape[1] else 0.0)
    beta_s = np.vstack([_wls_prior(d.price.Xs, y_c, W[:, k], spec.ridge) for k in range(K)])
    if d.price.Xc.shape[1]:
        Wc = (W / p.sigma[None, :] ** 2).sum(axis=1)
        yk = d.price.y[:, None] - d.price.Xs @ beta_s.T
        target = (W / p.sigma[None, :] ** 2 * yk).sum(axis=1) / np.maximum(Wc, 1e-300)
        beta_c = _wls_prior(d.price.Xc, target, Wc, spec.ridge)
    else:
        beta_c = np.zeros(0)

    tmp = SDFHMMParams(pi, P, beta_s, beta_c, p.sigma.copy(), p.nu,
                       p.factor_c.copy(), p.factor_a.copy(), p.factor_sd.copy(),
                       p.common_c, p.common_a, p.common_sd)
    ep = _price_resid(tmp, d)
    sig2 = (W * ep * ep).sum(axis=0) / np.maximum(gamma.sum(axis=0), 1e-12)
    sigma = np.maximum(np.sqrt(sig2), spec.min_sigma)
    zz = ep / sigma[None, :]
    best_nu, best = p.nu, -np.inf
    for nu in spec.nu_grid:
        val = float((gamma * t_logpdf(zz, nu)).sum())
        if val > best:
            best, best_nu = val, nu

    fc = np.zeros((K, d.m))
    fa = np.zeros((K, d.m))
    fs = np.zeros((K, d.m))
    for k in range(K):
        for j in range(d.m):
            ok = d.M[:, j]
            if ok.sum() < 30:
                fc[k, j], fa[k, j], fs[k, j] = p.common_c[j], p.common_a[j], p.common_sd[j]
                continue
            mu0 = p.factor_c[k, j] + p.factor_a[k, j] * d.F_lag[ok, j]
            z0 = (d.F[ok, j] - mu0) / max(p.factor_sd[k, j], spec.min_factor_sd)
            tu = (spec.factor_nu + 1.0) / (spec.factor_nu + z0 * z0)
            w = gamma[ok, k] * tu
            X = np.column_stack([np.ones(ok.sum()), d.F_lag[ok, j]])
            y = d.F[ok, j]
            prior = np.array([p.common_c[j], p.common_a[j]])
            b = _wls_prior(X, y, w, spec.ridge, prior, spec.factor_shrink)
            fc[k, j], fa[k, j] = b
            fa[k, j] = float(np.clip(fa[k, j], -0.995, 0.995))
            resid = y - X @ np.array([fc[k, j], fa[k, j]])
            num = float(np.sum(w * resid * resid) + spec.factor_scale_shrink * p.common_sd[j] ** 2)
            den = float(np.sum(gamma[ok, k]) + spec.factor_scale_shrink)
            fs[k, j] = max(math.sqrt(num / max(den, 1e-12)), spec.min_factor_sd)

    return SDFHMMParams(pi, P, beta_s, beta_c, sigma, float(best_nu), fc, fa, fs,
                        p.common_c.copy(), p.common_a.copy(), p.common_sd.copy())


def _params_from_gamma(gamma, d, spec, common):
    K = spec.K
    cc, ca, cs = common
    pi = gamma[d.price.starts].sum(axis=0) + 1e-3
    pi /= pi.sum()
    P = np.ones((K, K)) if K == 1 else np.full((K, K), 1.0 / K)
    p = SDFHMMParams(
        pi, P,
        np.zeros((K, d.price.Xs.shape[1])),
        np.zeros(d.price.Xc.shape[1]),
        np.full(K, max(float(np.std(d.price.y)), 0.1)),
        8.0,
        np.tile(cc, (K, 1)),
        np.tile(ca, (K, 1)),
        np.tile(cs, (K, 1)),
        cc, ca, cs,
    )
    xi = np.ones((K, K))
    if K > 1:
        for aa, zz in zip(d.price.starts, d.price.ends):
            lab = gamma[aa:zz].argmax(axis=1)
            for x, y in zip(lab[:-1], lab[1:]):
                xi[x, y] += 1
    return _m_step(p, d, spec, gamma, xi)


def _random_block_gamma(d, K, mean_len, rng):
    lab = np.zeros(d.T, int)
    for aa, zz in zip(d.price.starts, d.price.ends):
        t = aa
        k = int(rng.integers(K))
        while t < zz:
            L = int(rng.geometric(1.0 / mean_len))
            lab[t:min(zz, t + L)] = k
            t += L
            if K > 1:
                k = int((k + rng.integers(1, K)) % K)
    return np.eye(K)[lab]


def _initial_params(d, spec):
    common = fit_common_factors(d)
    if spec.K == 1:
        yield _params_from_gamma(np.ones((d.T, 1)), d, spec, common)
        return

    stat = np.zeros(d.T)
    jD = d.price.names_s.index("D") if "D" in d.price.names_s else None
    if jD is not None:
        raw = d.price.Xs[:, jD] * d.price.y
        for aa, zz in zip(d.price.starts, d.price.ends):
            stat[aa:zz] = pd.Series(raw[aa:zz]).rolling(12, min_periods=4).mean().bfill().fillna(0.0).to_numpy()
    qs = np.quantile(stat, np.linspace(0, 1, spec.K + 1)[1:-1])
    lab = np.searchsorted(qs, stat)
    yield _params_from_gamma(np.eye(spec.K)[lab], d, spec, common)
    n = 1

    innov = np.zeros(d.T)
    cnt = np.zeros(d.T)
    cc, ca, cs = common
    for j in range(d.m):
        ok = d.M[:, j]
        z = np.zeros(d.T)
        z[ok] = np.abs((d.F[ok, j] - cc[j] - ca[j] * d.F_lag[ok, j]) / max(cs[j], 1e-6))
        innov += z
        cnt += ok
    stat = innov / np.maximum(cnt, 1)
    qs = np.quantile(stat, np.linspace(0, 1, spec.K + 1)[1:-1])
    lab = np.searchsorted(qs, stat)
    yield _params_from_gamma(np.eye(spec.K)[lab], d, spec, common)
    n += 1

    rng = np.random.default_rng(spec.seed)
    while n < spec.n_init:
        yield _params_from_gamma(_random_block_gamma(d, spec.K, spec.init_block_mean, rng), d, spec, common)
        n += 1


def _penalty(p, spec):
    dc = p.factor_c - p.common_c[None, :]
    da = p.factor_a - p.common_a[None, :]
    return 0.5 * spec.factor_shrink * float(np.sum(dc * dc + da * da))


def fit(d: SDFHMMData, spec: SDFHMMSpec) -> SDFHMMResult:
    best = None
    init_ll = []
    for p0 in _initial_params(d, spec):
        p = p0
        hist = []
        prev = -np.inf
        converged = False
        for it in range(1, spec.max_iter + 1):
            gamma, filt, xi, ll, score = _fb(p, d, spec)
            # The factor block is tempered by omega_f, so its regularization
            # must be tempered by the same weight. In particular omega_f=0
            # must be a pure price-only benchmark: diagnostic factor fits may
            # still be updated, but they cannot affect convergence or init choice.
            obj = ll - spec.omega_f * _penalty(p, spec)
            hist.append(obj)
            if np.isfinite(prev) and abs(obj - prev) <= spec.tol * max(abs(prev), 1.0):
                converged = True
                break
            prev = obj
            p = _m_step(p, d, spec, gamma, xi)
        gamma, filt, xi, ll, score = _fb(p, d, spec)
        obj = ll - spec.omega_f * _penalty(p, spec)
        init_ll.append(float(obj))
        if best is None or obj > best[0]:
            best = (obj, p, ll, gamma, filt, score, it, converged, hist)

    _, p, ll, gamma, filt, score, it, conv, hist = best
    if "D" in d.price.names_s:
        order = np.argsort(p.beta_s[:, d.price.names_s.index("D")])
    else:
        order = np.argsort(p.sigma)
    p = p.permute(order)
    gamma, filt, _, ll, score = _fb(p, d, spec)
    return SDFHMMResult(p, ll, gamma, filt, score, it, conv, init_ll, hist)


def predictive_price_scores(p, d, spec):
    return _fb(p, d, spec)[4]


def episodes(result, d):
    lab = result.gamma.argmax(axis=1)
    out = []
    for si, (aa, zz) in enumerate(zip(d.price.starts, d.price.ends)):
        t = aa
        while t < zz:
            k = lab[t]
            e = t + 1
            while e < zz and lab[e] == k:
                e += 1
            out.append({"segment": si, "start": t, "end": e, "state": int(k),
                        "length": e - t, "censored": bool(t == aa or e == zz)})
            t = e
    return pd.DataFrame(out)


def parameter_table(result, d):
    p = result.params
    rows = []
    for k in range(p.K):
        row = {"state": k, "occupancy": float(result.gamma[:, k].mean()),
               "sigma_price": p.sigma[k], "nu": p.nu}
        row.update({f"b_{n}": p.beta_s[k, j] for j, n in enumerate(d.price.names_s)})
        for j, n in enumerate(d.factor_names):
            row[f"a_{n}"] = p.factor_a[k, j]
            row[f"a0_{n}"] = p.common_a[j]
            row[f"sd_{n}"] = p.factor_sd[k, j]
        rows.append(row)
    return pd.DataFrame(rows)
