"""
Volatility Dynamics 06c: Student-t emission HMM.

Motivation (VD 011 §2.4)
------------------------
In 06b the validation log-likelihood of the diagonal-Gaussian HMM rose
monotonically from K=2 to K=7 (the grid edge). One explanation is emission
misspecification: heavy-tailed data force a Gaussian HMM to spend extra
states approximating the tails. 06c replaces the Gaussian emission with a
multivariate Student-t and asks whether the state count then stabilizes,
and whether the states carry q99 information beyond the volatility benchmark.

Emission model
--------------
For state k, with location mu_k, diagonal scale Sigma_k = diag(s2_k) and
degrees of freedom nu_k (estimated per state):

    x_t | S_t = k  ~  t_{nu_k}(mu_k, Sigma_k)

equivalently x_t | u_t, k ~ N(mu_k, Sigma_k / u_t), u_t ~ Gamma(nu_k/2, nu_k/2).
One latent scale u_t is shared by all fields in a bar: an activity burst
inflates range, volume, trades, ... together. Missing fields are
marginalized exactly (the marginal of a multivariate t over the observed
coordinates is a t with the same nu and p_t = number of observed fields):

    log f = lgamma((nu+p)/2) - lgamma(nu/2) - (p/2) log(nu*pi)
            - 0.5 * sum_obs log s2 - ((nu+p)/2) log(1 + delta/nu),
    delta = sum_obs (x - mu)^2 / s2.

Estimation: Baum-Welch with ECME for the t part (Liu & Rubin 1995):
E[u_t | x_t, k] = (nu_k + p_t) / (nu_k + delta_tk); mu, s2 are weighted
by gamma_tk * E[u]; nu_k solves the standard score equation by bisection
on [2.05, 300]. nu at the upper bound means the state is effectively
Gaussian.

Everything else is the 06b pipeline unchanged (emissions, split, hour
normalization, sequences, benchmark, decision rule, duration check), imported
from volatility_hmm_states_b. Branch-internal reuse only. NumPy only.

Pre-registered (fixed before any 06c real-data output exists)
-------------------------------------------------------------
1. PRIMARY decision: identical to 06b (VD 010 §5.4). At h = 60 min,
   logloss(M3) - logloss(M1) < 0 with 95% day-bootstrap upper bound < 0 and
   >= 20 unique held-out onset timestamps; else FAIL / INSUFFICIENT_DATA.
2. K question: K grid 2..10. "Interior K" iff the selected K < max(grid)
   AND validation LL/row at max(grid) is not above the selected K's.
   Selection by validation LL, occupancy >= 1%, as in 06b.
3. Emission question: the Student-t HMM fits better than the Gaussian iff
   its validation LL/row at its selected K exceeds the Gaussian's at the
   Gaussian's selected K (both fitted here on identical sequences).
None of these makes any state a displacement (R) regime: the emissions
contain no displacement-from-VWAP variable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma, pi
from typing import Sequence

import numpy as np
import pandas as pd

from data_engineering.modeling import volatility_hmm_states_b as hb


VERSION = "0.1.0"
DEFAULT_K = (2, 3, 4, 5, 6, 7, 8, 9, 10)
DEFAULT_N_INIT = 3
DEFAULT_MAX_ITER = 80
DEFAULT_TOL = 1e-5
NU_MIN = 2.05
NU_MAX = 300.0
NU_INIT = 8.0

_lgamma = np.vectorize(lgamma, otypes=[float])


def digamma(x: np.ndarray) -> np.ndarray:
    """Digamma for x > 0 (recurrence to x >= 6, then asymptotic series)."""
    x = np.asarray(x, float).copy()
    res = np.zeros_like(x)
    for _ in range(6):
        small = x < 6
        if not small.any():
            break
        res[small] -= 1.0 / x[small]
        x[small] += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    res += np.log(x) - 0.5 * inv - inv2 * (1 / 12 - inv2 * (1 / 120 - inv2 * (1 / 252 - inv2 * (1 / 240 - inv2 / 132))))
    return res


def t_logB_and_weights(X: np.ndarray, means: np.ndarray, s2: np.ndarray, nu: np.ndarray):
    """(T,K) log density over observed dims, E[u|x,k], and p_t (observed dims)."""
    M = np.isfinite(X)
    X0 = np.where(M, X, 0.0)
    p = M.sum(axis=1).astype(float)
    T, K = len(X), len(means)
    lb = np.zeros((T, K))
    w = np.ones((T, K))
    s2 = np.maximum(s2, 1e-6)
    for k in range(K):
        q = (X0 - means[k]) ** 2 / s2[k]
        delta = np.sum(np.where(M, q, 0.0), axis=1)
        logdet = np.sum(np.where(M, np.log(s2[k])[None, :], 0.0), axis=1)
        nk = float(nu[k])
        lb[:, k] = (
            _lgamma((nk + p) / 2.0) - lgamma(nk / 2.0)
            - 0.5 * p * np.log(nk * pi) - 0.5 * logdet
            - 0.5 * (nk + p) * np.log1p(delta / nk)
        )
        w[:, k] = (nk + p) / (nk + delta)
    lb[p == 0] = 0.0
    return lb, w, p


def _fb_from_logB(lb: np.ndarray, pi_: np.ndarray, A: np.ndarray, need_xi: bool = True):
    mx = lb.max(axis=1, keepdims=True)
    B = np.exp(lb - mx)
    T, K = B.shape
    alpha = np.empty((T, K))
    c = np.empty(T)
    a = pi_ * B[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * B[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    ll = float(np.sum(np.log(c)) + mx.sum())
    if not need_xi:
        return ll, alpha, None, None
    beta = np.empty((T, K))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    if T > 1:
        xi = alpha[:-1, :, None] * A[None] * (B[1:] * beta[1:])[:, None, :] / c[1:, None, None]
        xi_sum = xi.sum(axis=0)
    else:
        xi_sum = np.zeros((K, K))
    return ll, alpha, gamma, xi_sum


@dataclass
class THMM:
    K: int
    pi: np.ndarray
    A: np.ndarray
    means: np.ndarray
    variances: np.ndarray       # diagonal scale^2 (not the t variance)
    nu: np.ndarray
    loglik: float
    n_iter: int
    converged: bool
    seed: int

    def _lb(self, X):
        return t_logB_and_weights(X, self.means, self.variances, self.nu)[0]

    def loglik_of(self, seqs: Sequence[np.ndarray]) -> float:
        return float(sum(_fb_from_logB(self._lb(X), self.pi, self.A, need_xi=False)[0] for X in seqs))

    def filter(self, X: np.ndarray) -> np.ndarray:
        return _fb_from_logB(self._lb(X), self.pi, self.A, need_xi=False)[1]

    def sample(self, T: int, mask: np.ndarray, rng: np.random.Generator):
        s = np.empty(T, int)
        s[0] = rng.choice(self.K, p=self.pi)
        cum = np.cumsum(self.A, axis=1)
        uu = rng.random(T)
        for t in range(1, T):
            s[t] = min(int(np.searchsorted(cum[s[t - 1]], uu[t])), self.K - 1)
        nu = np.minimum(self.nu[s], 1e6)
        u = rng.gamma(shape=nu / 2.0, scale=2.0 / nu)
        X = self.means[s] + rng.normal(size=(T, self.means.shape[1])) * np.sqrt(self.variances[s] / u[:, None])
        return np.where(mask, X, np.nan), s


def _update_nu(nu_old: float, gam: np.ndarray, w: np.ndarray, p: np.ndarray) -> float:
    """ECME/ECM score equation for one state's nu, solved by bisection."""
    m = (p > 0) & (gam > 0)
    if gam[m].sum() <= 1e-8:
        return nu_old
    g, wk, pk = gam[m], w[m], p[m]
    S = np.sum(g * (np.log(wk) - wk + digamma((nu_old + pk) / 2.0) - np.log((nu_old + pk) / 2.0))) / g.sum()

    def f(nu):
        return float(-digamma(np.array([nu / 2.0]))[0] + np.log(nu / 2.0) + 1.0 + S)

    lo, hi = NU_MIN, NU_MAX
    if f(hi) > 0:
        return NU_MAX
    if f(lo) < 0:
        return NU_MIN
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def fit_thmm(seqs: Sequence[np.ndarray], K: int, *, seed: int, max_iter: int = DEFAULT_MAX_ITER,
             tol: float = DEFAULT_TOL, min_variance: float = 1e-2, nu_init: float = NU_INIT) -> THMM:
    rng = np.random.default_rng(int(seed))
    Xall = np.vstack(seqs)
    d = Xall.shape[1]
    complete = Xall[np.isfinite(Xall).all(axis=1)]
    pool = complete if len(complete) >= K * 20 else np.nan_to_num(Xall)
    means = pool[rng.choice(len(pool), size=K, replace=False)].copy()
    gv = np.nanvar(Xall, axis=0)
    s2 = np.tile(np.where(np.isfinite(gv) & (gv > 1e-3), gv, 1.0), (K, 1))
    nu = np.full(K, float(nu_init))
    pi_ = np.full(K, 1.0 / K)
    A = rng.dirichlet(np.ones(K), size=K) + 4.0 * np.eye(K)
    A /= A.sum(axis=1, keepdims=True)
    prev, ll, it, converged = -np.inf, -np.inf, 0, False
    for it in range(1, int(max_iter) + 1):
        s0 = np.zeros(K)
        tr = np.zeros((K, K))
        gw_sum = np.zeros((K, d))
        gwx = np.zeros((K, d))
        gwx2 = np.zeros((K, d))
        g_obs = np.zeros((K, d))
        gam_all, w_all, p_all = [], [], []
        ll = 0.0
        for X in seqs:
            lb, w, p = t_logB_and_weights(X, means, s2, nu)
            l, _, g, xi = _fb_from_logB(lb, pi_, A)
            ll += l
            s0 += g[0]
            tr += xi
            M = np.isfinite(X).astype(float)
            X0 = np.nan_to_num(X)
            gw = g * w
            gw_sum += gw.T @ M
            gwx += gw.T @ X0
            gwx2 += gw.T @ (X0 ** 2)
            g_obs += g.T @ M
            gam_all.append(g)
            w_all.append(w)
            p_all.append(p)
        pi_ = (s0 + 1e-6) / (s0 + 1e-6).sum()
        A = (tr + 1e-6) / (tr + 1e-6).sum(axis=1, keepdims=True)
        means = gwx / np.maximum(gw_sum, 1e-8)
        s2 = (gwx2 - 2 * means * gwx + means ** 2 * gw_sum) / np.maximum(g_obs, 1e-8)
        s2 = np.maximum(s2, float(min_variance))
        G = np.vstack(gam_all)
        W = np.vstack(w_all)
        P = np.concatenate(p_all)
        nu = np.array([_update_nu(nu[k], G[:, k], W[:, k], P) for k in range(K)])
        if np.isfinite(prev) and abs(ll - prev) <= tol * (1 + abs(prev)):
            converged = True
            break
        prev = ll
    return THMM(K, pi_, A, means, s2, nu, float(ll), it, converged, int(seed))


def _occupancy(model, seqs) -> np.ndarray:
    occ = np.zeros(model.K)
    n = 0
    for X in seqs:
        lb = model._lb(X) if isinstance(model, THMM) else hb._logB(X, model.means, model.variances)
        occ += _fb_from_logB(lb, model.pi, model.A)[2].sum(axis=0)
        n += len(X)
    return occ / max(n, 1)


def select_and_fit_both(fit_seqs, valid_seqs, train_seqs, k_values=DEFAULT_K, *, n_init=DEFAULT_N_INIT,
                        n_init_gauss=1, max_iter=DEFAULT_MAX_ITER, seed_base=6800, run_gaussian=True,
                        progress=False):
    """Validation-LL K selection for Student-t (and Gaussian, for comparison) on identical sequences."""
    n_fit = sum(map(len, fit_seqs))
    n_val = sum(map(len, valid_seqs))
    rows = []
    for K in k_values:
        fits = [fit_thmm(fit_seqs, K, seed=seed_base + 100 * K + j, max_iter=max_iter) for j in range(n_init)]
        best = max(fits, key=lambda m: m.loglik)
        occ = _occupancy(best, fit_seqs)
        rows.append({
            "emission": "student_t", "K": K,
            "fit_loglik_per_row": best.loglik / n_fit,
            "valid_loglik_per_row": best.loglik_of(valid_seqs) / max(n_val, 1),
            "min_state_occupancy": float(occ.min()),
            "n_converged": int(sum(m.converged for m in fits)),
            "nu_by_state": ";".join(f"{v:.1f}" for v in best.nu),
        })
        if run_gaussian:
            gf = [hb.fit_hmm(fit_seqs, K, seed=seed_base + 50_000 + 100 * K + j, max_iter=max_iter) for j in range(n_init_gauss)]
            gb = max(gf, key=lambda m: m.loglik)
            gocc = _occupancy(gb, fit_seqs)
            rows.append({
                "emission": "gaussian", "K": K,
                "fit_loglik_per_row": gb.loglik / n_fit,
                "valid_loglik_per_row": gb.loglik_of(valid_seqs) / max(n_val, 1),
                "min_state_occupancy": float(gocc.min()),
                "n_converged": int(sum(m.converged for m in gf)),
                "nu_by_state": "",
            })
        if progress:
            last = [r for r in rows if r["K"] == K]
            print(f"K={K}: " + " | ".join(f"{r['emission']} valid ll/row {r['valid_loglik_per_row']:.4f}" for r in last))
    comp = pd.DataFrame(rows)

    def pick(em):
        c = comp[comp.emission == em]
        ok = c[c.min_state_occupancy >= 0.01]
        return int((ok if len(ok) else c).sort_values("valid_loglik_per_row", ascending=False).iloc[0]["K"])

    k_t = pick("student_t")
    comp["selected"] = False
    comp.loc[(comp.emission == "student_t") & (comp.K == k_t), "selected"] = True
    k_g = None
    if run_gaussian:
        k_g = pick("gaussian")
        comp.loc[(comp.emission == "gaussian") & (comp.K == k_g), "selected"] = True
    final = max((fit_thmm(train_seqs, k_t, seed=seed_base + 9000 + j, max_iter=max_iter) for j in range(n_init)),
                key=lambda m: m.loglik)
    return comp, final, k_t, k_g


def k_and_emission_questions(comp: pd.DataFrame, k_t: int, k_g: int | None) -> dict:
    t = comp[comp.emission == "student_t"].set_index("K")
    kmax = int(t.index.max())
    interior = bool(k_t < kmax and t.loc[kmax, "valid_loglik_per_row"] <= t.loc[k_t, "valid_loglik_per_row"])
    out = {
        "k_grid": [int(k) for k in t.index],
        "student_t_selected_K": int(k_t),
        "student_t_interior_K": interior,
        "student_t_valid_ll_per_row_selected": float(t.loc[k_t, "valid_loglik_per_row"]),
    }
    if k_g is not None:
        g = comp[comp.emission == "gaussian"].set_index("K")
        out.update({
            "gaussian_selected_K": int(k_g),
            "gaussian_interior_K": bool(k_g < kmax and g.loc[kmax, "valid_loglik_per_row"] <= g.loc[k_g, "valid_loglik_per_row"]),
            "gaussian_valid_ll_per_row_selected": float(g.loc[k_g, "valid_loglik_per_row"]),
            "student_t_fits_better": bool(t.loc[k_t, "valid_loglik_per_row"] > g.loc[k_g, "valid_loglik_per_row"]),
        })
    return out


def selftest() -> dict:
    # 1. digamma
    assert np.isclose(digamma(np.array([1.0]))[0], -0.5772156649, atol=1e-8)
    assert np.isclose(digamma(np.array([0.5]))[0], -1.9635100260, atol=1e-8)
    # 2. heavy-tailed 2-state data: t-HMM must beat the Gaussian HMM and find small nu
    rng = np.random.default_rng(3)
    A = np.array([[0.97, 0.03], [0.05, 0.95]])
    mu = np.array([[-1.0, -0.8, -0.5], [1.2, 1.0, 0.8]])
    seqs = []
    for _ in range(4):
        T = 600
        s = np.zeros(T, int)
        for t in range(1, T):
            s[t] = rng.choice(2, p=A[s[t - 1]])
        u = rng.gamma(2.0, 0.5, size=T)                       # nu = 4
        X = mu[s] + 0.6 * rng.normal(size=(T, 3)) / np.sqrt(u)[:, None]
        X[rng.random((T, 3)) < 0.1] = np.nan
        seqs.append(X)
    tm = fit_thmm(seqs, 2, seed=1, max_iter=60)
    gm = hb.fit_hmm(seqs, 2, seed=1, max_iter=60)
    assert tm.loglik > gm.loglik
    assert np.all(tm.nu < 15), tm.nu
    # 3. scaled forward pass equals brute-force log-space recursion
    X = seqs[0][:60]
    lb = tm._lb(X)
    la = np.log(tm.pi) + lb[0]
    for t in range(1, len(X)):
        m = la.max()
        la = lb[t] + np.log(np.exp(la - m) @ tm.A) + m
    brute = float(np.log(np.exp(la - la.max()).sum()) + la.max())
    assert np.isclose(brute, tm.loglik_of([X]), rtol=1e-8, atol=1e-6)
    # 4. t density integrates the Gaussian limit: huge nu ~ Gaussian log density
    Xg = rng.normal(size=(5, 3))
    lt, _, _ = t_logB_and_weights(Xg, np.zeros((1, 3)), np.ones((1, 3)), np.array([1e7]))
    lg = hb._logB(Xg, np.zeros((1, 3)), np.ones((1, 3)))
    assert np.allclose(lt, lg, atol=1e-4)
    # 5. duck-typing with the 06b duration check
    summ, _ = hb.duration_check(tm, seqs[:2], n_sims=2)
    assert len(summ) == 2
    return {"digamma": True, "t_beats_gaussian_on_t_data": True, "nu_recovered_small": True,
            "scaled_fb_matches_logspace": True, "gaussian_limit": True, "duration_check_compatible": True}
