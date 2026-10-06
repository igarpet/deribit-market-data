"""
Explicit-duration hidden semi-Markov model with a switching-regression
emission (framework v2.1 §7; Decisions part 2, §10).

Model, per asset
----------------
Latent regime s_t in {1..K}. A regime episode of type k lasts d bars,
d ~ p_k(d) on {1..D_max} (shifted negative binomial, or geometric for the
sticky-HMM baseline). At the end of an episode the next type is drawn from
A[k, .] with A[k, k] = 0 (K >= 2). Within a regime the observation is

    y_t = x_t' b_k + w_t' c + sigma_k * eps_t,      eps_t ~ Student-t(nu)

  y_t  standardized index return z_t = r_t / sigma^c_{t-1}
  x_t  state-specific regressors (intercept, D^c_{t-1})
  w_t  shared regressors (z_{t-1}, z_{t-2}, z_{t-3}): the short-range
       reversal is common to all states, so states are not labelled by it
  c    shared coefficients

An optional masked block o_t (option features) adds independent Gaussian
terms per state; masked entries contribute nothing (never imputed).

Representation
--------------
The HSMM is run as an HMM on the expanded "countdown" state (k, r), r the
remaining bars of the current episode including t:

    (k, r) -> (k, r - 1)                         if r > 1
    (k, 1) -> (j, d)  w.p. A[k, j] p_j(d)        episode ends

so forward-backward costs O(T K D_max). Sequences are split into
segments at grid gaps; each segment starts from pi_k p_k(d) and is
right-censored at its end.

Estimation: EM (ECM for the Student-t emission: latent precision weights
u = (nu + 1) / (nu + e^2 / sigma^2)), several initialisations, best
log-likelihood kept. States are ordered by the slope on the regressor
named "D" (ascending: most mean-reverting first), else by sigma.

Inference: smoothed P(s_t | all), filtered P(s_t | y_<=t), and the one-step
predictive density of y_t given y_<t (used for out-of-sample scoring).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import gammaln
from scipy.stats import nbinom


ENGINE_VERSION = "0.1"
LOG_2PI = math.log(2.0 * math.pi)


# --------------------------------------------------
# Data and parameters
# --------------------------------------------------

@dataclass
class HSMMData:
    """Concatenated segments. Rows t in [starts[i], ends[i]) form segment i."""

    y: np.ndarray                       # (T,)
    Xs: np.ndarray                      # (T, ps) state-specific regressors
    Xc: np.ndarray                      # (T, pc) shared regressors (pc may be 0)
    starts: np.ndarray                  # (S,)
    ends: np.ndarray                    # (S,)
    names_s: list[str] = field(default_factory=list)
    names_c: list[str] = field(default_factory=list)
    O: np.ndarray | None = None         # (T, q) masked block
    M: np.ndarray | None = None         # (T, q) True where observed
    names_o: list[str] = field(default_factory=list)

    def __post_init__(self):
        T = len(self.y)
        assert self.Xs.shape[0] == T and self.Xc.shape[0] == T
        assert np.all(np.isfinite(self.y)) and np.all(np.isfinite(self.Xs)) and np.all(np.isfinite(self.Xc))
        assert len(self.starts) == len(self.ends) and np.all(self.ends > self.starts)
        if self.O is not None:
            assert self.O.shape[0] == T and self.M.shape == self.O.shape
            self.O = np.where(self.M, self.O, 0.0)

    @property
    def T(self) -> int:
        return len(self.y)

    @property
    def has_block(self) -> bool:
        return self.O is not None and self.O.shape[1] > 0

    def without_block(self) -> "HSMMData":
        return HSMMData(self.y, self.Xs, self.Xc, self.starts, self.ends, self.names_s, self.names_c)

    def segment_ids(self) -> np.ndarray:
        ids = np.empty(self.T, dtype=int)
        for i, (a, b) in enumerate(zip(self.starts, self.ends)):
            ids[a:b] = i
        return ids


@dataclass
class HSMMSpec:
    K: int = 2
    d_max: int = 144
    duration: str = "negbin"           # negbin | geometric
    nu_grid: tuple = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 30.0, 100.0)
    max_iter: int = 300
    tol: float = 1e-6                  # relative change of the log-likelihood
    n_init: int = 4
    seed: int = 0
    ridge: float = 1e-6
    min_sigma: float = 1e-3
    min_block_sd: float = 1e-3
    init_block_mean: int = 24          # mean block length of random initial paths


@dataclass
class HSMMParams:
    pi: np.ndarray                      # (K,)
    A: np.ndarray                       # (K, K)
    pmf: np.ndarray                     # (K, D)
    dur: list                           # per state: ("negbin", r, p) or ("geometric", q)
    beta_s: np.ndarray                  # (K, ps)
    beta_c: np.ndarray                  # (pc,)
    sigma: np.ndarray                   # (K,)
    nu: float
    m: np.ndarray | None = None         # (K, q)
    s: np.ndarray | None = None         # (K, q)

    @property
    def K(self) -> int:
        return len(self.pi)

    def mean_duration(self) -> np.ndarray:
        d = np.arange(1, self.pmf.shape[1] + 1)
        return self.pmf @ d

    def permute(self, order: np.ndarray) -> "HSMMParams":
        o = np.asarray(order)
        return HSMMParams(
            pi=self.pi[o], A=self.A[np.ix_(o, o)], pmf=self.pmf[o], dur=[self.dur[i] for i in o],
            beta_s=self.beta_s[o], beta_c=self.beta_c.copy(), sigma=self.sigma[o], nu=self.nu,
            m=None if self.m is None else self.m[o], s=None if self.s is None else self.s[o])

    def to_dict(self) -> dict:
        f = lambda a: None if a is None else np.asarray(a).tolist()
        return {"pi": f(self.pi), "A": f(self.A), "dur": [list(map(lambda v: v if isinstance(v, str) else float(v), d)) for d in self.dur],
                "mean_duration": f(self.mean_duration()), "beta_s": f(self.beta_s), "beta_c": f(self.beta_c),
                "sigma": f(self.sigma), "nu": float(self.nu), "m": f(self.m), "s": f(self.s)}


# --------------------------------------------------
# Duration distributions
# --------------------------------------------------

def duration_pmf(kind: str, d_max: int, *theta) -> np.ndarray:
    """pmf on d = 1..d_max, renormalised (truncation at d_max)."""
    d = np.arange(1, d_max + 1)
    if kind == "geometric":
        q = float(np.clip(theta[0], 1e-6, 1.0))
        p = q * (1.0 - q) ** (d - 1)
    elif kind == "negbin":
        r, pr = theta
        p = nbinom.pmf(d - 1, r, pr)
    else:
        raise ValueError(kind)
    p = np.maximum(p, 1e-300)
    return p / p.sum()


R_GRID = np.exp(np.linspace(np.log(0.15), np.log(60.0), 48))


def fit_duration(counts: np.ndarray, kind: str, d_max: int) -> tuple[np.ndarray, tuple]:
    """Weighted MLE of the truncated duration pmf from expected counts eta(d)."""
    d = np.arange(1, d_max + 1)
    w = np.maximum(np.asarray(counts, float), 0.0)
    if w.sum() <= 1e-9:
        w = np.ones(d_max)
    mean_m1 = max(float((w * (d - 1)).sum() / w.sum()), 1e-3)
    if kind == "geometric":
        best, best_ll = None, -np.inf
        for scale in np.exp(np.linspace(-0.7, 0.7, 29)):   # truncation-aware search around 1/mean
            q = float(np.clip(scale / (mean_m1 + 1.0), 1e-6, 1.0))
            p = duration_pmf("geometric", d_max, q)
            ll = float((w * np.log(p)).sum())
            if ll > best_ll:
                best, best_ll = (q,), ll
        return duration_pmf("geometric", d_max, *best), ("geometric", *best)
    best, best_ll = None, -np.inf
    for r in R_GRID:
        for scale in (0.8, 0.9, 1.0, 1.1, 1.25):
            m = mean_m1 * scale
            pr = r / (r + m)
            p = duration_pmf("negbin", d_max, r, pr)
            ll = float((w * np.log(p)).sum())
            if ll > best_ll:
                best, best_ll = (float(r), float(pr)), ll
    return duration_pmf("negbin", d_max, *best), ("negbin", *best)


# --------------------------------------------------
# Emission densities
# --------------------------------------------------

def t_logpdf(x: np.ndarray, nu: float) -> np.ndarray:
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * math.log(nu * math.pi)
            - (nu + 1) / 2 * np.log1p(x * x / nu))


def residuals(params: HSMMParams, data: HSMMData) -> np.ndarray:
    """(T, K) regression residuals."""
    shared = data.Xc @ params.beta_c if data.Xc.shape[1] else 0.0
    return (data.y - shared)[:, None] - data.Xs @ params.beta_s.T


def emission_logdens(params: HSMMParams, data: HSMMData) -> tuple[np.ndarray, np.ndarray]:
    """(log density of y, log density of y and the masked block), each (T, K)."""
    e = residuals(params, data)
    ly = t_logpdf(e / params.sigma[None, :], params.nu) - np.log(params.sigma)[None, :]
    lb = ly.copy()
    if data.has_block and params.m is not None:
        for k in range(params.K):
            z = (data.O - params.m[k]) / params.s[k]
            lp = -0.5 * (z * z + LOG_2PI) - np.log(params.s[k])
            lb[:, k] += np.where(data.M, lp, 0.0).sum(axis=1)
    return ly, lb


# --------------------------------------------------
# Forward-backward on the countdown representation
# --------------------------------------------------

def forward(params: HSMMParams, data: HSMMData, logb: np.ndarray, logy: np.ndarray | None = None):
    """
    Scaled forward pass. Returns (alpha (T,K,D), c (T,), loglik, pred (T,K),
    y_logscore (T,) or None). pred[t] = P(s_t | obs_<t); y_logscore[t] =
    log p(y_t | obs_<t).
    """
    T, K = logb.shape
    D = params.pmf.shape[1]
    shift = logb.max(axis=1)
    b = np.exp(logb - shift[:, None])
    alpha = np.empty((T, K, D))
    c = np.empty(T)
    pred = np.empty((T, K))
    score = np.empty(T) if logy is not None else None
    pmf, A, pi = params.pmf, params.A, params.pi
    prior = np.empty((K, D))
    for a, z in zip(data.starts, data.ends):
        for t in range(a, z):
            if t == a:
                np.multiply(pi[:, None], pmf, out=prior)
            else:
                prev = alpha[t - 1]
                entry = prev[:, 0] @ A
                prior[:, :-1] = prev[:, 1:]
                prior[:, -1] = 0.0
                prior += entry[:, None] * pmf
            ps = prior.sum(axis=1)
            pred[t] = ps / ps.sum()
            if score is not None:
                ly = logy[t]
                mx = ly.max()
                score[t] = mx + math.log(float(pred[t] @ np.exp(ly - mx)))
            at = prior * b[t][:, None]
            ct = at.sum()
            c[t] = ct
            alpha[t] = at / ct
    loglik = float(np.log(c).sum() + shift.sum())
    return alpha, c, loglik, pred, score


def backward(params: HSMMParams, data: HSMMData, logb: np.ndarray, c: np.ndarray) -> np.ndarray:
    T, K = logb.shape
    D = params.pmf.shape[1]
    b = np.exp(logb - logb.max(axis=1)[:, None])
    beta = np.empty((T, K, D))
    pmf, A = params.pmf, params.A
    for a, z in zip(data.starts, data.ends):
        beta[z - 1] = 1.0
        for t in range(z - 2, a - 1, -1):
            nxt = beta[t + 1]
            G = b[t + 1] * (pmf * nxt).sum(axis=1) / c[t + 1]
            beta[t, :, 0] = A @ G
            beta[t, :, 1:] = b[t + 1][:, None] * nxt[:, :-1] / c[t + 1]
    return beta


def expectations(params: HSMMParams, data: HSMMData, logb, alpha, beta, c):
    """Posterior state marginals, expected episode-entry counts by duration,
    expected type transitions, expected initial types."""
    T, K, D = alpha.shape
    b = np.exp(logb - logb.max(axis=1)[:, None])
    gamma = (alpha * beta).sum(axis=2)
    gamma /= gamma.sum(axis=1, keepdims=True)
    start_mask = np.zeros(T, dtype=bool)
    start_mask[data.starts] = True
    # entries at segment starts
    t0 = data.starts
    init = params.pi[None, :, None] * params.pmf[None] * b[t0][:, :, None] * beta[t0] / c[t0][:, None, None]
    eta = init.sum(axis=0)
    pi_acc = init.sum(axis=(0, 2))
    # entries inside segments
    tt = np.flatnonzero(~start_mask)
    Gp = params.pmf[None] * b[tt][:, :, None] * beta[tt] / c[tt][:, None, None]      # (n, K, D)
    exit_prev = alpha[tt - 1, :, 0]                                                   # (n, K)
    entry = exit_prev @ params.A                                                      # (n, K)
    eta += (entry[:, :, None] * Gp).sum(axis=0)
    Gj = Gp.sum(axis=2)                                                               # (n, K)
    xi = (exit_prev.T @ Gj) * params.A
    return gamma, eta, xi, pi_acc


def smoothed_and_filtered(params: HSMMParams, data: HSMMData):
    _, logb = emission_logdens(params, data)
    alpha, c, ll, _, _ = forward(params, data, logb)
    beta = backward(params, data, logb, c)
    gamma, _, _, _ = expectations(params, data, logb, alpha, beta, c)
    filtered = alpha.sum(axis=2)
    return gamma, filtered, ll


def predictive_scores(params: HSMMParams, data: HSMMData) -> np.ndarray:
    """log p(y_t | observations before t), filtering on y and the block."""
    logy, logb = emission_logdens(params, data)
    _, _, _, _, score = forward(params, data, logb, logy)
    return score


# --------------------------------------------------
# M-step
# --------------------------------------------------

def _wls(X: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float) -> np.ndarray:
    Xw = X * w[:, None]
    G = X.T @ Xw + ridge * np.eye(X.shape[1])
    return np.linalg.solve(G, Xw.T @ y)


def m_step(params: HSMMParams, data: HSMMData, spec: HSMMSpec, gamma, eta, xi, pi_acc) -> HSMMParams:
    K = gamma.shape[1]
    D = params.pmf.shape[1]
    pi = pi_acc / pi_acc.sum()
    if K >= 2:
        A = xi.copy()
        np.fill_diagonal(A, 0.0)
        A = A + 1e-8 * (1 - np.eye(K))
        A /= A.sum(axis=1, keepdims=True)
    else:
        A = np.ones((1, 1))
    pmf = np.empty((K, D))
    dur = []
    for k in range(K):
        if K == 1:
            pmf[k], dur_k = np.r_[1.0, np.full(D - 1, 1e-300)] / (1 + (D - 1) * 1e-300), ("geometric", 1.0)
        else:
            pmf[k], dur_k = fit_duration(eta[k], spec.duration, D)
        dur.append(dur_k)

    # Student-t ECM weights with current parameters
    e = residuals(params, data)
    u = (params.nu + 1) / (params.nu + (e / params.sigma[None, :]) ** 2)
    W = gamma * u
    y_c = data.y - (data.Xc @ params.beta_c if data.Xc.shape[1] else 0.0)
    beta_s = np.vstack([_wls(data.Xs, y_c, W[:, k], spec.ridge) for k in range(K)])
    if data.Xc.shape[1]:
        Wc = (W / params.sigma[None, :] ** 2).sum(axis=1)
        yk = data.y[:, None] - data.Xs @ beta_s.T                       # (T, K)
        target = (W / params.sigma[None, :] ** 2 * yk).sum(axis=1) / np.maximum(Wc, 1e-300)
        beta_c = _wls(data.Xc, target, Wc, spec.ridge)
    else:
        beta_c = np.zeros(0)
    new = HSMMParams(pi, A, pmf, dur, beta_s, beta_c, params.sigma.copy(), params.nu, params.m, params.s)
    e = residuals(new, data)
    sig2 = (W * e * e).sum(axis=0) / np.maximum(gamma.sum(axis=0), 1e-300)
    new.sigma = np.maximum(np.sqrt(sig2), spec.min_sigma)
    # nu on a grid: expected log-likelihood of the y emission
    best_nu, best = params.nu, -np.inf
    zz = e / new.sigma[None, :]
    for nu in spec.nu_grid:
        val = float((gamma * t_logpdf(zz, nu)).sum())
        if val > best:
            best_nu, best = nu, val
    new.nu = float(best_nu)
    if data.has_block:
        q = data.O.shape[1]
        m = np.zeros((K, q))
        s = np.ones((K, q))
        for k in range(K):
            wk = gamma[:, k][:, None] * data.M
            n = np.maximum(wk.sum(axis=0), 1e-12)
            m[k] = (wk * data.O).sum(axis=0) / n
            v = (wk * (data.O - m[k]) ** 2).sum(axis=0) / n
            s[k] = np.maximum(np.sqrt(v), spec.min_block_sd)
        new.m, new.s = m, s
    return new


# --------------------------------------------------
# Initialisation
# --------------------------------------------------

def _rolling_stat(data: HSMMData, kind: str, window: int = 24) -> np.ndarray:
    out = np.full(data.T, np.nan)
    j = data.names_s.index("D") if "D" in data.names_s else None
    for a, z in zip(data.starts, data.ends):
        for t in range(a, z):
            lo = max(a, t - window // 2)
            hi = min(z, t + window // 2)
            if hi - lo < 6:
                continue
            yy = data.y[lo:hi]
            if kind == "vol":
                out[t] = np.mean(np.abs(yy))
            elif kind == "slope" and j is not None:
                xx = data.Xs[lo:hi, j]
                vx = np.var(xx)
                out[t] = np.cov(xx, yy, bias=True)[0, 1] / vx if vx > 0 else 0.0
    return np.where(np.isfinite(out), out, np.nanmedian(out))


def _labels_from_stat(stat: np.ndarray, K: int) -> np.ndarray:
    qs = np.quantile(stat, np.linspace(0, 1, K + 1)[1:-1]) if K > 1 else []
    return np.searchsorted(qs, stat) if K > 1 else np.zeros(len(stat), int)


def _random_labels(data: HSMMData, K: int, mean_len: int, rng) -> np.ndarray:
    lab = np.empty(data.T, dtype=int)
    for a, z in zip(data.starts, data.ends):
        t, k = a, rng.integers(K)
        while t < z:
            L = int(rng.geometric(1.0 / mean_len))
            lab[t:min(z, t + L)] = k
            t += L
            if K > 1:
                k = (k + rng.integers(1, K)) % K
    return lab


def params_from_labels(labels: np.ndarray, data: HSMMData, spec: HSMMSpec) -> HSMMParams:
    K, D = spec.K, (spec.d_max if spec.K > 1 else 1)
    gamma = np.eye(K)[labels]
    eta = np.zeros((K, D))
    xi = np.zeros((K, K))
    pi_acc = np.full(K, 1e-3)
    for a, z in zip(data.starts, data.ends):
        pi_acc[labels[a]] += 1
        t = a
        while t < z:
            k = labels[t]
            e = t
            while e < z and labels[e] == k:
                e += 1
            eta[k, min(e - t, D) - 1] += 1
            if e < z:
                xi[k, labels[e]] += 1
            t = e
    eta += 1e-3
    sd = float(np.std(data.y)) or 1.0
    base = HSMMParams(
        pi=np.full(K, 1.0 / K), A=np.full((K, K), 1.0 / max(K - 1, 1)) * (1 - np.eye(K)) if K > 1 else np.ones((1, 1)),
        pmf=np.full((K, D), 1.0 / D), dur=[("geometric", 1.0 / D)] * K,
        beta_s=np.zeros((K, data.Xs.shape[1])), beta_c=np.zeros(data.Xc.shape[1]),
        sigma=np.full(K, sd), nu=8.0)
    if data.has_block:
        base.m = np.zeros((K, data.O.shape[1]))
        base.s = np.ones((K, data.O.shape[1]))
    p = m_step(base, data, spec, gamma, eta, xi, pi_acc)
    p = m_step(p, data, spec, gamma, eta, xi, pi_acc)      # second pass with fitted sigma/nu
    return p


# --------------------------------------------------
# Fitting
# --------------------------------------------------

@dataclass
class FitResult:
    params: HSMMParams
    loglik: float
    n_params: int
    n_obs: int
    iterations: int
    converged: bool
    history: list
    gamma: np.ndarray
    filtered: np.ndarray
    init_logliks: list

    @property
    def bic(self) -> float:
        return -2 * self.loglik + self.n_params * math.log(self.n_obs)

    @property
    def aic(self) -> float:
        return -2 * self.loglik + 2 * self.n_params


def count_params(spec: HSMMSpec, data: HSMMData) -> int:
    K = spec.K
    n = K * data.Xs.shape[1] + data.Xc.shape[1] + K + 1          # regression, sigma, nu
    if K > 1:
        n += (K - 1) + K * (K - 2)                                   # pi, A (zero diagonal)
        n += K * (2 if spec.duration == "negbin" else 1)
    if data.has_block:
        n += 2 * K * data.O.shape[1]
    return n


def _em(params: HSMMParams, data: HSMMData, spec: HSMMSpec):
    history = []
    prev = -np.inf
    converged = False
    for it in range(1, spec.max_iter + 1):
        _, logb = emission_logdens(params, data)
        alpha, c, ll, _, _ = forward(params, data, logb)
        beta = backward(params, data, logb, c)
        gamma, eta, xi, pi_acc = expectations(params, data, logb, alpha, beta, c)
        history.append(ll)
        if np.isfinite(prev) and abs(ll - prev) <= spec.tol * abs(prev):
            converged = True
            break
        prev = ll
        params = m_step(params, data, spec, gamma, eta, xi, pi_acc)
    return params, ll, it, converged, history


def fit(data: HSMMData, spec: HSMMSpec, init_params: HSMMParams | None = None) -> FitResult:
    """Fit by EM from several initialisations; keep the best log-likelihood."""
    rng = np.random.default_rng(spec.seed)
    if spec.K == 1:
        inits = [params_from_labels(np.zeros(data.T, int), data, spec)]
    else:
        inits = [] if init_params is None else [init_params]
        kinds = ("slope", "vol") if "D" in data.names_s else ("vol",)
        for kind in kinds:
            if len(inits) >= spec.n_init:
                break
            inits.append(params_from_labels(_labels_from_stat(_rolling_stat(data, kind), spec.K), data, spec))
        while len(inits) < spec.n_init:
            inits.append(params_from_labels(_random_labels(data, spec.K, spec.init_block_mean, rng), data, spec))
    best = None
    init_ll = []
    for p0 in inits:
        p, ll, it, conv, hist = _em(p0, data, spec)
        init_ll.append(ll)
        if best is None or ll > best[1]:
            best = (p, ll, it, conv, hist)
    p, ll, it, conv, hist = best
    if "D" in data.names_s:
        order = np.argsort(p.beta_s[:, data.names_s.index("D")])
    else:
        order = np.argsort(p.sigma)
    p = p.permute(order)
    gamma, filtered, ll = smoothed_and_filtered(p, data)
    return FitResult(p, ll, count_params(spec, data), data.T, it, conv, hist, gamma, filtered, init_ll)


# --------------------------------------------------
# Episodes and simulation
# --------------------------------------------------

def episodes(gamma: np.ndarray, data: HSMMData) -> list[dict]:
    """Runs of the posterior-mode state; censored when touching a segment edge."""
    lab = gamma.argmax(axis=1)
    out = []
    for i, (a, z) in enumerate(zip(data.starts, data.ends)):
        t = a
        while t < z:
            k = lab[t]
            e = t
            while e < z and lab[e] == k:
                e += 1
            out.append({"segment": i, "start": t, "end": e, "state": int(k), "length": e - t,
                        "censored": bool(t == a or e == z)})
            t = e
    return out


def simulate_states(params: HSMMParams, T: int, rng) -> np.ndarray:
    K, D = params.pmf.shape
    lab = np.empty(T, dtype=int)
    k = rng.choice(K, p=params.pi)
    t = 0
    while t < T:
        d = rng.choice(D, p=params.pmf[k]) + 1
        lab[t:min(T, t + d)] = k
        t += d
        k = rng.choice(K, p=params.A[k]) if K > 1 else k
    return lab
