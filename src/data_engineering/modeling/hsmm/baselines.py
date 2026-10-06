"""
Baselines the HSMM must beat out of sample (framework v2.1 §6, board b1).

All models predict the same target y_t = z_t from information up to t-1 and
are scored on the same test rows by the one-step predictive log density:

  no_regime_lags   one regime: y ~ const + 3 lags                 (Student-t)
  no_regime_D      one regime: y ~ const + D + 3 lags             (Student-t)
  hsmm_K<k>        explicit-duration HSMM, negative-binomial durations (K from hsmm_K)
  sticky_hmm_K<k>  same emission, geometric durations (a plain HMM), at the smallest K >= 2
  ms_ar1_K2        Markov-switching AR(1): y ~ const_k + phi_k z_{t-1}, geometric
  jump_K2          statistical jump model on [D, z1, mean(z1..z3), VR6] with a jump
                   penalty (Nystrup et al.), online greedy labels out of sample,
                   per-state Student-t regressions
  rule_vr3         rule-based: terciles of the trailing variance ratio VR(6)

Score differences are tested with Newey-West (12 lags) standard errors.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from data_engineering.modeling.hsmm.design import Design
from data_engineering.modeling.hsmm.engine import HSMMSpec, fit, predictive_scores, t_logpdf


NU_GRID = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 30.0, 100.0)
JUMP_PENALTY = 10.0


# --------------------------------------------------
# Student-t regression
# --------------------------------------------------

def fit_t_regression(X: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple:
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    sigma, nu = float(np.std(e)) or 1.0, 8.0
    for _ in range(iters):
        u = (nu + 1) / (nu + (e / sigma) ** 2)
        Xw = X * u[:, None]
        beta = np.linalg.solve(X.T @ Xw + 1e-8 * np.eye(X.shape[1]), Xw.T @ y)
        e = y - X @ beta
        sigma = max(math.sqrt(float(np.mean(u * e * e))), 1e-6)
        nu = max(NU_GRID, key=lambda v: float(t_logpdf(e / sigma, v).sum()))
    return beta, sigma, nu


def t_score(X, y, params) -> np.ndarray:
    beta, sigma, nu = params
    return t_logpdf((y - X @ beta) / sigma, nu) - math.log(sigma)


# --------------------------------------------------
# Jump model
# --------------------------------------------------

def _jump_dp(C: np.ndarray, lam: float) -> np.ndarray:
    T, K = C.shape
    V = C[0].copy()
    back = np.zeros((T, K), int)
    for t in range(1, T):
        best_prev = V.argmin()
        stay = V
        switch = V[best_prev] + lam
        choose_switch = switch < stay
        back[t] = np.where(choose_switch, best_prev, np.arange(K))
        V = C[t] + np.minimum(stay, switch)
    lab = np.empty(T, int)
    lab[-1] = V.argmin()
    for t in range(T - 1, 0, -1):
        lab[t - 1] = back[t, lab[t]]
    return lab


def fit_jump(F: np.ndarray, segments: list, K: int, lam: float, rng, restarts: int = 5, iters: int = 20):
    best = None
    for _ in range(restarts):
        theta = F[rng.choice(len(F), K, replace=False)]
        for _ in range(iters):
            C = ((F[:, None, :] - theta[None]) ** 2).sum(axis=2)
            lab = np.concatenate([_jump_dp(C[a:b], lam) for a, b in segments])
            new = np.vstack([F[lab == k].mean(axis=0) if (lab == k).any() else theta[k] for k in range(K)])
            if np.allclose(new, theta):
                break
            theta = new
        C = ((F[:, None, :] - theta[None]) ** 2).sum(axis=2)
        obj = float(C[np.arange(len(F)), lab].sum() + lam * sum(np.count_nonzero(np.diff(lab[a:b])) for a, b in segments))
        if best is None or obj < best[0]:
            best = (obj, theta, lab)
    return best[1], best[2]


def online_jump_labels(F: np.ndarray, segments: list, theta: np.ndarray, lam: float) -> np.ndarray:
    lab = np.empty(len(F), int)
    for a, b in segments:
        prev = None
        for t in range(a, b):
            c = ((F[t] - theta) ** 2).sum(axis=1)
            if prev is not None:
                c = c + lam * (np.arange(len(theta)) != prev)
            prev = int(c.argmin())
            lab[t] = prev
    return lab


# --------------------------------------------------
# Comparison
# --------------------------------------------------

def hac_t(d: np.ndarray, lags: int = 12) -> float:
    """t-statistic of mean(d) with Newey-West long-run variance."""
    n = len(d)
    if n < 2:
        return np.nan
    m = float(np.mean(d))
    x = d - m
    s = float(x @ x) / n
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * float(x[L:] @ x[:-L]) / n
    return m / math.sqrt(max(s, 1e-300) / n)


def compare_models(design: Design, train: np.ndarray, test: np.ndarray, spec_kw: dict | None = None,
                   seed: int = 0, hsmm_K=(2, 3)) -> tuple[pd.DataFrame, dict]:
    """Fit every model on train rows; score on the common test rows."""
    spec_kw = dict(spec_kw or {})
    rng = np.random.default_rng(seed)
    test_rows = design.rows_of(test)
    train_rows = design.rows_of(train)
    fr = design.frame
    y_te = fr["y"].to_numpy()[test_rows]
    full_X = design.columns
    scores, fits = {}, {}

    def hsmm(name, K, duration="negbin", xs=("const", "D"), xc=("z1", "z2", "z3")):
        spec = HSMMSpec(K=K, duration=duration, seed=seed, **spec_kw)
        tr = design.data(list(xs), list(xc), rows=train)
        te = design.data(list(xs), list(xc), rows=test)
        res = fit(tr, spec)
        scores[name] = predictive_scores(res.params, te)
        fits[name] = res

    hsmm("no_regime_lags", 1, xs=("const",))
    hsmm("no_regime_D", 1)
    for K in sorted(set(int(k) for k in hsmm_K if int(k) >= 2)):
        hsmm(f"hsmm_K{K}", K)
    K_sticky = max(2, min(int(k) for k in hsmm_K)) if len(hsmm_K) else 2
    hsmm(f"sticky_hmm_K{K_sticky}", K_sticky, duration="geometric")
    hsmm("ms_ar1_K2", 2, duration="geometric", xs=("const", "z1"), xc=())

    # hard-label baselines: features known at t-1
    feats = np.column_stack([fr["D"], fr["z1"], fr[["z1", "z2", "z3"]].mean(axis=1), fr["vr_q6_prev"]])
    med = np.nanmedian(feats[train_rows], axis=0)
    feats = np.where(np.isfinite(feats), feats, med)
    mu, sd = feats[train_rows].mean(axis=0), feats[train_rows].std(axis=0) + 1e-12
    F = (feats - mu) / sd

    def segs(rows):
        brk = np.r_[True, np.diff(rows) != 1] if len(rows) else np.array([], bool)
        st = np.flatnonzero(brk)
        return list(zip(st, np.r_[st[1:], len(rows)]))

    X_all = full_X
    theta, lab_tr = fit_jump(F[train_rows], segs(train_rows), 2, JUMP_PENALTY, rng)
    lab_te = online_jump_labels(F[test_rows], segs(test_rows), theta, JUMP_PENALTY)
    regs = [fit_t_regression(X_all[train_rows][lab_tr == k], fr["y"].to_numpy()[train_rows][lab_tr == k])
            for k in range(2)]
    scores["jump_K2"] = np.array([t_score(X_all[test_rows][i:i + 1], y_te[i:i + 1], regs[lab_te[i]])[0]
                                  for i in range(len(test_rows))])
    fits["jump_K2"] = {"theta": theta, "regressions": regs, "train_share": np.bincount(lab_tr, minlength=2) / len(lab_tr)}

    vr = fr["vr_q6_prev"].to_numpy()
    qs = np.nanquantile(vr[train_rows], [1 / 3, 2 / 3])
    rule = lambda v: np.where(np.isfinite(v), np.searchsorted(qs, v), 1)
    lr_tr, lr_te = rule(vr[train_rows]), rule(vr[test_rows])
    regs_r = [fit_t_regression(X_all[train_rows][lr_tr == k], fr["y"].to_numpy()[train_rows][lr_tr == k]) for k in range(3)]
    scores["rule_vr3"] = np.array([t_score(X_all[test_rows][i:i + 1], y_te[i:i + 1], regs_r[lr_te[i]])[0]
                                   for i in range(len(test_rows))])
    fits["rule_vr3"] = {"terciles": qs, "regressions": regs_r}

    base = scores["no_regime_lags"]
    rows = []
    for name, s in scores.items():
        if len(s) != len(base):
            raise RuntimeError(f"{name}: scored {len(s)} rows, expected {len(base)}")
        d = s - base
        rows.append({"model": name, "test_rows": len(s), "mean_logscore": float(s.mean()),
                     "gain_vs_no_regime_x1000": 1000 * float(d.mean()), "hac_t": hac_t(d)})
    return pd.DataFrame(rows), {"scores": scores, "fits": fits, "test_rows": test_rows}
