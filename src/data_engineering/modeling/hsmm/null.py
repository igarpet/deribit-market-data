"""
Regime-free null for the HSMM (framework v2.1 §6, board b2).

Simulated returns replace the real ones on the real grid, and the whole
pipeline is rerun: index path, anchor, clock-adjusted scale, z, D, design
rows, and the HSMM fits. If K = 2 states with separated displacement slopes
appear as often on regime-free paths as on the real data, the real states
are construction artifacts.

Generators (missing real returns stay missing):

  ar_bootstrap  u_t = sum_{k=1}^{3} phi_k u_{t-k} + e_t, with phi fitted to the
                real standardized returns and e_t resampled from the fitted
                residuals; r*_t = sigma^c_{t-1} u_t on the real volatility path.
                Keeps clock, volatility clustering, heavy tails and the
                short-range reversal; has no regimes and no anchor effect.
  sign          r*_t = s_t |r_t|, s_t = +-1 i.i.d.: exact real magnitudes, no
                sign dependence at all.

Statistics per path (in sample, full data):
  lr            2 (loglik_K2 - loglik_K1)
  beta_spread   max_k beta_D,k - min_k beta_D,k (K = 2)
  oos_gain      mean out-of-sample log score K=2 minus K=1, x1000 (train/test split)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_engineering.modeling.hsmm.design import build_design
from data_engineering.modeling.hsmm.engine import HSMMSpec, fit, predictive_scores


def fit_ar(z: np.ndarray, p: int = 3) -> tuple[np.ndarray, np.ndarray]:
    s = pd.Series(z)
    X = pd.concat([s.shift(k) for k in range(1, p + 1)], axis=1)
    d = pd.concat([s.rename("y"), X], axis=1).dropna()
    A = np.column_stack([np.ones(len(d)), d.drop(columns="y").to_numpy()])
    b, *_ = np.linalg.lstsq(A, d["y"].to_numpy(), rcond=None)
    resid = d["y"].to_numpy() - A @ b
    return b[1:], resid - resid.mean()


def simulate_returns(g: pd.DataFrame, f_clock: np.ndarray, kind: str, rng, real_design=None) -> np.ndarray:
    r = g["r"].to_numpy(float)
    finite = np.isfinite(r)
    if kind == "sign":
        return np.where(finite, rng.choice([-1.0, 1.0], size=len(r)) * np.abs(r), np.nan)
    if kind != "ar_bootstrap":
        raise ValueError(kind)
    d = real_design if real_design is not None else build_design(g, f_clock)
    phi, resid = fit_ar(d.frame["y"].to_numpy())
    # real sigma^c_{t-1} path on the full grid
    sig = np.full(len(r), np.nan)
    sig[d.frame["source_row"].to_numpy()] = d.frame["sigma_c_prev"].to_numpy()
    fill = np.nanmedian(sig)
    sig = pd.Series(sig).ffill().fillna(fill).to_numpy()
    burn = 50
    e = rng.choice(resid, size=len(r) + burn)
    u = np.zeros(len(r) + burn)
    for t in range(3, len(u)):
        u[t] = phi[0] * u[t - 1] + phi[1] * u[t - 2] + phi[2] * u[t - 3] + e[t]
    return np.where(finite, sig * u[burn:], np.nan)


def path_statistics(design, train, test, spec: HSMMSpec, init=None) -> dict:
    full = design.data()
    r1 = fit(full, HSMMSpec(K=1))
    r2 = fit(full, spec, init_params=init)
    j = full.names_s.index("D")
    out = {"lr": 2 * (r2.loglik - r1.loglik), "beta_spread": float(np.ptp(r2.params.beta_s[:, j])),
           "beta_low": float(r2.params.beta_s[:, j].min()), "beta_high": float(r2.params.beta_s[:, j].max()),
           "sigma_ratio": float(r2.params.sigma.max() / r2.params.sigma.min()),
           "mean_duration_low": float(r2.params.mean_duration()[0]),
           "mean_duration_high": float(r2.params.mean_duration()[-1])}
    if train is not None:
        tr, te = design.data(rows=train), design.data(rows=test)
        s1 = predictive_scores(fit(tr, HSMMSpec(K=1)).params, te)
        s2 = predictive_scores(fit(tr, spec).params, te)
        out["oos_gain_x1000"] = 1000 * float(np.mean(s2 - s1))
    return out, r2


def run_null(g: pd.DataFrame, f_clock: np.ndarray, train_dates: set, n_paths: int, kind: str = "ar_bootstrap",
             spec: HSMMSpec | None = None, seed: int = 0, oos: bool = True, verbose: bool = True) -> tuple[dict, pd.DataFrame]:
    """Real statistics and the null distribution for one asset."""
    spec = spec or HSMMSpec(K=2, n_init=2, max_iter=150)
    rng = np.random.default_rng(seed)
    real = build_design(g, f_clock)
    tr = real.frame["session_date"].isin(train_dates).to_numpy() if oos else None
    real_stats, real_fit = path_statistics(real, tr, None if tr is None else ~tr, spec)
    rows = []
    for b in range(n_paths):
        r_star = simulate_returns(g, f_clock, kind, rng, real_design=real)
        d = build_design(g, f_clock, returns=r_star)
        trb = d.frame["session_date"].isin(train_dates).to_numpy() if oos else None
        st, _ = path_statistics(d, trb, None if trb is None else ~trb, spec, init=None)
        rows.append({"path": b, **st})
        if verbose and (b + 1) % 10 == 0:
            print(f"  {g['underlying'].iloc[0]}: {b + 1}/{n_paths} null paths")
    return real_stats, pd.DataFrame(rows)


def p_value(real: float, null: np.ndarray, side: str = "greater") -> float:
    null = np.asarray(null, float)
    if side == "greater":
        return float((1 + np.sum(null >= real)) / (1 + len(null)))
    return float((1 + np.sum(null <= real)) / (1 + len(null)))
