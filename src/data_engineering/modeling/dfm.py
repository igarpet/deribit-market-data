"""
Non-switching dynamic factor model (framework v3.2 §21 step 5, §6.12 B):

    F_t = c + A F_{t-1} + eta_t

fitted per asset on the training rows by equation-wise least squares with a
small ridge. Masked data (v3.2 §6.7): equation j uses the rows where F_{j,t}
and every regressor F_{t-1} are available and t-1 -> t is one bar on the
grid; nothing is filled.

Diagnostics:
  * eigenvalue moduli of A (persistence; > 1 explosive);
  * one-step out-of-sample R^2 per factor for the full A, a diagonal A (own
    lag only, i.e. no cross-factor propagation) and the training mean;
  * stability: A fitted on each half of the training rows (Frobenius
    distance, eigenvalue moduli).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _pairs(F: np.ndarray, contiguous: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices t (with t-1) usable for every equation: regressors complete at t-1."""
    t = np.arange(1, len(F))
    ok = contiguous[1:] & rows[1:] & np.all(np.isfinite(F[:-1]), axis=1)
    return t[ok], t[ok] - 1


def fit_var1(F: np.ndarray, contiguous: np.ndarray, rows: np.ndarray, ridge: float = 1e-3,
             diagonal: bool = False) -> dict:
    m = F.shape[1]
    t, s = _pairs(F, contiguous, rows)
    c = np.zeros(m)
    A = np.zeros((m, m))
    resid_var = np.full(m, np.nan)
    n_rows = np.zeros(m, dtype=int)
    for j in range(m):
        ok = np.isfinite(F[t, j])
        tj, sj = t[ok], s[ok]
        if len(tj) < 30:
            continue
        X = F[sj] if not diagonal else F[sj][:, [j]]
        X = np.column_stack([np.ones(len(tj)), X])
        y = F[tj, j]
        G = X.T @ X + ridge * len(tj) * np.diag(np.r_[0.0, np.ones(X.shape[1] - 1)])
        b = np.linalg.solve(G, X.T @ y)
        c[j] = b[0]
        if diagonal:
            A[j, j] = b[1]
        else:
            A[j] = b[1:]
        resid_var[j] = float(np.var(y - X @ b))
        n_rows[j] = len(tj)
    return {"c": c, "A": A, "resid_var": resid_var, "n_rows": n_rows}


def oos_r2(model: dict, F: np.ndarray, contiguous: np.ndarray, rows: np.ndarray, train_mean: np.ndarray) -> np.ndarray:
    t, s = _pairs(F, contiguous, rows)
    out = np.full(F.shape[1], np.nan)
    for j in range(F.shape[1]):
        ok = np.isfinite(F[t, j])
        if ok.sum() < 30:
            continue
        y = F[t[ok], j]
        pred = model["c"][j] + F[s[ok]] @ model["A"][j]
        out[j] = 1.0 - np.sum((y - pred) ** 2) / np.sum((y - train_mean[j]) ** 2)
    return out


def var1_report(F: pd.DataFrame, contiguous: np.ndarray, train: np.ndarray, test: np.ndarray) -> dict:
    """Fit on train, evaluate on test, and check stability across training halves."""
    X = F.to_numpy(float)
    names = list(F.columns)
    full = fit_var1(X, contiguous, train)
    diag = fit_var1(X, contiguous, train, diagonal=True)
    mean = np.nanmean(np.where(train[:, None], X, np.nan), axis=0)
    tri = np.flatnonzero(train)
    h = len(tri) // 2
    first = np.zeros(len(X), bool); first[tri[:h]] = True
    second = np.zeros(len(X), bool); second[tri[h:]] = True
    a1, a2 = fit_var1(X, contiguous, first)["A"], fit_var1(X, contiguous, second)["A"]
    eig = lambda A: np.sort(np.abs(np.linalg.eigvals(A)))[::-1]
    t_test, _ = _pairs(X, contiguous, test)
    return {
        "names": names,
        "A": pd.DataFrame(full["A"], index=names, columns=names),
        "c": pd.Series(full["c"], index=names),
        "eig_moduli": eig(full["A"]),
        "eig_moduli_half1": eig(a1), "eig_moduli_half2": eig(a2),
        "A_half_distance": float(np.linalg.norm(a1 - a2) / max(np.linalg.norm(full["A"]), 1e-12)),
        "r2": pd.DataFrame({"full_A": oos_r2(full, X, contiguous, test, mean),
                            "diagonal_A": oos_r2(diag, X, contiguous, test, mean)}, index=names),
        "train_rows": pd.Series(full["n_rows"], index=names),
        "test_rows": len(t_test),
    }
