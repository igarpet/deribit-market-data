"""
Compiled forward-backward kernels for the countdown HSMM (numba).

Identical arithmetic to the numpy reference in engine.py; used automatically
when numba is installed (pip install numba). Without numba the engine falls
back to the numpy implementation, which is correct but 20-100x slower for
K >= 4.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:          # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def wrap(f):
            return f
        return wrap(args[0]) if args and callable(args[0]) else wrap


@njit(cache=True)
def forward_kernel(b, pmf, A, pi, starts, ends):
    T, K = b.shape
    D = pmf.shape[1]
    alpha = np.empty((T, K, D))
    c = np.empty(T)
    pred = np.empty((T, K))
    prior = np.empty((K, D))
    entry = np.empty(K)
    for s in range(starts.shape[0]):
        a = starts[s]
        z = ends[s]
        for t in range(a, z):
            if t == a:
                for k in range(K):
                    for d in range(D):
                        prior[k, d] = pi[k] * pmf[k, d]
            else:
                for k in range(K):
                    acc = 0.0
                    for j in range(K):
                        acc += alpha[t - 1, j, 0] * A[j, k]
                    entry[k] = acc
                for k in range(K):
                    for d in range(D - 1):
                        prior[k, d] = alpha[t - 1, k, d + 1] + entry[k] * pmf[k, d]
                    prior[k, D - 1] = entry[k] * pmf[k, D - 1]
            tot_prior = 0.0
            for k in range(K):
                ps = 0.0
                for d in range(D):
                    ps += prior[k, d]
                pred[t, k] = ps
                tot_prior += ps
            ct = 0.0
            for k in range(K):
                pred[t, k] /= tot_prior
                for d in range(D):
                    v = prior[k, d] * b[t, k]
                    alpha[t, k, d] = v
                    ct += v
            c[t] = ct
            for k in range(K):
                for d in range(D):
                    alpha[t, k, d] /= ct
    return alpha, c, pred


@njit(cache=True)
def backward_kernel(b, pmf, A, c, starts, ends):
    T, K = b.shape
    D = pmf.shape[1]
    beta = np.empty((T, K, D))
    G = np.empty(K)
    for s in range(starts.shape[0]):
        a = starts[s]
        z = ends[s]
        for k in range(K):
            for d in range(D):
                beta[z - 1, k, d] = 1.0
        for t in range(z - 2, a - 1, -1):
            for j in range(K):
                acc = 0.0
                for d in range(D):
                    acc += pmf[j, d] * beta[t + 1, j, d]
                G[j] = b[t + 1, j] * acc / c[t + 1]
            for k in range(K):
                acc = 0.0
                for j in range(K):
                    acc += A[k, j] * G[j]
                beta[t, k, 0] = acc
                for d in range(1, D):
                    beta[t, k, d] = b[t + 1, k] * beta[t + 1, k, d - 1] / c[t + 1]
    return beta


@njit(cache=True)
def expectations_kernel(b, pmf, A, pi, c, alpha, beta, starts, ends):
    T, K = b.shape
    D = pmf.shape[1]
    gamma = np.zeros((T, K))
    eta = np.zeros((K, D))
    xi = np.zeros((K, K))
    pi_acc = np.zeros(K)
    Gj = np.empty(K)
    entry = np.empty(K)
    for s in range(starts.shape[0]):
        a = starts[s]
        z = ends[s]
        for t in range(a, z):
            tot = 0.0
            for k in range(K):
                acc = 0.0
                for d in range(D):
                    acc += alpha[t, k, d] * beta[t, k, d]
                gamma[t, k] = acc
                tot += acc
            for k in range(K):
                gamma[t, k] /= tot
            if t == a:
                for j in range(K):
                    for d in range(D):
                        e = pi[j] * pmf[j, d] * b[t, j] * beta[t, j, d] / c[t]
                        eta[j, d] += e
                        pi_acc[j] += e
            else:
                for j in range(K):
                    acc = 0.0
                    for k in range(K):
                        acc += alpha[t - 1, k, 0] * A[k, j]
                    entry[j] = acc
                    g = 0.0
                    for d in range(D):
                        gp = pmf[j, d] * b[t, j] * beta[t, j, d] / c[t]
                        eta[j, d] += entry[j] * gp
                        g += gp
                    Gj[j] = g
                for k in range(K):
                    ek = alpha[t - 1, k, 0]
                    for j in range(K):
                        xi[k, j] += ek * A[k, j] * Gj[j]
    return gamma, eta, xi, pi_acc
