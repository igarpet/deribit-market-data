"""
Multi-Scale HMM branch, MSH 01: do latent states found on 1h / 4h / 1D bars, built from
1-minute data, identify momentum or reversal regimes that predict the next bar's return
on Binance USD-M perpetuals, net of costs?

Pre-registration: discussions/Multi_Scale_HMM/001_claude_branch_preregistration.md

Pipeline (identical for every scale and for every null draw)
------------------------------------------------------------
1. Bars. UTC-aligned bars of 60, 240 or 1440 minutes built from Binance USD-M 1m klines.
   A bar is valid if >= 95% of its minutes are present. Bar label = bar END.
   Minute log return r_i = log C_i - log C_{i-1} (the first minute of a bar is measured from
   the previous bar's last close), so sum_i r_i = log C_t - log C_{t-1}.
2. Intrabar emissions (all from the bar's own 1-minute data):
       z      r_t / sigma_{t-1}                                    standardized bar return
       lrv    log(RV_t / sigma_{t-1}^2),  RV = sum r_i^2           relative realized variance
       lsig   log sigma_t                                          volatility level (persistent regimes)
       semi   (RS+ - RS-) / RV                                     up/down semivariance asymmetry
       er     r_t / sum |r_i|                                      signed efficiency (trend vs chop)
       clv    (2C - H - L) / (H - L)                               close location (wick structure)
       wick   1 - |C - O| / (H - L)                                wick share of the bar range
       jump   max(0, 1 - BV/RV),  BV = (pi/2) sum |r_i||r_{i-1}|   jump share of variance
       flow   sum(2 TBQ - QV) / sum QV                             signed taker imbalance
       lvol   log(QV_t / EWMA(QV)_{t-1})                           relative activity
   sigma_t^2 = EWMA of RV with half-life HALFLIFE[scale] bars (including bar t).
   Features are standardized per asset with the fit window's mean and sd, clipped at +-8.
3. Model. Multivariate Student-t HMM, full covariance, one shared nu, pooled over assets
   (one sequence per contiguous run of bars per asset). K and nu chosen by one-step
   predictive log-likelihood per bar on VALID (params fitted on TRAIN). The selected model
   is refitted on TRAIN+VALID and frozen.
4. Forecast. pi_t = filtered state probabilities at bar close t (causal forward filter).
   Per state, weighted least squares on the fit window (weights pi_tk):
       y_{t+1} = r_{t+1} / sigma_t = a_k + b_k z_t + e        (y winsorized at +-6 for fitting)
   b_k > 0: momentum state, b_k < 0: reversal state, a_k: drift.
   Forecast  m_{t+1} = sigma_t * sum_k pi_tk (a_k + b_k z_t)   (log-return units).
5. Trading on TEST (equal weight across assets, 1 unit notional per asset):
   always-in   s_t = sign(m_{t+1})                              -> statistic S (gross); must also beat
               the drift-only position sign(mean r on TRAIN+VALID) held throughout TEST
   net rule    pos_t = sign(m_{t+1}) * 1{|m_{t+1}| >= TAU}      -> economics E
               cost COST_SIDE per unit of |pos_t - pos_{t-1}|; flat at data gaps;
               funding: long pays the Binance funding rate at each funding time in (t, t+1].
6. Null. Every 1-minute return gets a random sign, shared across assets at the same minute
   (keeps |r| and cross-asset volatility co-movement, destroys all directional information);
   minute open/high/low are mirrored with it and taker buy volume becomes QV - TBQ. Bars,
   features, HMM fit (K, nu fixed at the real-data choice), regressions and forecasts are all
   recomputed. p = (1 + #{null >= real}) / (N + 1).
"""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

VERSION = "multiscale_hmm 1.0.0"

# ----------------------------------------------------------------------------- registration
ASSETS = ("BTC", "ETH", "SOL", "XRP")
SCALES = {"1h": 60, "4h": 240, "1D": 1440}
HALFLIFE = {"1h": 24, "4h": 18, "1D": 10}          # EWMA half-life in bars (1 day, 3 days, 10 days)
TSMOM_LOOKBACK = {"1h": 24, "4h": 30, "1D": 20}    # baseline only (1 day, 5 days, 20 days)
K_GRID = {"1h": (1, 2, 3, 4, 5, 6), "4h": (1, 2, 3, 4, 5, 6), "1D": (1, 2, 3, 4)}
NU_GRID = (4.0, 7.0, 15.0, 50.0)
N_NULL = {"1h": 99, "4h": 199, "1D": 199}
MIN_COVERAGE = 0.95
FEATURES = ("z", "lrv", "lsig", "semi", "er", "clv", "wick", "jump", "flow", "lvol")

TRAIN_START = pd.Timestamp("2024-10-01", tz="UTC")
VALID_START = pd.Timestamp("2025-09-01", tz="UTC")
TEST_START = pd.Timestamp("2026-02-01", tz="UTC")
TEST_END = pd.Timestamp("2026-10-18", tz="UTC")     # exclusive; later bars are the prospective window
PROSPECTIVE_START = TEST_END
PROSPECTIVE_MIN_DAYS = {"1h": 60, "4h": 90, "1D": 180}

COST_SIDE = 0.0006                                   # 5 bp taker + 1 bp slippage per side (as BT 01)
TAU = 2 * COST_SIDE                                  # trade only if |forecast| >= round-trip cost
ALPHA_S = 0.05 / 3                                   # Bonferroni over the three scales
MIN_ASSETS = 3                                       # of 4 must agree in sign
BOOT_B = 2000
BOOT_BLOCK_DAYS = {"1h": 1, "4h": 1, "1D": 7}
SEED = 20261017

EM_MAX_ITER = 300
EM_TOL = 1e-6
RESTARTS = 6
NULL_RESTARTS = 3
RIDGE = 1e-3
Y_CLIP = 6.0
X_CLIP = 8.0


@dataclass(frozen=True)
class Spec:
    assets: tuple = ASSETS
    scales: tuple = tuple(SCALES)
    restarts: int = RESTARTS
    null_restarts: int = NULL_RESTARTS
    n_null: dict = field(default_factory=lambda: dict(N_NULL))
    seed: int = SEED


def registration(spec: Spec = Spec()) -> dict:
    return {
        "version": VERSION, "assets": list(spec.assets), "scales_min": {s: SCALES[s] for s in spec.scales},
        "features": list(FEATURES), "halflife_bars": HALFLIFE, "k_grid": {k: list(v) for k, v in K_GRID.items()},
        "nu_grid": list(NU_GRID), "train": [str(TRAIN_START), str(VALID_START)],
        "valid": [str(VALID_START), str(TEST_START)], "test": [str(TEST_START), str(TEST_END)],
        "prospective_from": str(PROSPECTIVE_START), "prospective_min_days": PROSPECTIVE_MIN_DAYS,
        "cost_per_side": COST_SIDE, "tau": TAU, "alpha_S": ALPHA_S, "min_assets": MIN_ASSETS,
        "n_null": spec.n_null, "boot_B": BOOT_B, "boot_block_days": BOOT_BLOCK_DAYS,
        "restarts": spec.restarts, "null_restarts": spec.null_restarts, "seed": spec.seed,
        "source": "Binance USD-M perpetual 1m klines + funding (data.binance.vision); not Deribit execution",
    }


# ----------------------------------------------------------------------------- data
KLINE_COLS = ["open_time", "open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"]


def load_um_1m(path) -> pd.DataFrame:
    d = pd.read_parquet(path, columns=KLINE_COLS)
    return clean_1m(d)


def clean_1m(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    for c in KLINE_COLS[1:]:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
    d = d.dropna(subset=["open", "high", "low", "close"])
    d = d[(d["low"] > 0) & (d["high"] >= d["low"])]
    d = d.drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)
    d["quote_volume"] = d["quote_volume"].fillna(0.0).clip(lower=0.0)
    d["taker_buy_quote_volume"] = np.minimum(d["taker_buy_quote_volume"].fillna(0.0).clip(lower=0.0),
                                             d["quote_volume"])
    return d


def load_funding(path) -> pd.Series:
    """Binance fundingRate files: calc_time (ms), last_funding_rate. Index = funding time (UTC)."""
    f = pd.read_parquet(path)
    t = pd.to_datetime(pd.to_numeric(f["calc_time"], errors="coerce"), unit="ms", utc=True)
    s = pd.Series(pd.to_numeric(f["last_funding_rate"], errors="coerce").to_numpy(float), index=t)
    s = s[s.index.notna()].dropna()
    return s[~s.index.duplicated(keep="last")].sort_index()


def sign_flip(minutes: dict, seed: int) -> dict:
    """Null: one random sign per minute timestamp, shared across assets."""
    allt = np.unique(np.concatenate([m["open_time"].to_numpy("datetime64[ns]").astype(np.int64)
                                     for m in minutes.values()]))
    rng = np.random.default_rng(seed)
    sg = np.where(rng.random(len(allt)) < 0.5, -1.0, 1.0)
    out = {}
    for a, m in minutes.items():
        t = m["open_time"].to_numpy("datetime64[ns]").astype(np.int64)
        s = sg[np.searchsorted(allt, t)]
        lc = np.log(m["close"].to_numpy(float))
        lo, lh, ll = (np.log(m[c].to_numpy(float)) for c in ("open", "high", "low"))
        prev = np.empty_like(lc)
        prev[0] = lo[0]
        prev[1:] = lc[:-1]
        r, ro, rh, rl = lc - prev, lo - prev, lh - prev, ll - prev
        newc = np.cumsum(s * r) + prev[0]
        newprev = np.empty_like(newc)
        newprev[0] = prev[0]
        newprev[1:] = newc[:-1]
        hi_rel = np.where(s > 0, rh, -rl)
        lo_rel = np.where(s > 0, rl, -rh)
        q = m["quote_volume"].to_numpy(float)
        tb = m["taker_buy_quote_volume"].to_numpy(float)
        out[a] = pd.DataFrame({
            "open_time": m["open_time"].to_numpy(), "open": np.exp(newprev + s * ro),
            "high": np.exp(newprev + hi_rel), "low": np.exp(newprev + lo_rel), "close": np.exp(newc),
            "quote_volume": q, "taker_buy_quote_volume": np.where(s > 0, tb, q - tb)})
    return out


# ----------------------------------------------------------------------------- bars and features
def build_bars(m: pd.DataFrame, scale_min: int, min_coverage: float = MIN_COVERAGE) -> pd.DataFrame:
    """Valid UTC-aligned bars from 1m klines; bar_end = label. Includes minute-level aggregates."""
    t = m["open_time"].to_numpy("datetime64[ns]").astype(np.int64)
    lc = np.log(m["close"].to_numpy(float))
    prev = np.empty_like(lc)
    prev[0] = np.log(float(m["open"].iloc[0]))
    prev[1:] = lc[:-1]
    r = lc - prev
    ns = np.int64(scale_min) * 60 * 10**9
    b = t // ns
    first = np.r_[True, b[1:] != b[:-1]]
    ar = np.abs(r)
    bvp = np.zeros_like(r)
    bvp[1:] = np.where(first[1:], 0.0, ar[1:] * ar[:-1])          # |r_i||r_{i-1}| within bar
    g = pd.DataFrame({"b": b, "o": m["open"].to_numpy(float), "h": m["high"].to_numpy(float),
                      "l": m["low"].to_numpy(float), "c": m["close"].to_numpy(float), "r": r,
                      "r2": r * r, "rp2": np.where(r > 0, r * r, 0.0), "rn2": np.where(r < 0, r * r, 0.0),
                      "ar": ar, "bvp": bvp, "qv": m["quote_volume"].to_numpy(float),
                      "tb": m["taker_buy_quote_volume"].to_numpy(float)})
    agg = g.groupby("b", sort=True).agg(
        O=("o", "first"), H=("h", "max"), L=("l", "min"), C=("c", "last"), n=("r", "size"),
        RV=("r2", "sum"), RSP=("rp2", "sum"), RSN=("rn2", "sum"), SABS=("ar", "sum"), BVS=("bvp", "sum"),
        QV=("qv", "sum"), TBQ=("tb", "sum"))
    agg = agg[agg["n"] >= math.ceil(min_coverage * scale_min)]
    out = agg.reset_index()
    out["bar_end"] = pd.to_datetime((out["b"] + 1) * ns, utc=True)
    return out.drop(columns="b")


def bar_features(bars: pd.DataFrame, scale: str) -> pd.DataFrame:
    """Raw (unstandardized) features, targets and segment ids. Causal: row t uses bars <= t,
    except the explicit targets r_next / y_next / fund_next."""
    d = bars.sort_values("bar_end").reset_index(drop=True).copy()
    step = pd.Timedelta(minutes=SCALES[scale])
    adj = d["bar_end"].diff() == step
    d["r"] = np.where(adj, np.log(d["C"]) - np.log(d["C"].shift(1)), np.nan)
    hl = HALFLIFE[scale]
    alpha = 1.0 - 0.5 ** (1.0 / hl)
    s2 = d["RV"].ewm(alpha=alpha, adjust=False).mean()
    qv = d["QV"].clip(lower=1e-12).ewm(alpha=alpha, adjust=False).mean()
    d["sig"] = np.sqrt(s2)
    sp = d["sig"].shift(1)
    rng = (d["H"] - d["L"]).to_numpy()
    rv = d["RV"].clip(lower=1e-18)
    d["z"] = d["r"] / sp
    d["lrv"] = np.log(rv / sp ** 2)
    d["lsig"] = np.log(d["sig"].clip(lower=1e-12))
    d["semi"] = (d["RSP"] - d["RSN"]) / rv
    d["er"] = d["r"] / d["SABS"].clip(lower=1e-18)
    with np.errstate(invalid="ignore", divide="ignore"):
        d["clv"] = np.where(rng > 0, (2 * d["C"] - d["H"] - d["L"]) / rng, 0.0)
        d["wick"] = np.where(rng > 0, 1.0 - np.abs(d["C"] - d["O"]) / rng, 1.0)
    d["jump"] = np.clip(1.0 - (math.pi / 2) * d["BVS"] / rv, 0.0, 1.0)
    d["flow"] = np.where(d["QV"] > 0, (2 * d["TBQ"] - d["QV"]) / d["QV"].clip(lower=1e-12), 0.0)
    d["lvol"] = np.log(d["QV"].clip(lower=1e-12) / qv.shift(1))
    warm = 3 * hl
    d["valid"] = d[list(FEATURES)].notna().all(axis=1) & (np.arange(len(d)) >= warm)
    d["valid"] &= np.isfinite(d[list(FEATURES)].to_numpy(float)).all(axis=1)
    # segments: contiguous runs of valid, adjacent rows
    brk = (~d["valid"]) | (~adj)
    d["seg"] = np.cumsum(brk.to_numpy() | (~d["valid"].shift(1, fill_value=False)).to_numpy())
    nxt_adj = (d["bar_end"].shift(-1) - d["bar_end"]) == step
    d["r_next"] = np.where(nxt_adj, d["r"].shift(-1), np.nan)
    d["y_next"] = d["r_next"] / d["sig"]
    return d


def add_funding(d: pd.DataFrame, funding: pd.Series | None, scale: str) -> pd.DataFrame:
    """fund_next = sum of funding rates with timestamp in (bar_end_t, bar_end_t + scale]."""
    d = d.copy()
    if funding is None or len(funding) == 0:
        d["fund_next"] = np.nan
        return d
    step = pd.Timedelta(minutes=SCALES[scale])
    ft = funding.index.to_numpy("datetime64[ns]").astype(np.int64)
    cs = np.r_[0.0, np.cumsum(funding.to_numpy(float))]
    a = d["bar_end"].to_numpy("datetime64[ns]").astype(np.int64)
    b = (d["bar_end"] + step).to_numpy("datetime64[ns]").astype(np.int64)
    d["fund_next"] = cs[np.searchsorted(ft, b, side="right")] - cs[np.searchsorted(ft, a, side="right")]
    covered = (d["bar_end"] + step <= funding.index.max() + pd.Timedelta(hours=8))
    d.loc[~covered.to_numpy(), "fund_next"] = np.nan
    return d


def build_panel(minutes: dict, scale: str, funding: dict | None = None) -> dict:
    out = {}
    for a, m in minutes.items():
        f = bar_features(build_bars(m, SCALES[scale]), scale)
        out[a] = add_funding(f, (funding or {}).get(a), scale)
    return out


def window_mask(d: pd.DataFrame, start, end) -> np.ndarray:
    t = d["bar_end"]
    return (d["valid"] & (t > start) & (t <= end)).to_numpy()


def standardize(panel: dict, fit_start, fit_end) -> tuple[dict, dict]:
    """Per-asset z-scoring with the fit window's mean/sd; returns arrays X (all rows) per asset."""
    X, stats = {}, {}
    for a, d in panel.items():
        raw = d[list(FEATURES)].to_numpy(float)
        w = window_mask(d, fit_start, fit_end)
        mu = np.nanmean(raw[w], axis=0)
        sd = np.nanstd(raw[w], axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        X[a] = np.clip((raw - mu) / sd, -X_CLIP, X_CLIP)
        stats[a] = {"mean": mu.tolist(), "sd": sd.tolist()}
    return X, stats


def stack(panel: dict, X: dict, mask_fn) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Concatenate rows selected by mask_fn(d) into one array with segment start/end indices."""
    xs, starts, ends, keys = [], [], [], []
    n = 0
    for a, d in panel.items():
        msk = mask_fn(d) & d["valid"].to_numpy()
        idx = np.flatnonzero(msk)
        if len(idx) == 0:
            continue
        seg = d["seg"].to_numpy()[idx]
        cut = np.r_[True, (np.diff(idx) != 1) | (np.diff(seg) != 0)]
        st = np.flatnonzero(cut)
        en = np.r_[st[1:], len(idx)]
        xs.append(X[a][idx])
        starts.extend((st + n).tolist())
        ends.extend((en + n).tolist())
        keys.extend([(a, i) for i in idx])
        n += len(idx)
    if not xs:
        return np.zeros((0, len(FEATURES))), np.zeros(0, np.int64), np.zeros(0, np.int64), []
    return np.vstack(xs), np.array(starts, np.int64), np.array(ends, np.int64), keys


# ----------------------------------------------------------------------------- Student-t HMM
try:  # numba on the PC; pure Python fallback keeps tests runnable anywhere
    from numba import njit as _njit
    _jit = _njit(cache=True)
except Exception:  # pragma: no cover
    def _jit(f):
        return f


@_jit
def _fb_kernel(logb, A, pi0, starts, ends, want_xi):
    T, K = logb.shape
    alpha = np.zeros((T, K))
    beta = np.ones((T, K))
    c = np.zeros(T)
    mx = np.zeros(T)
    b = np.zeros((T, K))
    for t in range(T):
        m = logb[t, 0]
        for k in range(1, K):
            if logb[t, k] > m:
                m = logb[t, k]
        mx[t] = m
        for k in range(K):
            b[t, k] = math.exp(logb[t, k] - m)
    xi = np.zeros((K, K))
    ll = 0.0
    for s in range(len(starts)):
        t0 = starts[s]
        t1 = ends[s]
        tot = 0.0
        for k in range(K):
            alpha[t0, k] = pi0[k] * b[t0, k]
            tot += alpha[t0, k]
        if tot <= 0:
            tot = 1e-300
        c[t0] = tot
        for k in range(K):
            alpha[t0, k] /= tot
        ll += math.log(tot) + mx[t0]
        for t in range(t0 + 1, t1):
            tot = 0.0
            for j in range(K):
                acc = 0.0
                for i in range(K):
                    acc += alpha[t - 1, i] * A[i, j]
                alpha[t, j] = acc * b[t, j]
                tot += alpha[t, j]
            if tot <= 0:
                tot = 1e-300
            c[t] = tot
            for j in range(K):
                alpha[t, j] /= tot
            ll += math.log(tot) + mx[t]
        if want_xi:
            for k in range(K):
                beta[t1 - 1, k] = 1.0
            for t in range(t1 - 2, t0 - 1, -1):
                for i in range(K):
                    acc = 0.0
                    for j in range(K):
                        acc += A[i, j] * b[t + 1, j] * beta[t + 1, j]
                    beta[t, i] = acc / c[t + 1]
                for i in range(K):
                    for j in range(K):
                        xi[i, j] += alpha[t, i] * A[i, j] * b[t + 1, j] * beta[t + 1, j] / c[t + 1]
    return alpha, beta, xi, ll, c, mx


def t_logpdf(X: np.ndarray, mu: np.ndarray, cov: np.ndarray, nu: float) -> tuple[np.ndarray, np.ndarray]:
    """log density (T,K) and Mahalanobis distances (T,K) of a multivariate Student-t."""
    T, d = X.shape
    K = mu.shape[0]
    lp = np.empty((T, K))
    dm = np.empty((T, K))
    for k in range(K):
        L = np.linalg.cholesky(cov[k])
        z = np.linalg.solve(L, (X - mu[k]).T)
        dk = np.sum(z * z, axis=0)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        dm[:, k] = dk
        lp[:, k] = (gammaln((nu + d) / 2) - gammaln(nu / 2) - 0.5 * d * math.log(nu * math.pi)
                    - 0.5 * logdet - 0.5 * (nu + d) * np.log1p(dk / nu))
    return lp, dm


@dataclass
class THMM:
    K: int
    nu: float
    mu: np.ndarray
    cov: np.ndarray
    A: np.ndarray
    pi0: np.ndarray
    loglik: float = float("nan")
    n_iter: int = 0

    def to_dict(self) -> dict:
        return {"K": self.K, "nu": self.nu, "mu": self.mu.tolist(), "cov": self.cov.tolist(),
                "A": self.A.tolist(), "pi0": self.pi0.tolist(), "loglik": self.loglik, "n_iter": self.n_iter}

    @staticmethod
    def from_dict(d: dict) -> "THMM":
        return THMM(int(d["K"]), float(d["nu"]), np.array(d["mu"]), np.array(d["cov"]), np.array(d["A"]),
                    np.array(d["pi0"]), float(d.get("loglik", np.nan)), int(d.get("n_iter", 0)))


def _init(X: np.ndarray, K: int, rng: np.random.Generator) -> THMM:
    T, d = X.shape
    cen = [X[rng.integers(T)]]
    for _ in range(1, K):                                       # k-means++ seeding
        d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in cen], axis=0)
        p = d2 / d2.sum() if d2.sum() > 0 else None
        cen.append(X[rng.choice(T, p=p)])
    cen = np.array(cen)
    for _ in range(10):
        lab = np.argmin(((X[:, None, :] - cen[None]) ** 2).sum(-1), axis=1)
        for k in range(K):
            if np.any(lab == k):
                cen[k] = X[lab == k].mean(0)
    base = np.cov(X.T) + RIDGE * np.eye(d)
    cov = np.array([base.copy() for _ in range(K)])
    A = np.full((K, K), 0.1 / max(K - 1, 1)) if K > 1 else np.ones((1, 1))
    if K > 1:
        np.fill_diagonal(A, 0.9)
    return THMM(K, 0.0, cen, cov, A, np.full(K, 1.0 / K))


def fit_thmm(X, starts, ends, K: int, nu: float, restarts: int, seed: int,
             max_iter: int = EM_MAX_ITER, tol: float = EM_TOL) -> THMM:
    best = None
    d = X.shape[1]
    for r in range(restarts):
        rng = np.random.default_rng(seed + 1000 * r + K)
        m = _init(X, K, rng)
        m.nu = nu
        prev = -np.inf
        for it in range(max_iter):
            lp, dm = t_logpdf(X, m.mu, m.cov, nu)
            alpha, beta, xi, ll, _, _ = _fb_kernel(lp, m.A, m.pi0, starts, ends, True)
            g = alpha * beta
            g /= g.sum(1, keepdims=True)
            u = (nu + d) / (nu + dm)
            w = g * u
            for k in range(K):
                sw = w[:, k].sum()
                sg = g[:, k].sum()
                if sg < 1e-8:
                    continue
                mk = (w[:, k:k + 1] * X).sum(0) / sw
                Z = X - mk
                m.mu[k] = mk
                m.cov[k] = (w[:, k:k + 1] * Z).T @ Z / sg + RIDGE * np.eye(d)
            if K > 1:
                A = xi + 1e-6
                m.A = A / A.sum(1, keepdims=True)
            p0 = g[starts].sum(0) + 1e-6
            m.pi0 = p0 / p0.sum()
            m.n_iter = it + 1
            if abs(ll - prev) < tol * max(1.0, abs(ll)):
                prev = ll
                break
            prev = ll
        lp, _ = t_logpdf(X, m.mu, m.cov, nu)
        m.loglik = float(_fb_kernel(lp, m.A, m.pi0, starts, ends, False)[3])
        if best is None or m.loglik > best.loglik:
            best = m
    return best


def filter_probs(m: THMM, X, starts, ends) -> tuple[np.ndarray, float]:
    """Causal filtered state probabilities and total one-step predictive log-likelihood."""
    lp, _ = t_logpdf(X, m.mu, m.cov, m.nu)
    alpha, _, _, ll, _, _ = _fb_kernel(lp, m.A, m.pi0, starts, ends, False)
    return alpha, float(ll)


def n_params(K: int, d: int) -> int:
    return K * d + K * d * (d + 1) // 2 + K * (K - 1) + (K - 1)


# ----------------------------------------------------------------------------- forecasts
def state_regression(pi: np.ndarray, z: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Per-state WLS of y on [1, z] with weights pi; HC0 standard errors."""
    yc = np.clip(y, -Y_CLIP, Y_CLIP)
    ok = np.isfinite(yc) & np.isfinite(z)
    rows = []
    Z = np.column_stack([np.ones(ok.sum()), z[ok]])
    for k in range(pi.shape[1]):
        w = pi[ok, k]
        W = Z * w[:, None]
        XtWX = Z.T @ W + 1e-9 * np.eye(2)
        beta = np.linalg.solve(XtWX, W.T @ yc[ok])
        e = yc[ok] - Z @ beta
        meat = (Z * (w * e)[:, None]).T @ (Z * (w * e)[:, None])
        inv = np.linalg.inv(XtWX)
        se = np.sqrt(np.maximum(np.diag(inv @ meat @ inv), 0.0))
        rows.append({"state": k, "a": beta[0], "b": beta[1], "se_a": se[0], "se_b": se[1], "weight": w.sum()})
    return pd.DataFrame(rows)


def forecast(pi: np.ndarray, z: np.ndarray, sig: np.ndarray, reg: pd.DataFrame) -> np.ndarray:
    a = reg["a"].to_numpy()
    b = reg["b"].to_numpy()
    return sig * ((pi * a[None, :]).sum(1) + (pi * b[None, :]).sum(1) * z)


def tsmom_forecast(d: pd.DataFrame, scale: str) -> np.ndarray:
    L = TSMOM_LOOKBACK[scale]
    r = d["r"].fillna(0.0)
    return r.rolling(L, min_periods=L).sum().to_numpy()


# ----------------------------------------------------------------------------- economics
def pnl_frame(d: pd.DataFrame, mhat: np.ndarray, sel: np.ndarray, tau: float = TAU,
              always_in: bool = False) -> pd.DataFrame:
    """Per-bar P&L for one asset on the selected rows (in time order). Forced flat across gaps."""
    idx = np.flatnonzero(sel)
    f = d.iloc[idx]
    m = mhat[idx]
    rn = f["r_next"].to_numpy(float)
    fund = f["fund_next"].to_numpy(float)
    seg = f["seg"].to_numpy()
    if always_in:
        pos = np.sign(m)
    else:
        pos = np.where(np.abs(m) >= tau, np.sign(m), 0.0)
    pos = np.where(np.isfinite(m) & np.isfinite(rn), pos, 0.0)
    # previous position: zero at the start and after a break (segment change or skipped row)
    contiguous = np.r_[False, (np.diff(idx) == 1) & (np.diff(seg) == 0)]
    prev = np.r_[0.0, pos[:-1]]
    prev = np.where(contiguous, prev, 0.0)
    # positions whose next row is not contiguous are closed at the end of the bar
    nxt_contig = np.r_[contiguous[1:], False]
    trade = np.abs(pos - prev) + np.where(nxt_contig, 0.0, np.abs(pos))
    simple = np.expm1(np.nan_to_num(rn))
    gross = pos * simple
    fcost = np.where(np.isfinite(fund), pos * fund, 0.0)
    out = pd.DataFrame({"bar_end": f["bar_end"].to_numpy(), "pos": pos, "gross": gross,
                        "cost": trade * COST_SIDE, "funding": fcost})
    out["net"] = out["gross"] - out["cost"] - out["funding"]
    out["funding_missing"] = ~np.isfinite(fund) & (pos != 0)
    return out


def portfolio(per_asset: dict) -> pd.DataFrame:
    """Equal-weight portfolio per bar: mean over assets with a row at that bar_end."""
    fr = []
    for a, p in per_asset.items():
        x = p[["bar_end", "gross", "cost", "funding", "net"]].copy()
        x["asset"] = a
        fr.append(x)
    z = pd.concat(fr)
    n = len(per_asset)
    g = z.groupby("bar_end")[["gross", "cost", "funding", "net"]].sum() / n
    return g.reset_index()


def daily(p: pd.DataFrame) -> pd.Series:
    t = pd.to_datetime(p["bar_end"], utc=True) - pd.Timedelta(seconds=1)
    return p.groupby(t.dt.floor("D"))["net"].sum()


def block_bootstrap_mean(x: pd.Series, block_days: int, B: int = BOOT_B, seed: int = SEED) -> tuple:
    v = x.to_numpy(float)
    if len(v) < 3:
        return float("nan"), float("nan")
    nb = max(1, int(math.ceil(len(v) / block_days)))
    starts = np.arange(0, len(v), block_days)
    rng = np.random.default_rng(seed)
    means = np.empty(B)
    for i in range(B):
        pick = rng.choice(starts, size=nb, replace=True)
        s = np.concatenate([v[j:j + block_days] for j in pick])[:len(v)]
        means[i] = s.mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# ----------------------------------------------------------------------------- one pipeline
def _fit_window(panel, X, start, end):
    return stack(panel, X, lambda d: window_mask(d, start, end))


def select_model(panel: dict, scale: str, restarts: int, seed: int) -> pd.DataFrame:
    """Fit on TRAIN, score one-step predictive log-likelihood per bar on VALID."""
    Xs, _ = standardize(panel, TRAIN_START, VALID_START)
    Xtr, s_tr, e_tr, _ = _fit_window(panel, Xs, TRAIN_START, VALID_START)
    Xva, s_va, e_va, _ = _fit_window(panel, Xs, VALID_START, TEST_START)
    rows = []
    for K in K_GRID[scale]:
        for nu in NU_GRID:
            m = fit_thmm(Xtr, s_tr, e_tr, K, nu, restarts, seed)
            _, llv = filter_probs(m, Xva, s_va, e_va)
            rows.append({"scale": scale, "K": K, "nu": nu, "train_ll_per_bar": m.loglik / len(Xtr),
                         "valid_ll_per_bar": llv / len(Xva), "n_train": len(Xtr), "n_valid": len(Xva),
                         "bic_train": -2 * m.loglik + n_params(K, Xtr.shape[1]) * math.log(len(Xtr))})
    return pd.DataFrame(rows)


def run_pipeline(panel: dict, scale: str, K: int, nu: float, restarts: int, seed: int,
                 keep_detail: bool = True) -> dict:
    """Fit on TRAIN+VALID (frozen), forecast TEST; returns statistics (and details if asked)."""
    Xs, stats = standardize(panel, TRAIN_START, TEST_START)
    Xf, s_f, e_f, keys_f = _fit_window(panel, Xs, TRAIN_START, TEST_START)
    m = fit_thmm(Xf, s_f, e_f, K, nu, restarts, seed)
    # causal filter over all valid rows (fit window + test + later), per asset
    Xall, s_all, e_all, keys_all = stack(panel, Xs, lambda d: d["valid"].to_numpy())
    pi_all, _ = filter_probs(m, Xall, s_all, e_all)
    pos_of = {k: i for i, k in enumerate(keys_all)}
    # regression on the fit window
    ka = np.array([a for a, _ in keys_f])
    ki = np.array([i for _, i in keys_f])
    zf, yf = np.empty(len(keys_f)), np.empty(len(keys_f))
    for a in np.unique(ka):
        sel = ka == a
        zf[sel] = panel[a]["z"].to_numpy(float)[ki[sel]]
        yf[sel] = panel[a]["y_next"].to_numpy(float)[ki[sel]]
    pif = pi_all[[pos_of[k] for k in keys_f]]
    in_fit = np.isfinite(yf)
    reg = state_regression(pif[in_fit], zf[in_fit], yf[in_fit])
    reg1 = state_regression(np.ones((in_fit.sum(), 1)), zf[in_fit], yf[in_fit])     # K = 1 baseline
    res = {"K": K, "nu": nu, "per_asset": {}, "model": m, "reg": reg, "reg_k1": reg1, "stats": stats}
    gross_ai, net_rule, base_ai, base_rule, mom_ai, drift_ai = {}, {}, {}, {}, {}, {}
    detail = {}
    for a, d in panel.items():
        n = len(d)
        pi = np.full((n, K), np.nan)
        rows = [i for (aa, i) in keys_all if aa == a]
        if rows:
            pi[rows] = pi_all[[pos_of[(a, i)] for i in rows]]
        z = d["z"].to_numpy(float)
        sig = d["sig"].to_numpy(float)
        mh = forecast(np.nan_to_num(pi), z, sig, reg)
        mh = np.where(np.isfinite(pi).all(1), mh, np.nan)
        mb = forecast(np.ones((n, 1)), z, sig, reg1)
        mm = tsmom_forecast(d, scale)
        fitw = window_mask(d, TRAIN_START, TEST_START)
        drift_sign = np.sign(np.nanmean(d["r_next"].to_numpy(float)[fitw])) if fitw.any() else 0.0
        md = np.full(n, drift_sign if drift_sign != 0 else 1.0)
        test = window_mask(d, TEST_START, TEST_END) & np.isfinite(d["r_next"].to_numpy(float))
        gross_ai[a] = pnl_frame(d, mh, test, always_in=True)
        net_rule[a] = pnl_frame(d, mh, test)
        base_ai[a] = pnl_frame(d, mb, test, always_in=True)
        base_rule[a] = pnl_frame(d, mb, test)
        mom_ai[a] = pnl_frame(d, mm, test, always_in=True)
        drift_ai[a] = pnl_frame(d, md, test, always_in=True)
        rn = d["r_next"].to_numpy(float)[test]
        mt = mh[test]
        ok = np.isfinite(mt) & np.isfinite(rn)
        r2 = 1 - np.sum((rn[ok] - mt[ok]) ** 2) / np.sum(rn[ok] ** 2) if ok.sum() else np.nan
        res["per_asset"][a] = {
            "n_test": int(ok.sum()), "gross_bp_per_bar": 1e4 * gross_ai[a]["gross"].mean(),
            "hit_rate": float(np.mean(np.sign(mt[ok]) == np.sign(rn[ok]))) if ok.sum() else np.nan,
            "r2_os": float(r2), "net_total_pct": 100 * net_rule[a]["net"].sum(),
            "net_trades": float((net_rule[a]["cost"] > 0).sum()),
            "k1_gross_bp_per_bar": 1e4 * base_ai[a]["gross"].mean(),
            "k1_net_total_pct": 100 * base_rule[a]["net"].sum(),
            "tsmom_gross_bp_per_bar": 1e4 * mom_ai[a]["gross"].mean()}
        if keep_detail:
            detail[a] = pd.DataFrame({"bar_end": d["bar_end"], "mhat": mh, "mhat_k1": mb, "z": z, "sig": sig,
                                      "r_next": d["r_next"], **{f"pi{k}": pi[:, k] for k in range(K)}})
    pg = portfolio(gross_ai)
    res["S_stat_bp"] = 1e4 * pg["gross"].mean() if len(pg) else np.nan
    res["S_assets_positive"] = int(sum(v["gross_bp_per_bar"] > 0 for v in res["per_asset"].values()))
    res["drift_S_stat_bp"] = 1e4 * portfolio(drift_ai)["gross"].mean() if len(pg) else np.nan
    if keep_detail:
        pn = portfolio(net_rule)
        dn = daily(pn)
        lo, hi = block_bootstrap_mean(dn, BOOT_BLOCK_DAYS[scale])
        res.update(net_daily=dn, net_mean_bp_day=1e4 * dn.mean(), net_lo_bp_day=1e4 * lo, net_hi_bp_day=1e4 * hi,
                   net_cost_bp_day=1e4 * daily(pn.assign(net=pn["cost"])).mean(),
                   net_funding_bp_day=1e4 * daily(pn.assign(net=pn["funding"])).mean(),
                   funding_missing=bool(any(p["funding_missing"].any() for p in net_rule.values())),
                   E_assets_positive=int(sum(net_rule[a]["net"].sum() > 0 for a in net_rule)),
                   k1_S_stat_bp=1e4 * portfolio(base_ai)["gross"].mean(),
                   k1_net_mean_bp_day=1e4 * daily(portfolio(base_rule)).mean(),
                   tsmom_S_stat_bp=1e4 * portfolio(mom_ai)["gross"].mean(),
                   detail=detail, gross_ai=gross_ai, net_rule=net_rule, base_ai=base_ai,
                   pi_fit=pif, keys_fit=keys_f)
        # incremental value of states over K = 1 (same net rule, same costs): paired day-block bootstrap
        pb = portfolio(base_rule).set_index("bar_end")["net"].reindex(pn["bar_end"]).fillna(0.0).to_numpy()
        diff = daily(pn.assign(net=pn["net"] - pb))
        lo2, hi2 = block_bootstrap_mean(diff, BOOT_BLOCK_DAYS[scale], seed=SEED + 7)
        res.update(incr_mean_bp_day=1e4 * diff.mean(), incr_lo_bp_day=1e4 * lo2, incr_hi_bp_day=1e4 * hi2)
    return res


def state_table(res: dict, panel: dict, scale: str) -> pd.DataFrame:
    """Descriptive: occupancy, duration and raw feature means per state (fit window, filtered weights)."""
    m, reg, pif, keys = res["model"], res["reg"], res["pi_fit"], res["keys_fit"]
    raw = np.empty((len(keys), len(FEATURES)))
    ka = np.array([a for a, _ in keys])
    ki = np.array([i for _, i in keys])
    for a in np.unique(ka):
        sel = ka == a
        raw[sel] = panel[a][list(FEATURES)].to_numpy(float)[ki[sel]]
    rows = []
    for k in range(m.K):
        w = pif[:, k]
        sw = w.sum()
        means = (w[:, None] * raw).sum(0) / sw if sw > 0 else np.full(len(FEATURES), np.nan)
        stay = m.A[k, k]
        r = reg.iloc[k]
        tb, ta = r["b"] / r["se_b"] if r["se_b"] > 0 else np.nan, r["a"] / r["se_a"] if r["se_a"] > 0 else np.nan
        label = "momentum" if tb > 2 else "reversal" if tb < -2 else "no-autocorr"
        if abs(ta) > 2:
            label += "+drift_up" if ta > 0 else "+drift_down"
        rows.append({"scale": scale, "state": k, "occupancy": sw / len(w),
                     "exp_duration_bars": 1 / max(1 - stay, 1e-9),
                     "exp_duration_hours": SCALES[scale] / 60 / max(1 - stay, 1e-9),
                     **{f"mean_{f}": v for f, v in zip(FEATURES, means)},
                     "a": r["a"], "se_a": r["se_a"], "b": r["b"], "se_b": r["se_b"], "label": label})
    return pd.DataFrame(rows)


_NULL_MINUTES: dict = {}


def _init_null_worker(minutes):
    global _NULL_MINUTES
    _NULL_MINUTES = minutes


def _null_job(args):
    scale, K, nu, restarts, seed = args
    flipped = sign_flip(_NULL_MINUTES, seed)
    pan = build_panel(flipped, scale, None)
    r = run_pipeline(pan, scale, K, nu, restarts, seed, keep_detail=False)
    return {"seed": seed, "S_stat_bp": r["S_stat_bp"], "S_assets_positive": r["S_assets_positive"]}


def run_null(minutes: dict, scale: str, K: int, nu: float, n: int, restarts: int, seed: int,
             n_jobs: int = 1, progress=None) -> pd.DataFrame:
    """Sign-flip null; the 1m data is sent once per worker process, each task carries only its seed."""
    args = [(scale, K, nu, restarts, seed + 10_000 + i) for i in range(n)]
    out = []
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_null_worker,
                                 initargs=(minutes,)) as ex:
            for i, r in enumerate(ex.map(_null_job, args)):
                out.append(r)
                if progress:
                    progress(i + 1, n)
    else:
        _init_null_worker(minutes)
        for i, a in enumerate(args):
            out.append(_null_job(a))
            if progress:
                progress(i + 1, n)
    return pd.DataFrame(out)


def verdict(res: dict, null: pd.DataFrame) -> dict:
    s = res["S_stat_bp"]
    p = (1 + int((null["S_stat_bp"] >= s).sum())) / (len(null) + 1)
    S = bool(p <= ALPHA_S and res["S_assets_positive"] >= MIN_ASSETS and s > 0 and s > res["drift_S_stat_bp"])
    E = bool(res["net_lo_bp_day"] > 0 and res["E_assets_positive"] >= MIN_ASSETS and not res["funding_missing"])
    return {"S_p": p, "S_beats_drift": bool(s > res["drift_S_stat_bp"]), "S_null_median_bp": float(null["S_stat_bp"].median()), "S_pass": S,
            "S_verdict": "SKILL" if S else "NO_SKILL",
            "E_pass": E, "E_verdict": "NET_EDGE" if E else "NO_NET_EDGE",
            "states_add_value": bool(res["incr_lo_bp_day"] > 0),
            "scale_verdict": "CANDIDATE_FOR_PROSPECTIVE" if (S and E) else "NO_TRADEABLE_MOMENTUM_STATES"}


# ----------------------------------------------------------------------------- synthetic data
def synthetic_minutes(n_days: int = 760, seed: int = 0, start: str = "2024-10-01",
                      planted: float = 0.0, assets=ASSETS) -> dict:
    """1m klines for several assets with a common 2-state volatility process (4h persistence).
    `planted` > 0 adds hourly momentum in the high-volatility state: each hour's drift is
    planted * (previous hour's return) spread evenly over the hour's minutes."""
    rng = np.random.default_rng(seed)
    n = n_days * 1440
    t = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    hours = n // 60
    st = np.zeros(hours, int)
    for h in range(1, hours):
        st[h] = st[h - 1] if rng.random() < 0.97 else 1 - st[h - 1]
    vol_h = np.where(st == 1, 2.0, 1.0)
    common = rng.standard_t(5, size=n) * 0.6
    out = {}
    for j, a in enumerate(assets):
        base = 4e-4 * (1 + 0.3 * j)
        eps = (0.8 * common + 0.6 * rng.standard_t(5, size=n)) * base * np.repeat(vol_h, 60)[:n]
        r = eps.copy()
        if planted:
            for h in range(1, hours):
                if st[h] == 1:
                    prev = r[(h - 1) * 60:h * 60].sum()
                    r[h * 60:(h + 1) * 60] += planted * prev / 60.0
        lc = np.log(100.0 * (1 + j)) + np.cumsum(r)
        prev = np.r_[lc[0], lc[:-1]]
        spread = np.abs(rng.normal(0, base * 0.5, n))
        hi = np.maximum(lc, prev) + spread
        lo = np.minimum(lc, prev) - spread
        qv = rng.lognormal(10, 0.5, n) * np.repeat(vol_h, 60)[:n]
        frac = np.clip(0.5 + 20 * r + rng.normal(0, 0.05, n), 0.0, 1.0)
        out[a] = pd.DataFrame({"open_time": t, "open": np.exp(prev), "high": np.exp(hi), "low": np.exp(lo),
                               "close": np.exp(lc), "quote_volume": qv, "taker_buy_quote_volume": qv * frac})
    return out


def synthetic_funding(minutes: dict, rate: float = 1e-4) -> dict:
    out = {}
    for a, m in minutes.items():
        t = pd.date_range(m["open_time"].iloc[0].ceil("8h"), m["open_time"].iloc[-1], freq="8h")
        out[a] = pd.Series(rate, index=t)
    return out


def selftest() -> str:
    """Forward-backward against brute-force enumeration on a tiny problem."""
    rng = np.random.default_rng(0)
    T, K = 5, 2
    logb = rng.normal(size=(T, K))
    A = np.array([[0.8, 0.2], [0.3, 0.7]])
    pi0 = np.array([0.6, 0.4])
    _, _, _, ll, _, _ = _fb_kernel(logb, A, pi0, np.array([0]), np.array([T]), True)
    tot = 0.0
    for path in np.ndindex(*(K,) * T):
        p = pi0[path[0]] * math.exp(logb[0, path[0]])
        for t in range(1, T):
            p *= A[path[t - 1], path[t]] * math.exp(logb[t, path[t]])
        tot += p
    err = abs(ll - math.log(tot))
    assert err < 1e-9, err
    return f"{VERSION}: forward-backward selftest ok (err {err:.1e})"


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return str(o)
