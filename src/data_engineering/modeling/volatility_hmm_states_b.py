"""
Volatility Dynamics 06b: latent market-state HMM, repaired, with a benchmark.

Problems in Module 06 that 06b fixes
------------------------------------
1. Emissions: raw volume / trade count / book updates / spread / range were
   modelled as diagonal Gaussians and clipped at +/-6. 06b uses logs.
2. IV levels: three near-collinear IV levels (ATM, 25d put, 25d call)
   normalized by training-period medians drifted with the market, so the
   held-out state occupancy trended day by day (state 1: 94% -> 27%). 06b
   uses causal deviations from a trailing 24h median:
       atm_iv_dev = log ATM IV - median(log ATM IV over the previous 24h)
       skew_dev   = (IV25p - IV25c) - median(same over the previous 24h)
3. Normalization: exact 5m slots had ~12 training observations each (02b
   checks how much of that profile is noise). 06b uses asset x UTC hour.
4. Missing emissions broke sequences (median sequence 50 min, 47% of test
   rows dropped). 06b marginalizes missing dimensions in the emission
   likelihood (exact for a diagonal Gaussian); sequences break only on
   asset changes and timestamp gaps.
5. K was the edge of the grid under BIC. 06b selects K by validation
   log-likelihood on the last 20% of training days, then refits on all
   training days.
6. No benchmark. 06b compares, on the held-out period, logistic forecasts
   of a q99 onset within h minutes from
       M0 asset base rates
       M1 volatility benchmark: log RV over 15m and 1h, log range now and 15m
       M2 HMM filtered state probabilities
       M3 benchmark + HMM
   with day-block bootstrap intervals.
7. Duration: Module 06 compared filtered-argmax run lengths with the
   geometric mean 1/(1-A_kk). Filtered argmax runs are not draws from the
   state duration law (misclassification fragments runs). 06b simulates the
   fitted HMM with the real test sequence lengths and missingness masks,
   applies the same filter/argmax, and compares like with like.

Pre-registered decision (fixed before any 06b output exists)
------------------------------------------------------------
Primary horizon h = 60 min. The HMM is said to carry q99 information beyond
the volatility benchmark iff, on the held-out period,
    delta = logloss(M3) - logloss(M1) < 0
    AND the 95% day-block bootstrap upper bound of delta < 0
    AND >= 20 unique q99 onset timestamps fall in the held-out period.
Fewer than 20 -> INSUFFICIENT_DATA. Otherwise FAIL.
An HSMM is only worth building if the HMM passes AND the simulated-duration
check shows systematic misfit.

NumPy/Pandas only. Branch-internal reuse: volatility_vd_b_core (grid helpers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from data_engineering.modeling import volatility_vd_b_core as vb


VERSION = "0.1.0"
BARS_PER_DAY = 288
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_VALID_FRACTION_OF_TRAIN = 0.20
DEFAULT_K = (2, 3, 4, 5, 6, 7)
DEFAULT_N_INIT = 3
DEFAULT_MAX_ITER = 80
DEFAULT_TOL = 1e-5
DEFAULT_MIN_SEQUENCE = 12
DEFAULT_CLIP_Z = 8.0
HORIZONS_MIN = (30, 60, 120)
PRIMARY_HORIZON_MIN = 60
MIN_UNIQUE_TEST_EVENTS = 20

CORE_EMISSIONS = (
    "log_range", "log_volume", "log_trades", "log_book_updates",
    "log_spread", "atm_iv_dev", "skew_dev",
)
L2_EMISSIONS = (
    "log_obi_zero_cross_count", "log_obi_transition_count",
    "log_obi_zero_cross_rate", "log_obi_roughness",
)
REQUIRED_COLUMNS = (
    "intrabar_range_bps", "volume_usd", "trade_count", "book_update_count",
    "book_twa_spread_bps", "option_iv_atm_forward", "iv_put_25d", "iv_call_25d",
)
L2_COLUMNS = (
    "book_obi_20_raw_zero_cross_count", "book_obi_20_transition_count",
    "book_obi_20_raw_zero_cross_rate", "book_obi_20_path_roughness_per_min",
)
BENCHMARK_FEATURES = ("bm_lrv_15m", "bm_lrv_1h", "bm_lrange_now", "bm_lrange_15m")


# --------------------------------------------------------------------------
# Emissions
# --------------------------------------------------------------------------

def _num(frame: pd.DataFrame, c: str) -> np.ndarray:
    return pd.to_numeric(frame[c], errors="coerce").to_numpy(float)


def _log_pos(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(x) & (x > 0), np.log(x), np.nan)


def _log1p_nonneg(x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(x) & (x >= 0), np.log1p(x), np.nan)


def causal_rolling_median(grid: vb.Grid, values: np.ndarray, window: int = BARS_PER_DAY, min_periods: int = 144) -> np.ndarray:
    """Median of the previous ``window`` bars (current bar excluded), per asset."""
    d = grid.dense_values(values)
    out_d = np.full(len(d), np.nan)
    for a in range(len(grid.assets)):
        b = grid._base[a]
        seg = pd.Series(d[b:b + grid._span[a]])
        out_d[b:b + grid._span[a]] = seg.rolling(int(window), min_periods=int(min_periods)).median().shift(1).to_numpy()
    out = np.full(grid.n, np.nan)
    pos = grid.dense_pos(np.arange(grid.n))
    out[pos >= 0] = out_d[pos[pos >= 0]]
    return out


def build_emissions(grid: vb.Grid, *, include_l2: bool) -> pd.DataFrame:
    f = grid.frame
    e = pd.DataFrame(index=f.index)
    e["log_range"] = _log_pos(_num(f, "intrabar_range_bps"))
    e["log_volume"] = _log1p_nonneg(_num(f, "volume_usd"))
    e["log_trades"] = _log1p_nonneg(_num(f, "trade_count"))
    e["log_book_updates"] = _log1p_nonneg(_num(f, "book_update_count"))
    e["log_spread"] = _log_pos(_num(f, "book_twa_spread_bps"))
    liv = _log_pos(_num(f, "option_iv_atm_forward"))
    e["atm_iv_dev"] = liv - causal_rolling_median(grid, liv)
    skew = _num(f, "iv_put_25d") - _num(f, "iv_call_25d")
    e["skew_dev"] = skew - causal_rolling_median(grid, skew)
    if include_l2:
        e["log_obi_zero_cross_count"] = _log1p_nonneg(_num(f, "book_obi_20_raw_zero_cross_count"))
        e["log_obi_transition_count"] = _log1p_nonneg(_num(f, "book_obi_20_transition_count"))
        e["log_obi_zero_cross_rate"] = _log1p_nonneg(_num(f, "book_obi_20_raw_zero_cross_rate"))
        e["log_obi_roughness"] = _log1p_nonneg(_num(f, "book_obi_20_path_roughness_per_min"))
    return e


def benchmark_features(grid: vb.Grid) -> pd.DataFrame:
    """Volatility benchmark at origin t (bar t is known at t, so it is included)."""
    pos = grid.v[np.isfinite(grid.v) & (grid.v > 0)]
    eps = float(np.median(pos) * 1e-4) if len(pos) else 1e-12
    rng = _num(grid.frame, "intrabar_range_bps")
    rng = np.where(np.isfinite(rng) & (rng >= 0), rng, np.nan)
    rpos = rng[np.isfinite(rng) & (rng > 0)]
    reps = float(np.median(rpos) * 1e-3) if len(rpos) else 1e-6
    return pd.DataFrame({
        "bm_lrv_15m": np.log(vb.causal_window_mean(grid, grid.v, -2, 0, 2) + eps),
        "bm_lrv_1h": np.log(vb.causal_window_mean(grid, grid.v, -11, 0, 9) + eps),
        "bm_lrange_now": np.log(rng + reps),
        "bm_lrange_15m": np.log(vb.causal_window_mean(grid, rng, -2, 0, 2) + reps),
    })


# --------------------------------------------------------------------------
# Split and normalization
# --------------------------------------------------------------------------

@dataclass
class Split:
    test_start_ns: int
    valid_start_ns: int
    train_days: list
    valid_days: list
    test_days: list


def chronological_split(grid: vb.Grid, train_fraction: float = DEFAULT_TRAIN_FRACTION,
                        valid_fraction: float = DEFAULT_VALID_FRACTION_OF_TRAIN) -> Split:
    days = vb.valid_days(grid, min_coverage=0.5)
    if len(days) < 6:
        raise ValueError("Need at least six usable UTC days.")
    n_train = max(4, min(int(np.floor(len(days) * train_fraction)), len(days) - 2))
    n_valid = max(1, int(round(n_train * valid_fraction)))
    train_days = days[:n_train]
    test_days = days[n_train:]
    fit_days = train_days[: n_train - n_valid]
    valid_days = train_days[n_train - n_valid:]
    return Split(
        test_start_ns=int(test_days[0]) * BARS_PER_DAY * vb.BAR_NS,
        valid_start_ns=int(valid_days[0]) * BARS_PER_DAY * vb.BAR_NS,
        train_days=[int(d) for d in fit_days],
        valid_days=[int(d) for d in valid_days],
        test_days=[int(d) for d in test_days],
    )


def fit_hour_normalizer(grid: vb.Grid, E: pd.DataFrame, train_mask: np.ndarray, min_n: int = 30) -> pd.DataFrame:
    """Training-only robust location/scale per asset x UTC hour (asset fallback)."""
    hour = grid.slot // 12
    rows = []
    for col in E.columns:
        x = E[col].to_numpy(float)
        for a in range(len(grid.assets)):
            ma = (grid.asset_code == a) & train_mask & np.isfinite(x)
            xa = x[ma]
            if len(xa) < min_n:
                continue
            a_med = float(np.median(xa))
            q25, q75 = np.quantile(xa, [0.25, 0.75])
            a_sc = float((q75 - q25) / 1.349) or float(np.std(xa)) or 1.0
            for h in range(24):
                xh = x[ma & (hour == h)]
                if len(xh) >= min_n:
                    med = float(np.median(xh))
                    q25, q75 = np.quantile(xh, [0.25, 0.75])
                    sc = float((q75 - q25) / 1.349)
                    if not np.isfinite(sc) or sc <= 1e-9:
                        sc = a_sc
                    src = "asset_hour"
                else:
                    med, sc, src = a_med, a_sc, "asset_fallback"
                rows.append({"underlying": grid.assets[a], "hour": h, "emission": col,
                             "location": med, "scale": sc, "n": int(len(xh)), "source": src})
    return pd.DataFrame(rows)


def apply_normalizer(grid: vb.Grid, E: pd.DataFrame, params: pd.DataFrame, clip_z: float = DEFAULT_CLIP_Z) -> np.ndarray:
    hour = grid.slot // 12
    Z = np.full(E.shape, np.nan)
    a_idx = {a: i for i, a in enumerate(grid.assets)}
    for j, col in enumerate(E.columns):
        p = params[params.emission == col]
        loc = np.full((len(grid.assets), 24), np.nan)
        sc = np.full((len(grid.assets), 24), np.nan)
        for r in p.itertuples():
            loc[a_idx[r.underlying], r.hour] = r.location
            sc[a_idx[r.underlying], r.hour] = r.scale
        x = E[col].to_numpy(float)
        z = (x - loc[grid.asset_code, hour]) / sc[grid.asset_code, hour]
        Z[:, j] = np.clip(z, -clip_z, clip_z)
    return Z


def sequences(grid: vb.Grid, row_mask: np.ndarray, min_len: int = DEFAULT_MIN_SEQUENCE) -> list[np.ndarray]:
    """Contiguous exact-5m runs within ``row_mask``, per asset."""
    rows = np.flatnonzero(np.asarray(row_mask, bool))
    if not len(rows):
        return []
    rows = rows[np.lexsort((grid.bar[rows], grid.asset_code[rows]))]
    brk = np.ones(len(rows), bool)
    brk[1:] = (grid.asset_code[rows[1:]] != grid.asset_code[rows[:-1]]) | (grid.bar[rows[1:]] - grid.bar[rows[:-1]] != 1)
    starts = np.flatnonzero(brk)
    ends = np.append(starts[1:], len(rows))
    return [rows[s:e] for s, e in zip(starts, ends) if e - s >= int(min_len)]


# --------------------------------------------------------------------------
# HMM with missing-data marginalization (scaled forward-backward)
# --------------------------------------------------------------------------

def _logB(X: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """(T, K) log emission density over observed dims only (0 if none observed)."""
    M = np.isfinite(X)
    X0 = np.where(M, X, 0.0)
    var = np.maximum(variances, 1e-6)
    out = np.zeros((len(X), len(means)))
    for k in range(len(means)):
        q = (X0 - means[k]) ** 2 / var[k] + np.log(2 * np.pi * var[k])
        out[:, k] = -0.5 * np.sum(np.where(M, q, 0.0), axis=1)
    return out


def _fb(X, pi, A, means, variances, *, need_xi=True):
    lb = _logB(X, means, variances)
    mx = lb.max(axis=1, keepdims=True)
    B = np.exp(lb - mx)
    T, K = B.shape
    alpha = np.empty((T, K))
    c = np.empty(T)
    a = pi * B[0]
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
class HMM:
    K: int
    pi: np.ndarray
    A: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    loglik: float
    n_iter: int
    converged: bool
    seed: int

    def loglik_of(self, seqs: Sequence[np.ndarray]) -> float:
        return float(sum(_fb(X, self.pi, self.A, self.means, self.variances, need_xi=False)[0] for X in seqs))

    def filter(self, X: np.ndarray) -> np.ndarray:
        return _fb(X, self.pi, self.A, self.means, self.variances, need_xi=False)[1]

    def sample(self, T: int, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        s = np.empty(T, int)
        s[0] = rng.choice(self.K, p=self.pi)
        cum = np.cumsum(self.A, axis=1)
        u = rng.random(T)
        for t in range(1, T):
            s[t] = min(int(np.searchsorted(cum[s[t - 1]], u[t])), self.K - 1)
        X = self.means[s] + rng.normal(size=(T, self.means.shape[1])) * np.sqrt(self.variances[s])
        return np.where(mask, X, np.nan), s


def fit_hmm(seqs: Sequence[np.ndarray], K: int, *, seed: int, max_iter: int = DEFAULT_MAX_ITER,
            tol: float = DEFAULT_TOL, min_variance: float = 1e-2) -> HMM:
    rng = np.random.default_rng(int(seed))
    Xall = np.vstack(seqs)
    d = Xall.shape[1]
    complete = Xall[np.isfinite(Xall).all(axis=1)]
    pool = complete if len(complete) >= K * 20 else np.nan_to_num(Xall)
    means = pool[rng.choice(len(pool), size=K, replace=False)].copy()
    gv = np.nanvar(Xall, axis=0)
    variances = np.tile(np.where(np.isfinite(gv) & (gv > 1e-3), gv, 1.0), (K, 1))
    pi = np.full(K, 1.0 / K)
    A = rng.dirichlet(np.ones(K), size=K) + 4.0 * np.eye(K)
    A /= A.sum(axis=1, keepdims=True)
    prev = -np.inf
    converged = False
    it = 0
    ll = -np.inf
    for it in range(1, int(max_iter) + 1):
        s0 = np.zeros(K)
        tr = np.zeros((K, K))
        wsum = np.zeros((K, d))
        xsum = np.zeros((K, d))
        x2sum = np.zeros((K, d))
        ll = 0.0
        for X in seqs:
            l, _, g, xi = _fb(X, pi, A, means, variances)
            ll += l
            s0 += g[0]
            tr += xi
            M = np.isfinite(X).astype(float)
            X0 = np.nan_to_num(X)
            wsum += g.T @ M
            xsum += g.T @ X0
            x2sum += g.T @ (X0 ** 2)
        pi = (s0 + 1e-6) / (s0 + 1e-6).sum()
        A = (tr + 1e-6) / (tr + 1e-6).sum(axis=1, keepdims=True)
        w = np.maximum(wsum, 1e-8)
        means = xsum / w
        variances = np.maximum(x2sum / w - means ** 2, min_variance)
        if np.isfinite(prev) and abs(ll - prev) <= tol * (1 + abs(prev)):
            converged = True
            break
        prev = ll
    return HMM(K, pi, A, means, variances, float(ll), it, converged, int(seed))


def select_and_fit(fit_seqs, valid_seqs, train_seqs, k_values=DEFAULT_K, *, n_init=DEFAULT_N_INIT,
                   max_iter=DEFAULT_MAX_ITER, seed_base=6600, progress=False) -> tuple[pd.DataFrame, HMM, int]:
    rows = []
    n_fit = sum(map(len, fit_seqs))
    n_val = sum(map(len, valid_seqs))
    for K in k_values:
        fits = [fit_hmm(fit_seqs, K, seed=seed_base + 100 * K + j, max_iter=max_iter) for j in range(n_init)]
        best = max(fits, key=lambda m: m.loglik)
        vll = best.loglik_of(valid_seqs)
        occ = np.zeros(K)
        for X in fit_seqs:
            occ += _fb(X, best.pi, best.A, best.means, best.variances)[2].sum(axis=0)
        occ /= max(n_fit, 1)
        rows.append({
            "K": K, "fit_rows": n_fit, "valid_rows": n_val,
            "fit_loglik_per_row": best.loglik / n_fit,
            "valid_loglik_per_row": vll / max(n_val, 1),
            "min_state_occupancy": float(occ.min()),
            "n_converged": int(sum(m.converged for m in fits)),
            "best_minus_second_fit_loglik": float(sorted([m.loglik for m in fits])[-1] - sorted([m.loglik for m in fits])[-2]) if n_init > 1 else np.nan,
        })
        if progress:
            print(f"K={K}: valid ll/row {vll / max(n_val, 1):.4f}")
    comp = pd.DataFrame(rows)
    ok = comp[comp.min_state_occupancy >= 0.01]
    k_sel = int((ok if len(ok) else comp).sort_values("valid_loglik_per_row", ascending=False).iloc[0]["K"])
    final = max(
        (fit_hmm(train_seqs, k_sel, seed=seed_base + 7000 + j, max_iter=max_iter) for j in range(n_init)),
        key=lambda m: m.loglik,
    )
    comp["selected"] = comp["K"] == k_sel
    comp["k_at_grid_edge"] = k_sel == max(k_values)
    return comp, final, k_sel


def filtered_table(grid: vb.Grid, model: HMM, seqs_idx: list[np.ndarray], Z: np.ndarray, split: str) -> pd.DataFrame:
    parts = []
    for sid, ii in enumerate(seqs_idx):
        p = model.filter(Z[ii])
        df = pd.DataFrame(p, columns=[f"p_state_{k}" for k in range(model.K)])
        df.insert(0, "row", ii)
        df.insert(1, "sequence_id", sid)
        df["hard_state"] = p.argmax(axis=1)
        df["split"] = split
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# --------------------------------------------------------------------------
# q99 labels and benchmark comparison
# --------------------------------------------------------------------------

def future_onset_labels(grid: vb.Grid, onset: np.ndarray, horizons_min=HORIZONS_MIN) -> tuple[pd.DataFrame, np.ndarray]:
    """y_h(t) = 1 if an onset occurs in (t, t+h]; also the id (row) of the first such onset."""
    out = {}
    first = np.full(grid.n, -1, dtype=np.int64)
    rows = np.arange(grid.n)
    for h in horizons_min:
        k = int(h) // 5
        y = np.zeros(grid.n, bool)
        for s in range(k, 0, -1):
            r = grid.offset_rows(rows, s)
            hit = (r >= 0) & np.asarray(onset, bool)[np.maximum(r, 0)]
            y |= hit
            if h == max(horizons_min):
                first = np.where(hit, r, first)
        out[f"y_{int(h)}m"] = y
    return pd.DataFrame(out), first


def _logit_fit(X, y, ridge=1e-3, max_iter=50):
    n, d = X.shape
    w = np.zeros(d)
    R = ridge * n * np.eye(d)
    R[0, 0] = 0.0
    for _ in range(max_iter):
        z = np.clip(X @ w, -30, 30)
        p = 1 / (1 + np.exp(-z))
        W = p * (1 - p) + 1e-9
        H = X.T @ (X * W[:, None]) + R
        g = X.T @ (y - p) - R @ w
        step = np.linalg.solve(H, g)
        w += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def _design(df, cols, assets, mean=None, sd=None):
    D = [np.ones(len(df))]
    for a in assets[1:]:
        D.append((df["underlying"] == a).to_numpy(float))
    names = ["intercept"] + [f"asset_{a}" for a in assets[1:]]
    if cols:
        Xc = df[list(cols)].to_numpy(float)
        if mean is None:
            mean, sd = Xc.mean(axis=0), Xc.std(axis=0)
            sd = np.where(sd > 1e-12, sd, 1.0)
        D.extend(((Xc - mean) / sd).T)
        names += list(cols)
    return np.column_stack(D), mean, sd, names


def auc(y, p):
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(p).rank(method="average").to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def benchmark_comparison(train: pd.DataFrame, test: pd.DataFrame, state_cols: list[str], *,
                         horizons=HORIZONS_MIN, n_boot: int = 2000, seed: int = 606) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Logistic M0..M3 fit on train rows, scored on test rows (identical row sets)."""
    assets = sorted(set(train["underlying"]) | set(test["underlying"]))
    hmm_cols = state_cols[1:]                  # drop one (probabilities sum to 1)
    specs = {
        "M0_base_rate": [],
        "M1_vol_benchmark": list(BENCHMARK_FEATURES),
        "M2_hmm_states": hmm_cols,
        "M3_benchmark_plus_hmm": list(BENCHMARK_FEATURES) + hmm_cols,
    }
    rng = np.random.default_rng(int(seed))
    rows, boot_rows = [], []
    test_days = test["day"].to_numpy()
    uniq_days = np.unique(test_days)
    day_index = {d: np.flatnonzero(test_days == d) for d in uniq_days}
    for h in horizons:
        ycol = f"y_{int(h)}m"
        ytr = train[ycol].to_numpy(float)
        yte = test[ycol].to_numpy(float)
        losses = {}
        for name, cols in specs.items():
            Xtr, m, s, _ = _design(train, cols, assets)
            Xte, _, _, _ = _design(test, cols, assets, m, s)
            w = _logit_fit(Xtr, ytr)
            p = 1 / (1 + np.exp(-np.clip(Xte @ w, -30, 30)))
            p = np.clip(p, 1e-6, 1 - 1e-6)
            ll = -(yte * np.log(p) + (1 - yte) * np.log(1 - p))
            losses[name] = ll
            rows.append({
                "horizon_min": int(h), "model": name, "n_test_rows": int(len(yte)),
                "n_test_positive_rows": int(yte.sum()),
                "test_logloss": float(ll.mean()), "test_brier": float(np.mean((p - yte) ** 2)),
                "test_auc": auc(yte, p),
            })
        for a, b in (("M3_benchmark_plus_hmm", "M1_vol_benchmark"), ("M2_hmm_states", "M0_base_rate"),
                     ("M1_vol_benchmark", "M0_base_rate")):
            dl = losses[a] - losses[b]
            obs = float(dl.mean())
            bs = np.empty(int(n_boot))
            for i in range(int(n_boot)):
                pick = rng.choice(uniq_days, size=len(uniq_days), replace=True)
                idx = np.concatenate([day_index[d] for d in pick])
                bs[i] = dl[idx].mean()
            boot_rows.append({
                "horizon_min": int(h), "comparison": f"{a} - {b}",
                "delta_logloss": obs, "boot_q025": float(np.quantile(bs, 0.025)),
                "boot_q975": float(np.quantile(bs, 0.975)), "n_test_days": int(len(uniq_days)),
            })
    return pd.DataFrame(rows), pd.DataFrame(boot_rows)


def hmm_decision(boot: pd.DataFrame, n_unique_test_events: int) -> dict:
    r = boot[(boot.horizon_min == PRIMARY_HORIZON_MIN) & (boot.comparison == "M3_benchmark_plus_hmm - M1_vol_benchmark")]
    if r.empty:
        return {"status": "NOT_EVALUATED"}
    r = r.iloc[0]
    if n_unique_test_events < MIN_UNIQUE_TEST_EVENTS:
        status = "INSUFFICIENT_DATA"
    elif r.delta_logloss < 0 and r.boot_q975 < 0:
        status = "PASS"
    else:
        status = "FAIL"
    return {
        "rule": f"h={PRIMARY_HORIZON_MIN}m: logloss(M3)-logloss(M1) < 0 and day-bootstrap q97.5 < 0, >= {MIN_UNIQUE_TEST_EVENTS} unique test onset timestamps",
        "delta_logloss": float(r.delta_logloss), "boot_q975": float(r.boot_q975),
        "n_unique_test_onset_timestamps": int(n_unique_test_events), "status": status,
    }


def state_hazard_with_events(table: pd.DataFrame, K: int, first_event_row: np.ndarray, grid: vb.Grid,
                             horizons=HORIZONS_MIN) -> pd.DataFrame:
    rows = []
    for h in horizons:
        y = table[f"y_{int(h)}m"].to_numpy(float)
        base = float(y.mean())
        for k in range(K):
            m = table["hard_state"].to_numpy() == k
            p = table[f"p_state_{k}"].to_numpy(float)
            ev_rows = first_event_row[table["row"].to_numpy()[m & (y > 0)]]
            ev_rows = ev_rows[ev_rows >= 0]
            rows.append({
                "horizon_min": int(h), "state": k, "baseline": base,
                "soft_hazard": float((p * y).sum() / p.sum()) if p.sum() > 0 else np.nan,
                "hard_hazard": float(y[m].mean()) if m.any() else np.nan,
                "hard_rows": int(m.sum()),
                "distinct_events_behind_hard_hazard": int(len(np.unique(ev_rows))),
                "distinct_event_timestamps": int(len(np.unique(grid.time_ns[ev_rows]))) if len(ev_rows) else 0,
            })
    out = pd.DataFrame(rows)
    out["hard_lift"] = out.hard_hazard / out.baseline
    return out


# --------------------------------------------------------------------------
# Duration check: like-for-like simulation
# --------------------------------------------------------------------------

def _runs(hard: np.ndarray) -> list[tuple[int, int, bool, bool]]:
    out = []
    s = 0
    for i in range(1, len(hard) + 1):
        if i == len(hard) or hard[i] != hard[s]:
            out.append((int(hard[s]), i - s, s == 0, i == len(hard)))
            s = i
    return out


def run_stats(hards: list[np.ndarray], K: int, max_age: int = 36) -> tuple[np.ndarray, np.ndarray]:
    """Per-state mean uncensored run length and exit hazard by age (1..max_age)."""
    lens = {k: [] for k in range(K)}
    for h in hards:
        for st, L, lc, rc in _runs(h):
            if not lc and not rc:
                lens[st].append(L)
    mean = np.array([np.mean(lens[k]) if lens[k] else np.nan for k in range(K)])
    haz = np.full((K, max_age), np.nan)
    for k in range(K):
        d = np.asarray(lens[k])
        for a in range(1, max_age + 1):
            at = np.sum(d >= a)
            if at >= 5:
                haz[k, a - 1] = np.sum(d == a) / at
    return mean, haz


def duration_check(model: HMM, test_seqs: list[np.ndarray], n_sims: int = 30, seed: int = 909,
                   max_age: int = 36) -> tuple[pd.DataFrame, pd.DataFrame]:
    emp_hard = [model.filter(X).argmax(axis=1) for X in test_seqs]
    emp_mean, emp_haz = run_stats(emp_hard, model.K, max_age)
    rng = np.random.default_rng(int(seed))
    sim_means, sim_haz = [], []
    for _ in range(int(n_sims)):
        hards = []
        for X in test_seqs:
            Xs, _ = model.sample(len(X), np.isfinite(X), rng)
            hards.append(model.filter(Xs).argmax(axis=1))
        m, hz = run_stats(hards, model.K, max_age)
        sim_means.append(m)
        sim_haz.append(hz)
    SM = np.vstack(sim_means)
    SH = np.stack(sim_haz)
    summ = []
    for k in range(model.K):
        col = SM[:, k][np.isfinite(SM[:, k])]
        lo, hi = (np.quantile(col, [0.05, 0.95]) if len(col) else (np.nan, np.nan))
        summ.append({
            "state": k, "self_transition": float(model.A[k, k]),
            "geometric_mean_bars": 1 / max(1 - model.A[k, k], 1e-12),
            "empirical_filtered_mean_run_bars": emp_mean[k],
            "simulated_filtered_mean_run_q05": lo, "simulated_filtered_mean_run_q95": hi,
            "empirical_outside_sim_band": bool(np.isfinite(emp_mean[k]) and np.isfinite(lo) and (emp_mean[k] < lo or emp_mean[k] > hi)),
        })
    hz_rows = []
    for k in range(model.K):
        for a in range(max_age):
            col = SH[:, k, a]
            col = col[np.isfinite(col)]
            hz_rows.append({
                "state": k, "age_bars": a + 1, "age_min": 5 * (a + 1),
                "empirical_exit_hazard": emp_haz[k, a],
                "simulated_q05": float(np.quantile(col, 0.05)) if len(col) else np.nan,
                "simulated_q95": float(np.quantile(col, 0.95)) if len(col) else np.nan,
                "hmm_constant_hazard": 1 - float(model.A[k, k]),
            })
    return pd.DataFrame(summ), pd.DataFrame(hz_rows)


def event_alignment(grid: vb.Grid, table: pd.DataFrame, onset_rows: np.ndarray, K: int,
                    rel_bars=range(-24, 13)) -> pd.DataFrame:
    pos = pd.Series(np.arange(len(table)), index=table["row"].to_numpy())
    P = table[[f"p_state_{k}" for k in range(K)]].to_numpy(float)
    recs = []
    for e in onset_rows:
        for b in rel_bars:
            r = grid.offset_rows(np.array([e]), int(b))[0]
            if r >= 0 and r in pos.index:
                recs.append([grid.time_ns[e], b] + P[pos[r]].tolist())
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs, columns=["event_time_ns", "rel_bar"] + [f"p_state_{k}" for k in range(K)])
    ts = df.groupby(["event_time_ns", "rel_bar"]).mean().reset_index()
    out = ts.groupby("rel_bar")[[f"p_state_{k}" for k in range(K)]].mean().reset_index()
    out["rel_min"] = 5 * out.rel_bar
    out["n_unique_event_times"] = ts.groupby("rel_bar").size().to_numpy()
    return out


# --------------------------------------------------------------------------
# Self test
# --------------------------------------------------------------------------

def synthetic_market(n_days: int = 16, assets=("AAA", "BBB", "CCC"), seed: int = 4) -> pd.DataFrame:
    """Two hidden market states drive returns and every emission."""
    rng = np.random.default_rng(seed)
    T = n_days * BARS_PER_DAY
    A = np.array([[0.995, 0.005], [0.02, 0.98]])
    s = np.zeros(T, int)
    for t in range(1, T):
        s[t] = rng.choice(2, p=A[s[t - 1]])
    times = pd.date_range("2026-02-01", periods=T, freq="5min", tz="UTC")
    out = []
    for a in assets:
        lvl = np.where(s == 1, 1.0, 0.0)
        sig = 0.0015 * np.exp(0.9 * lvl + rng.normal(scale=0.2, size=T))
        df = pd.DataFrame({
            "state_time": times, "underlying": a,
            "return_5m": sig * rng.standard_t(5, size=T) / np.sqrt(5 / 3),
            "intrabar_range_bps": 1e4 * sig * np.abs(rng.normal(1.6, 0.3, size=T)),
            "volume_usd": np.exp(12 + 1.0 * lvl + rng.normal(scale=0.5, size=T)),
            "trade_count": rng.poisson(np.exp(4 + 0.8 * lvl)),
            "book_update_count": rng.poisson(np.exp(6 + 0.5 * lvl)),
            "book_twa_spread_bps": np.exp(0.5 + 0.4 * lvl + rng.normal(scale=0.3, size=T)),
            "option_iv_atm_forward": 0.5 * np.exp(0.05 * lvl + rng.normal(scale=0.01, size=T)),
            "iv_put_25d": 0.55 + 0.02 * lvl + rng.normal(scale=0.005, size=T),
            "iv_call_25d": 0.52 + rng.normal(scale=0.005, size=T),
        })
        miss = rng.random(T) < 0.1
        df.loc[miss, "option_iv_atm_forward"] = np.nan
        out.append(df)
    return pd.concat(out, ignore_index=True)


def selftest() -> dict:
    f = synthetic_market()
    g = vb.build_grid(f)
    E = build_emissions(g, include_l2=False)
    sp = chronological_split(g)
    tmask = g.time_ns < sp.test_start_ns
    params = fit_hour_normalizer(g, E, tmask)
    Z = apply_normalizer(g, E, params)
    fit_idx = sequences(g, g.time_ns < sp.valid_start_ns)
    val_idx = sequences(g, (g.time_ns >= sp.valid_start_ns) & tmask)
    tr_idx = sequences(g, tmask)
    te_idx = sequences(g, ~tmask)
    assert len(tr_idx) <= 3 * len(g.assets)          # missing IV does not fragment
    comp, model, k = select_and_fit([Z[i] for i in fit_idx], [Z[i] for i in val_idx], [Z[i] for i in tr_idx],
                                    k_values=(1, 2, 3), n_init=2, max_iter=40)
    assert k >= 2
    p = model.filter(Z[te_idx[0]])
    assert np.allclose(p.sum(axis=1), 1.0)
    # scaled forward-backward agrees with a brute-force log-space likelihood
    X = Z[te_idx[0]][:50]
    lb = _logB(X, model.means, model.variances)
    la = np.log(model.pi) + lb[0]
    for t in range(1, len(X)):
        m = la.max()
        la = lb[t] + np.log(np.exp(la - m) @ model.A) + m
    brute = float(np.log(np.exp(la - la.max()).sum()) + la.max())
    assert np.isclose(brute, model.loglik_of([X]), rtol=1e-8, atol=1e-6)
    summ, _ = duration_check(model, [Z[i] for i in te_idx], n_sims=3)
    assert len(summ) == model.K
    return {"missing_marginalized": True, "validation_K": True, "scaled_fb_matches_logspace": True,
            "duration_simulation": True}
