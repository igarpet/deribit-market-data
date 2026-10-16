"""
Signed Jumps 01: volatility engine + signed Hawkes jump layer on 1-minute Binance spot.

Protocol: discussions/Signed_Jumps/001_claude_sj01_preregistration.md (fixed before any real-data output).

Model (asset a, 1-minute bar t, sign s in {+, -})
-------------------------------------------------
r_t = sigma_c,t * eps_t + J_t

1. Deseasonalise: r~_t = r_t / s(slot, weekend); log s fitted on DEV by OLS of log|r| on a
   4-harmonic Fourier curve in time of day plus a weekend dummy.
2. Continuous engine (jump-robust, real time): bipower products bp_t = (pi/2)|r~_t||r~_{t-1}|.
   HAR on log bipower: log BV_fwd60(t) ~ 1 + log BV_30 + log BV_60 + log BV_240 + log BV_1440, all
   backward windows ending at t-1; fitted on DEV. sigma2_c,t = exp(fit + s2/2).
   Engine state (AM-T-HAR-Blend layout): EWMAs of log bp with half-lives 30 (fast) and 240 (slow)
   minutes through t-1;  L = slow - DEV mean,  F = fast - slow.
3. Jump: Z_t = r~_t / sigma_c,t;  J+_t = 1{Z_t >= c, r_t >= floor},  J-_t = 1{Z_t <= -c, r_t <= -floor};
   c = 4 primary, floor = 5 bp.
4. Jump-truncated skewness TSK_t: skewness of Z over bars t-60..t-1 with |Z| < c (>= 45 valid),
   standardised on DEV.
5. Discrete signed Hawkes:
      P(J^s_t = 1 | F_{t-1}) = 1 - exp(-lambda^s_t)
      lambda^s_t = exp(X^s_t b_s) + sum_{s'} eta[s, s'] E^{s'}_t
      E^{s'}_t = sum_{k>=1} g(k) J^{s'}_{t-k},  g(k) = (1-q) q^(k-1),  q = exp(-ln2 / half_life)
   Ladder (each nests the previous):
      (a) asset intercepts                                  homogeneous Poisson
      (b) + Fourier(4) + weekend + L + F                    engine-modulated Poisson (null)
      (c) + eta[s, +] = eta[s, -] = eta_s                   excitation, parent sign irrelevant
      (d) eta[s, s'] free                                   parent sign matters (conditional sign)
      (e) (d) + theta_s TSK_t                               pre-jump asymmetry
   Baseline coefficients are sign-specific; sign likelihoods separate given E, so each sign is fitted
   on its own (shared eta in (c) is per child sign, which keeps that separation).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

VERSION = "SJ01-0.1.1"   # 0.1.1: optional extra baseline covariates (models bf, f) for SJ 03; SJ 01 models unchanged
ASSETS = ("BTC", "ETH", "SOL", "XRP")
BARS_PER_DAY = 1440
SIGNS = ("up", "down")
MODELS = ("a", "b", "c", "d", "e")


@dataclass(frozen=True)
class Params:
    assets: tuple = ASSETS
    c: float = 4.0
    c_sensitivity: tuple = (3.5, 4.5, 5.0)
    floor_bp: float = 5.0
    fourier_k: int = 4
    har_windows: tuple = (30, 60, 240, 1440)
    har_target: int = 60
    min_cover: float = 0.75
    fast_half_life: int = 30
    slow_half_life: int = 240
    surprise_windows: tuple = (10, 60)
    tsk_window: int = 60
    tsk_min: int = 45
    half_lives: tuple = (1, 3, 10, 30, 90)
    dev_fraction: float = 0.70
    n_boot: int = 2000
    horizons: tuple = (5, 15, 60)
    primary_horizon: int = 15
    origin_step: int = 15
    mc_paths: int = 200
    min_test_jumps_per_sign: int = 200
    status: str = "PREREGISTERED_SJ001"


def registration(p: Params | None = None) -> dict:
    p = p or Params()
    d = asdict(p)
    d["version"] = VERSION
    d["gates"] = {
        "G1_excitation": "(c) vs (b): TEST logLik/row delta > 0, day-bootstrap q2.5 > 0, >= min jumps per sign",
        "G2a_parent_sign": "(d) vs (c): same rule at Bonferroni q1.25",
        "G2b_truncated_skew": "(e) vs (d): same rule at Bonferroni q1.25",
        "G2": "PASS if G2a or G2b passes",
        "G3_horizon": "signed occurrence log-loss at primary horizon, selected model vs (b): delta > 0, q2.5 > 0",
        "G4_economics": "deferred to SJ 02; built only if G2 and G3 pass",
    }
    return d


# ----------------------------------------------------------------------------- data

def load_spot_1m(path) -> pd.Series:
    """Binance spot 1m klines (open_time, close) -> log close indexed by bar END time on a full minute grid."""
    df = pd.read_parquet(path, columns=["open_time", "close"])
    t = pd.to_datetime(df["open_time"], utc=True) + pd.Timedelta(minutes=1)
    s = pd.Series(np.log(df["close"].astype(float).to_numpy()), index=t).sort_index()
    s = s[~s.index.duplicated()]
    grid = pd.date_range(s.index.min().ceil("D"), s.index.max().floor("D"), freq="1min", inclusive="left")
    return s.reindex(grid)


@dataclass
class Asset:
    name: str
    time: pd.DatetimeIndex
    r: np.ndarray                       # raw log return (NaN if missing)
    dev: np.ndarray                     # bool, DEV rows
    cols: dict = field(default_factory=dict)


def make_asset(name: str, logp: pd.Series, dev_end: pd.Timestamp) -> Asset:
    r = logp.diff().to_numpy()
    return Asset(name, logp.index, r, (logp.index < dev_end))


def split_date(times: list[pd.DatetimeIndex], frac: float) -> pd.Timestamp:
    """DEV/TEST boundary: day at `frac` of the union of calendar days (all assets)."""
    days = np.unique(np.concatenate([t.normalize().to_numpy() for t in times]))
    return pd.Timestamp(days[int(np.floor(frac * len(days)))]).tz_localize("UTC") if pd.Timestamp(days[0]).tzinfo is None \
        else pd.Timestamp(days[int(np.floor(frac * len(days)))])


# ----------------------------------------------------------------------------- engine

def clock_design(time: pd.DatetimeIndex, k: int) -> np.ndarray:
    slot = (time.hour * 60 + time.minute).to_numpy()
    u = 2 * np.pi * slot / BARS_PER_DAY
    cols = []
    for j in range(1, k + 1):
        cols += [np.sin(j * u), np.cos(j * u)]
    cols.append((time.dayofweek >= 5).astype(float))
    return np.column_stack(cols)


def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _roll_mean(x: np.ndarray, w: int, min_count: int) -> np.ndarray:
    """Mean over the w values ending at each index (inclusive); NaN if fewer than min_count finite."""
    s = pd.Series(x)
    return s.rolling(w, min_periods=min_count).mean().to_numpy()


def _lag(x: np.ndarray, k: int = 1) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    out[k:] = x[:-k]
    return out


def _ewma_lagged(x: np.ndarray, half_life: int) -> np.ndarray:
    """EWMA of x through t-1 (ignoring NaN), value at t."""
    m = pd.Series(x).ewm(halflife=half_life, ignore_na=True, min_periods=half_life).mean().to_numpy()
    return _lag(m)


def build_engine(a: Asset, p: Params) -> None:
    """Fills a.cols: z (deseasonalised return), sig2c, Z, L/F raw, Jup/Jdn, TSK raw, valid."""
    C = clock_design(a.time, p.fourier_k)
    X = np.column_stack([np.ones(len(C)), C])
    ok = a.dev & np.isfinite(a.r) & (a.r != 0)
    gs = _ols(X[ok], np.log(np.abs(a.r[ok])))
    s = np.exp(X @ gs)
    z = a.r / s
    bp = (np.pi / 2) * np.abs(z) * np.abs(_lag(z))
    # backward BV windows ending at t-1
    feats = []
    for w in p.har_windows:
        m = _lag(_roll_mean(bp, w, int(p.min_cover * w)))
        feats.append(np.log(m))
    F_ = np.column_stack(feats)
    # forward target: mean bp over t .. t+H-1
    H = p.har_target
    fwd = pd.Series(bp[::-1]).rolling(H, min_periods=int(p.min_cover * H)).mean().to_numpy()[::-1]
    y = np.log(fwd)
    Xh = np.column_stack([np.ones(len(F_)), F_])
    fit = a.dev & np.isfinite(y) & np.isfinite(Xh).all(axis=1)
    # the DEV target must not reach into TEST
    last_dev = np.flatnonzero(a.dev)
    if len(last_dev):
        fit[max(0, last_dev[-1] - H + 2):] = False
    gh = _ols(Xh[fit], y[fit])
    s2 = float(np.var(y[fit] - Xh[fit] @ gh))
    sig2c = np.exp(Xh @ gh + s2 / 2)
    Z = z / np.sqrt(sig2c)
    floor = p.floor_bp / 1e4
    up = (Z >= p.c) & (a.r >= floor)
    dn = (Z <= -p.c) & (a.r <= -floor)
    # engine state from log bp (floored)
    fl = np.nanmedian(bp[a.dev]) * 0.1
    lbp = np.log(np.where(np.isfinite(bp), np.maximum(bp, fl), np.nan))
    fast = _ewma_lagged(lbp, p.fast_half_life)
    slow = _ewma_lagged(lbp, p.slow_half_life)
    a.cols.update(dict(z=z, bp=bp, sig2c=sig2c, Z=Z, fast=fast, slow=slow, clock=C,
                       har_coef=gh, seas_coef=gs))
    a.cols["Jup"] = up.astype(float)
    a.cols["Jdn"] = dn.astype(float)
    a.cols["obs"] = np.isfinite(Z)
    a.cols["TSK"] = truncated_skew(Z, p.c, p.tsk_window, p.tsk_min)
    for w in p.surprise_windows:
        a.cols[f"TV{w}"] = truncated_var_ratio(Z, p.c, w, max(3, int(p.min_cover * w)))


def truncated_var_ratio(Z: np.ndarray, c: float, w: int, min_n: int) -> np.ndarray:
    """log mean Z^2 over bars t-w..t-1 with |Z| < c (jumps excluded): realised / forecast continuous
    variance. Absorbs volatility-forecast error so it cannot masquerade as jump excitation."""
    keep = np.isfinite(Z) & (np.abs(Z) < c)
    k = pd.Series(keep.astype(float)).rolling(w, min_periods=1).sum().to_numpy()
    s2 = pd.Series(np.where(keep, Z ** 2, 0.0)).rolling(w, min_periods=1).sum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.log(np.maximum(s2 / k, 1e-3))
    v[k < min_n] = np.nan
    return _lag(v)


def truncated_skew(Z: np.ndarray, c: float, w: int, min_n: int) -> np.ndarray:
    """Skewness of Z over bars t-w..t-1 with |Z| < c; value at t (strictly causal)."""
    keep = np.isfinite(Z) & (np.abs(Z) < c)
    x = np.where(keep, Z, 0.0)
    k = keep.astype(float)
    S0 = pd.Series(k).rolling(w, min_periods=1).sum().to_numpy()
    S1 = pd.Series(x).rolling(w, min_periods=1).sum().to_numpy()
    S2 = pd.Series(x ** 2).rolling(w, min_periods=1).sum().to_numpy()
    S3 = pd.Series(x ** 3).rolling(w, min_periods=1).sum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        m = S1 / S0
        m2 = S2 / S0 - m ** 2
        m3 = S3 / S0 - 3 * m * S2 / S0 + 2 * m ** 3
        sk = m3 / m2 ** 1.5
    sk[S0 < min_n] = np.nan
    return _lag(sk)


def excitation(J: np.ndarray, half_life: float) -> np.ndarray:
    """E_t = q E_{t-1} + (1-q) J_{t-1}  (kernel sums to 1)."""
    q = np.exp(-np.log(2) / half_life)
    j = np.nan_to_num(J, nan=0.0)
    return lfilter([0.0, 1 - q], [1.0, -q], j)


# ----------------------------------------------------------------------------- panel

@dataclass
class Panel:
    asset_idx: np.ndarray
    day: np.ndarray            # int day id (calendar, shared across assets)
    dev: np.ndarray
    time: np.ndarray
    clock: np.ndarray
    L: np.ndarray
    F: np.ndarray
    TSK: np.ndarray
    TV: np.ndarray             # jump-truncated variance-surprise columns
    J: dict                    # sign -> array
    Z: np.ndarray
    r: np.ndarray
    n_assets: int
    names: tuple


def build_panel(assets: list[Asset], p: Params) -> Panel:
    """Stack assets; keep rows with complete covariates and an observed return. Excitation is computed
    per asset BEFORE dropping rows (from all observed jumps)."""
    parts = []
    for i, a in enumerate(assets):
        c = a.cols
        dev_mean = np.nanmean(c["slow"][a.dev])
        L = c["slow"] - dev_mean
        F = c["fast"] - c["slow"]
        tsk_sd = np.nanstd(c["TSK"][a.dev])
        tsk_mu = np.nanmean(c["TSK"][a.dev])
        T = (c["TSK"] - tsk_mu) / tsk_sd
        TV = np.column_stack([c[f"TV{w}"] for w in p.surprise_windows])
        ok = c["obs"] & np.isfinite(L) & np.isfinite(F) & np.isfinite(T) & np.isfinite(TV).all(axis=1)
        parts.append(dict(asset=np.full(ok.sum(), i), time=a.time.to_numpy()[ok], dev=a.dev[ok],
                          clock=c["clock"][ok], L=L[ok], F=F[ok], TSK=T[ok], TV=TV[ok],
                          Jup=c["Jup"][ok], Jdn=c["Jdn"][ok], Z=c["Z"][ok], r=a.r[ok],
                          idx=np.flatnonzero(ok)))
    cat = {k: np.concatenate([q[k] for q in parts]) for k in parts[0] if k != "idx"}
    day = pd.DatetimeIndex(cat["time"]).normalize()
    _, day_id = np.unique(day.to_numpy(), return_inverse=True)
    pan = Panel(cat["asset"], day_id, cat["dev"], cat["time"], cat["clock"].astype(np.float32),
                cat["L"].astype(np.float32), cat["F"].astype(np.float32), cat["TSK"].astype(np.float32),
                cat["TV"].astype(np.float32),
                {"up": cat["Jup"], "down": cat["Jdn"]}, cat["Z"], cat["r"], len(assets),
                tuple(a.name for a in assets))
    pan._idx = [q["idx"] for q in parts]          # row -> original index per asset
    return pan


def panel_excitation(assets: list[Asset], pan: Panel, half_life: float) -> dict:
    """Excitation per parent sign on panel rows, computed on each asset's full grid."""
    out = {"up": [], "down": []}
    for a, idx in zip(assets, pan._idx):
        for s, key in (("up", "Jup"), ("down", "Jdn")):
            out[s].append(excitation(a.cols[key], half_life)[idx])
    return {s: np.concatenate(v) for s, v in out.items()}


def design(pan: Panel, model: str) -> np.ndarray:
    D = np.zeros((len(pan.asset_idx), pan.n_assets), np.float32)
    D[np.arange(len(D)), pan.asset_idx] = 1
    if model == "a":
        return D
    cols = [D, pan.clock, pan.L[:, None], pan.F[:, None], pan.TV]
    if model in ("e", "f"):
        cols.append(pan.TSK[:, None])
    if model in ("bf", "f"):
        cols.append(pan.extra)          # SJ 03 flow covariates, attached by signed_jumps_flow
    return np.column_stack(cols).astype(np.float32)


def design_names(pan: Panel, model: str, p: Params) -> list[str]:
    names = [f"asset_{a}" for a in pan.names]
    if model == "a":
        return names
    for j in range(1, p.fourier_k + 1):
        names += [f"sin{j}", f"cos{j}"]
    names += ["weekend", "L_slow", "F_fast_minus_slow"] + [f"TV{w}" for w in p.surprise_windows]
    if model in ("e", "f"):
        names.append("TSK")
    if model in ("bf", "f"):
        names += list(pan.extra_names)
    return names


def excitation_design(E: dict, model: str, child: str) -> np.ndarray | None:
    if model in ("a", "b", "bf"):
        return None
    if model == "c":
        return (E["up"] + E["down"])[:, None]
    return np.column_stack([E["up"], E["down"]])


# ----------------------------------------------------------------------------- likelihood

def loglik_terms(J: np.ndarray, lam: np.ndarray) -> np.ndarray:
    lam = np.maximum(lam, 1e-300)
    return np.where(J > 0.5, np.log(-np.expm1(-lam)), -lam)


def _negll(theta, X, Ex, J, nb):
    b = theta[:nb]
    eta = theta[nb:]
    mu = np.exp(np.clip(X @ b, -50, 5))
    lam = mu + (Ex @ eta if Ex is not None else 0.0)
    lam = np.maximum(lam, 1e-300)
    jj = J > 0.5
    ll = np.where(jj, np.log(-np.expm1(-lam)), -lam).sum()
    w = np.where(jj, 1.0 / np.expm1(lam), 0.0) - (~jj)
    g_b = X.T @ (w * mu)
    g = [g_b]
    if Ex is not None:
        g.append(Ex.T @ w)
    n = len(J)
    return -ll / n, -np.concatenate(g).astype(float) / n


@dataclass
class Fit:
    model: str
    sign: str
    b: np.ndarray
    eta: np.ndarray
    half_life: float | None
    ll_dev: float
    n_dev: int
    converged: bool


def fit_sign(X, Ex, J, model, sign, half_life, n_dummy: int) -> Fit:
    """The first n_dummy columns of X are asset intercept dummies (start at the log jump rate)."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    Ex = None if Ex is None else np.ascontiguousarray(Ex, dtype=np.float64)
    nb = X.shape[1]
    ne = 0 if Ex is None else Ex.shape[1]
    th0 = np.zeros(nb + ne)
    th0[:n_dummy] = np.log(max(J.mean(), 1e-6))
    th0[nb:] = 0.01
    bounds = [(None, None)] * nb + [(0.0, None)] * ne
    res = minimize(_negll, th0, args=(X, Ex, J, nb), jac=True, method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=500, gtol=1e-9, ftol=1e-13))
    return Fit(model, sign, res.x[:nb], res.x[nb:], half_life, -res.fun * len(J), len(J), bool(res.success))


def intensity(fit: Fit, X, Ex) -> np.ndarray:
    lam = np.exp(np.clip(X @ fit.b, -50, 5))
    if Ex is not None and len(fit.eta):
        lam = lam + Ex @ fit.eta
    return lam


def fit_ladder(assets, pan: Panel, p: Params, models=MODELS) -> dict:
    """Fits every model on DEV rows; excitation models profile the half-life on DEV likelihood."""
    dev = pan.dev
    fits = {}
    Ecache = {h: panel_excitation(assets, pan, h) for h in p.half_lives}
    for m in models:
        X = design(pan, m)
        for s in SIGNS:
            J = pan.J[s]
            if m in ("a", "b"):
                fits[(m, s)] = fit_sign(X[dev], None, J[dev], m, s, None, pan.n_assets)
                continue
            best = None
            for h in p.half_lives:
                Ex = excitation_design(Ecache[h], m, s)
                f = fit_sign(X[dev], Ex[dev], J[dev], m, s, h, pan.n_assets)
                if best is None or f.ll_dev > best.ll_dev:
                    best = f
            fits[(m, s)] = best
    return fits, Ecache


def row_loglik(fits, pan: Panel, Ecache, model: str) -> np.ndarray:
    """Per-row log-likelihood summed over both signs."""
    X = design(pan, model)
    tot = np.zeros(len(pan.asset_idx))
    for s in SIGNS:
        f = fits[(model, s)]
        Ex = excitation_design(Ecache[f.half_life], model, s) if f.half_life is not None else None
        tot += loglik_terms(pan.J[s], intensity(f, X, Ex))
    return tot


# ----------------------------------------------------------------------------- inference

def day_boot_delta(delta: np.ndarray, day: np.ndarray, n_boot: int, seed: int) -> dict:
    """Delta logLik per row with a calendar-day block bootstrap (days shared across assets)."""
    ud, inv = np.unique(day, return_inverse=True)
    S = np.bincount(inv, weights=delta)
    N = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ud), size=(n_boot, len(ud)))
    bs = S[draws].sum(1) / N[draws].sum(1)
    return dict(delta_per_row=float(S.sum() / N.sum()), q0125=float(np.quantile(bs, 0.0125)),
                q025=float(np.quantile(bs, 0.025)), q975=float(np.quantile(bs, 0.975)),
                q09875=float(np.quantile(bs, 0.98750)), n_days=int(len(ud)), n_rows=int(N.sum()))


def gate(boot: dict, n_jumps: dict, p: Params, bonferroni: bool = False) -> str:
    lo = boot["q0125"] if bonferroni else boot["q025"]
    if min(n_jumps.values()) < p.min_test_jumps_per_sign:
        return "INSUFFICIENT_EVENTS"
    return "PASS" if boot["delta_per_row"] > 0 and lo > 0 else "FAIL"


def eta_matrix(fits) -> pd.DataFrame:
    rows = []
    for s in SIGNS:
        f = fits[("d", s)]
        rows.append({"child": s, "from_up": f.eta[0], "from_down": f.eta[1], "half_life_min": f.half_life})
    return pd.DataFrame(rows)


def branching(fits, model="d") -> float:
    if model == "c":
        return float(fits[("c", "up")].eta[0] + fits[("c", "down")].eta[0])
    M = np.array([[fits[(model, "up")].eta[0], fits[(model, "up")].eta[1]],
                  [fits[(model, "down")].eta[0], fits[(model, "down")].eta[1]]])
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def cluster_se(fit: Fit, X, Ex, J, day) -> np.ndarray:
    """Cluster(day)-robust sandwich SEs: numerical Hessian of the analytic gradient, day-summed scores."""
    nb = X.shape[1]
    th = np.concatenate([fit.b, fit.eta])
    n = len(J)

    def grad(t):
        return _negll(t, X, Ex, J, nb)[1] * n

    H = np.zeros((len(th), len(th)))
    for k in range(len(th)):
        e = np.zeros_like(th)
        e[k] = 1e-5 * max(1.0, abs(th[k]))
        H[:, k] = (grad(th + e) - grad(th - e)) / (2 * e[k])
    H = (H + H.T) / 2
    b, eta = fit.b, fit.eta
    mu = np.exp(np.clip(X @ b, -50, 5))
    lam = np.maximum(mu + (Ex @ eta if Ex is not None else 0.0), 1e-300)
    jj = J > 0.5
    w = np.where(jj, 1.0 / np.expm1(lam), 0.0) - (~jj)
    sc = [X * (w * mu)[:, None]]
    if Ex is not None:
        sc.append(Ex * w[:, None])
    Sc = np.column_stack(sc)
    _, inv = np.unique(day, return_inverse=True)
    G = np.zeros((inv.max() + 1, Sc.shape[1]))
    np.add.at(G, inv, Sc)
    Hi = np.linalg.pinv(H)
    V = Hi @ (G.T @ G) @ Hi
    return np.sqrt(np.maximum(np.diag(V), 0))


# ----------------------------------------------------------------------------- horizon forecasts

def horizon_forecast(fits, model: str, assets, pan: Panel, p: Params, rows: np.ndarray, seed: int = 0):
    """For each origin row o (information through o-1): P(>=1 jump of each sign in bars o..o+h-1).
    Baseline: asset/state covariates frozen at o, clock terms from the true future slots.
    Model (b): exact. Excitation models: Monte Carlo with p.mc_paths paths.
    Returns dict horizon -> {sign: prob array}, plus realised indicators."""
    H = max(p.horizons)
    hmax = H
    out_p = {h: {s: np.zeros(len(rows)) for s in SIGNS} for h in p.horizons}
    out_y = {h: {s: np.zeros(len(rows)) for s in SIGNS} for h in p.horizons}
    # build per-asset arrays on original grids for future clock terms and realised jumps
    a_of = pan.asset_idx[rows]
    rng = np.random.default_rng(seed)
    X = design(pan, model)
    nb_clock = pan.clock.shape[1]
    # future clock contribution: recompute the clock design at o+j from the origin time
    times = pd.DatetimeIndex(pan.time[rows])
    if times.tz is None:
        times = times.tz_localize("UTC")
    base = {}
    for s in SIGNS:
        f = fits[(model, s)]
        b = f.b
        nA = pan.n_assets
        if model == "a":
            lin0 = X[rows] @ b
            base[s] = np.repeat(np.exp(lin0)[:, None], hmax, axis=1)
            continue
        bc = b[nA:nA + nb_clock]
        lin_now = X[rows] @ b - pan.clock[rows] @ bc
        B = np.empty((len(rows), hmax))
        for j in range(hmax):
            Cj = clock_design(times + pd.Timedelta(minutes=j), p.fourier_k)
            B[:, j] = np.exp(np.clip(lin_now + Cj @ bc, -50, 5))
        base[s] = B
    # realised jumps on the original grids
    for i_a, a in enumerate(assets):
        sel = np.flatnonzero(a_of == i_a)
        if not len(sel):
            continue
        idx = pan._idx[i_a]
        off = int(np.sum([len(x) for x in pan._idx[:i_a]]))
        g = idx[rows[sel] - off]
        for s, key in (("up", "Jup"), ("down", "Jdn")):
            Jg = np.nan_to_num(a.cols[key], nan=0.0)
            cs = np.concatenate([[0.0], np.cumsum(Jg)])
            obs = np.concatenate([[0.0], np.cumsum(a.cols["obs"].astype(float))])
            for h in p.horizons:
                end = np.minimum(g + h, len(Jg))
                cnt = cs[end] - cs[g]
                ob = obs[end] - obs[g]
                y = (cnt > 0).astype(float)
                y[ob < h * 0.9] = np.nan
                out_y[h][s][sel] = y
    if model in ("a", "b", "bf"):
        for s in SIGNS:
            cum = np.cumsum(base[s], axis=1)
            for h in p.horizons:
                out_p[h][s] = -np.expm1(-cum[:, h - 1])
        return out_p, out_y
    fu, fd = fits[(model, "up")], fits[(model, "down")]
    hl = fu.half_life
    q = np.exp(-np.log(2) / hl)
    eta_u = fu.eta if len(fu.eta) == 2 else np.repeat(fu.eta, 2)
    eta_d = fd.eta if len(fd.eta) == 2 else np.repeat(fd.eta, 2)
    E = panel_excitation(assets, pan, hl)
    M = p.mc_paths
    batch = max(1, 400_000 // M)
    # Conditional (Rao-Blackwellised) estimator. On {no target-sign jump yet} the target's excitation only
    # decays and the other sign evolves with target jumps absent, so
    #   P(no target jump in h) = E_Q[ prod_j exp(-lambda^target_j) ],
    # Q = simulate the other sign only. Unbiased, no smoothing, strictly inside (0, 1).
    for target in SIGNS:
        other = "down" if target == "up" else "up"
        et = eta_u if target == "up" else eta_d          # [from_up, from_down] for the target child
        eo = eta_d if target == "up" else eta_u          # for the other child
        it, io = (0, 1) if target == "up" else (1, 0)    # column of each parent sign in eta rows
        for b0 in range(0, len(rows), batch):
            sl = slice(b0, min(len(rows), b0 + batch))
            n = sl.stop - sl.start
            Et = np.repeat(E[target][rows[sl]][:, None], M, axis=1)
            Eo = np.repeat(E[other][rows[sl]][:, None], M, axis=1)
            logS = np.zeros((n, M))
            for j in range(hmax):
                lt = base[target][sl, j][:, None] + et[it] * Et + et[io] * Eo
                lo = base[other][sl, j][:, None] + eo[it] * Et + eo[io] * Eo
                logS -= lt
                jo = rng.random((n, M)) < -np.expm1(-lo)
                Et = q * Et
                Eo = q * Eo + (1 - q) * jo
                if (j + 1) in p.horizons:
                    out_p[j + 1][target][sl] = -np.expm1(np.log(np.exp(logS).mean(1)))
    return out_p, out_y


def occurrence_logloss(prob: dict, y: dict) -> np.ndarray:
    tot = 0.0
    for s in SIGNS:
        pp = np.clip(prob[s], 1e-9, 1 - 1e-9)
        yy = y[s]
        tot = tot + np.where(yy > 0.5, -np.log(pp), -np.log1p(-pp))
    return tot


# ----------------------------------------------------------------------------- synthetic data

def simulate(n_days=40, n_assets=2, eta=((0.0, 0.0), (0.0, 0.0)), half_life=10.0, base_rate=6e-4,
             vol_persist=0.9995, vol_sd=0.02, common=0.7, df=4, jump_mult=6.0, seed=0, sigma=8e-4,
             clock_amp=0.3, start="2024-01-01", skew_link=0.0, skew_persist=0.995,
             heat_link=0.0, heat_persist=0.995, flow=False, flow_heat=0.0, flow_vol=0.5):
    """Synthetic 1m log prices: common+idiosyncratic log-vol AR(1), Student-t diffusion, intraday clock,
    and a discrete signed Hawkes jump layer with branching matrix eta[child][parent].
    Down jumps are larger (unconditional size asymmetry).
    skew_link > 0 plants pre-jump asymmetry: a slow latent state k_t skews the diffusion and tilts the
    baselines, lambda+ x exp(skew_link k), lambda- x exp(-skew_link k).
    heat_link > 0 plants a latent "heat" state H_t (AR(1)) that multiplies both jump baselines by
    exp(heat_link H). flow=True also returns per-minute trades / quote volume / taker-buy quote volume, with
    trade intensity exp(flow_vol * log-vol + flow_heat * H): flow_heat = 0 means flow tracks volatility only.
    Heat and flow use a separate random stream, so the price paths for flow=False/heat_link=0 are unchanged.
    Returns the series dict, or (series, flow) when flow=True."""
    rng = np.random.default_rng(seed)
    n = n_days * BARS_PER_DAY
    t = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    slot = np.arange(n) % BARS_PER_DAY
    clock = np.exp(clock_amp * np.sin(2 * np.pi * slot / BARS_PER_DAY))
    hc = lfilter([vol_sd], [1, -vol_persist], rng.standard_normal(n))
    out = {}
    fl = {}
    rng2 = np.random.default_rng(seed + 10_000)
    eta = np.asarray(eta, float)
    q = np.exp(-np.log(2) / half_life)
    for a in range(n_assets):
        hi = lfilter([vol_sd], [1, -vol_persist], rng.standard_normal(n))
        h = common * hc + np.sqrt(1 - common ** 2) * hi
        sd = sigma * clock * np.exp(h)
        tt = rng.standard_t(df, n) * np.sqrt((df - 2) / df)
        kst = np.zeros(n)
        if skew_link:
            kst = lfilter([np.sqrt(1 - skew_persist ** 2)], [1, -skew_persist], rng.standard_normal(n))
            g = rng.standard_normal(n)
            tt = (g + 0.35 * kst * (g ** 2 - 1)) / np.sqrt(1 + 2 * (0.35 * kst) ** 2)
        r = sd * tt
        Eu = Ed = 0.0
        ju = np.zeros(n)
        jd = np.zeros(n)
        u = rng.random((n, 2))
        H = np.zeros(n)
        if heat_link or flow:
            H = lfilter([np.sqrt(1 - heat_persist ** 2)], [1, -heat_persist], rng2.standard_normal(n))
        bu = base_rate * np.exp(skew_link * kst + heat_link * H)
        bd = base_rate * np.exp(-skew_link * kst + heat_link * H)
        for i in range(n):
            lu = bu[i] + eta[0, 0] * Eu + eta[0, 1] * Ed
            ld = bd[i] + eta[1, 0] * Eu + eta[1, 1] * Ed
            ju[i] = u[i, 0] < -np.expm1(-lu)
            jd[i] = u[i, 1] < -np.expm1(-ld)
            Eu = q * Eu + (1 - q) * ju[i]
            Ed = q * Ed + (1 - q) * jd[i]
        size_u = sd * (jump_mult + rng.exponential(2.0, n))
        size_d = sd * (jump_mult + rng.exponential(3.0, n))
        r = r + ju * size_u - jd * size_d
        lp = np.log(100.0) + np.concatenate([[0.0], np.cumsum(r[1:])])
        out[f"S{a}"] = pd.Series(lp, index=t + pd.Timedelta(minutes=1))
        if flow:
            lam_tr = 40.0 * clock * np.exp(flow_vol * h / max(np.std(h), 1e-9) * 0.5 + flow_heat * H)
            trades = rng2.poisson(lam_tr).astype(float)
            qv = trades * np.exp(rng2.normal(8.0, 0.5, n))
            tb = qv * np.clip(rng2.normal(0.5, 0.08, n), 0, 1)
            fl[f"S{a}"] = pd.DataFrame({"trades": trades, "quote_volume": qv, "taker_buy_quote_volume": tb},
                                       index=t + pd.Timedelta(minutes=1))
    return (out, fl) if flow else out


# ----------------------------------------------------------------------------- end to end

def prepare(series: dict, p: Params):
    times = [s.index for s in series.values()]
    allt = np.unique(np.concatenate([t.normalize().to_numpy() for t in times]))
    dev_end = pd.Timestamp(allt[int(np.floor(p.dev_fraction * len(allt)))])
    if dev_end.tzinfo is None:
        dev_end = dev_end.tz_localize("UTC")
    assets = [make_asset(k, v, dev_end) for k, v in series.items()]
    for a in assets:
        build_engine(a, p)
    pan = build_panel(assets, p)
    return assets, pan, dev_end


def compare(fits, pan, Ecache, p: Params, seed=11) -> tuple[pd.DataFrame, dict]:
    test = ~pan.dev
    ll = {m: row_loglik(fits, pan, Ecache, m)[test] for m in MODELS if (m, "up") in fits}
    day = pan.day[test]
    njump = {s: int(pan.J[s][test].sum()) for s in SIGNS}
    pairs = [("b_vs_a", "b", "a", False), ("G1_c_vs_b", "c", "b", False), ("G2a_d_vs_c", "d", "c", True),
             ("G2b_e_vs_d", "e", "d", True), ("d_vs_b", "d", "b", False), ("e_vs_b", "e", "b", False)]
    rows = []
    for name, m1, m0, bon in pairs:
        if m1 not in ll or m0 not in ll:
            continue
        bt = day_boot_delta(ll[m1] - ll[m0], day, p.n_boot, seed)
        bt.update(test=name, status=gate(bt, njump, p, bonferroni=bon), bonferroni=bon)
        rows.append(bt)
    return pd.DataFrame(rows), njump


def g3_table(fits, model: str, assets, pan: Panel, p: Params, seed: int = 21) -> tuple[pd.DataFrame, pd.DataFrame]:
    """G3: signed occurrence log-loss of `model` vs (b) at each horizon, origins every p.origin_step bars
    on TEST. Also returns a calibration table of the predicted up-share among origins with >= 1 jump."""
    rows = np.flatnonzero(~pan.dev)[:: p.origin_step]
    pb, y = horizon_forecast(fits, "b", assets, pan, p, rows, seed=seed)
    pm, _ = horizon_forecast(fits, model, assets, pan, p, rows, seed=seed + 1)
    njump = {s: int(pan.J[s][~pan.dev].sum()) for s in SIGNS}
    out, cal = [], []
    for h in p.horizons:
        ok = np.isfinite(y[h]["up"]) & np.isfinite(y[h]["down"])
        yy = {s: y[h][s][ok] for s in SIGNS}
        lb = occurrence_logloss({s: pb[h][s][ok] for s in SIGNS}, yy)
        lm = occurrence_logloss({s: pm[h][s][ok] for s in SIGNS}, yy)
        bt = day_boot_delta(lb - lm, pan.day[rows][ok], p.n_boot, seed + h)
        bt.update(horizon_min=h, model=model, status=gate(bt, njump, p),
                  mean_p_up_b=float(pb[h]["up"][ok].mean()), mean_p_up_model=float(pm[h]["up"][ok].mean()),
                  obs_up=float(yy["up"].mean()), mean_p_dn_b=float(pb[h]["down"][ok].mean()),
                  mean_p_dn_model=float(pm[h]["down"][ok].mean()), obs_dn=float(yy["down"].mean()))
        out.append(bt)
        any_ = (yy["up"] + yy["down"]) > 0
        share = pm[h]["up"][ok] / (pm[h]["up"][ok] + pm[h]["down"][ok])
        if any_.sum() >= 50:
            dec = pd.qcut(share[any_], 5, labels=False, duplicates="drop")
            df = pd.DataFrame({"bin": dec, "pred_up_share": share[any_],
                               "up_only": ((yy["up"] > 0) & (yy["down"] == 0))[any_],
                               "down_only": ((yy["down"] > 0) & (yy["up"] == 0))[any_]})
            g = df.groupby("bin").agg(n=("pred_up_share", "size"), pred_up_share=("pred_up_share", "mean"),
                                      up_only=("up_only", "mean"), down_only=("down_only", "mean")).reset_index()
            g["horizon_min"] = h
            cal.append(g)
    return pd.DataFrame(out), (pd.concat(cal, ignore_index=True) if cal else pd.DataFrame())


def rejump(assets: list[Asset], p: Params) -> None:
    """Re-derive jumps and jump-truncated covariates for a different threshold p.c (engine unchanged)."""
    floor = p.floor_bp / 1e4
    for a in assets:
        Z = a.cols["Z"]
        a.cols["Jup"] = ((Z >= p.c) & (a.r >= floor)).astype(float)
        a.cols["Jdn"] = ((Z <= -p.c) & (a.r <= -floor)).astype(float)
        a.cols["TSK"] = truncated_skew(Z, p.c, p.tsk_window, p.tsk_min)
        for w in p.surprise_windows:
            a.cols[f"TV{w}"] = truncated_var_ratio(Z, p.c, w, max(3, int(p.min_cover * w)))


def selected_model(cmp: pd.DataFrame) -> str:
    st = dict(zip(cmp.test, cmp.status))
    if st.get("G2b_e_vs_d") == "PASS":
        return "e"
    if st.get("G2a_d_vs_c") == "PASS":
        return "d"
    if st.get("G1_c_vs_b") == "PASS":
        return "c"
    return "d"   # G3 still scored on the full signed model for the record


def decide(cmp: pd.DataFrame, g3: pd.DataFrame | None, p: Params) -> dict:
    st = dict(zip(cmp.test, cmp.status))
    g2 = "PASS" if "PASS" in (st.get("G2a_d_vs_c"), st.get("G2b_e_vs_d")) else "FAIL"
    out = {"version": VERSION, "status": p.status, "G1_excitation": st.get("G1_c_vs_b"),
           "G2a_parent_sign": st.get("G2a_d_vs_c"), "G2b_truncated_skew": st.get("G2b_e_vs_d"), "G2": g2}
    if g3 is not None and len(g3):
        prim = g3[g3.horizon_min == p.primary_horizon].iloc[0]
        out["G3_horizon"] = prim["status"]
        out["G3_model"] = prim["model"]
    out["G4_build_SJ02"] = bool(g2 == "PASS" and out.get("G3_horizon") == "PASS")
    return out
