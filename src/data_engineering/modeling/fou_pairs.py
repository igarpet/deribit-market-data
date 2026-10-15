"""Pairs fOU 01 (Stage A): Ivan's thesis pipeline (fOU statistical arbitrage) at 5 minutes on crypto perpetuals.

Protocol: discussions/Pairs_fOU/001_claude_stage_a_preregistration.md.
Source: Garcia Petrov (2026), "Fractional Ornstein-Uhlenbeck Statistical Arbitrage with Synthetic Equity Options".

Thesis -> 5-minute translation (Ivan's choices 2026-10-16): rolling 7-day formation refit every UTC day,
4-hour (48-bar) maximum conditional horizon, perpetual legs (options only if this stage shows a gross edge).

Per pair (fixed orientation: alphabetically first asset is y) and trade day d, formation = previous 2016 bars:
    integration screen  ADF(level) not rejected at 5%, ADF(diff) rejected at 5%, both legs
    cointegration       log y = a + b log x (OLS); Engle-Granger p <= 0.01; b > 0
    Hurst               H = 0.5 log2( sum D2^2 / sum D1^2 ), second differences at lag 1 and 2, clipped [0.01, 0.99]
    fOU                 mu = mean X; sigma^2 = mean[(D1)^2] / (4 - 2^{2H}); kappa = (sigma^2 Gamma(2H+1) / (2 Var X))^{1/(2H)}
    eligibility         H < 0.5, 0 < kappa < 2, structural P(tau <= 252 bars | Z0 = 1.5) >= 0.70
Trading on day d (parameters frozen for the day; positions may run past midnight on their entry-day parameters):
    signal at bar close t when flat and |Z_t| >= 1.5; conditional fGn simulation (last 60 inferred innovations,
    Gaussian conditioning) of 48 future increments, N paths; T70 = first T with P(first passage to mu by T) >= 0.70;
    admit if 1 < T70 <= 48; execute at the NEXT bar close; legs: -sign(Z) on y, +sign(Z) * b on x (unit y notional);
    exit at the close after convergence is first observed, or at bar t + T70 (the thesis 'expiry'), whichever first.
P&L per trade in bp of gross notional (1 + b); costs 6 bp per side per leg -> 12 bp round trip of gross notional.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import gamma

import numpy as np
import pandas as pd

VERSION = "PF01-0.1.0"
STEP = pd.Timedelta(minutes=5)


@dataclass(frozen=True)
class Spec:
    version: str = VERSION
    assets: tuple = ("BTC", "ETH", "SOL", "XRP")
    formation_bars: int = 2016            # 7 days of 5m bars
    entry_z: float = 1.5
    target_p: float = 0.70
    max_horizon: int = 48                 # 4 hours
    structural_horizon: int = 252         # thesis 252-step structural cap (here 21 h)
    memory: int = 60                      # inferred fGn innovations conditioned on
    paths: int = 5000
    structural_paths: int = 2000
    coint_p: float = 0.01
    adf_p: float = 0.05
    cost_bp_per_side_leg: float = 6.0     # 0.05% taker + 1 bp slippage
    test_start: str = "2025-10-01"        # dev = before, test = from (both reported; gate on test)
    n_boot: int = 4000
    seed: int = 42
    status: str = "HISTORICAL_EXPLORATORY"


def registration(spec: Spec | None = None) -> dict:
    return asdict(spec or Spec())


# ---------------------------------------------------------------- data

def log_close_5m(df1: pd.DataFrame) -> pd.Series:
    """Log close of complete 5m bars (index = bar END) from 1m klines (open_time, close)."""
    q = df1[["open_time", "close"]].copy()
    q["open_time"] = pd.to_datetime(q["open_time"], utc=True)
    q = q.drop_duplicates("open_time").sort_values("open_time")
    q["end"] = q["open_time"].dt.floor("5min") + STEP
    g = q.groupby("end").agg(n=("open_time", "size"), first=("open_time", "min"), close=("close", "last"))
    ok = (g["n"] == 5) & (g["first"] == g.index - STEP) & (pd.to_numeric(g["close"]) > 0)
    return np.log(pd.to_numeric(g.loc[ok, "close"]))


def panel(series: dict) -> pd.DataFrame:
    """Dense 5m grid; NaN where an asset's bar is incomplete."""
    df = pd.DataFrame(series).sort_index()
    grid = pd.date_range(df.index.min(), df.index.max(), freq=STEP)
    return df.reindex(grid)


def pairs(assets) -> list:
    a = sorted(assets)
    return [(a[i], a[j]) for i in range(len(a)) for j in range(i + 1, len(a))]


# ---------------------------------------------------------------- estimation

def hurst_cof(x: np.ndarray) -> float:
    d1 = x[2:] - 2 * x[1:-1] + x[:-2]
    d2 = x[4:] - 2 * x[2:-2] + x[:-4]
    num, den = np.sum(d2 ** 2), np.sum(d1 ** 2)
    if den <= 0 or num <= 0:
        return np.nan
    return float(np.clip(0.5 * np.log2(num / den), 0.01, 0.99))


def fou_fit(x: np.ndarray) -> dict:
    H = hurst_cof(x)
    mu, var = float(np.mean(x)), float(np.var(x))
    d1 = x[2:] - 2 * x[1:-1] + x[:-2]
    denom = 4.0 - 2.0 ** (2 * H) if np.isfinite(H) else np.nan
    s2 = float(np.mean(d1 ** 2) / denom) if np.isfinite(denom) and denom > 0 else np.nan
    kappa = (s2 * gamma(2 * H + 1) / (2 * var)) ** (1 / (2 * H)) if (np.isfinite(s2) and var > 0) else np.nan
    return {"H": H, "mu": mu, "var": var, "sd": float(np.sqrt(var)), "sigma": float(np.sqrt(s2)) if np.isfinite(s2) else np.nan,
            "kappa": float(kappa) if np.isfinite(kappa) else np.nan,
            "half_life_bars": float(np.log(2) / kappa) if np.isfinite(kappa) and kappa > 0 else np.nan}


def screen(y: np.ndarray, x: np.ndarray, spec: Spec) -> dict:
    """Integration screen + fixed-orientation Engle-Granger; returns dict with pass flags and a, b, residuals."""
    import warnings
    from statsmodels.tsa.stattools import adfuller, coint
    warnings.filterwarnings("ignore", module="statsmodels")
    warnings.filterwarnings("ignore", category=FutureWarning)
    out = {"integration_ok": False, "coint_ok": False}
    try:
        for s in (y, x):
            if adfuller(s, autolag="AIC")[1] <= spec.adf_p or adfuller(np.diff(s), autolag="AIC")[1] > spec.adf_p:
                return out
        out["integration_ok"] = True
        X = np.column_stack([np.ones(len(x)), x])
        a, b = np.linalg.lstsq(X, y, rcond=None)[0]
        p = coint(y, x, trend="c", autolag="AIC")[1]
    except Exception as exc:                     # numerical failure = not eligible
        out["error"] = repr(exc)
        return out
    out.update({"a": float(a), "b": float(b), "coint_p": float(p),
                "coint_ok": bool(p <= spec.coint_p and b > 0), "resid": y - a - b * x})
    return out


# ---------------------------------------------------------------- fractional Gaussian noise

def fgn_acov(H: float, n: int) -> np.ndarray:
    k = np.arange(n, dtype=float)
    return 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k - 1) ** (2 * H))


def fgn_cov(H: float, n: int) -> np.ndarray:
    g = fgn_acov(H, n)
    i = np.arange(n)
    return g[np.abs(i[:, None] - i[None, :])]


def conditional_operator(H: float, m: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Future n fGn increments given past m: mean = A @ past (oldest first), cov = L L^T."""
    S = fgn_cov(H, m + n)
    S11, S12, S22 = S[:m, :m], S[:m, m:], S[m:, m:]
    A = np.linalg.solve(S11 + 1e-10 * np.eye(m), S12).T
    C = S22 - A @ S12
    C = 0.5 * (C + C.T) + 1e-10 * np.eye(n)
    return A, np.linalg.cholesky(C)


def first_passage(x0: float, mu: float, kappa: float, sigma: float, inc: np.ndarray) -> np.ndarray:
    """Euler fOU paths from x0 with fGn increments inc (paths x n). Returns first step (1..n) reaching mu, or
    n + 1 if none. Direction: from above -> X <= mu; from below -> X >= mu."""
    paths, n = inc.shape
    x = np.full(paths, x0, float)
    tau = np.full(paths, n + 1, int)
    above = x0 > mu
    for k in range(n):
        x = x + kappa * (mu - x) + sigma * inc[:, k]
        hit = (x <= mu) if above else (x >= mu)
        new = hit & (tau == n + 1)
        tau[new] = k + 1
    return tau


def t70(tau: np.ndarray, n: int, p: float) -> tuple[int | None, float]:
    """Earliest T in 1..n with P(tau <= T) >= p; also P(tau <= n)."""
    cdf = np.array([(tau <= T).mean() for T in range(1, n + 1)])
    idx = np.flatnonzero(cdf >= p)
    return (int(idx[0] + 1) if len(idx) else None), float(cdf[-1])


def structural_ok(fit: dict, spec: Spec, rng) -> tuple[bool, int | None]:
    n = spec.structural_horizon
    L = np.linalg.cholesky(fgn_cov(fit["H"], n) + 1e-10 * np.eye(n))
    inc = rng.standard_normal((spec.structural_paths, n)) @ L.T
    x0 = fit["mu"] + spec.entry_z * fit["sd"]
    T, _ = t70(first_passage(x0, fit["mu"], fit["kappa"], fit["sigma"], inc), n, spec.target_p)
    return T is not None, T


def innovations(xhist: np.ndarray, fit: dict) -> np.ndarray:
    """eps_s = [X_s - X_{s-1} - kappa (mu - X_{s-1})] / sigma for the given history (len m+1 -> m innovations)."""
    return (np.diff(xhist) - fit["kappa"] * (fit["mu"] - xhist[:-1])) / fit["sigma"]


# ---------------------------------------------------------------- backtest

def run_pair(P: pd.DataFrame, y: str, x: str, spec: Spec | None = None, *, mode: str = "fou",
             days=None, progress=None, screen_cache: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """mode: 'fou' (thesis), 'ou' (same, but H fixed at 0.5 for dynamics and no H filter),
    'zscore' (cointegrated days only; enter |Z| >= 1.5, exit at mu or after max_horizon; no simulation).
    Returns (trades, forecasts, day_fits)."""
    spec = spec or Spec()
    ly, lx = P[y].to_numpy(float), P[x].to_numpy(float)
    t = P.index
    day = t.floor("D")
    udays = pd.DatetimeIndex(day.unique()).sort_values()
    if days is not None:
        udays = udays[udays.isin(pd.DatetimeIndex(days))]
    rng = np.random.default_rng(spec.seed)
    trades, forecasts, fits = [], [], []
    open_until = -1                                       # bar index before which the pair is busy
    for d in udays:
        i0 = int(np.searchsorted(t, d))                   # first bar of day d (bar ending 00:05 or later)
        i1 = int(np.searchsorted(t, d + pd.Timedelta(days=1)))
        f0 = i0 - spec.formation_bars
        if f0 < 0:
            continue
        fy, fx = ly[f0:i0], lx[f0:i0]
        rec = {"pair": f"{y}-{x}", "day": d, "mode": mode}
        if not (np.isfinite(fy).all() and np.isfinite(fx).all()):
            fits.append({**rec, "eligible": False, "reason": "gaps"}); continue
        key = (y, x, d)
        if screen_cache is not None and key in screen_cache:
            sc = screen_cache[key]
        else:
            sc = screen(fy, fx, spec)
            if screen_cache is not None:
                screen_cache[key] = sc
        rec.update({k: sc.get(k) for k in ("integration_ok", "coint_ok", "coint_p", "a", "b")})
        if not sc["coint_ok"]:
            fits.append({**rec, "eligible": False, "reason": "no_coint"}); continue
        fit = fou_fit(sc["resid"])
        if mode == "ou":
            fit = {**fit, "H": 0.5}
            d1 = sc["resid"][2:] - 2 * sc["resid"][1:-1] + sc["resid"][:-2]
            fit["sigma"] = float(np.sqrt(np.mean(d1 ** 2) / 2.0))
            fit["kappa"] = fit["sigma"] ** 2 / (2 * fit["var"])
        rec.update({k: fit[k] for k in ("H", "kappa", "sigma", "sd", "half_life_bars")})
        if mode == "fou" and not (fit["H"] < 0.5):
            fits.append({**rec, "eligible": False, "reason": "H>=0.5"}); continue
        if mode in ("fou", "ou"):
            if not (np.isfinite(fit["kappa"]) and 0 < fit["kappa"] < 2 and fit["sigma"] > 0 and fit["var"] > 0):
                fits.append({**rec, "eligible": False, "reason": "params"}); continue
            ok, Ts = structural_ok(fit, spec, rng)
            rec["T70_structural"] = Ts
            if not ok:
                fits.append({**rec, "eligible": False, "reason": "structural"}); continue
            A, L = conditional_operator(fit["H"], spec.memory, spec.max_horizon)
            W = rng.standard_normal((spec.paths, spec.max_horizon)) @ L.T       # common random numbers per pair-day
        fits.append({**rec, "eligible": True, "reason": ""})
        a, b, mu, sd = sc["a"], sc["b"], fit["mu"], fit["sd"]
        X = ly - a - b * lx                                                     # spread on this day's frozen fit
        i = max(i0, open_until)
        while i < i1:
            if not np.isfinite(X[i]) or abs(X[i] - mu) < spec.entry_z * sd:
                i += 1; continue
            z = (X[i] - mu) / sd
            if mode in ("fou", "ou"):
                hist = X[i - spec.memory:i + 1]
                if len(hist) != spec.memory + 1 or not np.isfinite(hist).all():
                    i += 1; continue
                cm = A @ innovations(hist, fit)
                tau = first_passage(X[i], mu, fit["kappa"], fit["sigma"], W + cm[None, :])
                T, p_max = t70(tau, spec.max_horizon, spec.target_p)
                pT = float((tau <= T).mean()) if T else np.nan
            else:
                T, p_max, pT = spec.max_horizon, np.nan, np.nan
            fc = {"pair": f"{y}-{x}", "mode": mode, "signal_bar": t[i], "z": float(z), "T70": T,
                  "p_at_T": pT, "p_max": p_max, "admitted": bool(T is not None and T > 1)}
            # realised first passage over (i, i+T] on the frozen spread (calibration)
            if fc["admitted"]:
                seg = X[i + 1:i + 1 + T]
                hit = (seg <= mu) if z > 0 else (seg >= mu)
                fc["complete"] = bool(len(seg) == T and np.isfinite(seg).all())
                fc["converged"] = bool(hit.any())
            forecasts.append(fc)
            if not fc["admitted"]:
                i += 1; continue
            e = i + 1                                                           # execute at next close
            expiry = i + T
            if e >= len(X) or not np.isfinite(ly[e]) or not np.isfinite(lx[e]):
                i += 1; continue
            k, exit_i, why = e, None, None
            while k < len(X) and k <= expiry:
                if k > e and np.isfinite(X[k]) and ((X[k] <= mu) if z > 0 else (X[k] >= mu)):
                    exit_i, why = min(k + 1, expiry), "converge"
                    break
                k += 1
            if exit_i is None:
                exit_i, why = expiry, "expiry"
            if exit_i >= len(X) or not (np.isfinite(ly[exit_i]) and np.isfinite(lx[exit_i])):
                j = exit_i
                while j < len(X) and not (np.isfinite(ly[j]) and np.isfinite(lx[j])):
                    j += 1
                if j >= len(X):
                    break
                exit_i, why = j, why + "_gapfill"
            s = -np.sign(z)
            gross = s * ((ly[exit_i] - ly[e]) - b * (lx[exit_i] - lx[e])) / (1 + b)
            trades.append({"pair": f"{y}-{x}", "mode": mode, "day": d, "signal_bar": t[i], "entry_bar": t[e],
                           "exit_bar": t[exit_i], "bars_held": exit_i - e, "z": float(z), "b": b, "T70": T,
                           "exit": why, "gross_bp": 1e4 * gross,
                           "net_bp": 1e4 * gross - 2 * spec.cost_bp_per_side_leg})
            open_until = exit_i + 1
            i = exit_i + 1
        if progress:
            progress(d)
    return pd.DataFrame(trades), pd.DataFrame(forecasts), pd.DataFrame(fits)


# ---------------------------------------------------------------- evaluation

def day_boot(v_by_day: pd.Series, reps: int, seed: int) -> tuple[float, float]:
    v = v_by_day.to_numpy(float)
    if len(v) < 10:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (reps, len(v)))].mean(1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))


def trade_summary(tr: pd.DataFrame, spec: Spec, label: str) -> dict:
    """Per-trade means with a bootstrap over UTC days (trade-weighted: resample days, average their trades)."""
    if tr.empty:
        return {"label": label, "trades": 0}
    g = tr.groupby(tr["entry_bar"].dt.floor("D"))
    rng = np.random.default_rng(spec.seed)
    keys = list(g.groups)
    sums = g[["gross_bp", "net_bp"]].sum(); cnt = g.size()
    S, C = sums.to_numpy(), cnt.to_numpy()
    idx = rng.integers(0, len(keys), (spec.n_boot, len(keys)))
    bs = S[idx].sum(1) / C[idx].sum(1)[:, None]
    lo, hi = np.quantile(bs, [.025, .975], axis=0)
    return {"label": label, "trades": int(len(tr)), "days_with_trades": int(len(keys)),
            "gross_bp": float(tr.gross_bp.mean()), "gross_lo": float(lo[0]), "gross_hi": float(hi[0]),
            "net_bp": float(tr.net_bp.mean()), "net_lo": float(lo[1]), "net_hi": float(hi[1]),
            "win_rate_net": float((tr.net_bp > 0).mean()), "converge_share": float(tr.exit.str.startswith("converge").mean()),
            "mean_bars_held": float(tr.bars_held.mean())}


def calibration(fc: pd.DataFrame) -> dict:
    a = fc[fc.get("admitted", False) == True] if len(fc) else fc
    if a.empty or "complete" not in a:
        return {"forecasts": 0}
    c = a[a.complete == True]
    y = c.converged.astype(float).to_numpy(); p = c.p_at_T.to_numpy(float)
    ok = np.isfinite(p)
    return {"forecasts": int(len(a)), "complete": int(len(c)), "mean_predicted": float(np.nanmean(p)),
            "observed": float(y.mean()) if len(y) else np.nan,
            "brier": float(np.mean((p[ok] - y[ok]) ** 2)) if ok.any() else np.nan}


def decision(summ: pd.DataFrame, synthetic=False) -> dict:
    if synthetic:
        return {"verdict": "SYNTHETIC_NO_MARKET_INFERENCE"}
    r = summ[(summ.label == "fou|test|all")]
    if r.empty or not r.iloc[0].get("trades", 0):
        return {"verdict": "NO_TRADES"}
    r = r.iloc[0]
    v = ("NET_EDGE_CANDIDATE" if r.net_lo > 0 else "GROSS_EDGE_ONLY" if r.gross_lo > 0 else
         "NO_EDGE" if r.gross_hi < 0 or r.net_hi < 0 else "INCONCLUSIVE")
    return {"verdict": v, "stage_b_unlocked": bool(r.gross_lo > 0), "test_trades": int(r.trades),
            "gross_bp": [r.gross_bp, r.gross_lo, r.gross_hi], "net_bp": [r.net_bp, r.net_lo, r.net_hi],
            "status": Spec().status}


# ---------------------------------------------------------------- synthetic

def synthetic_prices(n_days: int = 20, seed: int = 0, kappa: float = 0.05, H: float = 0.5, beta: float = 1.2,
                     spread_sd: float = 0.004):
    """x is a random walk; y = a + beta x + OU/fGn spread. Returns a panel with columns AAA (y) and BBB (x)."""
    rng = np.random.default_rng(seed)
    n = n_days * 288
    t = pd.date_range("2026-01-01 00:05", periods=n, freq="5min", tz="UTC")
    lx = np.log(100) + np.cumsum(0.0015 * rng.standard_normal(n))
    if H == 0.5:
        e = rng.standard_normal(n)
    else:                                   # Davies-Harte-free small sample: chunked Cholesky fGn
        e = np.concatenate([rng.standard_normal(500) @ np.linalg.cholesky(fgn_cov(H, 500)).T for _ in range(n // 500 + 1)])[:n]
    sig = spread_sd * np.sqrt(2 * kappa) if kappa > 0 else 0.1 * spread_sd   # kappa = 0: random-walk spread
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = s[i - 1] - kappa * s[i - 1] + sig * e[i]
    ly = 0.3 + beta * lx + s
    return pd.DataFrame({"AAA": ly, "BBB": lx}, index=t)
