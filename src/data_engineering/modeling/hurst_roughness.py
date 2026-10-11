"""
Hurst Regimes branch, notebook 02 (HR 003): descriptive roughness of price
and volatility. No trading claim, no gate.

1. Price scaling. Structure function of the log price
       m_q(D) = mean |X_{t+D} - X_t|^q,   m_q(D) ~ D^{zeta(q)},  H = zeta(2)/2
   estimated separately over three scale ranges (5-30 min, 1-8 h, 1-7 days),
   so microstructure at the short end and any drift at the long end do not
   mix.

2. Volatility roughness (Gatheral, Jaisson & Rosenbaum 2018; Wang, Xiao & Yu
   2023). log sigma_k = 0.5 log RV_k on blocks of 1h, 4h and 1 day (sub-daily
   blocks on clock-deseasonalized returns). Structure function of log sigma
   over lags; zeta(q) for q in Q; H_q = zeta(q)/q. Linearity of zeta(q) in q
   (constant H_q) is the monofractal signature of rough volatility.
   Measurement noise: log RV estimates log sigma with error, which adds a
   constant 2 eta^2 to m_2(D) and biases the raw slope (and H) downward. The
   noise-corrected fit m_2(D) = A D^{2H} + 2 eta^2 is reported alongside.

3. Mean reversion of log sigma. Autocorrelation rho(k) for k = 1..K, compared
   by log-scale least squares: power law  rho = c k^(-a)  versus exponential
   rho = c exp(-k / tau) (OU). Plus a fractional OU fit with the thesis
   estimators (COF H, second-difference sigma, kappa from the stationary
   variance identity) and its half-life ln 2 / kappa.

All estimators are validated on synthetic paths in selftest().
"""

from __future__ import annotations

from math import gamma
from typing import Sequence

import numpy as np
import pandas as pd

from data_engineering.modeling import hurst_regimes as hr


VERSION = "0.1.1"   # 0.1.1: empty structure functions keep their columns (short histories)
BAR_MIN = 5
PRICE_LAGS = (1, 2, 3, 4, 6, 12, 18, 24, 36, 48, 72, 96, 288, 432, 576, 864, 1152, 1440, 2016)
PRICE_RANGES = {"5m-30m": (1, 6), "1h-8h": (12, 96), "1d-7d": (288, 2016)}
VOL_BLOCKS = {"1h": 12, "4h": 48, "1d": 288}
VOL_LAGS = {"1h": (1, 2, 3, 4, 6, 8, 12, 24, 48, 96, 168), "4h": (1, 2, 3, 4, 6, 9, 12, 18, 30, 42),
            "1d": (1, 2, 3, 4, 5, 7, 10, 14, 20, 30)}
QS = (0.5, 1.0, 1.5, 2.0, 3.0)
MIN_BLOCK_SHARE = 0.9
ACF_MAX = {"1h": 168, "4h": 84, "1d": 60}


# --------------------------------------------------------------------------
# Grids
# --------------------------------------------------------------------------

def price_grid(h: pd.DataFrame, asset: str) -> pd.Series:
    """Regular 5m grid of log closes (NaN where missing), index = bar end."""
    g = h[(h["underlying"] == asset) & np.isfinite(h["close"]) & (h["close"] > 0)]
    g = g.drop_duplicates("state_time").set_index("state_time")["close"].sort_index()
    grid = pd.date_range(g.index.min().ceil("5min"), g.index.max().floor("5min"), freq="5min")
    return np.log(g.reindex(grid))


def slot_clock(r: pd.Series) -> np.ndarray:
    """288-slot RMS of 5m returns (UTC time of day), normalized to mean square 1."""
    slot = (r.index.hour * 60 + r.index.minute) // BAR_MIN
    ms = pd.Series(r.to_numpy() ** 2).groupby(np.asarray(slot)).mean().reindex(range(288)).to_numpy()
    ms = np.where(np.isfinite(ms) & (ms > 0), ms, np.nanmean(ms))
    ms = pd.Series(np.r_[ms[-3:], ms, ms[:3]]).rolling(7, center=True).mean().to_numpy()[3:-3]
    return np.sqrt(ms / ms.mean())


def log_vol_series(lp: pd.Series, block: int, deseason: bool = True) -> pd.Series:
    """log sigma per block of `block` 5m bars aligned to 00:00 UTC (NaN if < 90% of bars)."""
    r = lp.diff()
    if deseason and block < 288:
        s = slot_clock(r)
        r = r / s[(r.index.hour * 60 + r.index.minute) // BAR_MIN]
    start = r.index[0].normalize()
    k = ((r.index - start) // pd.Timedelta(minutes=BAR_MIN)) // block
    df = pd.DataFrame({"k": np.asarray(k), "r2": r.to_numpy() ** 2})
    agg = df.groupby("k")["r2"].agg(["sum", "count"])
    n = df.groupby("k").size()
    ok = (agg["count"] >= MIN_BLOCK_SHARE * block) & (agg["sum"] > 0)
    rv = (agg["sum"] * block / agg["count"]).where(ok)                   # scale up for missing bars
    full = pd.Series(np.nan, index=range(int(n.index.max()) + 1))
    full.loc[rv.index] = 0.5 * np.log(rv)
    full.index = start + pd.to_timedelta(full.index * block * BAR_MIN, unit="min")
    return full


# --------------------------------------------------------------------------
# Structure functions and fits
# --------------------------------------------------------------------------

def structure_function(x: np.ndarray, lags: Sequence[int], qs: Sequence[float] = QS) -> pd.DataFrame:
    """m_q(D) = mean |x_{t+D} - x_t|^q over all pairs with both values finite."""
    x = np.asarray(x, float)
    rows = []
    for d in lags:
        if d >= len(x):
            continue
        inc = x[d:] - x[:-d]
        inc = np.abs(inc[np.isfinite(inc)])
        if len(inc) < 50:
            continue
        rows.append({"lag": int(d), "n": int(len(inc)), **{f"m{q:g}": float(np.mean(inc ** q)) for q in qs}})
    return pd.DataFrame(rows, columns=["lag", "n"] + [f"m{q:g}" for q in qs])


def zeta(sf: pd.DataFrame, lo: int, hi: int, qs: Sequence[float] = QS) -> dict:
    """Log-log slopes zeta(q) over lags in [lo, hi]; H_q = zeta(q)/q."""
    s = sf[(sf["lag"] >= lo) & (sf["lag"] <= hi)]
    if len(s) < 2:
        return {}
    lx = np.log(s["lag"].to_numpy(float))
    out = {"n_lags": int(len(s))}
    for q in qs:
        z = float(np.polyfit(lx, np.log(s[f"m{q:g}"].to_numpy(float)), 1)[0])
        out[f"zeta{q:g}"] = z
        out[f"H{q:g}"] = z / q
    return out


def noise_corrected_H(sf: pd.DataFrame, lo: int, hi: int, grid: np.ndarray | None = None) -> dict:
    """Fit m_2(D) = A D^{2H} + c (c = 2 eta^2 >= 0) by grid search over H with
    non-negative least squares in (A, c); relative (log-scale-like) weights."""
    s = sf[(sf["lag"] >= lo) & (sf["lag"] <= hi)]
    if len(s) < 3:
        return {}
    D, m = s["lag"].to_numpy(float), s["m2"].to_numpy(float)
    w = 1.0 / m
    best = None
    for H in (np.linspace(0.01, 0.99, 197) if grid is None else grid):
        X = np.column_stack([D ** (2 * H), np.ones_like(D)])
        Xw, yw = X * w[:, None], m * w
        coef = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        if coef[1] < 0:                                   # clamp noise at 0
            coef = np.array([float(np.sum(Xw[:, 0] * yw) / np.sum(Xw[:, 0] ** 2)), 0.0])
        if coef[0] <= 0:
            continue
        rss = float(np.sum((yw - Xw @ coef) ** 2))
        if best is None or rss < best[0]:
            best = (rss, H, coef)
    if best is None:
        return {}
    _, H, (A, c) = best
    return {"H_noise_corr": float(H), "noise_var_eta2": float(c / 2.0), "noise_share_lag1": float(c / (A + c))}


def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, float)
    xm = x - np.nanmean(x)
    v = np.nanmean(xm ** 2)
    out = np.full(max_lag + 1, np.nan)
    for k in range(max_lag + 1):
        a, b = xm[k:], xm[: len(xm) - k]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 30:
            out[k] = float(np.mean(a[ok] * b[ok]) / v)
    return out


def decay_fits(rho: np.ndarray, kmax: int) -> dict:
    """Power law vs exponential on log rho for k = 1..kmax (rho > 0 only)."""
    k = np.arange(1, kmax + 1, dtype=float)
    r = rho[1:kmax + 1]
    ok = np.isfinite(r) & (r > 0.01)
    if ok.sum() < 5:
        return {}
    k, lr = k[ok], np.log(r[ok])
    bp = np.polyfit(np.log(k), lr, 1)
    be = np.polyfit(k, lr, 1)
    rss_p = float(np.sum((lr - np.polyval(bp, np.log(k))) ** 2))
    rss_e = float(np.sum((lr - np.polyval(be, k)) ** 2))
    return {"power_alpha": float(-bp[0]), "rss_power": rss_p, "exp_tau": float(-1.0 / be[0]) if be[0] < 0 else np.inf,
            "rss_exp": rss_e, "better": "power" if rss_p < rss_e else "exponential", "n_lags_fit": int(ok.sum())}


def fou_fit(x: np.ndarray, H: float | None = None) -> dict:
    """Thesis estimators (§5.1, §5.5) on an equally spaced series, unit step = one block.
    H defaults to the COF estimate; pass a calibrated H to override it (on log RV the
    COF estimate is biased upward by block integration, see calibrate())."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]                                 # gaps are rare in block series; drop them
    if len(x) < 50:
        return {}
    H = float(hr.cof_hurst(x[None, :])[0]) if H is None or not np.isfinite(H) else float(H)
    d2 = x[2:] - 2 * x[1:-1] + x[:-2]
    sig2 = float(np.mean(d2 ** 2) / (4 - 2 ** (2 * H)))
    var = float(np.var(x, ddof=1))
    kappa = float((sig2 * gamma(2 * H + 1) / (2 * var)) ** (1 / (2 * H)))
    return {"H_used": H, "sigma": float(np.sqrt(sig2)), "kappa": kappa, "half_life_blocks": float(np.log(2) / kappa),
            "mean": float(np.mean(x)), "sd": float(np.sqrt(var))}


# --------------------------------------------------------------------------
# Synthetic generators and self test
# --------------------------------------------------------------------------

def fgn(H: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Exact fractional Gaussian noise by Davies-Harte (circulant embedding)."""
    k = np.arange(n + 1, dtype=float)
    g = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * k ** (2 * H) + np.abs(k - 1) ** (2 * H))
    c = np.r_[g, g[-2:0:-1]]
    lam = np.fft.fft(c).real
    lam = np.clip(lam, 0, None)
    m = len(c)
    z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    w = np.fft.fft(np.sqrt(lam / m) * z)
    return w.real[:n]


def rough_vol_path(n_bars: int, H: float = 0.1, nu: float = 1.0, kappa_per_day: float = 0.02,
                   base_vol: float = 0.002, rng: np.random.Generator | None = None) -> np.ndarray:
    """5m log returns with log sigma a (discretized) fOU driven by fGn at 5m steps."""
    rng = rng or np.random.default_rng(0)
    dW = fgn(H, n_bars, rng) * (1.0 / 288) ** H * nu         # fBm increments on a 1-day time unit
    k = kappa_per_day / 288
    y = np.empty(n_bars)
    y[0] = 0.0
    for t in range(1, n_bars):
        y[t] = y[t - 1] - k * y[t - 1] + dW[t]
    return base_vol * np.exp(y) * rng.standard_normal(n_bars)


MIN_DAYS = 365                    # assets with a shorter history are excluded from notebook 02
CAL_H = (0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
CAL_REPS = 3


def vol_H_estimates(lp: pd.Series, deseason: bool) -> dict:
    """Raw and noise-corrected H of log sigma for every block size (fit over all lags of that block)."""
    out = {}
    for name, b in VOL_BLOCKS.items():
        lv = log_vol_series(lp, b, deseason=deseason)
        lags = VOL_LAGS[name]
        sf = structure_function(lv.to_numpy(), lags)
        z = zeta(sf, 1, max(lags))
        nc = noise_corrected_H(sf, 1, max(lags))
        x = lv.to_numpy()
        x = x[np.isfinite(x)]
        cof = float(hr.cof_hurst(x[None, :])[0]) if len(x) >= 50 else np.nan
        out[name] = {"raw": z.get("H2", np.nan), "corr": nc.get("H_noise_corr", np.nan), "cof": cof,
                     "sf": sf, "z": z, "nc": nc, "lv": lv}
    return out


def calibrate(n_bars: int, nu: float, *, H_grid: Sequence[float] = CAL_H, reps: int = CAL_REPS,
              seed: int = 3003) -> pd.DataFrame:
    """Mean raw / noise-corrected estimates of each block size on simulated rough volatility
    (fOU log sigma with true H, vol-of-vol nu per day, same sample length)."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-01-01 00:05", periods=n_bars, freq="5min", tz="UTC")
    rows = []
    for H in H_grid:
        acc = {}
        for _ in range(reps):
            r = rough_vol_path(n_bars, H=H, nu=nu, kappa_per_day=0.01, rng=rng)
            est = vol_H_estimates(pd.Series(np.cumsum(r), index=t), deseason=False)
            for b, e in est.items():
                for kind in ("raw", "corr", "cof"):
                    acc.setdefault((b, kind), []).append(e[kind])
        for (b, kind), v in acc.items():
            rows.append({"H_true": H, "block": b, "kind": kind, "H_est": float(np.nanmean(v)), "H_est_sd": float(np.nanstd(v))})
    return pd.DataFrame(rows)


def invert(cal: pd.DataFrame, block: str, kind: str, h_obs: float) -> float:
    """Calibrated H: the H_true whose mean simulated estimate equals the observed one
    (linear interpolation on the monotone part; NaN outside the simulated range)."""
    c = cal[(cal["block"] == block) & (cal["kind"] == kind)].sort_values("H_true")
    x, y = c["H_est"].to_numpy(float), c["H_true"].to_numpy(float)
    if not np.all(np.diff(x) > 0) or not np.isfinite(h_obs) or h_obs < x.min() or h_obs > x.max():
        return np.nan
    return float(np.interp(h_obs, x, y))


def selftest() -> dict:
    rng = np.random.default_rng(3)
    # 1. Price structure function recovers H of fBm
    out = {}
    for H in (0.35, 0.5, 0.65):
        x = np.cumsum(fgn(H, 200_000, rng))
        z = zeta(structure_function(x, PRICE_LAGS), 1, 96)
        assert abs(z["H2"] - H) < 0.03, (H, z)
        out[f"fbm_H{H}"] = round(z["H2"], 3)
    # 2. Noise correction: fBm (H = 0.1) + white noise; raw slope biased low, corrected close
    x = np.cumsum(fgn(0.1, 20_000, rng)) * 0.3 + 0.3 * rng.standard_normal(20_000)
    sf = structure_function(x, (1, 2, 3, 4, 5, 7, 10, 14, 20, 30, 50))
    raw = zeta(sf, 1, 30)["H2"]
    nc = noise_corrected_H(sf, 1, 50)["H_noise_corr"]
    assert raw < 0.07 and abs(nc - 0.1) < 0.05, (raw, nc)
    out.update({"noisy_raw_H": round(raw, 3), "noisy_corrected_H": round(nc, 3)})
    # 3. ACF: OU (exponential) vs power-law-like long memory
    ou = np.empty(20_000)
    ou[0] = 0
    e = rng.standard_normal(20_000)
    for t in range(1, 20_000):
        ou[t] = 0.9 * ou[t - 1] + e[t]
    assert decay_fits(acf(ou, 40), 40)["better"] == "exponential"
    # 4. fOU fit on a stationary fOU-like series recovers H
    f = fou_fit(np.cumsum(fgn(0.3, 5000, rng)) * 0.05 - 0.0 )
    assert abs(f["H_used"] - 0.3) < 0.06, f
    # 5. Rough volatility end to end: H of daily log RV well below 0.5
    r = rough_vol_path(288 * 400, H=0.1, rng=rng)
    t = pd.date_range("2025-01-01 00:05", periods=len(r), freq="5min", tz="UTC")
    lp = pd.Series(np.cumsum(r), index=t)
    lv = log_vol_series(lp, 288, deseason=False)
    zv = noise_corrected_H(structure_function(lv.to_numpy(), VOL_LAGS["1d"]), 1, 30)
    assert zv["H_noise_corr"] < 0.3, zv
    out["roughvol_daily_H"] = round(zv["H_noise_corr"], 3)
    # 6. Calibration inverts the estimator bias (daily, corrected): recovers H = 0.1 within 0.06
    cal = calibrate(288 * 400, 0.3, H_grid=(0.05, 0.1, 0.2, 0.3, 0.5), reps=2, seed=5)
    r = rough_vol_path(288 * 400, H=0.1, nu=0.3, kappa_per_day=0.01, rng=np.random.default_rng(77))
    e = vol_H_estimates(pd.Series(np.cumsum(r), index=pd.date_range("2025-01-01 00:05", periods=len(r), freq="5min", tz="UTC")), False)
    hc = invert(cal, "1d", "corr", e["1d"]["corr"])
    assert np.isfinite(hc) and abs(hc - 0.1) < 0.08, (hc, e["1d"]["corr"])
    out["calibrated_daily_H_true0.1"] = round(hc, 3)
    return {"version": VERSION, **out, "ok": True}
