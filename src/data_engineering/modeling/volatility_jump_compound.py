"""
Volatility Dynamics Module 09: compound-Poisson description of jumps.

Module 08 (VD 018) found that volatility-standardized jumps (|r~| > 4 local
sigma, two-sided jump-robust bipower) arrive like a homogeneous Poisson
process per asset, with no self-excitation and ~20% of jump bars shared by
several assets. Module 09 turns that into a compound-Poisson description that
an option model can use:

1. jump catalogue (Module 08 detection, unchanged: same functions, c = 4);
2. jump sizes: bps and local-sigma units, sign, tail (exponential vs GPD on
   the excess over the threshold);
3. arrival decomposition: market-wide (>= 2 assets in the same bar) versus
   idiosyncratic, with the co-jump count expected under independent Poisson
   arrivals as the reference;
4. tick check: jump size in index ticks (TRX had ~2x the other assets' rate);
5. P(at least one jump in the next tau) with exact (Garwood) intervals;
6. rate stability: first half vs second half of the sample.

Inputs: state_time, underlying, return_5m, futures_index_price.
NumPy/Pandas only. Branch-internal reuse: volatility_jump_hawkes (Module 08).

Pre-registered (fixed before any real-data Module 09 output)
-------------------------------------------------------------
P1 Common-jump component exists iff the number of bars with >= 2 assets
   jumping exceeds its expectation under independent per-asset Poisson
   arrivals with one-sided Poisson p < 0.01.
P2 An asset's jumps are a tick-discreteness artifact iff the median jump
   size is <= 3 index ticks (tick = the smallest positive |Delta index|
   observed for that asset, rounded). Such an asset is excluded from
   Module 10 with that reason.
P3 Tail: GPD vs exponential for the excess z - c by likelihood-ratio test
   (descriptive, reported only).
P4 Rate stability: first vs second half per asset, exact conditional
   binomial test (descriptive, Holm-adjusted across assets).
"""

from __future__ import annotations

from math import exp, lgamma, log, sqrt
from typing import Sequence

import numpy as np
import pandas as pd

from data_engineering.modeling import volatility_vd_b_core as vb
from data_engineering.modeling import volatility_jump_hawkes as jh


VERSION = "0.1.0"
C = jh.PRIMARY_C
TAU_HOURS = (1, 4, 8, 16, 24)
TICK_ARTIFACT_MAX_TICKS = 3.0
COJUMP_P = 0.01


# --------------------------------------------------------------------------
# Small exact distributions (no scipy)
# --------------------------------------------------------------------------

def _poisson_sf(k: int, mu: float) -> float:
    """P(X >= k), X ~ Poisson(mu)."""
    if k <= 0:
        return 1.0
    s = 0.0
    term = exp(-mu)
    for i in range(k):
        s += term
        term *= mu / (i + 1)
    return float(max(0.0, 1.0 - s))


def _chi2_ppf_via_gamma(p: float, k: float) -> float:
    """Quantile of chi2 with k dof via bisection on the regularized gamma."""
    lo, hi = 0.0, max(10.0, 10 * k)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _gammainc(k / 2.0, mid / 2.0) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _gammainc(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x) (series / continued fraction)."""
    if x <= 0:
        return 0.0
    if x < a + 1:
        term = 1.0 / a
        s = term
        n = a
        for _ in range(1000):
            n += 1
            term *= x / n
            s += term
            if abs(term) < abs(s) * 1e-14:
                break
        return float(s * exp(-x + a * log(x) - lgamma(a)))
    b = x + 1 - a
    c = 1e300
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1) < 1e-14:
            break
    return float(1.0 - exp(-x + a * log(x) - lgamma(a)) * h)


def garwood_interval(k: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact Poisson CI for the mean given k events."""
    lo = 0.0 if k == 0 else 0.5 * _chi2_ppf_via_gamma(alpha / 2, 2 * k)
    hi = 0.5 * _chi2_ppf_via_gamma(1 - alpha / 2, 2 * (k + 1))
    return lo, hi


def _binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    if n <= 0:
        return np.nan
    logs = np.array([lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) + i * log(p) + (n - i) * log(1 - p)
                     for i in range(n + 1)])
    probs = np.exp(logs)
    return float(min(1.0, probs[probs <= probs[k] * (1 + 1e-9)].sum()))


def _holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * p[j]))
        out[j] = run
    return out


# --------------------------------------------------------------------------
# Detection (Module 08, unchanged)
# --------------------------------------------------------------------------

def detect(grid: vb.Grid, c: float = C):
    panel = jh.build_panel(grid)
    train, _, _ = jh.chronological_rows(panel)
    S = jh.clock_factors(panel, train)
    Z, BV = jh.jump_statistics(panel, S)
    J, valid = jh.jump_indicator(Z, c)
    return panel, S, Z, BV, J, valid


def index_panel(grid: vb.Grid, panel: jh.Panel) -> np.ndarray:
    I = np.full((panel.T, panel.N), np.nan)
    if "futures_index_price" not in grid.frame:
        return I
    b0 = int(panel.bar[0])
    I[grid.bar - b0, grid.asset_code] = pd.to_numeric(grid.frame["futures_index_price"], errors="coerce").to_numpy(float)
    return I


def jump_catalog(panel, S, Z, BV, J, valid, I) -> pd.DataFrame:
    tt, aa = np.nonzero((J > 0) & valid)
    nsame = (J * valid).sum(axis=1)
    rows = []
    for t, a in zip(tt, aa):
        r = panel.R[t, a]
        local_sigma = float(np.sqrt(BV[t, a]) * S[panel.slot[t], a])     # back to raw-return units
        prev_i = I[t - 1, a] if t > 0 else np.nan
        rows.append({
            "underlying": panel.assets[a],
            "state_time": pd.Timestamp(int(panel.time_ns[t]), tz="UTC"),
            "bar_index": int(t),
            "return_5m": float(r),
            "abs_return_bps": float(abs(r) * 1e4),
            "sign": int(np.sign(r)),
            "z": float(Z[t, a]),
            "signed_z": float(np.sign(r) * Z[t, a]),
            "local_sigma_bps": local_sigma * 1e4,
            "index_price": float(I[t, a]),
            "abs_index_change": float(abs(I[t, a] - prev_i)) if np.isfinite(prev_i) and np.isfinite(I[t, a]) else np.nan,
            "n_assets_same_bar": int(nsame[t]),
            "hour_utc": int(panel.slot[t] // 12),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Sizes and tails
# --------------------------------------------------------------------------

def _gpd_negloglik(xi: float, sigma: float, x: np.ndarray) -> float:
    if sigma <= 0:
        return np.inf
    if abs(xi) < 1e-8:
        return len(x) * log(sigma) + x.sum() / sigma
    t = 1 + xi * x / sigma
    if np.any(t <= 0):
        return np.inf
    return len(x) * log(sigma) + (1 + 1 / xi) * np.log(t).sum()


def tail_fit(excess: np.ndarray) -> dict:
    """Exponential vs GPD on the excess over the threshold (profile over xi)."""
    x = np.asarray(excess, float)
    x = x[np.isfinite(x) & (x >= 0)]
    n = len(x)
    if n < 10:
        return {"n": n}
    sig_exp = float(x.mean())
    ll_exp = -(n * log(sig_exp) + x.sum() / sig_exp)
    best = (ll_exp, 0.0, sig_exp)
    for xi in np.linspace(-0.45, 1.5, 196):
        # profile sigma by golden-section on log scale
        lo, hi = log(sig_exp) - 4, log(sig_exp) + 4
        for _ in range(80):
            m1 = lo + (hi - lo) * 0.382
            m2 = lo + (hi - lo) * 0.618
            if _gpd_negloglik(xi, exp(m1), x) < _gpd_negloglik(xi, exp(m2), x):
                hi = m2
            else:
                lo = m1
        s = exp(0.5 * (lo + hi))
        ll = -_gpd_negloglik(xi, s, x)
        if ll > best[0]:
            best = (ll, float(xi), float(s))
    lr = 2 * (best[0] - ll_exp)
    p = float(1 - _gammainc(0.5, max(lr, 0) / 2))        # chi2(1) survival
    return {"n": n, "exp_scale": sig_exp, "loglik_exp": ll_exp,
            "gpd_xi": best[1], "gpd_scale": best[2], "loglik_gpd": best[0],
            "lr_stat": lr, "lr_p_chi2_1": p}


def size_summary(cat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in list(cat.groupby("underlying")) + [("ALL", cat)]:
        if g.empty:
            continue
        b = g["abs_return_bps"].to_numpy(float)
        z = g["z"].to_numpy(float)
        tf = tail_fit(z - C)
        rows.append({
            "underlying": name, "n_jumps": int(len(g)),
            "share_up": float(np.mean(g["sign"] > 0)),
            "bps_q25": float(np.quantile(b, 0.25)), "bps_median": float(np.median(b)),
            "bps_q75": float(np.quantile(b, 0.75)), "bps_q95": float(np.quantile(b, 0.95)), "bps_max": float(b.max()),
            "z_median": float(np.median(z)), "z_q90": float(np.quantile(z, 0.9)), "z_max": float(z.max()),
            "mean_excess_z": float(np.mean(z - C)),
            "gpd_xi": tf.get("gpd_xi", np.nan), "gpd_vs_exp_lr_p": tf.get("lr_p_chi2_1", np.nan),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Arrival decomposition and tick check
# --------------------------------------------------------------------------

def cojump_test(J: np.ndarray, valid: np.ndarray) -> dict:
    """Observed bars with >= 2 assets jumping vs expectation under independence."""
    V = valid.astype(float)
    Jv = J * V
    p_a = Jv.sum(axis=0) / np.maximum(V.sum(axis=0), 1)           # per-asset per-bar probability
    # expected P(>= 2 jumps in bar t) under independence, using the valid assets at t
    exp_count = 0.0
    for t in range(J.shape[0]):
        pa = p_a[V[t] > 0]
        if len(pa) < 2:
            continue
        q = np.prod(1 - pa)
        one = sum(pa[i] * q / (1 - pa[i]) for i in range(len(pa)))
        exp_count += 1 - q - one
    obs = int(((Jv.sum(axis=1)) >= 2).sum())
    return {"observed_bars_ge2": obs, "expected_bars_ge2_independent": float(exp_count),
            "ratio": obs / exp_count if exp_count > 0 else np.nan,
            "poisson_p_one_sided": _poisson_sf(obs, exp_count),
            "common_component": bool(_poisson_sf(obs, exp_count) < COJUMP_P)}


def arrival_rates(panel, J, valid, cat: pd.DataFrame) -> pd.DataFrame:
    """Per-asset total, common (bar shared with >= 1 other asset) and idiosyncratic rates per day."""
    rows = []
    for a, name in enumerate(panel.assets):
        n_bars = int(valid[:, a].sum())
        days = n_bars / jh.BARS_PER_DAY
        g = cat[cat.underlying == name]
        k = int(len(g))
        k_common = int((g["n_assets_same_bar"] >= 2).sum())
        lo, hi = garwood_interval(k)
        rows.append({
            "underlying": name, "valid_days": days, "n_jumps": k,
            "rate_per_day": k / days if days else np.nan,
            "rate_ci_low": lo / days if days else np.nan, "rate_ci_high": hi / days if days else np.nan,
            "n_common": k_common, "n_idiosyncratic": k - k_common,
            "common_share": k_common / k if k else np.nan,
        })
    return pd.DataFrame(rows)


def jump_probability_table(rates: pd.DataFrame, taus_h: Sequence[float] = TAU_HOURS) -> pd.DataFrame:
    rows = []
    for r in rates.itertuples():
        for h in taus_h:
            f = h / 24.0
            rows.append({
                "underlying": r.underlying, "tau_hours": h,
                "p_at_least_one_jump": 1 - exp(-r.rate_per_day * f),
                "p_ci_low": 1 - exp(-r.rate_ci_low * f), "p_ci_high": 1 - exp(-r.rate_ci_high * f),
            })
    return pd.DataFrame(rows)


def tick_check(grid: vb.Grid, cat: pd.DataFrame) -> pd.DataFrame:
    """Jump size in index ticks; tick = smallest positive |Delta index| per asset."""
    rows = []
    f = grid.frame
    for a, name in enumerate(grid.assets):
        ip = pd.to_numeric(f.loc[grid.asset_code == a, "futures_index_price"], errors="coerce").to_numpy(float) \
            if "futures_index_price" in f else np.array([])
        d = np.abs(np.diff(ip))
        d = d[np.isfinite(d) & (d > 0)]
        tick = float(np.min(d)) if len(d) else np.nan
        # robust tick: smallest value that occurs at least 5 times (guards a rounding outlier)
        if len(d):
            vals, cnt = np.unique(np.round(d, 12), return_counts=True)
            common = vals[cnt >= 5]
            tick = float(common.min()) if len(common) else tick
        g = cat[cat.underlying == name]
        ticks = g["abs_index_change"].to_numpy(float) / tick if np.isfinite(tick) and tick > 0 else np.full(len(g), np.nan)
        med = float(np.nanmedian(ticks)) if np.isfinite(ticks).any() else np.nan
        rows.append({
            "underlying": name, "index_tick": tick,
            "median_price": float(np.nanmedian(ip)) if len(ip) else np.nan,
            "tick_bps": tick / float(np.nanmedian(ip)) * 1e4 if len(ip) and np.isfinite(tick) else np.nan,
            "n_jumps": int(len(g)), "median_jump_ticks": med,
            "share_jumps_le3_ticks": float(np.nanmean(ticks <= 3)) if np.isfinite(ticks).any() else np.nan,
            "tick_artifact": bool(np.isfinite(med) and med <= TICK_ARTIFACT_MAX_TICKS),
        })
    return pd.DataFrame(rows)


def stability(panel, J, valid) -> pd.DataFrame:
    days = np.unique(panel.day[valid.any(axis=1)])
    mid = days[len(days) // 2]
    first = panel.day < mid
    rows = []
    for a, name in enumerate(panel.assets):
        n1, n2 = valid[first, a].sum(), valid[~first, a].sum()
        k1, k2 = int(J[first & valid[:, a], a].sum()), int(J[~first & valid[:, a], a].sum())
        p_exp = n1 / (n1 + n2) if n1 + n2 else 0.5
        rows.append({"underlying": name, "jumps_first_half": k1, "jumps_second_half": k2,
                     "rate_ratio_second_over_first": (k2 / n2) / (k1 / n1) if k1 and n1 and n2 else np.nan,
                     "binomial_p": _binom_two_sided(k1, k1 + k2, p_exp) if k1 + k2 else np.nan})
    out = pd.DataFrame(rows)
    out["holm_p"] = _holm(out["binomial_p"].to_numpy(float))
    return out


def run(grid: vb.Grid, c: float = C) -> dict:
    panel, S, Z, BV, J, valid = detect(grid, c)
    I = index_panel(grid, panel)
    cat = jump_catalog(panel, S, Z, BV, J, valid, I)
    rates = arrival_rates(panel, J, valid, cat)
    return {
        "panel": panel, "S": S, "Z": Z, "BV": BV, "J": J, "valid": valid,
        "catalog": cat, "sizes": size_summary(cat), "rates": rates,
        "probabilities": jump_probability_table(rates), "cojump": cojump_test(J, valid),
        "ticks": tick_check(grid, cat), "stability": stability(panel, J, valid),
    }


# --------------------------------------------------------------------------
# Self test
# --------------------------------------------------------------------------

def selftest() -> dict:
    # exact helpers
    lo, hi = garwood_interval(10)
    assert abs(lo - 4.7954) < 1e-3 and abs(hi - 18.3904) < 1e-3, (lo, hi)
    assert abs(_poisson_sf(3, 1.0) - 0.0803014) < 1e-6
    rng = np.random.default_rng(0)
    xe = rng.exponential(0.8, size=400)
    tf = tail_fit(xe)
    assert tf["lr_p_chi2_1"] > 0.01 and abs(tf["exp_scale"] - xe.mean()) < 1e-9
    tf2 = tail_fit((rng.pareto(2.5, size=600)) * 0.6)
    assert tf2["gpd_xi"] > 0.15
    # independent jumps: no common component; common jumps: detected
    f, _ = jh.simulate(n_days=21, N=7, base=0.002, seed=3)
    f["futures_index_price"] = 100 * np.exp(f.groupby("underlying")["return_5m"].cumsum())
    res = run(vb.build_grid(f))
    assert not res["cojump"]["common_component"], res["cojump"]
    T = 21 * 288
    R = np.random.default_rng(4).normal(scale=0.002, size=(T, 7))
    common = np.random.default_rng(5).random(T) < 0.002
    R[common] += 0.02 * np.random.default_rng(6).choice([-1, 1], size=(common.sum(), 1))
    times = pd.date_range("2026-03-01", periods=T, freq="5min", tz="UTC")
    g = pd.concat([pd.DataFrame({"state_time": times, "underlying": f"A{a}", "return_5m": R[:, a],
                                 "futures_index_price": 100 * np.exp(np.cumsum(R[:, a]))}) for a in range(7)])
    res2 = run(vb.build_grid(g))
    assert res2["cojump"]["common_component"], res2["cojump"]
    # tick artifact: an index that usually does not move within a bar, so a
    # single tick is many local sigmas (true move ~0.4 tick per bar)
    idx = np.round(1 + np.cumsum(np.random.default_rng(7).normal(scale=4e-5, size=T)), 4)
    h = pd.DataFrame({"state_time": times, "underlying": "COARSE", "return_5m": np.r_[np.nan, np.diff(np.log(idx))],
                      "futures_index_price": idx})
    r3 = run(vb.build_grid(h))
    assert r3["ticks"].iloc[0]["tick_artifact"], r3["ticks"].to_dict("records")
    assert not res2["ticks"]["tick_artifact"].any(), res2["ticks"].to_dict("records")
    return {"garwood": True, "poisson_sf": True, "tail_fit": True,
            "no_false_common_component": True, "common_component_detected": True, "tick_artifact_detected": True}
