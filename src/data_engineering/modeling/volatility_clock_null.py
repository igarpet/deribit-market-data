"""
Volatility Dynamics 02b: is the native five-minute clock profile signal or
sampling noise?

Module 02 measured an exact-UTC-slot profile of log r^2 from roughly 16-17
finite observations per slot per asset. With log r^2 having a standard
deviation of about 2.2, slot medians from ~17 draws are noisy, and the
q90-q10 spread of 288 noisy medians is large even with no clock effect.

This module separates the two at two resolutions (exact 5m slot, UTC hour):

1. Rotation null: for every UTC day, rotate that day's clock labels by an
   independent random circular offset. Contiguous bars stay contiguous, so
   volatility clustering within a day is preserved, but a clock effect that
   repeats across days is destroyed. (A plain label shuffle is
   anti-conservative here: hourly medians are built from 12 contiguous bars
   and inherit volatility clustering, which a shuffle removes.)
2. Split-half reliability: correlation between the clock profiles estimated
   on even vs odd UTC days. A real profile reproduces; noise does not.
3. Signal share: observed amplitude in excess of the null mean, as a share
   of the observed amplitude.

Inputs: state_time, underlying, return_5m only. NumPy/Pandas only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


VERSION = "0.2.0"
BARS_PER_DAY = 288


def _medians_by_label(x: np.ndarray, labels: np.ndarray, n_labels: int) -> np.ndarray:
    out = np.full(n_labels, np.nan)
    order = np.argsort(labels, kind="stable")
    ls = labels[order]
    xs = x[order]
    bounds = np.searchsorted(ls, np.arange(n_labels + 1))
    for k in range(n_labels):
        seg = xs[bounds[k]:bounds[k + 1]]
        if len(seg):
            out[k] = np.median(seg)
    return out


def _amplitude(med: np.ndarray) -> float:
    m = med[np.isfinite(med)]
    if len(m) < 4:
        return np.nan
    return float(np.quantile(m, 0.90) - np.quantile(m, 0.10))


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame[["state_time", "underlying", "return_5m"]].copy()
    f["state_time"] = pd.to_datetime(f["state_time"], utc=True)
    r = pd.to_numeric(f["return_5m"], errors="coerce").to_numpy(float)
    v = r ** 2
    out = []
    for a, g in f.assign(v=v).groupby("underlying", sort=True):
        vv = g["v"].to_numpy(float)
        pos = vv[np.isfinite(vv) & (vv > 0)]
        eps = float(np.median(pos) * 1e-4) if len(pos) else np.nan
        lv = np.log(vv + eps)
        ns = pd.DatetimeIndex(g["state_time"]).as_unit("ns").asi8
        bar = ns // (300 * 10 ** 9)
        out.append(pd.DataFrame({
            "underlying": a,
            "log_variance_5m": lv,
            "slot": (bar % BARS_PER_DAY).astype(int),
            "hour": ((bar % BARS_PER_DAY) // 12).astype(int),
            "day": (bar // BARS_PER_DAY).astype(int),
        }))
    p = pd.concat(out, ignore_index=True)
    return p[np.isfinite(p["log_variance_5m"])].reset_index(drop=True)


def clock_null(
    panel: pd.DataFrame,
    *,
    n_perm: int = 500,
    seed: int = 202,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Observed vs permutation-null clock amplitude and split-half reliability."""
    rng = np.random.default_rng(seed)
    rows = []
    profiles = []
    for a, g in panel.groupby("underlying", sort=True):
        x = g["log_variance_5m"].to_numpy(float)
        q25, q75 = np.quantile(x, [0.25, 0.75])
        pooled_iqr = float(q75 - q25)
        for res, col, n_lab in (("slot_5m", "slot", BARS_PER_DAY), ("hour", "hour", 24)):
            lab = g[col].to_numpy(int)
            med = _medians_by_label(x, lab, n_lab)
            obs = _amplitude(med)
            slot_lab = g["slot"].to_numpy(int)
            day = g["day"].to_numpy(int)
            udays, dinv = np.unique(day, return_inverse=True)
            null = np.empty(int(n_perm))
            for b in range(int(n_perm)):
                off = rng.integers(0, BARS_PER_DAY, size=len(udays))[dinv]
                rot = (slot_lab + off) % BARS_PER_DAY
                rl = rot if col == "slot" else rot // 12
                null[b] = _amplitude(_medians_by_label(x, rl, n_lab))
            even = (g["day"].to_numpy(int) % 2) == 0
            m_even = _medians_by_label(x[even], lab[even], n_lab)
            m_odd = _medians_by_label(x[~even], lab[~even], n_lab)
            ok = np.isfinite(m_even) & np.isfinite(m_odd)
            split_corr = float(np.corrcoef(m_even[ok], m_odd[ok])[0, 1]) if ok.sum() > 3 else np.nan
            # Spearman-Brown: reliability of the full-sample profile
            sb = 2 * split_corr / (1 + split_corr) if np.isfinite(split_corr) and split_corr > -1 else np.nan
            n_per = np.bincount(lab, minlength=n_lab)
            null_mean = float(np.mean(null))
            rows.append({
                "underlying": a,
                "resolution": res,
                "n_labels": n_lab,
                "median_obs_per_label": float(np.median(n_per[n_per > 0])),
                "observed_amplitude_q90_q10": obs,
                "null_mean_amplitude": null_mean,
                "null_q95_amplitude": float(np.quantile(null, 0.95)),
                "perm_p_value": float((1 + np.sum(null >= obs)) / (len(null) + 1)),
                "excess_amplitude": obs - null_mean,
                "signal_share_of_amplitude": (obs - null_mean) / obs if obs > 0 else np.nan,
                "pooled_log_variance_iqr": pooled_iqr,
                "observed_amplitude_to_iqr": obs / pooled_iqr if pooled_iqr > 0 else np.nan,
                "split_half_corr_even_odd_days": split_corr,
                "spearman_brown_reliability": sb,
            })
            for k in range(n_lab):
                profiles.append({
                    "underlying": a,
                    "resolution": res,
                    "label": k,
                    "n": int(n_per[k]),
                    "median_log_variance": med[k],
                    "median_even_days": m_even[k],
                    "median_odd_days": m_odd[k],
                })
    return pd.DataFrame(rows), pd.DataFrame(profiles)


def selftest() -> dict:
    rng = np.random.default_rng(0)
    times = pd.date_range("2026-01-01", periods=20 * BARS_PER_DAY, freq="5min", tz="UTC")
    slot = np.arange(len(times)) % BARS_PER_DAY
    hour_effect = 0.8 * np.sin(2 * np.pi * slot / BARS_PER_DAY)
    frames = []
    # persistent volatility clustering in BOTH series; only CLOCK has a clock effect
    lv = np.zeros(len(times))
    for t in range(1, len(times)):
        lv[t] = 0.99 * lv[t - 1] + rng.normal(scale=0.1)
    for a, eff in (("FLAT", 0.0), ("CLOCK", 1.0)):
        sig = 0.002 * np.exp(eff * hour_effect + lv)
        frames.append(pd.DataFrame({
            "state_time": times.as_unit("us"),
            "underlying": a,
            "return_5m": sig * rng.normal(size=len(times)),
        }))
    s, _ = clock_null(prepare(pd.concat(frames)), n_perm=100)
    flat = s[(s.underlying == "FLAT") & (s.resolution == "hour")].iloc[0]
    clk = s[(s.underlying == "CLOCK") & (s.resolution == "hour")].iloc[0]
    assert flat.perm_p_value > 0.05
    assert clk.perm_p_value < 0.05
    assert clk.split_half_corr_even_odd_days > flat.split_half_corr_even_odd_days
    return {"flat_not_detected": True, "real_clock_detected": True, "split_half": True}
