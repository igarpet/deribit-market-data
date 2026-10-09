"""
Standalone Volatility Dynamics research engine.

This module is intentionally independent of every other research branch.
It imports no project modeling helpers and consumes no research outputs,
frozen specifications, thresholds, coefficients, or feature selections.

Module 01 uses only timestamp, asset identifier, and raw 5-minute index
returns from the shared research database to establish the volatility
process from first principles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


VOLATILITY_DYNAMICS_VERSION = "0.2.1"

DEFAULT_WINDOWS: dict[str, int] = {
    "5m": 1,
    "15m": 3,
    "30m": 6,
    "60m": 12,
    "4h": 48,
}
DEFAULT_FORWARD_WINDOWS: dict[str, int] = {
    "30m": 6,
    "60m": 12,
}
DEFAULT_HIGH_VOL_Q = 0.90
DEFAULT_PATH_PRE_MIN = 120
DEFAULT_PATH_POST_MIN = 180


def _num(frame: pd.DataFrame, col: str) -> np.ndarray:
    if col not in frame:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(float)


def _ns_values(times: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Return UTC timestamps as int64 nanoseconds, independent of Pandas storage unit.

    Pandas 3 may keep DatetimeIndex values at microsecond resolution. Direct
    use of .asi8 is therefore unsafe for comparisons against Timedelta.value,
    which is nanoseconds. Normalize explicitly before any integer arithmetic.
    """
    t = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    return t.as_unit("ns").asi8


def infer_bar_minutes(times: pd.Series | pd.DatetimeIndex) -> float:
    """Infer the modal positive spacing in minutes."""
    t = pd.DatetimeIndex(pd.to_datetime(times, utc=True)).sort_values()
    if len(t) < 2:
        raise ValueError("Need at least two timestamps to infer bar spacing.")
    d = np.diff(_ns_values(t)).astype(float) / 60e9
    d = d[np.isfinite(d) & (d > 0)]
    if not len(d):
        raise ValueError("No positive timestamp differences.")
    vals, counts = np.unique(np.round(d, 9), return_counts=True)
    return float(vals[np.argmax(counts)])


def _contiguous_back_sum(
    values: np.ndarray,
    times: pd.DatetimeIndex,
    bars: int,
    bar_min: float,
) -> np.ndarray:
    """Inclusive backward sum requiring a complete exact grid and finite values."""
    x = np.asarray(values, float)
    out = np.full(len(x), np.nan)
    delta = pd.Timedelta(minutes=float(bar_min))
    for i in range(bars - 1, len(x)):
        j = i - bars + 1
        if times[i] - times[j] != delta * (bars - 1):
            continue
        if bars > 1 and not np.all(
            np.diff(_ns_values(times[j:i + 1])) == delta.value
        ):
            continue
        w = x[j:i + 1]
        if np.isfinite(w).all():
            out[i] = float(np.sum(w))
    return out


def _contiguous_forward_sum(
    values: np.ndarray,
    times: pd.DatetimeIndex,
    bars: int,
    bar_min: float,
) -> np.ndarray:
    """Future sum over t+1..t+bars requiring a complete exact grid."""
    x = np.asarray(values, float)
    out = np.full(len(x), np.nan)
    delta = pd.Timedelta(minutes=float(bar_min))
    for i in range(0, len(x) - bars):
        j = i + 1
        k = i + bars
        if times[k] - times[i] != delta * bars:
            continue
        if not np.all(
            np.diff(_ns_values(times[i:k + 1])) == delta.value
        ):
            continue
        w = x[j:k + 1]
        if np.isfinite(w).all():
            out[i] = float(np.sum(w))
    return out


def _exact_lag(
    values: np.ndarray,
    times: pd.DatetimeIndex,
    bars: int,
    bar_min: float,
) -> np.ndarray:
    x = np.asarray(values, float)
    out = np.full(len(x), np.nan)
    delta = pd.Timedelta(minutes=float(bar_min))
    if bars <= 0:
        return x.copy()
    for i in range(bars, len(x)):
        j = i - bars
        if times[i] - times[j] == delta * bars:
            out[i] = x[j]
    return out


def _log_with_scale_epsilon(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Log transform with an asset/window-local numerical epsilon.

    Epsilon is a technical device only: 1e-4 of the median positive value.
    It is estimated independently for each asset/window inside this branch.
    """
    x = np.asarray(x, float)
    pos = x[np.isfinite(x) & (x > 0)]
    if not len(pos):
        return np.full(len(x), np.nan), np.nan
    eps = float(np.median(pos) * 1e-4)
    if not np.isfinite(eps) or eps <= 0:
        eps = float(np.min(pos) * 1e-4)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(np.isfinite(x) & (x >= 0), np.log(x + eps), np.nan)
    return y, eps


@dataclass
class VolatilityPanelMeta:
    underlying: str
    bar_minutes: float
    n_rows: int
    first_time: str
    last_time: str
    finite_return_share: float


def build_volatility_panel(
    g: pd.DataFrame,
    *,
    windows: Mapping[str, int] = DEFAULT_WINDOWS,
    forward_windows: Mapping[str, int] = DEFAULT_FORWARD_WINDOWS,
) -> tuple[pd.DataFrame, VolatilityPanelMeta, pd.DataFrame]:
    """Construct realized-variance objects from raw 5-minute index returns only."""
    required = {"state_time", "underlying", "return_5m"}
    missing = required - set(g.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if g.empty:
        raise ValueError("Empty asset frame.")

    g = g.sort_values("state_time").reset_index(drop=True).copy()
    times = pd.DatetimeIndex(pd.to_datetime(g["state_time"], utc=True))
    asset_vals = g["underlying"].dropna().astype(str).unique()
    if len(asset_vals) != 1:
        raise ValueError("build_volatility_panel expects exactly one asset.")
    asset = str(asset_vals[0])

    bar_min = infer_bar_minutes(times)
    r = _num(g, "return_5m")
    r2 = r ** 2

    out = pd.DataFrame({
        "row_id": np.arange(len(g), dtype=int),
        "state_time": times,
        "underlying": asset,
        "return_5m": r,
        "sq_return_5m": r2,
    })

    eps_rows = []
    for label, bars in windows.items():
        s = _contiguous_back_sum(r2, times, int(bars), bar_min)
        rate = s / float(bars)
        out[f"back_rv_{label}"] = s
        out[f"back_rate_{label}"] = rate
        lg, eps = _log_with_scale_epsilon(rate)
        out[f"log_back_rate_{label}"] = lg
        eps_rows.append({
            "underlying": asset,
            "direction": "backward",
            "window": label,
            "bars": int(bars),
            "epsilon": eps,
        })

    for label, bars in forward_windows.items():
        s = _contiguous_forward_sum(r2, times, int(bars), bar_min)
        rate = s / float(bars)
        out[f"fwd_rv_{label}"] = s
        out[f"fwd_rate_{label}"] = rate
        lg, eps = _log_with_scale_epsilon(rate)
        out[f"log_fwd_rate_{label}"] = lg
        eps_rows.append({
            "underlying": asset,
            "direction": "forward",
            "window": label,
            "bars": int(bars),
            "epsilon": eps,
        })

    # Purely backward-looking volatility-shape descriptors.
    out["slope_15v60"] = (
        out["log_back_rate_15m"].to_numpy(float)
        - out["log_back_rate_60m"].to_numpy(float)
    )
    out["slope_30v4h"] = (
        out["log_back_rate_30m"].to_numpy(float)
        - out["log_back_rate_4h"].to_numpy(float)
    )
    base = out["log_back_rate_30m"].to_numpy(float)
    for label, bars in (("15m", 3), ("30m", 6), ("60m", 12)):
        lag = _exact_lag(base, times, bars, bar_min)
        out[f"delta_log_back30_{label}"] = base - lag

    meta = VolatilityPanelMeta(
        underlying=asset,
        bar_minutes=bar_min,
        n_rows=int(len(out)),
        first_time=str(times.min()),
        last_time=str(times.max()),
        finite_return_share=float(np.mean(np.isfinite(r))),
    )
    return out, meta, pd.DataFrame(eps_rows)


def data_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asset, g in panel.groupby("underlying"):
        for col in [
            "return_5m",
            "back_rate_5m",
            "back_rate_15m",
            "back_rate_30m",
            "back_rate_60m",
            "back_rate_4h",
            "fwd_rate_30m",
            "fwd_rate_60m",
        ]:
            x = pd.to_numeric(g[col], errors="coerce")
            rows.append({
                "underlying": asset,
                "series": col,
                "n_rows": int(len(g)),
                "n_finite": int(x.notna().sum()),
                "coverage": float(x.notna().mean()),
            })
    return pd.DataFrame(rows)


def rv_distribution(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = [
        "back_rate_5m",
        "back_rate_15m",
        "back_rate_30m",
        "back_rate_60m",
        "back_rate_4h",
        "fwd_rate_30m",
        "fwd_rate_60m",
    ]
    for asset, g in panel.groupby("underlying"):
        for col in cols:
            x = pd.to_numeric(g[col], errors="coerce").dropna().to_numpy(float)
            if not len(x):
                continue
            rows.append({
                "underlying": asset,
                "series": col,
                "n": int(len(x)),
                "mean": float(np.mean(x)),
                "std": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
                "q01": float(np.quantile(x, 0.01)),
                "q10": float(np.quantile(x, 0.10)),
                "q25": float(np.quantile(x, 0.25)),
                "median": float(np.quantile(x, 0.50)),
                "q75": float(np.quantile(x, 0.75)),
                "q90": float(np.quantile(x, 0.90)),
                "q99": float(np.quantile(x, 0.99)),
                "max": float(np.max(x)),
            })
    return pd.DataFrame(rows)


def clock_profile(
    panel: pd.DataFrame,
    *,
    series: str = "log_back_rate_30m",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure UTC-clock structure internally; no external clock result is assumed."""
    work = panel[["underlying", "state_time", series]].copy()
    t = pd.to_datetime(work["state_time"], utc=True)
    work["clock_slot"] = (t.dt.hour * 12 + t.dt.minute // 5).astype(int)
    rows = []

    for (asset, slot), g in work.groupby(["underlying", "clock_slot"]):
        x = pd.to_numeric(g[series], errors="coerce").dropna().to_numpy(float)
        if not len(x):
            continue
        rows.append({
            "underlying": asset,
            "clock_slot": int(slot),
            "clock_time_utc": f"{int(slot)//12:02d}:{(int(slot)%12)*5:02d}",
            "n": int(len(x)),
            "median": float(np.median(x)),
            "q25": float(np.quantile(x, 0.25)),
            "q75": float(np.quantile(x, 0.75)),
        })
    prof = pd.DataFrame(rows)

    summary = []
    for asset, g in prof.groupby("underlying"):
        m = pd.to_numeric(g["median"], errors="coerce").dropna().to_numpy(float)
        raw = pd.to_numeric(
            panel.loc[panel["underlying"] == asset, series],
            errors="coerce",
        ).dropna().to_numpy(float)
        if not len(m) or not len(raw):
            continue
        amp = float(np.quantile(m, 0.90) - np.quantile(m, 0.10))
        raw_iqr = float(np.quantile(raw, 0.75) - np.quantile(raw, 0.25))
        summary.append({
            "underlying": asset,
            "n_clock_slots": int(len(m)),
            "slot_median_q10": float(np.quantile(m, 0.10)),
            "slot_median_q90": float(np.quantile(m, 0.90)),
            "clock_amplitude_q90_q10": amp,
            "pooled_iqr": raw_iqr,
            "clock_amplitude_to_pooled_iqr": amp / raw_iqr if raw_iqr > 0 else np.nan,
        })
    return prof, pd.DataFrame(summary)


def _corr(a: np.ndarray, b: np.ndarray) -> tuple[int, float]:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return int(ok.sum()), np.nan
    aa, bb = a[ok], b[ok]
    if np.std(aa) <= 0 or np.std(bb) <= 0:
        return int(ok.sum()), np.nan
    return int(ok.sum()), float(np.corrcoef(aa, bb)[0, 1])


def persistence_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Descriptive persistence and current-vs-future association.

    These are not trading forecasts and do not import any prior branch model.
    """
    rows = []
    for asset, g in panel.groupby("underlying"):
        g = g.sort_values("state_time").reset_index(drop=True)
        times = pd.DatetimeIndex(pd.to_datetime(g["state_time"], utc=True))
        bar_min = infer_bar_minutes(times)
        x = pd.to_numeric(g["log_back_rate_30m"], errors="coerce").to_numpy(float)

        for label, bars in (("5m", 1), ("15m", 3), ("30m", 6), ("60m", 12), ("4h", 48)):
            lag = _exact_lag(x, times, bars, bar_min)
            n, c = _corr(x, lag)
            rows.append({
                "underlying": asset,
                "relationship": f"back30_autocorr_{label}",
                "n": n,
                "corr": c,
            })

        for h in ("30m", "60m"):
            y = pd.to_numeric(g[f"log_fwd_rate_{h}"], errors="coerce").to_numpy(float)
            n, c = _corr(x, y)
            rows.append({
                "underlying": asset,
                "relationship": f"back30_vs_fwd{h}_all",
                "n": n,
                "corr": c,
            })
            stride = 6 if h == "30m" else 12
            idx = np.arange(len(g)) % stride == 0
            n2, c2 = _corr(x[idx], y[idx])
            rows.append({
                "underlying": asset,
                "relationship": f"back30_vs_fwd{h}_nonoverlap",
                "n": n2,
                "corr": c2,
            })
    return pd.DataFrame(rows)


def high_vol_thresholds(
    panel: pd.DataFrame,
    *,
    q: float = DEFAULT_HIGH_VOL_Q,
    series: str = "back_rate_30m",
) -> pd.DataFrame:
    if not 0.5 < q < 1.0:
        raise ValueError("q must be between 0.5 and 1.")
    rows = []
    for asset, g in panel.groupby("underlying"):
        x = pd.to_numeric(g[series], errors="coerce").dropna().to_numpy(float)
        rows.append({
            "underlying": asset,
            "series": series,
            "quantile": float(q),
            "n": int(len(x)),
            "threshold": float(np.quantile(x, q)) if len(x) else np.nan,
        })
    return pd.DataFrame(rows)


def extract_high_vol_episodes(
    panel: pd.DataFrame,
    thresholds: pd.DataFrame,
    *,
    series: str = "back_rate_30m",
) -> pd.DataFrame:
    """Retrospective contiguous episodes above an asset-specific threshold.

    An episode anchor is the first bar crossing the threshold. Peak is the
    within-run maximum. No forecast model or external state definition enters.
    """
    thr = {
        str(r.underlying): float(r.threshold)
        for r in thresholds.itertuples()
        if np.isfinite(float(r.threshold))
    }
    recs = []
    eid = 0

    for asset, g0 in panel.groupby("underlying"):
        g = g0.sort_values("state_time").reset_index()
        if asset not in thr:
            continue
        times = pd.DatetimeIndex(pd.to_datetime(g["state_time"], utc=True))
        bar_min = infer_bar_minutes(times)
        delta = pd.Timedelta(minutes=bar_min)
        x = pd.to_numeric(g[series], errors="coerce").to_numpy(float)
        high = np.isfinite(x) & (x >= thr[asset])

        i = 0
        while i < len(g):
            if not high[i]:
                i += 1
                continue
            start = i
            end = i
            while (
                end + 1 < len(g)
                and high[end + 1]
                and times[end + 1] - times[end] == delta
            ):
                end += 1

            seg = x[start:end + 1]
            peak_rel = int(np.nanargmax(seg))
            peak = start + peak_rel
            eid += 1

            anchor_global = int(g.loc[start, "index"])
            peak_global = int(g.loc[peak, "index"])
            recs.append({
                "event_id": eid,
                "underlying": asset,
                "anchor_panel_index": anchor_global,
                "peak_panel_index": peak_global,
                "anchor_time": times[start],
                "peak_time": times[peak],
                "end_time": times[end],
                "n_bars": int(end - start + 1),
                "duration_min": float((end - start + 1) * bar_min),
                "threshold": float(thr[asset]),
                "anchor_value": float(x[start]),
                "peak_value": float(x[peak]),
                "right_censored": bool(end == len(g) - 1),
                "anchor_slope_15v60": float(g.loc[start, "slope_15v60"])
                    if pd.notna(g.loc[start, "slope_15v60"]) else np.nan,
                "anchor_slope_30v4h": float(g.loc[start, "slope_30v4h"])
                    if pd.notna(g.loc[start, "slope_30v4h"]) else np.nan,
                "anchor_delta_back30_15m": float(g.loc[start, "delta_log_back30_15m"])
                    if pd.notna(g.loc[start, "delta_log_back30_15m"]) else np.nan,
                "anchor_delta_back30_30m": float(g.loc[start, "delta_log_back30_30m"])
                    if pd.notna(g.loc[start, "delta_log_back30_30m"]) else np.nan,
                "anchor_delta_back30_60m": float(g.loc[start, "delta_log_back30_60m"])
                    if pd.notna(g.loc[start, "delta_log_back30_60m"]) else np.nan,
            })
            i = end + 1

    return pd.DataFrame(recs)


def episode_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    rows = []
    for asset, g in episodes.groupby("underlying"):
        d = pd.to_numeric(g["duration_min"], errors="coerce").dropna().to_numpy(float)
        p = pd.to_numeric(g["peak_value"], errors="coerce").dropna().to_numpy(float)
        rows.append({
            "underlying": asset,
            "n_episodes": int(len(g)),
            "n_right_censored": int(g["right_censored"].astype(bool).sum()),
            "median_duration_min": float(np.median(d)) if len(d) else np.nan,
            "q75_duration_min": float(np.quantile(d, 0.75)) if len(d) else np.nan,
            "q90_duration_min": float(np.quantile(d, 0.90)) if len(d) else np.nan,
            "median_peak_value": float(np.median(p)) if len(p) else np.nan,
            "median_anchor_slope_15v60": float(np.nanmedian(g["anchor_slope_15v60"])),
            "median_anchor_slope_30v4h": float(np.nanmedian(g["anchor_slope_30v4h"])),
            "median_anchor_delta_back30_15m": float(np.nanmedian(g["anchor_delta_back30_15m"])),
            "median_anchor_delta_back30_30m": float(np.nanmedian(g["anchor_delta_back30_30m"])),
            "median_anchor_delta_back30_60m": float(np.nanmedian(g["anchor_delta_back30_60m"])),
        })
    return pd.DataFrame(rows)


def event_path_summary(
    panel: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    pre_min: int = DEFAULT_PATH_PRE_MIN,
    post_min: int = DEFAULT_PATH_POST_MIN,
) -> pd.DataFrame:
    """Aggregate raw volatility paths around episode onsets by asset."""
    if episodes.empty:
        return pd.DataFrame()

    vars_ = [
        "log_back_rate_5m",
        "log_back_rate_15m",
        "log_back_rate_30m",
        "log_back_rate_60m",
        "log_back_rate_4h",
        "slope_15v60",
        "slope_30v4h",
        "delta_log_back30_15m",
        "delta_log_back30_30m",
        "delta_log_back30_60m",
    ]
    recs = []

    for asset, g0 in panel.groupby("underlying"):
        g = g0.sort_values("state_time").reset_index(drop=True)
        times = pd.DatetimeIndex(pd.to_datetime(g["state_time"], utc=True))
        bar_min = infer_bar_minutes(times)
        lookup = {t: i for i, t in enumerate(times)}
        e = episodes[episodes["underlying"] == asset]
        for ev in e.itertuples():
            anchor = pd.Timestamp(ev.anchor_time)
            for rel in np.arange(-pre_min, post_min + 0.1, bar_min):
                t = anchor + pd.Timedelta(minutes=float(rel))
                i = lookup.get(t)
                if i is None:
                    continue
                for v in vars_:
                    val = g.loc[i, v]
                    if pd.notna(val):
                        recs.append({
                            "event_id": int(ev.event_id),
                            "underlying": asset,
                            "rel_min": float(rel),
                            "variable": v,
                            "value": float(val),
                        })
    if not recs:
        return pd.DataFrame()

    long = pd.DataFrame(recs)
    return (
        long.groupby(["underlying", "rel_min", "variable"])["value"]
        .agg(
            n_episodes="count",
            median="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
        )
        .reset_index()
    )


def selftest() -> dict:
    """Synthetic integrity tests with no external project dependencies."""
    times = pd.date_range("2026-01-01", periods=80, freq="5min", tz="UTC")
    r = np.linspace(-0.01, 0.01, len(times))
    g = pd.DataFrame({
        "state_time": times,
        "underlying": "TEST",
        "return_5m": r,
    })

    p, meta, eps = build_volatility_panel(g)
    assert abs(meta.bar_minutes - 5.0) < 1e-12

    # Simulate Pandas 3-style microsecond-backed timestamps explicitly.
    times_us = times.as_unit("us")
    assert abs(infer_bar_minutes(times_us) - 5.0) < 1e-12
    us_back = _contiguous_back_sum(r ** 2, times_us, 6, 5.0)
    assert abs(us_back[5] / 6.0 - np.mean(r[:6] ** 2)) < 1e-15
    assert np.isnan(p.loc[0:4, "back_rate_30m"]).all()
    expected_back = np.mean(r[:6] ** 2)
    assert abs(p.loc[5, "back_rate_30m"] - expected_back) < 1e-15
    expected_fwd = np.mean(r[1:7] ** 2)
    assert abs(p.loc[0, "fwd_rate_30m"] - expected_fwd) < 1e-15
    assert len(eps) == len(DEFAULT_WINDOWS) + len(DEFAULT_FORWARD_WINDOWS)

    thr = high_vol_thresholds(p, q=0.90)
    ev = extract_high_vol_episodes(p, thr)
    assert len(ev) >= 1

    # Introduce a gap; windows crossing it must be missing.
    g2 = g.drop(index=20).reset_index(drop=True)
    p2, _, _ = build_volatility_panel(g2)
    after_gap = p2.index[p2["state_time"] == times[21]][0]
    assert np.isnan(p2.loc[after_gap, "back_rate_30m"])

    return {
        "bar_inference": True,
        "timestamp_unit_normalization": True,
        "backward_rv": True,
        "forward_rv": True,
        "gap_guard": True,
        "episodes": True,
    }
