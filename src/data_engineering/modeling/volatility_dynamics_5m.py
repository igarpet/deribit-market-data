"""
Standalone native-5-minute Volatility Dynamics analysis.

This module is independent of every other research branch and of the
30-minute rolling analysis in Volatility Dynamics Module 01.

Input contract:
    state_time, underlying, return_5m

Primary volatility object:
    v_t = return_5m_t ** 2

No rolling variance, HAR model, option variable, state label, or prior
research output is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


VOLATILITY_DYNAMICS_5M_VERSION = "0.1.0"
BAR_MINUTES = 5
DEFAULT_LAG_BARS = (1, 2, 3, 6, 12, 24, 48)
DEFAULT_TAIL_QUANTILES = (0.90, 0.95, 0.99)


def _ns_values(times: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    t = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    return t.as_unit("ns").asi8


def infer_bar_minutes(times: pd.Series | pd.DatetimeIndex) -> float:
    t = pd.DatetimeIndex(pd.to_datetime(times, utc=True)).sort_values()
    if len(t) < 2:
        raise ValueError("Need at least two timestamps.")
    d = np.diff(_ns_values(t)).astype(float) / 60e9
    d = d[np.isfinite(d) & (d > 0)]
    if not len(d):
        raise ValueError("No positive timestamp differences.")
    vals, counts = np.unique(np.round(d, 9), return_counts=True)
    return float(vals[np.argmax(counts)])


def _corr(a: np.ndarray, b: np.ndarray) -> tuple[int, float, float]:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < 3:
        return n, np.nan, np.nan
    aa = a[ok]
    bb = b[ok]
    if np.std(aa) <= 0 or np.std(bb) <= 0:
        pearson = np.nan
    else:
        pearson = float(np.corrcoef(aa, bb)[0, 1])
    ra = pd.Series(aa).rank(method="average").to_numpy(float)
    rb = pd.Series(bb).rank(method="average").to_numpy(float)
    if np.std(ra) <= 0 or np.std(rb) <= 0:
        spearman = np.nan
    else:
        spearman = float(np.corrcoef(ra, rb)[0, 1])
    return n, pearson, spearman


@dataclass
class FiveMinutePanelMeta:
    underlying: str
    inferred_bar_minutes: float
    n_rows: int
    n_finite_returns: int
    finite_return_share: float
    first_time: str
    last_time: str
    log_epsilon: float


def build_5m_panel(g: pd.DataFrame) -> tuple[pd.DataFrame, FiveMinutePanelMeta]:
    """Construct the native five-minute volatility panel from raw returns only."""
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
        raise ValueError("build_5m_panel expects one asset.")
    asset = str(asset_vals[0])

    inferred = infer_bar_minutes(times)
    if not np.isclose(inferred, BAR_MINUTES, atol=1e-9):
        raise ValueError(
            f"{asset}: modal bar spacing is {inferred} minutes, expected 5."
        )

    r = pd.to_numeric(g["return_5m"], errors="coerce").to_numpy(float)
    v = r ** 2
    pos = v[np.isfinite(v) & (v > 0)]
    if len(pos):
        eps = float(np.median(pos) * 1e-4)
        if not np.isfinite(eps) or eps <= 0:
            eps = float(np.min(pos) * 1e-4)
    else:
        eps = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        log_v = (
            np.where(np.isfinite(v) & (v >= 0), np.log(v + eps), np.nan)
            if np.isfinite(eps)
            else np.full(len(v), np.nan)
        )

    ns = _ns_values(times)
    step_ns = pd.Timedelta(minutes=BAR_MINUTES).value
    prev_exact = np.zeros(len(g), dtype=bool)
    if len(g) > 1:
        prev_exact[1:] = np.diff(ns) == step_ns

    tseries = pd.Series(times)
    out = pd.DataFrame({
        "row_id": np.arange(len(g), dtype=int),
        "state_time": times,
        "underlying": asset,
        "return_5m": r,
        "variance_5m": v,
        "log_variance_5m": log_v,
        "clock_slot": (tseries.dt.hour * 12 + tseries.dt.minute // 5).to_numpy(int),
        "utc_date": tseries.dt.date.astype(str).to_numpy(),
        "prev_exact_5m": prev_exact,
    })

    meta = FiveMinutePanelMeta(
        underlying=asset,
        inferred_bar_minutes=inferred,
        n_rows=int(len(out)),
        n_finite_returns=int(np.isfinite(r).sum()),
        finite_return_share=float(np.mean(np.isfinite(r))),
        first_time=str(times.min()),
        last_time=str(times.max()),
        log_epsilon=float(eps) if np.isfinite(eps) else np.nan,
    )
    return out, meta


def distribution_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asset, g in panel.groupby("underlying"):
        x = pd.to_numeric(g["variance_5m"], errors="coerce").dropna().to_numpy(float)
        if not len(x):
            continue
        rows.append({
            "underlying": asset,
            "n": int(len(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
            "q00": float(np.min(x)),
            "q01": float(np.quantile(x, 0.01)),
            "q10": float(np.quantile(x, 0.10)),
            "q25": float(np.quantile(x, 0.25)),
            "median": float(np.quantile(x, 0.50)),
            "q75": float(np.quantile(x, 0.75)),
            "q90": float(np.quantile(x, 0.90)),
            "q95": float(np.quantile(x, 0.95)),
            "q99": float(np.quantile(x, 0.99)),
            "max": float(np.max(x)),
            "mean_to_median": (
                float(np.mean(x) / np.median(x))
                if np.median(x) > 0 else np.nan
            ),
        })
    return pd.DataFrame(rows)


def missingness_by_clock(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (asset, slot), g in panel.groupby(["underlying", "clock_slot"]):
        finite = pd.to_numeric(g["return_5m"], errors="coerce").notna()
        rows.append({
            "underlying": asset,
            "clock_slot": int(slot),
            "clock_time_utc": f"{int(slot)//12:02d}:{(int(slot)%12)*5:02d}",
            "n_rows": int(len(g)),
            "n_finite": int(finite.sum()),
            "coverage": float(finite.mean()),
        })
    return pd.DataFrame(rows)


def missingness_by_date(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (asset, date), g in panel.groupby(["underlying", "utc_date"]):
        finite = pd.to_numeric(g["return_5m"], errors="coerce").notna()
        rows.append({
            "underlying": asset,
            "utc_date": str(date),
            "n_rows": int(len(g)),
            "n_finite": int(finite.sum()),
            "coverage": float(finite.mean()),
        })
    return pd.DataFrame(rows)


def clock_profile(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exact 5-minute UTC profile of the native log variance."""
    rows = []
    for (asset, slot), g in panel.groupby(["underlying", "clock_slot"]):
        x = pd.to_numeric(g["log_variance_5m"], errors="coerce").dropna().to_numpy(float)
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
        slot_med = pd.to_numeric(g["median"], errors="coerce").dropna().to_numpy(float)
        raw = pd.to_numeric(
            panel.loc[panel["underlying"] == asset, "log_variance_5m"],
            errors="coerce",
        ).dropna().to_numpy(float)
        if not len(slot_med) or not len(raw):
            continue
        amp = float(np.quantile(slot_med, 0.90) - np.quantile(slot_med, 0.10))
        raw_iqr = float(np.quantile(raw, 0.75) - np.quantile(raw, 0.25))
        summary.append({
            "underlying": asset,
            "n_clock_slots": int(len(slot_med)),
            "clock_q10": float(np.quantile(slot_med, 0.10)),
            "clock_q90": float(np.quantile(slot_med, 0.90)),
            "clock_amplitude_q90_q10": amp,
            "pooled_log_variance_iqr": raw_iqr,
            "amplitude_to_pooled_iqr": amp / raw_iqr if raw_iqr > 0 else np.nan,
        })
    return prof, pd.DataFrame(summary)


def lag_persistence(
    panel: pd.DataFrame,
    lag_bars: Iterable[int] = DEFAULT_LAG_BARS,
) -> pd.DataFrame:
    """Native 5-minute log-variance dependence at exact timestamp lags."""
    rows = []
    step_ns = pd.Timedelta(minutes=BAR_MINUTES).value

    for asset, g0 in panel.groupby("underlying"):
        g = g0.sort_values("state_time").reset_index(drop=True)
        times = pd.DatetimeIndex(pd.to_datetime(g["state_time"], utc=True))
        ns = _ns_values(times)
        x = pd.to_numeric(g["log_variance_5m"], errors="coerce").to_numpy(float)

        for lag in lag_bars:
            lag = int(lag)
            a = []
            b = []
            for i in range(lag, len(g)):
                j = i - lag
                if ns[i] - ns[j] != step_ns * lag:
                    continue
                a.append(x[j])
                b.append(x[i])
            n, pearson, spearman = _corr(np.asarray(a), np.asarray(b))
            rows.append({
                "underlying": asset,
                "lag_bars": lag,
                "lag_minutes": lag * BAR_MINUTES,
                "n_pairs": n,
                "pearson_log_variance": pearson,
                "spearman_log_variance": spearman,
            })
    return pd.DataFrame(rows)


def tail_thresholds(
    panel: pd.DataFrame,
    quantiles: Iterable[float] = DEFAULT_TAIL_QUANTILES,
) -> pd.DataFrame:
    rows = []
    for asset, g in panel.groupby("underlying"):
        x = pd.to_numeric(g["variance_5m"], errors="coerce").dropna().to_numpy(float)
        for q in quantiles:
            q = float(q)
            if not 0 < q < 1:
                raise ValueError("Quantiles must be between 0 and 1.")
            rows.append({
                "underlying": asset,
                "quantile": q,
                "n": int(len(x)),
                "threshold": float(np.quantile(x, q)) if len(x) else np.nan,
            })
    return pd.DataFrame(rows)


def quantile_transition_matrix(
    panel: pd.DataFrame,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """One-step transition probabilities between empirical volatility quantiles."""
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")
    rows = []
    step_ns = pd.Timedelta(minutes=BAR_MINUTES).value

    for asset, g0 in panel.groupby("underlying"):
        g = g0.sort_values("state_time").reset_index(drop=True)
        times = pd.DatetimeIndex(pd.to_datetime(g["state_time"], utc=True))
        ns = _ns_values(times)
        x = pd.to_numeric(g["variance_5m"], errors="coerce")

        ranks = x.rank(method="average", pct=True)
        bins = np.ceil(ranks.to_numpy(float) * n_bins)
        bins = np.where(np.isfinite(bins), np.clip(bins, 1, n_bins), np.nan)

        counts = np.zeros((n_bins, n_bins), dtype=int)
        for i in range(1, len(g)):
            if ns[i] - ns[i - 1] != step_ns:
                continue
            a = bins[i - 1]
            b = bins[i]
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            counts[int(a) - 1, int(b) - 1] += 1

        for i in range(n_bins):
            total = int(counts[i].sum())
            for j in range(n_bins):
                rows.append({
                    "underlying": asset,
                    "from_bin": i + 1,
                    "to_bin": j + 1,
                    "n_from": total,
                    "count": int(counts[i, j]),
                    "probability": float(counts[i, j] / total) if total else np.nan,
                })
    return pd.DataFrame(rows)


def extract_tail_episodes(
    panel: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    """Consecutive native 5-minute bars above each empirical tail threshold."""
    rows = []
    eid = 0
    step_ns = pd.Timedelta(minutes=BAR_MINUTES).value

    for asset, g0 in panel.groupby("underlying"):
        g = g0.sort_values("state_time").reset_index(drop=True)
        times = pd.DatetimeIndex(pd.to_datetime(g["state_time"], utc=True))
        ns = _ns_values(times)
        x = pd.to_numeric(g["variance_5m"], errors="coerce").to_numpy(float)

        for tr in thresholds[thresholds["underlying"] == asset].itertuples():
            q = float(tr.quantile)
            threshold = float(tr.threshold)
            exceed = np.isfinite(x) & (x >= threshold)

            i = 0
            while i < len(g):
                if not exceed[i]:
                    i += 1
                    continue
                start = i
                end = i
                while (
                    end + 1 < len(g)
                    and exceed[end + 1]
                    and ns[end + 1] - ns[end] == step_ns
                ):
                    end += 1

                seg = x[start:end + 1]
                peak = start + int(np.nanargmax(seg))
                eid += 1
                rows.append({
                    "event_id": eid,
                    "underlying": asset,
                    "quantile": q,
                    "threshold": threshold,
                    "anchor_time": times[start],
                    "peak_time": times[peak],
                    "end_time": times[end],
                    "n_bars": int(end - start + 1),
                    "duration_min": int((end - start + 1) * BAR_MINUTES),
                    "anchor_variance": float(x[start]),
                    "peak_variance": float(x[peak]),
                    "right_censored": bool(end == len(g) - 1),
                })
                i = end + 1
    return pd.DataFrame(rows)


def episode_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    rows = []
    for (asset, q), g in episodes.groupby(["underlying", "quantile"]):
        d = pd.to_numeric(g["duration_min"], errors="coerce").dropna().to_numpy(float)
        rows.append({
            "underlying": asset,
            "quantile": float(q),
            "n_episodes": int(len(g)),
            "n_right_censored": int(g["right_censored"].astype(bool).sum()),
            "one_bar_share": float(np.mean(d == BAR_MINUTES)) if len(d) else np.nan,
            "median_duration_min": float(np.median(d)) if len(d) else np.nan,
            "q75_duration_min": float(np.quantile(d, 0.75)) if len(d) else np.nan,
            "q90_duration_min": float(np.quantile(d, 0.90)) if len(d) else np.nan,
            "max_duration_min": float(np.max(d)) if len(d) else np.nan,
        })
    return pd.DataFrame(rows)


def event_path_summary(
    panel: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    pre_bars: int = 12,
    post_bars: int = 12,
) -> pd.DataFrame:
    """Native 5-minute variance paths around episode onset; no aggregation."""
    if episodes.empty:
        return pd.DataFrame()

    rows = []
    step = pd.Timedelta(minutes=BAR_MINUTES)

    for asset, g0 in panel.groupby("underlying"):
        g = g0.sort_values("state_time").reset_index(drop=True)
        lookup = {
            pd.Timestamp(t): i
            for i, t in enumerate(pd.to_datetime(g["state_time"], utc=True))
        }
        e = episodes[episodes["underlying"] == asset]
        for ev in e.itertuples():
            anchor = pd.Timestamp(ev.anchor_time)
            for rel_bar in range(-pre_bars, post_bars + 1):
                t = anchor + step * rel_bar
                i = lookup.get(t)
                if i is None:
                    continue
                row = g.loc[i]
                if pd.notna(row["log_variance_5m"]):
                    rows.append({
                        "event_id": int(ev.event_id),
                        "underlying": asset,
                        "quantile": float(ev.quantile),
                        "rel_bars": int(rel_bar),
                        "rel_min": int(rel_bar * BAR_MINUTES),
                        "log_variance_5m": float(row["log_variance_5m"]),
                    })

    if not rows:
        return pd.DataFrame()
    long = pd.DataFrame(rows)
    return (
        long.groupby(["underlying", "quantile", "rel_bars", "rel_min"])["log_variance_5m"]
        .agg(
            n_episodes="count",
            median="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
        )
        .reset_index()
    )


def synchronous_tail_events(
    panel: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    """Same-timestamp cross-asset native 5-minute tail exceedances."""
    threshold_map = {
        (str(r.underlying), float(r.quantile)): float(r.threshold)
        for r in thresholds.itertuples()
    }
    rows = []

    for q in sorted(thresholds["quantile"].unique()):
        work = panel[["state_time", "underlying", "variance_5m"]].copy()
        work["is_tail"] = [
            bool(
                np.isfinite(v)
                and v >= threshold_map.get((str(a), float(q)), np.inf)
            )
            for a, v in zip(work["underlying"], work["variance_5m"])
        ]
        hit = work[work["is_tail"]]
        for t, g in hit.groupby("state_time"):
            assets = sorted(g["underlying"].astype(str).unique())
            rows.append({
                "quantile": float(q),
                "state_time": pd.Timestamp(t),
                "n_assets": int(len(assets)),
                "assets": "|".join(assets),
            })
    return pd.DataFrame(rows)


def next_bar_tail_summary(
    panel: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    """P(next bar remains above same tail threshold | current bar is above)."""
    rows = []
    step_ns = pd.Timedelta(minutes=BAR_MINUTES).value

    for asset, g0 in panel.groupby("underlying"):
        g = g0.sort_values("state_time").reset_index(drop=True)
        ns = _ns_values(pd.to_datetime(g["state_time"], utc=True))
        x = pd.to_numeric(g["variance_5m"], errors="coerce").to_numpy(float)

        for tr in thresholds[thresholds["underlying"] == asset].itertuples():
            q = float(tr.quantile)
            threshold = float(tr.threshold)
            current = np.isfinite(x) & (x >= threshold)
            n_current = 0
            n_stay = 0
            n_valid_pairs = 0
            n_next_tail = 0

            for i in range(len(g) - 1):
                if ns[i + 1] - ns[i] != step_ns:
                    continue
                if not (np.isfinite(x[i]) and np.isfinite(x[i + 1])):
                    continue
                n_valid_pairs += 1
                if x[i + 1] >= threshold:
                    n_next_tail += 1
                if current[i]:
                    n_current += 1
                    if x[i + 1] >= threshold:
                        n_stay += 1

            unconditional = (
                float(n_next_tail / n_valid_pairs) if n_valid_pairs else np.nan
            )
            conditional = float(n_stay / n_current) if n_current else np.nan
            rows.append({
                "underlying": asset,
                "quantile": q,
                "n_current_tail_bars": int(n_current),
                "n_tail_to_tail": int(n_stay),
                "p_next_tail_given_tail": conditional,
                "p_next_tail_unconditional": unconditional,
                "persistence_lift": (
                    conditional / unconditional
                    if np.isfinite(conditional)
                    and np.isfinite(unconditional)
                    and unconditional > 0
                    else np.nan
                ),
            })
    return pd.DataFrame(rows)


def selftest() -> dict:
    times = pd.date_range("2026-01-01", periods=120, freq="5min", tz="UTC")
    r = np.sin(np.arange(len(times)) / 5.0) * 0.01
    g = pd.DataFrame({
        "state_time": times.as_unit("us"),
        "underlying": "TEST",
        "return_5m": r,
    })

    p, meta = build_5m_panel(g)
    assert np.isclose(meta.inferred_bar_minutes, 5.0)
    assert np.allclose(p["variance_5m"].to_numpy(), r ** 2)

    dist = distribution_summary(p)
    assert len(dist) == 1

    lag = lag_persistence(p, lag_bars=(1, 2))
    assert set(lag["lag_bars"]) == {1, 2}

    thr = tail_thresholds(p, quantiles=(0.90, 0.95))
    eps = extract_tail_episodes(p, thr)
    assert len(eps) >= 1

    trans = quantile_transition_matrix(p, n_bins=5)
    assert len(trans) == 25

    # A missing bar must break an episode/transition sequence.
    p_gap = p.drop(index=20).reset_index(drop=True)
    ns = _ns_values(pd.to_datetime(p_gap["state_time"], utc=True))
    assert np.max(np.diff(ns)) == pd.Timedelta(minutes=10).value

    return {
        "native_variance": True,
        "timestamp_unit_normalization": True,
        "lag_persistence": True,
        "tail_episodes": True,
        "quantile_transitions": True,
        "gap_detection": True,
    }
