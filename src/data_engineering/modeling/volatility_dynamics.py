"""
Volatility Dynamics research branch.

Module 01 builds causal expanding-window HAR volatility forecasts and uses
them to describe expected high-volatility episodes, positive volatility
surprises, forecast-high misses, and their surrounding market conditions.

This module is descriptive. Retrospective surprise/event labels are never
used as origin-time predictors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from data_engineering.modeling import vol_har as vh
from data_engineering.modeling import vol_targets as vt


VOLATILITY_DYNAMICS_VERSION = "0.1.0"
DEFAULT_MIN_TRAIN_SESSIONS = 7
DEFAULT_EVENT_Q = 0.90
DEFAULT_CLUSTER_GAP_MIN = 30
PROTECTED_CONFIRMATION_START = "2026-10-09"

DIAGNOSTIC_SPECS = {
    "abs_return_5m": ("return_5m", "abs"),
    "log_volume_usd": ("volume_usd", "log1p"),
    "log_trade_count": ("trade_count", "log1p"),
    "abs_tfi": ("trade_flow_imbalance", "abs"),
    "signed_volume": ("signed_volume_usd", "signed_log1p"),
    "abs_close_vs_vwap_bps": ("close_vs_vwap_bps", "abs"),
    "abs_perp_basis_bps": ("perpetual_mid_vs_index_bps", "abs"),
    "oi_return": ("open_interest_return_5m", "identity"),
    "log_spread_bps": ("book_twa_spread_bps", "log"),
    "abs_obi_20": ("book_twa_obi_20", "abs"),
    "abs_microprice_mid_bps": ("microprice_vs_mid_bps", "abs"),
    "log_atm_iv": ("option_iv_atm_forward", "log"),
    "atm_iv_d5": ("atm_iv_change_5m", "identity"),
    "atm_iv_d15": ("atm_iv_change_15m", "identity"),
    "atm_iv_d30": ("atm_iv_change_30m", "identity"),
    "log_option_spread_med": ("option_median_two_sided_spread_bps", "log"),
}


def _num(frame: pd.DataFrame, col: str) -> np.ndarray:
    if col not in frame:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(float)


def _transform(x: np.ndarray, kind: str) -> np.ndarray:
    x = np.asarray(x, float)
    if kind == "identity":
        return x
    if kind == "abs":
        return np.abs(x)
    if kind == "log1p":
        return np.where(x >= 0, np.log1p(x), np.nan)
    if kind == "log":
        return np.where(x > 0, np.log(x), np.nan)
    if kind == "signed_log1p":
        return np.sign(x) * np.log1p(np.abs(x))
    raise ValueError(f"Unknown transform {kind!r}")


def diagnostic_frame(g: pd.DataFrame) -> pd.DataFrame:
    """Compact contemporaneous diagnostics. Missing sources stay NaN."""
    out = pd.DataFrame(index=np.arange(len(g)))
    for name, (src, kind) in DIAGNOSTIC_SPECS.items():
        out[name] = _transform(_num(g, src), kind)

    bid = _num(g, "book_twa_bid_depth_20")
    ask = _num(g, "book_twa_ask_depth_20")
    depth = bid + ask
    out["log_total_depth20"] = np.where(
        np.isfinite(bid) & np.isfinite(ask) & (depth >= 0),
        np.log1p(depth),
        np.nan,
    )
    return out


def _causal_lag(values, times, sessions, bars: int) -> np.ndarray:
    """Exact-bar lag, requiring continuity and the same expiry session."""
    x = np.asarray(values, float)
    out = np.full(len(x), np.nan)
    if bars <= 0 or len(x) <= bars:
        return out
    t = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    sess = np.asarray(sessions)
    i = np.arange(bars, len(x))
    j = i - bars
    good = ((t[i] - t[j]) == pd.Timedelta(vt.BAR) * bars) & (sess[i] == sess[j])
    out[i[good]] = x[j[good]]
    return out


def _trailing_mean(values, times, sessions, bars: int) -> np.ndarray:
    """Trailing inclusive mean over exact contiguous bars in one session."""
    x = np.asarray(values, float)
    out = np.full(len(x), np.nan)
    if bars <= 0:
        return out
    t = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    sess = np.asarray(sessions)
    for i in range(bars - 1, len(x)):
        j = i - bars + 1
        if sess[j] != sess[i]:
            continue
        if (t[i] - t[j]) != pd.Timedelta(vt.BAR) * (bars - 1):
            continue
        w = x[j:i + 1]
        if np.isfinite(w).all():
            out[i] = float(w.mean())
    return out


def expanding_m1_state_panel(
    g: pd.DataFrame,
    *,
    min_train_sessions: int = DEFAULT_MIN_TRAIN_SESSIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Causal expanding-session M1 forecasts for one asset.

    Each evaluation session is forecast from nuisance quantities and an
    HAR+clock M1 fitted only on strictly earlier expiry sessions. M1 excludes
    IV, so the physical-volatility forecast remains independent of option IV.
    """
    if g.empty:
        raise ValueError("empty asset frame")
    g = g.sort_values("state_time").reset_index(drop=True).copy()
    g["state_time"] = pd.to_datetime(g["state_time"], utc=True)

    panel = vt.build_panel(g)
    sessions = sorted(pd.unique(panel.session_date))
    if len(sessions) <= min_train_sessions:
        raise ValueError(
            f"{panel.asset}: only {len(sessions)} sessions; need > {min_train_sessions}"
        )

    out = pd.DataFrame({
        "raw_index": np.arange(len(g), dtype=int),
        "state_id": g["state_id"].to_numpy() if "state_id" in g else np.arange(len(g)),
        "state_time": g["state_time"].to_numpy(),
        "underlying": panel.asset,
        "session": panel.session_date,
        "tte_min": panel.tte_min,
    })

    for w, k in vt.BACK_WINDOWS.items():
        rate = panel.back_rv[w] / float(k)
        out[f"back_rate_{w}"] = rate
        out[f"log_back_rate_{w}"] = np.where(rate > 0, np.log(rate), np.nan)

    raw_X = pd.DataFrame(index=np.arange(len(g)))
    fit_rows = []
    # Pre-create forecast columns so a failed fit produces NaNs rather than
    # changing the output schema or breaking downstream diagnostics.
    for h in vt.HORIZONS:
        for prefix in ("forecast_level_eps", "forecast_rv", "actual_rv", "eps", "surprise"):
            out[f"{prefix}_{h}"] = np.nan

    for si, session in enumerate(sessions):
        if si < min_train_sessions:
            continue
        train_sessions = set(sessions[:si])
        eval_rows = np.asarray(panel.session_date == session)
        try:
            nuis = vh.fit_nuisance(g, panel, train_sessions)
            mat = vh.materialize(panel, nuis, raw_X)
        except Exception as exc:
            fit_rows.append({
                "underlying": panel.asset,
                "session": session,
                "horizon": "ALL",
                "n_train_sessions": len(train_sessions),
                "n_train": 0,
                "status": f"nuisance_error:{type(exc).__name__}",
            })
            continue

        tr = np.isin(mat.sessions, sorted(train_sessions))
        for h in vt.HORIZONS:
            try:
                fv = vh.fit_vol(mat, h, "M1", [], tr)
                yhat = vh.predict_log(fv, mat, h)
                F = np.exp(yhat) * fv.smear
                eps = float(nuis.eps_target[h])
                actual = panel.fwd_rv[h]
                actual_level = actual + eps
                pred_rv = np.maximum(F - eps, 0.0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    surprise = np.log(actual_level / F)

                out.loc[eval_rows, f"forecast_level_eps_{h}"] = F[eval_rows]
                out.loc[eval_rows, f"forecast_rv_{h}"] = pred_rv[eval_rows]
                out.loc[eval_rows, f"actual_rv_{h}"] = actual[eval_rows]
                out.loc[eval_rows, f"eps_{h}"] = eps
                out.loc[eval_rows, f"surprise_{h}"] = surprise[eval_rows]

                fit_rows.append({
                    "underlying": panel.asset,
                    "session": session,
                    "horizon": h,
                    "n_train_sessions": len(train_sessions),
                    "n_train": int(fv.n_train),
                    "smear": float(fv.smear),
                    "eps": eps,
                    "status": "ok",
                })
            except Exception as exc:
                fit_rows.append({
                    "underlying": panel.asset,
                    "session": session,
                    "horizon": h,
                    "n_train_sessions": len(train_sessions),
                    "n_train": 0,
                    "status": f"fit_error:{type(exc).__name__}",
                })

    for h, bars in vt.HORIZONS.items():
        F = pd.to_numeric(out.get(f"forecast_level_eps_{h}"), errors="coerce").to_numpy(float)
        A = pd.to_numeric(out.get(f"actual_rv_{h}"), errors="coerce").to_numpy(float)
        out[f"forecast_rate_eps_{h}"] = F / float(bars)
        out[f"actual_rate_{h}"] = A / float(bars)
        out[f"log_forecast_rate_eps_{h}"] = np.where(
            F > 0, np.log(F / float(bars)), np.nan
        )
        out[f"log_actual_rate_{h}"] = np.where(
            A > 0, np.log(A / float(bars)), np.nan
        )

    f30 = out["log_forecast_rate_eps_30m"].to_numpy(float)
    f60 = out["log_forecast_rate_eps_60m"].to_numpy(float)
    out["forecast_term_slope_60v30"] = f60 - f30

    for bars, label in ((3, "15m"), (6, "30m")):
        lag = _causal_lag(f30, out["state_time"], out["session"], bars)
        out[f"forecast_accel_30m_{label}"] = f30 - lag

    out["back_term_slope_30v4h"] = (
        out["log_back_rate_30m"].to_numpy(float)
        - out["log_back_rate_4h"].to_numpy(float)
    )
    return out, pd.DataFrame(fit_rows)


def robust_clock_z(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    min_count: int = 8,
) -> list[str]:
    """Append robust exact-five-minute UTC clock z-scores by asset."""
    t = pd.to_datetime(frame["state_time"], utc=True)
    slot = (t.dt.hour * 12 + t.dt.minute // 5).to_numpy()
    asset = frame["underlying"].astype(str).to_numpy()
    zcols = []

    for col in columns:
        x = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
        z = np.full(len(frame), np.nan)

        for a in pd.unique(asset):
            ia = np.flatnonzero(asset == a)
            xa = x[ia]
            if np.isfinite(xa).any():
                med_a = float(np.nanmedian(xa))
                scale_a = float(1.4826 * np.nanmedian(np.abs(xa - med_a)))
                if not np.isfinite(scale_a) or scale_a <= 1e-12:
                    scale_a = float(np.nanstd(xa))
                if not np.isfinite(scale_a) or scale_a <= 1e-12:
                    scale_a = 1.0
            else:
                med_a, scale_a = 0.0, 1.0

            for s in pd.unique(slot[ia]):
                ii = ia[slot[ia] == s]
                xx = x[ii]
                good = np.isfinite(xx)
                if good.sum() >= min_count:
                    med = float(np.nanmedian(xx))
                    scale = float(1.4826 * np.nanmedian(np.abs(xx - med)))
                    if not np.isfinite(scale) or scale <= 1e-12:
                        scale = float(np.nanstd(xx))
                else:
                    med, scale = med_a, scale_a
                if not np.isfinite(scale) or scale <= 1e-12:
                    med, scale = med_a, scale_a
                z[ii[good]] = (xx[good] - med) / scale

        zname = f"{col}_clock_z"
        frame[zname] = z
        zcols.append(zname)
    return zcols


def merge_diagnostics(
    state_panel: pd.DataFrame,
    g: pd.DataFrame,
    *,
    min_clock_n: int = 8,
) -> tuple[pd.DataFrame, list[str]]:
    if len(state_panel) != len(g):
        raise ValueError("state/source length mismatch")
    d = diagnostic_frame(g.reset_index(drop=True))
    out = pd.concat([state_panel.reset_index(drop=True), d], axis=1)
    zcols = robust_clock_z(out, list(d.columns), min_count=min_clock_n)
    return out, zcols


@dataclass
class EventThreshold:
    underlying: str
    expected_cut: float
    realized_cut: float
    surprise_cut: float
    n_valid: int


def label_volatility_events(
    panel: pd.DataFrame,
    *,
    q: float = DEFAULT_EVENT_Q,
    horizon: str = "30m",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrospective descriptive event candidates.

    shock = realized high AND positive surprise high.
    expected_high = realized high AND forecast high, excluding shock.
    forecast_high_miss = forecast high but realized not high.
    """
    if not (0.5 < q < 1.0):
        raise ValueError("q must be between 0.5 and 1")
    out = panel.copy()
    out["event_candidate"] = ""
    thresholds = []

    fcol = f"log_forecast_rate_eps_{horizon}"
    acol = f"log_actual_rate_{horizon}"
    scol = f"surprise_{horizon}"

    for asset, idx in out.groupby("underlying").groups.items():
        ii = np.asarray(list(idx), int)
        f = pd.to_numeric(out.loc[ii, fcol], errors="coerce").to_numpy(float)
        a = pd.to_numeric(out.loc[ii, acol], errors="coerce").to_numpy(float)
        s = pd.to_numeric(out.loc[ii, scol], errors="coerce").to_numpy(float)
        ok = np.isfinite(f) & np.isfinite(a) & np.isfinite(s)

        if ok.sum() < 100:
            thresholds.append(
                EventThreshold(str(asset), np.nan, np.nan, np.nan, int(ok.sum())).__dict__
            )
            continue

        fc = float(np.quantile(f[ok], q))
        ac = float(np.quantile(a[ok], q))
        sc = float(np.quantile(s[ok], q))
        thresholds.append(EventThreshold(str(asset), fc, ac, sc, int(ok.sum())).__dict__)

        shock = ok & (a >= ac) & (s >= sc)
        expected = ok & (a >= ac) & (f >= fc) & ~shock
        miss = ok & (f >= fc) & (a < ac)

        labels = np.full(len(ii), "", dtype=object)
        labels[miss] = "forecast_high_miss"
        labels[expected] = "expected_high"
        labels[shock] = "shock"
        out.loc[ii, "event_candidate"] = labels

    return out, pd.DataFrame(thresholds)


def decluster_event_origins(
    panel: pd.DataFrame,
    *,
    cluster_gap_min: int = DEFAULT_CLUSTER_GAP_MIN,
    horizon: str = "30m",
) -> pd.DataFrame:
    """Collapse overlapping event candidates into non-duplicative episodes."""
    scores = {
        "shock": f"surprise_{horizon}",
        "expected_high": f"log_actual_rate_{horizon}",
        "forecast_high_miss": f"log_forecast_rate_eps_{horizon}",
    }
    candidates = panel[panel["event_candidate"].isin(scores)].copy()
    candidates["state_time"] = pd.to_datetime(candidates["state_time"], utc=True)
    recs = []
    eid = 0

    def emit(indices, asset, session, kind):
        nonlocal eid
        if not indices:
            return
        gg = panel.loc[indices]
        sc = pd.to_numeric(gg[scores[kind]], errors="coerce")
        j = int(sc.idxmax()) if sc.notna().any() else int(indices[0])
        row = panel.loc[j]
        eid += 1
        recs.append({
            "event_id": eid,
            "panel_index": j,
            "raw_index": int(row["raw_index"]),
            "underlying": asset,
            "session": session,
            "event_type": kind,
            "state_time": row["state_time"],
            "tte_min": float(row["tte_min"]),
            f"forecast_rate_eps_{horizon}": row[f"forecast_rate_eps_{horizon}"],
            f"actual_rate_{horizon}": row[f"actual_rate_{horizon}"],
            f"surprise_{horizon}": row[f"surprise_{horizon}"],
            "forecast_term_slope_60v30": row.get("forecast_term_slope_60v30", np.nan),
            "forecast_accel_30m_15m": row.get("forecast_accel_30m_15m", np.nan),
        })

    for (asset, session, kind), gg in candidates.groupby(
        ["underlying", "session", "event_candidate"], sort=True
    ):
        gg = gg.sort_values("state_time")
        cluster = []
        prev = None
        for j, row in gg.iterrows():
            t = row["state_time"]
            if prev is None or (t - prev) <= pd.Timedelta(minutes=cluster_gap_min):
                cluster.append(int(j))
            else:
                emit(cluster, asset, session, kind)
                cluster = [int(j)]
            prev = t
        emit(cluster, asset, session, kind)

    return pd.DataFrame(recs)


def forecast_quality(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asset, g in panel.groupby("underlying"):
        for h in vt.HORIZONS:
            y = pd.to_numeric(g[f"log_actual_rate_{h}"], errors="coerce").to_numpy(float)
            p = pd.to_numeric(g[f"log_forecast_rate_eps_{h}"], errors="coerce").to_numpy(float)
            actual = pd.to_numeric(g[f"actual_rv_{h}"], errors="coerce").to_numpy(float)
            eps = pd.to_numeric(g[f"eps_{h}"], errors="coerce").to_numpy(float)
            F = pd.to_numeric(g[f"forecast_level_eps_{h}"], errors="coerce").to_numpy(float)
            surprise = pd.to_numeric(g[f"surprise_{h}"], errors="coerce").to_numpy(float)

            ok = (
                np.isfinite(y) & np.isfinite(p) & np.isfinite(actual)
                & np.isfinite(eps) & np.isfinite(F) & (F > 0)
            )
            if ok.sum() < 10:
                continue
            yy, pp = y[ok], p[ok]
            sst = float(np.sum((yy - yy.mean()) ** 2))
            r2 = 1.0 - float(np.sum((yy - pp) ** 2)) / sst if sst > 0 else np.nan
            corr = (
                float(np.corrcoef(yy, pp)[0, 1])
                if np.std(yy) > 0 and np.std(pp) > 0 else np.nan
            )
            ratio = (actual[ok] + eps[ok]) / F[ok]
            qlike = ratio - np.log(ratio) - 1.0
            rows.append({
                "underlying": asset,
                "horizon": h,
                "n_rows": int(ok.sum()),
                "n_sessions": int(g.loc[g.index[ok], "session"].nunique()),
                "log_r2": r2,
                "log_corr": corr,
                "mean_qlike": float(np.mean(qlike)),
                "median_surprise": float(np.nanmedian(surprise[ok])),
            })
    return pd.DataFrame(rows)


def event_feature_summary(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    feature_cols: Iterable[str],
    *,
    pre_bars: int = 3,
) -> pd.DataFrame:
    """Origin and trailing-window summaries of clock-standardized diagnostics."""
    if events.empty:
        return pd.DataFrame()
    work = panel.copy()
    cols = [c for c in feature_cols if c in work]
    suffix = f"pre{pre_bars * int(vt.BAR_MIN)}m"

    for c in cols:
        work[f"{c}_{suffix}"] = _trailing_mean(
            pd.to_numeric(work[c], errors="coerce").to_numpy(float),
            work["state_time"],
            work["session"],
            pre_bars,
        )

    recs = []
    for _, ev in events.iterrows():
        i = int(ev["panel_index"])
        row = work.loc[i]
        for c in cols:
            recs.append({
                "event_id": int(ev["event_id"]),
                "underlying": ev["underlying"],
                "event_type": ev["event_type"],
                "feature": c,
                "origin": row[c],
                "pre_window": row[f"{c}_{suffix}"],
            })

    long = pd.DataFrame(recs)
    out = []
    for (asset, kind, feature), gg in long.groupby(
        ["underlying", "event_type", "feature"], dropna=False
    ):
        for view in ("origin", "pre_window"):
            x = pd.to_numeric(gg[view], errors="coerce").dropna().to_numpy(float)
            if len(x) == 0:
                continue
            out.append({
                "underlying": asset,
                "event_type": kind,
                "feature": feature,
                "view": view,
                "n_events": int(len(x)),
                "median": float(np.median(x)),
                "q25": float(np.quantile(x, 0.25)),
                "q75": float(np.quantile(x, 0.75)),
            })
    return pd.DataFrame(out)


def event_path_summary(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    variables: Iterable[str],
    *,
    pre_min: int = 60,
    post_min: int = 120,
) -> pd.DataFrame:
    """Aggregate exact-5m event paths, never crossing an expiry session."""
    if events.empty:
        return pd.DataFrame()
    vars_ = [v for v in variables if v in panel]
    work = panel.copy()
    work["state_time"] = pd.to_datetime(work["state_time"], utc=True)
    lookup = {
        (str(r.underlying), str(r.session), r.state_time): int(i)
        for i, r in work[["underlying", "session", "state_time"]].iterrows()
    }

    recs = []
    for _, ev in events.iterrows():
        et = pd.Timestamp(ev["state_time"])
        for rel in range(-pre_min, post_min + 1, int(vt.BAR_MIN)):
            key = (
                str(ev["underlying"]),
                str(ev["session"]),
                et + pd.Timedelta(minutes=rel),
            )
            i = lookup.get(key)
            if i is None:
                continue
            row = work.loc[i]
            for v in vars_:
                if pd.notna(row[v]):
                    recs.append({
                        "event_id": int(ev["event_id"]),
                        "underlying": ev["underlying"],
                        "event_type": ev["event_type"],
                        "rel_min": rel,
                        "variable": v,
                        "value": float(row[v]),
                    })

    if not recs:
        return pd.DataFrame()
    long = pd.DataFrame(recs)
    return (
        long.groupby(["underlying", "event_type", "rel_min", "variable"])["value"]
        .agg(
            n_events="count",
            median="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
        )
        .reset_index()
    )


def event_counts(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=["underlying", "event_type", "n_events", "n_sessions"]
        )
    return (
        events.groupby(["underlying", "event_type"])
        .agg(n_events=("event_id", "size"), n_sessions=("session", "nunique"))
        .reset_index()
    )


def selftest() -> dict:
    """Small in-kernel tests for alignment, clock z-scoring and declustering."""
    times = pd.date_range("2026-01-01 08:00", periods=24, freq="5min", tz="UTC")
    sess = np.array(["2026-01-01"] * 24)
    x = np.arange(24, dtype=float)

    lag = _causal_lag(x, pd.Series(times), sess, 3)
    assert np.isnan(lag[:3]).all()
    assert np.allclose(lag[3:], x[:-3])

    m = _trailing_mean(x, pd.Series(times), sess, 3)
    assert np.isnan(m[:2]).all()
    expected = np.array([x[i - 2:i + 1].mean() for i in range(2, len(x))])
    assert np.allclose(m[2:], expected)

    tf = pd.DataFrame({
        "underlying": ["X"] * 8,
        "state_time": pd.to_datetime([
            "2026-01-01 08:00Z", "2026-01-01 08:05Z",
            "2026-01-02 08:00Z", "2026-01-02 08:05Z",
            "2026-01-03 08:00Z", "2026-01-03 08:05Z",
            "2026-01-04 08:00Z", "2026-01-04 08:05Z",
        ]),
        "v": [10, 20, 11, 21, 9, 19, 10, 20],
    })
    zc = robust_clock_z(tf, ["v"], min_count=3)[0]
    assert np.isfinite(tf[zc]).all()
    assert abs(float(tf.loc[[0, 2, 4, 6], zc].median())) < 1e-12

    p = pd.DataFrame({
        "raw_index": np.arange(4),
        "underlying": ["X"] * 4,
        "session": ["2026-01-01"] * 4,
        "state_time": pd.date_range("2026-01-01 10:00", periods=4, freq="5min", tz="UTC"),
        "tte_min": [300, 295, 290, 285],
        "event_candidate": ["shock", "shock", "shock", ""],
        "surprise_30m": [1.0, 2.0, 1.5, 0.0],
        "log_actual_rate_30m": [0.0] * 4,
        "log_forecast_rate_eps_30m": [0.0] * 4,
        "forecast_rate_eps_30m": [1.0] * 4,
        "actual_rate_30m": [1.0] * 4,
        "forecast_term_slope_60v30": [0.0] * 4,
        "forecast_accel_30m_15m": [0.0] * 4,
    })
    ev = decluster_event_origins(p, cluster_gap_min=30)
    assert len(ev) == 1
    assert int(ev.iloc[0]["panel_index"]) == 1

    return {"lag": True, "trailing": True, "clock_z": True, "decluster": True}
