"""
Volatility Dynamics 05b: prospective test of the frozen -95 minute L2 motif.

Module 05 characterized a five-candle motif (-105,-100,-95,-90,-85 min before
clean q99 onsets) after the -95m region was spotted visually. The only part
that differed from the neighbouring candles was the sparse L2 family:

    L2 notch   = P(-95) - 0.5 * (pre + post),
                 pre = mean(P(-105), P(-100)), post = mean(P(-90), P(-85))
    L2 rebound = post - P(-95)

(P = event percentile vs matched controls, averaged over the four L2
features; Module 05 definitions, unchanged.)

Discovery estimate (Module 05, 13 unique clean event timestamps, all in the
same data that suggested the lag): notch -0.097, rebound +0.093.

Because the lag was chosen after looking, the discovery sample cannot test it.
05b freezes everything and evaluates only q99 onsets that occur AFTER the
discovery data ended.

Frozen specification (fixed before any prospective data are inspected)
---------------------------------------------------------------------
- DISCOVERY_END = 2026-10-08 08:00 UTC (Module 05's last bar was 07:55).
- q99 thresholds: per-asset 0.99 quantile of r^2 on rows < DISCOVERY_END
  (identical to Module 05's full-sample thresholds if history is unchanged).
- Prospective events: q99 onsets (frozen thresholds) at or after
  DISCOVERY_END with no q99 bar in the preceding 120 minutes.
- Primary procedure: Module 05's own code path (Module 04 native-vol-matched
  controls on clean origins, Module 03 percentiles, Module 05 components).
  The notch is a contrast between ADJACENT lags, so the volatility-level
  confound that broke Module 04 largely cancels in it; replicating the exact
  procedure is the faithful test of what was observed.
- Primary statistic: mean over unique prospective timestamps of l2_notch_95.
- PASS iff  n_unique_timestamps >= 20  AND  mean notch < 0  AND  one-sided
  sign-flip randomization p < 0.05.
  n < 20 -> INSUFFICIENT_DATA (estimate reported, no conclusion).
  Otherwise -> FAIL.
- Secondary (reported, no decision): rebound > 0 with the same rule; the
  same notch under 04b M2 bias-corrected matching.
- Context only (NOT a test): notch at every center from -115 to -10 min in
  the prospective sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_engineering.modeling import volatility_q99_precursors as qp
from data_engineering.modeling import volatility_q99_conditional as qc
from data_engineering.modeling import volatility_q99_motif95 as qm
from data_engineering.modeling import volatility_vd_b_core as vb
from data_engineering.modeling import volatility_q99_matching_b as mb


VERSION = "0.1.0"
DISCOVERY_END = pd.Timestamp("2026-10-08 08:00:00", tz="UTC")
MIN_UNIQUE_TIMESTAMPS = 20
ALPHA = 0.05
DISCOVERY_NOTCH = -0.097
DISCOVERY_REBOUND = 0.093
DISCOVERY_N_UNIQUE = 13
SCAN_CENTERS = tuple(range(-23, -1))      # -115 .. -10 min (needs center +/- 2)


def frozen_events(grid: vb.Grid, discovery_end: pd.Timestamp = DISCOVERY_END) -> tuple[pd.DataFrame, dict]:
    """All q99 onsets under thresholds frozen on pre-discovery_end rows,
    in the Module 03 event-table format (event_panel_index = grid.frame row)."""
    cutoff_ns = pd.DatetimeIndex([discovery_end]).as_unit("ns").asi8[0]
    thr = vb.q99_thresholds(grid, row_mask=grid.time_ns < cutoff_ns)
    tail = vb.tail_flags(grid, thr)
    ons = np.flatnonzero(vb.onset_flags(grid, tail))
    prev = grid.offset_rows(ons, -1)
    ev = pd.DataFrame({
        "event_id": np.arange(1, len(ons) + 1),
        "underlying": [grid.assets[c] for c in grid.asset_code[ons]],
        "event_time": pd.to_datetime(grid.time_ns[ons], utc=True),
        "event_panel_index": ons,
        "clock_slot": grid.slot[ons],
        "q99_threshold": [thr[grid.assets[c]] for c in grid.asset_code[ons]],
        "event_variance_5m": grid.v[ons],
        "previous_variance_5m": np.where(prev >= 0, grid.v[np.maximum(prev, 0)], np.nan),
    })
    return ev, thr


def _notch_from_P(P_by_bar: dict[int, np.ndarray], center: int) -> np.ndarray:
    """Per-row notch: center - 0.5*(mean(c-2,c-1) + mean(c+1,c+2))."""
    pre = 0.5 * (P_by_bar[center - 2] + P_by_bar[center - 1])
    post = 0.5 * (P_by_bar[center + 1] + P_by_bar[center + 2])
    return P_by_bar[center] - 0.5 * (pre + post)


def module05_procedure(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    prospective_mask: np.ndarray,
    l2_features: list[str],
    *,
    max_controls: int = 8,
    min_controls: int = 4,
    min_env_dims: int = 4,
) -> dict:
    """Exact Module 04/05 code path on the prospective clean events."""
    origins = qp.build_origin_index(frame, events)
    clean_events, clean_origins, audit = qm.clean_event_and_origin_sets(frame, events, origins)
    pros = clean_events[clean_events["event_id"].isin(events.loc[prospective_mask, "event_id"])].copy()
    out = {"audit": audit, "prospective_events": pros}
    if pros.empty:
        out.update({"components": pd.DataFrame(), "scores": pd.DataFrame(), "controls": pd.DataFrame()})
        return out
    controls, support = qc.choose_volatility_matched_controls(
        frame, clean_origins, pros, max_controls=max_controls, min_env_dims=min_env_dims,
    )
    scores = qp.percentile_scores(frame, l2_features, pros, controls, range(-24, 1), min_controls=min_controls)
    contract = qm.motif_contract(l2_features)
    comps = qm.motif_component_scores(scores, contract) if len(scores) else pd.DataFrame()
    out.update({"components": comps, "scores": scores, "controls": controls, "support": support})
    return out


def timestamp_series(components: pd.DataFrame, column: str) -> pd.Series:
    if components.empty or column not in components:
        return pd.Series(dtype=float)
    ts = components.groupby("event_time")[column].mean()
    return ts[np.isfinite(ts)]


def decide(values: pd.Series, *, expected_sign: int, rng: np.random.Generator) -> dict:
    x = values.to_numpy(float)
    n = int(len(x))
    mean = float(np.mean(x)) if n else np.nan
    if n >= 2:
        p = vb.sign_flip_pvalue(x, rng, n=20000, alternative="less" if expected_sign < 0 else "greater")
    else:
        p = np.nan
    if n < MIN_UNIQUE_TIMESTAMPS:
        status = "INSUFFICIENT_DATA"
    elif np.isfinite(mean) and np.sign(mean) == expected_sign and np.isfinite(p) and p < ALPHA:
        status = "PASS"
    else:
        status = "FAIL"
    return {"n_unique_timestamps": n, "mean": mean, "one_sided_p": p, "status": status}


def center_scan(scores: pd.DataFrame) -> pd.DataFrame:
    """Context only: L2 notch at every center (same components logic)."""
    if scores.empty:
        return pd.DataFrame()
    l2 = scores[scores["feature"].isin(qm.L2_FEATURES)]
    piv = l2.groupby(["event_id", "event_time", "rel_bar"])["percentile_vs_controls"].mean().unstack("rel_bar")
    rows = []
    for c in SCAN_CENTERS:
        need = [c - 2, c - 1, c, c + 1, c + 2]
        if not all(b in piv.columns for b in need):
            continue
        sub = piv[need].dropna()
        if sub.empty:
            continue
        notch = _notch_from_P({b: sub[b].to_numpy() for b in need}, c)
        ts = pd.Series(notch, index=sub.index.get_level_values("event_time")).groupby(level=0).mean()
        rows.append({
            "center_rel_bar": c, "center_rel_min": 5 * c,
            "n_unique_timestamps": int(len(ts)), "mean_l2_notch": float(ts.mean()),
            "is_frozen_center": c == qm.MOTIF_CENTER_BAR,
        })
    return pd.DataFrame(rows)


def secondary_04b_notch(
    grid: vb.Grid,
    events: pd.DataFrame,
    prospective_ids: set,
    clean_eligible: np.ndarray,
    l2_features: list[str],
) -> pd.Series:
    """Same notch with 04b M2 bias-corrected matched controls."""
    rows = events.loc[events["event_id"].isin(prospective_ids), "event_panel_index"].to_numpy(np.int64)
    if not len(rows):
        return pd.Series(dtype=float)
    env, cols, _ = mb.environment(grid)
    if not all(c in cols for c in mb.M2_REQUIRED):
        return pd.Series(dtype=float)
    X = grid.feature_matrix(l2_features)
    Z, _ = vb.natural_scale(X)
    bars = list(qm.MOTIF_BARS)
    req = [cols.index(c) for c in mb.M2_REQUIRED]
    betas = vb.fit_adjustment(grid, Z, env[:, req], clean_eligible, bars)
    cc, _ = vb.matched_controls(grid, rows, clean_eligible, env, required_dims=req)
    P = vb.percentile_tensor(grid, Z, rows, cc, bars, env=env[:, req], betas=betas)
    with np.errstate(invalid="ignore"):
        Pl2 = np.nanmean(P, axis=2)                       # mean over L2 features
    by = {b: Pl2[:, i] for i, b in enumerate(bars)}
    notch = _notch_from_P(by, qm.MOTIF_CENTER_BAR)
    s = pd.Series(notch, index=pd.to_datetime(grid.time_ns[rows], utc=True))
    s = s[np.isfinite(s)]
    return s.groupby(level=0).mean()


def selftest() -> dict:
    # synthetic: 20 days, motif-free L2 proxies; cutoff after day 14
    f = vb.synthetic_frame(n_days=20, seed=31)
    rng = np.random.default_rng(5)
    for c in qm.L2_FEATURES:
        f[c] = np.abs(rng.normal(size=len(f)))
    g = vb.build_grid(f)
    cutoff = pd.Timestamp("2026-01-15", tz="UTC")
    ev, thr = frozen_events(g, cutoff)
    pros_mask = (ev["event_time"] >= cutoff).to_numpy()
    res = module05_procedure(g.frame, ev, pros_mask, list(qm.L2_FEATURES))
    assert (res["prospective_events"]["event_time"] >= cutoff).all()
    ts = timestamp_series(res["components"], "l2_notch_95")
    d = decide(ts, expected_sign=-1, rng=np.random.default_rng(0))
    assert d["status"] in {"INSUFFICIENT_DATA", "FAIL", "PASS"}
    # no motif injected -> must not PASS
    assert d["status"] != "PASS"
    # frozen thresholds must not use prospective rows
    t_all = vb.q99_thresholds(g)
    assert any(abs(thr[a] - t_all[a]) > 0 for a in thr)
    return {"frozen_thresholds": True, "prospective_only": True, "module05_code_path": True, "no_false_pass": True}
