"""
Volatility Dynamics "b" run: shared, vectorized core.

Used by notebooks 03b, 04b and 05b (Volatility Dynamics branch only). It
re-implements the Module 03 event/control/percentile machinery on a dense
per-asset five-minute grid so that permutation nulls (hundreds of reruns of
the full event study) are computationally feasible, and adds:

- causal multi-horizon volatility-environment covariates;
- volatility-matched controls drawn from a +/- clock window across all dates
  (instead of the ~15 exact-slot candidates Module 04 had per event);
- a day-relabel placebo generator that preserves clock slot, asset identity
  and cross-asset synchronicity of the real event set.

Branch independence: imports only NumPy/Pandas. Nothing is imported from any
other research branch. Within the branch it re-implements (and, in tests,
reproduces) Module 03 definitions.

Conventions
-----------
- ``state_time`` is the bar END; ``return_5m`` at t covers (t-5m, t].
- A feature at relative bar -k of an origin t is the row at t - 5k minutes.
- "Causal" covariates for an origin at t use bars t-1 ... t-k only (strictly
  before the origin bar), unless stated otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


VERSION = "0.1.0"
BAR_MINUTES = 5
BAR_NS = BAR_MINUTES * 60 * 1_000_000_000
BARS_PER_DAY = 288
Q99 = 0.99


# --------------------------------------------------------------------------
# Dense grid
# --------------------------------------------------------------------------

@dataclass
class Grid:
    """Rows of ``frame`` plus a dense (asset, bar) -> row lookup."""

    frame: pd.DataFrame          # sorted by underlying, state_time; RangeIndex
    assets: list[str]
    asset_code: np.ndarray       # int per row
    bar: np.ndarray              # int64 bar number (epoch ns // BAR_NS) per row
    slot: np.ndarray             # 0..287 UTC five-minute slot per row
    day: np.ndarray              # int64 UTC day number per row
    time_ns: np.ndarray          # int64 epoch ns per row
    r: np.ndarray                # return_5m
    v: np.ndarray                # return_5m ** 2
    _base: np.ndarray            # dense offset per asset code
    _bar0: np.ndarray            # first bar per asset code
    _span: np.ndarray            # number of dense bars per asset code
    _dense: np.ndarray           # dense position -> row (-1 if absent)

    @property
    def n(self) -> int:
        return int(len(self.frame))

    def dense_pos(self, rows: np.ndarray, offset: int = 0) -> np.ndarray:
        """Dense position of (row's asset, row's bar + offset); -1 if outside."""
        rows = np.asarray(rows, dtype=np.int64)
        out = np.full(rows.shape, -1, dtype=np.int64)
        ok = rows >= 0
        if not ok.any():
            return out
        a = self.asset_code[rows[ok]]
        rel = self.bar[rows[ok]] + int(offset) - self._bar0[a]
        inside = (rel >= 0) & (rel < self._span[a])
        pos = np.full(rel.shape, -1, dtype=np.int64)
        pos[inside] = self._base[a[inside]] + rel[inside]
        out[ok] = pos
        return out

    def offset_rows(self, rows: np.ndarray, offset: int) -> np.ndarray:
        """Row index at exact relative bar ``offset`` (same asset); -1 if absent."""
        pos = self.dense_pos(rows, offset)
        out = np.full(pos.shape, -1, dtype=np.int64)
        ok = pos >= 0
        out[ok] = self._dense[pos[ok]]
        return out

    def row_at(self, asset_code: np.ndarray, bar: np.ndarray) -> np.ndarray:
        asset_code = np.asarray(asset_code, dtype=np.int64)
        bar = np.asarray(bar, dtype=np.int64)
        rel = bar - self._bar0[asset_code]
        inside = (rel >= 0) & (rel < self._span[asset_code])
        out = np.full(bar.shape, -1, dtype=np.int64)
        out[inside] = self._dense[self._base[asset_code[inside]] + rel[inside]]
        return out

    def dense_values(self, values: np.ndarray) -> np.ndarray:
        """Scatter a per-row array onto the dense grid (NaN where absent)."""
        values = np.asarray(values, dtype=float)
        d = np.full(len(self._dense), np.nan)
        ok = self._dense >= 0
        d[ok] = values[self._dense[ok]]
        return d

    def feature_matrix(self, features: Sequence[str]) -> np.ndarray:
        cols = []
        for c in features:
            s = self.frame[c]
            if pd.api.types.is_bool_dtype(s.dtype):
                x = s.astype("Float64").to_numpy(dtype=float, na_value=np.nan)
            else:
                x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            cols.append(x)
        if not cols:
            return np.empty((self.n, 0))
        return np.column_stack(cols)


def _ns(times) -> np.ndarray:
    t = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    return t.as_unit("ns").asi8


def build_grid(frame: pd.DataFrame) -> Grid:
    required = {"state_time", "underlying", "return_5m"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    f = frame.copy()
    f["state_time"] = pd.to_datetime(f["state_time"], utc=True)
    f["underlying"] = f["underlying"].astype(str)
    f = f.sort_values(["underlying", "state_time"]).reset_index(drop=True)
    if f.duplicated(["underlying", "state_time"]).any():
        raise ValueError("Duplicate (underlying, state_time) rows.")

    t_ns = _ns(f["state_time"])
    if np.any(t_ns % BAR_NS != 0):
        raise ValueError("state_time values are not on the five-minute grid.")
    bar = t_ns // BAR_NS
    assets = sorted(f["underlying"].unique())
    code_map = {a: i for i, a in enumerate(assets)}
    code = f["underlying"].map(code_map).to_numpy(dtype=np.int64)

    n_a = len(assets)
    bar0 = np.zeros(n_a, dtype=np.int64)
    span = np.zeros(n_a, dtype=np.int64)
    for i in range(n_a):
        b = bar[code == i]
        bar0[i] = int(b.min())
        span[i] = int(b.max() - b.min() + 1)
    base = np.concatenate([[0], np.cumsum(span)[:-1]]).astype(np.int64)
    dense = np.full(int(span.sum()), -1, dtype=np.int64)
    dense[base[code] + (bar - bar0[code])] = np.arange(len(f), dtype=np.int64)

    r = pd.to_numeric(f["return_5m"], errors="coerce").to_numpy(dtype=float)
    return Grid(
        frame=f,
        assets=assets,
        asset_code=code,
        bar=bar,
        slot=(bar % BARS_PER_DAY).astype(np.int64),
        day=(bar // BARS_PER_DAY).astype(np.int64),
        time_ns=t_ns,
        r=r,
        v=r ** 2,
        _base=base,
        _bar0=bar0,
        _span=span,
        _dense=dense,
    )


# --------------------------------------------------------------------------
# q99 events
# --------------------------------------------------------------------------

def q99_thresholds(grid: Grid, row_mask: np.ndarray | None = None, q: float = Q99) -> dict[str, float]:
    """Per-asset empirical quantile of r^2 over rows in ``row_mask``."""
    mask = np.ones(grid.n, bool) if row_mask is None else np.asarray(row_mask, bool)
    out = {}
    for i, a in enumerate(grid.assets):
        x = grid.v[(grid.asset_code == i) & mask]
        x = x[np.isfinite(x)]
        if len(x) >= 100:
            out[a] = float(np.quantile(x, q))
    return out


def tail_flags(grid: Grid, thresholds: dict[str, float]) -> np.ndarray:
    thr = np.array([thresholds.get(a, np.inf) for a in grid.assets])[grid.asset_code]
    return np.isfinite(grid.v) & (grid.v >= thr)


def previous_flag(grid: Grid, flag: np.ndarray) -> np.ndarray:
    """flag at the exact previous five-minute bar (False if that bar is absent)."""
    prev = grid.offset_rows(np.arange(grid.n), -1)
    out = np.zeros(grid.n, bool)
    ok = prev >= 0
    out[ok] = np.asarray(flag, bool)[prev[ok]]
    return out


def onset_flags(grid: Grid, tail: np.ndarray) -> np.ndarray:
    """q99 run onsets: tail bar whose exact previous bar is absent or not tail."""
    return np.asarray(tail, bool) & ~previous_flag(grid, tail)


def eligible_origins(grid: Grid, tail: np.ndarray) -> np.ndarray:
    """Module 03 control eligibility: finite, not q99, previous exact bar not q99."""
    return np.isfinite(grid.v) & ~np.asarray(tail, bool) & ~previous_flag(grid, tail)


def window_count(grid: Grid, flag: np.ndarray, start: int, end: int) -> np.ndarray:
    """Number of flagged rows at relative bars start..end (inclusive), same asset."""
    if start > end:
        raise ValueError("start must be <= end")
    d = np.nan_to_num(grid.dense_values(np.asarray(flag, float)), nan=0.0)
    cs = np.concatenate([[0.0], np.cumsum(d)])
    out = np.zeros(grid.n)
    rows = np.arange(grid.n)
    for a in range(len(grid.assets)):
        rr = rows[grid.asset_code == a]
        rel = grid.bar[rr] - grid._bar0[a]
        lo = np.clip(rel + start, 0, grid._span[a])
        hi = np.clip(rel + end + 1, 0, grid._span[a])
        b = grid._base[a]
        out[rr] = np.where(hi > lo, cs[b + hi] - cs[b + lo], 0.0)
    return out


def causal_window_mean(
    grid: Grid,
    values: np.ndarray,
    start: int,
    end: int,
    min_count: int,
) -> np.ndarray:
    """Mean of finite ``values`` at relative bars start..end; NaN if too few."""
    vals = grid.dense_values(values)
    fin = np.isfinite(vals)
    cs = np.concatenate([[0.0], np.cumsum(np.where(fin, vals, 0.0))])
    cc = np.concatenate([[0.0], np.cumsum(fin.astype(float))])
    out = np.full(grid.n, np.nan)
    rows = np.arange(grid.n)
    for a in range(len(grid.assets)):
        rr = rows[grid.asset_code == a]
        rel = grid.bar[rr] - grid._bar0[a]
        lo = np.clip(rel + start, 0, grid._span[a])
        hi = np.clip(rel + end + 1, 0, grid._span[a])
        b = grid._base[a]
        s = np.where(hi > lo, cs[b + hi] - cs[b + lo], 0.0)
        c = np.where(hi > lo, cc[b + hi] - cc[b + lo], 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = s / c
        out[rr] = np.where(c >= int(min_count), m, np.nan)
    return out


def event_table(grid: Grid, rows: np.ndarray, thresholds: dict[str, float] | None = None) -> pd.DataFrame:
    rows = np.asarray(rows, dtype=np.int64)
    df = pd.DataFrame({
        "row": rows,
        "underlying": [grid.assets[c] for c in grid.asset_code[rows]],
        "event_time": pd.to_datetime(grid.time_ns[rows], utc=True),
        "clock_slot": grid.slot[rows],
        "event_variance_5m": grid.v[rows],
    })
    if thresholds is not None:
        df["q99_threshold"] = df["underlying"].map(thresholds)
    return df


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------

def clock_controls(
    grid: Grid,
    event_rows: np.ndarray,
    eligible: np.ndarray,
    *,
    max_controls: int = 8,
) -> np.ndarray:
    """Module 03 controls: same asset, exact slot, eligible, nearest dates.

    Ties are broken by earlier time (as in Module 03). The event row itself
    is never its own control (relevant for placebo events, which are eligible).
    Returns (n_events, max_controls) row indices, -1 padded.
    """
    event_rows = np.asarray(event_rows, dtype=np.int64)
    out = np.full((len(event_rows), int(max_controls)), -1, dtype=np.int64)
    elig_rows = np.flatnonzero(np.asarray(eligible, bool))
    key = grid.asset_code[elig_rows] * BARS_PER_DAY + grid.slot[elig_rows]
    order = np.lexsort((grid.time_ns[elig_rows], key))
    elig_sorted = elig_rows[order]
    key_sorted = key[order]
    for k, e in enumerate(event_rows):
        kk = grid.asset_code[e] * BARS_PER_DAY + grid.slot[e]
        lo, hi = np.searchsorted(key_sorted, [kk, kk + 1])
        cand = elig_sorted[lo:hi]
        cand = cand[cand != e]
        if not len(cand):
            continue
        dist = np.abs(grid.time_ns[cand] - grid.time_ns[e])
        o = np.lexsort((grid.time_ns[cand], dist))[: int(max_controls)]
        out[k, : len(o)] = cand[o]
    return out


def volatility_environment(
    grid: Grid,
    *,
    range_values: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[int, int, int]]]:
    """Causal volatility-environment covariates for every origin row.

    Return-based (matching set M1): log mean r^2 over bars
        -3..-1 (15m), -12..-1 (1h), -48..-1 (4h), -288..-1 (24h)
    with at least 2/3, 9/12, 36/48 and 216/288 finite bars.

    If ``range_values`` (intrabar range) is supplied, also adds (set M2):
        log mean range over -3..-1 and -12..-1.

    Everything is strictly before the origin bar.
    """
    specs = {
        "env_lrv_15m": (-3, -1, 2),
        "env_lrv_1h": (-12, -1, 9),
        "env_lrv_4h": (-48, -1, 36),
        "env_lrv_24h": (-288, -1, 216),
    }
    pos = grid.v[np.isfinite(grid.v) & (grid.v > 0)]
    eps = float(np.median(pos) * 1e-4) if len(pos) else 1e-12
    cols = {}
    for name, (s, e, m) in specs.items():
        cols[name] = np.log(causal_window_mean(grid, grid.v, s, e, m) + eps)
    if range_values is not None:
        rv = np.asarray(range_values, float)
        rpos = rv[np.isfinite(rv) & (rv > 0)]
        reps = float(np.median(rpos) * 1e-3) if len(rpos) else 1e-6
        rv = np.where(np.isfinite(rv) & (rv >= 0), rv, np.nan)
        for name, (s, e, m) in {
            "env_lrange_15m": (-3, -1, 2),
            "env_lrange_1h": (-12, -1, 9),
        }.items():
            cols[name] = np.log(causal_window_mean(grid, rv, s, e, m) + reps)
            specs[name] = (s, e, m)
    return pd.DataFrame(cols), specs


def robust_asset_scales(grid: Grid, env: np.ndarray) -> np.ndarray:
    """(n_assets, d) IQR/1.349 scales of the environment columns."""
    env = np.asarray(env, float)
    sc = np.ones((len(grid.assets), env.shape[1]))
    for a in range(len(grid.assets)):
        x = env[grid.asset_code == a]
        for j in range(env.shape[1]):
            col = x[:, j]
            col = col[np.isfinite(col)]
            if len(col) < 10:
                continue
            q25, q75 = np.quantile(col, [0.25, 0.75])
            s = (q75 - q25) / 1.349
            if not np.isfinite(s) or s <= 1e-12:
                s = float(np.std(col))
            sc[a, j] = s if np.isfinite(s) and s > 1e-12 else 1.0
    return sc


def matched_controls(
    grid: Grid,
    event_rows: np.ndarray,
    eligible: np.ndarray,
    env: np.ndarray,
    *,
    required_dims: Sequence[int],
    max_controls: int = 8,
    clock_window_slots: int = 6,
    min_separation_bars: int = 48,
    scales: np.ndarray | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Volatility-matched controls.

    Candidates: same asset; eligible (Module 03 rule); circular UTC clock
    distance <= ``clock_window_slots``; at least ``min_separation_bars`` away
    from the event (so the two -120m windows do not overlap); finite on every
    ``required_dims`` environment column and on every column the event has.

    Distance: RMS of robust-scaled differences over the dimensions finite for
    the event. Ties broken by clock distance, then date distance.
    """
    event_rows = np.asarray(event_rows, dtype=np.int64)
    env = np.asarray(env, float)
    if scales is None:
        scales = robust_asset_scales(grid, env)
    required_dims = list(required_dims)
    out = np.full((len(event_rows), int(max_controls)), -1, dtype=np.int64)
    elig = np.asarray(eligible, bool)
    req_ok = np.isfinite(env[:, required_dims]).all(axis=1) if required_dims else np.ones(grid.n, bool)
    pool_mask = elig & req_ok
    pool_by_asset = {a: np.flatnonzero(pool_mask & (grid.asset_code == a)) for a in range(len(grid.assets))}
    support = []

    for k, e in enumerate(event_rows):
        a = int(grid.asset_code[e])
        ev = env[e]
        dims = np.flatnonzero(np.isfinite(ev))
        if not set(required_dims).issubset(set(dims.tolist())):
            support.append((k, 0, 0, np.nan, np.nan))
            continue
        cand = pool_by_asset[a]
        cd = np.abs(grid.slot[cand] - grid.slot[e])
        cd = np.minimum(cd, BARS_PER_DAY - cd)
        sep = np.abs(grid.bar[cand] - grid.bar[e])
        keep = (cd <= int(clock_window_slots)) & (sep >= int(min_separation_bars))
        cand = cand[keep]
        cd = cd[keep]
        if len(cand):
            C = env[np.ix_(cand, dims)]
            fin = np.isfinite(C).all(axis=1)
            cand, cd, C = cand[fin], cd[fin], C[fin]
        n_pool = int(len(cand))
        if not n_pool:
            support.append((k, 0, 0, np.nan, np.nan))
            continue
        z = (C - ev[dims][None, :]) / scales[a, dims][None, :]
        dist = np.sqrt(np.mean(z ** 2, axis=1))
        datedist = np.abs(grid.time_ns[cand] - grid.time_ns[e])
        o = np.lexsort((datedist, cd, dist))[: int(max_controls)]
        out[k, : len(o)] = cand[o]
        support.append((k, n_pool, len(o), float(np.median(dist[o])), float(np.max(dist[o]))))

    sup = pd.DataFrame(support, columns=[
        "event_index", "n_candidate_pool", "n_controls_selected",
        "median_selected_distance", "max_selected_distance",
    ])
    return out, sup


# --------------------------------------------------------------------------
# Bias-corrected matching (Abadie-Imbens style regression adjustment)
# --------------------------------------------------------------------------

def _norm_ppf(p: np.ndarray) -> np.ndarray:
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    p = np.asarray(p, float)
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    out = np.full(p.shape, np.nan)
    lo = (p > 0) & (p < 0.02425)
    hi = (p > 1 - 0.02425) & (p < 1)
    mid = (p >= 0.02425) & (p <= 1 - 0.02425)
    q = np.sqrt(-2 * np.log(p[lo]))
    out[lo] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
              ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = np.sqrt(-2 * np.log(1 - p[hi]))
    out[hi] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p[mid] - 0.5
    r = q * q
    out[mid] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    return out


def normal_scores(grid: Grid, X: np.ndarray) -> np.ndarray:
    """Per-asset rank -> normal-score transform of each column (monotone).

    Percentile ranks are invariant to it; it makes linear adjustment sensible
    for skewed fields (volume, counts, spreads) and for booleans/ties.
    """
    X = np.asarray(X, float)
    Z = np.full(X.shape, np.nan)
    for a in range(len(grid.assets)):
        m = grid.asset_code == a
        sub = pd.DataFrame(X[m])
        n_fin = sub.notna().sum(axis=0).to_numpy(float)
        rk = sub.rank(method="average").to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            u = (rk - 0.5) / n_fin[None, :]
        Z[m] = _norm_ppf(u)
    return Z


def natural_scale(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Monotone per-column transform used for regression adjustment.

    - boolean / binary columns: unchanged;
    - non-negative columns: log(x + c), c = 1% of the median positive value
      (log for volumes, counts, spreads, ranges, IV levels, realized vol);
    - signed columns: unchanged.
    Monotone, so percentile ranks are identical to the raw scale. Unlike
    rank-normal scores it does not saturate in the tail, so linear
    extrapolation to extreme q99 environments does not overshoot.
    """
    X = np.asarray(X, float)
    out = X.copy()
    kinds = []
    for j in range(X.shape[1]):
        x = X[:, j]
        fin = x[np.isfinite(x)]
        if len(fin) == 0 or len(np.unique(fin[: 5000])) <= 2:
            kinds.append("identity_binary")
            continue
        if np.min(fin) >= 0:
            pos = fin[fin > 0]
            c = 0.01 * float(np.median(pos)) if len(pos) else 1.0
            out[:, j] = np.log(x + c)
            kinds.append("log")
        else:
            kinds.append("identity_signed")
    return out, kinds


def fit_adjustment(
    grid: Grid,
    Z: np.ndarray,
    env: np.ndarray,
    origin_mask: np.ndarray,
    rel_bars: Sequence[int],
    *,
    min_rows: int = 200,
    ridge: float = 1e-3,
) -> np.ndarray:
    """OLS slopes of Z(origin + b) on env(origin), per rel bar, asset, feature.

    Fitted over all origins in ``origin_mask`` (non-event rows) with finite
    env. Returns betas of shape (n_rel, n_assets, d, n_feat); NaN where the
    fit had fewer than ``min_rows`` rows (no adjustment applied there).
    """
    env = np.asarray(env, float)
    d = env.shape[1]
    nf = Z.shape[1]
    B = np.full((len(rel_bars), len(grid.assets), d, nf), np.nan)
    base = np.flatnonzero(np.asarray(origin_mask, bool) & np.isfinite(env).all(axis=1))
    for j, b in enumerate(rel_bars):
        tgt = grid.offset_rows(base, int(b))
        ok = tgt >= 0
        o, t = base[ok], tgt[ok]
        for a in range(len(grid.assets)):
            ma = grid.asset_code[o] == a
            if ma.sum() < min_rows:
                continue
            E = env[o[ma]]
            E = E - E.mean(axis=0, keepdims=True)
            Y = Z[t[ma]]
            for fj in range(nf):
                y = Y[:, fj]
                fin = np.isfinite(y)
                if fin.sum() < min_rows:
                    continue
                Ef = E[fin] - E[fin].mean(axis=0, keepdims=True)
                yf = y[fin] - y[fin].mean()
                G = Ef.T @ Ef + ridge * len(yf) * np.eye(d)
                B[j, a, :, fj] = np.linalg.solve(G, Ef.T @ yf)
    return B


# --------------------------------------------------------------------------
# Percentile scores
# --------------------------------------------------------------------------

def percentile_tensor(
    grid: Grid,
    X: np.ndarray,
    event_rows: np.ndarray,
    controls: np.ndarray,
    rel_bars: Sequence[int],
    *,
    min_controls: int = 4,
    env: np.ndarray | None = None,
    betas: np.ndarray | None = None,
) -> np.ndarray:
    """Event percentile among its controls, for every rel bar and feature.

    Identical rule to Module 03: (less + 0.5 * equal) / n_finite_controls,
    valid only if the event value is finite and >= min_controls finite
    control values exist. Returns (n_events, n_rel, n_features), NaN invalid.

    If ``env`` and ``betas`` (from ``fit_adjustment`` with the same rel_bars)
    are given, every control value is bias-corrected before ranking:
        C_adj = C + beta' (env[event origin] - env[control origin])
    so the control is moved to the event's volatility environment.
    """
    event_rows = np.asarray(event_rows, dtype=np.int64)
    controls = np.asarray(controls, dtype=np.int64)
    n_ev, n_c = controls.shape
    f = X.shape[1]
    P = np.full((n_ev, len(rel_bars), f), np.nan)
    has_enough_origins = (controls >= 0).sum(axis=1) >= int(min_controls)
    flat_c = controls.reshape(-1)
    adjust = env is not None and betas is not None
    if adjust:
        env = np.asarray(env, float)
        dE = env[event_rows][:, None, :] - np.where(
            (controls >= 0)[:, :, None], env[np.maximum(controls, 0)], np.nan
        )                                                     # (n_ev, n_c, d)
        a_ev = grid.asset_code[event_rows]
    for j, b in enumerate(rel_bars):
        er = grid.offset_rows(event_rows, int(b))
        cr = grid.offset_rows(flat_c, int(b)).reshape(n_ev, n_c)
        E = np.full((n_ev, f), np.nan)
        okE = er >= 0
        E[okE] = X[er[okE]]
        C = np.full((n_ev, n_c, f), np.nan)
        okC = cr >= 0
        C[okC] = X[cr[okC]]
        if adjust:
            Bj = np.nan_to_num(betas[j][a_ev])                # (n_ev, d, f)
            shift = np.einsum("ncd,ndf->ncf", np.nan_to_num(dE), Bj)
            shift = np.where(np.isfinite(dE).all(axis=2)[:, :, None], shift, np.nan)
            C = C + shift
        finC = np.isfinite(C)
        n_fin = finC.sum(axis=1)
        with np.errstate(invalid="ignore"):
            less = ((C < E[:, None, :]) & finC).sum(axis=1)
            equal = ((C == E[:, None, :]) & finC).sum(axis=1)
        pct = (less + 0.5 * equal) / np.maximum(n_fin, 1)
        valid = np.isfinite(E) & (n_fin >= int(min_controls)) & has_enough_origins[:, None]
        P[:, j, :] = np.where(valid, pct, np.nan)
    return P


def collapse_by_time(time_ns: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average asset-events sharing one timestamp (NaN-aware)."""
    time_ns = np.asarray(time_ns)
    uniq, inv = np.unique(time_ns, return_inverse=True)
    shape = (len(uniq),) + P.shape[1:]
    s = np.zeros(shape)
    c = np.zeros(shape)
    fin = np.isfinite(P)
    np.add.at(s, inv, np.where(fin, P, 0.0))
    np.add.at(c, inv, fin.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        T = s / c
    T[c == 0] = np.nan
    return uniq, T


def module03_rankings(
    P: np.ndarray,
    event_time_ns: np.ndarray,
    event_asset_code: np.ndarray,
    rel_bars: Sequence[int],
    rank_bars: Sequence[int],
    *,
    min_event_times: int = 12,
    min_assets: int = 3,
) -> pd.DataFrame:
    """Module 03 ``precursor_rankings`` on a percentile tensor.

    Returns one row per feature index with the Module 03 discovery score plus
    the pre-registered b-run summary ``window_mean_percentile`` (mean over the
    rank lags per timestamp, then over timestamps).
    """
    rel_bars = [int(b) for b in rel_bars]
    rank_idx = [rel_bars.index(int(b)) for b in sorted(rank_bars) if int(b) in rel_bars]
    Pr = P[:, rank_idx, :]
    _, T = collapse_by_time(event_time_ns, Pr)                 # (ts, lags, f)
    n_ts = np.isfinite(T).sum(axis=0)                          # (lags, f)
    with np.errstate(invalid="ignore"):
        mean_p = np.nanmean(np.where(np.isfinite(T), T, np.nan), axis=0)
    dev = mean_p - 0.5
    absdev = np.abs(dev)

    codes = np.asarray(event_asset_code)
    asset_ids = np.unique(codes)
    asset_mean = np.full((len(asset_ids), len(rank_idx), P.shape[2]), np.nan)
    for i, a in enumerate(asset_ids):
        x = Pr[codes == a]
        with np.errstate(invalid="ignore"):
            cnt = np.isfinite(x).sum(axis=0)
            m = np.nansum(np.where(np.isfinite(x), x, 0.0), axis=0) / np.maximum(cnt, 1)
        m[cnt == 0] = np.nan
        asset_mean[i] = m

    with np.errstate(invalid="ignore"):
        ts_window = np.nanmean(T, axis=1)                      # (ts, f)
    win_n = np.isfinite(ts_window).sum(axis=0)
    with np.errstate(invalid="ignore"):
        win_mean = np.nanmean(ts_window, axis=0)

    rows = []
    for fj in range(P.shape[2]):
        keep = np.flatnonzero(n_ts[:, fj] >= int(min_event_times))
        rec = {
            "feature_index": fj,
            "window_mean_percentile": float(win_mean[fj]) if win_n[fj] else np.nan,
            "window_n_unique_event_times": int(win_n[fj]),
            "discovery_score": np.nan,
        }
        if len(keep):
            # sort: abs_dev desc, n desc, then original lag order (stable)
            order = sorted(keep, key=lambda li: (-absdev[li, fj], -n_ts[li, fj]))
            best = order[0]
            am = asset_mean[:, best, fj]
            am = am[np.isfinite(am)]
            n_assets = int(len(am))
            if n_assets >= int(min_assets):
                direction = 1.0 if dev[best, fj] >= 0 else -1.0
                consistent = int(np.sum(np.sign(am - 0.5) == direction))
                lag_dirs = np.sign(dev[keep, fj])
                lag_cons = float(np.mean(lag_dirs == direction))
                rec.update({
                    "best_rel_bar": int(sorted(rank_bars)[best]),
                    "best_n_unique_event_times": int(n_ts[best, fj]),
                    "best_mean_percentile": float(mean_p[best, fj]),
                    "n_assets": n_assets,
                    "asset_direction_consistency": consistent / n_assets,
                    "lag_direction_consistency": lag_cons,
                    "discovery_score": float(absdev[best, fj] * (consistent / n_assets) * lag_cons),
                })
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Day-relabel placebo
# --------------------------------------------------------------------------

def valid_days(grid: Grid, min_coverage: float = 0.5) -> np.ndarray:
    """UTC day numbers where every asset has >= min_coverage finite returns."""
    days = np.unique(grid.day)
    keep = []
    for d in days:
        ok = True
        for a in range(len(grid.assets)):
            m = (grid.day == d) & (grid.asset_code == a)
            if np.isfinite(grid.r[m]).sum() < min_coverage * BARS_PER_DAY:
                ok = False
                break
        if ok:
            keep.append(int(d))
    return np.asarray(keep, dtype=np.int64)


def day_relabel_events(
    grid: Grid,
    event_rows: np.ndarray,
    eligible: np.ndarray,
    days: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Placebo events: map each event's UTC day through a random permutation
    of ``days``; keep asset and exact clock slot. Events on the same day move
    together, so cross-asset synchronicity and within-day clustering of the
    real event set are preserved. Placebos landing on a missing or
    non-eligible row are dropped. Events on days outside ``days`` are dropped.
    """
    event_rows = np.asarray(event_rows, dtype=np.int64)
    days = np.asarray(days, dtype=np.int64)
    perm = rng.permutation(days)
    mapping = dict(zip(days.tolist(), perm.tolist()))
    ed = grid.day[event_rows]
    new_day = np.array([mapping.get(int(d), -1) for d in ed], dtype=np.int64)
    ok = new_day >= 0
    bars = new_day * BARS_PER_DAY + grid.slot[event_rows]
    rows = np.full(len(event_rows), -1, dtype=np.int64)
    rows[ok] = grid.row_at(grid.asset_code[event_rows[ok]], bars[ok])
    rows = rows[rows >= 0]
    rows = rows[np.asarray(eligible, bool)[rows]]
    return np.unique(rows)


def sign_flip_pvalue(x: np.ndarray, rng: np.random.Generator, n: int = 10000, alternative: str = "less") -> float:
    """Randomization p-value for mean(x) via random sign flips (H0: symmetric about 0)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    obs = float(np.mean(x))
    s = rng.choice([-1.0, 1.0], size=(int(n), len(x)))
    null = (s * x[None, :]).mean(axis=1)
    if alternative == "less":
        return float((1 + np.sum(null <= obs)) / (n + 1))
    if alternative == "greater":
        return float((1 + np.sum(null >= obs)) / (n + 1))
    return float((1 + np.sum(np.abs(null) >= abs(obs))) / (n + 1))


# --------------------------------------------------------------------------
# Synthetic data + self test
# --------------------------------------------------------------------------

def synthetic_frame(
    n_days: int = 20,
    assets: Sequence[str] = ("AAA", "BBB", "CCC"),
    seed: int = 1,
    missing_share: float = 0.03,
) -> pd.DataFrame:
    """GARCH-like synthetic panel with a common volatility factor.

    Features: intrabar_range_bps (vol proxy), volume_usd (vol + noise),
    noise_feature (pure noise). Useful for unit tests and notebook dry runs.
    """
    rng = np.random.default_rng(seed)
    times = pd.date_range("2026-01-01", periods=n_days * BARS_PER_DAY, freq="5min", tz="UTC")
    T = len(times)
    common = np.zeros(T)
    for t in range(1, T):
        common[t] = 0.985 * common[t - 1] + rng.normal(scale=0.12)
    clock = 0.4 * np.sin(2 * np.pi * (np.arange(T) % BARS_PER_DAY) / BARS_PER_DAY)
    rows = []
    for k, a in enumerate(assets):
        idio = np.zeros(T)
        for t in range(1, T):
            idio[t] = 0.98 * idio[t - 1] + rng.normal(scale=0.08)
        logsig = np.log(0.0015 * (1 + 0.3 * k)) + common + idio + clock
        sig = np.exp(logsig)
        r = sig * rng.standard_t(5, size=T) / np.sqrt(5 / 3)
        rng_bps = 1e4 * sig * np.abs(rng.normal(1.6, 0.4, size=T))
        vol = np.exp(12 + 1.2 * (logsig - logsig.mean()) + rng.normal(scale=0.5, size=T))
        miss = rng.random(T) < missing_share
        r = np.where(miss, np.nan, r)
        rows.append(pd.DataFrame({
            "state_time": times,
            "underlying": a,
            "return_5m": r,
            "intrabar_range_bps": rng_bps,
            "volume_usd": vol,
            "noise_feature": rng.normal(size=T),
        }))
    return pd.concat(rows, ignore_index=True)


def selftest() -> dict:
    f = synthetic_frame(n_days=12, seed=3)
    g = build_grid(f)
    # offset lookup
    i = 500
    assert g.offset_rows(np.array([i]), -1)[0] == i - 1
    assert g.offset_rows(np.array([0]), -1)[0] == -1
    # pandas 3 microsecond-backed timestamps must still map to the grid
    f2 = f.copy()
    f2["state_time"] = pd.DatetimeIndex(f2["state_time"]).as_unit("us")
    g2 = build_grid(f2)
    assert np.array_equal(g2.bar, g.bar)
    # window mean vs brute force
    m = causal_window_mean(g, g.v, -3, -1, 2)
    j = 1000
    vals = g.v[j - 3:j]
    vals = vals[np.isfinite(vals)]
    assert (len(vals) < 2 and np.isnan(m[j])) or np.isclose(m[j], vals.mean())
    # q99 and onsets
    thr = q99_thresholds(g)
    tail = tail_flags(g, thr)
    ons = onset_flags(g, tail)
    assert ons.sum() > 0 and ons.sum() <= tail.sum()
    elig = eligible_origins(g, tail)
    assert not np.any(elig & tail)
    # day relabel keeps slot and asset
    ev = np.flatnonzero(ons)
    rng = np.random.default_rng(0)
    pl = day_relabel_events(g, ev, elig, valid_days(g), rng)
    assert set(g.slot[pl]).issubset(set(g.slot[ev]))
    assert np.all(elig[pl])
    # controls never include the event itself
    cc = clock_controls(g, pl, elig)
    assert not np.any(cc == pl[:, None])
    return {
        "dense_grid": True,
        "us_timestamp_backing": True,
        "causal_window_mean": True,
        "q99_onsets": True,
        "day_relabel_placebo": True,
        "no_self_control": True,
    }
