"""
Volatility targets and clock allocation for Module 12 (discussions/001-004).

All quantities are indexed by an ORIGIN row t of one asset, sorted by
state_time. A row's state_time is the instant of the index snapshot that
closes the five-minute return r_t = ln(S_t / S_{t-1}); every feature on row t
is known at state_time t. Targets use only returns r_{t+1}, ..., so no target
return can share a row with any feature of its origin.

Primary targets (fixed horizon m bars, m = 6 -> 30m, m = 12 -> 60m):

    RV_{t,m} = sum_{j=1..m} r_{t+j}^2

defined only when the m forward bars are present and contiguous and the last
one closes no later than the origin session's 08:00 UTC expiry (targets that
cross 08:00 are dropped, never spliced).

Provisional terminal target (descriptive only, discussion 004 §3):

    RV^settle_{t} = sum_{j>t} w_j^2 (Delta S_j / S_t)^2

with w_j^2 the bar average of w(u)^2, w(u) = min(1, (T - u) / 30 min): the
martingale variance of the 30-minute settlement average on a five-minute
grid. The log-return analogue sum w_j^2 r_j^2 is reported beside it.

Clock allocation (discussion 004 §4): the market only identifies total
implied variance to expiry W_imp = IV^2 (T - t). The share of it falling in
(t, t + h] is allocated with the training-only New York volatility clock
f(u) of clock_baseline (return_5m, log_abs):

    W~_imp(t, t+h) = W_imp * sum_{j in (t,t+h]} f_j^2 / sum_{j in (t,T]} f_j^2

This is a model-based allocation, not a market-observed forward variance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_engineering.modeling.clock_baseline import (
    FittedFeatureBaseline,
    add_clock_coordinates,
)


VOL_TARGETS_VERSION = "0.1"

BAR = pd.Timedelta(minutes=5)
BAR_MIN = 5.0
EXPIRY_HOUR_UTC = 8
SETTLEMENT_MINUTES = 30.0
MINUTES_PER_YEAR = 365.0 * 1440.0
HORIZONS = {"30m": 6, "60m": 12}
BACK_WINDOWS = {"5m": 1, "30m": 6, "1h": 12, "4h": 48}


# --------------------------------------------------
# Session / expiry keys
# --------------------------------------------------

def expiry_time(times: pd.Series) -> pd.Series:
    """08:00 UTC expiry of the session each state belongs to (the state at
    exactly 08:00 opens the next session)."""
    t = pd.to_datetime(times, utc=True)
    return (t - pd.Timedelta(hours=EXPIRY_HOUR_UTC)).dt.floor("D") + pd.Timedelta(days=1, hours=EXPIRY_HOUR_UTC)


def settlement_weight_sq(bar_end_minutes_to_expiry: np.ndarray) -> np.ndarray:
    """Bar average of w(u)^2 for a bar closing `k` minutes before expiry.

    w(u) = min(1, (T - u) / 30). For the bar (T - k - 5, T - k]:
        mean w^2 = ((k + 5)^3 - k^3) / (3 * 5 * 30^2)   when k + 5 <= 30,
    1 when the bar closes at or before T - 30, and the mixed case in between.
    Negative k (bar after expiry) gets weight 0.
    """
    k = np.asarray(bar_end_minutes_to_expiry, float)
    a = np.clip(k, 0.0, None)                       # distance of bar end
    b = a + BAR_MIN                                 # distance of bar start
    L = SETTLEMENT_MINUTES
    lo, hi = np.minimum(a, L), np.minimum(b, L)
    inside = (hi ** 3 - lo ** 3) / (3.0 * L ** 2)  # integral of (x/L)^2 over [lo, hi]
    outside = np.clip(b, L, None) - np.clip(a, L, None)
    out = (inside + outside) / BAR_MIN
    return np.where(k < 0, 0.0, out)


# --------------------------------------------------
# Raw (unfitted) panel
# --------------------------------------------------

def _num(g: pd.DataFrame, col: str) -> np.ndarray:
    if col not in g:
        return np.full(len(g), np.nan)
    return pd.to_numeric(g[col], errors="coerce").to_numpy(float)


def _bool(g: pd.DataFrame, col: str) -> np.ndarray:
    if col not in g:
        return np.zeros(len(g), bool)
    s = g[col]
    if s.dtype == object:
        s = s.map({"True": True, "False": False, "true": True, "false": False, True: True, False: False})
    return s.fillna(False).astype(bool).to_numpy()


def _contig(times: pd.Series, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """times[b] - times[a] == (b - a) bars, for index arrays a <= b (in range)."""
    t = times.to_numpy()
    return (t[b] - t[a]) == (b - a) * BAR.to_timedelta64()


def forward_sum(x: np.ndarray, times: pd.Series, m: int, expiry: pd.Series) -> np.ndarray:
    """sum_{j=1..m} x_{t+j}; NaN unless all present, contiguous and the
    last bar closes no later than the origin's expiry."""
    n = len(x)
    out = np.full(n, np.nan)
    if n <= m:
        return out
    i = np.arange(n - m)
    j = i + m
    ok = _contig(times, i, j)
    ok &= times.to_numpy()[j] <= expiry.to_numpy()[i]
    c = np.r_[0.0, np.cumsum(np.where(np.isfinite(x), x, 0.0))]
    cnt = np.r_[0, np.cumsum(np.isfinite(x).astype(int))]
    s = c[j + 1] - c[i + 1]
    full = (cnt[j + 1] - cnt[i + 1]) == m
    out[i] = np.where(ok & full, s, np.nan)
    return out


def backward_sum(x: np.ndarray, times: pd.Series, k: int) -> np.ndarray:
    """sum_{i=t-k+1..t} x_i; NaN unless all present and contiguous."""
    n = len(x)
    out = np.full(n, np.nan)
    if n < k:
        return out
    j = np.arange(k - 1, n)
    i = j - (k - 1)
    ok = _contig(times, i, j)
    c = np.r_[0.0, np.cumsum(np.where(np.isfinite(x), x, 0.0))]
    cnt = np.r_[0, np.cumsum(np.isfinite(x).astype(int))]
    s = c[j + 1] - c[i]
    full = (cnt[j + 1] - cnt[i]) == k
    out[j] = np.where(ok & full, s, np.nan)
    return out


def _grid(times: pd.Series, expiry: pd.Series) -> tuple[pd.Timestamp, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Regular five-minute grid covering the rows and their expiries.
    Returns (t0, grid, row positions, expiry positions)."""
    t0 = times.min().floor("D") - pd.Timedelta(days=1)
    t1 = expiry.max() + pd.Timedelta(days=1)
    grid = pd.date_range(t0, t1, freq=BAR, tz="UTC")
    pos = ((times - t0) / BAR).to_numpy().round().astype(int)
    epos = ((expiry - t0) / BAR).to_numpy().round().astype(int)
    return t0, grid, pos, epos


def _weighted_suffix(values: np.ndarray, good: np.ndarray, pos: np.ndarray, epos: np.ndarray,
                     weighted: bool) -> np.ndarray:
    """For each origin p: sum over grid bars j in (pos_p, epos_p] of
    w_j^2 * values_j (w_j^2 = 1 when not weighted); NaN if any bar is bad."""
    out = np.full(len(pos), np.nan)
    for ep in np.unique(epos):
        rows = np.flatnonzero(epos == ep)
        lo = int(pos[rows].min())
        bars = np.arange(lo + 1, ep + 1)
        if len(bars) == 0:
            continue
        w2 = settlement_weight_sq((ep - bars) * BAR_MIN) if weighted else np.ones(len(bars))
        v = np.where(good[bars], values[bars], 0.0) * w2
        sfx = np.r_[np.cumsum(v[::-1])[::-1], 0.0]
        bad = np.r_[np.cumsum((~good[bars])[::-1])[::-1], 0]
        off = pos[rows] - lo
        ok = (bad[off] == 0) & (off < len(bars))
        out[rows] = np.where(ok, sfx[off], np.nan)
    return out


def terminal_targets(r: np.ndarray, S: np.ndarray, times: pd.Series, expiry: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(level-increment target, log-return analogue) to the origin's expiry.

    Every bar from t+1 to the bar closing at the expiry must be present with
    a finite return and positive index levels at both ends; otherwise NaN."""
    _, grid, pos, epos = _grid(times, expiry)
    n = len(grid)
    have = np.zeros(n, bool)
    have[pos] = True
    Sg = np.full(n, np.nan)
    Sg[pos] = S
    rg = np.full(n, np.nan)
    rg[pos] = r
    dS = np.r_[np.nan, np.diff(Sg)]
    good = have & np.r_[False, have[:-1]] & np.isfinite(rg) & np.isfinite(dS) & (Sg > 0) & (np.r_[np.nan, Sg[:-1]] > 0)
    lvl_sum = _weighted_suffix(np.where(good, dS, 0.0) ** 2, good, pos, epos, weighted=True)
    lg = _weighted_suffix(np.where(good, rg, 0.0) ** 2, good, pos, epos, weighted=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        lvl = np.where(S > 0, lvl_sum / S ** 2, np.nan)
    return lvl, lg


def perp_range_estimators(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar Parkinson and Garman-Klass variance from perp TRADE OHLC on
    bars passing the perp integrity gate. Trade ranges include bid-ask
    bounce, so these are regressors, never the primary target."""
    o, h, l, c = (_num(g, k) for k in ("open", "high", "low", "close"))
    ok = _bool(g, "perp_integrity_ok") & (l > 0) & (h >= l) & (o > 0) & (c > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.log(h / l)
        co = np.log(c / o)
    park = hl ** 2 / (4.0 * np.log(2.0))
    gk = 0.5 * hl ** 2 - (2.0 * np.log(2.0) - 1.0) * co ** 2
    return np.where(ok, park, np.nan), np.where(ok, np.maximum(gk, 0.0), np.nan)


def iv_decimal(iv: np.ndarray) -> tuple[np.ndarray, float]:
    """ATM IV as an annualized decimal. Values quoted in percent (median
    above 3) are divided by 100. Returns (iv, divisor)."""
    med = np.nanmedian(iv) if np.isfinite(iv).any() else np.nan
    div = 100.0 if np.isfinite(med) and med > 3.0 else 1.0
    return iv / div, div


@dataclass
class Panel:
    """Unfitted per-asset quantities, aligned with the rows of g."""

    asset: str
    times: pd.Series
    session_date: np.ndarray
    expiry: pd.Series
    tte_min: np.ndarray
    r: np.ndarray
    fwd_rv: dict          # horizon -> RV_{t,m}
    back_rv: dict         # window -> backward sum of r^2
    settle_rv: np.ndarray
    settle_rv_log: np.ndarray
    park_back30: np.ndarray
    gk_back30: np.ndarray
    iv: np.ndarray        # decimal, atm scope only
    iv_divisor: float
    weekend: np.ndarray


def build_panel(g: pd.DataFrame) -> Panel:
    """g: one asset sorted by state_time, with state_time, return_5m (or r),
    futures_index_price, is_weekend and the option / perp columns."""
    g = g.reset_index(drop=True)
    times = pd.to_datetime(g["state_time"], utc=True).reset_index(drop=True)
    if times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError("build_panel: state_time must be unique and increasing within an asset.")
    r = _num(g, "r") if "r" in g else _num(g, "return_5m")
    S = _num(g, "futures_index_price")
    expiry = expiry_time(times)
    tte = ((expiry - times).dt.total_seconds() / 60.0).to_numpy()
    session = (expiry - pd.Timedelta(hours=EXPIRY_HOUR_UTC)).dt.strftime("%Y-%m-%d").to_numpy()

    r2 = r ** 2
    fwd = {h: forward_sum(r2, times, m, expiry) for h, m in HORIZONS.items()}
    back = {w: backward_sum(r2, times, k) for w, k in BACK_WINDOWS.items()}
    lvl, lg = terminal_targets(r, S, times, expiry)
    park, gk = perp_range_estimators(g)

    settle = _bool(g, "in_settlement_window")
    atm = _bool(g, "option_atm_quote_ok") & ~settle
    iv, div = iv_decimal(_num(g, "option_iv_atm_forward"))
    iv = np.where(atm & (iv > 0), iv, np.nan)

    return Panel(
        asset=str(g["underlying"].iloc[0]) if "underlying" in g else "asset",
        times=times, session_date=session, expiry=expiry, tte_min=tte, r=r,
        fwd_rv=fwd, back_rv=back, settle_rv=lvl, settle_rv_log=lg,
        park_back30=backward_sum(park, times, 6), gk_back30=backward_sum(gk, times, 6),
        iv=iv, iv_divisor=div, weekend=_bool(g, "is_weekend"),
    )


# --------------------------------------------------
# Clock profile on a regular grid
# --------------------------------------------------

def clock_factor_sq(fb: FittedFeatureBaseline, times: pd.DatetimeIndex, underlying: str) -> np.ndarray:
    """f(u)^2 = exp(2 (m(u) - c)) of a fitted return_5m (log_abs) baseline at
    arbitrary timestamps: deterministic, so evaluating it ahead of t is not
    lookahead."""
    if fb.feature_spec.transform != "log_abs" or fb.vol_factor_const is None:
        raise ValueError("clock_factor_sq needs the fitted return_5m (log_abs) baseline.")
    t = pd.Series(pd.to_datetime(times, utc=True))
    frame = pd.DataFrame({
        "state_id": np.arange(len(t)),
        "state_time": t,
        "underlying": underlying,
        "is_weekend": (t.dt.dayofweek >= 5).to_numpy(),
    })
    frame = add_clock_coordinates(frame)
    m = fb._location(frame)
    return np.exp(2.0 * (m - fb.vol_factor_const))


@dataclass
class ClockAllocation:
    """Clock sums for every origin row."""

    fwd_mean: dict        # horizon -> mean f^2 over (t, t+h]
    back_mean_30m: np.ndarray
    share: dict           # horizon -> sum_(t,t+h] f^2 / sum_(t,T] f^2
    settle_share: np.ndarray   # sum_(t,T] w^2 f^2 / sum_(t,T] f^2


def clock_allocation(panel: Panel, fb: FittedFeatureBaseline) -> ClockAllocation:
    _, grid, pos, epos = _grid(panel.times, panel.expiry)
    f2 = clock_factor_sq(fb, grid, panel.asset)
    c = np.r_[0.0, np.cumsum(f2)]

    def s(a, b):
        """sum of f2 over grid bars a+1..b."""
        return c[b + 1] - c[a + 1]

    to_exp = s(pos, epos)
    safe = np.where(to_exp > 0, to_exp, np.nan)
    fwd_mean, share = {}, {}
    for h, m in HORIZONS.items():
        fwd_mean[h] = s(pos, pos + m) / m
        share[h] = np.where(pos + m <= epos, s(pos, pos + m) / safe, np.nan)
    back = (c[pos + 1] - c[pos - 5]) / 6.0
    allgood = np.ones(len(f2), bool)
    settle_share = _weighted_suffix(f2, allgood, pos, epos, weighted=True) / safe
    return ClockAllocation(fwd_mean=fwd_mean, back_mean_30m=back, share=share, settle_share=settle_share)


def implied_variance(panel: Panel, alloc: ClockAllocation) -> dict:
    """Total implied variance to expiry and its clock-allocated pieces
    (variance units of a sum of squared log returns)."""
    tte_years = np.clip(panel.tte_min, 0.0, None) / MINUTES_PER_YEAR
    W = panel.iv ** 2 * tte_years
    out = {"W_imp_total": W, "W_imp_settle": W * alloc.settle_share}
    for h in HORIZONS:
        out[f"W_imp_alloc_{h}"] = W * alloc.share[h]
    return out


def alignment_selftest(g: pd.DataFrame, n_cuts: int = 5, seed: int = 0) -> dict:
    """Causality check on real rows: scramble every return and price after a
    cut row c and rebuild the panel. Backward quantities at rows <= c must be
    unchanged; forward targets at c must change. Raises on failure."""
    g = g.reset_index(drop=True)
    base = build_panel(g)
    rng = np.random.default_rng(seed)
    ok_rows = np.flatnonzero(np.isfinite(base.fwd_rv["60m"]) & np.isfinite(base.back_rv["4h"]))
    if len(ok_rows) == 0:
        raise RuntimeError("alignment_selftest: no row with both a 60m target and 4h history.")
    cuts = rng.choice(ok_rows, size=min(n_cuts, len(ok_rows)), replace=False)
    for c in cuts:
        h = g.copy()
        after = np.arange(len(h)) > c
        col = "r" if "r" in h else "return_5m"
        h.loc[after, col] = pd.to_numeric(h.loc[after, col], errors="coerce") * 3.0 + 1e-4
        for k in ("futures_index_price", "open", "high", "low", "close"):
            if k in h:
                h.loc[after, k] = pd.to_numeric(h.loc[after, k], errors="coerce") * 1.01
        p = build_panel(h)
        upto = np.arange(len(h)) <= c
        for w in BACK_WINDOWS:
            a, b = base.back_rv[w][upto], p.back_rv[w][upto]
            if not np.allclose(a, b, equal_nan=True):
                raise RuntimeError(f"alignment_selftest: backward RV {w} uses rows after the origin.")
        for name in ("park_back30", "gk_back30", "iv"):
            if not np.allclose(getattr(base, name)[upto], getattr(p, name)[upto], equal_nan=True):
                raise RuntimeError(f"alignment_selftest: {name} uses rows after the origin.")
        if np.isclose(base.fwd_rv["60m"][c], p.fwd_rv["60m"][c]):
            raise RuntimeError("alignment_selftest: the 60m target ignores returns after the origin.")
    return {"cuts_tested": int(len(cuts)), "passed": True}
