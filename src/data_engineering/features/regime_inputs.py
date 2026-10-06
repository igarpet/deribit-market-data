"""
Regime inputs: anchors, displacement and persistence statistics
(framework v2.1 Step 3, sections 5.1-5.2).

All quantities are causal (state t uses information up to t only) and
computed per asset on the exact 5-minute grid. A gap in the grid, an
ineligible index or a session boundary (for session-scoped anchors)
restarts the affected recursion; the number of bars since the restart is
stored so consumers can require a warm-up.

Anchors A_t and their gaps G_t = log P_t - log A_t:

  * index EWMA, half-life H bars (H = 6, 12, 24), restarted at 08:00 UTC:
        log A_t = lam log A_{t-1} + (1 - lam) log I_t,   lam = 0.5^(1/H)
    P_t = Deribit index I_t.
  * session VWAP of the perpetual since 08:00 UTC, from integrity-gated
    bars with volume only:  A_t = sum v_i VWAP_i / sum v_i.
    P_t = perpetual mid (same instrument, so the basis cancels).
  * rolling VWAP of the perpetual over the trailing W bars (W = 12, 48),
    gated bars only, requiring at least ROLLING_MIN_COVERAGE of the window.
    P_t = perpetual mid.

Displacement in units of expected movement over the anchor's horizon:

    D_t = G_t / (sigma_t sqrt(h)),

    sigma_t: trailing per-bar volatility (1h realized volatility of index
             returns, de-annualized). This is the RAW scale; the regime
             layer rescales with the clock-adjusted scale
             (modeling.clock_baseline), so G_t is stored as well.
    h:       H for EWMA anchors; bars since 08:00 for session VWAP;
             W / 2 for rolling VWAP (centre of mass of the window).

Persistence statistics on index returns over the trailing N bars (all N
returns must exist and be eligible; they may cross the 08:00 boundary
because the index is continuous):

    VR_t(q) = mean_j (sum_{i=0}^{q-1} r_{j-i})^2 / (q * mean r^2)
              (overlapping q-bar sums, zero-mean; > 1 persistence,
               < 1 anti-persistence),  q = 3, 6; N = 48
    ER_t(N) = |sum r_i| / sum |r_i|,  N = 12, 48  (0 noise .. 1 straight line)

Version history
  0.1  first version (6 Oct 2026).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


REGIME_INPUT_VERSION = "0.1"

PERIODS_PER_YEAR = 365.0 * 24.0 * 12.0
BAR = pd.Timedelta(minutes=5)

EWMA_HALF_LIVES = (6, 12, 24)
ROLLING_VWAP_WINDOWS = (12, 48)
ROLLING_MIN_COVERAGE = 0.75
SESSION_VWAP_MIN_BARS = 12

VR_HORIZONS = (3, 6)
VR_WINDOW = 48
ER_WINDOWS = (12, 48)

REQUIRED_COLUMNS = (
    "state_id", "state_time", "underlying", "session_id",
    "futures_index_price", "return_5m", "realized_vol_1h_annualized",
)


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.map({"True": True, "False": False, "true": True,
                             "false": False, True: True, False: False})
    return series.fillna(False).astype(bool)


def _contiguous(times: pd.Series) -> np.ndarray:
    """True where the previous row of the same asset is exactly one bar earlier."""
    return (times.diff() == BAR).to_numpy()


# --------------------------------------------------
# Anchors
# --------------------------------------------------

def ewma_anchor(log_price: np.ndarray, restart: np.ndarray, half_life: int) -> tuple[np.ndarray, np.ndarray]:
    """Causal EWMA of log price. `restart[i]` starts a new recursion at i.
    Returns (log anchor, bars since restart incl. current). NaN prices
    break the recursion; the next valid price restarts it."""

    lam = 0.5 ** (1.0 / half_life)
    out = np.full(len(log_price), np.nan)
    age = np.zeros(len(log_price), dtype=int)
    prev, n = np.nan, 0
    for i, x in enumerate(log_price):
        if not np.isfinite(x):
            prev, n = np.nan, 0
            continue
        if restart[i] or not np.isfinite(prev):
            prev, n = x, 1
        else:
            prev, n = lam * prev + (1.0 - lam) * x, n + 1
        out[i], age[i] = prev, n
    return out, age


def session_vwap(vwap: np.ndarray, volume: np.ndarray, usable: np.ndarray,
                 restart: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Volume-weighted VWAP since the last restart, from usable bars.
    Returns (anchor, usable bars accumulated)."""

    out = np.full(len(vwap), np.nan)
    bars = np.zeros(len(vwap), dtype=int)
    num = den = 0.0
    n = 0
    for i in range(len(vwap)):
        if restart[i]:
            num = den = 0.0
            n = 0
        if usable[i]:
            num += vwap[i] * volume[i]
            den += volume[i]
            n += 1
        if den > 0:
            out[i] = num / den
        bars[i] = n
    return out, bars


def rolling_vwap(vwap: np.ndarray, volume: np.ndarray, usable: np.ndarray,
                 present: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing W-bar VWAP over usable bars. `present` marks rows that are
    contiguous with the previous one; a gap resets the window. Returns
    (anchor, coverage = usable bars / W)."""

    out = np.full(len(vwap), np.nan)
    cov = np.zeros(len(vwap))
    pv = np.where(usable, vwap * volume, 0.0)
    v = np.where(usable, volume, 0.0)
    u = usable.astype(float)
    run_start = 0
    for i in range(len(vwap)):
        if not present[i]:
            run_start = i
        lo = max(run_start, i - window + 1)
        if i - lo + 1 < window:
            continue
        den = v[lo:i + 1].sum()
        cov[i] = u[lo:i + 1].sum() / window
        if den > 0 and cov[i] >= ROLLING_MIN_COVERAGE:
            out[i] = pv[lo:i + 1].sum() / den
    return out, cov


# --------------------------------------------------
# Persistence statistics
# --------------------------------------------------

def trailing_complete(r: np.ndarray, present: np.ndarray, window: int) -> np.ndarray:
    """True where the last `window` returns exist and are contiguous."""

    ok = np.isfinite(r)
    run = np.zeros(len(r), dtype=int)
    for i in range(len(r)):
        if not ok[i]:
            run[i] = 0
        elif i > 0 and present[i]:
            run[i] = run[i - 1] + 1
        else:
            run[i] = 1
    return run >= window


def variance_ratio(r: np.ndarray, complete: np.ndarray, q: int, window: int) -> np.ndarray:
    out = np.full(len(r), np.nan)
    for i in np.flatnonzero(complete):
        x = r[i - window + 1:i + 1]
        denom = q * np.mean(x ** 2)
        if denom <= 0:
            continue
        sums = np.convolve(x, np.ones(q), mode="valid")
        out[i] = np.mean(sums ** 2) / denom
    return out


def efficiency_ratio(r: np.ndarray, complete: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(r), np.nan)
    for i in np.flatnonzero(complete):
        x = r[i - window + 1:i + 1]
        path = np.abs(x).sum()
        if path > 0:
            out[i] = abs(x.sum()) / path
    return out


# --------------------------------------------------
# Assembly
# --------------------------------------------------

def compute_asset(g: pd.DataFrame) -> pd.DataFrame:
    """All regime inputs for one asset (rows sorted by state_time)."""

    g = g.sort_values("state_time")
    times = pd.to_datetime(g["state_time"], utc=True)
    present = _contiguous(times)
    session = g["session_id"].astype(str).to_numpy()
    new_session = np.r_[True, session[1:] != session[:-1]]

    eligible = (
        _as_bool(g["index_return_input_eligible"]).to_numpy()
        if "index_return_input_eligible" in g
        else pd.to_numeric(g["return_5m"], errors="coerce").notna().to_numpy()
    )
    index = pd.to_numeric(g["futures_index_price"], errors="coerce").to_numpy(float)
    log_index = np.where(eligible & (index > 0), np.log(np.where(index > 0, index, 1.0)), np.nan)
    r = pd.to_numeric(g["return_5m"], errors="coerce").to_numpy(float)
    sigma = pd.to_numeric(g["realized_vol_1h_annualized"], errors="coerce").to_numpy(float) / math.sqrt(PERIODS_PER_YEAR)
    sigma = np.where(sigma > 0, sigma, np.nan)

    out = pd.DataFrame({
        "state_id": g["state_id"].to_numpy(),
        "state_time": g["state_time"].to_numpy(),
        "underlying": g["underlying"].to_numpy(),
        "session_id": session,
        "log_index": log_index,
        "sigma_bar_raw": sigma,
    })

    restart = new_session | ~present
    for H in EWMA_HALF_LIVES:
        a, age = ewma_anchor(log_index, restart, H)
        gap = log_index - a
        out[f"ewma_hl{H}_gap"] = gap
        out[f"ewma_hl{H}_age"] = age
        out[f"D_ewma_hl{H}"] = gap / (sigma * math.sqrt(H))
        out[f"D_ewma_hl{H}_ready"] = np.isfinite(out[f"D_ewma_hl{H}"]) & (age >= H)

    perp_ok = _as_bool(g["perp_integrity_ok"]).to_numpy() if "perp_integrity_ok" in g else np.zeros(len(g), bool)
    if {"vwap", "volume_usd", "futures_tradable_mid_price"} <= set(g.columns):
        vwap = pd.to_numeric(g["vwap"], errors="coerce").to_numpy(float)
        vol = pd.to_numeric(g["volume_usd"], errors="coerce").to_numpy(float)
        mid = pd.to_numeric(g["futures_tradable_mid_price"], errors="coerce").to_numpy(float)
        usable = perp_ok & np.isfinite(vwap) & (vwap > 0) & np.isfinite(vol) & (vol > 0)
        log_mid = np.where(perp_ok & (mid > 0), np.log(np.where(mid > 0, mid, 1.0)), np.nan)

        sv, bars = session_vwap(np.nan_to_num(vwap), np.nan_to_num(vol), usable, new_session)
        gap = log_mid - np.log(sv)
        out["session_vwap_gap"] = gap
        out["session_vwap_bars"] = bars
        out["D_session_vwap"] = gap / (sigma * np.sqrt(np.maximum(bars, 1)))
        out["D_session_vwap_ready"] = np.isfinite(out["D_session_vwap"]) & (bars >= SESSION_VWAP_MIN_BARS)

        for W in ROLLING_VWAP_WINDOWS:
            rv, cov = rolling_vwap(np.nan_to_num(vwap), np.nan_to_num(vol), usable, present, W)
            gap = log_mid - np.log(rv)
            out[f"rvwap_{W}_gap"] = gap
            out[f"rvwap_{W}_coverage"] = cov
            out[f"D_rvwap_{W}"] = gap / (sigma * math.sqrt(W / 2.0))
            out[f"D_rvwap_{W}_ready"] = np.isfinite(out[f"D_rvwap_{W}"])

    windows = sorted(set(ER_WINDOWS) | {VR_WINDOW})
    complete = {N: trailing_complete(r, present, N) for N in windows}
    for q in VR_HORIZONS:
        out[f"vr_q{q}_n{VR_WINDOW}"] = variance_ratio(r, complete[VR_WINDOW], q, VR_WINDOW)
    for N in ER_WINDOWS:
        out[f"er_n{N}"] = efficiency_ratio(r, complete[N], N)

    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].where(np.isfinite(out[c]))
    return out


def compute_regime_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in frame]
    if missing:
        raise ValueError(f"Regime input source lacks columns: {missing}")
    if frame["state_id"].duplicated().any():
        raise ValueError("Duplicate state_id in regime input source.")
    parts = [compute_asset(g) for _, g in frame.groupby("underlying", sort=True)]
    out = pd.concat(parts, ignore_index=True)
    out["regime_input_version"] = REGIME_INPUT_VERSION
    return out
