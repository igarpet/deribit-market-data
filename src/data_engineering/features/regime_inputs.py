"""
Regime inputs: anchors, displacement and persistence statistics
(framework v2.2 Step 3, sections 5.1-5.2).

All quantities are causal (state t uses information up to t only) and
computed per asset on the exact 5-minute grid. A gap in the grid, an
ineligible index or a session boundary (for session-scoped anchors)
restarts the affected recursion; the number of bars since the restart is
stored so consumers can require a warm-up.

Primary anchor (framework v2.2): SESSION VWAP of the perpetual.

  The collector aggregates every perpetual trade into 5-minute bars with
      volume_base_i = sum of trade amounts in base currency
      vwap_i        = sum(price * amount_base) / volume_base_i
  and volume_usd_i = sum of USD notionals. For both contract types
  (inverse: amount in USD, base = USD / price; linear: amount in base,
  USD = amount * price) the identity volume_base_i = volume_usd_i / vwap_i
  holds exactly, so the session VWAP of all gated trades since 08:00 UTC is
  reconstructed exactly from the bar aggregates:

      A_t = sum_{i<=t, s(i)=s(t)} w_i vwap_i / sum w_i,   w_i = volume_usd_i / vwap_i

  (base-currency weights; v0.2 weighted by USD notional, which is not the
  traded VWAP of an inverse contract).

  Trade gate: a bar's trades enter the VWAP unless the bar has a known
  exchange event or trades printed outside the book BBO (the trade-quality
  flags of the futures integrity gate). Book-quality flags (stale or wide
  book) do not remove genuine trades. Without the flag columns the full
  integrity gate perp_integrity_ok is used.

  Price: P_t = perpetual mid at the bar close (same instrument as the
  VWAP, so the perpetual-index basis cancels), required to pass the full
  integrity gate.

  Exact standardization. Under a random walk with per-bar volatility
  sigma_j and trades spread evenly within each bar, an increment at
  fraction u of bar j enters G_t = log P_t - log A_t with coefficient
  (C_{j-1} + u w_j) / C_t, C_j being the cumulative weight through bar j of
  the session. Integrating over u:

      Var(G_t) = sum_j sigma_j^2 (C_{j-1}^2 + C_{j-1} w_j + w_j^2 / 3) / C_t^2.

  So

      D_t = G_t / sqrt(Var(G_t))

  has unit variance at every point of the session (Var about sigma^2 n / 3
  with constant volume after n bars, sigma^2 / 3 in the first bar), and the
  shrinking sensitivity of a cumulative VWAP late in the session is
  accounted for. h_t = Var(G_t) / sigma_t^2 is stored as the effective
  horizon in bars.

Secondary anchors (diagnostics only):

  * index EWMA, half-life H bars (H = 6, 12, 24), restarted at 08:00 UTC,
    h_eff(H) = lam / (1 - lam^2);
  * rolling VWAP of the perpetual over W bars (W = 12, 48), base weights,
    h = W / 2.

Persistence statistics on index returns over the trailing N bars (all N
returns must exist and be eligible; they may cross the 08:00 boundary
because the index is continuous):

    VR_t(q) = mean_j (sum_{i=0}^{q-1} r_{j-i})^2 / (q * mean r^2),  q = 3, 6; N = 48
    ER_t(N) = |sum r_i| / sum |r_i|,  N = 12, 48

Version history
  0.1  first version (6 Oct 2026).
  0.2  EWMA displacement in standard units, h = h_eff(H).
  0.3  session VWAP is the primary anchor: base-currency weights (exact
       trade VWAP), trade-quality gate, exact variance standardization;
       helpers shared with the HSMM design and its null.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


REGIME_INPUT_VERSION = "0.3"

PERIODS_PER_YEAR = 365.0 * 24.0 * 12.0
BAR = pd.Timedelta(minutes=5)

EWMA_HALF_LIVES = (6, 12, 24)
ROLLING_VWAP_WINDOWS = (12, 48)
ROLLING_MIN_COVERAGE = 0.75
SESSION_VWAP_MIN_TRADE_BARS = 1

VR_HORIZONS = (3, 6)
VR_WINDOW = 48
ER_WINDOWS = (12, 48)

REQUIRED_COLUMNS = (
    "state_id", "state_time", "underlying", "session_id",
    "futures_index_price", "return_5m", "realized_vol_1h_annualized",
)

TRADE_FLAG_COLUMNS = ("known_exchange_event", "integrity_flag_trades_outside_bbo")
VWAP_SOURCE_COLUMNS = ("vwap", "volume_usd", "futures_tradable_mid_price", "perp_integrity_ok")


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.map({"True": True, "False": False, "true": True,
                             "false": False, True: True, False: False})
    return series.fillna(False).astype(bool)


def _contiguous(times: pd.Series) -> np.ndarray:
    """True where the previous row of the same asset is exactly one bar earlier."""
    return (times.diff() == BAR).to_numpy()


def session_key(times: pd.Series) -> np.ndarray:
    """Session (expiry date) of each state: sessions run 08:00 -> 08:00 UTC."""
    t = pd.to_datetime(times, utc=True)
    return ((t - pd.Timedelta(hours=8)).dt.floor("D") + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d").to_numpy()


# --------------------------------------------------
# Session VWAP inputs (shared with the HSMM design and its null)
# --------------------------------------------------

def base_volume(volume_usd: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    """Base-currency volume of a bar, exact for inverse and linear perpetuals."""
    v = np.asarray(volume_usd, float)
    p = np.asarray(vwap, float)
    ok = np.isfinite(v) & np.isfinite(p) & (v > 0) & (p > 0)
    return np.where(ok, v / np.where(ok, p, 1.0), 0.0)


def vwap_trade_gate(g: pd.DataFrame) -> np.ndarray:
    """Bars whose trades enter the VWAP: trades present, positive VWAP and
    volume, and no trade-quality flag (known event, prints outside the BBO)."""
    vwap = pd.to_numeric(g["vwap"], errors="coerce").to_numpy(float)
    vol = pd.to_numeric(g["volume_usd"], errors="coerce").to_numpy(float)
    ok = np.isfinite(vwap) & (vwap > 0) & np.isfinite(vol) & (vol > 0)
    if "trade_count" in g:
        ok &= pd.to_numeric(g["trade_count"], errors="coerce").fillna(0).to_numpy() > 0
    flags = [c for c in TRADE_FLAG_COLUMNS if c in g]
    if flags:
        for c in flags:
            ok &= ~_as_bool(g[c]).to_numpy()
    elif "perp_integrity_ok" in g:
        ok &= _as_bool(g["perp_integrity_ok"]).to_numpy()
    return ok


def perp_log_mid(g: pd.DataFrame) -> np.ndarray:
    """log perpetual mid at the bar close where the full integrity gate passes."""
    mid = pd.to_numeric(g["futures_tradable_mid_price"], errors="coerce").to_numpy(float)
    ok = np.isfinite(mid) & (mid > 0)
    if "perp_integrity_ok" in g:
        ok &= _as_bool(g["perp_integrity_ok"]).to_numpy()
    return np.where(ok, np.log(np.where(ok, mid, 1.0)), np.nan)


def session_vwap_displacement(log_price: np.ndarray, bar_vwap: np.ndarray, weight: np.ndarray,
                              trade_ok: np.ndarray, new_session: np.ndarray,
                              sigma: np.ndarray) -> dict[str, np.ndarray]:
    """
    Session VWAP anchor and exactly standardized displacement.

    log_price: log P_t (NaN where unusable); bar_vwap: bar VWAP level;
    weight: base-currency volume; trade_ok: bars whose trades enter;
    new_session: True at the first row of each session; sigma: per-bar
    volatility at each row (NaN allowed: such bars use the current sigma).

    Returns arrays: log_anchor, gap, var (Var(G_t)), h (Var / sigma_t^2,
    effective horizon in bars), D, weight_cum (base units), trade_bars.
    """
    n = len(log_price)
    out = {k: np.full(n, np.nan) for k in ("log_anchor", "gap", "var", "h", "D", "weight_cum")}
    out["trade_bars"] = np.zeros(n, dtype=int)
    w = np.where(np.asarray(trade_ok, bool), np.asarray(weight, float), 0.0)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    v = np.asarray(bar_vwap, float)
    s = np.asarray(sigma, float)
    C = num = Qk = Qu = 0.0
    nb = 0
    last_sigma = np.nan
    for i in range(n):
        if new_session[i]:
            C = num = Qk = Qu = 0.0
            nb = 0
        c2 = C * C + C * w[i] + w[i] * w[i] / 3.0
        if np.isfinite(s[i]) and s[i] > 0:
            Qk += s[i] * s[i] * c2
            last_sigma = s[i]
        else:
            Qu += c2
        if w[i] > 0:
            C += w[i]
            num += w[i] * v[i]
            nb += 1
        out["trade_bars"][i] = nb
        if C <= 0:
            continue
        out["weight_cum"][i] = C
        la = math.log(num / C)
        out["log_anchor"][i] = la
        cur = s[i] if np.isfinite(s[i]) and s[i] > 0 else last_sigma
        if not np.isfinite(cur):
            continue
        var = (Qk + Qu * cur * cur) / (C * C)
        out["var"][i] = var
        out["h"][i] = var / (cur * cur)
        if np.isfinite(log_price[i]) and var > 0:
            out["gap"][i] = log_price[i] - la
            out["D"][i] = out["gap"][i] / math.sqrt(var)
    return out


def synthetic_perp_path(log_index_star: np.ndarray, basis: np.ndarray, sigma: np.ndarray,
                        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Null construction: simulated perpetual log mid = simulated log index +
    real basis. Each bar's VWAP is the time average of a Brownian path between
    the bar's start and end prices with trades spread evenly:

        log VWAP*_i = (log P*_{i-1} + log P*_i) / 2 + eta_i,  eta_i ~ N(0, sigma_i^2 / 12)

    (the Brownian-bridge average), which is the assumption of the exact
    standardization. Returns (log mid*, bar VWAP* level)."""
    lp = np.asarray(log_index_star, float) + np.asarray(basis, float)
    prev = np.r_[np.nan, lp[:-1]]
    s = np.asarray(sigma, float)
    s = np.where(np.isfinite(s) & (s > 0), s, np.nanmedian(s[np.isfinite(s) & (s > 0)]) if np.any(np.isfinite(s) & (s > 0)) else 0.0)
    eta = rng.standard_normal(len(lp)) * s / math.sqrt(12.0)
    centre = np.where(np.isfinite(prev), 0.5 * (prev + lp), lp)
    return lp, np.exp(centre + eta)


# --------------------------------------------------
# Secondary anchors
# --------------------------------------------------

def ewma_h_eff(half_life: int) -> float:
    """Stationary Var(log I - EWMA anchor) / per-bar variance for a random walk."""
    lam = 0.5 ** (1.0 / half_life)
    return lam / (1.0 - lam * lam)


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


def rolling_vwap(vwap: np.ndarray, weight: np.ndarray, usable: np.ndarray,
                 present: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing W-bar VWAP over usable bars (base-currency weights). `present`
    marks rows contiguous with the previous one; a gap resets the window.
    Returns (anchor, coverage = usable bars / W)."""

    out = np.full(len(vwap), np.nan)
    cov = np.zeros(len(vwap))
    pv = np.where(usable, vwap * weight, 0.0)
    v = np.where(usable, weight, 0.0)
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

    # primary anchor: session VWAP
    if set(VWAP_SOURCE_COLUMNS[:3]) <= set(g.columns):
        vwap = pd.to_numeric(g["vwap"], errors="coerce").to_numpy(float)
        w = base_volume(pd.to_numeric(g["volume_usd"], errors="coerce").to_numpy(float), vwap)
        trade_ok = vwap_trade_gate(g)
        log_mid = perp_log_mid(g)
        sv = session_vwap_displacement(log_mid, vwap, w, trade_ok, new_session, sigma)
        out["vwap_trade_ok"] = trade_ok
        out["session_vwap"] = np.exp(sv["log_anchor"])
        out["session_vwap_gap"] = sv["gap"]
        out["session_vwap_h"] = sv["h"]
        out["session_vwap_trade_bars"] = sv["trade_bars"]
        out["session_vwap_weight_base"] = sv["weight_cum"]
        out["D_session_vwap"] = sv["D"]
        out["D_session_vwap_ready"] = np.isfinite(sv["D"]) & (sv["trade_bars"] >= SESSION_VWAP_MIN_TRADE_BARS)

        for W in ROLLING_VWAP_WINDOWS:
            rv, cov = rolling_vwap(np.nan_to_num(vwap), w, trade_ok, present, W)
            gap = log_mid - np.log(rv)
            out[f"rvwap_{W}_gap"] = gap
            out[f"rvwap_{W}_coverage"] = cov
            out[f"D_rvwap_{W}"] = gap / (sigma * math.sqrt(W / 2.0))
            out[f"D_rvwap_{W}_ready"] = np.isfinite(out[f"D_rvwap_{W}"])

    restart = new_session | ~present
    for H in EWMA_HALF_LIVES:
        a, age = ewma_anchor(log_index, restart, H)
        gap = log_index - a
        out[f"ewma_hl{H}_gap"] = gap
        out[f"ewma_hl{H}_age"] = age
        out[f"D_ewma_hl{H}"] = gap / (sigma * math.sqrt(ewma_h_eff(H)))
        out[f"D_ewma_hl{H}_ready"] = np.isfinite(out[f"D_ewma_hl{H}"]) & (age >= H)

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
