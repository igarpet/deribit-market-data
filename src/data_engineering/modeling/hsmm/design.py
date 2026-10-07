"""
Design matrices for the HSMM from per-asset state rows (feature contract v0).

For one asset, on the exact 5-minute grid:

    r~_t      = r_t / f_t                         (clock-adjusted return; f from clock_baseline)
    sigma~_t  = EWMA_{hl 12}(r~^2)^(1/2)          (causal, restarted on gaps)
    sigma^c_t = f_t * sigma~_t
    z_t       = r_t / sigma^c_{t-1}               (standardized return)
    Anchor (framework v2.2): session VWAP of the perpetual since 08:00 UTC,
    rebuilt exactly from the collector's bar aggregates (base-currency
    weights w_i = volume_usd_i / vwap_i, trade-quality gate):
    G_t       = log P_t - log A_t                 (P = perpetual mid at the bar close, full integrity gate)
    D_t       = G_t / sqrt(Var(G_t)),  Var(G_t) = sum_j (sigma^c_j)^2 (C_{j-1}^2 + C_{j-1} w_j + w_j^2/3) / C_t^2
                                                  (exact random-walk variance; regime_inputs v0.3)
    The VWAP resets at 08:00; the underlying chain stays continuous (contract v0):
    D is defined from the first traded bar of each session, so segments do not
    break at 08:00.

    anchor="ewma" keeps the v0.1 index EWMA anchor (half-life H, h_eff(H)) for
    comparison only.

Model row t:  y_t = z_t;  state-specific [1, D_{t-1}];  shared [z_{t-1}, z_{t-2}, z_{t-3}]
(all regressors known at t-1). Rows need every term finite and the grid
contiguous from t-3 to t; contiguous runs of valid rows form segments.

The same builder is used for real returns and for simulated (null) returns, so
the null passes through exactly the same pipeline.

Exogenous regressors (`extra`, e.g. factor scores from modeling.factors) are
taken as given at state t, lagged one bar like D (known at t-1 for row t),
and appended to the columns after z3; unavailable values are masked terms.
They are not rebuilt under the null. For simulated returns the
perpetual path is rebuilt as simulated index + real basis, the bar VWAPs as
Brownian-bridge averages within each bar, with the real volume weights and
the real trade and price gates (regime_inputs.synthetic_perp_path).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data_engineering.features.regime_inputs import (
    base_volume,
    ewma_anchor,
    ewma_h_eff,
    perp_log_mid,
    session_vwap_displacement,
    synthetic_perp_path,
    trailing_complete,
    variance_ratio,
    vwap_trade_gate,
)
from data_engineering.modeling.hsmm.engine import HSMMData


SIGMA_HALF_LIFE = 12
SIGMA_SEED_BARS = 12
ANCHOR_HALF_LIFE = 12
DEFAULT_ANCHOR = "session_vwap"
ANCHORS = ("session_vwap", "ewma")
VWAP_COLUMNS = ("vwap", "volume_usd", "futures_tradable_mid_price", "futures_index_price")
N_LAGS = 3
MIN_SEGMENT = 24
BAR = pd.Timedelta(minutes=5)

ALL_COLUMNS = ["const", "D", "z1", "z2", "z3"]
STRUCTURAL = (["const", "D"], ["z1", "z2", "z3"])


def ewma_sigma(r_tilde: np.ndarray, contiguous: np.ndarray, half_life: int = SIGMA_HALF_LIFE,
               seed_bars: int = SIGMA_SEED_BARS) -> np.ndarray:
    """Causal EWMA volatility; restarts (with a seed window) on gaps or missing returns."""
    lam = 0.5 ** (1.0 / half_life)
    out = np.full(len(r_tilde), np.nan)
    v, n, seed = np.nan, 0, []
    for i, x in enumerate(r_tilde):
        if not contiguous[i] or not np.isfinite(x):
            v, n, seed = np.nan, 0, []
            if not np.isfinite(x):
                continue
        if n < seed_bars:
            seed.append(x * x)
            n += 1
            if n == seed_bars:
                v = float(np.mean(seed))
                out[i] = math.sqrt(v) if v > 0 else np.nan
            continue
        v = lam * v + (1 - lam) * x * x
        out[i] = math.sqrt(v) if v > 0 else np.nan
    return out


@dataclass
class Design:
    asset: str
    frame: pd.DataFrame          # one row per model row: state_time, session_date, y, regressors, block
    columns: np.ndarray          # (T, 5) ALL_COLUMNS
    starts: np.ndarray
    ends: np.ndarray
    block_names: list
    block: np.ndarray | None
    block_mask: np.ndarray | None
    anchor: str = DEFAULT_ANCHOR
    coverage: dict | None = None
    col_names: list = field(default_factory=lambda: list(ALL_COLUMNS))

    @property
    def extra_names(self) -> list:
        return [c for c in self.col_names if c not in ALL_COLUMNS]

    def data(self, xs=STRUCTURAL[0], xc=STRUCTURAL[1], rows: np.ndarray | None = None,
             with_block: bool = False, block_weight: float = 1.0) -> HSMMData:
        """HSMMData for a column choice, optionally restricted to a boolean row mask
        (segments are cut where the mask is False). block_weight tempers the
        block log density (engine HSMMData.block_weight)."""
        idx_s = [self.col_names.index(c) for c in xs]
        idx_c = [self.col_names.index(c) for c in xc]
        keep = np.ones(len(self.frame), bool) if rows is None else np.asarray(rows, bool)
        seg = np.zeros(len(self.frame), int)
        for i, (a, b) in enumerate(zip(self.starts, self.ends)):
            seg[a:b] = i
        pos = np.flatnonzero(keep)
        if len(pos) == 0:
            raise ValueError("No rows selected.")
        brk = np.r_[True, (np.diff(pos) != 1) | (seg[pos][1:] != seg[pos][:-1])]
        st = np.flatnonzero(brk)
        en = np.r_[st[1:], len(pos)]
        ok = (en - st) >= MIN_SEGMENT
        sel = np.concatenate([np.arange(a, b) for a, b in zip(st[ok], en[ok])]) if ok.any() else np.array([], int)
        lengths = (en - st)[ok]
        new_st = np.r_[0, np.cumsum(lengths)[:-1]] if len(lengths) else np.array([], int)
        new_en = np.cumsum(lengths)
        rows_ = pos[sel]
        O = M = None
        names_o = []
        if with_block and self.block is not None:
            O, M, names_o = self.block[rows_], self.block_mask[rows_], self.block_names
        return HSMMData(self.frame["y"].to_numpy()[rows_], self.columns[rows_][:, idx_s],
                        self.columns[rows_][:, idx_c], new_st.astype(int), new_en.astype(int),
                        list(xs), list(xc), O, M, names_o, block_weight=float(block_weight))

    def rows_of(self, rows: np.ndarray | None = None) -> np.ndarray:
        """Row positions (into frame) that data(rows=...) keeps, in order."""
        keep = np.ones(len(self.frame), bool) if rows is None else np.asarray(rows, bool)
        seg = np.zeros(len(self.frame), int)
        for i, (a, b) in enumerate(zip(self.starts, self.ends)):
            seg[a:b] = i
        pos = np.flatnonzero(keep)
        brk = np.r_[True, (np.diff(pos) != 1) | (seg[pos][1:] != seg[pos][:-1])]
        st = np.flatnonzero(brk)
        en = np.r_[st[1:], len(pos)]
        ok = (en - st) >= MIN_SEGMENT
        return pos[np.concatenate([np.arange(a, b) for a, b in zip(st[ok], en[ok])])] if ok.any() else np.array([], int)


def _session_vwap_D(g: pd.DataFrame, r: np.ndarray, sigma_c: np.ndarray, new_session: np.ndarray,
                    simulated: bool, rng: np.random.Generator | None) -> tuple[np.ndarray, np.ndarray, dict]:
    missing = [c for c in VWAP_COLUMNS if c not in g]
    if missing:
        raise ValueError(f"Session VWAP anchor needs columns {missing}")
    vwap = pd.to_numeric(g["vwap"], errors="coerce").to_numpy(float)
    w = base_volume(pd.to_numeric(g["volume_usd"], errors="coerce").to_numpy(float), vwap)
    trade_ok = vwap_trade_gate(g)
    log_mid = perp_log_mid(g)
    if simulated:
        idx = pd.to_numeric(g["futures_index_price"], errors="coerce").to_numpy(float)
        log_idx = np.where(idx > 0, np.log(np.where(idx > 0, idx, 1.0)), np.nan)
        basis = log_mid - log_idx
        fill = np.nanmedian(basis) if np.isfinite(basis).any() else 0.0
        basis_f = pd.Series(basis).ffill().fillna(fill).to_numpy()
        first = np.flatnonzero(np.isfinite(log_idx))
        start = log_idx[first[0]] if len(first) else 0.0
        log_idx_star = start + np.cumsum(np.where(np.isfinite(r), r, 0.0))
        lp_full, vwap = synthetic_perp_path(log_idx_star, basis_f, sigma_c, rng or np.random.default_rng(0))
        log_mid = np.where(np.isfinite(log_mid), lp_full, np.nan)        # real price gate
    sv = session_vwap_displacement(log_mid, vwap, w, trade_ok, new_session, sigma_c)
    cov = {"vwap_trade_bars": float(trade_ok.mean()), "price_ok": float(np.isfinite(log_mid).mean()),
           "D_finite": float(np.isfinite(sv["D"]).mean())}
    return sv["D"], sv["h"], cov


def build_design(g: pd.DataFrame, f_clock: np.ndarray, returns: np.ndarray | None = None,
                 log_price: np.ndarray | None = None, block: pd.DataFrame | None = None,
                 block_mask: pd.DataFrame | None = None, anchor: str = DEFAULT_ANCHOR,
                 anchor_half_life: int = ANCHOR_HALF_LIFE, rng: np.random.Generator | None = None,
                 extra: pd.DataFrame | None = None) -> Design:
    """
    g: one asset, sorted by state_time, with state_time and (unless `returns`
    is given) r; for the session VWAP anchor also vwap, volume_usd,
    futures_tradable_mid_price, futures_index_price and the gate columns
    (perp_integrity_ok; trade_count, known_exchange_event,
    integrity_flag_trades_outside_bbo when available).

    returns: simulated returns (null). The anchor inputs are then rebuilt from
    them (see module docstring); rng drives the within-bar VWAP noise.
    log_price: EWMA anchor only; defaults to log futures_index_price.
    extra: exogenous regressors aligned with the rows of g; lagged one bar. NaN
    marks an unavailable factor (framework v3.2 §6.7): its term is masked out
    of the mean for that row (numerically a zero regressor value, so the
    slope is estimated only from rows where the factor is available); the
    factor score itself is never set to a neutral value.
    """
    if anchor not in ANCHORS:
        raise ValueError(f"anchor must be one of {ANCHORS}")
    asset = str(g["underlying"].iloc[0]) if "underlying" in g else "asset"
    times = pd.to_datetime(g["state_time"], utc=True).reset_index(drop=True)
    contiguous = np.r_[False, (times.diff().iloc[1:] == BAR).to_numpy()]
    r = np.asarray(g["r"] if returns is None else returns, float)
    f = np.asarray(f_clock, float)
    f = np.where(np.isfinite(f) & (f > 0), f, 1.0)

    if anchor == "ewma" and log_price is None:
        if returns is None and "futures_index_price" in g:
            px = pd.to_numeric(g["futures_index_price"], errors="coerce").to_numpy(float)
            log_price = np.where(np.isfinite(r) & (px > 0), np.log(np.where(px > 0, px, 1.0)), np.nan)
        else:
            log_price = np.full(len(r), np.nan)
            level = 0.0
            for i in range(len(r)):
                if not contiguous[i]:
                    level = 0.0
                if np.isfinite(r[i]):
                    level += r[i]
                    log_price[i] = level

    sigma_c = ewma_sigma(r / f, contiguous) * f
    sig_prev = np.r_[np.nan, sigma_c[:-1]]
    sig_prev[~contiguous] = np.nan
    z = r / sig_prev
    session = ((times - pd.Timedelta(hours=8)).dt.floor("D") + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d").to_numpy()
    new_session = np.r_[True, session[1:] != session[:-1]]
    coverage = None
    if anchor == "session_vwap":
        D, h, coverage = _session_vwap_D(g, r, sigma_c, new_session, returns is not None, rng)
    else:
        a, age = ewma_anchor(log_price, ~contiguous, anchor_half_life)
        D = (log_price - a) / (sigma_c * math.sqrt(ewma_h_eff(anchor_half_life)))
        D[age < anchor_half_life] = np.nan
        h = np.full(len(D), ewma_h_eff(anchor_half_life))

    gap_ok = {k: (times - times.shift(k) == k * BAR).to_numpy() for k in range(1, N_LAGS + 1)}

    def lag(x, k):
        """x_{t-k}, NaN unless the grid is contiguous from t-k to t."""
        out = np.full(len(x), np.nan)
        out[k:] = x[:-k]
        ok = gap_ok[k] if k in gap_ok else (times - times.shift(k) == k * BAR).to_numpy()
        return np.where(ok, out, np.nan)

    D1 = lag(D, 1)
    h1 = lag(h, 1)
    z1, z2, z3 = lag(z, 1), lag(z, 2), lag(z, 3)
    cols = np.column_stack([np.ones(len(r)), D1, z1, z2, z3])
    valid = np.isfinite(z) & np.all(np.isfinite(cols), axis=1) & gap_ok[N_LAGS]
    col_names = list(ALL_COLUMNS)
    if extra is not None and len(extra.columns):
        if len(extra) != len(r):
            raise ValueError("extra must be aligned with the rows of g")
        ex = np.nan_to_num(extra.to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
        ex1 = np.column_stack([np.nan_to_num(lag(ex[:, j], 1), nan=0.0) for j in range(ex.shape[1])])
        cols = np.column_stack([cols, ex1])
        col_names += [str(c) for c in extra.columns]

    # VR(6) on clock-adjusted returns over the 48 bars up to t-1 (rule baseline)
    rt = r / f
    comp = trailing_complete(rt, contiguous, 48)
    vr = lag(variance_ratio(rt, comp, 6, 48), 1)

    pos = np.flatnonzero(valid)
    brk = np.r_[True, np.diff(pos) != 1] if len(pos) else np.array([], bool)
    st = np.flatnonzero(brk)
    en = np.r_[st[1:], len(pos)]
    ok = (en - st) >= MIN_SEGMENT
    rows = np.concatenate([np.arange(a, b) for a, b in zip(st[ok], en[ok])]) if ok.any() else np.array([], int)
    lengths = (en - st)[ok]
    starts = np.r_[0, np.cumsum(lengths)[:-1]].astype(int) if len(lengths) else np.array([], int)
    ends = np.cumsum(lengths).astype(int)
    src = pos[rows]

    frame = pd.DataFrame({
        "state_time": times.to_numpy()[src],
        "session_date": session[src],
        "y": z[src], "D": D1[src], "z1": z1[src], "z2": z2[src], "z3": z3[src], "h_prev": h1[src],
        "vr_q6_prev": vr[src], "sigma_c_prev": sig_prev[src], "r": r[src], "source_row": src,
    })
    B = Bm = None
    names = []
    if block is not None:
        names = list(block.columns)
        B = block.to_numpy(float)[src]
        Bm = (block_mask.to_numpy(bool) if block_mask is not None else np.isfinite(B))[src] & np.isfinite(B)
        B = np.where(Bm, B, 0.0)
    if coverage is not None:
        coverage["model_rows"] = float(len(src) / max(len(r), 1))
    return Design(asset, frame, cols[src], starts, ends, names, B, Bm, anchor, coverage, col_names)
