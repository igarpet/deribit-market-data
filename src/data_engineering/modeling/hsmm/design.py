"""
Design matrices for the HSMM from per-asset state rows (feature contract v0).

For one asset, on the exact 5-minute grid:

    r~_t      = r_t / f_t                         (clock-adjusted return; f from clock_baseline)
    sigma~_t  = EWMA_{hl 12}(r~^2)^(1/2)          (causal, restarted on gaps)
    sigma^c_t = f_t * sigma~_t
    z_t       = r_t / sigma^c_{t-1}               (standardized return)
    G_t       = log I_t - log A_t                 (index EWMA anchor, half-life H, restarted on gaps only:
                                                   the underlying chain is continuous across 08:00, contract v0)
    D_t       = G_t / (sigma^c_t * sqrt(h_eff(H)))

Model row t:  y_t = z_t;  state-specific [1, D_{t-1}];  shared [z_{t-1}, z_{t-2}, z_{t-3}]
(all regressors known at t-1). Rows need every term finite and the grid
contiguous from t-3 to t; contiguous runs of valid rows form segments.

The same builder is used for real returns and for simulated (null) returns, so
the null passes through exactly the same pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_engineering.features.regime_inputs import ewma_anchor, ewma_h_eff, trailing_complete, variance_ratio
from data_engineering.modeling.hsmm.engine import HSMMData


SIGMA_HALF_LIFE = 12
SIGMA_SEED_BARS = 12
ANCHOR_HALF_LIFE = 12
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

    def data(self, xs=STRUCTURAL[0], xc=STRUCTURAL[1], rows: np.ndarray | None = None,
             with_block: bool = False) -> HSMMData:
        """HSMMData for a column choice, optionally restricted to a boolean row mask
        (segments are cut where the mask is False)."""
        idx_s = [ALL_COLUMNS.index(c) for c in xs]
        idx_c = [ALL_COLUMNS.index(c) for c in xc]
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
                        list(xs), list(xc), O, M, names_o)

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


def build_design(g: pd.DataFrame, f_clock: np.ndarray, returns: np.ndarray | None = None,
                 log_price: np.ndarray | None = None, block: pd.DataFrame | None = None,
                 block_mask: pd.DataFrame | None = None, anchor_half_life: int = ANCHOR_HALF_LIFE) -> Design:
    """
    g: one asset, sorted by state_time, with state_time, session_date and (unless
    `returns` is given) r. log_price defaults to log futures_index_price where
    returns are eligible; for simulated returns pass returns and leave
    log_price None (the path is rebuilt as a cumulative sum within contiguous runs).
    """
    asset = str(g["underlying"].iloc[0]) if "underlying" in g else "asset"
    times = pd.to_datetime(g["state_time"], utc=True).reset_index(drop=True)
    contiguous = np.r_[False, (times.diff().iloc[1:] == BAR).to_numpy()]
    r = np.asarray(g["r"] if returns is None else returns, float)
    f = np.asarray(f_clock, float)
    f = np.where(np.isfinite(f) & (f > 0), f, 1.0)

    if log_price is None:
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
    restart = ~contiguous
    anchor, age = ewma_anchor(log_price, restart, anchor_half_life)
    D = (log_price - anchor) / (sigma_c * math.sqrt(ewma_h_eff(anchor_half_life)))
    D[age < anchor_half_life] = np.nan

    gap_ok = {k: (times - times.shift(k) == k * BAR).to_numpy() for k in range(1, N_LAGS + 1)}

    def lag(x, k):
        """x_{t-k}, NaN unless the grid is contiguous from t-k to t."""
        out = np.full(len(x), np.nan)
        out[k:] = x[:-k]
        ok = gap_ok[k] if k in gap_ok else (times - times.shift(k) == k * BAR).to_numpy()
        return np.where(ok, out, np.nan)

    D1 = lag(D, 1)
    z1, z2, z3 = lag(z, 1), lag(z, 2), lag(z, 3)
    cols = np.column_stack([np.ones(len(r)), D1, z1, z2, z3])
    valid = np.isfinite(z) & np.all(np.isfinite(cols), axis=1) & gap_ok[N_LAGS]

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
        "session_date": ((times - pd.Timedelta(hours=8)).dt.floor("D") + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d").to_numpy()[src],
        "y": z[src], "D": D1[src], "z1": z1[src], "z2": z2[src], "z3": z3[src],
        "vr_q6_prev": vr[src], "sigma_c_prev": sig_prev[src], "r": r[src], "source_row": src,
    })
    B = Bm = None
    names = []
    if block is not None:
        names = list(block.columns)
        B = block.to_numpy(float)[src]
        Bm = (block_mask.to_numpy(bool) if block_mask is not None else np.isfinite(B))[src] & np.isfinite(B)
        B = np.where(Bm, B, 0.0)
    return Design(asset, frame, cols[src], starts, ends, names, B, Bm)
