"""
Signed Jumps 02a: what does the price path look like after a jump, and is it tradeable?

Protocol: discussions/Signed_Jumps/003_claude_sj02a_preregistration.md (fixed before any real-data output).

Events
------
Jumps from the SJ 01 engine unchanged (spot 1m, c = 4, 5 bp floor). PRIMARY events are cluster starters:
a jump at bar t with no jump of either sign on the same asset in bars t-5..t-1. Event sign s = +1 / -1.

Signal (no refit)
-----------------
SJ 01 model (e) with the fitted DEV parameters from notebooks/Signed_Jumps/01/01_fits.csv. At bar t+1
(covariates through t, excitation including the jump at t) the predicted continuation share is
    pi_c = lambda^{same}_{t+1} / (lambda^{same}_{t+1} + lambda^{opp}_{t+1}).

Path (tradeable instrument: Binance USD-M perpetual, 1m closes)
--------------------------------------------------------------
T = end of jump bar t.  Immediate entry (PRIMARY): price at T.  Delayed entry (secondary): price at T + 1 min.
    CAR(k) = d * (log P_perp(T_entry + k) - log P_perp(T_entry)) * 1e4  bp,  k in HORIZONS minutes,
d = +s (continuation) or -s (reversal).  MFE / MAE over 15 minutes.

Rule selection (DEV only) and gates (TEST)
-----------------------------------------
Grid: horizon k x direction {continuation, reversal} x filter {all, top pi_c quintile, bottom pi_c quintile}
(quintile cut points from DEV). Rule = argmax over cells of the DEV 95% day-bootstrap LOWER bound of net bp
at the primary cost (the mean picks small noisy cells). TEST, the selected rule:
    NET_EDGE          : net mean 95% day-bootstrap lower bound > 0
    GROSS_EDGE_ONLY   : gross lower bound > 0, net not
    NO_EDGE           : otherwise
The full TEST CAR surface for every cell is reported, never used for the decision.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from . import signed_jumps as sj

VERSION = "SJ02a-0.1.0"
HORIZONS = (1, 2, 3, 5, 10, 15, 30, 60)


@dataclass(frozen=True)
class EventParams:
    horizons: tuple = HORIZONS
    cluster_gap: int = 5
    excursion_window: int = 15
    cost_primary_bp: float = 12.0       # Binance perp taker 5 bp + 1 bp slippage, per side (project standard)
    cost_grid_bp: tuple = (7.0, 12.0, 14.0)
    n_boot: int = 2000
    min_test_events: int = 200
    status: str = "PREREGISTERED_SJ003"


def registration(p: EventParams | None = None) -> dict:
    p = p or EventParams()
    d = asdict(p)
    d["version"] = VERSION
    d["selection"] = "argmax DEV net 95% lower bound over horizon x {continuation, reversal} x {all, top, bottom pi_c quintile}"
    d["gates"] = {"NET_EDGE": "TEST net lower 95% > 0", "GROSS_EDGE_ONLY": "TEST gross lower 95% > 0, net not",
                  "NO_EDGE": "otherwise"}
    return d


# ----------------------------------------------------------------------------- SJ 01 parameters

def fits_from_table(tab: pd.DataFrame, pan: sj.Panel, p: sj.Params, model: str = "e") -> dict:
    """Rebuild SJ 01 Fit objects from 01_fits.csv (no refit)."""
    out = {}
    names = sj.design_names(pan, model, p)
    for s in sj.SIGNS:
        r = tab[(tab.model == model) & (tab.sign == s)].iloc[0]
        b = np.array([float(r[n]) for n in names])
        eta = np.array([float(r[c]) for c in ("eta0", "eta1") if c in r and pd.notna(r[c])])
        hl = float(r["half_life_min"]) if pd.notna(r["half_life_min"]) else None
        out[(model, s)] = sj.Fit(model, s, b, eta, hl, float(r["ll_dev"]), int(r["n_dev"]), bool(r["converged"]))
    return out


def intensities(fits: dict, assets, pan: sj.Panel, model: str = "e") -> dict:
    X = sj.design(pan, model)
    hl = fits[(model, "up")].half_life
    E = sj.panel_excitation(assets, pan, hl)
    return {s: sj.intensity(fits[(model, s)], X, sj.excitation_design(E, model, s)) for s in sj.SIGNS}


# ----------------------------------------------------------------------------- events

def find_events(pan: sj.Panel, lam: dict, gap: int) -> pd.DataFrame:
    """One row per jump. pi_c is read at the next panel row of the same asset if it is exactly 1 minute later."""
    up, dn = pan.J["up"] > 0.5, pan.J["down"] > 0.5
    rows = []
    time = pd.DatetimeIndex(pan.time)
    if time.tz is None:
        time = time.tz_localize("UTC")
    n = len(pan.r)
    for i_a in range(pan.n_assets):
        idx = np.flatnonzero(pan.asset_idx == i_a)
        any_j = (up | dn)[idx]
        cs = np.concatenate([[0], np.cumsum(any_j)])
        for k in np.flatnonzero(any_j):
            i = idx[k]
            s = 1 if up[i] else -1
            prior = cs[k] - cs[max(0, k - gap)]
            nxt = idx[k + 1] if k + 1 < len(idx) else None
            pic = np.nan
            if nxt is not None and (time[nxt] - time[i]) == pd.Timedelta(minutes=1):
                same, opp = ("up", "down") if s > 0 else ("down", "up")
                pic = lam[same][nxt] / (lam[same][nxt] + lam[opp][nxt])
            rows.append({"asset": pan.names[i_a], "row": i, "time": time[i], "sign": s,
                         "starter": bool(prior == 0), "pi_c": pic, "dev": bool(pan.dev[i]),
                         "day": int(pan.day[i]), "jump_bp": float(pan.r[i] * 1e4), "absZ": float(abs(pan.Z[i]))})
    return pd.DataFrame(rows)


def attach_paths(ev: pd.DataFrame, perp: dict, p: EventParams) -> pd.DataFrame:
    """perp: asset -> log close Series on a minute grid indexed by bar END time.
    Adds g_imm_k / g_del_k (signed by the jump sign: continuation-positive, bp) and MFE/MAE (immediate)."""
    ev = ev.copy()
    H = max(p.horizons)
    W = p.excursion_window
    for a, grp in ev.groupby("asset"):
        s = perp[a]
        v = s.to_numpy()
        pos = s.index.get_indexer(pd.DatetimeIndex(grp.time))
        ok = pos >= 0
        sign = grp.sign.to_numpy()
        for mode, off in (("imm", 0), ("del", 1)):
            e0 = pos + off
            base = np.where(ok & (e0 < len(v)), v[np.clip(e0, 0, len(v) - 1)], np.nan)
            for k in p.horizons:
                j = e0 + k
                val = np.where(ok & (j < len(v)), v[np.clip(j, 0, len(v) - 1)], np.nan)
                ev.loc[grp.index, f"g_{mode}_{k}"] = sign * (val - base) * 1e4
        path = np.full((len(grp), W), np.nan)
        for j in range(1, W + 1):
            jj = pos + j
            path[:, j - 1] = np.where(ok & (jj < len(v)), v[np.clip(jj, 0, len(v) - 1)], np.nan)
        b0 = np.where(ok, v[np.clip(pos, 0, len(v) - 1)], np.nan)
        rel = sign[:, None] * (path - b0[:, None]) * 1e4
        with np.errstate(all="ignore"):
            ev.loc[grp.index, "mfe_15"] = np.nanmax(rel, axis=1)
            ev.loc[grp.index, "mae_15"] = np.nanmin(rel, axis=1)
    return ev


# ----------------------------------------------------------------------------- statistics

def boot_mean(x: np.ndarray, day: np.ndarray, n_boot: int, seed: int) -> dict:
    ok = np.isfinite(x)
    x, day = x[ok], day[ok]
    if len(x) == 0:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, n=0, n_days=0)
    ud, inv = np.unique(day, return_inverse=True)
    S = np.bincount(inv, weights=x)
    N = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    d = rng.integers(0, len(ud), size=(n_boot, len(ud)))
    bs = S[d].sum(1) / N[d].sum(1)
    return dict(mean=float(x.mean()), lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)),
                n=int(len(x)), n_days=int(len(ud)))


def quintile_cuts(ev: pd.DataFrame) -> np.ndarray:
    d = ev[ev.dev & ev.starter & ev.pi_c.notna()].pi_c
    return np.quantile(d, [0.2, 0.8]) if len(d) >= 10 else np.array([np.nan, np.nan])


def cell_mask(ev: pd.DataFrame, flt: str, cuts: np.ndarray) -> np.ndarray:
    if flt == "all":
        return np.ones(len(ev), bool)
    if flt == "top":
        return (ev.pi_c >= cuts[1]).to_numpy()
    return (ev.pi_c <= cuts[0]).to_numpy()


def surface(ev: pd.DataFrame, p: EventParams, cuts, period: str, mode: str = "imm", seed: int = 3) -> pd.DataFrame:
    """Mean signed return (gross, bp) for every horizon x direction x filter, primary events."""
    base = ev[ev.starter & (ev.dev if period == "dev" else ~ev.dev)]
    rows = []
    for flt in ("all", "top", "bottom"):
        m = cell_mask(base, flt, cuts)
        sub = base[m]
        for k in p.horizons:
            g = sub[f"g_{mode}_{k}"].to_numpy()
            for direction, sgn in (("continuation", 1), ("reversal", -1)):
                b = boot_mean(sgn * g, sub.day.to_numpy(), p.n_boot, seed + k)
                rows.append({"period": period, "entry": mode, "filter": flt, "horizon_min": k,
                             "direction": direction, "gross_bp": b["mean"], "gross_lo": b["lo"],
                             "gross_hi": b["hi"], "n": b["n"], "n_days": b["n_days"],
                             "net_bp_primary": b["mean"] - p.cost_primary_bp})
    return pd.DataFrame(rows)


def select_rule(dev_surface: pd.DataFrame, p: EventParams) -> dict:
    """Rank DEV cells by the 95% day-bootstrap LOWER bound of net bp (gross_lo - primary cost), not by the
    mean: selecting on the mean picks small noisy cells (winner's curse; found on synthetic data)."""
    d = dev_surface[dev_surface.n >= 30].assign(net_lo=lambda x: x.gross_lo - p.cost_primary_bp)
    d = d.sort_values("net_lo", ascending=False)
    r = d.iloc[0]
    return {"filter": r["filter"], "horizon_min": int(r.horizon_min), "direction": r.direction,
            "dev_gross_bp": float(r.gross_bp), "dev_gross_lo": float(r.gross_lo), "dev_n": int(r.n)}


def evaluate(ev: pd.DataFrame, rule: dict, cuts, p: EventParams, mode: str = "imm", seed: int = 77) -> dict:
    base = ev[ev.starter & ~ev.dev]
    sub = base[cell_mask(base, rule["filter"], cuts)]
    sgn = 1 if rule["direction"] == "continuation" else -1
    g = sgn * sub[f"g_{mode}_{rule['horizon_min']}"].to_numpy()
    day = sub.day.to_numpy()
    gross = boot_mean(g, day, p.n_boot, seed)
    out = {"entry": mode, "rule": rule, "test_n": gross["n"], "test_days": gross["n_days"],
           "gross_bp": gross["mean"], "gross_lo": gross["lo"], "gross_hi": gross["hi"]}
    for c in p.cost_grid_bp:
        b = boot_mean(g - c, day, p.n_boot, seed)
        out[f"net_{c:g}bp"] = b["mean"]
        out[f"net_{c:g}bp_lo"] = b["lo"]
    net_lo = out[f"net_{p.cost_primary_bp:g}bp_lo"]
    if gross["n"] < p.min_test_events:
        out["verdict"] = "INSUFFICIENT_EVENTS"
    elif net_lo > 0:
        out["verdict"] = "NET_EDGE"
    elif gross["lo"] > 0:
        out["verdict"] = "GROSS_EDGE_ONLY"
    else:
        out["verdict"] = "NO_EDGE"
    return out


def car_curve(ev: pd.DataFrame, p: EventParams, cuts, mode: str = "imm") -> pd.DataFrame:
    """TEST mean continuation-signed path by parent sign and pi_c quintile (descriptive)."""
    base = ev[ev.starter & ~ev.dev & ev.pi_c.notna()].copy()
    q = pd.qcut(base.pi_c, 5, labels=False, duplicates="drop") if len(base) >= 25 else 0
    base["q"] = q
    rows = []
    for (sgn, qq), g in base.groupby(["sign", "q"]):
        for k in p.horizons:
            x = g[f"g_{mode}_{k}"]
            rows.append({"sign": "up" if sgn > 0 else "down", "pi_c_quintile": int(qq), "pi_c_mean": g.pi_c.mean(),
                         "horizon_min": k, "mean_bp": x.mean(), "median_bp": x.median(),
                         "share_positive": (x > 0).mean(), "n": int(x.notna().sum())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- synthetic

def plant_drift(series: dict, assets, drift_bp: float, minutes: int) -> dict:
    """Synthetic perp = spot + small noise; after each detected jump add drift_bp per minute in the jump
    direction for `minutes` minutes (drift_bp = 0: no post-jump drift)."""
    out = {}
    rng = np.random.default_rng(0)
    for a in assets:
        s = series[a.name]
        r = np.nan_to_num(np.diff(s.to_numpy(), prepend=s.iloc[0]))
        add = np.zeros(len(r))
        sgn = a.cols["Jup"] - a.cols["Jdn"]
        for i in np.flatnonzero(sgn != 0):
            add[i + 1: i + 1 + minutes] += sgn[i] * drift_bp / 1e4
        lp = s.iloc[0] + np.cumsum(r + add)
        out[a.name] = pd.Series(lp + rng.normal(0, 2e-5, len(lp)), index=s.index)
    return out
