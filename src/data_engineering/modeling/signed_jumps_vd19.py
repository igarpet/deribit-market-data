"""
Signed Jumps 05: does the SJ 01 jump hazard add to VD 19's model-versus-implied information?

Protocol: discussions/Signed_Jumps/011_claude_sj05_preregistration.md (fixed before any real-data output).
Cross-branch use approved by Ivan (2026-10-16): VD 19 v0.1.1 rows are READ ONLY; VD 19 itself is unchanged and
its prospective re-score keeps its own frozen rule.

For each VD 19 row (session, asset, entry hour, horizon h minutes to the 08:00 UTC expiry):
    entry time      t_e = session 08:00 UTC - h minutes
    origin          the SJ 01 panel row ending at t_e + 1 min (its covariates use bars through t_e)
    Lambda(t_e, T)  expected number of SJ 01 jumps (both signs) in the h minutes to expiry, model (e) with the
                    fitted DEV parameters (01_fits.csv), covariates frozen at the origin except the clock terms,
                    which follow the true future slots; the excitation decays deterministically:
        Lambda = sum_s sum_{j<h} [ exp(X_o b_s - C_o bc_s + C_{o+j} bc_s) + (eta_s . E_o) q^j ]
    xJ = log Lambda
Primary (entry 04:00, outcome y_rv_index = log(Deribit-index path RV / sigma_imp^2), VD 19 session bootstrap):
    y = a + beta x + gamma xJ
    JUMP_HAZARD_ADDS_INFORMATION iff gamma 95% lower bound > 0; JUMP_HAZARD_ANTI iff upper < 0; else NO_EVIDENCE.
Reported: entries 06:00 / 12:00; spot-RV outcome; xJ alone; corr(x, xJ); VD 19-selected straddle R_net split
at the median xJ of the 04:00 rows.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from . import signed_jumps as sj

VERSION = "SJ05-0.1.0"
EXPIRY_HOUR_UTC = 8


@dataclass(frozen=True)
class VdParams:
    primary_entry_hour: int = 4
    primary_outcome: str = "y_rv_index"
    secondary_outcomes: tuple = ("y_rv", "y")
    vd19_selection_log_ratio: float = float(np.log(1.25))
    origin_tolerance_min: int = 5
    status: str = "PREREGISTERED_SJ011"


def registration(vp: VdParams | None = None) -> dict:
    vp = vp or VdParams()
    d = asdict(vp)
    d["version"] = VERSION
    d["gate"] = ("entry 04:00, y_rv_index ~ x + xJ, VD19 session bootstrap: gamma lo > 0 => JUMP_HAZARD_ADDS_INFORMATION; "
                 "gamma hi < 0 => JUMP_HAZARD_ANTI; else NO_EVIDENCE")
    return d


def entry_times(rows: pd.DataFrame) -> pd.Series:
    sess = pd.to_datetime(rows["session"]).dt.tz_localize("UTC") if pd.to_datetime(rows["session"]).dt.tz is None \
        else pd.to_datetime(rows["session"])
    return sess + pd.Timedelta(hours=EXPIRY_HOUR_UTC) - pd.to_timedelta(rows["h"].astype(int), unit="min")


def origin_rows(pan: sj.Panel, rows: pd.DataFrame, vp: VdParams) -> np.ndarray:
    """Panel row index of the bar ending at entry + 1 min (or the first within tolerance); -1 if none."""
    time = pd.DatetimeIndex(pan.time)
    if time.tz is None:
        time = time.tz_localize("UTC")
    out = np.full(len(rows), -1)
    te = entry_times(rows)
    for i_a, name in enumerate(pan.names):
        sel = np.flatnonzero((rows["underlying"] == name).to_numpy())
        if not len(sel):
            continue
        idx = np.flatnonzero(pan.asset_idx == i_a)
        t_a = time[idx]
        want = te.iloc[sel] + pd.Timedelta(minutes=1)
        pos = t_a.searchsorted(want)
        for k, (j, w) in enumerate(zip(pos, want)):
            if j < len(t_a) and (t_a[j] - w) <= pd.Timedelta(minutes=vp.origin_tolerance_min):
                out[sel[k]] = idx[j]
    return out


def hazard_to_expiry(fits: dict, assets, pan: sj.Panel, p: sj.Params, orig: np.ndarray, h: np.ndarray,
                     model: str = "e") -> tuple[np.ndarray, np.ndarray]:
    """(Lambda_total, Lambda_baseline_only) for each origin row and horizon h (minutes)."""
    X = sj.design(pan, model)
    nA, nC = pan.n_assets, pan.clock.shape[1]
    hl = fits[(model, "up")].half_life
    E = sj.panel_excitation(assets, pan, hl) if hl is not None else None
    q = np.exp(-np.log(2) / hl) if hl is not None else 0.0
    time = pd.DatetimeIndex(pan.time)
    if time.tz is None:
        time = time.tz_localize("UTC")
    tot = np.full(len(orig), np.nan)
    base_only = np.full(len(orig), np.nan)
    for k, (o, hh) in enumerate(zip(orig, h)):
        if o < 0:
            continue
        hh = int(hh)
        fut = time[o] + pd.to_timedelta(np.arange(hh), unit="min")
        Cf = sj.clock_design(fut, p.fourier_k)
        lb, le = 0.0, 0.0
        for s in sj.SIGNS:
            f = fits[(model, s)]
            bc = f.b[nA:nA + nC]
            lin0 = float(X[o] @ f.b - pan.clock[o] @ bc)
            lb += float(np.exp(np.clip(lin0 + Cf @ bc, -50, 5)).sum())
            if E is not None and len(f.eta):
                exc = f.eta[0] * E["up"][o] + f.eta[1] * E["down"][o]
                le += float(exc * (1 - q ** hh) / (1 - q))
        base_only[k] = lb
        tot[k] = lb + le
    return tot, base_only


def attach(rows: pd.DataFrame, lam: np.ndarray, lam_base: np.ndarray) -> pd.DataFrame:
    d = rows.copy()
    d["Lambda"] = lam
    d["Lambda_base"] = lam_base
    with np.errstate(divide="ignore", invalid="ignore"):
        d["xJ"] = np.log(lam)
        d["xJ_per_hour"] = np.log(lam / (d["h"] / 60.0))
    return d


def information(d: pd.DataFrame, mv, vp: VdParams) -> pd.DataFrame:
    """Encompassing regressions by entry hour and outcome (VD 19 session bootstrap)."""
    P = mv.Params()
    out = []
    for hr, g in d.groupby("entry_hour"):
        for ycol in (vp.primary_outcome, *vp.secondary_outcomes):
            if ycol not in g or not np.isfinite(g[ycol]).any():
                continue
            for cols, tag in ((["x", "xJ"], "x+xJ"), (["xJ"], "xJ_only"), (["x"], "x_only")):
                r = mv.session_bootstrap(g, cols, ycol, P, seed_offset=hr * 100 + len(ycol) * 7 + len(tag))
                out.append({"entry_hour": hr, "outcome": ycol, "model": tag,
                            "corr_x_xJ": float(np.corrcoef(g["x"], g["xJ"])[0, 1]) if len(g) > 2 else np.nan, **r})
    return pd.DataFrame(out)


def economics(d: pd.DataFrame, mv, vp: VdParams) -> pd.DataFrame:
    """VD 19-selected straddles (x >= log 1.25), R_net by xJ above / below the median xJ of the same entry hour."""
    P = mv.Params()
    out = []
    for hr, g in d.groupby("entry_hour"):
        med = float(np.nanmedian(g["xJ"]))
        for sel_name, m in (("vd19_selected", g["x"] >= vp.vd19_selection_log_ratio), ("all", np.ones(len(g), bool))):
            for half, mm in (("xJ_high", g["xJ"] >= med), ("xJ_low", g["xJ"] < med)):
                s = g[m & mm]
                r = mv.session_mean_boot(s, "R_net", P) if len(s) else {"n": 0}
                out.append({"entry_hour": hr, "set": sel_name, "half": half, "xJ_median": med, **r})
    return pd.DataFrame(out)


def decide(info: pd.DataFrame, vp: VdParams) -> dict:
    r = info[(info.entry_hour == vp.primary_entry_hour) & (info.outcome == vp.primary_outcome) & (info.model == "x+xJ")]
    if not len(r) or "xJ_lo" not in r or not np.isfinite(r.iloc[0].get("xJ_lo", np.nan)):
        return {"version": VERSION, "verdict": "INSUFFICIENT_SESSIONS"}
    r = r.iloc[0]
    v = ("JUMP_HAZARD_ADDS_INFORMATION" if r.xJ_lo > 0 else "JUMP_HAZARD_ANTI" if r.xJ_hi < 0 else "NO_EVIDENCE")
    return {"version": VERSION, "verdict": v, "gamma": r.xJ, "gamma_ci": [r.xJ_lo, r.xJ_hi], "beta": r.x,
            "beta_ci": [r.x_lo, r.x_hi], "n": int(r.n), "sessions": int(r.sessions), "corr_x_xJ": r.corr_x_xJ,
            "status": vp.status, "prospective_confirmation": False}


def synthetic_rows(n_sessions=25, assets=("BTC", "ETH", "SOL", "XRP"), gamma=0.0, seed=0) -> pd.DataFrame:
    """VD 19-shaped rows with a known dependence of y on x and xJ (for tests)."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in pd.date_range("2026-09-21", periods=n_sessions, freq="D"):
        shock = rng.normal(0, 0.3)
        for a in assets:
            x = rng.normal(0, 0.3)
            xJ = 0.5 * x + rng.normal(0, 0.3)
            y = 0.2 + 1.0 * x + gamma * xJ + shock + rng.normal(0, 0.4)
            rows.append({"session": s.date().isoformat(), "underlying": a, "entry_hour": 4, "h": 240, "x": x, "xJ": xJ,
                         "y_rv_index": y, "y_rv": y + rng.normal(0, 0.05), "y": y + rng.normal(0, 1.0),
                         "R_net": 0.3 * (y - 0.4) + rng.normal(0, 0.3)})
    return pd.DataFrame(rows)
