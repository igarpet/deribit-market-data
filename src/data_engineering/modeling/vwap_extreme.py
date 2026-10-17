"""VWAP Events 04: fading EXTREME displacement from VWAP (any time, not only after a cross).

Protocol: discussions/VWAP_Events/006_claude_ve04_preregistration.md (fixed before any real-data output).

Event: bar t (>= 30 min after the anchor) whose close is displaced from the VWAP through bar t by at least a
threshold, then a 30-minute cooldown per (anchor spec, threshold). Thresholds:
    sigma-scaled   |log C_t - log VWAP_t| >= k * sigma_{t-1} * sqrt(15),  k in {3, 4, 5}
    percent        |log C_t - log VWAP_t| >= x,                            x in {0.5%, 0.75%, 1.0%}
Trade: fade toward VWAP, entry at the close of bar t+1, exits as VE03 (vwap_fade.simulate): target T bp or a
VWAP touch, stop S bp, or time; same-bar ties -> stop. Grid: T in {20, 40, 80, vwap}, S in {20, 40, 80},
max hold {60, 240} min.
Selection on DEV (entry before 2026-04-01): the REAL cell (threshold x T x S x hold) with the highest
day-bootstrap 95% lower bound of mean gross bp, n_dev >= 50.
Gates on TEST (entry from 2026-04-01), the selected cell only:
    G1 VWAP-specific  hour-matched real - placebo mean gross > 0, q2.5 > 0
    G2 tradeable      real mean gross q2.5 > 7 bp
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace

import numpy as np
import pandas as pd

from . import vwap_events as ve
from . import vwap_fade as vf
from . import vwap_touch_stats as vt

VERSION = "VE04-0.1.0"
TEST_START = pd.Timestamp("2026-04-01", tz="UTC")


@dataclass(frozen=True)
class XParams:
    k_sigma: tuple = (3.0, 4.0, 5.0)
    pct: tuple = (0.005, 0.0075, 0.010)
    stretch_m: int = 15
    warmup_min: int = 30
    cooldown: int = 30
    targets: tuple = (20, 40, 80, "vwap")
    stops: tuple = (20, 40, 80)
    holds: tuple = (60, 240)
    min_dev_n: int = 50
    tradeable_bp: float = 7.0
    costs_bp: tuple = (4.0, 7.0, 12.0)
    n_boot: int = 2000
    status: str = "PREREGISTERED_VE006"


def registration(xp: XParams | None = None) -> dict:
    xp = xp or XParams()
    d = asdict(xp)
    d["version"] = VERSION
    d["test_start"] = str(TEST_START)
    d["selection"] = "DEV real cell with max day-bootstrap q2.5 of mean gross bp (n_dev >= 50)"
    d["gates"] = {"G1": "TEST hour-matched real - placebo > 0, q2.5 > 0", "G2": "TEST real gross q2.5 > 7 bp"}
    return d


def thresholds(xp: XParams) -> list[tuple[str, str, float]]:
    return [("sigma", f"k{k:g}", k) for k in xp.k_sigma] + [("pct", f"p{x * 100:g}", x) for x in xp.pct]


def extreme_events(fr: pd.DataFrame, st: pd.DataFrame, kind: str, val: float, xp: XParams) -> pd.DataFrame:
    aid = st["aid"].to_numpy()
    v_now = st["lev"].shift(-1).to_numpy()
    same_next = np.r_[aid[1:] == aid[:-1], False]
    v_now = np.where(same_next, v_now, np.nan)
    logc = fr["logc"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = logc - np.log(v_now)
    dist = np.abs(sd)
    if kind == "sigma":
        sig_prev = np.r_[np.nan, fr["sigma"].to_numpy()[:-1]]
        thr = val * sig_prev * np.sqrt(xp.stretch_m)
    else:
        thr = np.full(len(dist), val)
    since = st["since"].to_numpy()
    cand = np.flatnonzero((dist >= thr) & (since >= xp.warmup_min) & np.isfinite(dist))
    keep, last = [], -10 ** 12
    for i in cand:
        if i - last >= xp.cooldown:
            keep.append(i)
            last = i
    keep = np.asarray(keep, int)
    return pd.DataFrame({"i": keep, "side": np.sign(sd[keep]).astype(int), "dist_bp": dist[keep] * 1e4})


def run_asset(df: pd.DataFrame, asset: str, xp: XParams, specs=vt.SPECS) -> pd.DataFrame:
    fr = ve.prepare(df, ve.Params())
    rows = []
    for sp in specs:
        st = vt.bar_states(fr, sp.period_min, sp.phase_min)
        for kind, label, val in thresholds(xp):
            ev = extreme_events(fr, st, kind, val, xp)
            ev = ev[ev["i"] + 1 < len(fr)]
            if not len(ev):
                continue
            ts = fr.index[ev["i"].to_numpy() + 1]
            test = np.asarray(ts >= TEST_START)
            day = np.asarray((ts - pd.Timestamp("1970-01-01", tz="UTC")).days, dtype=np.int64)   # unit-safe day id
            hour = ts.hour.to_numpy()
            d = -ev["side"].to_numpy()
            for H in xp.holds:
                fp = replace(vf.FadeParams(), max_hold=H)
                for T in xp.targets:
                    for S in xp.stops:
                        pnl = vf.simulate(fr, st, ev, d, T, S, fp)
                        rows.append(pd.DataFrame({"asset": asset, "anchor": sp.name, "kind": sp.kind,
                                                  "phase_min": sp.phase_min, "thr": label, "target": str(T),
                                                  "stop": np.int16(S), "hold": np.int16(H), "day": day.astype(np.int32),
                                                  "hour": hour.astype(np.int8), "test": test,
                                                  "dist_bp": ev["dist_bp"].to_numpy().astype("float32"),
                                                  "pnl": pnl.astype("float32")}))
    R = pd.concat(rows, ignore_index=True)
    for c in ("asset", "anchor", "kind", "thr", "target"):
        R[c] = R[c].astype("category")
    return R


CELL = ["thr", "target", "stop", "hold"]


def dev_table(R: pd.DataFrame, xp: XParams) -> pd.DataFrame:
    out = []
    for key, g in R[(R.kind == "real") & ~R.test].groupby(CELL, observed=True):
        m, lo, hi, n = vf.boot(g.pnl.to_numpy(float), g.day.to_numpy(), xp.n_boot, 3)
        out.append(dict(zip(CELL, key), dev_gross=m, dev_lo=lo, dev_hi=hi, dev_n=n,
                        dev_win=float((g.pnl > 0).mean())))
    return pd.DataFrame(out)


def select(dev: pd.DataFrame, xp: XParams) -> dict:
    d = dev[dev.dev_n >= xp.min_dev_n].sort_values("dev_lo", ascending=False)
    r = d.iloc[0]
    return {c: (r[c] if c != "stop" and c != "hold" else int(r[c])) for c in CELL} | {
        "dev_gross": float(r.dev_gross), "dev_lo": float(r.dev_lo), "dev_n": int(r.dev_n)}


def evaluate(R: pd.DataFrame, cell: dict, xp: XParams) -> dict:
    m = (R.thr == cell["thr"]) & (R.target == cell["target"]) & (R.stop == cell["stop"]) & (R.hold == cell["hold"]) & R.test
    real, plc = R[m & (R.kind == "real")], R[m & (R.kind == "placebo")]
    g, lo, hi, n = vf.boot(real.pnl.to_numpy(float), real.day.to_numpy(), xp.n_boot, 21)
    a = real.assign(pnl=real.pnl.astype(float)); b = plc.assign(pnl=plc.pnl.astype(float))
    dm, dlo, dhi = vf.boot_diff_hour(a, b, xp.n_boot, 23)
    x = a.pnl.to_numpy(); x = x[np.isfinite(x)]
    g1 = "PASS" if dm > 0 and dlo > 0 else "FAIL"
    g2 = "PASS" if lo > xp.tradeable_bp else "FAIL"
    res = {"version": VERSION, "status": xp.status, "cell": cell, "test_n": n, "test_gross": g, "test_lo": lo,
           "test_hi": hi, "placebo_gross": float(np.nanmean(b.pnl)), "real_minus_placebo_hour_matched": dm,
           "diff_lo": dlo, "diff_hi": dhi, "test_win_rate": float((x > 0).mean()) if len(x) else np.nan,
           "G1_vwap_specific": g1, "G2_tradeable": g2}
    for c in xp.costs_bp:
        res[f"test_net_{c:g}bp"] = g - c
    res["verdict"] = ("VWAP_EXTREME_EDGE" if g1 == "PASS" and g2 == "PASS" else
                      "VWAP_SPECIFIC_BUT_UNTRADEABLE" if g1 == "PASS" else
                      "GENERIC_EDGE_NOT_VWAP" if g2 == "PASS" else "NO_EDGE")
    return res


def full_table(R: pd.DataFrame, xp: XParams) -> pd.DataFrame:
    """All cells, real and placebo, DEV and TEST (descriptive)."""
    out = []
    for key, g in R.groupby(["kind", "test", *CELL], observed=True):
        x = g.pnl.to_numpy(float)
        m, lo, hi, n = vf.boot(x, g.day.to_numpy(), max(200, xp.n_boot // 5), 5)
        ok = np.isfinite(x)
        out.append(dict(zip(["kind", "test", *CELL], key), gross=m, lo=lo, hi=hi, n=n,
                        win=float((x[ok] > 0).mean()) if ok.any() else np.nan,
                        mean_dist_bp=float(g.dist_bp.mean())))
    return pd.DataFrame(out)
