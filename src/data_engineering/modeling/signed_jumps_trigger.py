"""
Signed Jumps 04b: a REAL-TIME seconds trigger. Does price keep going after a burst is detected, for a system
that reacts within seconds?

Protocol: discussions/Signed_Jumps/009_claude_sj04b_preregistration.md (fixed before any real-data output).

Replaces SJ 04 P3, which was look-ahead biased (onset defined from the full jump minute; SJ 008 §4).

Trigger (only information available at second tau)
    sigma_min(T): SJ 01 engine 1-minute return s.d. for the minute ending at T, forecast with data through T-1 min:
                  s(slot_T) * sqrt(sig2c_T)   (deseasonalisation x continuous-variance forecast).
    For a second tau in minute (T-60s, T]:  R60(tau) = log P(tau) - log P(tau - 60 s),  z(tau) = R60 / sigma_min(T).
    Fire at the first tau with |z| >= c_trig and |R60| >= 5 bp, then disarm for `cooldown_s` seconds.
    Direction d = sign(R60(tau)).
Outcome
    G(delta, H) = d * (log P(tau + delta + H) - log P(tau + delta)) * 1e4  bp  (spot 1s closes).
Gate (primary c_trig = 4, delta = 2 s, H = 60 s): gross day-block bootstrap q2.5 > 0 -> CONTINUATION.
Economic flags: gross >= 7 bp (cheapest perp taker) / >= 12 bp (project standard). Grids reported only.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

VERSION = "SJ04b-0.1.0"


@dataclass(frozen=True)
class TrigParams:
    c_trig: float = 4.0
    c_grid: tuple = (3.0, 4.0, 5.0)
    floor_bp: float = 5.0
    window_s: int = 60
    cooldown_s: int = 300
    delta_s: int = 2
    h_s: int = 60
    grid_delta: tuple = (1, 2, 5, 10, 30)
    grid_h: tuple = (10, 30, 60, 300)
    costs_bp: tuple = (7.0, 12.0)
    n_boot: int = 2000
    status: str = "PREREGISTERED_SJ009"


def registration(tp: TrigParams | None = None) -> dict:
    tp = tp or TrigParams()
    d = asdict(tp)
    d["version"] = VERSION
    d["gate"] = "c_trig=4, delta=2 s, H=60 s: gross day-bootstrap q2.5 > 0 => CONTINUATION; else NO_CONTINUATION"
    return d


def sigma_minute(asset) -> pd.Series:
    """Per-minute return s.d. forecast (bar-end index), from the SJ 01 engine already built on `asset`."""
    C = asset.cols["clock"]
    s = np.exp(np.column_stack([np.ones(len(C)), C]) @ asset.cols["seas_coef"])
    return pd.Series(s * np.sqrt(asset.cols["sig2c"]), index=asset.time)


def triggers(sec: pd.DataFrame, sig_min: pd.Series, c: float, tp: TrigParams) -> pd.DataFrame:
    """Real-time trigger times on a 1s grid (bar-end index, column logp)."""
    lp = sec["logp"].to_numpy()
    idx = sec.index
    w = tp.window_s
    R = np.full(len(lp), np.nan)
    R[w:] = lp[w:] - lp[:-w]
    T = idx.ceil("min")                                # minute (T-60s, T] containing each second
    sig = sig_min.reindex(T).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        z = R / sig
    cand = np.flatnonzero((np.abs(z) >= c) & (np.abs(R) >= tp.floor_bp / 1e4))
    fired, last = [], -10 ** 12
    for i in cand:
        if i - last >= tp.cooldown_s:
            fired.append(i)
            last = i
    fired = np.asarray(fired, int)
    return pd.DataFrame({"pos": fired, "tau": idx[fired], "dir": np.sign(R[fired]).astype(int),
                         "z": z[fired], "R60_bp": R[fired] * 1e4})


def forward(tr: pd.DataFrame, sec: pd.DataFrame, tp: TrigParams) -> pd.DataFrame:
    lp = sec["logp"].to_numpy()
    out = tr.copy()
    pos, d = tr["pos"].to_numpy(), tr["dir"].to_numpy()
    for dl in tp.grid_delta:
        for h in tp.grid_h:
            a, b = pos + dl, pos + dl + h
            ok = b < len(lp)
            v = np.full(len(tr), np.nan)
            v[ok] = d[ok] * (lp[b[ok]] - lp[a[ok]]) * 1e4
            out[f"g_{dl}_{h}"] = v
    return out


def boot_mean(x, day, n_boot, seed):
    ok = np.isfinite(x)
    x, day = x[ok], day[ok]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    ud, inv = np.unique(day, return_inverse=True)
    S, N = np.bincount(inv, weights=x), np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    dd = rng.integers(0, len(ud), size=(n_boot, len(ud)))
    bs = S[dd].sum(1) / N[dd].sum(1)
    return float(x.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), int(len(x))


def grid_table(ev: pd.DataFrame, tp: TrigParams, c: float) -> pd.DataFrame:
    day = pd.DatetimeIndex(ev["tau"]).normalize().asi8
    rows = []
    for dl in tp.grid_delta:
        for h in tp.grid_h:
            x = ev[f"g_{dl}_{h}"].to_numpy()
            m, lo, hi, n = boot_mean(x, day, tp.n_boot, 17 + dl + h)
            rows.append({"c_trig": c, "delta_s": dl, "h_s": h, "gross_bp": m, "lo": lo, "hi": hi, "n": n,
                         "share_positive": float(np.nanmean(x > 0)) if n else np.nan})
    return pd.DataFrame(rows)


def decide(tab: pd.DataFrame, tp: TrigParams) -> dict:
    r = tab[(tab.c_trig == tp.c_trig) & (tab.delta_s == tp.delta_s) & (tab.h_s == tp.h_s)].iloc[0]
    out = {"version": VERSION, "gross_bp": r.gross_bp, "q025": r.lo, "q975": r.hi, "n": int(r.n),
           "verdict": "CONTINUATION" if r.lo > 0 else "NO_CONTINUATION"}
    for c in tp.costs_bp:
        out[f"gross_ge_{c:g}bp"] = bool(r.gross_bp >= c)
        out[f"net_{c:g}bp"] = r.gross_bp - c
    return out
