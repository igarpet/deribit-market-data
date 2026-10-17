"""VWAP Events 03: does a VWAP fade scalp have gross edge, and is it specific to the real VWAP?

Protocol: discussions/VWAP_Events/004_claude_ve03_preregistration.md (fixed before any real-data output).

Data/anchors/VWAP: as VWAP Events 01/02 (vwap_events.py, vwap_touch_stats.py), Binance USD-M perp 1m.

Event (stretch fade)
    After each cross (VE02 definition) price sits on side s. The event is the FIRST bar t of that segment whose
    close is stretched:  |log C_t - log VWAP_t| >= k * sigma_{t-1} * sqrt(15)   (sigma: causal EWMA of 1m
    log returns, VE01), with t >= 30 min after the anchor. One event per crossing segment.
    Position d = -s (bet on reversion toward VWAP).  Variant "follow": d = +s.
Entry: close of bar t+1 (one-minute latency).
Exit (first of):
    target  T bp in favour (T in {5, 10, 20, 40}) or "vwap": a touch of the VWAP level known at the bar's open,
            exiting at that level;
    stop    S bp against (S in {5, 10, 20, 40});
    time    close of bar entry + 60.
    If target and stop are both inside one bar, the stop is assumed (conservative). Exits at exact levels
    (no gap slippage).
Primary (fixed): k = 1, fade, exit "vwap", S = 20 bp, BTC+ETH, session + 4H pooled.
Gates (day-block bootstrap, 2000 draws):
    G1 VWAP-specific  hour-matched mean gross(real) - mean gross(placebo) > 0 with q2.5 > 0 (within each UTC
                      entry hour, weighted by real counts; removes time-of-day composition differences)
    G2 tradeable      mean gross(real) q2.5 > 7 bp
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from . import vwap_events as ve
from . import vwap_touch_stats as vt

VERSION = "VE03-0.1.0"


@dataclass(frozen=True)
class FadeParams:
    k_grid: tuple = (1.0, 2.0)
    k_primary: float = 1.0
    stretch_m: int = 15
    warmup_min: int = 30
    max_hold: int = 60
    targets: tuple = (5, 10, 20, 40, "vwap")
    stops: tuple = (5, 10, 20, 40)
    primary_target: object = "vwap"
    primary_stop: int = 20
    costs_bp: tuple = (4.0, 7.0, 12.0)
    tradeable_bp: float = 7.0
    n_boot: int = 2000
    status: str = "PREREGISTERED_VE004"


def registration(fp: FadeParams | None = None) -> dict:
    fp = fp or FadeParams()
    d = asdict(fp)
    d["version"] = VERSION
    d["gates"] = {"G1": "hour-matched real - placebo mean gross bp, day-bootstrap q2.5 > 0 (primary cell)",
                  "G2": "real mean gross bp q2.5 > 7 (primary cell)"}
    return d


def stretch_events(fr: pd.DataFrame, st: pd.DataFrame, k: float, fp: FadeParams) -> pd.DataFrame:
    """First stretched bar after each cross (one per crossing segment)."""
    v_now = st["lev"].shift(-1).to_numpy()            # VWAP through bar t = level of bar t+1 (same window)
    aid = st["aid"].to_numpy()
    same_next = np.r_[aid[1:] == aid[:-1], False]
    v_now = np.where(same_next, v_now, np.nan)
    side = st["side"].to_numpy()
    since = st["since"].to_numpy()
    logc = fr["logc"].to_numpy()
    sig_prev = np.r_[np.nan, fr["sigma"].to_numpy()[:-1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        dist = np.abs(logc - np.log(v_now))
    stretched = dist >= k * sig_prev * np.sqrt(fp.stretch_m)
    out_t, out_s = [], []
    last_side, armed = 0, False
    n = len(st)
    for i in range(n):
        if i > 0 and aid[i] != aid[i - 1]:
            last_side, armed = 0, False
        s = side[i]
        if s != 0:
            if last_side != 0 and s == -last_side:
                armed = True                          # a new crossing segment begins
            last_side = s
        if armed and s != 0 and since[i] >= fp.warmup_min and stretched[i]:
            out_t.append(i)
            out_s.append(s)
            armed = False
    return pd.DataFrame({"i": np.asarray(out_t, int), "side": np.asarray(out_s, int)})


def simulate(fr: pd.DataFrame, st: pd.DataFrame, ev: pd.DataFrame, d: np.ndarray, target, stop: float,
             fp: FadeParams) -> np.ndarray:
    """Gross P&L (bp) for positions d (+1 long, -1 short) entered at close of bar i+1."""
    hi = np.log(fr["high"].to_numpy())
    lo = np.log(fr["low"].to_numpy())
    cl = fr["logc"].to_numpy()
    lev = np.log(st["lev"].to_numpy())
    n = len(cl)
    e = ev["i"].to_numpy() + 1
    H = fp.max_hold
    ok = e + H < n
    out = np.full(len(e), np.nan)
    if not ok.any():
        return out
    e, dd = e[ok], d[ok]
    idx = e[:, None] + np.arange(1, H + 1)[None, :]
    p0 = cl[e][:, None]
    up = np.where(dd[:, None] > 0, hi[idx] - p0, -(lo[idx] - p0)) * 1e4      # favourable excursion
    dn = np.where(dd[:, None] > 0, -(lo[idx] - p0), hi[idx] - p0) * 1e4      # adverse excursion
    stop_hit = dn >= stop
    if target == "vwap":
        L = lev[idx]
        tgt_hit = (lo[idx] <= L) & (hi[idx] >= L)
        tgt_pnl = dd[:, None] * (L - p0) * 1e4
    else:
        tgt_hit = up >= float(target)
        tgt_pnl = np.full(up.shape, float(target))
    big = H + 1
    fs = np.where(stop_hit.any(1), stop_hit.argmax(1), big)
    ft = np.where(tgt_hit.any(1), tgt_hit.argmax(1), big)
    fin = np.isfinite(cl[idx]).all(1) & np.isfinite(hi[idx]).all(1)
    pnl = dd * (cl[e + H] - cl[e]) * 1e4                                      # time exit
    take_t = ft < fs
    take_s = (fs <= ft) & (fs < big)
    rows = np.arange(len(e))
    pnl = np.where(take_t, tgt_pnl[rows, np.minimum(ft, H - 1)], pnl)
    pnl = np.where(take_s, -stop, pnl)
    pnl = np.where(fin, pnl, np.nan)
    out[np.flatnonzero(ok)] = pnl
    return out


def run_asset(df: pd.DataFrame, asset: str, fp: FadeParams, specs=vt.SPECS) -> pd.DataFrame:
    fr = ve.prepare(df, ve.Params())
    rows = []
    for sp in specs:
        st = vt.bar_states(fr, sp.period_min, sp.phase_min)
        for k in fp.k_grid:
            ev = stretch_events(fr, st, k, fp)
            if not len(ev):
                continue
            ts = fr.index[ev["i"].to_numpy() + 1]
            day = ts.normalize().asi8
            hour = ts.hour.to_numpy()
            for mode, sgn in (("fade", -1), ("follow", 1)):
                d = sgn * ev["side"].to_numpy()
                for T in fp.targets:
                    if mode == "follow" and T == "vwap":
                        continue
                    for S in fp.stops:
                        pnl = simulate(fr, st, ev, d, T, S, fp)
                        rows.append(pd.DataFrame({"asset": asset, "anchor": sp.name, "kind": sp.kind,
                                                  "phase_min": sp.phase_min, "k": k, "mode": mode,
                                                  "target": str(T), "stop": S, "day": day, "hour": hour,
                                                  "pnl": pnl}))
    R = pd.concat(rows, ignore_index=True)
    for c in ("asset", "anchor", "kind", "mode", "target"):
        R[c] = R[c].astype("category")
    R["pnl"] = R["pnl"].astype("float32")
    return R


def boot(x: np.ndarray, day: np.ndarray, n_boot: int, seed: int):
    ok = np.isfinite(x)
    x, day = x[ok], day[ok]
    ud, inv = np.unique(day, return_inverse=True)
    S, N = np.bincount(inv, weights=x), np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    dd = rng.integers(0, len(ud), size=(n_boot, len(ud)))
    bs = S[dd].sum(1) / N[dd].sum(1)
    return float(x.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), int(len(x))


def boot_diff(a: pd.DataFrame, b: pd.DataFrame, n_boot: int, seed: int):
    """Mean(a) - mean(b), joint day-block bootstrap over the union of days."""
    days = np.unique(np.r_[a.day.to_numpy(), b.day.to_numpy()])
    def agg(f):
        f = f[np.isfinite(f.pnl)]
        i = np.searchsorted(days, f.day.to_numpy())
        return np.bincount(i, weights=f.pnl.to_numpy(), minlength=len(days)), np.bincount(i, minlength=len(days))
    Sa, Na = agg(a); Sb, Nb = agg(b)
    rng = np.random.default_rng(seed)
    d = rng.integers(0, len(days), size=(n_boot, len(days)))
    bs = Sa[d].sum(1) / Na[d].sum(1) - Sb[d].sum(1) / Nb[d].sum(1)
    est = Sa.sum() / Na.sum() - Sb.sum() / Nb.sum()
    return float(est), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def boot_diff_hour(a: pd.DataFrame, b: pd.DataFrame, n_boot: int, seed: int):
    """Hour-matched real - placebo: within each UTC hour of entry, mean(a) - mean(b), weighted by a's hour counts.
    Day-block bootstrap over the union of days. Removes time-of-day composition differences between anchors."""
    a = a[np.isfinite(a.pnl)]
    b = b[np.isfinite(b.pnl)]
    days = np.unique(np.r_[a.day.to_numpy(), b.day.to_numpy()])
    def cube(f):
        i = np.searchsorted(days, f.day.to_numpy())
        h = f.hour.to_numpy()
        S = np.zeros((len(days), 24)); N = np.zeros((len(days), 24))
        np.add.at(S, (i, h), f.pnl.to_numpy()); np.add.at(N, (i, h), 1)
        return S, N
    Sa, Na = cube(a); Sb, Nb = cube(b)
    def stat(sa, na, sb, nb):
        with np.errstate(invalid="ignore", divide="ignore"):
            ma, mb = sa / na, sb / nb
        w = np.where(np.isfinite(ma) & np.isfinite(mb), na, 0)
        return np.nansum(w * (np.nan_to_num(ma) - np.nan_to_num(mb))) / w.sum()
    est = stat(Sa.sum(0), Na.sum(0), Sb.sum(0), Nb.sum(0))
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for k in range(n_boot):
        d = rng.integers(0, len(days), len(days))
        bs[k] = stat(Sa[d].sum(0), Na[d].sum(0), Sb[d].sum(0), Nb[d].sum(0))
    return float(est), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def synthetic_path_klines(n_days: int = 60, seed: int = 0, start: str = "2026-01-01", base_vol: float = 0.0006,
                          sub: int = 30) -> pd.DataFrame:
    """1m klines aggregated from a 2-second martingale path (true high/low), intraday volatility seasonality
    peaking near 08:00 and 14:30 UTC, slow volatility clustering. No VWAP effect by construction."""
    rng = np.random.default_rng(seed)
    n = n_days * 1440
    t = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    h = t.hour.to_numpy() + t.minute.to_numpy() / 60
    season = 1 + 0.5 * np.exp(-((h - 14.5) ** 2) / 2) + 0.3 * np.exp(-((h - 8) ** 2) / 2)
    lv = np.zeros(n); e = rng.standard_normal(n)
    for i in range(1, n):
        lv[i] = 0.999 * lv[i - 1] + 0.02 * e[i]
    sig = base_vol * season * np.exp(lv)
    steps = rng.standard_normal((n, sub)) * (sig / np.sqrt(sub))[:, None]
    path = np.log(100) + np.cumsum(steps.ravel()).reshape(n, sub)
    o = np.r_[np.log(100), path[:-1, -1]]
    c = path[:, -1]
    hi = np.maximum(path.max(1), o); lo = np.minimum(path.min(1), o)
    vol = rng.gamma(2.0, 50.0, n) * season
    typ = np.exp((hi + lo + c) / 3)
    return pd.DataFrame({"open_time": t, "open": np.exp(o), "high": np.exp(hi), "low": np.exp(lo), "close": np.exp(c),
                         "volume": vol, "quote_volume": vol * typ})


def grid_table(R: pd.DataFrame, fp: FadeParams) -> pd.DataFrame:
    out = []
    for key, g in R.groupby(["kind", "k", "mode", "target", "stop"]):
        x = g.pnl.to_numpy()
        m, lo, hi, n = boot(x, g.day.to_numpy(), fp.n_boot, 7)
        ok = np.isfinite(x)
        out.append(dict(zip(["kind", "k", "mode", "target", "stop"], key), gross_bp=m, lo=lo, hi=hi, n=n,
                        win_rate=float((x[ok] > 0).mean()), mean_win=float(x[ok][x[ok] > 0].mean()) if (x[ok] > 0).any() else np.nan,
                        mean_loss=float(x[ok][x[ok] <= 0].mean()) if (x[ok] <= 0).any() else np.nan))
    return pd.DataFrame(out)


def primary(R: pd.DataFrame, fp: FadeParams) -> dict:
    sel = (R.k == fp.k_primary) & (R["mode"] == "fade") & (R.target == str(fp.primary_target)) & (R.stop == fp.primary_stop)
    real, plc = R[sel & (R.kind == "real")], R[sel & (R.kind == "placebo")]
    m, lo, hi, n = boot(real.pnl.to_numpy(), real.day.to_numpy(), fp.n_boot, 11)
    dm, dlo, dhi = boot_diff_hour(real, plc, fp.n_boot, 13)
    um, ulo, uhi = boot_diff(real, plc, fp.n_boot, 13)
    pm = float(np.nanmean(plc.pnl))
    x = real.pnl.to_numpy(); x = x[np.isfinite(x)]
    g1 = "PASS" if dm > 0 and dlo > 0 else "FAIL"
    g2 = "PASS" if lo > fp.tradeable_bp else "FAIL"
    res = {"version": VERSION, "status": fp.status, "n_real": n, "real_gross_bp": m, "real_lo": lo, "real_hi": hi,
           "placebo_gross_bp": pm, "real_minus_placebo_hour_matched": dm, "diff_lo": dlo, "diff_hi": dhi,
           "real_minus_placebo_unmatched": um, "unmatched_lo": ulo, "unmatched_hi": uhi,
           "win_rate_real": float((x > 0).mean()), "G1_vwap_specific": g1, "G2_tradeable": g2}
    for c in fp.costs_bp:
        res[f"real_net_{c:g}bp"] = m - c
    res["verdict"] = ("VWAP_FADE_EDGE" if g1 == "PASS" and g2 == "PASS" else
                      "VWAP_SPECIFIC_BUT_UNTRADEABLE" if g1 == "PASS" else
                      "GENERIC_EDGE_NOT_VWAP" if g2 == "PASS" else "NO_EDGE")
    return res
