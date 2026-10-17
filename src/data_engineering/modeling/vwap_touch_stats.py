"""VWAP Events 02 (descriptive): how often do BTC/ETH touch and cross their session and 4H VWAP, and after a
cross, how long until price comes back to it from the other side?

Data: Binance USD-M perpetual 1m klines (open_time, high, low, close, volume, quote_volume), as VWAP Events 01.
Anchors and VWAP exactly as VWAP Events 01 (vwap_events.py): session = 08:00 UTC (Deribit expiry),
4H = 00/04/08/12/16/20 UTC; VWAP_t = cumulative quote_volume / volume since the anchor through bar t.

Definitions (bar t, inside one anchor window; the window's first bar is skipped because no level exists yet)
    level L_t       = VWAP through bar t-1 (the value known when bar t opens), as in VWAP Events 01
    touch           low_t <= L_t <= high_t
    touch episode   a run of consecutive touch bars counts once
    side            s_t = sign(close_t - VWAP_t)   (0 = exactly on it, ignored)
    cross           s_t = -s_prev, where s_prev is the last nonzero side in the same window
After a cross at bar c that leaves price on side s (+1 above, -1 below):
    return touch    first bar k > c with low_k <= L_k <= high_k   (price comes back to VWAP from side s)
    cross back      first bar k > c with s_k = -s                 (price crosses to the other side again)
    detached        first bar d > c with no touch; "return after detaching" = first touch after d
Times are in minutes; a cross whose window ends first is right-censored (reported as share not returned).
Kaplan-Meier medians account for censoring.

Reference: identical statistics on placebo VWAPs anchored at times nobody uses (same period, other phases),
so real-vs-placebo shows whether the real levels are touched / crossed / revisited differently.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import vwap_events as ve

VERSION = "VE02-0.1.0"


@dataclass(frozen=True)
class Spec:
    name: str
    period_min: int
    phase_min: int
    kind: str            # "real" or "placebo"


SPECS = (
    Spec("session", 1440, 8 * 60, "real"),
    Spec("session", 1440, 2 * 60, "placebo"),
    Spec("session", 1440, 14 * 60, "placebo"),
    Spec("session", 1440, 20 * 60, "placebo"),
    Spec("h4", 240, 0, "real"),
    Spec("h4", 240, 60, "placebo"),
    Spec("h4", 240, 120, "placebo"),
    Spec("h4", 240, 180, "placebo"),
)


def bar_states(fr: pd.DataFrame, period_min: int, phase_min: int) -> pd.DataFrame:
    """Per-bar level, touch, side and window id."""
    v, since, aid = ve.anchored_vwap(fr, period_min, phase_min)
    lev = np.r_[np.nan, v[:-1]]
    same = np.r_[False, aid[1:] == aid[:-1]]
    lev = np.where(same, lev, np.nan)
    hi, lo, cl = fr["high"].to_numpy(), fr["low"].to_numpy(), fr["close"].to_numpy()
    ok = np.isfinite(lev) & np.isfinite(hi) & np.isfinite(lo) & np.isfinite(cl)
    touch = ok & (lo <= lev) & (hi >= lev)
    with np.errstate(invalid="ignore"):
        side = np.where(ok & np.isfinite(v), np.sign(cl - v), 0).astype(int)
    return pd.DataFrame({"aid": aid, "since": since, "lev": lev, "ok": ok, "touch": touch, "side": side},
                        index=fr.index)


def events(st: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Crosses with return times, plus touch counts."""
    aid = st["aid"].to_numpy()
    touch = st["touch"].to_numpy()
    side = st["side"].to_numpy()
    ok = st["ok"].to_numpy()
    n = len(st)
    # touch episodes: touch bar whose previous bar (same window) was not a touch
    prev_touch = np.r_[False, touch[:-1]] & np.r_[False, aid[1:] == aid[:-1]]
    episodes = int((touch & ~prev_touch).sum())
    # window end index for censoring
    starts = np.r_[0, np.flatnonzero(np.diff(aid)) + 1]
    ends = np.r_[starts[1:], n]
    win_end = np.empty(n, int)
    for s, e in zip(starts, ends):
        win_end[s:e] = e
    # last nonzero side within window
    rows = []
    last_side = 0
    for i in range(n):
        if i > 0 and aid[i] != aid[i - 1]:
            last_side = 0
        if not ok[i] or side[i] == 0:
            continue
        if last_side != 0 and side[i] == -last_side:
            rows.append(i)
        last_side = side[i]
    crosses = np.asarray(rows, int)
    # next touch / next opposite side / detach, searched forward within the window
    t_touch, t_back, t_ret_det = [], [], []
    touch_idx = np.flatnonzero(touch)
    for c in crosses:
        e = win_end[c]
        s = side[c]
        j = np.searchsorted(touch_idx, c + 1)
        nt = touch_idx[j] if j < len(touch_idx) and touch_idx[j] < e else -1
        seg = side[c + 1:e]
        kb = np.flatnonzero(seg == -s)
        nb = c + 1 + kb[0] if len(kb) else -1
        segt = touch[c + 1:e]
        kd = np.flatnonzero(~segt & ok[c + 1:e])
        if len(kd):
            d = c + 1 + kd[0]
            j2 = np.searchsorted(touch_idx, d + 1)
            nr = touch_idx[j2] if j2 < len(touch_idx) and touch_idx[j2] < e else -1
            rd = (nr - d) if nr >= 0 else np.nan
        else:
            d, rd = -1, np.nan
        t_touch.append((nt - c) if nt >= 0 else np.nan)
        t_back.append((nb - c) if nb >= 0 else np.nan)
        t_ret_det.append(rd)
    ev = pd.DataFrame({"t": st.index[crosses], "direction": np.where(side[crosses] > 0, "up", "down"),
                       "min_to_window_end": win_end[crosses] - crosses,
                       "min_to_touch": t_touch, "min_to_cross_back": t_back, "min_return_after_detach": t_ret_det})
    counts = {"bars": int(ok.sum()), "windows": int(len(np.unique(aid[ok]))), "touch_bars": int(touch.sum()),
              "touch_episodes": episodes, "crosses": int(len(crosses)),
              "crosses_up": int((side[crosses] > 0).sum()), "crosses_down": int((side[crosses] < 0).sum())}
    return ev, counts


def km_median(time: np.ndarray, censor_time: np.ndarray) -> float:
    """Kaplan-Meier median of time-to-event with right censoring at censor_time where time is NaN."""
    obs = np.isfinite(time)
    t = np.where(obs, time, censor_time).astype(float)
    order = np.argsort(t, kind="stable")
    t, e = t[order], obs[order]
    n = len(t)
    s = 1.0
    i = 0
    while i < n:
        j = i
        while j < n and t[j] == t[i]:
            j += 1
        d = e[i:j].sum()
        s *= 1 - d / (n - i)
        if s <= 0.5:
            return float(t[i])
        i = j
    return float("nan")


def summarise(ev: pd.DataFrame, col: str) -> dict:
    x = ev[col].to_numpy(float)
    obs = np.isfinite(x)
    out = {"n": int(len(x)), "share_returned_before_reset": float(obs.mean()) if len(x) else np.nan,
           "mean_min_if_returned": float(np.nanmean(x)) if obs.any() else np.nan,
           "median_min_if_returned": float(np.nanmedian(x)) if obs.any() else np.nan,
           "km_median_min": km_median(x, ev["min_to_window_end"].to_numpy(float)) if len(x) else np.nan}
    for k in (1, 5, 15, 60):
        out[f"share_within_{k}m"] = float((x <= k).mean()) if len(x) else np.nan
    return out


def run_asset(df: pd.DataFrame, asset: str, specs=SPECS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fr = ve.prepare(df, ve.Params())
    cnt, summ, evs = [], [], []
    days = (fr.index[-1] - fr.index[0]).total_seconds() / 86400
    for sp in specs:
        st = bar_states(fr, sp.period_min, sp.phase_min)
        ev, c = events(st)
        ev["asset"], ev["anchor"], ev["kind"], ev["phase_min"] = asset, sp.name, sp.kind, sp.phase_min
        evs.append(ev)
        c.update(asset=asset, anchor=sp.name, kind=sp.kind, phase_min=sp.phase_min, days=days,
                 touch_episodes_per_window=c["touch_episodes"] / max(c["windows"], 1),
                 crosses_per_window=c["crosses"] / max(c["windows"], 1),
                 crosses_per_day=c["crosses"] / days)
        cnt.append(c)
        for col in ("min_to_touch", "min_to_cross_back", "min_return_after_detach"):
            for d in ("all", "up", "down"):
                sub = ev if d == "all" else ev[ev.direction == d]
                summ.append({"asset": asset, "anchor": sp.name, "kind": sp.kind, "phase_min": sp.phase_min,
                             "measure": col, "direction": d, **summarise(sub, col)})
    return pd.DataFrame(cnt), pd.DataFrame(summ), pd.concat(evs, ignore_index=True)


def real_vs_placebo(summ: pd.DataFrame) -> pd.DataFrame:
    """Real anchor vs mean of its placebo phases, per asset/anchor/measure (direction = all)."""
    s = summ[summ.direction == "all"]
    keys = ["asset", "anchor", "measure"]
    real = s[s.kind == "real"].set_index(keys)
    plc = s[s.kind == "placebo"].groupby(keys).mean(numeric_only=True)
    cols = ["share_returned_before_reset", "median_min_if_returned", "km_median_min", "share_within_5m"]
    out = real[cols].join(plc[cols], rsuffix="_placebo")
    return out.reset_index()
