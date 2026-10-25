"""Turtle Breakout 05: is the turtle breakout fractal? The original rules on candles from 1 minute to 1 day.

Protocol: discussions/Turtle_Breakout/007_claude_tb05_preregistration.md (fixed before any real-data output).

Rules on every timeframe (20 / 10 / 20 candles, as TB 01): long when price trades above the highest high of the
last 20 completed candles; exit when price trades through max(entry - 2 ATR_20, lowest low of the last 10
completed candles); ATR = Wilder-style
EWM (alpha = 1/20) of the candle true range; no re-entry in the candle of an exit; long only; one position per
asset; unit notional per trade (no sizing / compounding: the question is the edge per trade).

Fills are resolved on base bars: 1-minute bars for candles of 1-3 minutes, 5-minute bars otherwise.
FILL CONVENTION (differs from TB 01): every fill is the CLOSE of the base bar in which the trigger trades
(breakout bar for entries, stop bar for exits). Filling exactly at the level is optimistic by the overshoot of a
discrete crossing (synthetic martingale check: +0.07 R per trade at 1-60 min, an artefact that grows as candles
shrink); the bar close is an unbiased stand-in for the price at the crossing (martingale), so no timeframe is
favoured. A stop is checked from the bar after the entry bar. Random-entry null: entry at the candle open.

Per trade:  R_gross = (exit - entry) / (2 ATR)          edge in risk units (fractal if constant across timeframes)
            bp_gross = 1e4 (exit / entry - 1)            edge in price units
            bp_net   = bp_gross - 1e4 c (1 + exit/entry) - funding,  c = 0.05% taker + 1 bp slippage per side
Null (per timeframe): random entries at candle opens with the real entry rate, same exits; statistic = pooled mean
R_gross per trade.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import turtle_breakout as tb

VERSION = "TB05-0.1.0"
TIMEFRAMES = (1, 2, 3, 5, 10, 15, 30, 60, 120, 240, 480, 720, 1440)     # minutes; 1440 = 17:00 New York day
N_NULL = {1: 50, 2: 50, 3: 50, 5: 100, 10: 100, 15: 100, 30: 200, 60: 200, 120: 200, 240: 200, 480: 200,
          720: 200, 1440: 400}


@dataclass(frozen=True)
class FrParams:
    entry_n: int = 20
    exit_n: int = 10
    atr_n: int = 20
    stop_atr: float = 2.0
    side_cost: float = 0.0006        # 0.05% taker + 1 bp slippage per side
    maker_side_cost: float = 0.0001  # reported scenario: maker fill, 1 bp adverse selection allowance per side
    min_bar_share: float = 0.8
    alpha: float = 0.05
    n_boot: int = 2000
    seed: int = 20261028
    status: str = "PREREGISTERED_TB007"


def registration(fp: FrParams | None = None) -> dict:
    fp = fp or FrParams()
    d = asdict(fp)
    d.update(version=VERSION, timeframes_min=list(TIMEFRAMES), n_null=N_NULL, assets=list(tb.ASSETS),
             window=[str(tb.COMMON_START), "end of data"], fills="close of the trigger base bar (entries and exits)",
             gates={"G_edge": "z = (real - mean null) / sd null over the random-entry runs, one-sided p = 1 - Phi(z) < alpha/13 "
                               "(normal approximation: 50-400 runs cannot resolve p < 0.004 empirically; the empirical "
                               "share is reported)",
                    "G_net": "pooled mean bp_net (taker) > 0 with month-cluster bootstrap (1 - alpha/13) lower bound > 0",
                    "verdict": "EDGE_TRADEABLE (both) / EDGE_GROSS_ONLY (G_edge) / NO_EDGE"},
             fractal_summary="real - null mean R_gross by timeframe (reported; fractal if flat in log timeframe)")
    return d


# ---------------------------------------------------------------- bars

def bars_1m(df1m: pd.DataFrame) -> pd.DataFrame:
    d = df1m[["open_time", "open", "high", "low", "close"]].copy()
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d = d.drop_duplicates("open_time").sort_values("open_time")
    b = d.set_index("open_time")[["open", "high", "low", "close"]].astype(float)
    b.index = b.index + pd.Timedelta(minutes=1)
    b.index.name = "end"
    return b


def base_for(tf: int) -> int:
    return 1 if tf < 5 else 5


@dataclass
class Frame:
    tf: int
    base: int
    end: pd.DatetimeIndex
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    fcum: np.ndarray
    s: np.ndarray
    e: np.ndarray
    pend: pd.DatetimeIndex
    full: np.ndarray
    ph: np.ndarray
    pl: np.ndarray
    hi_n: np.ndarray
    lo_x: np.ndarray
    atr: np.ndarray
    first_up: np.ndarray
    smin: np.ndarray


def candle_id(end: pd.DatetimeIndex, tf: int) -> np.ndarray:
    if tf == 1440:
        return tb.period_id(end, "day")
    return np.asarray((end - pd.Timedelta(seconds=1) - pd.Timestamp("1970-01-01", tz="UTC"))
                      // pd.Timedelta(minutes=tf)).astype(np.int64)


def prepare(bars: pd.DataFrame, fund: pd.DataFrame | None, tf: int, fp: FrParams) -> Frame:
    base = int(round((bars.index[1] - bars.index[0]).total_seconds() / 60)) if len(bars) > 1 else base_for(tf)
    end = pd.DatetimeIndex(bars.index)
    o, h, l, c = (bars[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    n = len(c)
    fr = np.zeros(n)
    if fund is not None and len(fund):
        j = end.searchsorted(pd.DatetimeIndex(fund["t"]), side="left")
        ok = j < n
        np.add.at(fr, j[ok], fund["rate"].to_numpy()[ok] * c[j[ok]])
    fcum = np.cumsum(fr)
    cid = candle_id(end, tf)
    first = np.r_[True, cid[1:] != cid[:-1]]
    last = np.r_[cid[1:] != cid[:-1], True]
    s = np.flatnonzero(first); e = np.flatnonzero(last)
    pids = cid[s]
    K = len(s)
    kbar = np.cumsum(first) - 1                                   # candle index of every bar (only existing candles)
    cnt = e - s + 1
    full = cnt >= fp.min_bar_share * max(1, tf // base)
    ph = np.maximum.reduceat(h, s); pl = np.minimum.reduceat(l, s); pc = c[e]
    # gaps in candle ids (missing candles) are ignored: rolling windows count existing candles
    prev_c = np.r_[np.nan, pc[:-1]]
    tr = np.nanmax(np.c_[ph - pl, np.abs(ph - prev_c), np.abs(pl - prev_c)], axis=1)
    atr_c = pd.Series(tr).ewm(alpha=1.0 / fp.atr_n, adjust=False, min_periods=fp.atr_n).mean().to_numpy()
    shift = lambda x: np.r_[np.nan, x[:-1]]
    hi_n = shift(pd.Series(ph).rolling(fp.entry_n, min_periods=fp.entry_n).max().to_numpy())
    lo_x = shift(pd.Series(pl).rolling(fp.exit_n, min_periods=fp.exit_n).min().to_numpy())
    atr = shift(atr_c)
    up = np.flatnonzero(h >= hi_n[kbar])
    first_up = np.full(K, -1)
    if len(up):
        kk = kbar[up]
        firsts = np.r_[True, kk[1:] != kk[:-1]]
        first_up[kk[firsts]] = up[firsts]
    smin = pd.Series(l[::-1]).groupby(kbar[::-1]).cummin().to_numpy()[::-1]
    pend = tb.period_end(pids, "day") if tf == 1440 else pd.to_datetime((pids + 1) * tf * 60, unit="s", utc=True)
    return Frame(tf, base, end, o, h, l, c, fcum, s, e, pend, full, ph, pl, hi_n, lo_x, atr, first_up, smin)


# ---------------------------------------------------------------- event-driven simulation

def _exit_after(F: Frame, k0: int, fixed: float, K: int) -> tuple[int, float]:
    """First candle k >= k0 whose low trades through max(fixed, lo_x[k]); returns (k, stop) or (-1, nan)."""
    kk, ch = k0, 64
    while kk < K:
        sl = slice(kk, min(K, kk + ch))
        lx = F.lo_x[sl]
        st = np.where(np.isfinite(lx), np.maximum(fixed, lx), fixed)
        hit = F.pl[sl] <= st
        if hit.any():
            i = int(np.argmax(hit))
            return kk + i, float(st[i])
        kk += ch
        ch = min(ch * 2, 1 << 16)
    return -1, np.nan


def _first_bar_hit(F: Frame, a: int, b: int, stop: float) -> int:
    seg = F.l[a:b + 1] <= stop
    return a + int(np.argmax(seg))


def simulate(F: Frame, fp: FrParams, k_start: int = 0, mode: str = "rule", p_entry: float = 0.0,
             rng: np.random.Generator | None = None) -> tuple[pd.DataFrame, dict]:
    """Single asset, unit notional. Returns trades (closed only) and stats (eligible flat candles, entries)."""
    K = len(F.s)
    elig = F.full & np.isfinite(F.atr) & (F.atr > 0) & np.isfinite(F.hi_n) & np.isfinite(F.lo_x)
    elig[:k_start] = False
    E_k = np.flatnonzero(elig & (F.first_up >= 0)) if mode == "rule" else None
    G_k = np.flatnonzero(elig)
    rows = []
    k, flat_elig = k_start, 0
    while True:
        if mode == "rule":
            i = int(np.searchsorted(E_k, k))
            if i >= len(E_k):
                break
            ke = int(E_k[i])
            j = int(F.first_up[ke])
            px = F.c[j]
            a0 = j + 1
        else:
            i0 = int(np.searchsorted(G_k, k))
            g = int(rng.geometric(p_entry)) if p_entry > 0 else 10 ** 12
            if i0 + g - 1 >= len(G_k):
                break
            ke = int(G_k[i0 + g - 1])
            j = int(F.s[ke]); px = F.o[j]; a0 = j
        flat_elig += int(np.searchsorted(G_k, ke) - np.searchsorted(G_k, k))
        atr = F.atr[ke]
        fixed = px - fp.stop_atr * atr
        st = max(fixed, F.lo_x[ke]) if np.isfinite(F.lo_x[ke]) else fixed
        ex_k = -1
        if a0 <= F.e[ke] and F.smin[a0] <= st:
            jx = _first_bar_hit(F, a0, F.e[ke], st)
            fill = F.c[jx]; ex_k = ke; reason = "same_candle"
        else:
            ex_k, st2 = _exit_after(F, ke + 1, fixed, K)
            if ex_k < 0:
                break                                                  # still open at the end: excluded
            jx = _first_bar_hit(F, F.s[ex_k], F.e[ex_k], st2)
            fill = F.c[jx]; reason = "trail" if st2 > fixed else "fixed"
        rows.append((ke, ex_k, j, jx, px, fill, atr, reason))
        k = ex_k + 1
    T = pd.DataFrame(rows, columns=["k_entry", "k_exit", "je", "jx", "entry_px", "exit_px", "atr", "reason"])
    if len(T):
        T["R_gross"] = (T.exit_px - T.entry_px) / (fp.stop_atr * T.atr)
        T["bp_gross"] = 1e4 * (T.exit_px / T.entry_px - 1)
        T["bp_funding"] = 1e4 * (F.fcum[T.jx] - F.fcum[T.je]) / T.entry_px
        T["bp_net"] = T.bp_gross - 1e4 * fp.side_cost * (1 + T.exit_px / T.entry_px) - T.bp_funding
        T["bp_net_maker"] = T.bp_gross - 1e4 * fp.maker_side_cost * (1 + T.exit_px / T.entry_px) - T.bp_funding
        T["R_net"] = T.bp_net / 1e4 * T.entry_px / (fp.stop_atr * T.atr)
        T["entry_time"] = F.end[T.je.to_numpy()]
        T["exit_time"] = F.end[T.jx.to_numpy()]
        T["hold_min"] = (T.exit_time - T.entry_time).dt.total_seconds() / 60 + F.base
        T["cost_R"] = 2e4 * fp.side_cost / 1e4 * T.entry_px / (fp.stop_atr * T.atr)
    return T, {"entries": len(T), "flat_eligible": flat_elig,
               "p_entry": len(T) / flat_elig if flat_elig else 0.0}


def k_start_for(F: Frame, start: pd.Timestamp) -> int:
    return int(np.searchsorted(F.pend, start))


# ---------------------------------------------------------------- statistics

def null_means(frames: dict, fp: FrParams, p_entry: dict, k0: dict, n_runs: int, seed: int) -> np.ndarray:
    """Pooled mean R_gross per trade of random-entry runs (same exits)."""
    ss = np.random.SeedSequence(seed)
    out = np.empty(n_runs)
    for r, child in enumerate(ss.spawn(n_runs)):
        rng = np.random.default_rng(child)
        tot, cnt = 0.0, 0
        for a, F in frames.items():
            T, _ = simulate(F, fp, k0[a], mode="random", p_entry=p_entry[a], rng=rng)
            if len(T):
                tot += T.R_gross.sum(); cnt += len(T)
        out[r] = tot / cnt if cnt else np.nan
    return out


def month_boot(x: np.ndarray, t: pd.DatetimeIndex, n_boot: int, q: float, seed: int) -> tuple[float, float, float]:
    ok = np.isfinite(x)
    x = x[ok]; m = pd.DatetimeIndex(t[ok]).tz_convert("UTC").strftime("%Y-%m").to_numpy()
    u, inv = np.unique(m, return_inverse=True)
    S = np.bincount(inv, weights=x); N = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=n_boot).astype(float)
    bs = (W @ S) / (W @ N)
    lo, hi = np.quantile(bs, [q / 2, 1 - q / 2])
    return float(x.mean()), float(lo), float(hi)


def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def summarize(tf: int, T: pd.DataFrame, null: np.ndarray, fp: FrParams, days: float) -> dict:
    a = fp.alpha / len(TIMEFRAMES)
    real = float(T.R_gross.mean()) if len(T) else np.nan
    share = float((null >= real).mean()) if len(T) else np.nan
    mu, sd = float(np.nanmean(null)), float(np.nanstd(null, ddof=1))
    z = (real - mu) / sd if (len(T) and sd > 0) else np.nan
    p_z = float(1 - _norm_cdf(z)) if np.isfinite(z) else np.nan
    net, lo, hi = month_boot(T.bp_net.to_numpy(), pd.DatetimeIndex(T.entry_time), fp.n_boot, a, fp.seed + tf) if len(T) > 10 else (np.nan,) * 3
    g_edge = bool(np.isfinite(p_z) and p_z < a)
    g_net = bool(np.isfinite(lo) and lo > 0)
    pos = T.bp_net[T.bp_net > 0].sum(); neg = -T.bp_net[T.bp_net <= 0].sum()
    return {"tf_min": tf, "trades": int(len(T)), "trades_per_day": len(T) / days if days else np.nan,
            "median_hold_min": float(T.hold_min.median()) if len(T) else np.nan,
            "win_rate_gross": float((T.bp_gross > 0).mean()) if len(T) else np.nan,
            "mean_R_gross": real, "null_mean_R_gross": float(np.nanmedian(null)), "edge_R": real - float(np.nanmedian(null)),
            "null_sd": sd, "z": z, "p_z": p_z, "null_share_ge_real": share,
            "mean_bp_gross": float(T.bp_gross.mean()) if len(T) else np.nan,
            "breakeven_rt_cost_bp": float((T.bp_gross - T.bp_funding).mean()) if len(T) else np.nan,
            "median_cost_R": float(T.cost_R.median()) if len(T) else np.nan,
            "mean_bp_net": net, "net_lo": lo, "net_hi": hi,
            "mean_bp_net_maker": float(T.bp_net_maker.mean()) if len(T) else np.nan,
            "mean_R_net": float(T.R_net.mean()) if len(T) else np.nan,
            "profit_factor_net": float(pos / neg) if neg > 0 else np.nan,
            "G_edge": "PASS" if g_edge else "FAIL", "G_net": "PASS" if g_net else "FAIL",
            "verdict": "EDGE_TRADEABLE" if (g_edge and g_net) else "EDGE_GROSS_ONLY" if g_edge else "NO_EDGE"}
