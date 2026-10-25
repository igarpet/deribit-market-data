"""Turtle Breakout 07: what happens after a LOSING daily turtle trade is stopped out, and does stop-and-reverse pay?

Protocol: discussions/Turtle_Breakout/009_claude_tb07_preregistration.md (fixed before any real-data output).

Long trades: TB 01 daily rules (turtle_breakout.Params() defaults), via turtle_trade_features.run_trades.
A losing exit = a long closed by its stop (trail / fixed_stop / stop_same_period) with net R < 0, which is known
at the exit fill. Universe as TB 04: DISCOVERY = 26 other USDT perps (full history), CONFIRMATION =
BTC/ETH/SOL/XRP from 2020-10-15. TON after 2026-08-07 is excluded (flat-series artefact, TB 007 §0).

Stop-and-reverse (decision): at every losing exit, open a SHORT at the exit fill, unit notional, mirrored exits
    fixed stop  = reverse price + 2 ATR (ATR known at the start of the exit day); never moves
    trailing    = highest high of the last 10 completed days, recalculated daily
    exit        = min(fixed, trailing) traded through, fill max(stop, bar open), checked from the bar after
                  the reverse bar; the short is also closed at the next long breakout of the same asset, at
                  that long's entry fill (a stop hit in the same bar comes first)
    costs       taker 0.05% + 1 bp per side on the short; funding (a short receives positive funding)
    R_rev       net return per notional / (2 ATR / reverse price)
Null: shorts opened at the open of random FLAT days (no long open) of the same asset, same exits and costs,
same number per asset as the real reversals; 2,000 runs; empirical one-sided p.
Event study (reported): forward return at +1/3/5/10 days after losing exits, winning exits and random flat-day
opens, in ATR units and bp; plus the net R of a fixed-horizon short held h days.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import turtle_breakout as tb

VERSION = "TB07-0.1.0"
HORIZONS_D = (1, 3, 5, 10)
STOP_REASONS = ("trail", "fixed_stop", "stop_same_period")
TON_CUTOFF = pd.Timestamp("2026-08-07", tz="UTC")


@dataclass(frozen=True)
class RParams:
    stop_atr: float = 2.0
    side_cost: float = 0.0006          # taker 0.05% + 1 bp slippage, per side
    n_null: int = 2000
    n_boot: int = 2000
    alpha: float = 0.05
    seed: int = 20261030
    status: str = "PREREGISTERED_TB009"


def registration(rp: RParams | None = None) -> dict:
    d = asdict(rp or RParams())
    d.update(version=VERSION, horizons_days=list(HORIZONS_D), losing_exit="long closed by a stop with net R < 0",
             reverse="short at the exit fill; fixed stop +2 ATR, trailing 10-day high; closed at the next long entry",
             gates={"G1": "confirmation: mean R_rev > 0 with month-cluster bootstrap 95% lower bound > 0",
                    "G2": "confirmation: mean R_rev above random flat-day shorts, empirical one-sided p < 0.05 (2,000 runs)",
                    "verdict": "REVERSE_PAYS (G1 & G2) / BEATS_RANDOM_SHORTS_ONLY (G2) / NO_EDGE"},
             reported=["discovery set, same statistics", "event study vs winners and random flat days",
                       "fixed-horizon shorts", "long-only vs long + reverse curves", "by exit reason, by year, by asset"])
    return d


# ---------------------------------------------------------------- mechanics

def _k_of(d: tb.AssetData, j: int) -> int:
    return int(tb.period_id(pd.DatetimeIndex([d.end[j]]), "day")[0]) - d.pid0


def _longs(T: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if not len(T):
        return np.array([], int), np.array([])
    o = np.argsort(T.je.to_numpy())
    return T.je.to_numpy()[o].astype(int), T.entry_px.to_numpy()[o].astype(float)


def short_exit(d: tb.AssetData, j0: int, k0: int, fixed: float, longs: tuple[np.ndarray, np.ndarray],
               first_bar: int | None = None) -> tuple[int, float, str]:
    """Walk a short opened at bar j0 (period k0). Stop checks start at first_bar (default j0 + 1).
    Returns (exit bar, fill, reason) or (-1, nan, 'open')."""
    lj, lpx = longs
    i = int(np.searchsorted(lj, j0, side="right"))
    jl = int(lj[i]) if i < len(lj) else -1
    a = j0 + 1 if first_bar is None else first_bar
    for k in range(k0, len(d.s)):
        if d.s[k] < 0:
            continue
        lo, hi = (a, d.e[k]) if k == k0 else (d.s[k], d.e[k])
        if lo > hi:
            continue
        st = min(fixed, d.hi_x[k]) if np.isfinite(d.hi_x[k]) else fixed
        hit = np.flatnonzero(d.h[lo:hi + 1] >= st)
        js = lo + int(hit[0]) if len(hit) else -1
        if lo <= jl <= hi and (js < 0 or jl < js):
            return jl, float(lpx[i]), "next_long"
        if js >= 0:
            return js, float(max(st, d.o[js])), "stop"
    return -1, np.nan, "open"


def _short_R(d, j0, jx, px, fill, atr, rp):
    fund = (d.fcum[jx] - d.fcum[j0]) / px
    ret = (px - fill) / px - rp.side_cost * (1 + fill / px) + fund
    return ret, ret / (rp.stop_atr * atr / px), fund


def losing_exits(T: pd.DataFrame) -> pd.DataFrame:
    return T[(T.R < 0) & T.reason.isin(STOP_REASONS)].copy()


def reverse_trades(d: tb.AssetData, T: pd.DataFrame, rp: RParams) -> pd.DataFrame:
    longs = _longs(T)
    rows = []
    for r in losing_exits(T).itertuples(index=False):
        j0 = int(r.jx); k0 = _k_of(d, j0)
        atr = d.atr[k0] if np.isfinite(d.atr[k0]) and d.atr[k0] > 0 else r.atr_entry
        px = float(r.exit_px)
        jx, fill, why = short_exit(d, j0, k0, px + rp.stop_atr * atr, longs)
        if jx < 0:
            continue
        ret, R, fund = _short_R(d, j0, jx, px, fill, atr, rp)
        rows.append({"asset": d.name, "long_entry_time": r.entry_time, "rev_time": d.end[j0], "rev_exit_time": d.end[jx],
                     "long_R": float(r.R), "long_reason": r.reason, "rev_px": px, "rev_exit_px": fill, "atr_rev": atr,
                     "rev_reason": why, "rev_ret": ret, "R_rev": R, "R_rev_gross": (px - fill) / (rp.stop_atr * atr),
                     "hold_days": (d.end[jx] - d.end[j0]) / pd.Timedelta(days=1), "funding_bp": 1e4 * fund})
    return pd.DataFrame(rows)


def flat_days(d: tb.AssetData, T: pd.DataFrame, k_lo: int, k_hi: int | None = None) -> np.ndarray:
    """Periods in [k_lo, k_hi] with data, ATR and a 10-day high, with no long open at the period start."""
    K = len(d.s)
    k_hi = K - 1 if k_hi is None else min(k_hi, K - 1)
    ks = np.arange(K)
    ok = (ks >= k_lo) & (ks <= k_hi) & (d.s >= 0) & np.isfinite(d.atr) & (d.atr > 0) & np.isfinite(d.hi_x)
    if len(T):
        st = np.array([_k_of(d, int(j)) for j in T.je]); en = np.array([_k_of(d, int(j)) for j in T.jx])
        busy = np.zeros(K + 1, int)
        np.add.at(busy, np.minimum(st + 1, K), 1); np.add.at(busy, np.minimum(en + 1, K), -1)  # open at start of st+1..en
        ok &= np.cumsum(busy)[:K] == 0
    return np.flatnonzero(ok)


def random_short_pool(d: tb.AssetData, T: pd.DataFrame, k_lo: int, rp: RParams, k_hi: int | None = None) -> pd.DataFrame:
    """The (deterministic) outcome of a short opened at the open of every flat day; the null samples from it."""
    longs = _longs(T)
    rows = []
    for k in flat_days(d, T, k_lo, k_hi):
        j0 = int(d.s[k]); px = float(d.o[j0]); atr = float(d.atr[k])
        jx, fill, why = short_exit(d, j0, k, px + rp.stop_atr * atr, longs, first_bar=j0)
        if jx < 0:
            continue
        _, R, _ = _short_R(d, j0, jx, px, fill, atr, rp)
        rows.append({"asset": d.name, "t": d.end[j0], "R": R, "reason": why})
    return pd.DataFrame(rows)


def null_means(pools: dict, counts: dict, rp: RParams, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros(rp.n_null); tot = 0
    for a, n in counts.items():
        x = pools[a].R.to_numpy() if a in pools and len(pools[a]) else np.array([])
        if n == 0 or not len(x):
            continue
        out += x[rng.integers(0, len(x), size=(rp.n_null, n))].sum(1); tot += n
    return out / max(tot, 1)


def event_study(d: tb.AssetData, T: pd.DataFrame, pool: pd.DataFrame, rp: RParams) -> pd.DataFrame:
    """Forward returns from losing exits, winning exits and random flat-day opens (ATR of the start day).
    short_R_net = net R of a short opened at the start and closed exactly h days later."""
    starts = [("after_loss" if r.R < 0 else "after_win", int(r.jx), float(r.exit_px))
              for r in T[T.reason.isin(STOP_REASONS)].itertuples(index=False)]
    if len(pool):
        js = d.end.searchsorted(pd.DatetimeIndex(pool.t))
        starts += [("random_flat", int(j), float(d.o[j])) for j in js]
    rows = []
    for kind, j0, px in starts:
        atr = d.atr[_k_of(d, j0)]
        if not (np.isfinite(atr) and atr > 0):
            continue
        for h in HORIZONS_D:
            j1 = int(d.end.searchsorted(d.end[j0] + pd.Timedelta(days=h)))
            if j1 >= len(d.c) or (d.end[j1] - d.end[j0]) > pd.Timedelta(days=h, hours=1):
                continue
            c1 = float(d.c[j1])
            _, R, _ = _short_R(d, j0, j1, px, c1, atr, rp)
            rows.append({"asset": d.name, "kind": kind, "t": d.end[j0], "h": h, "bp": 1e4 * math.log(c1 / px),
                         "atr_units": (c1 - px) / atr, "short_R_net": R})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- statistics

def month_boot_mean(x, t, n_boot: int, seed: int, q: float = 0.05) -> tuple[float, float, float]:
    x = np.asarray(x, float); ok = np.isfinite(x)
    x = x[ok]
    if len(x) < 3:
        return (float(x.mean()) if len(x) else np.nan), np.nan, np.nan
    m = pd.DatetimeIndex(pd.to_datetime(np.asarray(t)[ok], utc=True)).strftime("%Y-%m").to_numpy()
    u, inv = np.unique(m, return_inverse=True)
    S = np.bincount(inv, weights=x); N = np.bincount(inv).astype(float)
    W = np.random.default_rng(seed).multinomial(len(u), np.full(len(u), 1 / len(u)), size=n_boot).astype(float)
    bs = (W @ S) / np.maximum(W @ N, 1e-12)
    lo, hi = np.quantile(bs, [q / 2, 1 - q / 2])
    return float(x.mean()), float(lo), float(hi)


def null_z(real_mean: float, nm: np.ndarray) -> tuple[float, float]:
    mu, sd = float(np.nanmean(nm)), float(np.nanstd(nm, ddof=1))
    z = (real_mean - mu) / sd if sd > 0 else np.nan
    return z, (0.5 * math.erfc(z / math.sqrt(2)) if np.isfinite(z) else np.nan)


def evaluate(R: pd.DataFrame, nm: np.ndarray, rp: RParams, seed: int) -> dict:
    if not len(R):
        return {"reversals": 0, "verdict": "NO_EDGE", "G1": "FAIL", "G2": "FAIL"}
    m, lo, hi = month_boot_mean(R.R_rev, R.rev_time, rp.n_boot, seed)
    z, _ = null_z(m, nm)
    p = float((1 + np.sum(nm >= m)) / (1 + len(nm)))          # empirical one-sided p (R is skewed: z is reported only)
    g1 = bool(np.isfinite(lo) and lo > 0); g2 = bool(np.isfinite(p) and p < rp.alpha)
    return {"reversals": int(len(R)), "mean_R_rev": m, "lo95": lo, "hi95": hi, "win_rate": float((R.R_rev > 0).mean()),
            "mean_R_rev_gross": float(R.R_rev_gross.mean()), "sum_R_rev": float(R.R_rev.sum()),
            "median_hold_days": float(R.hold_days.median()), "mean_funding_bp": float(R.funding_bp.mean()),
            "share_closed_by_next_long": float((R.rev_reason == "next_long").mean()),
            "null_mean_R": float(np.nanmean(nm)), "null_sd": float(np.nanstd(nm, ddof=1)), "z": z, "p": p,
            "G1": "PASS" if g1 else "FAIL", "G2": "PASS" if g2 else "FAIL",
            "verdict": "REVERSE_PAYS" if (g1 and g2) else ("BEATS_RANDOM_SHORTS_ONLY" if g2 else "NO_EDGE")}


def event_table(E: pd.DataFrame, rp: RParams) -> pd.DataFrame:
    rows = []
    for kind in ("after_loss", "after_win", "random_flat"):
        for h in HORIZONS_D:
            a = E[(E.kind == kind) & (E.h == h)] if len(E) else E
            m, lo, hi = month_boot_mean(a.atr_units, a.t, rp.n_boot, rp.seed + h)
            sm, slo, shi = month_boot_mean(a.short_R_net, a.t, rp.n_boot, rp.seed + 100 + h)
            rows.append({"kind": kind, "h_days": h, "n": len(a), "mean_move_atr": m, "move_lo95": lo, "move_hi95": hi,
                         "mean_bp": float(a.bp.mean()) if len(a) else np.nan,
                         "share_down": float((a.bp < 0).mean()) if len(a) else np.nan,
                         "short_R_net": sm, "short_lo95": slo, "short_hi95": shi})
    return pd.DataFrame(rows)


def curves(T_all: pd.DataFrame, R_all: pd.DataFrame) -> pd.DataFrame:
    """Daily cumulative R (booked at exit), unit risk per trade: long-only, reverse-only, long + reverse."""
    a = pd.Series(T_all.R.to_numpy(), index=pd.DatetimeIndex(T_all.exit_time))
    b = pd.Series(R_all.R_rev.to_numpy(), index=pd.DatetimeIndex(R_all.rev_exit_time)) if len(R_all) \
        else pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    lo = min(a.index.min(), b.index.min()) if len(b) else a.index.min()
    hi = max(a.index.max(), b.index.max()) if len(b) else a.index.max()
    days = pd.date_range(lo.floor("D"), hi.ceil("D"), freq="D")
    A = a.groupby(a.index.ceil("D")).sum().reindex(days, fill_value=0).cumsum()
    B = b.groupby(b.index.ceil("D")).sum().reindex(days, fill_value=0).cumsum() if len(b) else pd.Series(0.0, index=days)
    return pd.DataFrame({"long_only": A, "reverse_only": B, "long_plus_reverse": A + B})


def max_dd(c: pd.Series) -> float:
    return float((c - c.cummax()).min()) if len(c) else np.nan
