"""Turtle Breakout 01: Richard Dennis' turtle breakout (Rallio's simplified version) on crypto perpetuals.

Protocol: discussions/Turtle_Breakout/001_claude_tb01_preregistration.md (fixed before any real-data output).

Rules (Rallio Trading's public description, as reviewed in the Algo Vibes video Ivan supplied)
---------------------------------------------------------------------------------------------
Trading day = 17:00 -> 17:00 America/New_York. Levels move once per day and use COMPLETED days only.
    entry      long when price trades above the highest high of the last 20 days (stop order, filled
               at max(level, bar open))
    stop 1     fixed: entry - 2 * ATR_20 (Wilder average of the daily true range); never moves
    stop 2     trailing: lowest low of the last 10 days, recalculated each day
    exit       whichever stop is higher is hit (filled at min(stop, bar open))
    no take-profit, no pyramiding, one position per market
    size       risk 2% of the account's highest equity (high-water mark): units = 0.02 HWM / (2 ATR)
Primary: long only (Rallio). Reported: short only, long + short (original turtles), 1-hour version (20 / 10 /
20 hours instead of days). Rallio's 'day-trading' variant (flat 16:30-18:00 NY) exists to avoid CFD swap at
17:00; perpetual funding is charged at 00/08/16 UTC, so it has no purpose here and is not tested.

Instrument and costs: Binance USD-M perpetual 1m klines aggregated to 5-minute bars (intraday fills),
taker fee 0.05% + 1 bp slippage per side (Deribit perpetual schedule, as BT 001), Binance funding rate
applied to the held notional at every funding time (long pays positive funding).

Nulls (both as in the reviewed video)
    N1 trade twins  each real trade re-drawn at a random 5m bar of the same asset, same side, same holding
                    time, same costs; 2,000 runs; share of runs whose summed net return per unit notional
                    is >= the real trades'
    N2 random portfolio  same exits, 2% HWM sizing, but the entry is a coin: on each eligible flat day enter
                    at the day's open with probability = the real run's entry rate for that asset;
                    1,000 runs; share with CAGR >= real
Benchmark: buy-and-hold of the same assets, equal notional, at the strategy's mean gross leverage, with funding.

Decision (primary = daily, long-only, 4-asset portfolio, window from COMMON_START):
    G1 beats luck          N2 share of random portfolios with CAGR >= real < 0.05
    G2 beats doing nothing Sharpe(strategy) - Sharpe(buy-and-hold) > 0 with 21-day block-bootstrap q2.5 > 0
    verdict TREND_EDGE (G1 & G2) / TIMING_ONLY (G1) / RISK_PROFILE_ONLY (G2) / NO_EDGE
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

VERSION = "TB01-0.1.0"
ASSETS = ("BTC", "ETH", "SOL", "XRP")
NY = "America/New_York"
COMMON_START = pd.Timestamp("2020-10-15", tz="UTC")     # all four perps listed + warm-up (SOLUSDT perp 2020-09)
HALF_SPLIT = pd.Timestamp("2023-07-01", tz="UTC")
BAR_MIN = 5


@dataclass(frozen=True)
class Params:
    period: str = "day"            # "day" (17:00 NY) or "hour"
    side: str = "long"             # "long", "short", "both"
    entry_n: int = 20
    exit_n: int = 10
    atr_n: int = 20
    stop_atr: float = 2.0
    risk_frac: float = 0.02
    max_pos_lev: float = 3.0       # notional per position <= 3 x last marked equity (binds only for tiny ATR)
    start_equity: float = 100_000.0
    fee: float = 0.0005
    slip: float = 0.0001
    funding: bool = True
    min_bar_share: float = 0.8     # a period needs >= 80% of its 5m bars to allow a new entry
    n_twin: int = 2000
    n_null: int = 1000
    block_days: int = 21
    n_boot: int = 2000
    alpha: float = 0.05
    seed: int = 19831
    status: str = "PREREGISTERED_TB001"


def registration(p: Params | None = None) -> dict:
    p = p or Params()
    d = asdict(p)
    d.update(version=VERSION, assets=list(ASSETS), common_start=str(COMMON_START), half_split=str(HALF_SPLIT),
             trading_day="17:00-17:00 America/New_York", atr="Wilder RMA of true range",
             leverage_cap="position notional <= max_pos_lev x last marked equity; no entries at equity <= 0",
             primary="period=day, side=long, 4-asset portfolio, 2% HWM risk, fees+slippage+funding",
             gates={"G1": "N2 random-entry portfolios with CAGR >= real: share < 0.05",
                    "G2": "Sharpe(strategy) - Sharpe(B&H at mean leverage): 21d block bootstrap q2.5 > 0"},
             reported=["short", "both", "hour", "no funding", "per asset N1", "halves", "years"])
    return d


# ---------------------------------------------------------------- data

def bars_5m(df1m: pd.DataFrame) -> pd.DataFrame:
    """1m klines (open_time = bar open, UTC) -> 5m OHLC indexed by bar END. Missing minutes are not filled."""
    d = df1m[["open_time", "open", "high", "low", "close"]].copy()
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d = d.drop_duplicates("open_time").sort_values("open_time")
    g = d.groupby(d["open_time"].dt.floor(f"{BAR_MIN}min"))
    b = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last()})
    b.index = b.index + pd.Timedelta(minutes=BAR_MIN)
    b.index.name = "end"
    return b.astype(float)


def funding_events(f: pd.DataFrame) -> pd.DataFrame:
    """Binance fundingRate files: calc_time (ms), last_funding_rate."""
    if f is None or not len(f):
        return pd.DataFrame(columns=["t", "rate"])
    t = pd.to_numeric(f["calc_time"], errors="coerce")
    t = pd.to_datetime(t, unit="us" if t.max() > 10 ** 14 else "ms", utc=True)
    out = pd.DataFrame({"t": t, "rate": pd.to_numeric(f["last_funding_rate"], errors="coerce")}).dropna()
    return out.drop_duplicates("t").sort_values("t").reset_index(drop=True)


def period_id(end: pd.DatetimeIndex, period: str) -> np.ndarray:
    """Integer period of each bar (bar END). Day d = (17:00 NY of day d-1, 17:00 NY of day d]."""
    if period == "day":
        loc = (end - pd.Timedelta(seconds=1)).tz_convert(NY).tz_localize(None)
        return ((loc - pd.Timestamp("1970-01-01 17:00")) // pd.Timedelta(days=1)).to_numpy().astype(np.int64)
    if period == "hour":
        return np.asarray((end - pd.Timedelta(seconds=1) - pd.Timestamp("1970-01-01", tz="UTC"))
                          // pd.Timedelta(hours=1)).astype(np.int64)
    raise ValueError(period)


def period_end(pid: np.ndarray, period: str) -> pd.DatetimeIndex:
    if period == "day":
        loc = pd.Timestamp("1970-01-02 17:00") + pd.to_timedelta(pid, unit="D")
        return pd.DatetimeIndex(loc).tz_localize(NY, ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")
    return pd.to_datetime((pid + 1) * 3600, unit="s", utc=True)


def _rma(x: np.ndarray, n: int) -> np.ndarray:
    """Wilder RMA, seeded with the mean of the first n finite values; NaN gaps carry the last value."""
    out = np.full(len(x), np.nan)
    ok = np.flatnonzero(np.isfinite(x))
    if len(ok) < n:
        return out
    start = ok[n - 1]
    v = np.nanmean(x[ok[:n]])
    out[start] = v
    for i in range(start + 1, len(x)):
        if np.isfinite(x[i]):
            v = v + (x[i] - v) / n
        out[i] = v
    return out


@dataclass
class AssetData:
    name: str
    end: pd.DatetimeIndex          # 5m bar ends
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    fcum: np.ndarray               # cumulative sum of funding rate x price, per bar (inclusive)
    pid0: int                      # first period id
    s: np.ndarray                  # per period (index k = pid - pid0): first bar, last bar (-1 if empty)
    e: np.ndarray
    full: np.ndarray               # enough bars for a new entry
    ph: np.ndarray
    pl: np.ndarray
    po: np.ndarray
    pc: np.ndarray
    hi_n: np.ndarray               # levels known at the start of period k (completed periods only)
    lo_n: np.ndarray
    lo_x: np.ndarray
    hi_x: np.ndarray
    atr: np.ndarray
    first_up: np.ndarray           # first bar in period k with high >= hi_n[k] (-1 if none)
    first_dn: np.ndarray
    smin: np.ndarray               # per bar: min low from the bar to the end of its period
    smax: np.ndarray


def prepare_asset(name: str, b5: pd.DataFrame, fund: pd.DataFrame | None, p: Params) -> AssetData:
    end = pd.DatetimeIndex(b5.index)
    o, h, l, c = (b5[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    n = len(c)
    fr = np.zeros(n)
    if fund is not None and len(fund) and p.funding:
        j = end.searchsorted(pd.DatetimeIndex(fund["t"]), side="left")       # first bar ending at/after the event
        ok = j < n
        np.add.at(fr, j[ok], fund["rate"].to_numpy()[ok] * c[j[ok]])
    fcum = np.cumsum(fr)
    pid = period_id(end, p.period)
    pid0 = int(pid.min()); K = int(pid.max()) - pid0 + 1
    k = pid - pid0
    s = np.full(K, -1); e = np.full(K, -1)
    first = np.r_[True, k[1:] != k[:-1]]
    last = np.r_[k[1:] != k[:-1], True]
    s[k[first]] = np.flatnonzero(first); e[k[last]] = np.flatnonzero(last)
    cnt = np.bincount(k, minlength=K)
    expected = (1440 if p.period == "day" else 60) // BAR_MIN
    full = cnt >= p.min_bar_share * expected
    ph = np.full(K, np.nan); pl = np.full(K, np.nan); po = np.full(K, np.nan); pc = np.full(K, np.nan)
    has = s >= 0
    ph[has] = np.maximum.reduceat(h, s[has]); pl[has] = np.minimum.reduceat(l, s[has])
    po[has] = o[s[has]]; pc[has] = c[e[has]]
    pcs = pd.Series(pc).ffill().to_numpy()
    prev_c = np.r_[np.nan, pcs[:-1]]
    tr = np.where(has, np.nanmax(np.c_[ph - pl, np.abs(ph - prev_c), np.abs(pl - prev_c)], axis=1), np.nan)
    tr[has & ~np.isfinite(prev_c)] = (ph - pl)[has & ~np.isfinite(prev_c)]
    atr_c = _rma(tr, p.atr_n)                                  # through period k
    roll = lambda x, m, f: getattr(pd.Series(x).rolling(m, min_periods=m), f)().to_numpy()
    shift = lambda x: np.r_[np.nan, x[:-1]]
    hi_n = shift(roll(ph, p.entry_n, "max")); lo_n = shift(roll(pl, p.entry_n, "min"))
    lo_x = shift(roll(pl, p.exit_n, "min")); hi_x = shift(roll(ph, p.exit_n, "max"))
    atr = shift(atr_c)
    # first breakout bar per period
    kb = k
    up = np.flatnonzero(h >= hi_n[kb]); dn = np.flatnonzero(l <= lo_n[kb])
    first_up = np.full(K, -1); first_dn = np.full(K, -1)
    if len(up):
        u = pd.Series(up).groupby(kb[up]).min(); first_up[u.index.to_numpy()] = u.to_numpy()
    if len(dn):
        u = pd.Series(dn).groupby(kb[dn]).min(); first_dn[u.index.to_numpy()] = u.to_numpy()
    smin = pd.Series(l[::-1]).groupby(k[::-1]).cummin().to_numpy()[::-1]
    smax = pd.Series(h[::-1]).groupby(k[::-1]).cummax().to_numpy()[::-1]
    return AssetData(name, end, o, h, l, c, fcum, pid0, s, e, full, ph, pl, po, pc, hi_n, lo_n, lo_x, hi_x, atr,
                     first_up, first_dn, smin, smax)


# ---------------------------------------------------------------- portfolio simulation

TRADE_COLS = ["asset", "dir", "kind", "je", "jx", "entry_px", "exit_px", "units", "pnl", "fees", "funding",
              "ret_notional", "reason", "entry_time", "exit_time", "hold_bars"]


def _stop_level(pos, d: AssetData, k: int) -> float:
    trail = d.lo_x[k] if pos["dir"] == 1 else d.hi_x[k]
    if not np.isfinite(trail):
        return pos["fixed"]
    return max(pos["fixed"], trail) if pos["dir"] == 1 else min(pos["fixed"], trail)


def _first_hit(d: AssetData, a: int, b: int, dirn: int, stop: float) -> int:
    seg = d.l[a:b + 1] <= stop if dirn == 1 else d.h[a:b + 1] >= stop
    return a + int(np.argmax(seg)) if seg.any() else -1


def simulate(data: dict, p: Params, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
             mode: str = "rule", p_entry: dict | None = None, rng: np.random.Generator | None = None,
             record: bool = True):
    """Joint portfolio simulation over periods. Returns (equity Series by period end, trades DataFrame, stats)."""
    names = list(data)
    pid_lo = min(d.pid0 for d in data.values())
    pid_hi = max(d.pid0 + len(d.s) - 1 for d in data.values())
    if start is not None:
        pid_lo = max(pid_lo, int(period_id(pd.DatetimeIndex([start + pd.Timedelta(minutes=BAR_MIN)]), p.period)[0]))
    if end is not None:
        pid_hi = min(pid_hi, int(period_id(pd.DatetimeIndex([end]), p.period)[0]))
    cost = p.fee + p.slip
    sides = {"long": (1,), "short": (-1,), "both": (1, -1)}[p.side]
    realized = 0.0
    hwm = p.start_equity
    e_prev = p.start_equity
    pos = {a: None for a in names}
    eq = np.full(pid_hi - pid_lo + 1, np.nan)
    gross = np.zeros(pid_hi - pid_lo + 1)
    trades = []
    elig = {a: 0 for a in names}; entries = {a: 0 for a in names}

    def close(a, d, P, jx, px, reason):
        nonlocal realized
        u, dirn = P["units"], P["dir"]
        fund = dirn * u * (d.fcum[jx] - d.fcum[P["je"]])
        fees = cost * u * (P["px"] + px)
        pnl = dirn * u * (px - P["px"]) - fees - fund
        realized += pnl
        if record:
            trades.append((a, dirn, P["kind"], P["je"], jx, P["px"], px, u, pnl, fees, fund,
                           pnl / (u * P["px"]), reason, d.end[P["je"]], d.end[jx], jx - P["je"]))

    def open_(a, d, k, j, px, dirn, kind):
        atr = d.atr[k]
        u = min(p.risk_frac * hwm / (p.stop_atr * atr), p.max_pos_lev * e_prev / px)
        P = {"dir": dirn, "je": j, "px": px, "units": u, "kind": kind,
             "fixed": px - dirn * p.stop_atr * atr}
        pos[a] = P
        entries[a] += 1
        # same-period stop after the entry bar (entry bar itself excluded unless entered at the open)
        st = _stop_level(P, d, k)
        a0 = j if kind == "random" else j + 1
        if a0 <= d.e[k]:
            sm = d.smin[a0] if dirn == 1 else d.smax[a0]
            if (dirn == 1 and sm <= st) or (dirn == -1 and sm >= st):
                jx = _first_hit(d, a0, d.e[k], dirn, st)
                fill = min(st, d.o[jx]) if dirn == 1 else max(st, d.o[jx])
                if jx == j:
                    fill = st
                close(a, d, P, jx, fill, "stop_same_period")
                pos[a] = None

    for t, pid in enumerate(range(pid_lo, pid_hi + 1)):
        for a in names:
            d = data[a]
            k = pid - d.pid0
            if k < 0 or k >= len(d.s) or d.s[k] < 0:
                continue
            P = pos[a]
            exited = False
            if P is not None:
                P["jlast"] = d.e[k]
                st = _stop_level(P, d, k)
                if (P["dir"] == 1 and d.pl[k] <= st) or (P["dir"] == -1 and d.ph[k] >= st):
                    jx = _first_hit(d, d.s[k], d.e[k], P["dir"], st)
                    fill = min(st, d.o[jx]) if P["dir"] == 1 else max(st, d.o[jx])
                    reason = "trail" if st != P["fixed"] else "fixed_stop"
                    close(a, d, P, jx, fill, reason)
                    pos[a] = None
                    exited = True
            if pos[a] is None and not exited and e_prev > 0 and d.full[k] and np.isfinite(d.atr[k]) and d.atr[k] > 0 \
                    and np.isfinite(d.hi_n[k]) and np.isfinite(d.lo_x[k]):
                elig[a] += 1
                if mode == "rule":
                    cand = []
                    if 1 in sides and d.first_up[k] >= 0:
                        cand.append((d.first_up[k], 1))
                    if -1 in sides and d.first_dn[k] >= 0:
                        cand.append((d.first_dn[k], -1))
                    if cand:
                        j, dirn = min(cand)
                        lvl = d.hi_n[k] if dirn == 1 else d.lo_n[k]
                        px = max(lvl, d.o[j]) if dirn == 1 else min(lvl, d.o[j])
                        open_(a, d, k, j, px, dirn, "breakout")
                else:
                    if rng.random() < p_entry[a]:
                        dirn = sides[0] if len(sides) == 1 else (1 if rng.random() < 0.5 else -1)
                        open_(a, d, k, d.s[k], d.o[d.s[k]], dirn, "random")
        # mark to market at the period close
        unreal = 0.0; g = 0.0
        for a in names:
            P = pos[a]
            if P is None:
                continue
            d = data[a]
            k = pid - d.pid0
            if 0 <= k < len(d.e) and d.e[k] >= 0:
                P["jlast"] = d.e[k]
            j = max(P.get("jlast", P["je"]), P["je"])
            unreal += P["dir"] * P["units"] * (d.c[j] - P["px"]) - cost * P["units"] * P["px"] \
                - P["dir"] * P["units"] * (d.fcum[j] - d.fcum[P["je"]])
            g += P["units"] * d.c[j]
        E = p.start_equity + realized + unreal
        eq[t] = E
        gross[t] = g / E if E > 0 else np.nan
        hwm = max(hwm, E)
        e_prev = E
    # close open positions at the last bar (reason 'end')
    for a in names:
        P = pos[a]
        if P is not None:
            d = data[a]
            k = pid_hi - d.pid0
            jx = d.e[min(k, len(d.e) - 1)]
            jx = max(jx if jx >= 0 else len(d.c) - 1, P["je"])
            close(a, d, P, jx, d.c[jx], "end")
    idx = period_end(np.arange(pid_lo, pid_hi + 1), p.period)
    E = pd.Series(eq, index=idx).dropna()
    T = pd.DataFrame(trades, columns=TRADE_COLS)
    stats = {"eligible": elig, "entries": entries,
             "p_entry": {a: entries[a] / elig[a] if elig[a] else 0.0 for a in names},
             "mean_gross_leverage": float(np.nanmean(gross)), "max_gross_leverage": float(np.nanmax(gross)),
             "gross": pd.Series(gross, index=idx)}
    return E, T, stats


# ---------------------------------------------------------------- metrics

def daily(E: pd.Series) -> pd.Series:
    return E.resample("1D").last().dropna()


def metrics(E: pd.Series, T: pd.DataFrame | None = None) -> dict:
    D = daily(E)
    if len(D) < 3:
        return {}
    yrs = (D.index[-1] - D.index[0]).total_seconds() / (365.25 * 86400)
    tot = D.iloc[-1] / D.iloc[0]
    r = D.pct_change().dropna()
    dd = (D / D.cummax() - 1).min()
    out = {"start": str(D.index[0].date()), "end": str(D.index[-1].date()), "years": yrs,
           "final_equity": float(D.iloc[-1]), "total_return": float(tot - 1),
           "cagr": (float(tot ** (1 / yrs) - 1) if tot > 0 else -1.0) if yrs > 0 else np.nan,
           "sharpe": float(r.mean() / r.std() * math.sqrt(365)) if r.std() > 0 else np.nan,
           "max_dd": float(dd)}
    if T is not None and len(T):
        w, ls = T.pnl[T.pnl > 0].sum(), -T.pnl[T.pnl <= 0].sum()
        out.update(trades=int(len(T)), win_rate=float((T.pnl > 0).mean()),
                   profit_factor=float(w / ls) if ls > 0 else np.nan,
                   avg_hold_days=float(T.hold_bars.mean() * BAR_MIN / 1440),
                   fees=float(T.fees.sum()), funding=float(T.funding.sum()))
    return out


def yearly(E: pd.Series) -> pd.Series:
    D = daily(E)
    y = D.groupby(D.index.year).last()
    prev = pd.concat([pd.Series([D.iloc[0]], index=[y.index[0] - 1]), y]).shift(1).iloc[1:]
    return (y / prev - 1).rename("return")


def buy_and_hold(data: dict, p: Params, start: pd.Timestamp, end: pd.Timestamp | None, leverage: float) -> pd.Series:
    """Equal notional per asset at start (total = leverage x equity), never rebalanced, with entry cost and funding."""
    names = list(data)
    eq = None
    E0 = p.start_equity
    for a in names:
        d = data[a]
        j0 = int(d.end.searchsorted(start, side="left"))
        j1 = len(d.c) - 1 if end is None else int(d.end.searchsorted(end, side="right")) - 1
        px0 = d.c[j0]
        u = leverage * E0 / len(names) / px0
        pnl = u * (d.c[j0:j1 + 1] - px0) - (p.fee + p.slip) * u * px0 - u * (d.fcum[j0:j1 + 1] - d.fcum[j0])
        s = pd.Series(pnl, index=d.end[j0:j1 + 1]).resample("1D").last()
        eq = s if eq is None else eq.add(s, fill_value=np.nan)
    return (E0 + eq.ffill()).dropna()


def sharpe_diff(Ea: pd.Series, Eb: pd.Series, p: Params, seed: int = 0) -> dict:
    a = daily(Ea).pct_change(); b = daily(Eb).pct_change()
    x = pd.concat([a, b], axis=1).dropna().to_numpy()
    n = len(x)
    sh = lambda z: z.mean(0) / z.std(0) * math.sqrt(365)
    est = sh(x)
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / p.block_days))
    bs = np.empty(p.n_boot)
    for i in range(p.n_boot):
        st = rng.integers(0, n - p.block_days + 1, nb)
        idx = (st[:, None] + np.arange(p.block_days)).ravel()[:n]
        s = sh(x[idx]); bs[i] = s[0] - s[1]
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return {"sharpe_strategy": float(est[0]), "sharpe_bh": float(est[1]), "diff": float(est[0] - est[1]),
            "lo": float(lo), "hi": float(hi)}


# ---------------------------------------------------------------- nulls

def trade_twins(data: dict, T: pd.DataFrame, p: Params, seed: int = 0) -> pd.DataFrame:
    """N1: per asset, each real trade re-drawn at a random bar, same side and holding bars; net return per unit
    notional (close to close, fees+slippage both sides, funding). Returns real vs null summary per asset + ALL."""
    rng = np.random.default_rng(seed)
    cost = p.fee + p.slip
    rows, null_all, real_all = [], np.zeros(p.n_twin), 0.0
    for a, g in T[T.reason != "end"].groupby("asset"):
        d = data[a]
        n = len(d.c)
        H = g.hold_bars.to_numpy(); dirn = g.dir.to_numpy()
        real = float(g.ret_notional.sum())
        S = np.zeros(p.n_twin)
        for hb, dr in zip(H, dirn):
            j0 = rng.integers(0, n - hb - 1, p.n_twin)
            j1 = j0 + hb
            r = dr * (d.c[j1] / d.c[j0] - 1) - cost * (1 + d.c[j1] / d.c[j0]) - dr * (d.fcum[j1] - d.fcum[j0]) / d.c[j0]
            S += r
        rows.append({"asset": a, "trades": int(len(g)), "real_sum_ret": real, "null_median": float(np.median(S)),
                     "null_q95": float(np.quantile(S, 0.95)), "share_null_ge_real": float((S >= real).mean())})
        null_all += S; real_all += real
    rows.append({"asset": "ALL", "trades": int((T.reason != "end").sum()), "real_sum_ret": real_all,
                 "null_median": float(np.median(null_all)), "null_q95": float(np.quantile(null_all, 0.95)),
                 "share_null_ge_real": float((null_all >= real_all).mean())})
    return pd.DataFrame(rows)


def random_portfolios(data: dict, p: Params, p_entry: dict, start, end=None, n: int | None = None,
                      seed: int = 0) -> pd.DataFrame:
    """N2: random-entry portfolios with the turtle exits and sizing."""
    n = n or p.n_null
    ss = np.random.SeedSequence(seed)
    out = []
    for i, child in enumerate(ss.spawn(n)):
        E, _, st = simulate(data, p, start, end, mode="random", p_entry=p_entry,
                            rng=np.random.default_rng(child), record=False)
        m = metrics(E)
        out.append({"run": i, "cagr": m.get("cagr", np.nan), "sharpe": m.get("sharpe", np.nan),
                    "max_dd": m.get("max_dd", np.nan), "final_equity": m.get("final_equity", np.nan),
                    "mean_gross_leverage": st["mean_gross_leverage"]})
    return pd.DataFrame(out)


def decide(real: dict, null: pd.DataFrame, sd: dict, p: Params) -> dict:
    share = float((null.cagr >= real["cagr"]).mean())
    g1 = "PASS" if share < p.alpha else "FAIL"
    g2 = "PASS" if sd["lo"] > 0 else "FAIL"
    v = ("TREND_EDGE" if g1 == g2 == "PASS" else "TIMING_ONLY" if g1 == "PASS"
         else "RISK_PROFILE_ONLY" if g2 == "PASS" else "NO_EDGE")
    return {"version": VERSION, "status": p.status, "real_cagr": real["cagr"], "real_sharpe": real["sharpe"],
            "real_max_dd": real["max_dd"], "null_cagr_median": float(null.cagr.median()),
            "null_share_cagr_ge_real": share, **{f"sharpe_{k}": v_ for k, v_ in sd.items()},
            "G1_beats_luck": g1, "G2_beats_buy_and_hold": g2, "verdict": v}


# ---------------------------------------------------------------- synthetic

def synthetic_5m(n_days: int = 900, seed: int = 0, start: str = "2020-01-01", vol: float = 0.6,
                 trend: float = 0.0, block_days: int = 20) -> pd.DataFrame:
    """5m OHLC (index = bar END). trend > 0 adds persistent drift blocks (sign random per block) - a trending
    market where breakouts should beat random entries; trend = 0 is a driftless random walk."""
    rng = np.random.default_rng(seed)
    n = n_days * 288
    sig = vol / math.sqrt(365 * 288)
    mu = np.zeros(n)
    if trend:
        for s0 in range(0, n, block_days * 288):
            mu[s0:s0 + block_days * 288] = rng.choice([-1.0, 1.0]) * trend * sig
    sub = 4
    r = rng.normal(mu[:, None] / sub, sig / math.sqrt(sub), size=(n, sub))
    path = math.log(100.0) + np.cumsum(r.ravel()).reshape(n, sub)
    o = np.r_[math.log(100.0), path[:-1, -1]]
    end = pd.date_range(pd.Timestamp(start, tz="UTC") + pd.Timedelta(minutes=5), periods=n, freq="5min")
    return pd.DataFrame({"open": np.exp(o), "high": np.exp(np.maximum(path.max(1), o)),
                         "low": np.exp(np.minimum(path.min(1), o)), "close": np.exp(path[:, -1])},
                        index=pd.DatetimeIndex(end, name="end"))


def synthetic_funding(b5: pd.DataFrame, rate: float = 0.0001) -> pd.DataFrame:
    t = pd.date_range(b5.index[0].ceil("8h"), b5.index[-1], freq="8h")
    return pd.DataFrame({"t": t, "rate": rate})
