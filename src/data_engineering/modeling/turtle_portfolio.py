"""Turtle Breakout 11: one account across all coins — risk-allocation rules R0..R4, long only and long + reverse.

Protocol: discussions/Turtle_Breakout/013_claude_tb11_preregistration.md (fixed before any real-data output).

Trades: per coin, TB 01 daily longs and (variant long_rev) the TB 07 reverse short at each losing stop exit
(turtle_sizing.trade_list). Entries and exits do not depend on sizing; the rules only set the size (or skip).
One account: equity marked at every 17:00 New York close; sizing uses the last mark (HWM, equity), as TB 01.

Base risk per trade r = f x HWM (f = 2%), units = r / (2 ATR), notional <= 3 x equity per position.
    R0  base risk, no portfolio rule
    R1  heat cap: open risk (sum of risk at entry of open positions) + r <= H x HWM; the new trade takes what is
        left, and is skipped if that is < 25% of r.  H chosen on DISCOVERY from {4, 6, 8, 10, 15, 20, 30}%
    R2  correlation scaling: r / (1 + sum over open positions j of max(0, d_i d_j rho_ij)), rho = trailing 60-day
        correlation of daily close-to-close log returns (>= 30 observations; else the cross-sectional median, else 0.7)
    R3  cluster caps: coins in K = 4 clusters (average linkage on 1 - rho, trailing 90 days, refitted monthly; coins
        with < 60 observations join BTC's cluster); open risk per cluster + r <= C x HWM, same take/skip as R1.
        C chosen on DISCOVERY from {2, 4, 6, 8}%
    R4  walk-forward coin weights: r x m_c, m_c = clip(s_c / median(s), 0, 3), s_c = mean / sd of the coin's trade R
        over trades closed before the quarter starts (>= 10 trades; else m_c = 1; m = 1 if the median <= 0)
Windows: DISCOVERY 2020-01-01 .. 2023-12-31 (choose H, C by Calmar), EVALUATION 2024-01-01 .. end (decision).
Benchmark for the decision: R0m = R0 with f scaled so its mean open risk equals the rule's (same exposure, no
rule), because any rule that cuts exposure improves Calmar mechanically when risk is above the growth-optimal level.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import turtle_breakout as tb

VERSION = "TB11-0.3.0"   # 0.3.0 (TB 14): optional gross-notional cap, faster row access, ann for calmar_paths; defaults reproduce 0.2.0. 0.2.0 (TB 12): size_base "equity" (risk, caps and open risk on the last marked equity); defaults reproduce 0.1.0
RULES = ("R0", "R1", "R2", "R3", "R4")
DISC = (pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"))
EVAL_START = pd.Timestamp("2024-01-01", tz="UTC")


@dataclass(frozen=True)
class PParams:
    f: float = 0.02
    stop_atr: float = 2.0
    pos_cap: float = 3.0
    gross_cap: float | None = None   # TB 14: total entry notional of open positions <= gross_cap x equity (None = off)
    size_base: str = "hwm"          # "hwm" (TB 11) or "equity" (TB 12): base for r, H, C and open risk
    start_equity: float = 100_000.0
    H_grid: tuple = (0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30)
    C_grid: tuple = (0.02, 0.04, 0.06, 0.08)
    K: int = 4
    min_take: float = 0.25
    corr_win: int = 60
    corr_min: int = 30
    corr_fallback: float = 0.7
    clus_win: int = 90
    clus_min: int = 60
    r4_min_trades: int = 10
    r4_max: float = 3.0
    block_days: int = 21
    n_boot: int = 2000
    alpha: float = 0.05
    seed: int = 20261101
    status: str = "PREREGISTERED_TB013"


def registration(pp: PParams | None = None) -> dict:
    d = asdict(pp or PParams())
    d.update(version=VERSION, rules=list(RULES), discovery=[str(x) for x in DISC], evaluation_start=str(EVAL_START),
             selection="H (R1) and C (R3) maximise Calmar on DISCOVERY, per variant; ties -> smaller value",
             decision="EVALUATION: Calmar(rule) - Calmar(R0m) for R1, R2, R3 x {long, long_rev}; R0m = R0 with f scaled "
                      "to the rule's mean open risk in the window (same exposure, no rule); paired 21-day block bootstrap, "
                      "one-sided p, Holm over 6 at alpha 0.05 -> IMPROVES / NO_DIFFERENCE",
             reported=["each rule vs plain R0", "R4", "long_rev vs long under each rule", "equal-weight sleeves", "full period", "4-coin portfolio",
                       "growth, max DD, Sharpe, max open risk, max gross leverage, skipped/scaled entries"])
    return d


# ---------------------------------------------------------------- data: one compact book per coin

@dataclass
class Book:
    name: str
    pid0: int
    close: np.ndarray          # per period (k = pid - pid0): close at the period's last bar, forward-filled
    fcum: np.ndarray           # cumulative funding x price at that bar, forward-filled
    trades: pd.DataFrame       # dir, t_in, t_out, pid_in, pid_out, px_in, px_out, atr, cost, fc_in, fc_out, kind, parent_t, R


def make_book(d: tb.AssetData, L: pd.DataFrame, p: tb.Params) -> Book:
    ok = d.e >= 0
    close = np.where(ok, d.c[np.maximum(d.e, 0)], np.nan)
    fc = np.where(ok, d.fcum[np.maximum(d.e, 0)], np.nan)
    close = pd.Series(close).ffill().to_numpy(); fc = pd.Series(fc).ffill().fillna(0.0).to_numpy()
    T = L.copy().reset_index(drop=True)
    T["t_in"] = d.end[T.je.to_numpy()]; T["t_out"] = d.end[T.jx.to_numpy()]
    T["pid_in"] = tb.period_id(pd.DatetimeIndex(T.t_in), "day"); T["pid_out"] = tb.period_id(pd.DatetimeIndex(T.t_out), "day")
    T["fc_in"] = d.fcum[T.je.to_numpy()]; T["fc_out"] = d.fcum[T.jx.to_numpy()]
    parent = T.t_in.copy()
    for i in range(1, len(T)):
        if T.kind.iat[i] == "reverse":
            parent.iat[i] = T.t_in.iat[i - 1]
    T["parent_t"] = parent
    ret = T.dir * (T.px_out - T.px_in) / T.px_in - T.cost * (1 + T.px_out / T.px_in) - T.dir * (T.fc_out - T.fc_in) / T.px_in
    T["R"] = ret / (p.stop_atr * T.atr / T.px_in)
    keep = ["dir", "t_in", "t_out", "pid_in", "pid_out", "px_in", "px_out", "atr", "cost", "fc_in", "fc_out", "kind", "parent_t", "R"]
    return Book(d.name, int(d.pid0), close, fc, T[keep])


def returns_panel(books: dict) -> pd.DataFrame:
    """Daily log returns of each coin's period close, indexed by global period id."""
    cols = {}
    for a, b in books.items():
        pid = b.pid0 + np.arange(len(b.close))
        cols[a] = pd.Series(np.log(b.close), index=pid).diff()
    return pd.DataFrame(cols).sort_index()


# ---------------------------------------------------------------- rule helpers

def corr_at(R: pd.DataFrame, pid: int, win: int, min_obs: int) -> pd.DataFrame:
    X = R.loc[pid - win:pid - 1]
    return X.corr(min_periods=min_obs)


def average_linkage(C: pd.DataFrame, K: int) -> dict:
    """Agglomerative average linkage on distance 1 - rho; returns {coin: cluster id}."""
    names = list(C.index)
    Dm = 1 - C.to_numpy(float)
    clusters = [[i] for i in range(len(names))]
    while len(clusters) > K:
        best, bi, bj = np.inf, 0, 1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                v = np.nanmean(Dm[np.ix_(clusters[i], clusters[j])])
                if v < best:
                    best, bi, bj = v, i, j
        clusters[bi] = clusters[bi] + clusters[bj]
        del clusters[bj]
    return {names[i]: c for c, members in enumerate(clusters) for i in members}


def clusters_at(R: pd.DataFrame, pid: int, pp: PParams, anchor: str = "BTC") -> dict:
    X = R.loc[pid - pp.clus_win:pid - 1]
    good = [a for a in X.columns if X[a].notna().sum() >= pp.clus_min]
    out = {}
    if len(good) > pp.K:
        out = average_linkage(X[good].corr(min_periods=pp.clus_min).fillna(0.0), pp.K)
    elif good:
        out = {a: i for i, a in enumerate(good)}
    home = out.get(anchor, 0)
    for a in X.columns:
        out.setdefault(a, home)
    return out


def r4_multipliers(books: dict, t_q: pd.Timestamp, pp: PParams) -> dict:
    s = {}
    for a, b in books.items():
        r = b.trades.loc[b.trades.t_out < t_q, "R"].to_numpy()
        if len(r) >= pp.r4_min_trades and r.std(ddof=1) > 0:
            s[a] = r.mean() / r.std(ddof=1)
    med = float(np.median(list(s.values()))) if s else 0.0
    if med <= 0:
        return {a: 1.0 for a in books}
    return {a: float(np.clip(s[a] / med, 0, pp.r4_max)) if a in s else 1.0 for a in books}


# ---------------------------------------------------------------- the joint simulation

def simulate(books: dict, rule: str, pp: PParams, t0: pd.Timestamp, t1: pd.Timestamp | None = None,
             H: float | None = None, C: float | None = None, R: pd.DataFrame | None = None) -> dict:
    """One account over [t0, t1]. Trades enter if t_in >= t0 (a reverse only if its long also entered >= t0) and
    t_in <= t1; positions still open at t1 are closed at the t1 period close (with exit cost)."""
    if rule in ("R2", "R3") and R is None:
        R = returns_panel(books)
    p_lo = int(tb.period_id(pd.DatetimeIndex([t0 + pd.Timedelta(minutes=5)]), "day")[0])
    p_hi_all = max(b.pid0 + len(b.close) - 1 for b in books.values())
    p_hi = p_hi_all if t1 is None else min(p_hi_all, int(tb.period_id(pd.DatetimeIndex([t1]), "day")[0]))
    ev = []
    rows = {a: list(b.trades.itertuples(index=False)) for a, b in books.items()}
    for a, b in books.items():
        T = b.trades
        m = (T.t_in >= t0) & (T.pid_in <= p_hi) & (T.parent_t >= t0)
        for i in np.flatnonzero(m.to_numpy()):
            r = rows[a][i]
            ev.append((r.t_in, 1, a, i, int(r.pid_in)))
            if r.pid_out <= p_hi:
                ev.append((r.t_out, 0 if r.t_out > r.t_in else 2, a, i, int(r.pid_out)))
    ev.sort(key=lambda x: (x[0], x[1]))
    realized, hwm, e_prev = 0.0, pp.start_equity, pp.start_equity
    pos = {}                                   # (a, i) -> dict(u, risk, dir, px_in, fc_in, cost, cluster)
    marks, risk_open, gross = [], [], []
    stats = {"entries": 0, "skipped": 0, "scaled": 0, "ruined_skip": 0}
    clus, clus_month, mult, mult_q, corr_cache = {}, None, {}, None, {}
    gross_open = 0.0
    ei = 0
    for pid in range(p_lo, p_hi + 1):
        pend = tb.period_end(np.array([pid]), "day")[0]
        while ei < len(ev) and ev[ei][4] <= pid:
            t, typ, a, i, _ = ev[ei]; ei += 1
            b = books[a]; r = rows[a][i]
            if typ == 1:
                if e_prev <= 0:
                    stats["ruined_skip"] += 1; continue
                base = hwm if pp.size_base == "hwm" else e_prev
                risk = pp.f * base
                if rule == "R1":
                    open_r = sum(q["risk"] for q in pos.values())
                    avail = H * base - open_r
                    if avail < pp.min_take * risk:
                        stats["skipped"] += 1; continue
                    if avail < risk:
                        risk = avail; stats["scaled"] += 1
                elif rule == "R2":
                    if pos:
                        if pid not in corr_cache:
                            corr_cache = {pid: corr_at(R, pid, pp.corr_win, pp.corr_min)}
                        Cm = corr_cache[pid]
                        vals = Cm.to_numpy()[np.triu_indices(len(Cm), 1)] if len(Cm) > 1 else np.array([])
                        med = float(np.nanmedian(vals)) if np.isfinite(vals).any() else pp.corr_fallback
                        pen = 0.0
                        for (a2, _), q in pos.items():
                            rho = Cm.at[a, a2] if (a in Cm.index and a2 in Cm.columns) else np.nan
                            rho = med if not np.isfinite(rho) else rho
                            if a2 == a:
                                rho = 1.0
                            pen += max(0.0, r.dir * q["dir"] * rho)
                        if pen > 0:
                            risk = risk / (1 + pen); stats["scaled"] += 1
                elif rule == "R3":
                    month = (pend.year, pend.month)
                    if month != clus_month:
                        clus, clus_month = clusters_at(R, pid, pp), month
                    c = clus.get(a, clus.get("BTC", 0))
                    open_c = sum(q["risk"] for (a2, _), q in pos.items() if clus.get(a2, -1) == c)
                    avail = C * base - open_c
                    if avail < pp.min_take * risk:
                        stats["skipped"] += 1; continue
                    if avail < risk:
                        risk = avail; stats["scaled"] += 1
                elif rule == "R4":
                    q_start = pd.Timestamp(year=t.year, month=3 * ((t.month - 1) // 3) + 1, day=1, tz="UTC")
                    if q_start != mult_q:
                        mult, mult_q = r4_multipliers(books, q_start, pp), q_start
                    risk = risk * mult.get(a, 1.0)
                    if risk <= 0:
                        stats["skipped"] += 1; continue
                u = risk / (pp.stop_atr * r.atr)
                u = min(u, pp.pos_cap * e_prev / r.px_in)
                if pp.gross_cap is not None:
                    avail = pp.gross_cap * e_prev - gross_open
                    if avail < pp.min_take * u * r.px_in:
                        stats["skipped"] += 1; continue
                    if avail < u * r.px_in:
                        u = avail / r.px_in; stats["scaled"] += 1
                pos[(a, i)] = {"u": u, "risk": u * pp.stop_atr * r.atr, "dir": int(r.dir), "notional": u * r.px_in}
                gross_open += u * r.px_in
                stats["entries"] += 1
            elif (a, i) in pos:
                q = pos.pop((a, i))
                u = q["u"]; gross_open -= q["notional"]
                realized += r.dir * u * (r.px_out - r.px_in) - r.cost * u * (r.px_in + r.px_out) - r.dir * u * (r.fc_out - r.fc_in)
        unreal = 0.0; g = 0.0
        for (a, i), q in pos.items():
            b = books[a]; r = rows[a][i]
            k = min(max(pid - b.pid0, 0), len(b.close) - 1)
            c, fc = b.close[k], b.fcum[k]
            unreal += r.dir * q["u"] * (c - r.px_in) - r.cost * q["u"] * r.px_in - r.dir * q["u"] * (fc - r.fc_in)
            g += q["u"] * c
        E = pp.start_equity + realized + unreal
        hb = hwm if pp.size_base == "hwm" else E
        marks.append((pend, E)); risk_open.append(sum(q["risk"] for q in pos.values()) / hb if hb > 0 else np.nan)
        gross.append(g / E if E > 0 else np.nan)
        hwm = max(hwm, E); e_prev = E
    Eser = pd.Series([m[1] for m in marks], index=pd.DatetimeIndex([m[0] for m in marks]))
    return {"equity": Eser, "stats": stats, "open_risk": pd.Series(risk_open, index=Eser.index), "max_open_risk": float(np.nanmax(risk_open)) if risk_open else np.nan,
            "mean_open_risk": float(np.nanmean(risk_open)) if risk_open else np.nan,
            "max_gross_lev": float(np.nanmax(gross)) if gross else np.nan}


# ---------------------------------------------------------------- metrics and tests

FLOOR = 1e-6


def daily_curve(E: pd.Series, e0: float) -> pd.Series:
    D = E.resample("1D").last().dropna()
    ruined = (D <= 0).cummax()
    return D.where(~ruined, 0.0).clip(lower=FLOOR * e0)


def metrics(D: pd.Series, e0: float) -> dict:
    lr = np.log(D / e0).diff().fillna(np.log(D.iloc[0] / e0))
    yrs = len(D) / 365.25
    ruined = bool(D.iloc[-1] <= FLOOR * e0 * 1.0001)
    tot = D.iloc[-1] / e0
    cagr = float(tot ** (1 / yrs) - 1) if (yrs > 0 and not ruined) else -1.0
    dd = float((D / np.maximum(D.cummax(), e0) - 1).min())
    r = np.expm1(lr)
    return {"cagr": cagr, "log_growth": float(lr.mean() * 365.25), "max_dd": dd, "calmar": cagr / abs(dd) if dd < 0 else np.nan,
            "sharpe": float(r.mean() / r.std() * math.sqrt(365)) if r.std() > 0 else np.nan, "ruined": ruined,
            "final_multiple": float(tot)}


def calmar_paths(L: np.ndarray, ann: float = 365.25) -> np.ndarray:
    """L: (..., n) daily log returns; Calmar of each path (start at 1). ann = periods per year."""
    cum = np.cumsum(L, axis=-1)
    peak = np.maximum.accumulate(np.maximum(cum, 0.0), axis=-1)
    mdd = np.expm1((cum - peak).min(axis=-1))
    cagr = np.expm1(cum[..., -1] * ann / L.shape[-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(mdd < 0, cagr / np.abs(mdd), np.nan)


def calmar_diff(Da: pd.Series, Db: pd.Series, pp: PParams, seed: int) -> dict:
    """Calmar(a) - Calmar(b), paired circular block bootstrap; one-sided p = P(diff <= 0)."""
    from .turtle_sizing import block_index
    e0 = pp.start_equity
    la = np.log(pd.concat([pd.Series([e0]), Da]).to_numpy()); lb = np.log(pd.concat([pd.Series([e0]), Db]).to_numpy())
    x = np.diff(la); y = np.diff(lb)
    n = min(len(x), len(y)); x, y = x[:n], y[:n]
    est = float(calmar_paths(x[None])[0] - calmar_paths(y[None])[0])
    idx = block_index(n, pp.block_days, pp.n_boot, seed)
    bs = calmar_paths(x[idx]) - calmar_paths(y[idx])
    bs = bs[np.isfinite(bs)]
    p = float((1 + np.sum(bs <= 0)) / (1 + len(bs)))
    return {"calmar_diff": est, "lo95": float(np.quantile(bs, .025)), "hi95": float(np.quantile(bs, .975)), "p": p}


def holm(p) -> np.ndarray:
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[o])); out = np.empty(m); out[o] = adj
    return out


def select(grid_results: pd.DataFrame, key: str) -> float:
    """Highest Calmar on DISCOVERY; ties (to 1e-9) -> smallest parameter. Ruined runs never selected."""
    g = grid_results[~grid_results.ruined].copy()
    if not len(g):
        return float(grid_results[key].min())
    best = g.calmar.max()
    return float(g.loc[g.calmar >= best - 1e-9, key].min())


# ---------------------------------------------------------------- synthetic

def synthetic_books(n_coins: int = 6, days: int = 1500, seed: int = 0, rho: float = 0.7, trend: float = 0.01,
                    with_reverse: bool = False, block_days: int = 20) -> dict:
    """Correlated synthetic coins: one common factor carries shared drift blocks (sign random per block), so
    breakouts cluster across coins; books are built with the real trade-list code."""
    from . import turtle_sizing as ts
    rng = np.random.default_rng(seed)
    names = ["BTC"] + [f"C{i}" for i in range(1, n_coins)]
    n = days * 288
    sig = 0.6 / math.sqrt(365 * 288)
    mu = np.zeros(n)
    if trend:
        for s0 in range(0, n, block_days * 288):
            mu[s0:s0 + block_days * 288] = rng.choice([-1.0, 1.0]) * trend * sig
    common = rng.standard_normal(n)
    end = pd.date_range(pd.Timestamp("2021-06-01", tz="UTC") + pd.Timedelta(minutes=5), periods=n, freq="5min")
    P = tb.Params(); out = {}
    for j, a in enumerate(names):
        e = math.sqrt(rho) * common + math.sqrt(1 - rho) * rng.standard_normal(n)
        r = mu + sig * e
        c = math.log(100.0) + np.cumsum(r); o = np.r_[math.log(100.0), c[:-1]]
        wig = np.abs(rng.normal(0, sig / 2, n))
        b = pd.DataFrame({"open": np.exp(o), "high": np.exp(np.maximum(o, c) + wig), "low": np.exp(np.minimum(o, c) - wig),
                          "close": np.exp(c)}, index=pd.DatetimeIndex(end, name="end"))
        d = tb.prepare_asset(a, b, tb.synthetic_funding(b, 0.0), P)
        L = ts.trade_list(d, P, end[0] + pd.Timedelta(days=30), with_reverse=with_reverse)
        out[a] = make_book(d, L, P)
    return out
