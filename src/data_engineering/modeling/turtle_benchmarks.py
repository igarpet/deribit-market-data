"""Turtle Breakout 02: does the TB 01 turtle portfolio beat PROPER buy-and-hold benchmarks?

Protocol: discussions/Turtle_Breakout/003_claude_tb02_preregistration.md (fixed before any real-data output).

TB 01's G2 benchmark (equal notional, never rebalanced, perpetuals with funding) was distorted: SOL's notional
grew ~80x in 2021 and paid funding on it, and at 1x leverage the benchmark's equity went below zero in Nov 2022.
TB 02 replaces it with constant-mix benchmarks. The strategy is the unchanged TB 01 primary (turtle_breakout,
Params() defaults, window COMMON_START -> end of data); it is re-simulated deterministically.

Benchmarks (equal weight across BTC/ETH/SOL/XRP, total notional = L x equity, marked at the daily 17:00 NY close)
    B1 perp, monthly rebalance   funding paid on the held notional, taker + slippage on every rebalance trade
    B2 perp, daily rebalance     same, rebalanced every day
    B3 spot proxy, monthly       perpetual price path WITHOUT funding (spot-perp basis ~1 bp, Pairs fOU 004), same
                                 trading costs; idle cash earns nothing (as for the strategy)
    L = the strategy's mean gross leverage (TB 01: 0.41). Sharpe of a constant-mix portfolio is (almost) leverage-
    free, so the Sharpe tests do not depend on this choice; CAGR / drawdown comparisons do.
Reported only
    B4 inverse-volatility weights (60-day realised vol, monthly), perp with funding
    B1 at the strategy's realised volatility (leverage solved so the daily vol matches): CAGR and drawdown at equal risk
    alpha / beta: OLS of strategy daily returns on B1 (L = 1) daily returns, Newey-West (5 lags) t of alpha
    years, halves, rolling 365-day Sharpe

Decision (intersection-union test: each comparison at 95%, no multiplicity correction needed because ALL must pass)
    For each of B1, B2, B3: dSR = Sharpe(strategy) - Sharpe(B), daily returns, 21-day circular block bootstrap
    (2,000), pass if q2.5 > 0.
    BEATS_ALL_BENCHMARKS if B1, B2 and B3 all pass; otherwise FAILS_<names>.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import turtle_breakout as tb

VERSION = "TB02-0.1.0"


@dataclass(frozen=True)
class BParams:
    vol_window_days: int = 60
    nw_lags: int = 5
    block_days: int = 21
    n_boot: int = 2000
    seed: int = 20261025
    status: str = "PREREGISTERED_TB003"


def registration(bp: BParams | None = None) -> dict:
    d = asdict(bp or BParams())
    d.update(version=VERSION, strategy="TB 01 primary, unchanged (turtle_breakout Params defaults)",
             gated=["B1 perp monthly", "B2 perp daily", "B3 spot-proxy monthly"],
             reported=["B4 inverse-vol perp monthly", "B1 at equal realised vol", "alpha/beta vs B1 (L=1)",
                       "years", "halves", "rolling 365d Sharpe"],
             rule="each gated benchmark: Sharpe diff 21d block bootstrap q2.5 > 0; verdict needs all three")
    return d


def daily_panel(data: dict, start: pd.Timestamp, end: pd.Timestamp | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily (17:00 NY) close and cumulative funding (rate x price) per asset, aligned on period ends."""
    closes, funds = {}, {}
    for a, d in data.items():                       # daily 17:00 NY closes, whatever the strategy's own timeframe
        dpid = tb.period_id(d.end, "day")
        last = np.r_[dpid[1:] != dpid[:-1], True]
        idx = tb.period_end(dpid[last], "day")
        closes[a] = pd.Series(d.c[last], index=idx)
        funds[a] = pd.Series(d.fcum[last], index=idx)
    C = pd.DataFrame(closes).sort_index()
    F = pd.DataFrame(funds).sort_index().ffill()
    m = C.index >= start
    if end is not None:
        m &= C.index <= end
    C, F = C[m], F[m]
    C = C.ffill()
    return C, F


def constant_mix(C: pd.DataFrame, F: pd.DataFrame, leverage: float, rebalance: str, funding: bool,
                 cost: float, weights: str = "equal", vol_window: int = 60, E0: float = 100_000.0) -> pd.Series:
    """Constant-mix portfolio. rebalance: 'D' (every day) or 'M' (first day of each calendar month, NY date)."""
    P = C.to_numpy(float); Fc = F.to_numpy(float)
    n, m = P.shape
    lr = np.log(C).diff()
    vol = lr.rolling(vol_window, min_periods=vol_window).std().shift(0).to_numpy()
    month = C.index.tz_convert(tb.NY).month
    units = np.zeros(m)
    E = np.empty(n)
    eq = E0
    for t in range(n):
        if t > 0:
            dp = P[t] - P[t - 1]
            df = (Fc[t] - Fc[t - 1]) if funding else np.zeros(m)
            pnl = units * np.nan_to_num(dp) - (units * np.nan_to_num(df) if funding else 0.0)
            eq = eq + float(pnl.sum())
        reb = t == 0 or rebalance == "D" or month[t] != month[t - 1]
        if reb and eq > 0:
            if weights == "invvol" and t > 0 and np.isfinite(vol[t]).all():
                w = 1.0 / vol[t]; w = w / w.sum()
            else:
                w = np.full(m, 1.0 / m)
            target = leverage * eq * w / P[t]
            eq -= cost * float(np.abs(target - units) @ P[t])
            units = target
        E[t] = eq
    return pd.Series(E, index=C.index)


def realised_vol(E: pd.Series) -> float:
    return float(tb.daily(E).pct_change().std() * math.sqrt(365))


def vol_matched(C, F, target_vol: float, cost: float, L0: float = 1.0) -> tuple[pd.Series, float]:
    L = L0
    for _ in range(4):
        S = constant_mix(C, F, L, "M", True, cost)
        v = realised_vol(S)
        if not np.isfinite(v) or v <= 0:
            break
        L = L * target_vol / v
    return constant_mix(C, F, L, "M", True, cost), L


def sharpe_diff(Ea: pd.Series, Eb: pd.Series, bp: BParams, seed: int) -> dict:
    a = tb.daily(Ea).pct_change(); b = tb.daily(Eb).pct_change()
    x = pd.concat([a, b], axis=1).dropna().to_numpy()
    n = len(x)
    sh = lambda z: z.mean(0) / z.std(0) * math.sqrt(365)
    est = sh(x)
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / bp.block_days))
    bs = np.empty(bp.n_boot)
    for i in range(bp.n_boot):
        st = rng.integers(0, n, nb)
        idx = ((st[:, None] + np.arange(bp.block_days)) % n).ravel()[:n]       # circular blocks
        s = sh(x[idx]); bs[i] = s[0] - s[1]
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return {"sharpe_strategy": float(est[0]), "sharpe_benchmark": float(est[1]), "diff": float(est[0] - est[1]),
            "lo": float(lo), "hi": float(hi), "pass": bool(lo > 0), "days": int(n)}


def alpha_beta(Es: pd.Series, Eb: pd.Series, lags: int = 5) -> dict:
    a = tb.daily(Es).pct_change(); b = tb.daily(Eb).pct_change()
    z = pd.concat([a, b], axis=1).dropna().to_numpy()
    y, x = z[:, 0], z[:, 1]
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * u[:, None]).T @ (X * u[:, None])
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = (X[L:] * u[L:, None]).T @ (X[:-L] * u[:-L, None])
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    return {"alpha_daily": float(beta[0]), "alpha_annual_simple": float(365 * beta[0]), "beta": float(beta[1]),
            "t_alpha_nw": float(beta[0] / se[0]), "t_beta_nw": float(beta[1] / se[1]),
            "corr": float(np.corrcoef(x, y)[0, 1]), "days": int(len(y))}


def rolling_sharpe(E: pd.Series, window: int = 365) -> pd.Series:
    r = tb.daily(E).pct_change()
    return r.rolling(window).mean() / r.rolling(window).std() * math.sqrt(365)


def decide(tests: dict) -> dict:
    fails = [k for k, v in tests.items() if not v["pass"]]
    return {"version": VERSION, "status": BParams().status,
            "verdict": "BEATS_ALL_BENCHMARKS" if not fails else "FAILS_" + "_".join(fails), "tests": tests}
