"""Turtle Breakout 17: the turtle on BTC-USD daily (Yahoo) since 2014 — does the crypto edge predate 2020?

Protocol: discussions/Turtle_Breakout/019_claude_tb17_preregistration.md (fixed before any real-data output).

One account, TB 01 rules (2% of HWM per trade, notional <= 3x equity), long and long + reverse (TB 07), on
Yahoo's BTC-USD daily bars (a spot index, 7 days a week), O->H->L->C fills (turtle_stocks.pseudo_bars).
Costs 6 bp per side (taker 0.05% + 1 bp, as TB 01); no funding (spot), no borrow on shorts; 15 bp sensitivity.

Windows: PRE  2014-09-17 .. 2019-12-31 (never used: every earlier crypto test starts in 2019/2020) - decision
         FULL 2014-09-17 .. last bar; POST 2020-01-01 .. last bar (consistency with TB 01-12) - reported
Decision (Bonferroni over 2):
    G1 timing  PRE: mean net R of the real longs - mean R of random-entry longs (same exits, costs, entry rate;
               1,000 runs) >= 0.48R = the 97.5% quantile of that difference on driftless synthetic BTC-like series
               (daily vol 4%, 48 intraday steps, no overnight gap, 1,932 days; 300 paths: mean +0.078, sd 0.183)
    G2 reversal PRE: Calmar(long + reverse) - Calmar(long), 21-day block bootstrap lower bound (2.5%) > 0
Regimes (reporting labels, dated ex post from BTC's cycle tops and bottoms): see REGIMES.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import turtle_breakout as tb
from . import turtle_sizing as ts
from . import turtle_stock_accounts as ta
from . import turtle_stocks as tk

VERSION = "TB17-0.1.0"
WINDOWS = {"PRE": ("2014-09-17", "2019-12-31"), "FULL": ("2014-09-17", None), "POST": ("2020-01-01", None)}
THRESHOLD_R = {"PRE": 0.48, "FULL": 0.33, "POST": None}      # 97.5% martingale quantiles (1,932 / 4,425 days)
REGIMES = [("2014-09-17", "2015-01-14", "bear 2014"), ("2015-01-15", "2015-10-31", "base 2015"),
           ("2015-11-01", "2017-12-17", "bull 2015-17"), ("2017-12-18", "2018-12-15", "bear 2018"),
           ("2018-12-16", "2021-11-10", "bull 2019-21"), ("2021-11-11", "2022-11-21", "bear 2022"),
           ("2022-11-22", "2025-10-06", "bull 2023-25"), ("2025-10-07", "2100-01-01", "after the 2025 top")]


@dataclass(frozen=True)
class BParams:
    side_cost: float = 0.0006
    side_cost_high: float = 0.0015
    n_null: int = 1000
    alpha: float = 0.05
    n_tests: int = 2
    seed: int = 20261029
    status: str = "PREREGISTERED_TB019"


def registration(bp: BParams | None = None) -> dict:
    d = asdict(bp or BParams())
    d.update(version=VERSION, windows=WINDOWS, threshold_R=THRESHOLD_R, regimes=REGIMES,
             rules="TB 01: 2% of HWM, 3x cap; long and long + reverse (TB 07)",
             gates={"G1": "PRE: real - random mean R >= 0.48R (and reported empirical p)",
                    "G2": "PRE: Calmar(long_rev) - Calmar(long) block-bootstrap lower bound > 0"})
    return d


def regime_of(t) -> str:
    t = pd.Timestamp(t).tz_localize(None) if pd.Timestamp(t).tzinfo else pd.Timestamp(t)
    for a, b, lab in REGIMES:
        if pd.Timestamp(a) <= t <= pd.Timestamp(b):
            return lab
    return "?"


def run_window(daily: pd.DataFrame, start: str, end: str | None, p: tb.Params, rp, ap: ta.AParams, n_null: int,
               rng: np.random.Generator, cost: float | None = None) -> dict:
    """Fresh account on [start, end]: trade lists, accounts (long, long + reverse), buy and hold, random-entry null."""
    d = daily[(daily.index >= pd.Timestamp(start)) & ((daily.index <= pd.Timestamp(end)) if end else True)]
    ses = pd.DatetimeIndex(d.index)
    D = tb.prepare_asset("BTC", tk.pseudo_bars(d, ses), None, p)
    t0 = D.end[0] + pd.Timedelta(days=30)
    L0 = ts.trade_list(D, p, t0, rp, with_reverse=False); L1 = ts.trade_list(D, p, t0, rp, with_reverse=True)
    if cost is not None:
        L0, L1 = L0.assign(cost=cost), L1.assign(cost=cost)
    a, _, sa = ta.account(D, L0, p, ap, t0, ses); b, _, _ = ta.account(D, L1, p, ap, t0, ses)
    c = d.close[d.index >= a.index[0]]; h = p.start_equity * c / c.iloc[0]
    closed = L0[~L0.end]
    R0 = ta.trade_R(closed, D)
    R1 = ta.trade_R(L1, D)
    _, T, st = tb.simulate({"BTC": D}, p, start=t0)
    nul = ta.random_long_R(D, p, t0, st["p_entry"]["BTC"], n_null, rng) if n_null else np.zeros((0, 2))
    null_means = nul[:, 0] / np.maximum(nul[:, 1], 1) if len(nul) else np.array([])
    trades = pd.DataFrame({"kind": np.where(L1.dir == 1, "long", "reverse"), "entry": tk.to_real(D.end[L1.je.to_numpy()], ses),
                           "exit": tk.to_real(D.end[L1.jx.to_numpy()], ses), "R": R1.to_numpy(), "open_at_end": L1.end.to_numpy()})
    trades["regime"] = [regime_of(t) for t in trades.entry]
    return {"long": a, "long_rev": b, "hold": h, "R_long": R0.to_numpy(), "null_means": null_means, "p_entry": st["p_entry"]["BTC"],
            "trades": trades, "mean_gross": sa["mean_gross_leverage"], "D": D, "t0": t0, "sessions": ses}


def random_by_regime(D, p, t0, rate, n, rng, sessions) -> pd.DataFrame:
    """Mean R of random-entry longs by the regime of their entry date (pooled over n runs)."""
    rows = []
    for _ in range(n):
        _, T, _ = tb.simulate({"BTC": D}, p, start=t0, mode="random", p_entry={"BTC": rate}, rng=rng)
        T = T[T.reason != "end"]
        if not len(T):
            continue
        k = tb.period_id(pd.DatetimeIndex(T.entry_time), p.period) - D.pid0
        R = T.ret_notional.to_numpy() / (p.stop_atr * D.atr[k] / T.entry_px.to_numpy())
        ent = tk.to_real(pd.DatetimeIndex(T.entry_time), sessions)
        rows.append(pd.DataFrame({"regime": [regime_of(t) for t in ent], "R": R}))
    X = pd.concat(rows) if rows else pd.DataFrame(columns=["regime", "R"])
    return X.groupby("regime").R.agg(["mean", "size"]).rename(columns={"mean": "random_mean_R", "size": "random_trades"})
