"""Turtle Breakout 15: the original turtle (long) and long + reverse on every S&P 500 stock, each in its own account.

Protocol: discussions/Turtle_Breakout/017_claude_tb15_preregistration.md (fixed before any real-data output).

Per stock, one account, exactly the TB 01 rules: risk 2% of the account's high-water mark per trade
(units = 0.02 HWM / (2 ATR)), notional <= 3x equity; TB 07 reverse short at every losing stop exit in the
long + reverse variant. Daily Yahoo data via turtle_stocks (sessions as consecutive pseudo-days, O->H->L->C
fills), 5 bp per side, short borrow 0.5%/yr. Window: each stock from 2000-01-03 (or its first session) + 30
sessions to the last session.

Decision (Bonferroni over 2, alpha = 0.025 each, one-sided):
    G1 timing edge  pooled mean net R of the real long trades of all stocks above the pooled mean R of
                    random-entry longs with the same exits, costs and per-stock entry rate (50 runs; z-test), AND
                    the difference >= 0.10R: on driftless synthetic stocks whose OHLC come from an intraday path,
                    the daily-bar fills give breakouts +0.04R (sd 0.035 per 8 stocks) over random entries
    G2 reversal     number of stocks with Calmar(long + reverse) > Calmar(long): sign test
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import turtle_breakout as tb
from . import turtle_portfolio as tp
from . import turtle_sizing as ts
from . import turtle_stocks as tk

VERSION = "TB15-0.2.0"   # 0.2.0 (TB 16): decide_paired (per-instrument t-test against the calibrated fill artefact); 0.1.0 functions unchanged


@dataclass(frozen=True)
class AParams:
    risk_frac: float = 0.02
    pos_cap: float = 3.0
    size_base: str = "hwm"
    side_cost: float = 0.0005
    borrow: float = 0.005
    n_null: int = 50
    min_effect_R: float = 0.10          # G1 also needs real - null >= 0.10R (martingale calibration of the daily fills: +0.04R)
    alpha: float = 0.05
    n_tests: int = 2
    seed: int = 20261027
    status: str = "PREREGISTERED_TB017"


def registration(ap: AParams | None = None) -> dict:
    d = asdict(ap or AParams())
    d.update(version=VERSION, rules="TB 01 per stock: 2% of HWM per trade, 3x notional cap; reverse = TB 07",
             gates={"G1": "pooled mean R of real longs > random-entry longs (50 runs, z-test, p < alpha/2) and difference >= 0.10R",
                    "G2": "stocks with Calmar(long_rev) > Calmar(long): sign test p < alpha/2"},
             reported=["per stock CAGR, max DD, Calmar, Sharpe, ruin for long, long + reverse, buy and hold",
                       "share of stocks with positive CAGR; beating buy and hold on Calmar", "mean R of reverse trades",
                       "by GICS sector", "per-stock timing edge sign test", "1% risk"])
    return d


def borrow_list(L: pd.DataFrame, d: tb.AssetData, annual: float, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Raise the buy-back price of reverse shorts by px_in x annual x calendar days / 365 (borrow paid at exit)."""
    L = L.copy()
    m = (L.dir == -1).to_numpy()
    if m.any() and annual:
        tin = tk.to_real(d.end[L.je[m].to_numpy()], sessions); tout = tk.to_real(d.end[L.jx[m].to_numpy()], sessions)
        days = ((tout - tin) / pd.Timedelta(days=1)).to_numpy()
        L.loc[m, "px_out"] = L.px_out[m].to_numpy() + L.px_in[m].to_numpy() * annual * days / 365.0
    return L


def trade_R(L: pd.DataFrame, d: tb.AssetData) -> pd.Series:
    ret = L.dir * (L.px_out - L.px_in) / L.px_in - L.cost * (1 + L.px_out / L.px_in) \
        - L.dir * (d.fcum[L.jx.to_numpy()] - d.fcum[L.je.to_numpy()]) / L.px_in
    return ret / (2.0 * L.atr / L.px_in)


def account(d: tb.AssetData, L: pd.DataFrame, p: tb.Params, ap: AParams, start, sessions, f: float | None = None):
    """Equity of one stock's own account (TB 01 sizing) -> session-dated daily curve."""
    E, T, st = ts.equity_from_trades(d, L, p, ap.risk_frac if f is None else f, start, ap.pos_cap, ap.size_base)
    D = tp.daily_curve(E, p.start_equity)
    return tk.real_daily(D, sessions), T, st


def random_long_R(d: tb.AssetData, p: tb.Params, start, rate: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, 2): sum of net R and count of closed random-entry long trades per run (same exits, costs)."""
    out = np.zeros((n, 2))
    if rate <= 0:
        return out
    for i in range(n):
        _, T, _ = tb.simulate({d.name: d}, p, start=start, mode="random", p_entry={d.name: rate}, rng=rng)
        T = T[T.reason != "end"]
        if not len(T):
            continue
        k = tb.period_id(pd.DatetimeIndex(T.entry_time), p.period) - d.pid0
        R = T.ret_notional.to_numpy() / (p.stop_atr * d.atr[k] / T.entry_px.to_numpy())
        out[i] = (np.nansum(R), np.isfinite(R).sum())
    return out


def z_test(real_mean: float, null_means: np.ndarray) -> tuple[float, float]:
    mu, sd = float(np.mean(null_means)), float(np.std(null_means, ddof=1))
    z = (real_mean - mu) / sd if sd > 0 else np.nan
    return z, (0.5 * math.erfc(z / math.sqrt(2)) if np.isfinite(z) else np.nan)


def sign_test(k: int, n: int) -> float:
    return float(sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n) if n else np.nan


def decide(real_sum: float, real_n: int, null_sum: np.ndarray, null_n: np.ndarray, calmar_L: np.ndarray, calmar_LR: np.ndarray,
           ap: AParams) -> dict:
    a = ap.alpha / ap.n_tests
    nm = null_sum / np.maximum(null_n, 1)
    z, p1 = z_test(real_sum / max(real_n, 1), nm)
    ok = np.isfinite(calmar_L) & np.isfinite(calmar_LR)
    k = int((calmar_LR[ok] > calmar_L[ok]).sum()); n = int(ok.sum())
    p2 = sign_test(k, n)
    diff = real_sum / max(real_n, 1) - float(nm.mean())
    g1 = bool(np.isfinite(p1) and p1 < a and diff >= ap.min_effect_R)
    g2 = bool(np.isfinite(p2) and p2 < a)
    return {"real_mean_R": real_sum / max(real_n, 1), "real_trades": int(real_n), "null_mean_R": float(nm.mean()), "diff_R": diff,
            "null_sd": float(nm.std(ddof=1)), "z": z, "p_G1": p1, "G1": "PASS" if g1 else "FAIL",
            "stocks": n, "reverse_better": k, "p_G2": p2, "G2": "PASS" if g2 else "FAIL",
            "verdict": {(True, True): "EDGE_AND_REVERSAL", (True, False): "TIMING_EDGE_ONLY",
                        (False, True): "REVERSAL_ONLY", (False, False): "NO_EDGE"}[(g1, g2)]}


# ---------------------------------------------------------------- TB 16

FILL_ARTEFACT_R = 0.045      # driftless synthetic contracts (20 sets x 41, 26 years): real - random = +0.045R per instrument (sd 0.086)


def t_test_greater(x: np.ndarray, mu0: float) -> tuple[float, float]:
    """One-sided test that mean(x) > mu0 (normal approximation of the t statistic; n ~ 40)."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    t = (x.mean() - mu0) / (x.std(ddof=1) / math.sqrt(len(x)))
    return float(t), float(0.5 * math.erfc(t / math.sqrt(2)))


def decide_paired(real_sum: float, real_n: int, null_sum: np.ndarray, null_n: np.ndarray, per_real: np.ndarray, per_null: np.ndarray,
                  calmar_L: np.ndarray, calmar_LR: np.ndarray, ap: AParams) -> dict:
    """G1: pooled real - random >= min_effect_R AND per-instrument differences exceed the fill artefact (t-test,
    p < alpha/2). G2 as decide(). The conditional z-test of TB 15 ignores path noise across instruments and gave
    false passes on 8 driftless synthetic ETFs; with ~40 instruments the paired test is calibrated (5% at 0.045)."""
    a = ap.alpha / ap.n_tests
    nm = null_sum / np.maximum(null_n, 1)
    diff = real_sum / max(real_n, 1) - float(nm.mean())
    t, p1 = t_test_greater(np.asarray(per_real) - np.asarray(per_null), FILL_ARTEFACT_R)
    ok = np.isfinite(calmar_L) & np.isfinite(calmar_LR)
    k = int((calmar_LR[ok] > calmar_L[ok]).sum()); n = int(ok.sum())
    p2 = sign_test(k, n)
    g1 = bool(np.isfinite(p1) and p1 < a and diff >= ap.min_effect_R)
    g2 = bool(np.isfinite(p2) and p2 < a)
    return {"real_mean_R": real_sum / max(real_n, 1), "real_trades": int(real_n), "null_mean_R": float(nm.mean()), "diff_R": diff,
            "per_instrument_mean_diff": float(np.nanmean(np.asarray(per_real) - np.asarray(per_null))), "t": t, "p_G1": p1,
            "G1": "PASS" if g1 else "FAIL", "instruments": n, "reverse_better": k, "p_G2": p2, "G2": "PASS" if g2 else "FAIL",
            "verdict": {(True, True): "EDGE_AND_REVERSAL", (True, False): "TIMING_EDGE_ONLY",
                        (False, True): "REVERSAL_ONLY", (False, False): "NO_EDGE"}[(g1, g2)]}
