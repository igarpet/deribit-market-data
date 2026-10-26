"""Turtle Breakout 14: the frozen specification on US equities — S&P 500 constituents and index ETFs, daily data.

Protocol: discussions/Turtle_Breakout/016_claude_tb14_preregistration.md (fixed before any real-data output).

Data: Yahoo Finance daily OHLC, adjusted for splits and dividends (total return), 2000-01-03 .. last session.
Universe: the S&P 500 constituent list of Ivan's thesis (data/raw/constituents.csv of igarpet/thesis @ f967975,
a 2026 snapshot: survivorship-biased), and the index ETFs SPY, QQQ, DIA, IWM.

The turtle engine works on 17:00 New York periods built from intraday bars. For equities:
  * trading sessions are mapped to consecutive pseudo-days, so 20 periods = 20 trading days (weekends and
    holidays do not dilute the 20/10/20-day windows); results are mapped back to the real session dates;
  * each daily bar becomes three ordered pseudo-bars O->H, H->L, L->C. A long that enters at the high and is
    stopped at the low on the same day is filled in that (adverse) order; an open gap through a stop fills at the
    open. Daily ATR, highs and lows are unchanged.
Costs: 5 bp per side (commission + spread + slippage; 10 bp sensitivity); shorts pay a borrow fee of 0.5%/yr
(2% sensitivity), charged at the exit; no funding; idle cash earns nothing.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

VERSION = "TB14-0.1.0"
INDEX_ETFS = ("SPY", "QQQ", "DIA", "IWM")
BASE = pd.Timestamp("2000-01-03", tz="UTC")       # pseudo-day 0
BAR_HOURS = (13, 14, 15)                          # pseudo-bar ends (UTC): inside the pseudo-day's 17:00 NY period
ANN = 252.0


@dataclass(frozen=True)
class SParams:
    start: str = "2000-01-03"
    side_cost: float = 0.0005
    side_cost_high: float = 0.0010
    borrow: float = 0.005
    borrow_high: float = 0.02
    gross_cap: float = 2.0                         # Reg-T overnight: entry notional of open positions <= 2 x equity
    f_sp500: tuple = (0.00025, 0.0005, 0.001)
    f_sp500_primary: float = 0.0005
    f_index: tuple = (0.0025, 0.005, 0.01)
    f_index_primary: float = 0.005
    warmup_sessions: int = 30
    status: str = "PREREGISTERED_TB016"


def registration(sp: SParams | None = None) -> dict:
    d = asdict(sp or SParams())
    d.update(version=VERSION, index_etfs=list(INDEX_ETFS), universe="thesis S&P 500 snapshot (igarpet/thesis f967975)",
             frozen_spec="one account, TB 01 longs + TB 07 reverse, R0, f x equity, 3x notional per position, gross <= 2x",
             gates={"A": "long_rev at primary f: mean daily log growth, block-bootstrap lower bound at alpha/4 > 0",
                    "B": "Calmar(long_rev) - Calmar(long) at primary f, lower bound at alpha/4 > 0",
                    "sets": "SP500 and INDEX; Bonferroni over 4; verdict per set HOLDS / EDGE_ONLY / REVERSAL_ONLY / FAILS"})
    return d


# ---------------------------------------------------------------- calendar mapping

def pseudo_day(i: np.ndarray | int):
    return BASE + pd.to_timedelta(np.asarray(i), unit="D")


def pseudo_bars(daily: pd.DataFrame, sessions: pd.DatetimeIndex, order: str = "adverse", seed: int = 0) -> pd.DataFrame:
    """daily: OHLC indexed by session date (tz-naive or UTC). Returns 3 pseudo-bars per session on the pseudo
    timeline (index 'end'). order: 'adverse' (TB 14: always O->H->L->C), 'nearest' (the extreme nearer to the
    open comes first) or 'random' (O->H->L->C or O->L->H->C with probability 1/2, seeded)."""
    d = daily.copy()
    d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize() if d.index.tz is not None else pd.DatetimeIndex(d.index).normalize()
    ses = pd.DatetimeIndex(sessions).tz_localize(None).normalize() if sessions.tz is not None else pd.DatetimeIndex(sessions).normalize()
    d = d[d.index.isin(ses)].dropna(subset=["open", "high", "low", "close"])
    d = d[(d[["open", "high", "low", "close"]] > 0).all(axis=1)]
    pos = ses.get_indexer(d.index)
    O, Hh, Ll, Cc = (d[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    H = np.maximum.reduce([Hh, O, Cc]); L = np.minimum.reduce([Ll, O, Cc])
    if order == "adverse":
        hf = np.ones(len(O), bool)
    elif order == "nearest":
        hf = (H - O) <= (O - L)
    elif order == "random":
        hf = np.random.default_rng(seed).random(len(O)) < 0.5
    else:
        raise ValueError(order)
    X1 = np.where(hf, H, L); X2 = np.where(hf, L, H)                 # first and second extreme
    day = pseudo_day(pos)
    ends = [day + pd.Timedelta(hours=h) for h in BAR_HOURS]
    bars = [pd.DataFrame({"open": O, "high": np.maximum(O, X1), "low": np.minimum(O, X1), "close": X1}, index=ends[0]),
            pd.DataFrame({"open": X1, "high": H, "low": L, "close": X2}, index=ends[1]),
            pd.DataFrame({"open": X2, "high": np.maximum(X2, Cc), "low": np.minimum(X2, Cc), "close": Cc}, index=ends[2])]
    out = pd.concat(bars).sort_index()
    out.index = pd.DatetimeIndex(out.index, name="end")
    return out


def to_real(index: pd.DatetimeIndex, sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Pseudo timestamps -> the real session date of their pseudo-day."""
    n = ((pd.DatetimeIndex(index).tz_convert("UTC").normalize() - BASE) // pd.Timedelta(days=1)).to_numpy()
    n = np.clip(n, 0, len(sessions) - 1)
    return pd.DatetimeIndex(sessions[n])


def real_daily(D: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """tp.daily_curve output on the pseudo timeline -> one value per real session (last)."""
    s = pd.Series(D.to_numpy(), index=to_real(D.index, sessions))
    return s.groupby(level=0).last()


def apply_borrow(book, annual: float, sessions: pd.DatetimeIndex):
    """Charge the short borrow fee at exit: buy-back price raised by px_in x annual x calendar days / 365."""
    T = book.trades.copy()
    m = (T.dir == -1).to_numpy()
    if m.any():
        tin = to_real(pd.DatetimeIndex(T.t_in[m]), sessions); tout = to_real(pd.DatetimeIndex(T.t_out[m]), sessions)
        days = ((tout - tin) / pd.Timedelta(days=1)).to_numpy()
        T.loc[m, "px_out"] = T.px_out[m].to_numpy() + T.px_in[m].to_numpy() * annual * days / 365.0
        ret = T.dir * (T.px_out - T.px_in) / T.px_in - T.cost * (1 + T.px_out / T.px_in) - T.dir * (T.fc_out - T.fc_in) / T.px_in
        T["R"] = ret / (2.0 * T.atr / T.px_in)
    out = type(book)(book.name, book.pid0, book.close, book.fcum, T)
    return out


# ---------------------------------------------------------------- metrics on session-dated curves

def metrics(D: pd.Series, e0: float, floor: float = 1e-6) -> dict:
    D = D.clip(lower=floor * e0)
    lr = np.diff(np.log(np.r_[e0, D.to_numpy()]))
    yrs = (D.index[-1] - D.index[0]).days / 365.25 if len(D) > 1 else np.nan
    ruined = bool(D.iloc[-1] <= floor * e0 * 1.0001)
    tot = D.iloc[-1] / e0
    cagr = float(tot ** (1 / yrs) - 1) if (yrs and yrs > 0 and not ruined) else -1.0
    dd = float((D / np.maximum(D.cummax(), e0) - 1).min())
    r = np.expm1(lr)
    return {"cagr": cagr, "max_dd": dd, "calmar": cagr / abs(dd) if dd < 0 else np.nan,
            "sharpe": float(r.mean() / r.std() * math.sqrt(ANN)) if r.std() > 0 else np.nan,
            "vol": float(r.std() * math.sqrt(ANN)), "ruined": ruined, "final_multiple": float(tot), "years": yrs}


def equal_weight_hold(closes: dict) -> pd.Series:
    L = pd.DataFrame({a: np.log(c).diff() for a, c in closes.items()})
    return L.mean(axis=1, skipna=True).dropna()


def period_slices():
    return {"2000-2009": ("2000-01-01", "2009-12-31"), "2010-2019": ("2010-01-01", "2019-12-31"), "2020-": ("2020-01-01", "2100-01-01")}


def synthetic_daily(n: int, seed: int, vol: float = 0.02, drift: float = 0.0, steps: int = 26, gap: float = 0.3,
                    start: str = "2000-01-03") -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Daily OHLC taken from an intraday random-walk path (steps per session) plus an overnight gap, so highs and
    lows are real excursions of the path (used to calibrate the daily-bar fills)."""
    rng = np.random.default_rng(seed)
    ses = pd.bdate_range(start, periods=n)
    k = steps + gap * steps
    r = rng.normal(drift / k, vol / math.sqrt(k), (n, steps)); g = rng.normal(drift * gap * steps / k, vol * math.sqrt(gap * steps / k), n)
    lp = math.log(100.0); O = np.empty(n); H = np.empty(n); L = np.empty(n); C = np.empty(n)
    for i in range(n):
        o = lp + g[i]; p = o + np.cumsum(r[i])
        O[i], H[i], L[i], C[i] = o, max(o, p.max()), min(o, p.min()), p[-1]; lp = p[-1]
    return pd.DataFrame({"open": np.exp(O), "high": np.exp(H), "low": np.exp(L), "close": np.exp(C)}, index=ses), ses
