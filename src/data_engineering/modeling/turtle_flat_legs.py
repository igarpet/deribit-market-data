"""Turtle Breakout 20: two trend-exit legs for the days the BTC turtle is flat - long after a sharp drop (leg A) and
short at a stalled peak (leg B) - held with the turtle's exits and no time limit.

Protocol: discussions/Turtle_Breakout/022_claude_tb20_preregistration.md (fixed before any real-data output).

Common rules (Ivan, 2026-10-28): signals only at the close of a FLAT day (the TB 01 long-only turtle holds no long);
entry at that close C_t; exits exactly the turtle's - fixed stop entry -/+ 2 ATR_t (never moves) and a trailing stop
at the lowest low / highest high of the last 10 completed days, recalculated daily; the stop that is closer to the
price applies; filled at the stop, or at the open if the open gaps through it; NO take-profit and NO time limit
(a trade ends only at its stop or at the end of the data). One position per leg at a time; signals while a leg is
in a trade are ignored. Size = the turtle's: units = 0.02 HWM / (2 ATR_t), notional <= 3x equity (each leg its own
account). Costs 6 bp per side; on the perpetual, funding (a long pays positive funding, a short receives it).
    Leg A  long   r_t <= 20th percentile of r over the previous 365 FLAT days (>= 100)   [TB 19's signal]
    Leg B  short  C_t >= max(H_{t-54..t}) - ATR_t   (within 1 ATR of the 55-day high)  AND
                  |ln(C_t / C_{t-20})| <= 0.5 x sd20, sd20 = sd of daily log returns over the previous 365 days x sqrt(20)
Null per leg: entries on random FLAT days with the same direction, exits, costs (pool = every FLAT day's trade),
same number of trades, 10,000 draws. Gates per leg and data set (Yahoo 2014-19, Yahoo 2020-, Binance perp): mean
net R above the null with p < 0.01 AND mean net return > 0; a leg is EDGE if it passes all three.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

VERSION = "TB20-0.1.0"


@dataclass(frozen=True)
class LParams:
    stop_atr: float = 2.0
    trail_n: int = 10
    atr_n: int = 20
    risk_frac: float = 0.02
    pos_cap: float = 3.0
    side_cost: float = 0.0006
    a_quantile: float = 0.20
    a_lookback: int = 365
    a_min_flat: int = 100
    b_high_n: int = 55
    b_atr_band: float = 1.0
    b_flat_sd: float = 0.5
    b_sd_lookback: int = 365
    n_null: int = 10000
    p_max: float = 0.01
    seed: int = 20261101
    start_equity: float = 1e5
    status: str = "PREREGISTERED_TB022"


def registration(lp: LParams | None = None) -> dict:
    d = asdict(lp or LParams())
    d.update(version=VERSION, exits="turtle: fixed 2 ATR + 10-day trailing, no take-profit, no time limit",
             gates="per leg x {Yahoo 2014-19, Yahoo 2020-, Binance perp}: mean net R > random FLAT-day entries (same exits), p < 0.01, "
                   "and mean net return > 0; leg EDGE if all three pass")
    return d


def wilder_atr(d: pd.DataFrame, n: int) -> np.ndarray:
    pc = d.close.shift(1)
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean().to_numpy()


def signals_b(d: pd.DataFrame, flat: pd.Series, lp: LParams) -> pd.Series:
    atr = pd.Series(wilder_atr(d, lp.atr_n), index=d.index)
    h55 = d.high.rolling(lp.b_high_n, min_periods=lp.b_high_n).max()
    r = np.log(d.close).diff()
    sd20 = r.shift(1).rolling(lp.b_sd_lookback, min_periods=100).std() * np.sqrt(20)
    r20 = np.log(d.close / d.close.shift(20))
    sig = (d.close >= h55 - lp.b_atr_band * atr) & (r20.abs() <= lp.b_flat_sd * sd20) & flat.astype(bool)
    return sig.fillna(False)


def walk(d_o, d_h, d_l, d_c, atr, fcum, i: int, side: int, lp: LParams) -> tuple[int, float, str]:
    """Trade entered at the close of day i; returns (exit day, fill, reason). No time limit."""
    n = len(d_c); px = d_c[i]; fixed = px - side * lp.stop_atr * atr[i]
    for t in range(i + 1, n):
        a = max(0, t - lp.trail_n)
        if side == 1:
            stop = max(fixed, d_l[a:t].min())
            if d_o[t] <= stop:
                return t, d_o[t], "gap"
            if d_l[t] <= stop:
                return t, stop, "stop"
        else:
            stop = min(fixed, d_h[a:t].max())
            if d_o[t] >= stop:
                return t, d_o[t], "gap"
            if d_h[t] >= stop:
                return t, stop, "stop"
    return n - 1, d_c[n - 1], "end"


def _arrays(d: pd.DataFrame, funding: pd.Series | None):
    f = (funding if funding is not None else pd.Series(0.0, index=d.index)).reindex(d.index, fill_value=0.0).to_numpy()
    return d.open.to_numpy(), d.high.to_numpy(), d.low.to_numpy(), d.close.to_numpy(), np.r_[0.0, np.cumsum(f)]


def trade_record(d, arrs, atr, i, side, lp):
    o, h, l, c, fc = arrs
    j, fill, why = walk(o, h, l, c, atr, fc, i, side, lp)
    px = c[i]
    fund = side * (fc[j + 1] - fc[i + 1])                       # funding of days i+1..j
    ret = side * (fill / px - 1) - lp.side_cost * (1 + fill / px) - fund
    unit = lp.stop_atr * atr[i] / px
    return {"entry": d.index[i], "exit": d.index[j], "i": i, "j": j, "side": side, "px_in": px, "px_out": fill, "reason": why,
            "hold_days": j - i, "ret": ret, "R": ret / unit, "unit": unit, "fund": fund}


def simulate_leg(d: pd.DataFrame, sig: pd.Series, side: int, lp: LParams, funding: pd.Series | None = None):
    """One position at a time; turtle sizing on the leg's own account (HWM). Returns (trades, daily equity)."""
    arrs = _arrays(d, funding); o, h, l, c, fc = arrs
    atr = wilder_atr(d, lp.atr_n); s = sig.reindex(d.index, fill_value=False).to_numpy()
    n = len(c); E = np.full(n, np.nan); realized = 0.0; hwm = e_prev = lp.start_equity
    pos = None; rows = []
    for t in range(n):
        if pos is not None and t == pos["j"]:
            r = pos["rec"]; u = pos["u"]
            realized += u * r["px_in"] * r["ret"]
            rows.append({**r, "units": u}); pos = None
        if pos is None and s[t] and np.isfinite(atr[t]) and atr[t] > 0 and e_prev > 0:
            rec = trade_record(d, arrs, atr, t, side, lp)
            u = min(lp.risk_frac * hwm / (lp.stop_atr * atr[t]), lp.pos_cap * e_prev / c[t])
            pos = {"j": rec["j"] if rec["reason"] != "end" else n, "rec": rec, "u": u}
        unreal = 0.0
        if pos is not None:
            r = pos["rec"]; u = pos["u"]
            unreal = u * (side * (c[t] - r["px_in"]) - lp.side_cost * r["px_in"] - side * r["px_in"] * (fc[t + 1] - fc[r["i"] + 1]))
        E[t] = lp.start_equity + realized + unreal
        hwm = max(hwm, E[t]); e_prev = E[t]
    if pos is not None:
        rows.append({**pos["rec"], "units": pos["u"]})
    return pd.DataFrame(rows), pd.Series(E, index=d.index)


def pool(d: pd.DataFrame, flat: pd.Series, side: int, lp: LParams, funding: pd.Series | None = None) -> pd.DataFrame:
    """The trade of every FLAT day (independent of each other): the random-entry null samples from it."""
    arrs = _arrays(d, funding); atr = wilder_atr(d, lp.atr_n)
    idx = np.flatnonzero(flat.reindex(d.index, fill_value=False).to_numpy() & np.isfinite(atr) & (atr > 0))
    return pd.DataFrame([trade_record(d, arrs, atr, i, side, lp) for i in idx if i < len(d) - 1])


def gate(T: pd.DataFrame, P: pd.DataFrame, lp: LParams, seed: int) -> dict:
    T = T[T.reason != "end"] if len(T) else T; P = P[P.reason != "end"] if len(P) else P
    n = len(T)
    if n == 0 or len(P) == 0:
        return {"trades": n, "pass": False}
    rng = np.random.default_rng(seed)
    draws = P.R.to_numpy()[rng.integers(0, len(P), size=(lp.n_null, n))].mean(1)
    m = float(T.R.mean())
    p = float((1 + np.sum(draws >= m)) / (1 + lp.n_null))
    out = {"trades": n, "mean_R": m, "null_mean_R": float(draws.mean()), "null_sd": float(draws.std(ddof=1)), "p": p,
           "mean_ret_pct": float(T.ret.mean() * 100), "win_rate": float((T.ret > 0).mean()), "median_hold_days": float(T.hold_days.median()),
           "share_R_ge_3": float((T.R >= 3).mean())}
    out["pass"] = bool(p < lp.p_max and out["mean_ret_pct"] > 0)
    return out
