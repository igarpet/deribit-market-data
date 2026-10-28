"""Turtle Breakout 19: a 1-day reversal rule on the days the BTC turtle is out of the market.

Protocol: discussions/Turtle_Breakout/021_claude_tb19_preregistration.md (fixed before any real-data output).

Rule (frozen from TB 18, which found AC1 = -0.09 on FLAT days):
    at the close of day t, if the turtle (TB 01, long only) holds no long at that close (FLAT) and today's log
    return r_t is at or below the 20th percentile of the r of the previous 365 FLAT days (>= 100 of them),
    buy at the close C_t and sell at the next close C_{t+1}. Long only, one day, no stop.
Size: notional = min(0.02 / (2 ATR_t / C_t), 3) x equity (the turtle's 2% risk unit; ATR = Wilder 20 of
    completed days); results also in R = net return / (2 ATR_t / C_t).
Costs: 6 bp per side; on the Binance perpetual, funding over the holding day (a long pays positive funding).
Null: 1-day longs on random FLAT days (all FLAT days are the pool), same number of trades, 10,000 draws.
Data / gates (each: mean net R of the rule above the null with empirical p < 0.05/3, AND mean net return > 0):
    G1 Yahoo BTC-USD 2014-10 .. 2019-12   G2 Yahoo BTC-USD 2020-01 .. end   G3 Binance BTCUSDT perpetual daily
    (00:00 UTC bars from 1m klines, 2019-09 .. end; its own turtle state) - the check that the reversal is not an
    artefact of Yahoo's composite daily close. Verdict REVERSAL_EDGE if G1, G2 and G3 pass.
All three use data that TB 18 used to find the pattern: this confirms that the rule is tradable after costs and
stable across halves and price sources; it is not out-of-sample evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

VERSION = "TB19-0.1.0"


@dataclass(frozen=True)
class RParams:
    quantile: float = 0.20
    lookback_flat_days: int = 365
    min_flat_days: int = 100
    hold: int = 1
    side_cost: float = 0.0006
    side_cost_high: float = 0.0015
    risk_frac: float = 0.02
    pos_cap: float = 3.0
    atr_n: int = 20
    n_null: int = 10000
    alpha: float = 0.05
    n_tests: int = 3
    seed: int = 20261031
    status: str = "PREREGISTERED_TB021"


def registration(rp: RParams | None = None) -> dict:
    d = asdict(rp or RParams())
    d.update(version=VERSION, gates={"G1": "Yahoo 2014-10..2019-12", "G2": "Yahoo 2020-01..end", "G3": "Binance perp daily 2019-09..end",
                                     "each": "mean net R > random FLAT-day 1-day longs (p < alpha/3) and mean net return > 0",
                                     "verdict": "REVERSAL_EDGE if all three pass, else NO_TRADABLE_REVERSAL"},
             reported=["short leg after top-quintile days", "10th / 30th percentile", "2-day hold", "15 bp costs",
                       "account curves; correlation with the turtle; turtle + reversal combined (half capital each)", "by year"])
    return d


def daily_from_1m(df1m: pd.DataFrame) -> pd.DataFrame:
    """1m klines (open_time UTC) -> daily OHLC by UTC calendar day, index = day (tz-naive)."""
    d = df1m[["open_time", "open", "high", "low", "close"]].copy()
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d = d.drop_duplicates("open_time").sort_values("open_time")
    g = d.groupby(d["open_time"].dt.floor("D"))
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last(),
                        "minutes": g["close"].size()}).astype(float)
    out.index = out.index.tz_localize(None)
    return out[out.minutes >= 1380].drop(columns="minutes")        # >= 23 h of minutes


def daily_funding(fund: pd.DataFrame | None, days: pd.DatetimeIndex) -> pd.Series:
    """Sum of funding rates per UTC day (events during the day); 0 when unknown."""
    if fund is None or not len(fund):
        return pd.Series(0.0, index=days)
    t = pd.DatetimeIndex(fund["t"]).tz_convert("UTC").tz_localize(None).floor("D")
    return pd.Series(fund["rate"].to_numpy(), index=t).groupby(level=0).sum().reindex(days, fill_value=0.0)


def wilder_atr(d: pd.DataFrame, n: int) -> pd.Series:
    pc = d.close.shift(1)
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()     # through day t (known at its close)


def signals(r: pd.Series, flat: pd.Series, rp: RParams, side: str = "long", q: float | None = None) -> pd.Series:
    """Signal at the close of t: FLAT at t and r_t <= q-quantile (long) / >= (1-q)-quantile (short) of r on the previous
    lookback FLAT days (strictly before t)."""
    q = rp.quantile if q is None else q
    rf = r.where(flat.astype(bool))
    hist = rf.dropna()
    thr = pd.Series(np.nan, index=r.index)
    vals = hist.to_numpy(); idx = hist.index
    pos = np.searchsorted(idx.values, r.index.values, side="left")        # number of FLAT days strictly before t
    for i, p in enumerate(pos):
        if p >= rp.min_flat_days:
            w = vals[max(0, p - rp.lookback_flat_days):p]
            thr.iat[i] = np.quantile(w, q if side == "long" else 1 - q)
    cond = (r <= thr) if side == "long" else (r >= thr)
    return (cond & flat.astype(bool) & thr.notna() & r.notna())


def trades(d: pd.DataFrame, sig: pd.Series, rp: RParams, funding: pd.Series | None = None, side: str = "long",
           hold: int | None = None, cost: float | None = None) -> pd.DataFrame:
    """One trade per signal day t: enter at C_t, exit at C_{t+hold}."""
    hold = rp.hold if hold is None else hold; cost = rp.side_cost if cost is None else cost
    s = 1 if side == "long" else -1
    C = d.close.to_numpy(); atr = wilder_atr(d, rp.atr_n).to_numpy()
    fsum = np.r_[0.0, np.cumsum((funding if funding is not None else pd.Series(0.0, index=d.index)).reindex(d.index, fill_value=0.0).to_numpy())]
    rows = []
    for i in np.flatnonzero(sig.reindex(d.index, fill_value=False).to_numpy()):
        j = i + hold
        if j >= len(C) or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        g = s * (C[j] / C[i] - 1)
        fund = s * (fsum[j + 1] - fsum[i + 1])                       # funding of days i+1..j (positions held through them)
        ret = g - cost * (1 + C[j] / C[i]) - fund
        unit = 2 * atr[i] / C[i]
        rows.append({"entry": d.index[i], "exit": d.index[j], "gross": g, "ret": ret, "R": ret / unit, "unit": unit,
                     "notional_frac": min(rp.risk_frac / unit, rp.pos_cap)})
    return pd.DataFrame(rows)


def account(T: pd.DataFrame, days: pd.DatetimeIndex, e0: float = 1e5) -> pd.Series:
    """Equity compounding the trades in time order (sized on current equity), marked once per day."""
    eq = pd.Series(np.nan, index=days); E = e0
    pnl = pd.Series(0.0, index=days)
    if len(T):
        for r in T.sort_values("entry").itertuples():
            m = r.notional_frac * r.ret
            pnl[r.exit] += np.log1p(m) if m > -1 else -50
    return e0 * np.exp(pnl.cumsum())


def null_test(real_R: np.ndarray, pool_R: np.ndarray, rp: RParams, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(real_R)
    if n == 0 or len(pool_R) == 0:
        return {"trades": n, "mean_R": np.nan, "null_mean_R": np.nan, "p": np.nan}
    draws = pool_R[rng.integers(0, len(pool_R), size=(rp.n_null, n))].mean(1)
    m = float(np.mean(real_R))
    return {"trades": n, "mean_R": m, "null_mean_R": float(draws.mean()), "null_sd": float(draws.std(ddof=1)),
            "p": float((1 + np.sum(draws >= m)) / (1 + rp.n_null))}


def gate(real: pd.DataFrame, pool: pd.DataFrame, rp: RParams, seed: int) -> dict:
    t = null_test(real.R.to_numpy() if len(real) else np.array([]), pool.R.to_numpy() if len(pool) else np.array([]), rp, seed)
    t["mean_ret_pct"] = float(real.ret.mean() * 100) if len(real) else np.nan
    t["win_rate"] = float((real.ret > 0).mean()) if len(real) else np.nan
    a = rp.alpha / rp.n_tests
    t["pass"] = bool(np.isfinite(t["p"]) and t["p"] < a and t["mean_ret_pct"] > 0)
    return t
