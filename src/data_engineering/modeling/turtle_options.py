"""Turtle Breakout 06: the daily turtle with a protective put at the entry strike, and an option-only (long call)
version, on BTC and ETH.

Protocol: discussions/Turtle_Breakout/008_claude_tb06_preregistration.md (fixed before any real-data output).

Trades: TB 01 daily rules on BTC/ETH perps (turtle_breakout.Params() defaults), entries from DVOL availability.
Option horizon: DTE_TARGET = 38 days = mean holding time of the R >= 3 trades in the TB 04 DISCOVERY set
(26 other perps; 83 trades, mean 37.8 d) - fixed from data the confirmation coins never touch. The option expiry
is the first Friday 08:00 UTC at or after entry + 38 days (Deribit weekly/monthly expiries).

Strategies (one option per unit of underlying; per-trade results in R = 2 ATR at entry, as TB 04)
    S0 perp          TB 01 net of fees, slippage and funding
    S1 perp + put    put struck at the ENTRY price; sold at the bid when the turtle exits, or settles at intrinsic
                     at expiry if that comes first (no roll: the protection covers the first DTE_TARGET days)
    S2 call ATM      no perp; call struck at the entry price, sold at the bid when the turtle exits; if it expires
                     first it settles and a new ATM call (same DTE rule) is bought while the signal is still on
    S3 call OTM      as S2 with strike = entry + 1 ATR ("a bit above the current price")
Reported variants: S1 holding the put to expiry regardless of the turtle exit; S2 / S3 without rolling.

Pricing: Black-76 on the perp price, sigma = DVOL (30-day ATM implied vol, hourly, as of the bar) for every strike
and tenor (no skew / term structure: disclosed limitation; strike-at-entry and +1 ATR are near the money).
Bought at sigma x (1 + h), sold at sigma x (1 - h), h = relative IV half-spread (primary 2%, sensitivity 0 / 5%).
Fees: taker 0.03% of the underlying per option, capped at 12.5% of the option price; delivery fee 0.015% capped
at 12.5% of the intrinsic value at expiry.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import option_pricing as op
from . import turtle_breakout as tb

VERSION = "TB06-0.1.0"
ASSETS = ("BTC", "ETH")
DTE_TARGET_DAYS = 38
YEAR_S = 365.0 * 86400


@dataclass(frozen=True)
class OParams:
    dte_days: int = DTE_TARGET_DAYS
    half_spread: float = 0.02
    fee_rate: float = 0.0003
    fee_cap: float = 0.125
    delivery_rate: float = 0.00015
    otm_atr: float = 1.0
    n_boot: int = 2000
    block_days: int = 21
    alpha: float = 0.05
    seed: int = 20261029
    status: str = "PREREGISTERED_TB008"


def registration(op_: OParams | None = None) -> dict:
    o = op_ or OParams()
    d = asdict(o)
    d.update(version=VERSION, assets=list(ASSETS), dte_source="TB 04 discovery: mean hold of R>=3 trades = 37.8 d",
             strategies={"S0": "perp", "S1": "perp + put K=entry, sold at turtle exit or settled at expiry",
                         "S2": "call K=entry, sold at turtle exit, rolled at expiry", "S3": "call K=entry+1 ATR"},
             pricing="Black-76, sigma = DVOL as of the bar; buy at sigma(1+h), sell at sigma(1-h)",
             tests={"Q1": "S1 vs S0", "Q2": "S2 vs S0", "alpha_each": o.alpha / 2,
                    "stat": "paired Sharpe difference of daily cumulative-R curves (21d circular block bootstrap) "
                            "-> IMPROVES (lower > 0) / WORSENS (upper < 0) / NO_DIFFERENCE; per-trade mean R "
                            "difference (month-cluster bootstrap) and max drawdown reported"})
    return d


# ---------------------------------------------------------------- data and helpers

def load_dvol(path) -> pd.Series:
    d = pd.read_csv(path, parse_dates=["t"])
    s = pd.Series(d["close"].astype(float).to_numpy() / 100.0, index=pd.to_datetime(d["t"], utc=True)).sort_index()
    return s[~s.index.duplicated()]


def sigma_at(dvol: pd.Series, t) -> np.ndarray:
    """DVOL of the last completed hour at or before t (hour bars are labelled by their start)."""
    t = pd.DatetimeIndex(np.atleast_1d(t))
    i = dvol.index.searchsorted(t - pd.Timedelta(hours=1), side="right") - 1
    v = np.where(i >= 0, dvol.to_numpy()[np.maximum(i, 0)], np.nan)
    age = np.where(i >= 0, (t - dvol.index[np.maximum(i, 0)]) / pd.Timedelta(hours=1), np.inf)
    return np.where(age <= 6, v, np.nan)


def expiry_for(t: pd.Timestamp, days: int) -> pd.Timestamp:
    """First Friday 08:00 UTC at or after t + days."""
    x = t + pd.Timedelta(days=days)
    d = x.normalize() + pd.Timedelta(hours=8)
    if d < x:
        d += pd.Timedelta(days=1)
    while d.weekday() != 4:
        d += pd.Timedelta(days=1)
    return d


def _px(F, K, T, s, call):
    return float(op.black76_usd(F, K, max(T, 0.0), s, call))


def _fee(price, F, o: OParams):
    return min(o.fee_rate * F, o.fee_cap * price)


def _price_at(d: tb.AssetData, t: pd.Timestamp) -> tuple[float, int]:
    """Close of the last 5m bar ending at or before t."""
    j = int(d.end.searchsorted(t, side="right")) - 1
    return float(d.c[max(j, 0)]), max(j, 0)


def option_leg(d: tb.AssetData, dvol: pd.Series, t0, t1, K: float, call: bool, o: OParams, h: float,
               t_exit_signal=None) -> dict:
    """Buy one option at t0 (expiry chosen from t0), close at min(t1, expiry). Returns USD P&L per unit."""
    E = expiry_for(pd.Timestamp(t0), o.dte_days)
    F0, _ = _price_at(d, t0)
    s0 = float(sigma_at(dvol, t0)[0])
    T0 = (E - t0).total_seconds() / YEAR_S
    buy = _px(F0, K, T0, s0 * (1 + h), call)
    fee0 = _fee(buy, F0, o)
    if t1 < E:
        F1, _ = _price_at(d, t1)
        s1 = float(sigma_at(dvol, t1)[0])
        sell = _px(F1, K, (E - t1).total_seconds() / YEAR_S, s1 * (1 - h), call)
        fee1 = _fee(sell, F1, o); end = t1; how = "sold"
    else:
        F1, _ = _price_at(d, E)
        sell = max(F1 - K, 0.0) if call else max(K - F1, 0.0)
        fee1 = min(o.delivery_rate * F1, o.fee_cap * sell) if sell > 0 else 0.0; end = E; how = "expired"
    return {"t0": t0, "t_end": end, "expiry": E, "K": K, "sigma0": s0, "premium": buy, "proceeds": sell,
            "fees": fee0 + fee1, "pnl": sell - buy - fee0 - fee1, "how": how, "rolled_into_next": False}


def trade_options(d: tb.AssetData, dvol: pd.Series, tr, o: OParams, h: float) -> dict:
    """All option variants for one turtle trade (row of turtle_breakout trades + atr_entry)."""
    te, tx = pd.Timestamp(tr.entry_time), pd.Timestamp(tr.exit_time)
    E, A = float(tr.entry_px), float(tr.atr_entry)
    Ru = 2 * A
    out = {}
    # S1: put at entry strike, sold at exit or expired (no roll); variant: held to expiry
    p = option_leg(d, dvol, te, tx, E, False, o, h)
    out["put_R"] = p["pnl"] / Ru
    out["put_premium_R"] = p["premium"] / Ru
    out["put_how"] = p["how"]
    p2 = option_leg(d, dvol, te, pd.Timestamp("2262-01-01", tz="UTC"), E, False, o, h)
    out["put_hold_R"] = p2["pnl"] / Ru
    # S2 / S3: calls, rolled while the signal is on; variants without roll
    for name, K0 in (("callATM", E), ("callOTM", E + o.otm_atr * A)):
        t, pnl, legs, prem0 = te, 0.0, 0, None
        while True:
            K = K0 if legs == 0 else (_price_at(d, t)[0] + (o.otm_atr * A if name == "callOTM" else 0.0))
            leg = option_leg(d, dvol, t, tx, K, True, o, h)
            pnl += leg["pnl"]; legs += 1
            if prem0 is None:
                prem0 = leg["premium"]
                out[f"{name}_noroll_R"] = leg["pnl"] / Ru
            if leg["how"] == "sold" or leg["t_end"] >= tx:
                break
            t = leg["t_end"]
        out[f"{name}_R"] = pnl / Ru
        out[f"{name}_legs"] = legs
        out[f"{name}_premium_R"] = prem0 / Ru
    out["sigma_entry"] = float(sigma_at(dvol, te)[0])
    return out


def run_asset(d: tb.AssetData, dvol: pd.Series, T: pd.DataFrame, o: OParams, h: float) -> pd.DataFrame:
    rows = []
    for tr in T.itertuples(index=False):
        rows.append({"asset": d.name, "entry_time": tr.entry_time, "exit_time": tr.exit_time, "entry_px": tr.entry_px,
                     "exit_px": tr.exit_px, "atr_entry": tr.atr_entry, "perp_R": tr.R, "reason": tr.reason,
                     **trade_options(d, dvol, tr, o, h)})
    X = pd.DataFrame(rows)
    if len(X):
        X["perp_put_R"] = X.perp_R + X.put_R
        X["perp_put_hold_R"] = X.perp_R + X.put_hold_R
    return X


# ---------------------------------------------------------------- daily cumulative-R curves (mark to market)

def daily_curve(d: tb.AssetData, dvol: pd.Series, X: pd.DataFrame, kind: str, o: OParams, h: float,
                days: pd.DatetimeIndex) -> pd.Series:
    """Cumulative R of one asset at daily 17:00 NY closes: closed trades at their realised R, open trades marked
    (perp at the close; options at Black-76 mid with DVOL; entry costs already deducted)."""
    out = np.zeros(len(days))
    for r in X.itertuples(index=False):
        te, tx = pd.Timestamp(r.entry_time), pd.Timestamp(r.exit_time)
        Ru = 2 * r.atr_entry
        final = {"perp": r.perp_R, "perp_put": r.perp_put_R, "callATM": r.callATM_R, "callOTM": r.callOTM_R}[kind]
        out[days >= tx] += final
        open_days = days[(days >= te) & (days < tx)]
        if not len(open_days):
            continue
        idx = np.flatnonzero((days >= te) & (days < tx))
        F = np.array([_price_at(d, t)[0] for t in open_days])
        perp_mtm = (F - r.entry_px) / Ru - 2 * 0.0006 * r.entry_px / Ru / 2
        if kind == "perp":
            out[idx] += perp_mtm; continue
        s = sigma_at(dvol, open_days)
        E = expiry_for(te, o.dte_days)
        T = np.maximum((E - open_days).total_seconds().to_numpy() / YEAR_S, 0.0)
        if kind == "perp_put":
            K = r.entry_px
            p0 = _px(r.entry_px, K, (E - te).total_seconds() / YEAR_S, float(sigma_at(dvol, te)[0]) * (1 + h), False)
            v = np.where(T > 0, op.black76_usd(F, K, T, s, False), np.maximum(K - F, 0.0))
            out[idx] += perp_mtm + (v - p0 - _fee(p0, r.entry_px, o)) / Ru
        else:
            K = r.entry_px + (o.otm_atr * r.atr_entry if kind == "callOTM" else 0.0)
            p0 = _px(r.entry_px, K, (E - te).total_seconds() / YEAR_S, float(sigma_at(dvol, te)[0]) * (1 + h), True)
            v = np.where(T > 0, op.black76_usd(F, K, T, s, True), np.maximum(F - K, 0.0))
            out[idx] += (v - p0 - _fee(p0, r.entry_px, o)) / Ru      # first leg only (rolls are rare and short)
    return pd.Series(out, index=days)


def daily_days(start, end) -> pd.DatetimeIndex:
    pid = tb.period_id(pd.DatetimeIndex([start, end]), "day")
    return tb.period_end(np.arange(pid[0], pid[1] + 1), "day")


# ---------------------------------------------------------------- statistics

def sharpe_diff_R(ca: pd.Series, cb: pd.Series, o: OParams, q: float, seed: int) -> dict:
    """Paired Sharpe difference of daily changes in cumulative R (additive curves: no compounding)."""
    x = np.c_[ca.diff().to_numpy(), cb.diff().to_numpy()][1:]
    x = x[np.isfinite(x).all(1)]
    n = len(x)
    sh = lambda z: z.mean(0) / z.std(0) * math.sqrt(365)
    est = sh(x)
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / o.block_days))
    bs = np.empty(o.n_boot)
    for i in range(o.n_boot):
        st = rng.integers(0, n, nb)
        idx = ((st[:, None] + np.arange(o.block_days)) % n).ravel()[:n]
        s = sh(x[idx]); bs[i] = s[0] - s[1]
    lo, hi = np.quantile(bs, [q / 2, 1 - q / 2])
    res = "IMPROVES" if lo > 0 else "WORSENS" if hi < 0 else "NO_DIFFERENCE"
    return {"sharpe_a": float(est[0]), "sharpe_b": float(est[1]), "diff": float(est[0] - est[1]), "lo": float(lo),
            "hi": float(hi), "ci_level": 1 - q, "days": int(n), "result": res}


def mean_diff_month(a: np.ndarray, b: np.ndarray, t: pd.DatetimeIndex, o: OParams, q: float, seed: int) -> dict:
    x = a - b
    m = pd.DatetimeIndex(t).tz_convert("UTC").strftime("%Y-%m").to_numpy()
    u, inv = np.unique(m, return_inverse=True)
    S = np.bincount(inv, weights=x); N = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=o.n_boot).astype(float)
    bs = (W @ S) / (W @ N)
    lo, hi = np.quantile(bs, [q / 2, 1 - q / 2])
    return {"mean_diff_R": float(x.mean()), "lo": float(lo), "hi": float(hi), "trades": int(len(x))}


def max_dd(c: pd.Series) -> float:
    return float((c - c.cummax()).min())


def synthetic_dvol(index_start, index_end, level=0.6) -> pd.Series:
    t = pd.date_range(pd.Timestamp(index_start).floor("h"), pd.Timestamp(index_end).ceil("h"), freq="1h")
    return pd.Series(level, index=t)
