"""Turtle Breakout 13: the frozen TB 12 specification on data no earlier decision used.

Protocol: discussions/Turtle_Breakout/015_claude_tb13_preregistration.md (fixed before any real-data output).

Frozen specification (from TB 12): one account, TB 01 daily longs + TB 07 reverse short at every losing stop
exit, rule R0 (no portfolio rule), risk per trade f x current equity (primary f = 0.5%), notional <= 3 x equity.

Test sets, chosen by rule before any strategy output:
    NEW   USD-M USDT perpetuals in the Binance archive (delisted included) that are NOT among the 30 coins used
          in TB 04-12; >= 400 days of daily klines, first day <= 2025-06-30, median daily quote volume >= $10M;
          excluded: stablecoins, index contracts, dated futures; data cut at the first gap > 7 days.
          60 drawn at random (seed 20261026) if more are eligible. Window: each coin's history (after 30 days).
    SPOT  Binance spot USDT pairs with a first day <= 2018-12-31, median daily quote volume >= $1M over
          2017-08-17 .. 2019-12-31; excluded: stablecoins, leveraged tokens (UP/DOWN/BULL/BEAR); window ends
          2019-12-31 (before the perpetual era of every earlier test). Funding 0 (spot has none).
Decision per set, Bonferroni over 4 tests (alpha = 0.0125 each, one-sided):
    A  the strategy grows: long + reverse at f = 0.5%, mean daily log growth, 21-day block bootstrap lower bound > 0
    B  the reversal helps: Calmar(long + reverse) - Calmar(long) at f = 0.5%, lower bound > 0
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

VERSION = "TB13-0.1.0"
USED_30 = ("BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "LINK", "DOT", "LTC", "BCH", "UNI", "NEAR", "APT",
           "ARB", "OP", "SUI", "FIL", "ATOM", "ETC", "INJ", "AAVE", "TON", "WIF", "HBAR", "XLM", "SEI", "TIA",
           "ENA", "WLD")
STABLE = ("USDC", "BUSD", "TUSD", "FDUSD", "USDP", "DAI", "PAX", "USDS", "USDSB", "SUSD", "UST", "USTC", "EUR", "GBP",
          "AUD", "TRY", "BRL", "RUB", "UAH", "NGN", "BIDR", "IDRT", "BVND", "ZAR", "AEUR", "EURI", "USDE", "USD1", "XUSD",
          "BFUSD", "PYUSD", "RLUSD", "USDQ")
INDEX = ("BTCDOM", "DEFI", "FOOTBALL", "BLUEBIRD", "ALTINDEX", "MEMEINDEX")
LEVERAGED = ("UP", "DOWN", "BULL", "BEAR")


@dataclass(frozen=True)
class OParams:
    seed: int = 20261026
    new_n: int = 60
    new_min_days: int = 400
    new_first_by: str = "2025-06-30"
    new_min_qvol: float = 10e6
    spot_first_by: str = "2018-12-31"
    spot_start: str = "2017-08-17"
    spot_end: str = "2019-12-31"
    spot_min_qvol: float = 1e6
    max_gap_days: int = 7
    f_primary: float = 0.005
    f_report: tuple = (0.0025, 0.005, 0.01)
    alpha: float = 0.05
    n_tests: int = 4
    block_days: int = 21
    n_boot: int = 4000
    spot_cost_high: float = 0.0011          # sensitivity: 0.1% spot taker (2018) + 1 bp
    spot_funding_sens: float = 0.0001       # sensitivity: +0.01% per 8 h, longs pay / shorts receive
    status: str = "PREREGISTERED_TB015"


def registration(op: OParams | None = None) -> dict:
    d = asdict(op or OParams())
    d.update(version=VERSION, used_30=list(USED_30), excluded_stable=list(STABLE), excluded_index=list(INDEX),
             frozen_spec="one account, TB 01 longs + TB 07 reverse, R0, f x equity (0.5%), 3x notional cap per position",
             gates={"A": "long_rev f=0.5%: mean daily log growth, block-bootstrap lower bound at alpha/4 > 0",
                    "B": "Calmar(long_rev) - Calmar(long) at f=0.5%, lower bound at alpha/4 > 0",
                    "verdict per set": "HOLDS (A and B) / EDGE_ONLY (A) / REVERSAL_ONLY (B) / FAILS"})
    return d


# ---------------------------------------------------------------- universe rules

def base_of(symbol: str) -> str:
    return symbol[:-4] if symbol.endswith("USDT") else symbol


def core_base(base: str) -> str:
    """'1000PEPE' -> 'PEPE', '1000000MOG' -> 'MOG', '1MBABYDOGE' -> 'BABYDOGE'."""
    for p in ("1000000", "100000", "10000", "1000", "1M"):
        if base.startswith(p) and len(base) > len(p):
            return base[len(p):]
    return base


def new_candidate(symbol: str) -> bool:
    if not symbol.endswith("USDT") or "_" in symbol:
        return False
    b = base_of(symbol); c = core_base(b)
    return b not in USED_30 and c not in USED_30 and c not in STABLE and b not in STABLE and c not in INDEX


def spot_candidate(symbol: str) -> bool:
    if not symbol.endswith("USDT") or "_" in symbol:
        return False
    b = base_of(symbol)
    if b in STABLE or b in INDEX:
        return False
    return not any(b.endswith(s) and len(b) > len(s) for s in LEVERAGED)


def cut_at_gap(daily: pd.DataFrame, max_gap_days: int) -> pd.DataFrame:
    """Keep rows before the first gap > max_gap_days (a delisting; a later relisting may be another coin)."""
    if len(daily) < 2:
        return daily
    d = pd.to_datetime(pd.Series(daily["day"].to_numpy()), utc=True).diff()
    gap = np.flatnonzero((d > pd.Timedelta(days=max_gap_days)).to_numpy())
    return daily.iloc[: gap[0]] if len(gap) else daily


def screen(stats: pd.DataFrame, op: OParams, kind: str) -> pd.DataFrame:
    """stats: one row per symbol with first, last, days, med_qvol (computed after cut_at_gap / window).
    Returns the eligible rows with an 'eligible' reason column."""
    s = stats.copy()
    if kind == "new":
        s["eligible"] = (s.days >= op.new_min_days) & (pd.to_datetime(s["first"], utc=True) <= pd.Timestamp(op.new_first_by, tz="UTC")) \
            & (s.med_qvol >= op.new_min_qvol)
    else:
        s["eligible"] = (pd.to_datetime(s["first"], utc=True) <= pd.Timestamp(op.spot_first_by, tz="UTC")) & (s.med_qvol >= op.spot_min_qvol)
    return s


def draw(eligible: list, op: OParams) -> list:
    e = sorted(eligible)
    if len(e) <= op.new_n:
        return e
    rng = np.random.default_rng(op.seed)
    return sorted(rng.choice(e, size=op.new_n, replace=False).tolist())


# ---------------------------------------------------------------- statistics

def block_index(n: int, block: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block))
    st = rng.integers(0, n, size=(n_boot, nb))
    return ((st[:, :, None] + np.arange(block)) % n).reshape(n_boot, -1)[:, :n]


def log_returns(D: pd.Series, e0: float) -> np.ndarray:
    return np.diff(np.log(np.r_[e0, D.to_numpy()]))


def gate_growth(D: pd.Series, e0: float, op: OParams, seed: int) -> dict:
    x = log_returns(D, e0)
    idx = block_index(len(x), op.block_days, op.n_boot, seed)
    bs = x[idx].mean(1) * 365.25
    a = op.alpha / op.n_tests
    lo = float(np.quantile(bs, a))
    return {"growth": float(x.mean() * 365.25), "lo": lo, "p": float((1 + np.sum(bs <= 0)) / (1 + len(bs))), "pass": bool(lo > 0)}


def gate_calmar(Da: pd.Series, Db: pd.Series, e0: float, op: OParams, seed: int) -> dict:
    """Calmar(a) - Calmar(b), paired circular block bootstrap."""
    from .turtle_portfolio import calmar_paths
    x = log_returns(Da, e0); y = log_returns(Db, e0)
    n = min(len(x), len(y)); x, y = x[:n], y[:n]
    est = float(calmar_paths(x[None])[0] - calmar_paths(y[None])[0])
    idx = block_index(n, op.block_days, op.n_boot, seed)
    bs = calmar_paths(x[idx]) - calmar_paths(y[idx]); bs = bs[np.isfinite(bs)]
    a = op.alpha / op.n_tests
    lo = float(np.quantile(bs, a))
    return {"calmar_diff": est, "lo": lo, "p": float((1 + np.sum(bs <= 0)) / (1 + len(bs))), "pass": bool(lo > 0)}


def verdict(a: bool, b: bool) -> str:
    return "HOLDS" if (a and b) else ("EDGE_ONLY" if a else ("REVERSAL_ONLY" if b else "FAILS"))


def equal_weight_hold(closes: dict) -> pd.Series:
    """Equal-weight, daily-rebalanced buy-and-hold of the coins trading each day (log returns)."""
    L = pd.DataFrame({a: np.log(c).diff() for a, c in closes.items()})
    return L.mean(axis=1, skipna=True).dropna()
