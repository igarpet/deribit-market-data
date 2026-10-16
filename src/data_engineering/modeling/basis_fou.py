"""Pairs fOU 02: the thesis fOU pipeline on spreads tied together BY CONSTRUCTION (Binance).

Protocol: discussions/Pairs_fOU/003_claude_pf02_basis_preregistration.md.

Families (y - x, beta fixed at 1 by construction; no Engle-Granger estimation):
    SP  spot-perp basis       y = USD-M perpetual, x = spot              (funding on y)
    PQ  calendar basis        y = front USD-M quarterly, x = perpetual   (funding on x)
The formation screen replaces Engle-Granger with ADF on the basis level (p <= 1%, unit root rejected);
everything else (Hurst by change of frequency, fOU moments, structural and conditional T70, |Z| >= 1.5,
next-bar execution, convergence/expiry exits) is the PF01 engine unchanged (fou_pairs.run_pair).

Costs (taker, per side, per leg): spot 0.10% + 1 bp = 11 bp; USD-M futures 0.05% + 1 bp = 6 bp.
Round trip in bp of ONE leg's notional: SP 34 bp, PQ 24 bp (run_pair reports per gross notional = /2).
Funding: the perp leg pays/receives the actual Binance funding rates at their timestamps.
Short spot (perp cheap vs spot) needs margin borrowing; its borrow cost is NOT modelled, so the
no-borrow side (short perp / long spot) is reported separately.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from data_engineering.modeling import fou_pairs as fp

VERSION = "PF02-0.1.0"
UNIVERSE = ("BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "LINK", "DOT", "LTC", "BCH", "UNI", "NEAR", "APT",
            "ARB", "OP", "SUI", "FIL", "ATOM", "ETC", "INJ", "AAVE", "TON", "WIF", "HBAR", "XLM", "SEI", "TIA",
            "ENA", "WLD")   # fixed before any data was seen; AVAX/TRX/HYPE excluded at Ivan's earlier request
SPOT_COST_BP = 11.0
FUT_COST_BP = 6.0
ROLL_DAYS = 14


@dataclass(frozen=True)
class Params:
    version: str = VERSION
    universe: tuple = UNIVERSE
    adf_p: float = 0.01
    spot_cost_bp: float = SPOT_COST_BP
    fut_cost_bp: float = FUT_COST_BP
    roll_days_before_expiry: int = ROLL_DAYS
    status: str = "HISTORICAL_EXPLORATORY"


def registration() -> dict:
    return {**asdict(Params()), "engine": fp.registration()}


def screen_basis(fy: np.ndarray, fx: np.ndarray, spec) -> dict:
    """Construction-tied screen: basis = y - x; ADF on the basis level rejects a unit root at 1%."""
    import warnings
    from statsmodels.tsa.stattools import adfuller
    warnings.filterwarnings("ignore", module="statsmodels")
    warnings.filterwarnings("ignore", category=FutureWarning)
    r = fy - fx
    out = {"integration_ok": True, "a": 0.0, "b": 1.0, "resid": r}
    try:
        p = float(adfuller(r, autolag="AIC")[1])
    except Exception as exc:
        return {**out, "coint_ok": False, "coint_p": np.nan, "error": repr(exc)}
    return {**out, "coint_p": p, "coint_ok": bool(p <= Params().adf_p)}


# ---------------------------------------------------------------- data

def quarterly_expiries(start: str, end: str) -> list[pd.Timestamp]:
    """Last Friday of Mar/Jun/Sep/Dec, 08:00 UTC."""
    out = []
    for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 2):
        for m in (3, 6, 9, 12):
            last = pd.Timestamp(year=y, month=m, day=1, tz="UTC") + pd.offsets.MonthEnd(0)
            while last.weekday() != 4:
                last -= pd.Timedelta(days=1)
            out.append(last + pd.Timedelta(hours=8))
    return [e for e in out if pd.Timestamp(start, tz="UTC") <= e <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=200)]


def front_series(contracts: dict, roll_days: int = ROLL_DAYS) -> pd.Series:
    """contracts: {expiry Timestamp: 5m log-close Series}. Continuous front-quarter series: at each bar the
    earliest expiry at least roll_days away. The first bar after every roll is NaN, so no PF formation window
    (which must be gap-free) ever mixes two contracts."""
    if not contracts:
        return pd.Series(dtype=float)
    idx = sorted(set().union(*[s.index for s in contracts.values()]))
    idx = pd.DatetimeIndex(idx)
    exps = sorted(contracts)
    val = np.full(len(idx), np.nan)
    which = np.full(len(idx), -1)
    for k, e in enumerate(exps):
        s = contracts[e].reindex(idx)
        use = (which == -1) & (idx <= e - pd.Timedelta(days=roll_days)) & s.notna().to_numpy()
        val[use] = s.to_numpy()[use]
        which[use] = k
    switch = np.r_[False, (which[1:] != which[:-1]) & (which[:-1] >= 0) & (which[1:] >= 0)]
    val[switch] = np.nan
    return pd.Series(val, index=idx)


def funding_series(df: pd.DataFrame) -> pd.Series:
    """Binance fundingRate files: calc_time (ms), last_funding_rate. Index = funding time (UTC)."""
    t = pd.to_numeric(df["calc_time"], errors="coerce")
    unit = "us" if t.max() > 10 ** 14 else "ms"
    s = pd.Series(pd.to_numeric(df["last_funding_rate"], errors="coerce").to_numpy(float),
                  index=pd.to_datetime(t, unit=unit, utc=True).dt.round("min"))
    return s[~s.index.duplicated()].sort_index().dropna()


def deviation_scale(P: pd.DataFrame, y: str, x: str, window: int = 2016) -> pd.Series:
    """Descriptive: rolling 7-day s.d. of the basis in bp, i.e. what a 1.5-sigma entry is worth before costs."""
    b = (P[y] - P[x]) * 1e4
    return b.rolling(window, min_periods=window // 2).std()


# ---------------------------------------------------------------- synthetic

def synthetic_basis(n_days: int = 14, seed: int = 0, kappa: float = 0.1, sd_bp: float = 20.0, H: float = 0.5):
    """Spot random walk; perp = spot + mean-reverting basis (sd in bp); 8-hourly funding 1e-4."""
    rng = np.random.default_rng(seed)
    n = n_days * 288
    t = pd.date_range("2026-01-01 00:05", periods=n, freq="5min", tz="UTC")
    spot = np.log(100) + np.cumsum(0.0015 * rng.standard_normal(n))
    sd = sd_bp / 1e4
    e = rng.standard_normal(n)
    b = np.zeros(n)
    for i in range(1, n):
        b[i] = b[i - 1] - kappa * b[i - 1] + sd * np.sqrt(2 * kappa) * e[i]
    P = pd.DataFrame({"PERP_AAA": spot + b, "SPOT_AAA": spot}, index=t)
    f = pd.Series(1e-4, index=pd.date_range("2026-01-01 08:00", periods=n_days * 3, freq="8h", tz="UTC"))
    return P, f
