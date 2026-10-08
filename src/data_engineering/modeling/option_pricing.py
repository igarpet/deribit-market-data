"""
Black-76 pricing utilities for the Volatility Risk Premium branch.

Deribit option prices are forward-style. For a family with forward F, strike
K, time to expiry T (years) and volatility sigma:

    d1 = (ln(F/K) + sigma^2 T / 2) / (sigma sqrt(T)),  d2 = d1 - sigma sqrt(T)
    call_usd = F N(d1) - K N(d2),   put_usd = K N(-d2) - F N(-d1)

Discounting is ignored (0DTE: the rate effect is below a basis point of
premium). Quote units depend on the family:

    linear  (USDC-settled): price quoted in USD  -> native = usd
    inverse ("reversed", coin-settled): quoted in coin -> native = usd / F

All functions are vectorized with numpy (scipy.special.ndtr when available,
an exact per-element math.erf fallback otherwise).
"""

from __future__ import annotations

import math

import numpy as np


OPTION_PRICING_VERSION = "0.1"
SQRT2 = math.sqrt(2.0)


try:  # scipy is already a project dependency (sdf_hmm, vol_hmm_null)
    from scipy.special import ndtr as _ndtr
except Exception:  # pragma: no cover
    _ndtr = None


def norm_cdf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    if _ndtr is not None:
        return _ndtr(x)
    return 0.5 * (1.0 + _erf(x / SQRT2))


def norm_pdf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _erf(x: np.ndarray) -> np.ndarray:
    """Vectorized erf (numpy has no erf); math.erf is exact per element."""
    v = np.vectorize(math.erf, otypes=[float])
    return v(x)


def is_inverse(instrument_type) -> np.ndarray:
    s = np.asarray(instrument_type, dtype=object).astype(str)
    return np.char.find(np.char.lower(s.astype(str)), "revers") >= 0


def black76_usd(F, K, T, sigma, is_call) -> np.ndarray:
    """Undiscounted Black-76 price in USD (quote currency of the strike)."""
    F, K, T, sigma = (np.asarray(a, float) for a in (F, K, T, sigma))
    is_call = np.asarray(is_call, bool)
    sT = sigma * np.sqrt(np.clip(T, 0.0, None))
    intrinsic = np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    ok = (F > 0) & (K > 0) & (sT > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * sT * sT) / sT
        d2 = d1 - sT
        call = F * norm_cdf(d1) - K * norm_cdf(d2)
        put = K * norm_cdf(-d2) - F * norm_cdf(-d1)
    px = np.where(is_call, call, put)
    return np.where(ok, px, np.where((F > 0) & (K > 0), intrinsic, np.nan))


def to_native(price_usd, F, inverse) -> np.ndarray:
    return np.where(np.asarray(inverse, bool), np.asarray(price_usd, float) / np.asarray(F, float), price_usd)


def to_usd(price_native, F, inverse) -> np.ndarray:
    return np.where(np.asarray(inverse, bool), np.asarray(price_native, float) * np.asarray(F, float), price_native)


def black76_delta(F, K, T, sigma, is_call) -> np.ndarray:
    """dPrice_usd/dF (forward delta of the USD price)."""
    F, K, T, sigma = (np.asarray(a, float) for a in (F, K, T, sigma))
    sT = sigma * np.sqrt(np.clip(T, 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * sT * sT) / sT
    d = norm_cdf(d1)
    return np.where(np.asarray(is_call, bool), d, d - 1.0)


def black76_gamma(F, K, T, sigma) -> np.ndarray:
    F, K, T, sigma = (np.asarray(a, float) for a in (F, K, T, sigma))
    sT = sigma * np.sqrt(np.clip(T, 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * sT * sT) / sT
        return norm_pdf(d1) / (F * sT)


def implied_vol(price_usd, F, K, T, is_call, lo: float = 1e-4, hi: float = 20.0, iters: int = 80) -> np.ndarray:
    """Black-76 implied volatility by vectorized bisection. NaN where the
    price is outside the no-arbitrage band (below intrinsic or above the
    asset/strike bound)."""
    price, F, K, T = (np.asarray(a, float) for a in (price_usd, F, K, T))
    is_call = np.asarray(is_call, bool)
    intrinsic = np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    upper = np.where(is_call, F, K)
    ok = np.isfinite(price) & (F > 0) & (K > 0) & (T > 0) & (price > intrinsic) & (price < upper)
    a = np.full(price.shape, lo)
    b = np.full(price.shape, hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        pm = black76_usd(F, K, T, m, is_call)
        high = pm > price
        b = np.where(high, m, b)
        a = np.where(high, a, m)
    return np.where(ok, 0.5 * (a + b), np.nan)


def straddle_usd(F, K, T, sigma) -> np.ndarray:
    return black76_usd(F, K, T, sigma, True) + black76_usd(F, K, T, sigma, False)


def straddle_implied_vol(straddle_price_usd, F, K, T, iters: int = 80) -> np.ndarray:
    """Volatility that reproduces a call+put price at one strike."""
    p, F, K, T = (np.asarray(a, float) for a in (straddle_price_usd, F, K, T))
    ok = np.isfinite(p) & (F > 0) & (K > 0) & (T > 0) & (p > np.abs(F - K))
    a = np.full(p.shape, 1e-4)
    b = np.full(p.shape, 20.0)
    for _ in range(iters):
        m = 0.5 * (a + b)
        high = straddle_usd(F, K, T, m) > p
        b = np.where(high, m, b)
        a = np.where(high, a, m)
    return np.where(ok, 0.5 * (a + b), np.nan)
