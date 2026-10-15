"""CS03e: does passive (limit-order) execution rescue the CS03c perp flow signal?

Protocol: discussions/Candle_Streaks/003e_claude_passive_fill_preregistration.md.

For each CS03c TEST signal (|P(up)-P(down)| >= TRAIN q90, price_vol_signed, perp target) a limit order is
rested k bp behind the last trade at the 5m bar end t:   L = P0 * (1 - side * k / 1e4).
It lives for the forecast bar (t, t+5m]. It FILLS only on a strict trade-through (a buy needs a trade
below L; trades AT L are assumed to be ahead of us in the queue). The position is closed at the first
trade at or after t+H (H = 5 primary, 15 secondary). Unfilled orders contribute zero.

Benchmarks on the same origins: taker entry (first trade after t), and a random-side passive order on
every TEST origin (the adverse selection a passive order suffers with no information).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "CS03e-0.1.0"
MIN_MS = 60_000


@dataclass(frozen=True)
class Specification:
    version: str = VERSION
    assets: tuple = ("BTC", "ETH")
    window_start: str = "2026-07-01"           # inside CS03c TEST (2026-05-17 -> 2026-10-10)
    window_end: str = "2026-09-30"             # inclusive
    offsets_bp: tuple = (0.0, 1.0, 2.0, 5.0)
    fill_window_min: int = 5                   # order lives for the forecast bar only
    hold_min: tuple = (5, 15)                  # exit at first trade >= t + H
    primary_hold: int = 5
    markout_s: tuple = (60, 300)
    # costs per side, bp
    maker_fee_bp: float = 2.0                  # Binance USD-M regular maker 0.02%
    taker_cost_bp: float = 6.0                 # 0.05% taker + 1 bp slippage (as CS03c)
    bootstrap_reps: int = 2000
    seed: int = 2607
    status: str = "HISTORICAL_EXPLORATORY"


COSTS = {  # (entry bp, exit bp) per filled order
    "maker_in_taker_out": ("maker", "taker"),  # GATE: realistic retail
    "maker_both": ("maker", "maker"),          # optimistic: assumes passive exit at the mark
    "zero": (None, None),                      # pure gross / adverse-selection measure
}


def registration() -> dict:
    return asdict(Specification())


def cost_bp(name: str, spec: Specification) -> float:
    m = {"maker": spec.maker_fee_bp, "taker": spec.taker_cost_bp, None: 0.0}
    a, b = COSTS[name]
    return m[a] + m[b]


# ---------------------------------------------------------------- trade data

def to_ms(x) -> np.ndarray:
    """UTC epoch milliseconds, independent of the datetime64 resolution (ns/us/ms)."""
    s = pd.to_datetime(pd.Series(x), utc=True)
    return ((s - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).to_numpy("int64")


def day_path(root: Path, symbol: str, d) -> Path:
    return root / symbol / f"{symbol}-aggTrades-{pd.Timestamp(d):%Y-%m-%d}.parquet"


def load_window(root: Path, symbol: str, d, pre_min: int = 15, post_min: int = 45) -> tuple[np.ndarray, np.ndarray]:
    """Trades of UTC day d plus the tail of d-1 and head of d+1 (for P0 and exits across midnight)."""
    d = pd.Timestamp(d).normalize()
    parts = []
    for dd, lo, hi in ((d - pd.Timedelta(days=1), d - pd.Timedelta(minutes=pre_min), d),
                       (d, d, d + pd.Timedelta(days=1)),
                       (d + pd.Timedelta(days=1), d + pd.Timedelta(days=1),
                        d + pd.Timedelta(days=1, minutes=post_min))):
        p = day_path(root, symbol, dd)
        if not p.exists():
            if dd == d:
                raise FileNotFoundError(p)
            continue
        x = pd.read_parquet(p, columns=["time_ms", "price"])
        lo_ms, hi_ms = int(to_ms([lo])[0]), int(to_ms([hi])[0])
        x = x[(x.time_ms >= lo_ms) & (x.time_ms < hi_ms)]
        parts.append(x)
    x = pd.concat(parts, ignore_index=True).sort_values("time_ms", kind="stable")
    return x.time_ms.to_numpy("int64"), x.price.to_numpy("float64")


# ---------------------------------------------------------------- simulation

def simulate(origin_ms: np.ndarray, side: np.ndarray, t: np.ndarray, p: np.ndarray, *, k_bp: float,
             fill_min: int, hold_min: int, markout_s=(60, 300)) -> pd.DataFrame:
    """One row per order. Log returns signed by side; NaN where undefined."""
    n = len(origin_ms)
    out = {k: np.full(n, np.nan) for k in ("p0", "limit", "fill_ms", "exit_px", "gross_passive",
                                            "gross_taker", *[f"markout_{s}s" for s in markout_s])}
    filled = np.zeros(n, bool)
    for i in range(n):
        o, s = int(origin_ms[i]), float(side[i])
        j0 = np.searchsorted(t, o, "right")              # first trade strictly after o
        if j0 == 0:
            continue
        p0 = p[j0 - 1]
        L = p0 * (1.0 - s * k_bp / 1e4)
        j1 = np.searchsorted(t, o + fill_min * MIN_MS, "right")
        je = np.searchsorted(t, o + hold_min * MIN_MS, "left")
        if je >= len(t):
            continue                                      # exit not observable
        ex = p[je]
        out["p0"][i], out["limit"][i], out["exit_px"][i] = p0, L, ex
        if j0 < len(t):
            out["gross_taker"][i] = s * (np.log(ex) - np.log(p[j0]))
        w = p[j0:j1]
        hit = (w < L) if s > 0 else (w > L)
        if hit.any():
            jf = j0 + int(np.argmax(hit))
            filled[i] = True
            out["fill_ms"][i] = t[jf]
            out["gross_passive"][i] = s * (np.log(ex) - np.log(L))
            for ms in markout_s:
                jm = np.searchsorted(t, t[jf] + ms * 1000, "right") - 1
                out[f"markout_{ms}s"][i] = s * (np.log(p[jm]) - np.log(L))
    df = pd.DataFrame(out)
    df.insert(0, "side", side)
    df.insert(0, "origin_ms", origin_ms)
    df["filled"] = filled
    return df


def day_table(orders: pd.DataFrame, cost_total_bp: float, passive: bool = True) -> pd.DataFrame:
    """Per-day sums in bp. orders needs date, filled, gross_passive / gross_taker."""
    x = orders.copy()
    if passive:
        x["n_trades"] = x["filled"].astype(int)
        x["gross_bp"] = np.where(x["filled"], 1e4 * x["gross_passive"], 0.0)
    else:
        x["n_trades"] = np.isfinite(x["gross_taker"]).astype(int)
        x["gross_bp"] = np.where(x["n_trades"] == 1, 1e4 * x["gross_taker"], 0.0)
    x["net_bp"] = x["gross_bp"] - x["n_trades"] * cost_total_bp
    return x.groupby("date").agg(orders=("side", "size"), trades=("n_trades", "sum"),
                                 gross_bp=("gross_bp", "sum"), net_bp=("net_bp", "sum")).reset_index()


def summarize(days: pd.DataFrame, reps: int, seed: int, alpha: float = 0.05) -> dict:
    v = days["net_bp"].to_numpy(float)
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), size=(reps, len(v)))].mean(1) if len(v) >= 20 else np.array([np.nan])
    tr = days["trades"].sum()
    return {"days": int(len(days)), "orders": int(days["orders"].sum()), "trades": int(tr),
            "fill_rate": float(tr / days["orders"].sum()) if days["orders"].sum() else np.nan,
            "gross_bp_per_trade": float(days["gross_bp"].sum() / tr) if tr else np.nan,
            "net_bp_per_trade": float(days["net_bp"].sum() / tr) if tr else np.nan,
            "net_bp_per_day": float(v.mean()) if len(v) else np.nan,
            "net_bp_per_day_lo": float(np.nanquantile(b, alpha / 2)),
            "net_bp_per_day_hi": float(np.nanquantile(b, 1 - alpha / 2))}


def decision(summary: pd.DataFrame, spec: Specification | None = None, *, synthetic=False) -> dict:
    """GATE (frozen): for BTC and ETH, H = 5, cost maker_in_taker_out, signal orders: some k in offsets_bp has
    net bp/day lower bound > 0 at the Bonferroni level (alpha = .05 / len(offsets)), the SAME k for both."""
    spec = spec or Specification()
    if synthetic:
        return {"verdict": "SYNTHETIC_NO_MARKET_INFERENCE"}
    s = summary[(summary.arm == "signal") & (summary.hold_min == spec.primary_hold) &
                (summary.cost == "maker_in_taker_out") & (summary.kind == "passive")]
    rows, passing = [], []
    for k in spec.offsets_bp:
        ok = True
        for a in spec.assets:
            r = s[(s.asset == a) & (s.k_bp == k)]
            good = len(r) == 1 and np.isfinite(r.iloc[0].net_bp_per_day_lo_bonf) and r.iloc[0].net_bp_per_day_lo_bonf > 0
            rows.append({"asset": a, "k_bp": k, "net_bp_per_day": float(r.iloc[0].net_bp_per_day) if len(r) else None,
                         "lower_bonferroni": float(r.iloc[0].net_bp_per_day_lo_bonf) if len(r) else None, "pass": bool(good)})
            ok &= good
        if ok:
            passing.append(k)
    return {"verdict": "PASSIVE_CANDIDATE" if passing else "NO_PASSIVE_EDGE", "passing_offsets_bp": passing,
            "gates": rows, "status": spec.status, "prospective_confirmation": False,
            "note": "Fill on strict trade-through ignores queue position and latency: an UPPER bound on fills."}


# ---------------------------------------------------------------- synthetic

def synthetic_trades(n_min: int = 600, seed: int = 0, per_min: int = 60, start="2026-07-01"):
    rng = np.random.default_rng(seed)
    n = n_min * per_min
    t0 = int(to_ms([pd.Timestamp(start, tz="UTC")])[0])
    t = t0 + np.sort(rng.integers(0, n_min * MIN_MS, n)).astype("int64")
    p = 100.0 * np.exp(np.cumsum(2e-5 * rng.standard_normal(n)))
    return t, np.round(p, 3)
