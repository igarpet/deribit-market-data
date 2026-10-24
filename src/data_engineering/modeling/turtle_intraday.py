"""Turtle Breakout 03: intraday versions of the turtle breakout and volatility-engine position sizing.

Protocol: discussions/Turtle_Breakout/005_claude_tb03_preregistration.md (fixed before any real-data output).

Nine strategies = 3 timeframes x 3 sizing modes, all long only on BTC/ETH/SOL/XRP perpetuals, 2% HWM risk,
fees + slippage + funding, 3x per-position cap, window COMMON_START -> end of data (turtle_breakout.simulate).

Timeframes (Ivan, 2026-10-25: "set the 20 day window to a 4 hour window", both readings tested)
    daily        original: 20-day high entry, 10-day low trailing stop, ATR over 20 days (TB 01)
    h4_candles   same rules on 4-hour bars (UTC 00/04/../20): 20-bar (80 h) high, 10-bar (40 h) low, ATR 20 bars
    last_4h      the window itself is 4 hours: 1-hour bars, entry = high of the last 4 hours, trailing stop = low
                 of the last 2 hours, ATR over 4 hours (the 20:10:20 proportions kept)
Sizing (Ivan: both engine roles on all tests)
    atr          units = 0.02 HWM / (2 ATR); fixed stop = entry - 2 ATR                    (original)
    engine       units = 0.02 HWM / (2 U);   fixed stop = entry - 2 ATR                    (engine sizes only)
    engine_stop  units = 0.02 HWM / (2 U);   fixed stop = entry - 2 U                      (engine sizes and stops)
    U = c x sigma_engine x price, sigma_engine = frozen VD18 HAR_4 sd of the log return over one bar of the
    timeframe (1440 / 240 / 60 min) as of the last 5m bar before entry; c = causal expanding median of
    ATR / (sigma_engine x price) over past periods, so the engine changes WHEN risk is large, not its average scale.

Per strategy (9 strategies -> Bonferroni alpha = 0.05 / 9)
    G1 beats luck        share of random-entry portfolios (same exits, sizing, costs) with CAGR >= real < alpha,
                         and the strategy's own net CAGR > 0
    G2 beats benchmarks  Sharpe diff vs B1 perp monthly, B2 perp daily, B3 spot proxy monthly (TB 02), each with
                         a two-sided (1 - alpha) circular 21-day block-bootstrap interval whose lower bound > 0;
                         benchmark leverage = min(strategy mean leverage, 1) (Sharpe is ~leverage-free; the cap
                         avoids ruining a 4x buy-and-hold)
Contrasts (8, Bonferroni alpha = 0.05 / 8, paired Sharpe differences)
    C1-C3 engine - atr       for each timeframe
    C4-C6 engine_stop - atr  for each timeframe
    C7    h4_candles - daily (atr)
    C8    last_4h - daily (atr)
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from . import turtle_breakout as tb
from . import turtle_benchmarks as bm

VERSION = "TB03-0.1.0"
TIMEFRAMES = {"daily": dict(period="day"),
              "h4_candles": dict(period="h4"),
              "last_4h": dict(period="hour", entry_n=4, exit_n=2, atr_n=4)}
SIZINGS = ("atr", "engine", "engine_stop")
N_NULL = {"daily": 1000, "h4_candles": 400, "last_4h": 200}
CONTRASTS = ([(f"C{i + 1}", f"{tf}|engine", f"{tf}|atr") for i, tf in enumerate(TIMEFRAMES)]
             + [(f"C{i + 4}", f"{tf}|engine_stop", f"{tf}|atr") for i, tf in enumerate(TIMEFRAMES)]
             + [("C7", "h4_candles|atr", "daily|atr"), ("C8", "last_4h|atr", "daily|atr")])


@dataclass(frozen=True)
class CParams:
    n_strategies: int = 9
    n_contrasts: int = 8
    alpha: float = 0.05
    block_days: int = 21
    n_boot: int = 2000
    seed: int = 20261026
    bench_lev_cap: float = 1.0
    status: str = "PREREGISTERED_TB005"


def registration(cp: CParams | None = None) -> dict:
    cp = cp or CParams()
    d = asdict(cp)
    d.update(version=VERSION, timeframes=TIMEFRAMES, sizings=list(SIZINGS), n_null=N_NULL,
             contrasts=[{"id": c, "a": a, "b": b} for c, a, b in CONTRASTS],
             base=asdict(tb.Params()),
             rules={"G1": "random-entry share CAGR >= real < alpha/9 and net CAGR > 0",
                    "G2": "Sharpe diff vs B1, B2, B3: (1-alpha/9) interval lower bound > 0, all three",
                    "contrasts": "paired Sharpe diff, (1-alpha/8) interval excludes 0"})
    return d


def strategy_params(tf: str, sizing: str) -> tb.Params:
    return replace(tb.Params(), sizing=sizing, **TIMEFRAMES[tf])


def sharpe_diff_q(Ea: pd.Series, Eb: pd.Series, q: float, cp: CParams, seed: int) -> dict:
    a = tb.daily(Ea).pct_change(); b = tb.daily(Eb).pct_change()
    x = pd.concat([a, b], axis=1).dropna().to_numpy()
    n = len(x)
    sh = lambda z: z.mean(0) / z.std(0) * math.sqrt(365)
    est = sh(x)
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / cp.block_days))
    bs = np.empty(cp.n_boot)
    for i in range(cp.n_boot):
        st = rng.integers(0, n, nb)
        idx = ((st[:, None] + np.arange(cp.block_days)) % n).ravel()[:n]
        s = sh(x[idx]); bs[i] = s[0] - s[1]
    lo, hi = np.quantile(bs, [q / 2, 1 - q / 2])
    return {"sharpe_a": float(est[0]), "sharpe_b": float(est[1]), "diff": float(est[0] - est[1]),
            "lo": float(lo), "hi": float(hi), "ci_level": 1 - q, "days": int(n)}


def strategy_verdict(g1: bool, g2: bool) -> str:
    return ("BEATS_LUCK_AND_BENCHMARKS" if g1 and g2 else "BEATS_LUCK_ONLY" if g1
            else "BEATS_BENCHMARKS_ONLY" if g2 else "NO_EDGE")


def evaluate_strategy(name: str, E: pd.Series, T: pd.DataFrame, st: dict, null: pd.DataFrame, C: pd.DataFrame,
                      F: pd.DataFrame, cost: float, cp: CParams) -> tuple[dict, dict]:
    a = cp.alpha / cp.n_strategies
    m = tb.metrics(E, T)
    share = float((null.cagr >= m["cagr"]).mean())
    L = min(st["mean_gross_leverage"], cp.bench_lev_cap)
    bench = {"B1": bm.constant_mix(C, F, L, "M", True, cost), "B2": bm.constant_mix(C, F, L, "D", True, cost),
             "B3": bm.constant_mix(C, F, L, "M", False, cost)}
    tests = {k: sharpe_diff_q(E, v, a, cp, cp.seed + i) for i, (k, v) in enumerate(bench.items())}
    g1 = share < a and m["cagr"] > 0          # beating ruined random portfolios while losing money is not an edge
    g2 = all(t["lo"] > 0 for t in tests.values())
    row = {"strategy": name, **{k: m.get(k) for k in ("cagr", "sharpe", "max_dd", "trades", "win_rate",
                                                     "profit_factor", "avg_hold_days", "fees", "funding")},
           "mean_leverage": st["mean_gross_leverage"], "max_leverage": st["max_gross_leverage"],
           "engine_used": sum(st["engine_used"].values()), "engine_fallback": sum(st["engine_fallback"].values()),
           "leverage_capped_entries": sum(st["leverage_capped"].values()),
           "null_runs": len(null), "null_cagr_median": float(null.cagr.median()), "null_share_cagr_ge_real": share,
           **{f"{k}_diff": t["diff"] for k, t in tests.items()}, **{f"{k}_lo": t["lo"] for k, t in tests.items()},
           "bench_leverage": L, "G1": "PASS" if g1 else "FAIL", "G2": "PASS" if g2 else "FAIL",
           "verdict": strategy_verdict(g1, g2)}
    return row, tests


def contrasts(E: dict, cp: CParams) -> pd.DataFrame:
    q = cp.alpha / cp.n_contrasts
    rows = []
    for i, (cid, a, b) in enumerate(CONTRASTS):
        r = sharpe_diff_q(E[a], E[b], q, cp, cp.seed + 100 + i)
        sig = "A_BETTER" if r["lo"] > 0 else "B_BETTER" if r["hi"] < 0 else "NO_DIFFERENCE"
        rows.append({"contrast": cid, "a": a, "b": b, **r, "result": sig})
    return pd.DataFrame(rows)
