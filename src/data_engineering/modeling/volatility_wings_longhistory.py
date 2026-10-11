"""
Volatility Dynamics Module 10c: long-history physical tail for 0DTE wings.

Why
---
Module 10b found the wings RICH against realized outcomes, but the outcome
pool was 18 sessions. That is a peso problem: one ±8σ day per asset-year
would erase the bid-side premium. 10c replaces the pool with about two years
of out-of-sample history.

Outcome pool
------------
Clock: D uses exactly Module 10's clock (Module 08's, fitted on the
research-DB index), so pool outcomes and quotes are standardized
identically. A clock refitted on the long history is reported as a
sensitivity check. This was fixed after a synthetic run, before any real
data: with a refitted clock the gate failed on the synthetic DB even with
identical prices, because the two clocks differed.

5-minute closes from tools/download_price_history.py. The primary source is
Binance spot <ASSET>USDT, liquid constituents of the Deribit index; the
fallback is the Deribit perpetual. Every outcome is built exactly as in 10b,
using Module 08's clock, the backward 4h bipower and settlement-TWAP weights:

    u_t = ln(S_T / I_t) / sqrt(D_t)

The pool holds only sessions dated <= POOL_LAST_SESSION (2026-09-19), i.e.
strictly before the first Module 10 quote. It is therefore fully out of
sample and no LOSO is needed.

Consistency gate (pre-registered)
---------------------------------
On the quote window, the long-history u is matched with the index-based u of
Module 10b by (asset, session, minutes to expiry). An asset PASSES iff
corr >= 0.90 and sd ratio is in [0.85, 1.15].
- Failing assets are removed from the pool.
- If more than 2 assets fail, every verdict is GATE_FAIL.

Pre-registered decisions (fixed before any real-data Module 10c output)
-----------------------------------------------------------------------
Rows: Module 10's primary sample, unchanged (10/10_priced_quotes.csv.gz).

Price, one per row from the pooled long-history u:

    P = sum_i w_i payoff(F e^{sqrt(D) u_i} / mbar, K)

Here D is Module 10's quote D. The outer 0.1% of outcomes on each side are
kept raw; the centre is 500 bin means.

V_bid (DECISION): M_bid = median over sessions of the session median of
ln(bid / P). This is what a seller receives.
V_mid (continuity with 10/10b): the same statistic with the mid.

Uncertainty: joint bootstrap, B = 1000. Each replicate resamples quote
sessions and history dates independently; a history date resamples all
assets together, which keeps the common jump component.
- RICH iff the 2.5% quantile > 0; CHEAP iff the 97.5% quantile < 0;
  CONSISTENT otherwise.
- Zero-price rows are dropped; INSUFFICIENT_TAIL if more than 25% are dropped.

Consequence stated in advance:
- V_bid = RICH: proceed to a pre-registered, defined-risk wing-selling
  backtest (Module 11), gated on expected shortfall, not on mean P&L.
- Any other V_bid: no wing-selling strategy research from this branch until
  more quote sessions exist.

Reported, no decision:
- the bid net of fees (Deribit schedule: min(0.03% of F, 12.5% of premium);
  verify the current schedule);
- per asset; z x TTE buckets; TTE-conditioned pool; own-asset pool;
- tail frequencies overall and by volatility tercile;
- break-even unobserved-disaster mass at +/- 6, 8, 10 sigma, for bid and mid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from data_engineering.modeling import volatility_vd_b_core as vb
from data_engineering.modeling import volatility_jump_compound as jc
from data_engineering.modeling import volatility_jump_option_wings as ow
from data_engineering.modeling import volatility_wings_empirical as we


VERSION = "0.1.0"
POOL_LAST_SESSION = "2026-09-19"
GATE_MIN_CORR = 0.90
GATE_SD_RATIO = (0.85, 1.15)
GATE_MAX_FAILS = 2
TAIL_RAW_LONG = 0.001
N_BOOT = 1000
FEE_RATE_UNDERLYING = 0.0003
FEE_CAP_PREMIUM = 0.125
BREAKEVEN_USTAR = (6.0, 8.0, 10.0)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_history(directory: Path, assets: Sequence[str]) -> pd.DataFrame:
    frames = []
    for a in assets:
        p = Path(directory) / f"{a}.csv.gz"
        if not p.exists():
            continue
        h = pd.read_csv(p)
        h["underlying"] = a
        frames.append(h)
    if not frames:
        raise FileNotFoundError(f"No history files in {directory}")
    h = pd.concat(frames, ignore_index=True)
    h["state_time"] = pd.to_datetime(h["state_time"], utc=True)
    return h


def return_frame(h: pd.DataFrame) -> pd.DataFrame:
    """5m log returns on consecutive bars only; `futures_index_price` := close (the I_t of 10b)."""
    out = []
    for a, g in h.sort_values(["underlying", "state_time"]).groupby("underlying"):
        g = g.drop_duplicates("state_time")
        t = g["state_time"]
        c = pd.to_numeric(g["close"], errors="coerce")
        consec = t.diff().eq(pd.Timedelta(minutes=5))
        r = np.log(c / c.shift(1)).where(consec)
        out.append(pd.DataFrame({"state_time": t, "underlying": a, "return_5m": r,
                                 "futures_index_price": c, "source": g.get("source", "unknown")}))
    f = pd.concat(out, ignore_index=True)
    return f[np.isfinite(f["futures_index_price"])].reset_index(drop=True)


def module10_clock(db_grid: vb.Grid) -> dict:
    """{asset: (288,) clock} exactly as Module 10 used it (Module 08 clock on the research-DB index)."""
    panel, S, *_ = jc.detect(db_grid)
    return {a: S[:, i].copy() for i, a in enumerate(panel.assets)}


def long_outcomes(frame: pd.DataFrame, clock: dict | None = None) -> pd.DataFrame:
    """Outcomes on the long history. `clock` (from module10_clock) makes D identical in
    construction to the quotes' D; without it the clock is refitted on the long history."""
    grid = vb.build_grid(frame)
    panel, S, *_ = jc.detect(grid)
    if clock is not None:
        S = S.copy()
        for i, a in enumerate(panel.assets):
            if a in clock:
                S[:, i] = clock[a]
    BVb = ow.backward_bipower(panel, S)
    I = jc.index_panel(grid, panel)
    oc = we.origin_outcomes(panel, S, BVb, I, panel.assets)
    oc = oc[np.isfinite(oc["u"]) & np.isfinite(oc["k"])].reset_index(drop=True)
    return oc


def split(oc: pd.DataFrame, quote_sessions: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = oc[oc["session"] <= POOL_LAST_SESSION].reset_index(drop=True)
    overlap = oc[oc["session"].isin(set(quote_sessions))].reset_index(drop=True)
    return pool, overlap


def consistency_gate(overlap: pd.DataFrame, outcomes_10b: pd.DataFrame) -> pd.DataFrame:
    m = overlap.merge(outcomes_10b[["underlying", "session", "minutes_to_expiry", "u"]],
                      on=["underlying", "session", "minutes_to_expiry"], suffixes=("_long", "_10b"))
    rows = []
    for a in sorted(set(overlap["underlying"]) | set(outcomes_10b["underlying"])):
        g = m[m["underlying"] == a]
        if len(g) < 200:
            rows.append({"underlying": a, "n_matched": int(len(g)), "corr": np.nan, "sd_ratio": np.nan, "pass": False})
            continue
        c = float(np.corrcoef(g["u_long"], g["u_10b"])[0, 1])
        sr = float(g["u_long"].std() / g["u_10b"].std())
        rows.append({"underlying": a, "n_matched": int(len(g)), "corr": c, "sd_ratio": sr,
                     "pass": bool(c >= GATE_MIN_CORR and GATE_SD_RATIO[0] <= sr <= GATE_SD_RATIO[1])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Pricing and statistics
# --------------------------------------------------------------------------

def price(q: pd.DataFrame, u: np.ndarray, tail: float = TAIL_RAW_LONG, q_nodes: int = we.Q_NODES) -> np.ndarray:
    nodes, w = we.compress(u, q_nodes, tail)
    F, K, D = (q[c].to_numpy(float) for c in ("option_forward_price", "strike", "D"))
    call = q["is_call"].to_numpy(bool)
    P = np.empty(len(q))
    for s0 in range(0, len(q), 2000):
        e = slice(s0, s0 + 2000)
        P[e] = we.empirical_price(F[e], K[e], D[e], call[e], nodes, w)
    return P


def net_bid(q: pd.DataFrame) -> np.ndarray:
    F = q["option_forward_price"].to_numpy(float)
    b = q["bid_usd"].to_numpy(float)
    return b - np.minimum(FEE_RATE_UNDERLYING * F, FEE_CAP_PREMIUM * b)


def _median_inverted(v: np.ndarray) -> float:
    return float(np.quantile(v, 0.5, method="inverted_cdf"))


def session_M(q: pd.DataFrame, P: np.ndarray, price_col: str, *, weights: dict | None = None) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(q[price_col].to_numpy(float) / P)
    ok = np.isfinite(lr)
    sm = pd.Series(lr[ok]).groupby(q["session"].to_numpy()[ok]).median()
    if weights is None:
        return float(np.median(sm.to_numpy()))
    vals = np.concatenate([[sm[s]] * m for s, m in weights.items() if s in sm.index])
    return _median_inverted(vals) if len(vals) else np.nan


def _verdict(lo, hi):
    return "RICH" if lo > 0 else ("CHEAP" if hi < 0 else "CONSISTENT")


def decide(q: pd.DataFrame, pool: pd.DataFrame, *, n_boot: int = N_BOOT, seed: int = 1014,
           price_cols: Sequence[str] = ("bid_usd", "mid_usd"), gate_fail: bool = False) -> dict:
    """Point estimates + joint bootstrap (quote sessions x history dates) for each price column."""
    P = price(q, pool["u"].to_numpy())
    zero_share = float(np.mean(P <= 0))
    out = {"n_rows": int(len(q)), "n_sessions": int(q["session"].nunique()),
           "n_pool_origins": int(len(pool)), "n_pool_dates": int(pool["session"].nunique()),
           "share_model_price_zero": zero_share}
    for c in price_cols:
        out[c] = {"M": session_M(q, P, c)}
    dates = np.array(sorted(pool["session"].unique()))
    u_by_date = {d: g["u"].to_numpy(float) for d, g in pool.groupby("session")}
    qs = np.array(sorted(q["session"].unique()))
    q_by = {s: g for s, g in q.groupby("session")}
    rng = np.random.default_rng(int(seed))
    boots = {c: np.empty(int(n_boot)) for c in price_cols}
    for b in range(int(n_boot)):
        dd = rng.choice(dates, size=len(dates), replace=True)
        ud, cnt = np.unique(dd, return_counts=True)
        u = np.concatenate([np.tile(u_by_date[d], m) for d, m in zip(ud, cnt)])
        sd = rng.choice(qs, size=len(qs), replace=True)
        us, sc = np.unique(sd, return_counts=True)
        qq = pd.concat([q_by[s] for s in us])
        Pb = price(qq, u)
        wts = dict(zip(us, sc))
        for c in price_cols:
            boots[c][b] = session_M(qq, Pb, c, weights=wts)
    for c in price_cols:
        v = boots[c][np.isfinite(boots[c])]
        lo, hi = float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))
        verdict = _verdict(lo, hi)
        if zero_share > we.MAX_ZERO_PRICE_SHARE:
            verdict = "INSUFFICIENT_TAIL"
        if gate_fail:
            verdict = "GATE_FAIL"
        out[c].update({"ci_low": lo, "ci_high": hi, "verdict": verdict, "bootstrap_verdict": _verdict(lo, hi),
                       "n_boot": int(len(v))})
    return out


def quote_only(q: pd.DataFrame, P: np.ndarray, price_col: str, n_boot: int = 1000) -> dict:
    with np.errstate(divide="ignore", invalid="ignore"):
        x = q.assign(lr=np.log(q[price_col].to_numpy(float) / P))
    x = x[np.isfinite(x["lr"])]
    r = ow.session_median_stat(x, "lr", n_boot=n_boot)
    r["share_model_price_zero"] = float(np.mean(P <= 0))
    return r


def breakeven(q: pd.DataFrame, pool_u: np.ndarray, price_col: str, ustar: float) -> float:
    """Symmetric extra mass p at +/- ustar (per origin) that sets M to 0; bisection on log p."""
    nodes, w = we.compress(pool_u, we.Q_NODES, TAIL_RAW_LONG)
    F, K, D = (q[c].to_numpy(float) for c in ("option_forward_price", "strike", "D"))
    call = q["is_call"].to_numpy(bool)

    def M(p):
        n2 = np.concatenate([nodes, [ustar, -ustar]])
        w2 = np.concatenate([w * (1 - p), [p / 2, p / 2]])
        P = np.empty(len(q))
        for s0 in range(0, len(q), 2000):
            e = slice(s0, s0 + 2000)
            P[e] = we.empirical_price(F[e], K[e], D[e], call[e], n2, w2)
        return session_M(q, P, price_col)

    if M(0.0) <= 0:
        return 0.0
    lo, hi = 1e-6, 0.2
    if M(hi) > 0:
        return np.inf
    for _ in range(16):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if M(mid) > 0 else (lo, mid)
    return float(np.sqrt(lo * hi))


def vol_tercile_tails(pool: pd.DataFrame) -> pd.DataFrame:
    p = pool.copy()
    p["vol_tercile"] = p.groupby("underlying")["v"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str))
    rows = []
    for t, g in p.groupby("vol_tercile"):
        u = g["u"].to_numpy(float)
        rows.append({"vol_tercile": t, "n_origins": int(len(u)), "u_sd": float(u.std()),
                     **{f"P_abs_u_gt_{x:g}": float(np.mean(np.abs(u) > x)) for x in we.TAIL_X}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Self test
# --------------------------------------------------------------------------

def selftest() -> dict:
    out = {}
    rng = np.random.default_rng(10)
    # 1. return_frame only differences consecutive bars.
    t = pd.to_datetime(["2026-01-01 00:05", "2026-01-01 00:10", "2026-01-01 00:20"], utc=True)
    h = pd.DataFrame({"state_time": t, "close": [100.0, 101.0, 102.0], "underlying": "X", "source": "s"})
    rf = return_frame(h)
    out["returns_consecutive_only"] = bool(np.isnan(rf["return_5m"].iloc[0]) and np.isclose(rf["return_5m"].iloc[1], np.log(1.01))
                                           and np.isnan(rf["return_5m"].iloc[2]))
    # 2. Fat-tailed world: long pool from the true law -> CONSISTENT at mid; thin pool -> RICH.
    t3 = lambda n: rng.standard_t(3, n) / np.sqrt(3.0)
    q = we._fake_quotes(rng, 16, 60, t3, noise=0.05)
    q["bid_usd"] = q["mid_usd"] * 0.95
    pool_ok = pd.DataFrame({"session": np.repeat([f"h{d:03d}" for d in range(300)], 200), "u": t3(60_000)})
    pool_thin = pd.DataFrame({"session": np.repeat([f"h{d:03d}" for d in range(300)], 200), "u": rng.standard_normal(60_000)})
    r_ok = decide(q, pool_ok, n_boot=100, price_cols=("mid_usd",))
    r_thin = decide(q, pool_thin, n_boot=100, price_cols=("mid_usd",))
    out["true_pool_consistent"] = r_ok["mid_usd"]["verdict"] == "CONSISTENT"
    out["thin_pool_rich"] = r_thin["mid_usd"]["verdict"] == "RICH"
    # 3. Gate: identical u passes; scaled u fails.
    o = pd.DataFrame({"underlying": "A", "session": "s", "minutes_to_expiry": np.arange(500.0), "u": rng.standard_normal(500)})
    g1 = consistency_gate(o, o)
    g2 = consistency_gate(o.assign(u=o["u"] * 1.3), o)
    out["gate_pass_identical"] = bool(g1["pass"].all())
    out["gate_fail_scaled"] = bool(~g2["pass"].all())
    # 4. Break-even: positive for a rich book, 0 when already cheap.
    be = breakeven(q.assign(mid_usd=q["mid_usd"] * 3), pool_ok["u"].to_numpy(), "mid_usd", 8.0)
    out["breakeven_positive_when_rich"] = bool(0 < be < 0.2)
    out["breakeven_zero_when_cheap"] = breakeven(q.assign(mid_usd=q["mid_usd"] * 0.3), pool_ok["u"].to_numpy(), "mid_usd", 8.0) == 0.0
    # 5. Fees reduce the bid, never below 87.5% of it.
    nb = net_bid(q)
    out["fees_bounded"] = bool(np.all(nb <= q["bid_usd"]) and np.all(nb >= 0.875 * q["bid_usd"] - 1e-12))
    return out


if __name__ == "__main__":
    print(selftest())
