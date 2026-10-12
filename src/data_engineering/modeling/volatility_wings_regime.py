"""
Volatility Dynamics Module 10d (VD 028): regime x time-to-expiry conditioned
benchmark for 0DTE wings, at the bid (seller) and the ask (buyer).

Why (VD 024 section 5)
----------------------
10c priced every wing quote from one pooled two-year outcome distribution and
found the bid CONSISTENT. Post hoc, the pool is a mixture: standardized tails
are about 10x fatter in the calm volatility tercile than in the volatile one,
and the standardized variance rises with time to expiry. If the market prices
that structure, conditioning changes nothing; if it does not, calm-regime
wings may be CHEAP at the ask (a long edge) or volatile-regime wings RICH at
the bid (a short edge).

Construction (everything else as 10c)
-------------------------------------
- Volatility state of a quote: its backward 4h bipower v at the quote time,
  taken from the long-history outcomes on the quote sessions (matched on
  asset, session and minutes to expiry), so quotes and pool share one
  definition of v.
- Terciles: per asset, the 1/3 and 2/3 quantiles of that asset's two-year
  pool v. Pool origins and quotes are classified with the same cutoffs.
- Cell = (tercile, TTE bucket of 10). Each quote is priced from the pool
  outcomes of its own cell (all assets pooled, as in 10c).
- Sample: 10's primary sample (z in [2, 5), not floor-bound, tick-artifact
  assets excluded) with a matched v.

Pre-registered decisions (two, Holm at family alpha = 0.05, one-sided)
---------------------------------------------------------------------
    D_long  (low tercile):  M_ask = median over sessions of session median ln(ask_net / P)
            ask_net = ask + min(0.0003 F, 0.125 ask). CHEAP (long edge) if rejected
            H0: M_ask >= 0, p = share of bootstrap M_ask >= 0.
    D_short (high tercile): M_bid = same with bid_net = bid - min(0.0003 F, 0.125 bid).
            RICH (short edge) if rejected H0: M_bid <= 0, p = share of bootstrap M_bid <= 0.
Joint bootstrap B = 1000 (quote sessions x history dates, as 10c). Holm: the
smaller p is compared with 0.025, the larger with 0.05.
Consequence stated in advance: a rejection opens a defined-risk backtest
pre-registration for that side only; no rejection closes the wing line.
Reported, no decision: raw bid/mid/ask by tercile x TTE; counts; the mid tercile.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from data_engineering.modeling import volatility_wings_longhistory as wl


VERSION = "0.1.0"
N_BOOT = 1000
SEED = 1028
ALPHA = 0.05
TERCILES = ("low", "mid", "high")


def cutoffs(pool: pd.DataFrame) -> dict:
    return {a: (float(g["v"].quantile(1 / 3)), float(g["v"].quantile(2 / 3))) for a, g in pool.groupby("underlying")}


def classify(df: pd.DataFrame, cuts: dict, col: str = "v") -> np.ndarray:
    out = np.full(len(df), "na", dtype=object)
    for a, (c1, c2) in cuts.items():
        m = (df["underlying"] == a).to_numpy() & np.isfinite(df[col].to_numpy(float))
        v = df.loc[m, col].to_numpy(float)
        out[m] = np.where(v <= c1, "low", np.where(v <= c2, "mid", "high"))
    return out


def attach_quote_v(q: pd.DataFrame, overlap: pd.DataFrame) -> pd.DataFrame:
    """Quote v from the long-history outcome at the same asset/session/minutes (5-minute rounded)."""
    x = q.copy()
    x["_m"] = (x["minutes_to_expiry"].astype(float) / 5).round().astype(int) * 5
    o = overlap[["underlying", "session", "minutes_to_expiry", "v"]].copy()
    o["_m"] = (o["minutes_to_expiry"].astype(float) / 5).round().astype(int) * 5
    o = o.drop_duplicates(["underlying", "session", "_m"]).drop(columns="minutes_to_expiry")
    return x.merge(o, on=["underlying", "session", "_m"], how="left").drop(columns="_m")


def net_prices(q: pd.DataFrame) -> pd.DataFrame:
    x = q.copy()
    F = x["option_forward_price"].to_numpy(float)
    b, a = x["bid_usd"].to_numpy(float), x["ask_usd"].to_numpy(float)
    x["bid_net_usd"] = b - np.minimum(wl.FEE_RATE_UNDERLYING * F, wl.FEE_CAP_PREMIUM * b)
    x["ask_net_usd"] = a + np.minimum(wl.FEE_RATE_UNDERLYING * F, wl.FEE_CAP_PREMIUM * a)
    return x


def price_by_cell(q: pd.DataFrame, pool: pd.DataFrame, u_by_cell: dict | None = None) -> np.ndarray:
    """Model price of each quote from its (tercile, TTE) cell of the pool."""
    P = np.full(len(q), np.nan)
    for (t, tb), g in q.groupby(["tercile", "tte_bucket"]):
        u = (u_by_cell or {}).get((t, tb))
        if u is None:
            u = pool.loc[(pool["tercile"] == t) & (pool["tte_bucket"] == tb), "u"].to_numpy(float)
        if len(u) < 500:
            continue
        P[q.index.get_indexer(g.index)] = wl.price(g, u)
    return P


def _stat(q: pd.DataFrame, P: np.ndarray, col: str, weights: dict | None = None) -> float:
    ok = np.isfinite(P) & (P > 0)
    return wl.session_M(q[ok], P[ok], col, weights=weights)


def decide(q: pd.DataFrame, pool: pd.DataFrame, *, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """q: quotes with tercile, tte_bucket, bid_net_usd, ask_net_usd; pool: outcomes with tercile, tte_bucket, u, session."""
    q = q.reset_index(drop=True)
    low = q[q["tercile"] == "low"].reset_index(drop=True)
    high = q[q["tercile"] == "high"].reset_index(drop=True)
    P_low, P_high = price_by_cell(low, pool), price_by_cell(high, pool)
    point = {"M_ask_low": _stat(low, P_low, "ask_net_usd"), "M_bid_high": _stat(high, P_high, "bid_net_usd")}
    dates = np.array(sorted(pool["session"].unique()))
    cell_date = {k: {d: g2["u"].to_numpy(float) for d, g2 in g.groupby("session")}
                 for k, g in pool.groupby(["tercile", "tte_bucket"])}
    rng = np.random.default_rng(int(seed))
    boots = {"M_ask_low": np.empty(int(n_boot)), "M_bid_high": np.empty(int(n_boot))}
    for b in range(int(n_boot)):
        dd = rng.choice(dates, size=len(dates), replace=True)
        ud, cnt = np.unique(dd, return_counts=True)
        u_cell = {k: np.concatenate([np.tile(v[d], m) for d, m in zip(ud, cnt) if d in v] or [np.empty(0)])
                  for k, v in cell_date.items()}
        for name, sub, col in (("M_ask_low", low, "ask_net_usd"), ("M_bid_high", high, "bid_net_usd")):
            qs = np.array(sorted(sub["session"].unique()))
            sd = rng.choice(qs, size=len(qs), replace=True)
            us, sc = np.unique(sd, return_counts=True)
            qq = pd.concat([sub[sub["session"] == s] for s in us])
            Pb = price_by_cell(qq, pool, u_cell)
            boots[name][b] = _stat(qq, Pb, col, weights=dict(zip(us, sc)))
    out = {"n_low": int(len(low)), "n_high": int(len(high)), "sessions_low": int(low["session"].nunique()),
           "sessions_high": int(high["session"].nunique())}
    p = {"M_ask_low": float(np.mean(boots["M_ask_low"] >= 0)), "M_bid_high": float(np.mean(boots["M_bid_high"] <= 0))}
    order = sorted(p, key=p.get)
    rejected = {}
    if p[order[0]] <= ALPHA / 2:
        rejected[order[0]] = True
        rejected[order[1]] = p[order[1]] <= ALPHA
    else:
        rejected = {k: False for k in p}
    for k in ("M_ask_low", "M_bid_high"):
        v = boots[k][np.isfinite(boots[k])]
        out[k] = {"M": point[k], "ci_low": float(np.quantile(v, 0.025)), "ci_high": float(np.quantile(v, 0.975)),
                  "p_one_sided": p[k], "holm_rejected": bool(rejected[k])}
    out["D_long_verdict"] = "CHEAP_AT_ASK" if out["M_ask_low"]["holm_rejected"] else "NOT_CHEAP"
    out["D_short_verdict"] = "RICH_AT_BID" if out["M_bid_high"]["holm_rejected"] else "NOT_RICH"
    return out


def cell_table(q: pd.DataFrame, pool: pd.DataFrame, n_boot: int = 500) -> pd.DataFrame:
    q = q.reset_index(drop=True)
    P = price_by_cell(q, pool)
    rows = []
    for (t, tb), g in q.groupby(["tercile", "tte_bucket"]):
        Pg = P[g.index]
        for c in ("bid_net_usd", "mid_usd", "ask_net_usd"):
            r = wl.quote_only(g, Pg, c, n_boot=n_boot) if np.isfinite(Pg).any() else {}
            rows.append({"tercile": t, "tte_bucket": tb, "price": c, **{k: r.get(k) for k in
                         ("n_rows", "n_sessions", "M", "ci_low", "ci_high", "verdict")}})
    return pd.DataFrame(rows)


def registration() -> dict:
    return {"version": VERSION, "n_boot": N_BOOT, "seed": SEED, "alpha_holm": ALPHA,
            "decisions": {"D_long": "low tercile, ask net of fee, CHEAP if Holm-rejected H0 M>=0",
                          "D_short": "high tercile, bid net of fee, RICH if Holm-rejected H0 M<=0"},
            "cell": "(own-asset v tercile from 2-year pool cutoffs) x (Module 10 TTE bucket)"}


def selftest() -> dict:
    rng = np.random.default_rng(3)
    pool = pd.DataFrame({"underlying": "X", "v": rng.random(3000)})
    c = cutoffs(pool)
    t = classify(pool, c)
    assert abs(np.mean(t == "low") - 1 / 3) < 0.01 and abs(np.mean(t == "high") - 1 / 3) < 0.01
    q = pd.DataFrame({"underlying": ["X", "X"], "session": ["s", "s"], "minutes_to_expiry": [719.99, 600.0]})
    o = pd.DataFrame({"underlying": ["X"], "session": ["s"], "minutes_to_expiry": [720.0], "v": [0.5]})
    m = attach_quote_v(q, o)
    assert m["v"].iloc[0] == 0.5 and np.isnan(m["v"].iloc[1])
    n = net_prices(pd.DataFrame({"option_forward_price": [1000.0], "bid_usd": [1.0], "ask_usd": [2.0]}))
    assert np.isclose(n["bid_net_usd"].iloc[0], 1 - 0.125) and np.isclose(n["ask_net_usd"].iloc[0], 2 + 0.25)
    return {"version": VERSION, "ok": True}
