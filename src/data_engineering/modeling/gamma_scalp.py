"""Regime Structure 06 — delta-hedged long 0DTE straddle (gamma scalping).

Pre-registration: discussions/Regime_Structure/011_claude_rs06_preregistration.md.
Ivan lifted the no-hedging constraint for this test only (chat, 2026-10-13).

Option leg (VRP 02 code, approved): long ATM 0DTE straddle of the primary family, bought at the ask + taker
fee at every full-hour entry with 120..1200 minutes to the 08:00 UTC expiry, settled at the exchange delivery
price. Coin-premium conversion as VRP 02 (cost x S_T / F for inverse families).

Hedge leg: Deribit perpetual, 5-minute collector grid.
    Price P_t        perpetual tradable mid (fallback: Deribit index), forward-filled up to 15 minutes.
    Delta            Black-76 straddle delta at P_t, strike K, remaining T, entry straddle IV held fixed.
    Rebalance        every H minutes from entry to the last bar before expiry; position -delta (coin units).
    Hedge P&L        sum_k -delta_k (P_{k+1} - P_k); the position is closed at the last bar before 08:00.
    Hedge cost       |change in position| x P x (half-spread + fee), per side, including entry and exit.
                     Half-spread = futures_spread_bps / 2 at that bar (fallback 1 bp); fee = taker 0.05%.

    R_hedged = (payoff + hedge P&L - hedge cost) / ((ask + option fee) x conv) - 1
    Primary: H = 15 minutes, taker fee. Statistic and verdict exactly as VRP 02 (session mean, bootstrap,
    LONG_EDGE_CANDIDATE / NO_LONG_EDGE / INCONCLUSIVE).
Reported: H in {5, 15, 30, 60}; fee scenarios taker 0.05% + half-spread / 0.02% + half-spread / zero cost;
hedged minus unhedged; realised variance at H over implied variance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from data_engineering.modeling import option_pricing as op
from data_engineering.modeling import vrp_long_straddle as ls

VERSION = "gamma_scalp v1.0"
BAR = pd.Timedelta(minutes=5)


@dataclass(frozen=True)
class Params:
    hedge_min: int = 15
    hedge_grid: tuple = (5, 15, 30, 60)
    taker_fee: float = 0.0005
    low_fee: float = 0.0002
    default_half_spread_bps: float = 1.0
    max_ffill_bars: int = 3
    max_missing_share: float = 0.10
    n_boot: int = 4000


SCENARIOS = {"taker": "taker_fee", "low_fee": "low_fee", "zero_cost": None}


def registration(p: Params | None = None) -> dict:
    p = p or Params()
    return {**asdict(p), "version": VERSION, "vrp": ls.registration(),
            "primary": "R_hedged at H=15min, taker 0.05% + half-spread; VRP 02 session statistic and verdict"}


# ---------------------------------------------------------------- hedge price paths

def price_paths(ts: pd.DataFrame, p: Params) -> dict:
    """{underlying: DataFrame indexed by 5m state_time with P (hedge price), hs (half-spread, fraction), src}."""
    out = {}
    d = ts.copy()
    d["state_time"] = pd.to_datetime(d["state_time"], utc=True)
    for u, g in d.groupby("underlying"):
        g = g.drop_duplicates("state_time").set_index("state_time").sort_index()
        grid = pd.date_range(g.index[0], g.index[-1], freq=BAR)
        g = g.reindex(grid)
        mid = pd.to_numeric(g.get("futures_tradable_mid_price"), errors="coerce") if "futures_tradable_mid_price" in g else None
        idx = pd.to_numeric(g.get("futures_index_price"), errors="coerce")
        if mid is not None:
            P = mid.where(mid > 0)
            src = np.where(P.notna(), "perp_mid", "index")
            P = P.fillna(idx.where(idx > 0))
        else:
            P, src = idx.where(idx > 0), np.full(len(g), "index")
        P = P.ffill(limit=p.max_ffill_bars)
        if "futures_spread_bps" in g:
            hs = pd.to_numeric(g["futures_spread_bps"], errors="coerce").clip(lower=0) / 2e4
        else:
            hs = pd.Series(np.nan, index=g.index)
        hs = hs.fillna(p.default_half_spread_bps / 1e4)
        out[u] = pd.DataFrame({"P": P, "hs": hs, "src": src}, index=grid)
    return out


# ---------------------------------------------------------------- one hedged straddle

def hedge_one(path: pd.DataFrame, entry: pd.Timestamp, expiry: pd.Timestamp, K: float, iv: float,
              H: int, fee: float | None, hs_on: bool, p: Params) -> dict:
    """Hedge P&L and cost of a long straddle (1 coin notional) from entry to expiry, rebalanced every H min."""
    times = pd.date_range(entry, expiry - BAR, freq=pd.Timedelta(minutes=H))
    if len(times) < 2:
        return {"ok": False}
    seg = path.reindex(times)
    P = seg["P"].to_numpy(float)
    miss = ~np.isfinite(P)
    if miss.mean() > p.max_missing_share or miss[0]:
        return {"ok": False}
    P = pd.Series(P).ffill().to_numpy()
    last = path["P"].reindex([expiry - BAR]).to_numpy(float)[0]         # unwind at the last bar before expiry
    if not np.isfinite(last):
        last = P[-1]
    T = ((expiry - times) / pd.Timedelta(minutes=1)).to_numpy(float) / ls.YEAR_MIN
    delta = (op.black76_delta(P, K, T, iv, True) + op.black76_delta(P, K, T, iv, False))
    pos = -delta                                                        # perp position (coins)
    P_next = np.r_[P[1:], last]
    pnl = float(np.sum(pos * (P_next - P)))
    trades = np.abs(np.diff(np.r_[0.0, pos, 0.0]))                      # entry, rebalances, exit
    px = np.r_[P, last]
    hs = seg["hs"].ffill().to_numpy(float)
    hs = np.r_[hs, hs[-1]] if hs_on else np.zeros(len(px))
    f = 0.0 if fee is None else fee
    cost = float(np.sum(trades * px * (hs + f)))
    r = np.diff(np.log(np.r_[P, last]))
    rv_H = float(np.sum(r ** 2))
    return {"ok": True, "hedge_pnl": pnl, "hedge_cost": cost, "n_rebal": int(len(times)), "rv_H": rv_H,
            "turnover": float(trades.sum())}


def run(x: pd.DataFrame, paths: dict, p: Params) -> pd.DataFrame:
    """x: VRP straddle_pnl output. Adds hedged returns for every H and fee scenario."""
    rows = []
    for i, s in x.iterrows():
        path = paths.get(s["underlying"])
        if path is None:
            continue
        entry, expiry = s["state_time"], s["expiry_time"]
        conv = (s["S_T"] / s["F"]) if bool(s["inverse"]) else 1.0
        cost_opt = (s["ask_usd"] + s["fee_usd"]) * conv
        rec = {"row": i}
        for H in p.hedge_grid:
            for name, attr in SCENARIOS.items():
                fee = getattr(p, attr) if attr else None
                h = hedge_one(path, entry, expiry, s["K"], s["iv_straddle_mid"], H, fee, attr is not None, p)
                if not h["ok"]:
                    rec[f"R_h{H}_{name}"] = np.nan
                    continue
                rec[f"R_h{H}_{name}"] = (s["payoff_usd"] + h["hedge_pnl"] - h["hedge_cost"]) / cost_opt - 1.0
                if name == "taker":
                    rec[f"cost_h{H}_share"] = h["hedge_cost"] / cost_opt
                    rec[f"hpnl_h{H}_share"] = h["hedge_pnl"] / cost_opt
                    rec[f"rv_over_iv_h{H}"] = h["rv_H"] / (s["iv_straddle_mid"] ** 2 * s["T"])
                    rec[f"rebal_h{H}"] = h["n_rebal"]
        rows.append(rec)
    r = pd.DataFrame(rows).set_index("row")
    return x.join(r, how="inner")


def summarise(y: pd.DataFrame, p: Params) -> tuple[dict, pd.DataFrame]:
    prim_col = f"R_h{p.hedge_min}_taker"
    d = y[np.isfinite(y[prim_col])]
    st = ls.bootstrap_mean(ls.session_means(d, prim_col).to_numpy(), n_boot=p.n_boot)
    prim = {**st, "verdict": ls.verdict(st), "n_straddles": int(len(d)), "metric": prim_col}
    rows = [{"metric": "R_net (unhedged, same entries)", **ls.bootstrap_mean(ls.session_means(d, "R_net").to_numpy(), n_boot=1000)}]
    for H in p.hedge_grid:
        for name in SCENARIOS:
            c = f"R_h{H}_{name}"
            if c in d:
                rows.append({"metric": c, **ls.bootstrap_mean(ls.session_means(d, c).to_numpy(), n_boot=1000)})
        dd = d.assign(_diff=d[f"R_h{H}_taker"] - d["R_net"])
        rows.append({"metric": f"hedged_minus_unhedged_h{H}_taker",
                     **ls.bootstrap_mean(ls.session_means(dd, "_diff").to_numpy(), n_boot=1000)})
        for c in (f"rv_over_iv_h{H}", f"cost_h{H}_share", f"hpnl_h{H}_share"):
            if c in d:
                rows.append({"metric": c, **ls.bootstrap_mean(ls.session_means(d, c).to_numpy(), n_boot=1000)})
    return prim, pd.DataFrame(rows)


# ---------------------------------------------------------------- variance ratio by sampling (Binance, descriptive)

def vr_by_sampling(r5: np.ndarray, windows: np.ndarray, ks: tuple = (1, 3, 6, 12)) -> pd.DataFrame:
    """sum(session net^2) / sum(session RV_k^2), RV from k x 5-minute bars. 1 = no intraday autocorrelation."""
    net2 = r5.sum(1) ** 2
    rows = []
    for k in ks:
        rk = r5.reshape(len(r5), -1, k).sum(2)
        rv2 = (rk ** 2).sum(1)
        for w in ("dev", "test"):
            m = windows == w
            rows.append({"window": w, "sampling_min": 5 * k, "vr": float(net2[m].sum() / rv2[m].sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- synthetic

def synthetic_sessions(n: int, vol: float, iv: float, reversion: float = 0.0, half_spread: float = 0.0,
                       seed: int = 0, under: str = "BTC", start: str = "2026-04-01") -> tuple:
    """n sessions of a 5m path (GBM, optional negative AR(1) at 5m) and one fair-priced 12:00 straddle each.
    Returns (VRP-style straddle frame with settlement and P&L columns, price-path frame)."""
    rng = np.random.default_rng(seed)
    dt = 5 / ls.YEAR_MIN
    rows, path_rows = [], []
    for d in range(n):
        exp = pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=d + 1, hours=8)
        t0 = exp - pd.Timedelta(hours=20)
        m = int(pd.Timedelta(hours=20) / BAR)
        e = rng.standard_normal(m) * vol * np.sqrt(dt)
        r = e.copy()
        r[1:] = e[1:] - reversion * e[:-1]
        S0 = 100_000.0
        S = S0 * np.exp(np.r_[0.0, np.cumsum(r)])
        times = t0 + BAR * np.arange(m + 1)
        for t, v in zip(times, S):
            path_rows.append({"state_time": t, "underlying": under, "futures_index_price": v,
                              "futures_tradable_mid_price": v, "futures_spread_bps": 0.0})
        T = 20 * 60 / ls.YEAR_MIN
        fair = float(op.straddle_usd(S0, S0, T, iv))
        ask = fair * (1 + half_spread)
        rows.append({"state_time": t0, "expiry_time": exp, "session": exp.strftime("%Y-%m-%d"), "underlying": under,
                     "K": S0, "F": S0, "T": T, "inverse": False, "ask_usd": ask, "bid_usd": fair * (1 - half_spread),
                     "mid_usd": fair, "fee_usd": 0.0, "fee_bid_usd": 0.0, "iv_straddle_mid": iv, "S_T": float(S[-1]),
                     "ask_usd_call": ask / 2, "ask_usd_put": ask / 2, "minutes_to_expiry": 1200.0,
                     "tte_bucket": "16-20h"})
    x = ls.straddle_pnl(pd.DataFrame(rows))
    return x, pd.DataFrame(path_rows)
