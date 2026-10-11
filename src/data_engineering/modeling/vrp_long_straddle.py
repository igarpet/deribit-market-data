"""
Volatility Risk Premium branch, step 2 (VRP 007): the unconditional LONG
at-the-money 0DTE straddle, bought at the ask and held to settlement.

Question
--------
Does a buyer of the 0DTE ATM straddle, paying the executable ask and the
exchange fee, recover its cost at settlement on average? This is the
unconditional long-volatility P&L. Ivan does not sell options (VRP 007 §1),
so the long side is the decision; the short side at the bid is reported only
as the mirror image.

No hedge. Hold-to-settlement pays the spread once (at entry) and settlement
is cash at the exchange delivery price, so no perpetual hedge is needed and
no hedge spread is paid.

Pre-registered (fixed in this file before any real-data VRP 02 output)
---------------------------------------------------------------------
Universe   Primary 0DTE option family of each underlying
           (primary_option_surface_feature_state), sessions (expiry dates)
           before VRP_CONFIRMATION_START = 2026-11-07. Sessions on or after it
           are the branch's confirmation window (VRP 002 §2.4) and are never
           loaded here.
Entries    Every full hour (state_time minute == 0) with
           MIN_TTE_MIN <= minutes to expiry <= MAX_TTE_MIN (2h..20h).
Straddle   The strike nearest the family forward at which BOTH the call and
           the put are two-sided (bid > 0, ask >= bid). Ties: lower strike.
Cost       call ask + put ask (USD at entry) + taker fee per leg
           min(FEE_RATE x F, FEE_CAP x leg ask).
Payoff     |S_T - K|, S_T = exchange delivery price of the session
           (public/get_delivery_prices, 08:00 UTC). If unavailable, the
           approximate TWAP (mean index of the bars ending 07:35..08:00) is
           used and the row is flagged.
Inverse    Coin-settled families: premium and fee are paid in coin, so their
           USD value at settlement is scaled by S_T / F (VD 10 convention).
Unit       R = payoff / cost - 1   (return on premium, per straddle)
Statistic  M = mean over sessions of the session mean of R, where the
           session mean pools all assets and entries of that session
           (entries in one session share the settlement path).
Decision   95% session-bootstrap CI of M (N_BOOT, seed 7007):
             CI > 0  -> LONG_EDGE_CANDIDATE (must then hold on the
                        confirmation window; not a strategy yet)
             CI < 0  -> NO_LONG_EDGE (unconditional)
             else    -> INCONCLUSIVE
Reported, no decision: per asset; by TTE bucket; gross (no fee) and
delivery-fee sensitivity; mid-entry (upper bound, not executable); short at
the bid; P&L in bps of forward; straddle-implied vs realized move;
leave-one-session-out range of M; settlement-source agreement.

Fee constants are Deribit's published schedule as understood at
implementation (taker 0.03% of underlying per option, capped at 12.5% of the
option price; daily options pay no delivery fee). They could not be
re-verified from the build environment; Ivan to confirm (VRP 007 §5).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from data_engineering.modeling import option_pricing as op


VERSION = "0.1.0"
VRP_CONFIRMATION_START = pd.Timestamp("2026-11-07", tz="UTC")
MIN_TTE_MIN = 120
MAX_TTE_MIN = 1200
ENTRY_GRID_MIN = 60
FEE_RATE = 0.0003            # taker, fraction of underlying per option
FEE_CAP = 0.125              # cap as fraction of the option price
DELIVERY_FEE_RATE = 0.00015  # sensitivity only (daily options: assumed none)
N_BOOT = 4000
SEED = 7007
TTE_EDGES_MIN = (120, 240, 480, 960, 1200)
TTE_LABELS = ("2-4h", "4-8h", "8-16h", "16-20h")
DELIVERY_URL = "https://www.deribit.com/api/v2/public/get_delivery_prices"
YEAR_MIN = 365.0 * 24 * 60


# --------------------------------------------------------------------------
# Quotes -> straddles
# --------------------------------------------------------------------------

CHAIN_COLUMNS = ("state_time", "underlying", "option_family_id", "option_type", "strike",
                 "best_bid_price", "best_ask_price", "option_forward_price",
                 "minutes_to_expiry", "instrument_type", "settlement_currency")


def prepare_legs(chain: pd.DataFrame) -> pd.DataFrame:
    """USD prices, two-sided flag, entry grid and session for every option row."""
    q = chain.copy()
    q["state_time"] = pd.to_datetime(q["state_time"], utc=True)
    for c in ("strike", "option_forward_price", "minutes_to_expiry", "best_bid_price", "best_ask_price"):
        q[c] = pd.to_numeric(q[c], errors="coerce")
    q["is_call"] = q["option_type"].astype(str).str.lower().str[0].eq("c")
    q["inverse"] = op.is_inverse(q["instrument_type"])
    F = q["option_forward_price"].to_numpy(float)
    q["bid_usd"] = op.to_usd(q["best_bid_price"].to_numpy(float), F, q["inverse"])
    q["ask_usd"] = op.to_usd(q["best_ask_price"].to_numpy(float), F, q["inverse"])
    mins = q["minutes_to_expiry"]
    keep = ((q["state_time"].dt.minute % ENTRY_GRID_MIN == 0) & (q["state_time"].dt.second == 0)
            & (mins >= MIN_TTE_MIN) & (mins <= MAX_TTE_MIN)
            & np.isfinite(F) & (F > 0) & np.isfinite(q["strike"]))
    q = q[keep].copy()
    q["two_sided"] = (q["bid_usd"] > 0) & (q["ask_usd"] >= q["bid_usd"])
    q["expiry_time"] = (q["state_time"] + pd.to_timedelta(q["minutes_to_expiry"].round(), unit="min")).dt.round("5min")
    q["session"] = q["expiry_time"].dt.strftime("%Y-%m-%d")
    q = q[q["expiry_time"] < VRP_CONFIRMATION_START + pd.Timedelta(hours=8)]
    q = q[pd.to_datetime(q["session"], utc=True) < VRP_CONFIRMATION_START]
    return q.reset_index(drop=True)


def select_straddles(legs: pd.DataFrame) -> pd.DataFrame:
    """One ATM straddle per (entry time, underlying, family): nearest-forward
    strike with both legs two-sided; ties broken by the lower strike."""
    x = legs[legs["two_sided"]]
    key = ["state_time", "underlying", "option_family_id", "strike"]
    calls = x[x["is_call"]].drop_duplicates(key)
    puts = x[~x["is_call"]].drop_duplicates(key)
    keep = key + ["option_forward_price", "minutes_to_expiry", "inverse", "session", "expiry_time",
                  "bid_usd", "ask_usd", "settlement_currency"]
    keep = [c for c in keep if c in x.columns]
    m = calls[keep].merge(puts[key + ["bid_usd", "ask_usd"]], on=key, suffixes=("_call", "_put"))
    if m.empty:
        return m
    m["dist"] = np.abs(np.log(m["strike"] / m["option_forward_price"]))
    m = m.sort_values(["state_time", "underlying", "option_family_id", "dist", "strike"])
    s = m.drop_duplicates(["state_time", "underlying", "option_family_id"]).drop(columns="dist")
    s = s.rename(columns={"option_forward_price": "F", "strike": "K"})
    s["T"] = s["minutes_to_expiry"] / YEAR_MIN
    s["ask_usd"] = s["ask_usd_call"] + s["ask_usd_put"]
    s["bid_usd"] = s["bid_usd_call"] + s["bid_usd_put"]
    s["mid_usd"] = 0.5 * (s["ask_usd"] + s["bid_usd"])
    s["fee_usd"] = (np.minimum(FEE_RATE * s["F"], FEE_CAP * s["ask_usd_call"])
                    + np.minimum(FEE_RATE * s["F"], FEE_CAP * s["ask_usd_put"]))
    s["fee_bid_usd"] = (np.minimum(FEE_RATE * s["F"], FEE_CAP * s["bid_usd_call"])
                        + np.minimum(FEE_RATE * s["F"], FEE_CAP * s["bid_usd_put"]))
    s["spread_share"] = (s["ask_usd"] - s["bid_usd"]) / s["mid_usd"]
    s["iv_straddle_mid"] = op.straddle_implied_vol(s["mid_usd"].to_numpy(float), s["F"].to_numpy(float),
                                                   s["K"].to_numpy(float), s["T"].to_numpy(float))
    s["tte_bucket"] = pd.cut(s["minutes_to_expiry"], TTE_EDGES_MIN, labels=TTE_LABELS,
                             include_lowest=True).astype(str)
    return s.reset_index(drop=True)


# --------------------------------------------------------------------------
# Settlement
# --------------------------------------------------------------------------

def index_name(underlying: str, settlement_currency: str | None) -> str:
    cur = str(settlement_currency or "").upper()
    return f"{underlying.lower()}_usdc" if cur == "USDC" else f"{underlying.lower()}_usd"


def fetch_delivery_prices(index_names: Iterable[str], cache_csv: Path | None = None,
                          count: int = 120, timeout: float = 20.0) -> pd.DataFrame:
    """Exchange delivery prices (public endpoint, no key). Cached to CSV; an
    index that cannot be fetched is simply absent (the approximation is used)."""
    import requests
    frames = []
    if cache_csv is not None and Path(cache_csv).exists():
        frames.append(pd.read_csv(cache_csv, dtype={"session": str}))
    for name in sorted(set(index_names)):
        try:
            r = requests.get(DELIVERY_URL, params={"index_name": name, "offset": 0, "count": count}, timeout=timeout)
            r.raise_for_status()
            data = r.json().get("result", {}).get("data", [])
        except Exception as e:  # network or unknown index
            print(f"delivery prices for {name}: unavailable ({type(e).__name__})")
            continue
        if data:
            frames.append(pd.DataFrame({"index_name": name, "session": [d["date"] for d in data],
                                        "delivery_price": [float(d["delivery_price"]) for d in data]}))
        time.sleep(0.2)
    if not frames:
        return pd.DataFrame(columns=["index_name", "session", "delivery_price"])
    out = (pd.concat(frames, ignore_index=True)
           .drop_duplicates(["index_name", "session"], keep="last")
           .sort_values(["index_name", "session"]).reset_index(drop=True))
    if cache_csv is not None:
        out.to_csv(cache_csv, index=False)
    return out


def approx_settlement(index_frame: pd.DataFrame, sessions: Sequence[str]) -> pd.DataFrame:
    """Approximate delivery: mean index of the 5m bars ending 07:35..08:00
    (columns state_time, underlying, futures_index_price). VD 10 convention."""
    t = pd.to_datetime(index_frame["state_time"], utc=True)
    p = pd.to_numeric(index_frame["futures_index_price"], errors="coerce")
    rows = []
    for s in sessions:
        lo, hi = pd.Timestamp(f"{s} 07:35", tz="UTC"), pd.Timestamp(f"{s} 08:00", tz="UTC")
        m = (t >= lo) & (t <= hi) & p.notna()
        g = pd.DataFrame({"u": index_frame.loc[m, "underlying"], "p": p[m]}).groupby("u")["p"].agg(["mean", "count"])
        for u, r in g.iterrows():
            if r["count"] >= 5:
                rows.append({"session": s, "underlying": u, "settle_approx": float(r["mean"])})
    return pd.DataFrame(rows, columns=["session", "underlying", "settle_approx"])


def attach_settlement(st: pd.DataFrame, delivery: pd.DataFrame, approx: pd.DataFrame) -> pd.DataFrame:
    x = st.copy()
    x["index_name"] = [index_name(u, c) for u, c in zip(x["underlying"], x.get("settlement_currency", [None] * len(x)))]
    if len(delivery):
        x = x.merge(delivery[["index_name", "session", "delivery_price"]], on=["index_name", "session"], how="left")
    else:
        x["delivery_price"] = np.nan
    x = x.merge(approx, on=["session", "underlying"], how="left")
    x["settle_source"] = np.where(x["delivery_price"].notna(), "exchange",
                                  np.where(x["settle_approx"].notna(), "approx_twap", "none"))
    x["S_T"] = x["delivery_price"].fillna(x["settle_approx"])
    return x[x["S_T"].notna()].reset_index(drop=True)


# --------------------------------------------------------------------------
# P&L
# --------------------------------------------------------------------------

def straddle_pnl(x: pd.DataFrame) -> pd.DataFrame:
    """Per-straddle outcomes. Coin premium valued at settlement (S_T / F)."""
    x = x.copy()
    ST, K, F = (x[c].to_numpy(float) for c in ("S_T", "K", "F"))
    conv = np.where(x["inverse"].to_numpy(bool), ST / F, 1.0)
    pay = np.abs(ST - K)
    gross_cost = x["ask_usd"].to_numpy(float) * conv
    cost = (x["ask_usd"].to_numpy(float) + x["fee_usd"].to_numpy(float)) * conv
    deliv = np.minimum(DELIVERY_FEE_RATE * ST, FEE_CAP * pay)   # only the ITM leg has value to charge
    x["payoff_usd"] = pay
    x["R_net"] = pay / cost - 1.0                                  # PRIMARY
    x["R_gross"] = pay / gross_cost - 1.0
    x["R_net_delivery_fee"] = (pay - deliv) / cost - 1.0
    x["R_mid"] = pay / (x["mid_usd"].to_numpy(float) * conv) - 1.0
    x["pnl_long_net_bps"] = (pay - cost) / F * 1e4
    x["pnl_short_bid_net_bps"] = ((x["bid_usd"].to_numpy(float) - x["fee_bid_usd"].to_numpy(float)) * conv - pay) / F * 1e4
    x["move_over_implied"] = np.abs(np.log(ST / F)) / (x["iv_straddle_mid"].to_numpy(float)
                                                       * np.sqrt(x["T"].to_numpy(float)) * np.sqrt(2.0 / np.pi))
    return x


def session_means(x: pd.DataFrame, col: str) -> pd.Series:
    """Session mean of col, pooling every asset and entry of the session."""
    return x.groupby("session")[col].mean()


def bootstrap_mean(values: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return {"n_sessions": int(len(v))}
    rng = np.random.default_rng(seed)
    b = rng.choice(v, size=(int(n_boot), len(v)), replace=True).mean(axis=1)
    loso = np.array([np.delete(v, i).mean() for i in range(len(v))])
    return {"n_sessions": int(len(v)), "M": float(v.mean()),
            "ci_low": float(np.quantile(b, 0.025)), "ci_high": float(np.quantile(b, 0.975)),
            "loso_min": float(loso.min()), "loso_max": float(loso.max()),
            "share_sessions_positive": float(np.mean(v > 0))}


def verdict(stat: dict) -> str:
    if "ci_low" not in stat:
        return "INSUFFICIENT_SESSIONS"
    if stat["ci_low"] > 0:
        return "LONG_EDGE_CANDIDATE"
    if stat["ci_high"] < 0:
        return "NO_LONG_EDGE"
    return "INCONCLUSIVE"


def primary(x: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    st = bootstrap_mean(session_means(x, "R_net").to_numpy(), n_boot=n_boot)
    return {**st, "verdict": verdict(st), "n_straddles": int(len(x)),
            "n_session_assets": int(x.groupby(["session", "underlying"]).ngroups)}


def report_table(x: pd.DataFrame, by: str | None, cols: Sequence[str], n_boot: int = 1000) -> pd.DataFrame:
    groups = [("all", x)] if by is None else list(x.groupby(by))
    rows = []
    for g, d in groups:
        for c in cols:
            rows.append({"group": by or "all", "level": g, "metric": c, "n_straddles": int(len(d)),
                         **bootstrap_mean(session_means(d, c).to_numpy(), n_boot=n_boot)})
    return pd.DataFrame(rows)


REPORT_COLS = ("R_net", "R_gross", "R_net_delivery_fee", "R_mid", "pnl_long_net_bps",
               "pnl_short_bid_net_bps", "move_over_implied", "spread_share")


def registration() -> dict:
    return {"version": VERSION, "confirmation_start_excluded": str(VRP_CONFIRMATION_START.date()),
            "tte_min": [MIN_TTE_MIN, MAX_TTE_MIN], "entry_grid_min": ENTRY_GRID_MIN,
            "fee_rate": FEE_RATE, "fee_cap": FEE_CAP, "delivery_fee_rate_sensitivity": DELIVERY_FEE_RATE,
            "n_boot": N_BOOT, "seed": SEED, "primary_unit": "R_net = payoff/(ask+fee) - 1",
            "primary_statistic": "mean over sessions of session-pooled mean R_net",
            "decision": "CI>0 LONG_EDGE_CANDIDATE; CI<0 NO_LONG_EDGE; else INCONCLUSIVE"}


# --------------------------------------------------------------------------
# Self test (synthetic, no database)
# --------------------------------------------------------------------------

def _fake_chain(rng, n_sessions=12, vol=0.6, iv=0.6, half_spread=0.03, inverse=False, under="BTC") -> tuple:
    """Black-76-priced chains around a GBM; returns (chain, true settlement)."""
    rows, settle = [], []
    for d in range(n_sessions):
        s = (pd.Timestamp("2026-04-01", tz="UTC") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        exp = pd.Timestamp(f"{s} 08:00", tz="UTC")
        S0 = 100_000.0
        times = [exp - pd.Timedelta(minutes=m) for m in range(MAX_TTE_MIN, MIN_TTE_MIN - 1, -60)]
        S = S0
        prev = times[0] - pd.Timedelta(minutes=60)
        for t in times:
            dt = (t - prev).total_seconds() / 60 / YEAR_MIN
            S *= np.exp(vol * np.sqrt(dt) * rng.normal() - 0.5 * vol ** 2 * dt)
            prev = t
            mins = (exp - t).total_seconds() / 60
            T = mins / YEAR_MIN
            for K in np.arange(round(S, -3) - 3000, round(S, -3) + 3001, 1000.0):
                for cp in ("call", "put"):
                    fair = float(op.black76_usd(S, K, T, iv, cp == "call"))
                    bid, ask = fair * (1 - half_spread), fair * (1 + half_spread)
                    conv = S if inverse else 1.0
                    rows.append({"state_time": t, "underlying": under, "option_family_id": under + "|f",
                                 "option_type": cp, "strike": K, "best_bid_price": bid / conv,
                                 "best_ask_price": ask / conv, "option_forward_price": S,
                                 "minutes_to_expiry": mins,
                                 "instrument_type": "reversed" if inverse else "linear",
                                 "settlement_currency": under if inverse else "USDC"})
        dt = MIN_TTE_MIN / YEAR_MIN
        S_T = S * np.exp(vol * np.sqrt(dt) * rng.normal() - 0.5 * vol ** 2 * dt)
        settle.append({"session": s, "underlying": under, "settle_approx": S_T})
    return pd.DataFrame(rows), pd.DataFrame(settle)


def _run_synthetic(vol, iv, half_spread, inverse=False, seed=1, n_sessions=60):
    rng = np.random.default_rng(seed)
    chain, settle = _fake_chain(rng, n_sessions, vol, iv, half_spread, inverse)
    st = select_straddles(prepare_legs(chain))
    x = straddle_pnl(attach_settlement(st, pd.DataFrame(columns=["index_name", "session", "delivery_price"]), settle))
    return x, primary(x, n_boot=500)


def selftest() -> dict:
    # Payoff arithmetic, linear
    one = pd.DataFrame({"S_T": [110.0], "K": [100.0], "F": [100.0], "inverse": [False], "ask_usd": [8.0],
                        "fee_usd": [0.0], "mid_usd": [7.5], "bid_usd": [7.0], "fee_bid_usd": [0.0],
                        "iv_straddle_mid": [0.5], "T": [0.001]})
    o = straddle_pnl(one).iloc[0]
    assert abs(o["R_net"] - (10 / 8 - 1)) < 1e-12 and abs(o["pnl_short_bid_net_bps"] + 300.0) < 1e-9
    # Inverse: coin premium revalued at S_T / F
    one["inverse"] = True
    o = straddle_pnl(one).iloc[0]
    assert abs(o["R_net"] - (10 / (8 * 1.1) - 1)) < 1e-12
    # Fair options, no spread: R ~ 0 (mean of a ratio is slightly biased; wide tolerance)
    _, fair = _run_synthetic(vol=0.6, iv=0.6, half_spread=0.0, n_sessions=150)
    assert fair["verdict"] in ("INCONCLUSIVE",), fair
    # Overpriced IV with spread: no long edge
    _, rich = _run_synthetic(vol=0.4, iv=0.8, half_spread=0.05, n_sessions=60)
    assert rich["verdict"] == "NO_LONG_EDGE", rich
    # Underpriced IV: long edge
    _, cheap = _run_synthetic(vol=1.2, iv=0.5, half_spread=0.02, n_sessions=60, inverse=True)
    assert cheap["verdict"] == "LONG_EDGE_CANDIDATE", cheap
    return {"version": VERSION, "fair_M": round(fair["M"], 4), "rich_M": round(rich["M"], 4),
            "cheap_M": round(cheap["M"], 4), "ok": True}
