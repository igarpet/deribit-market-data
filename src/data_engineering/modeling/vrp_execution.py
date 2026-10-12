"""
Volatility Risk Premium branch, step 3 (VRP 009): passive execution of the
ATM straddle, both sides.

Question
--------
VRP 02 paid (or received) the full spread: buying at the ask lost 27% of
premium, while the mid-entry bound was -14%. How much of the spread can a
passive limit order near the mid recover once fills and adverse selection
are accounted for? The answer decides whether any 0DTE option strategy can
clear its cost hurdle (VRP 008 section 4.1).

Fill proxy (5-minute snapshots; no trade tape is stored)
--------------------------------------------------------
A resting BUY limit at price P, placed at entry t, is FILLED at the first later
snapshot s in (t, t + W] where either
    best ask_s <= P                     (sellers came to our price), or
    a trade printed (volume_24h rose or last_price changed since s-1) with
    last_price_s <= P                   (someone sold at or through our price).
A SELL limit is symmetric (best bid_s >= P, or a trade with last_price_s >= P).
P strictly improves the touch (buy: P > best bid; sell: P < best ask), so we
would have had queue priority at our price; one contract, no partial fills.
This is an optimistic proxy at the 5-minute grid (fills between snapshots that
reverse are invisible in both directions), stated as such.

Limit prices (native quote units, rounded to the family tick away from the mid)
    buy  P_b(q) = mid - q (mid - bid)        sell  P_s(q) = mid + q (ask - mid)
    q in {0 (mid), 0.5 (halfway to the touch)}; W in {5, 15, 30, 60} minutes.
    If the spread is one tick, no improving price exists: the order joins the
    touch (q = 1) and is flagged.

Policy scored (every entry trades, so it is paired with VRP 02's entries)
    "passive-then-cross": each leg rests at P for W minutes; a leg not filled
    by t + W is completed at the opposite touch of snapshot t + W.
    Baseline: both legs crossed at t (VRP 02, aggressive).

Settlement, fees and units as VRP 02 (exchange delivery price; fee
min(0.0003 F, 0.125 premium) per leg, maker assumed equal to taker).
Native units: inverse families pay and receive coin, so R is computed in coin
(identical to VRP 02's S_T/F convention); linear families in USD.

Pre-registered decision (primary: q = 0.5, W = 15 min)
------------------------------------------------------
For each side (long = buy straddle, short = sell straddle), the paired
improvement Delta_e = R_policy,e - R_aggressive,e per entry (R = return on
premium for the long; P&L over premium received for the short). Statistic:
mean over sessions of the session mean of Delta, 95% session bootstrap
(4,000 draws, seed 9003).
    CI > 0  -> PASSIVE_HELPS     CI < 0 -> PASSIVE_HURTS     else INCONCLUSIVE
Reported with it, no decision: the policy's absolute R per side (CI); fill
rates by q, W, TTE and asset; markouts (mid move after fill, in half-spreads);
the both-legs-passively-filled subsample; adverse-selection gap
(mid-entry R of filled vs all entries).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from data_engineering.modeling import vrp_long_straddle as ls


VERSION = "0.1.0"
Q_LEVELS = (0.0, 0.5)
WINDOWS_MIN = (5, 15, 30, 60)
PRIMARY = (0.5, 15)
SNAP_MIN = 5
N_BOOT = 4000
SEED = 9003
CHAIN_COLUMNS = ls.CHAIN_COLUMNS + ("last_price", "volume_24h", "best_bid_amount", "best_ask_amount")


# --------------------------------------------------------------------------
# Data shaping
# --------------------------------------------------------------------------

def prepare_snapshots(chain: pd.DataFrame) -> pd.DataFrame:
    """All 5m snapshots of the near-ATM legs, keyed by instrument (family, strike, type)."""
    q = chain.copy()
    q["state_time"] = pd.to_datetime(q["state_time"], utc=True)
    for c in ("strike", "best_bid_price", "best_ask_price", "option_forward_price", "minutes_to_expiry",
              "last_price", "volume_24h"):
        if c in q:
            q[c] = pd.to_numeric(q[c], errors="coerce")
        else:
            q[c] = np.nan
    q["is_call"] = q["option_type"].astype(str).str.lower().str[0].eq("c")
    q["inverse"] = ls.op.is_inverse(q["instrument_type"])
    q["expiry_time"] = (q["state_time"] + pd.to_timedelta(q["minutes_to_expiry"].round(), unit="min")).dt.round("5min")
    q["session"] = q["expiry_time"].dt.strftime("%Y-%m-%d")
    q = q[pd.to_datetime(q["session"], utc=True) < ls.VRP_CONFIRMATION_START]
    q["inst"] = (q["option_family_id"].astype(str) + "|" + q["strike"].astype(str) + "|"
                 + np.where(q["is_call"], "C", "P"))
    q = q.sort_values(["inst", "state_time"]).drop_duplicates(["inst", "state_time"])
    return q.reset_index(drop=True)


def family_ticks(snap: pd.DataFrame, min_count: int = 20) -> pd.Series:
    """Native price tick per option family: smallest positive price increment seen >= min_count times."""
    out = {}
    for fam, g in snap.groupby("option_family_id"):
        p = np.unique(np.round(np.r_[g["best_bid_price"].dropna(), g["best_ask_price"].dropna()], 10))
        p = p[p > 0]
        d = np.round(np.diff(p), 10)
        d = d[d > 0]
        if not len(d):
            continue
        vals, cnt = np.unique(d, return_counts=True)
        common = vals[cnt >= min_count]
        out[fam] = float(common.min() if len(common) else vals.min())
    return pd.Series(out, name="tick")


def limit_price(mid, bid, ask, tick, side: str, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Rounded limit and a flag for one-tick spreads (order joins the touch)."""
    mid, bid, ask, tick = (np.asarray(x, float) for x in (mid, bid, ask, tick))
    if side == "buy":
        p = np.floor((mid - q * (mid - bid)) / tick + 1e-9) * tick
        touch = p <= bid + 1e-12
        p = np.where(touch, bid, p)
    else:
        p = np.ceil((mid + q * (ask - mid)) / tick - 1e-9) * tick
        touch = p >= ask - 1e-12
        p = np.where(touch, ask, p)
    return p, touch


def leg_paths(snap: pd.DataFrame, entries: pd.DataFrame, max_w: int = max(WINDOWS_MIN)) -> dict:
    """For each entry leg, the next max_w/5 snapshots of the same instrument as arrays (k = 1..K).
    Missing snapshots are NaN (never filled on a missing snapshot)."""
    K = max_w // SNAP_MIN
    s = snap.set_index(["inst", "state_time"])
    traded_any = (s.groupby(level=0)["volume_24h"].diff() > 0) | (s.groupby(level=0)["last_price"].diff().fillna(0) != 0)
    s = s.assign(traded=traded_any.to_numpy())
    cols = ("best_bid_price", "best_ask_price", "last_price", "traded")
    out = {c: np.full((len(entries), K), np.nan) for c in cols}
    for k in range(1, K + 1):
        key = pd.MultiIndex.from_arrays([entries["inst"].to_numpy(),
                                         (entries["state_time"] + pd.Timedelta(minutes=SNAP_MIN * k)).to_numpy()])
        got = s.reindex(key)
        for c in cols:
            out[c][:, k - 1] = got[c].to_numpy(float)
    return out


def first_fill(paths: dict, price: np.ndarray, side: str, k_max: int) -> np.ndarray:
    """Index (1..k_max) of the first fill snapshot, 0 if not filled within k_max."""
    bid, ask, last, tr = (paths[c][:, :k_max] for c in ("best_bid_price", "best_ask_price", "last_price", "traded"))
    p = price[:, None]
    if side == "buy":
        hit = (ask <= p + 1e-12) | ((tr > 0.5) & (last <= p + 1e-12))
    else:
        hit = (bid >= p - 1e-12) | ((tr > 0.5) & (last >= p - 1e-12))
    hit = np.where(np.isfinite(bid) | np.isfinite(ask), hit, False)
    any_hit = hit.any(axis=1)
    return np.where(any_hit, hit.argmax(axis=1) + 1, 0)


# --------------------------------------------------------------------------
# Entries and P&L
# --------------------------------------------------------------------------

def straddle_entries(snap: pd.DataFrame) -> pd.DataFrame:
    """VRP 02's hourly ATM straddle entries (same selection), with each leg's instrument key."""
    hourly = snap[(snap["state_time"].dt.minute % ls.ENTRY_GRID_MIN == 0)]
    legs = ls.prepare_legs(hourly[list(ls.CHAIN_COLUMNS)])
    st = ls.select_straddles(legs)
    if st.empty:
        return st
    for leg, cp in (("call", "C"), ("put", "P")):
        st[f"inst_{leg}"] = st["option_family_id"].astype(str) + "|" + st["K"].astype(str) + "|" + cp
    return st


def _leg_native(st: pd.DataFrame, leg: str, col: str) -> np.ndarray:
    usd = st[f"{col}_usd_{leg}"].to_numpy(float)
    return np.where(st["inverse"].to_numpy(bool), usd / st["F"].to_numpy(float), usd)


def fee_native(price_native: np.ndarray, F: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    F_native = np.where(inverse, 1.0, F)          # 0.03% of one unit of underlying, in quote units
    return np.minimum(ls.FEE_RATE * F_native, ls.FEE_CAP * price_native)


def payoff_native(st: pd.DataFrame) -> np.ndarray:
    ST, K = st["S_T"].to_numpy(float), st["K"].to_numpy(float)
    pay = np.abs(ST - K)
    return np.where(st["inverse"].to_numpy(bool), pay / ST, pay)


def simulate_policy(st: pd.DataFrame, snap: pd.DataFrame, ticks: pd.Series, side: str, q: float, w_min: int) -> pd.DataFrame:
    """Per-entry cost (buy) or proceeds (sell) of the straddle under passive-then-cross, native units."""
    k = w_min // SNAP_MIN
    inv = st["inverse"].to_numpy(bool)
    F = st["F"].to_numpy(float)
    tick = st["option_family_id"].map(ticks).to_numpy(float)
    total = np.zeros(len(st))
    fees = np.zeros(len(st))
    both = np.ones(len(st), bool)
    out = {}
    for leg in ("call", "put"):
        bid, ask = _leg_native(st, leg, "bid"), _leg_native(st, leg, "ask")
        mid = 0.5 * (bid + ask)
        p, touch = limit_price(mid, bid, ask, tick, side, q)
        e = pd.DataFrame({"inst": st[f"inst_{leg}"].to_numpy(), "state_time": st["state_time"].to_numpy()})
        paths = leg_paths(snap, e)
        kf = first_fill(paths, p, side, k)
        filled = kf > 0
        cross_col = "best_ask_price" if side == "buy" else "best_bid_price"
        cross = paths[cross_col][:, k - 1]
        cross = np.where(np.isfinite(cross), cross, ask if side == "buy" else bid)     # missing: original touch
        px = np.where(filled, p, cross)
        total += px
        fees += fee_native(px, F, inv)
        both &= filled
        half = 0.5 * (ask - bid)
        mid_after = 0.5 * (paths["best_bid_price"] + paths["best_ask_price"])
        idx = np.clip(np.where(filled, kf - 1 + 30 // SNAP_MIN, k - 1), 0, mid_after.shape[1] - 1)
        m_after = mid_after[np.arange(len(st)), idx]
        sign = 1.0 if side == "buy" else -1.0
        out[f"filled_{leg}"] = filled
        out[f"touch_{leg}"] = touch
        out[f"markout_{leg}_halfspreads"] = np.where(filled, sign * (m_after - p) / np.where(half > 0, half, np.nan), np.nan)
    out.update({"px_total": total, "fees": fees, "both_filled": both})
    return pd.DataFrame(out, index=st.index)


def returns(st: pd.DataFrame, px_total: np.ndarray, fees: np.ndarray, side: str) -> np.ndarray:
    pay = payoff_native(st)
    if side == "buy":
        return pay / (px_total + fees) - 1.0
    return (px_total - fees - pay) / px_total


def aggressive(st: pd.DataFrame, side: str) -> tuple[np.ndarray, np.ndarray]:
    inv, F = st["inverse"].to_numpy(bool), st["F"].to_numpy(float)
    col = "ask" if side == "buy" else "bid"
    legs = [_leg_native(st, l, col) for l in ("call", "put")]
    return legs[0] + legs[1], fee_native(legs[0], F, inv) + fee_native(legs[1], F, inv)


def mid_entry(st: pd.DataFrame, side: str) -> np.ndarray:
    inv, F = st["inverse"].to_numpy(bool), st["F"].to_numpy(float)
    legs = [0.5 * (_leg_native(st, l, "bid") + _leg_native(st, l, "ask")) for l in ("call", "put")]
    return returns(st, legs[0] + legs[1], fee_native(legs[0], F, inv) + fee_native(legs[1], F, inv), side)


def session_boot(values: np.ndarray, sessions: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    s = pd.Series(np.asarray(values, float)).groupby(np.asarray(sessions)).mean().dropna()
    return {**ls.bootstrap_mean(s.to_numpy(), n_boot=n_boot, seed=seed)}


def verdict(stat: dict) -> str:
    if "ci_low" not in stat:
        return "INSUFFICIENT_SESSIONS"
    return "PASSIVE_HELPS" if stat["ci_low"] > 0 else ("PASSIVE_HURTS" if stat["ci_high"] < 0 else "INCONCLUSIVE")


def evaluate(st: pd.DataFrame, snap: pd.DataFrame, ticks: pd.Series, n_boot: int = N_BOOT) -> dict:
    """Full grid of (side, q, W); primary decision per side."""
    rows, per_entry = [], {}
    sess = st["session"].to_numpy()
    for side in ("buy", "sell"):
        agg_px, agg_fee = aggressive(st, side)
        r_agg = returns(st, agg_px, agg_fee, side)
        r_mid = mid_entry(st, side)
        for q in Q_LEVELS:
            for w in WINDOWS_MIN:
                sim = simulate_policy(st, snap, ticks, side, q, w)
                r_pol = returns(st, sim["px_total"].to_numpy(), sim["fees"].to_numpy(), side)
                delta = r_pol - r_agg
                both = sim["both_filled"].to_numpy(bool)
                d = session_boot(delta, sess, n_boot)
                a = session_boot(r_pol, sess, n_boot)
                rows.append({"side": side, "q": q, "window_min": w, "n_entries": int(len(st)),
                             "fill_rate_call": float(sim["filled_call"].mean()), "fill_rate_put": float(sim["filled_put"].mean()),
                             "fill_rate_both": float(both.mean()),
                             "touch_share": float(np.mean(sim["touch_call"] | sim["touch_put"])),
                             "markout_halfspreads_median": float(np.nanmedian(np.r_[sim["markout_call_halfspreads"], sim["markout_put_halfspreads"]])),
                             "delta_M": d.get("M"), "delta_ci_low": d.get("ci_low"), "delta_ci_high": d.get("ci_high"),
                             "delta_verdict": verdict(d),
                             "R_policy_M": a.get("M"), "R_policy_ci_low": a.get("ci_low"), "R_policy_ci_high": a.get("ci_high"),
                             "R_both_filled_mean": float(np.nanmean(r_pol[both])) if both.any() else np.nan,
                             "adverse_selection_gap_mid": float(np.nanmean(r_mid[both]) - np.nanmean(r_mid)) if both.any() else np.nan})
                if (q, w) == PRIMARY:
                    per_entry[side] = pd.DataFrame({"session": sess, "underlying": st["underlying"].to_numpy(),
                                                    "tte_bucket": st["tte_bucket"].to_numpy(), "R_aggressive": r_agg,
                                                    "R_mid": r_mid, "R_policy": r_pol, "delta": delta, "both_filled": both})
        base = session_boot(r_agg, sess, n_boot)
        rows.append({"side": side, "q": np.nan, "window_min": 0, "n_entries": int(len(st)), "R_policy_M": base.get("M"),
                     "R_policy_ci_low": base.get("ci_low"), "R_policy_ci_high": base.get("ci_high"),
                     "delta_verdict": "BASELINE_AGGRESSIVE"})
    grid = pd.DataFrame(rows)
    prim = grid[(grid["q"] == PRIMARY[0]) & (grid["window_min"] == PRIMARY[1])].set_index("side")
    decision = {side: {"verdict": prim.loc[side, "delta_verdict"], "delta_M": prim.loc[side, "delta_M"],
                       "delta_ci": [prim.loc[side, "delta_ci_low"], prim.loc[side, "delta_ci_high"]],
                       "R_policy_M": prim.loc[side, "R_policy_M"],
                       "R_policy_ci": [prim.loc[side, "R_policy_ci_low"], prim.loc[side, "R_policy_ci_high"]],
                       "fill_rate_both": prim.loc[side, "fill_rate_both"]} for side in ("buy", "sell")}
    return {"grid": grid, "per_entry": per_entry, "decision": decision}


def registration() -> dict:
    return {"version": VERSION, "q_levels": list(Q_LEVELS), "windows_min": list(WINDOWS_MIN), "primary": list(PRIMARY),
            "n_boot": N_BOOT, "seed": SEED, "policy": "passive-then-cross", "entries": "VRP 02 (hourly, TTE 2-20h, ATM)",
            "decision": "per side: session-bootstrap CI of paired Delta R (policy - aggressive) > 0 PASSIVE_HELPS, < 0 PASSIVE_HURTS"}


# --------------------------------------------------------------------------
# Self test
# --------------------------------------------------------------------------

def selftest() -> dict:
    # Limit prices and ticks
    p, t = limit_price([1.0], [0.9], [1.1], [0.01], "buy", 0.5)
    assert np.isclose(p[0], 0.95) and not t[0]
    p, t = limit_price([1.0], [0.99], [1.01], [0.01], "buy", 0.0)
    assert np.isclose(p[0], 1.0) and not t[0]
    p, t = limit_price([1.005], [1.0], [1.01], [0.01], "sell", 0.5)
    assert np.isclose(p[0], 1.01) and t[0]                        # one-tick spread joins the touch
    # Fill rule
    paths = {"best_bid_price": np.array([[0.9, 0.9, 0.9]]), "best_ask_price": np.array([[1.1, 0.95, 1.1]]),
             "last_price": np.array([[1.0, 1.0, 1.0]]), "traded": np.array([[0.0, 0.0, 0.0]])}
    assert first_fill(paths, np.array([0.95]), "buy", 3)[0] == 2
    assert first_fill(paths, np.array([0.94]), "buy", 3)[0] == 0
    paths["traded"] = np.array([[1.0, 0.0, 0.0]])
    paths["last_price"] = np.array([[0.93, 1.0, 1.0]])
    assert first_fill(paths, np.array([0.94]), "buy", 3)[0] == 1
    assert first_fill(paths, np.array([0.94]), "buy", 0 + 1)[0] == 1
    # Short return arithmetic
    st = pd.DataFrame({"S_T": [100.0], "K": [100.0], "inverse": [False]})
    assert np.isclose(returns(st, np.array([10.0]), np.array([0.0]), "sell")[0], 1.0)
    assert np.isclose(returns(st.assign(S_T=[115.0]), np.array([10.0]), np.array([0.0]), "sell")[0], -0.5)
    return {"version": VERSION, "ok": True}
