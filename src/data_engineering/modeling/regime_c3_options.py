"""Regime Structure 02 — does the C3 signal (up-displacement -> shock) pay for long 0DTE options?

Pre-registration: discussions/Regime_Structure/005_claude_rs02_preregistration.md.

Signal (from RS 01, scale 15 min, thresholds frozen on the development window, regime_structure v1.2):
a clock-aligned 15-minute window labelled DISP with net > 0 on Binance USD-M 1m klines. Signal time t_s =
window end. Assets BTC, ETH, SOL, XRP.

Part A — Binance, 2 years (power check: does the signal raise the move an option is paid on?)
    z_h = |log P(t_s + h) - log P(t_s)| / (sigma_pre * sqrt(h / 5 min)),
    h in {60 min, to the next 08:00 UTC settlement}; settlement horizon only for 120..1200 minutes to 08:00.
    Ratio = mean z over signals / time-of-day-matched mean z over all other eligible 15m window ends
    (non-signal hour-of-day means weighted by the signals' hour distribution). Session-day bootstrap.

Part B — Deribit collector quotes (executable, few sessions)
    Long ATM 0DTE straddle at the ask + taker fee, held to the 08:00 UTC settlement, entered at the first
    5-minute collector snapshot at or after t_s (within 10 min), TTE 120..1200 min (VRP 02 conventions,
    reused with Ivan's approval, RS 005 §2). Comparison set: every 5-minute entry of the same sessions.
    Primary statistic: M_signal = mean over sessions of the session mean of R_net over signal entries.
    Secondary: Delta = session mean R_net(signal) - session mean R_net(all entries), sessions with a signal;
    ATM call only; straddle sold at the bid 60 minutes later (exit variant).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from data_engineering.modeling import option_pricing as op
from data_engineering.modeling import regime_structure as rs
from data_engineering.modeling import vrp_long_straddle as ls

VERSION = "regime_c3_options v1.0"
TEST_START = rs.TEST_START
ASSETS = rs.ASSETS


@dataclass(frozen=True)
class Params:
    scale: int = 3                  # 15-minute windows (5m bars)
    entry_grid_min: int = 5
    entry_tolerance_min: int = 10
    exit_after_min: int = 60
    horizon_short_min: int = 60
    n_boot: int = 4000
    seed: int = 20261013
    alpha_a: float = 0.05           # Part A: two ratios, Bonferroni -> 97.5% CIs

    @property
    def ci_a(self) -> float:
        return 1 - self.alpha_a / 2


def registration(p: Params | None = None) -> dict:
    p = p or Params()
    d = asdict(p)
    d.update(version=VERSION, signal="RS 01 v1.2 label DISP with net>0 at the 15-minute scale",
             test_start=str(TEST_START), vrp_conventions=ls.registration(),
             part_a="ratio of matched mean normalised |move| (60 min, to settlement); 97.5% day-bootstrap CI; test window",
             part_b="M_signal (session mean R_net of signal straddles), 95% session-bootstrap CI; VRP verdict labels")
    return d


# ---------------------------------------------------------------- signals (Binance)

def signal_frame(df: pd.DataFrame, p: Params, rsp: rs.Params | None = None) -> tuple:
    """All 15m windows with label, signal flag, end time, price and sigma at the window end.
    Returns (frame, minute log-price path from t0, t0)."""
    rsp = rsp or rs.Params()
    m = rs.prepare_minutes(df)
    b = rs.bars5(m, rsp)
    h = rs.windows(b, p.scale, rsp)
    th = rs.finalize_thresholds(rs.thresholds(h, rsp), rsp)
    lab = rs.label(h, th, rsp)
    out = lab[["t", "label", "up", "window"]].copy()
    out["t_end"] = out["t"] + pd.Timedelta(minutes=p.scale * rsp.bar_min)
    out["signal"] = (out["label"] == "DISP") & out["up"]
    # sigma (per 5m bar) at the last bar of the window; log price at the window end
    nw = len(out)
    out["sigma5"] = b["sig"][p.scale - 1: nw * p.scale: p.scale][:nw]
    logc = np.r_[0.0, np.cumsum(m.r1)]                        # log price relative to t0, minute grid
    idx_end = ((out["t_end"] - m.t0) / pd.Timedelta(minutes=1)).astype(int).to_numpy()
    out["logp"] = logc[np.minimum(idx_end, len(logc) - 1)]
    return out, logc, m.t0


def part_a_rows(sf: pd.DataFrame, logc: np.ndarray, t0: pd.Timestamp, p: Params) -> pd.DataFrame:
    """Normalised absolute moves after every eligible window end (signal or not)."""
    n = len(logc) - 1
    te = sf["t_end"]
    i0 = ((te - t0) / pd.Timedelta(minutes=1)).astype(int).to_numpy()
    settle = te.dt.floor("1D") + pd.Timedelta(hours=8)
    settle = settle.where(settle > te, settle + pd.Timedelta(days=1))
    tte = ((settle - te) / pd.Timedelta(minutes=1)).to_numpy()
    rows = sf[["t_end", "signal", "window", "sigma5"]].copy()
    rows["hour"] = te.dt.hour
    rows["session"] = settle.dt.strftime("%Y-%m-%d")
    for name, hz in (("z60", np.full(len(sf), p.horizon_short_min)), ("zsettle", tte)):
        j = i0 + hz.astype(int)
        ok = (j <= n) & np.isfinite(sf["sigma5"].to_numpy()) & (sf["sigma5"].to_numpy() > 0) & sf["label"].notna().to_numpy()
        if name == "zsettle":
            ok &= (tte >= ls.MIN_TTE_MIN) & (tte <= ls.MAX_TTE_MIN)
        jj = np.minimum(j, n)
        z = np.abs(logc[jj] - logc[np.minimum(i0, n)]) / (sf["sigma5"].to_numpy() * np.sqrt(hz / 5.0))
        rows[name] = np.where(ok, z, np.nan)
    return rows


def matched_ratio(rows: pd.DataFrame, col: str, p: Params, rng: np.random.Generator) -> dict:
    """mean z(signal) / hour-matched mean z(non-signal); bootstrap over settlement days."""
    d = rows[np.isfinite(rows[col])]
    sig, non = d[d["signal"]], d[~d["signal"]]
    if len(sig) < 30:
        return {"n_signal": int(len(sig)), "ratio": np.nan, "lo": np.nan, "hi": np.nan}
    days = np.array(sorted(d["session"].unique()))
    di = {s: k for k, s in enumerate(days)}
    D = len(days)
    # per day x hour sums and counts
    def cube(x):
        S = np.zeros((D, 24)); N = np.zeros((D, 24))
        np.add.at(S, (x["session"].map(di).to_numpy(), x["hour"].to_numpy()), x[col].to_numpy())
        np.add.at(N, (x["session"].map(di).to_numpy(), x["hour"].to_numpy()), 1.0)
        return S, N
    Ss, Ns = cube(sig)
    Sn, Nn = cube(non)

    def stat(w):
        ss, ns = w @ Ss, w @ Ns
        sn, nn = w @ Sn, w @ Nn
        with np.errstate(invalid="ignore", divide="ignore"):
            mh = np.where(nn > 0, sn / nn, np.nan)
            wh = ns / ns.sum()
            ok = (ns > 0) & np.isfinite(mh)
            return (ss.sum() / ns.sum()) / np.sum(wh[ok] * mh[ok]) * (wh[ok].sum())
    est = stat(np.ones(D))
    W = rng.multinomial(D, np.full(D, 1.0 / D), size=p.n_boot).astype(float)
    bs = np.array([stat(w) for w in W])
    a = (1 - p.ci_a) / 2
    lo, hi = np.nanquantile(bs, [a, 1 - a])
    return {"n_signal": int(len(sig)), "n_other": int(len(non)), "days": D,
            "mean_z_signal": float(sig[col].mean()), "ratio": float(est), "lo": float(lo), "hi": float(hi)}


# ---------------------------------------------------------------- Part B (collector quotes)

def prepare_legs_grid(chain: pd.DataFrame, grid_min: int) -> pd.DataFrame:
    """VRP 02 prepare_legs with a configurable entry grid (VRP uses 60 min)."""
    q = chain.copy()
    q["state_time"] = pd.to_datetime(q["state_time"], utc=True)
    for c in ("strike", "option_forward_price", "minutes_to_expiry", "best_bid_price", "best_ask_price"):
        q[c] = pd.to_numeric(q[c], errors="coerce")
    q["is_call"] = q["option_type"].astype(str).str.lower().str[0].eq("c")
    q["inverse"] = op.is_inverse(q["instrument_type"])
    F = q["option_forward_price"].to_numpy(float)
    q["bid_usd"] = op.to_usd(q["best_bid_price"].to_numpy(float), F, q["inverse"])
    q["ask_usd"] = op.to_usd(q["best_ask_price"].to_numpy(float), F, q["inverse"])
    keep = ((q["state_time"].dt.minute % grid_min == 0) & np.isfinite(F) & (F > 0) & np.isfinite(q["strike"])
            & q["minutes_to_expiry"].notna())
    q = q[keep].copy()
    q["two_sided"] = (q["bid_usd"] > 0) & (q["ask_usd"] >= q["bid_usd"])
    q["expiry_time"] = (q["state_time"] + pd.to_timedelta(q["minutes_to_expiry"].round(), unit="min")).dt.round("5min")
    q["session"] = q["expiry_time"].dt.strftime("%Y-%m-%d")
    q = q[pd.to_datetime(q["session"], utc=True) < ls.VRP_CONFIRMATION_START]     # VRP holdout never loaded
    return q.reset_index(drop=True)


def entry_straddles(legs: pd.DataFrame) -> pd.DataFrame:
    """VRP ATM straddle per (time, underlying, family), restricted to 120..1200 minutes to expiry."""
    st = ls.select_straddles(legs)
    return st[(st["minutes_to_expiry"] >= ls.MIN_TTE_MIN) & (st["minutes_to_expiry"] <= ls.MAX_TTE_MIN)].reset_index(drop=True)


def tag_signals(st: pd.DataFrame, signals: pd.DataFrame, p: Params) -> pd.DataFrame:
    """signal_entry = this straddle is the first snapshot at/after a signal time (within tolerance)."""
    x = st.copy()
    x["signal_entry"] = False
    for u, g in x.groupby("underlying"):
        s = signals[signals["asset"] == u][["t_end"]].sort_values("t_end")
        if s.empty:
            continue
        times = g[["state_time"]].drop_duplicates().sort_values("state_time")
        m = pd.merge_asof(s, times, left_on="t_end", right_on="state_time", direction="forward",
                          tolerance=pd.Timedelta(minutes=p.entry_tolerance_min)).dropna()
        hit = set(m["state_time"])
        x.loc[g.index, "signal_entry"] = g["state_time"].isin(hit).to_numpy()
    return x


def add_variants(x: pd.DataFrame, legs: pd.DataFrame, p: Params) -> pd.DataFrame:
    """ATM call only (held to settlement) and straddle sold at the bid after exit_after_min."""
    x = x.copy()
    ST, K, F = (x[c].to_numpy(float) for c in ("S_T", "K", "F"))
    conv = np.where(x["inverse"].to_numpy(bool), ST / F, 1.0)
    fee_call = np.minimum(ls.FEE_RATE * F, ls.FEE_CAP * x["ask_usd_call"].to_numpy(float))
    x["R_call_net"] = np.maximum(ST - K, 0.0) / ((x["ask_usd_call"].to_numpy(float) + fee_call) * conv) - 1.0
    # exit leg: same family and strike, exit_after_min later, both legs two-sided
    two = legs[legs["two_sided"]]
    key = ["state_time", "underlying", "option_family_id", "strike"]
    c = two[two["is_call"]].drop_duplicates(key)[key + ["bid_usd", "option_forward_price"]]
    pu = two[~two["is_call"]].drop_duplicates(key)[key + ["bid_usd"]]
    ex = c.merge(pu, on=key, suffixes=("_call", "_put"))
    ex["exit_bid_usd"] = ex["bid_usd_call"] + ex["bid_usd_put"]
    Fx = ex["option_forward_price"].to_numpy(float)
    ex["exit_fee_usd"] = (np.minimum(ls.FEE_RATE * Fx, ls.FEE_CAP * ex["bid_usd_call"].to_numpy(float))
                          + np.minimum(ls.FEE_RATE * Fx, ls.FEE_CAP * ex["bid_usd_put"].to_numpy(float)))
    ex = ex.rename(columns={"state_time": "exit_time", "strike": "K"})[["exit_time", "underlying", "option_family_id", "K",
                                                                         "exit_bid_usd", "exit_fee_usd"]]
    x["exit_time"] = x["state_time"] + pd.Timedelta(minutes=p.exit_after_min)
    x = x.merge(ex, on=["exit_time", "underlying", "option_family_id", "K"], how="left")
    # USD values at each time (coin-premium FX effect over one hour ignored; reported, not a decision)
    x["R_exit60_net"] = (x["exit_bid_usd"] - x["exit_fee_usd"]) / (x["ask_usd"] + x["fee_usd"]) - 1.0
    return x


def session_stat(x: pd.DataFrame, col: str, mask: pd.Series | None, n_boot: int, seed: int) -> dict:
    d = x if mask is None else x[mask]
    return ls.bootstrap_mean(ls.session_means(d, col).to_numpy(), n_boot=n_boot, seed=seed)


def delta_stat(x: pd.DataFrame, col: str, n_boot: int, seed: int) -> dict:
    """Session mean over signal entries minus session mean over all entries, sessions with a signal."""
    a = x[x["signal_entry"]].groupby("session")[col].mean()
    b = x.groupby("session")[col].mean().reindex(a.index)
    v = (a - b).dropna().to_numpy()
    return ls.bootstrap_mean(v, n_boot=n_boot, seed=seed)


def part_b(x: pd.DataFrame, p: Params) -> dict:
    sig = x["signal_entry"]
    prim = session_stat(x, "R_net", sig, p.n_boot, ls.SEED)
    out = {"primary": {**prim, "verdict": ls.verdict(prim), "n_signal_straddles": int(sig.sum()),
                       "n_all_straddles": int(len(x)), "n_sessions_with_signal": int(x.loc[sig, "session"].nunique())}}
    rows = []
    for col in ("R_net", "R_call_net", "R_exit60_net", "R_gross", "R_mid", "move_over_implied"):
        if col not in x:
            continue
        for name, msk in (("signal", sig), ("all", None)):
            rows.append({"metric": col, "set": name, **session_stat(x, col, msk, 1000, ls.SEED)})
        rows.append({"metric": col, "set": "delta_signal_minus_all", **delta_stat(x, col, 1000, ls.SEED)})
    out["table"] = pd.DataFrame(rows)
    return out


# ---------------------------------------------------------------- synthetic helpers

def synthetic_chain(n_sessions: int = 30, seed: int = 0, under: str = "BTC") -> tuple:
    """VRP fake chain (hourly snapshots, fair-ish prices with spread); returns (chain, settlement)."""
    return ls._fake_chain(np.random.default_rng(seed), n_sessions=n_sessions, vol=0.6, iv=0.6,
                          half_spread=0.03, under=under)
