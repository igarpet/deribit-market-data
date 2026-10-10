"""
Volatility Dynamics Module 10: physical jump model vs 0DTE option wings.

Cross-branch reuse approved by Ivan (2026-10-10): this module uses the
Volatility Risk Premium branch's Black-76 utilities (option_pricing) and its
audit findings (VRP 006): Black-76 with our F, T reprices quotes; deep OTM
0DTE quotes are tick-priced, so exchange IV is unreliable there. Module 10
therefore compares in PRICE space against bid/ask and flags floor-bound
quotes. It never uses sessions after the VRP confirmation boundary
(VRP_CONFIRMATION_START = 2026-11-07).

Question
--------
Do out-of-the-money 0DTE options price jumps at the rate and size measured
in the physical index (Modules 08-09)?

Physical model price for each quote (Merton-type, compound Poisson)
-------------------------------------------------------------------
    P_jump = sum_{n=0}^{3} Pois(n; Lambda) E[ B76(F exp(S_n - Lambda (m - 1)), K, D) ]

- D: ex-ante diffusive variance to expiry = v_t * sum_j s(slot_j)^2 w_j^2,
  v_t the BACKWARD 4h jump-robust bipower of deseasonalized returns at the
  quote time (no future data), s the Module 08 smooth clock, w_j the
  settlement TWAP weight of bar j (1 before 07:30; (08:00 - u + 2.5)/30 for a
  bar ending at u inside the 07:30-08:00 averaging window).
- Lambda = lambda_bar * sum_j w_j: expected jumps to expiry (jumps inside the
  averaging window counted with weight w_j; documented approximation).
- J: signed jump sizes = (pooled signed z of Module 08/09 jumps) x per-bar
  local sigma at the quote time; m = E[e^J] (martingale correction).
- lambda_bar (per asset) and the z distribution are estimated LEAVE-ONE-
  SESSION-OUT: a session's own jumps never price that session's options.
- n = 0 exact, n = 1 over 40 quantile nodes, n = 2, 3 by 64 fixed random
  draws of the nodes; P(n >= 4) is added to n = 3.
P_diff (no jumps) = B76(F, K, D) is reported for decomposition.

Pre-registered (fixed before any real-data Module 10 output)
------------------------------------------------------------
Sample: two-sided OTM quotes of the primary 0DTE family, 60 <= minutes to
expiry <= 1200, quote times on the 15-minute grid, not floor-bound (bid >
one tick), assets not flagged as tick artifacts by Module 09 P2, sessions
<= 2026-11-06.
PRIMARY bucket: physical moneyness z = |ln(K/F)| / sqrt(D) in [2, 5).
Statistic M: median over sessions of the session median of ln(mid / P_jump).
  Wings RICH vs the physical jump model  iff session-bootstrap 95% CI of M > 0;
  Wings CHEAP                            iff the CI < 0;
  otherwise CONSISTENT.
Reported, no decision: every z x TTE bucket; per asset; ln(mid / P_diff);
sensitivities (diffusion variance x 0.85 for the VRP 006 Part E 5m-noise
inflation; lambda x 0.5 and x 2); share of quotes with P_jump outside
[bid, ask]; hold-to-settlement P&L (buy at ask / sell at bid) against the
approximate settlement TWAP (mean index of the 07:35-08:00 bars), session
bootstrap.
"""

from __future__ import annotations

from math import exp, factorial, log
from typing import Sequence

import numpy as np
import pandas as pd

from data_engineering.modeling import option_pricing as op
from data_engineering.modeling import volatility_vd_b_core as vb
from data_engineering.modeling import volatility_jump_hawkes as jh


VERSION = "0.1.0"
VRP_CONFIRMATION_START = pd.Timestamp("2026-11-07", tz="UTC")
MIN_TTE_MIN = 60
MAX_TTE_MIN = 1200
QUOTE_GRID_MIN = 15
PRIMARY_Z = (2.0, 5.0)
Z_EDGES = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0, np.inf)
TTE_EDGES_MIN = (60, 120, 240, 480, 960, 1200)
N_NODES = 40
N_DRAWS = 64
N_BOOT = 4000
SETTLE_WINDOW_MIN = 30
BACKWARD_PAIRS = 47
YEAR_MIN = 365.0 * 24 * 60


# --------------------------------------------------------------------------
# Fast vectorized Black-76 (no scipy on the research PC; option_pricing falls
# back to a per-element math.erf, far too slow for ~10^8 evaluations)
# --------------------------------------------------------------------------

def _erfc(x: np.ndarray) -> np.ndarray:
    """Chebyshev erfc (Numerical Recipes erfcc), RELATIVE error < 1.2e-7 for all x:
    accurate in the far tail, which is where deep-OTM 0DTE prices live."""
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    r = t * np.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (
        -0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return np.where(x >= 0, r, 2.0 - r)


def ncdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * _erfc(-np.asarray(x, float) / np.sqrt(2.0))


def b76(F, K, T, sigma, is_call) -> np.ndarray:
    """Undiscounted Black-76 in USD; same convention as option_pricing.black76_usd.
    Puts use N(-d) directly (no parity subtraction) to keep tail precision."""
    F, K, T, sigma = (np.asarray(a, float) for a in (F, K, T, sigma))
    is_call = np.asarray(is_call, bool)
    sT = sigma * np.sqrt(np.clip(T, 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * sT * sT) / sT
        d2 = d1 - sT
        call = F * ncdf(d1) - K * ncdf(d2)
        put = K * ncdf(-d2) - F * ncdf(-d1)
    px = np.where(is_call, call, put)
    intrinsic = np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    return np.where(sT > 0, np.maximum(px, 0.0), intrinsic)


# --------------------------------------------------------------------------
# Ex-ante diffusion
# --------------------------------------------------------------------------

def backward_bipower(panel: jh.Panel, S: np.ndarray) -> np.ndarray:
    """Per-bar deseasonalized bipower over the previous 4h, current bar excluded."""
    A = np.abs(panel.R / S[panel.slot, :])
    T, N = A.shape
    prod = np.vstack([np.full((1, N), np.nan), A[1:] * A[:-1]])
    fin = np.isfinite(prod)
    cs = np.vstack([np.zeros((1, N)), np.cumsum(np.where(fin, prod, 0.0), axis=0)])
    cc = np.vstack([np.zeros((1, N)), np.cumsum(fin, axis=0)])
    BV = np.full((T, N), np.nan)
    t = np.arange(BACKWARD_PAIRS + 1, T)
    s_ = cs[t] - cs[t - BACKWARD_PAIRS]
    c_ = cc[t] - cc[t - BACKWARD_PAIRS]
    with np.errstate(invalid="ignore", divide="ignore"):
        v = (np.pi / 2.0) * s_ / c_
    v[c_ < 0.75 * BACKWARD_PAIRS] = np.nan
    BV[t] = v
    return BV


def settlement_weight(minutes_before_expiry_at_bar_end: np.ndarray) -> np.ndarray:
    """TWAP weight of a 5m bar ending m minutes before the 08:00 expiry."""
    m = np.asarray(minutes_before_expiry_at_bar_end, float)
    return np.where(m >= SETTLE_WINDOW_MIN, 1.0, np.clip((m + 2.5) / SETTLE_WINDOW_MIN, 0.0, 1.0))


def remaining_sums(quote_slot: np.ndarray, n_bars: np.ndarray, S_asset: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For a quote at bar slot q with n future bars to expiry:
       sum s^2 w^2 (diffusion), sum w (jump count), rms s (per-bar jump scale)."""
    q = np.asarray(quote_slot, int)
    n = np.asarray(n_bars, int)
    out_d = np.zeros(len(q))
    out_w = np.zeros(len(q))
    out_s = np.zeros(len(q))
    for i in range(len(q)):
        if n[i] <= 0:
            continue
        slots = (q[i] + np.arange(1, n[i] + 1)) % jh.BARS_PER_DAY
        mins_before = 5.0 * (n[i] - np.arange(1, n[i] + 1))
        w = settlement_weight(mins_before)
        s2 = S_asset[slots] ** 2
        out_d[i] = np.sum(s2 * w * w)
        out_w[i] = np.sum(w)
        out_s[i] = np.sqrt(np.mean(s2))
    return out_d, out_w, out_s


# --------------------------------------------------------------------------
# Jump parameters (leave-one-session-out)
# --------------------------------------------------------------------------

def session_of_bar(panel: jh.Panel) -> np.ndarray:
    """0DTE session id = UTC date of the next 08:00 expiry (bar end <= 08:00)."""
    t = pd.to_datetime(panel.time_ns, utc=True)
    exp_date = (t - pd.Timedelta(minutes=1) - pd.Timedelta(hours=8)).floor("D") + pd.Timedelta(days=1)
    return exp_date.strftime("%Y-%m-%d").to_numpy()


def loso_jump_params(panel, J, valid, Z, R, sessions_of_bar: np.ndarray, assets_used: Sequence[str]):
    """Per session: per-asset lambda per bar and pooled signed-z quantile nodes, excluding that session."""
    a_idx = [panel.assets.index(a) for a in assets_used]
    jt, ja = np.nonzero((J > 0) & valid)
    keep = np.isin(ja, a_idx)
    jt, ja = jt[keep], ja[keep]
    signed_z = np.sign(R[jt, ja]) * Z[jt, ja]
    jsess = sessions_of_bar[jt]
    params = {}
    probs = (np.arange(N_NODES) + 0.5) / N_NODES
    for s in np.unique(sessions_of_bar):
        out_rows = sessions_of_bar != s
        lam = {}
        for a in a_idx:
            nb = valid[out_rows, a].sum()
            k = J[out_rows & valid[:, a], a].sum()
            lam[panel.assets[a]] = float(k / nb) if nb else np.nan
        zs = signed_z[jsess != s]
        nodes = np.quantile(zs, probs) if len(zs) >= 10 else np.full(N_NODES, np.nan)
        params[s] = {"lambda_bar": lam, "z_nodes": nodes, "n_jumps_used": int(len(zs))}
    return params


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------

def merton_price(F, K, T, D, Lam, sigma_bar, z_nodes, is_call, rng_seed: int = 1010) -> np.ndarray:
    """Compound-Poisson + diffusion price (USD, undiscounted); vectorized over rows.

    F, K, T, D, Lam, sigma_bar, is_call: (n,) arrays; z_nodes: (n, Q) signed-z nodes.
    """
    F, K, T, D, Lam, sigma_bar = (np.asarray(x, float) for x in (F, K, T, D, Lam, sigma_bar))
    is_call = np.asarray(is_call, bool)
    nodes = np.asarray(z_nodes, float)
    Jn = nodes * sigma_bar[:, None]                                  # (n, Q) log jump sizes
    m = np.mean(np.exp(Jn), axis=1)                                  # E[e^J]
    comp = -Lam * (m - 1.0)
    sig = np.sqrt(np.maximum(D, 1e-18) / T)
    p = np.exp(-Lam)
    w = [p, p * Lam, p * Lam ** 2 / 2.0]
    w.append(np.maximum(1.0 - (w[0] + w[1] + w[2]), 0.0))          # n >= 3 mass on n = 3

    def bb(Fx):
        return b76(Fx, K[:, None], T[:, None], sig[:, None], is_call[:, None])

    price = w[0] * b76(F * np.exp(comp), K, T, sig, is_call)
    price = price + w[1] * bb(F[:, None] * np.exp(comp[:, None] + Jn)).mean(axis=1)
    rng = np.random.default_rng(int(rng_seed))
    Q = nodes.shape[1]
    for n_j, wj in ((2, w[2]), (3, w[3])):
        pick = rng.integers(0, Q, size=(N_DRAWS, n_j))
        Sn = Jn[:, pick].sum(axis=2)                                  # (n, draws)
        price = price + wj * bb(F[:, None] * np.exp(comp[:, None] + Sn)).mean(axis=1)
    return price


def diffusion_price(F, K, T, D, is_call) -> np.ndarray:
    return b76(F, K, T, np.sqrt(np.maximum(np.asarray(D, float), 1e-18) / np.asarray(T, float)), is_call)


# --------------------------------------------------------------------------
# Quotes
# --------------------------------------------------------------------------

def prepare_quotes(chain: pd.DataFrame) -> pd.DataFrame:
    """Normalize the VRP-style chain: USD prices, OTM, two-sided, TTE window, 15m grid."""
    q = chain.copy()
    q["state_time"] = pd.to_datetime(q["state_time"], utc=True)
    q = q[q["state_time"] < VRP_CONFIRMATION_START]
    q["is_call"] = q["option_type"].astype(str).str.lower().str[0].eq("c")
    q["inverse"] = op.is_inverse(q["instrument_type"])
    for c in ("strike", "option_forward_price", "minutes_to_expiry", "best_bid_price", "best_ask_price"):
        q[c] = pd.to_numeric(q[c], errors="coerce")
    F = q["option_forward_price"].to_numpy(float)
    q["bid_usd"] = op.to_usd(q["best_bid_price"].to_numpy(float), F, q["inverse"])
    q["ask_usd"] = op.to_usd(q["best_ask_price"].to_numpy(float), F, q["inverse"])
    two = (q["bid_usd"] > 0) & (q["ask_usd"] >= q["bid_usd"])
    otm = np.where(q["is_call"], q["strike"] > q["option_forward_price"], q["strike"] < q["option_forward_price"])
    mins = q["minutes_to_expiry"]
    grid_ok = (q["state_time"].dt.minute % QUOTE_GRID_MIN) == 0
    q = q[two & otm & (mins >= MIN_TTE_MIN) & (mins <= MAX_TTE_MIN) & grid_ok].copy()
    q["mid_usd"] = 0.5 * (q["bid_usd"] + q["ask_usd"])
    q["T"] = q["minutes_to_expiry"] / YEAR_MIN
    q["expiry_time"] = (q["state_time"] + pd.to_timedelta(q["minutes_to_expiry"].round(), unit="min")).dt.round("5min")
    q["session"] = q["expiry_time"].dt.strftime("%Y-%m-%d")
    return q.reset_index(drop=True)


def tick_floors(q: pd.DataFrame) -> pd.DataFrame:
    """Minimum quoted price increment per (underlying, instrument_type): the
    smallest native bid that occurs at least 20 times (guards stray prints)."""
    rows = []
    for (u, it), g in q.groupby(["underlying", "instrument_type"]):
        b = g["best_bid_price"].to_numpy(float)
        b = b[np.isfinite(b) & (b > 0)]
        if not len(b):
            continue
        vals, cnt = np.unique(np.round(b, 10), return_counts=True)
        common = vals[cnt >= 20]
        tick = float(common.min()) if len(common) else float(vals.min())
        rows.append({"underlying": u, "instrument_type": it, "min_tick_native": tick,
                     "share_bid_at_tick": float(np.mean(np.isclose(b, tick, rtol=1e-6)))})
    return pd.DataFrame(rows)


def attach_model(q: pd.DataFrame, panel: jh.Panel, S: np.ndarray, BVb: np.ndarray, params: dict,
                 *, diff_scale: float = 1.0, lam_scale: float = 1.0) -> pd.DataFrame:
    """Model prices for every quote (ex-ante diffusion + LOSO jump parameters)."""
    q = q.copy()
    t_ns = q["state_time"].astype("int64").to_numpy()
    if pd.api.types.is_datetime64_any_dtype(q["state_time"]):
        t_ns = pd.DatetimeIndex(q["state_time"]).as_unit("ns").asi8
    bar = t_ns // vb.BAR_NS
    ti = bar - panel.bar[0]
    a_map = {a: i for i, a in enumerate(panel.assets)}
    ai = q["underlying"].map(a_map).to_numpy()
    ok = (ti >= 0) & (ti < panel.T) & pd.notna(ai)
    q = q[ok].reset_index(drop=True)
    ti, ai = ti[ok].astype(int), ai[ok].astype(int)
    v = BVb[ti, ai]
    n_bars = np.round(q["minutes_to_expiry"].to_numpy(float) / 5.0).astype(int)
    slots = panel.slot[ti]
    D_units = np.zeros(len(q))
    W = np.zeros(len(q))
    srms = np.zeros(len(q))
    for a in np.unique(ai):
        m = ai == a
        d_, w_, s_ = remaining_sums(slots[m], n_bars[m], S[:, a])
        D_units[m], W[m], srms[m] = d_, w_, s_
    q["D"] = diff_scale * v * D_units
    q["sigma_bar"] = np.sqrt(v) * srms
    lam = np.array([params.get(s, {}).get("lambda_bar", {}).get(u, np.nan)
                    for s, u in zip(q["session"], q["underlying"])])
    q["Lambda"] = lam_scale * lam * W
    nodes = np.vstack([params.get(s, {}).get("z_nodes", np.full(N_NODES, np.nan)) for s in q["session"]])
    good = np.isfinite(q["D"]) & (q["D"] > 0) & np.isfinite(q["Lambda"]) & np.isfinite(nodes).all(axis=1)
    q = q[good].reset_index(drop=True)
    nodes = nodes[good.to_numpy()]
    F, K, T = (q[c].to_numpy(float) for c in ("option_forward_price", "strike", "T"))
    q["P_diff"] = diffusion_price(F, K, T, q["D"].to_numpy(), q["is_call"].to_numpy())
    P = np.empty(len(q))
    for s in range(0, len(q), 20000):
        e = slice(s, s + 20000)
        P[e] = merton_price(F[e], K[e], T[e], q["D"].to_numpy()[e], q["Lambda"].to_numpy()[e],
                            q["sigma_bar"].to_numpy()[e], nodes[e], q["is_call"].to_numpy()[e])
    q["P_jump"] = P
    q["z_phys"] = np.abs(np.log(K / F)) / np.sqrt(q["D"].to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        q["log_mid_over_jump"] = np.log(q["mid_usd"] / q["P_jump"])
        q["log_mid_over_diff"] = np.log(q["mid_usd"] / q["P_diff"])
    q["jump_below_bid"] = q["P_jump"] < q["bid_usd"]
    q["jump_above_ask"] = q["P_jump"] > q["ask_usd"]
    q["z_bucket"] = pd.cut(q["z_phys"], Z_EDGES, right=False).astype(str)
    q["tte_bucket"] = pd.cut(q["minutes_to_expiry"], TTE_EDGES_MIN, right=True).astype(str)
    return q


def reprice(q: pd.DataFrame, params: dict, *, diff_scale: float = 1.0, lam_scale: float = 1.0) -> np.ndarray:
    """ln(mid / P_jump) on the SAME rows with scaled diffusion variance / jump rate
    (sensitivities keep the pre-registered sample fixed)."""
    nodes = np.vstack([params[s_]["z_nodes"] for s_ in q["session"]])
    F, K, T = (q[c].to_numpy(float) for c in ("option_forward_price", "strike", "T"))
    D = diff_scale * q["D"].to_numpy(float)
    L = lam_scale * q["Lambda"].to_numpy(float)
    P = np.empty(len(q))
    for s0 in range(0, len(q), 20000):
        e = slice(s0, s0 + 20000)
        P[e] = merton_price(F[e], K[e], T[e], D[e], L[e], q["sigma_bar"].to_numpy(float)[e], nodes[e],
                            q["is_call"].to_numpy()[e])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(q["mid_usd"].to_numpy(float) / P)


def flag_floor(q: pd.DataFrame, floors: pd.DataFrame) -> pd.DataFrame:
    q = q.merge(floors[["underlying", "instrument_type", "min_tick_native"]], on=["underlying", "instrument_type"], how="left")
    q["floor_bound"] = q["best_bid_price"] <= q["min_tick_native"] * (1 + 1e-6)
    return q


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def session_median_stat(df: pd.DataFrame, col: str, *, n_boot: int = N_BOOT, seed: int = 1011) -> dict:
    x = df[np.isfinite(df[col])]
    if x.empty:
        return {"n_rows": 0, "n_sessions": 0}
    sm = x.groupby("session")[col].median()
    vals = sm.to_numpy(float)
    rng = np.random.default_rng(int(seed))
    boots = np.array([np.median(rng.choice(vals, size=len(vals), replace=True)) for _ in range(int(n_boot))])
    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    verdict = "RICH" if lo > 0 else ("CHEAP" if hi < 0 else "CONSISTENT")
    return {"n_rows": int(len(x)), "n_sessions": int(len(vals)), "M": float(np.median(vals)),
            "ci_low": lo, "ci_high": hi, "verdict": verdict,
            "mean_ratio_mid_over_model": float(np.exp(x[col]).mean())}


def primary_sample(q: pd.DataFrame, excluded_assets: Sequence[str] = ()) -> pd.DataFrame:
    return q[(~q["floor_bound"]) & (~q["underlying"].isin(list(excluded_assets)))
             & (q["z_phys"] >= PRIMARY_Z[0]) & (q["z_phys"] < PRIMARY_Z[1])]


def bucket_table(q: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    rows = []
    base = q[~q["floor_bound"]]
    for (zb, tb), g in base.groupby(["z_bucket", "tte_bucket"]):
        r = session_median_stat(g, "log_mid_over_jump", n_boot=n_boot)
        r2 = session_median_stat(g, "log_mid_over_diff", n_boot=n_boot)
        rows.append({"z_bucket": zb, "tte_bucket": tb, **{f"jump_{k}": v for k, v in r.items()},
                     "diff_M": r2.get("M", np.nan),
                     "share_model_below_bid": float(g["jump_below_bid"].mean()),
                     "share_model_above_ask": float(g["jump_above_ask"].mean())})
    fl = q.groupby(["z_bucket"])["floor_bound"].mean().rename("share_floor_bound").reset_index()
    return pd.DataFrame(rows).merge(fl, on="z_bucket", how="left")


def settlement_prices(grid: vb.Grid, sessions: Sequence[str]) -> pd.DataFrame:
    """Approximate TWAP = mean index of the bars ending 07:35..08:00 of each expiry date."""
    f = grid.frame
    idx = pd.to_numeric(f["futures_index_price"], errors="coerce")
    t = pd.to_datetime(f["state_time"], utc=True)
    rows = []
    for s in sessions:
        lo = pd.Timestamp(f"{s} 07:35", tz="UTC")
        hi = pd.Timestamp(f"{s} 08:00", tz="UTC")
        m = (t >= lo) & (t <= hi)
        g = pd.DataFrame({"u": f.loc[m, "underlying"], "p": idx[m]}).groupby("u")["p"].agg(["mean", "count"])
        for u, r in g.iterrows():
            if r["count"] >= 5:
                rows.append({"session": s, "underlying": u, "settle_twap_approx": float(r["mean"])})
    return pd.DataFrame(rows)


def settlement_pnl(q: pd.DataFrame, settle: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    x = q.merge(settle, on=["session", "underlying"], how="inner")
    if x.empty:
        return pd.DataFrame()
    ST = x["settle_twap_approx"].to_numpy(float)
    K = x["strike"].to_numpy(float)
    F = x["option_forward_price"].to_numpy(float)
    pay = np.where(x["is_call"], np.maximum(ST - K, 0.0), np.maximum(K - ST, 0.0))
    conv = np.where(x["inverse"], ST / F, 1.0)                      # coin premium valued at settlement
    x["pnl_buy_ask_bps"] = (pay - x["ask_usd"] * conv) / F * 1e4
    x["pnl_sell_bid_bps"] = (x["bid_usd"] * conv - pay) / F * 1e4
    x["model_edge_buy_bps"] = (x["P_jump"] - x["ask_usd"]) / F * 1e4
    rows = []
    rng = np.random.default_rng(1012)
    for zb, g in x[~x["floor_bound"]].groupby("z_bucket"):
        for col in ("pnl_buy_ask_bps", "pnl_sell_bid_bps", "model_edge_buy_bps"):
            sm = g.groupby("session")[col].mean().to_numpy(float)
            if len(sm) < 3:
                continue
            boots = np.array([rng.choice(sm, size=len(sm), replace=True).mean() for _ in range(int(n_boot))])
            rows.append({"z_bucket": zb, "metric": col, "n_rows": int(len(g)), "n_sessions": int(len(sm)),
                         "mean_of_session_means": float(sm.mean()),
                         "ci_low": float(np.quantile(boots, 0.025)), "ci_high": float(np.quantile(boots, 0.975))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Self test
# --------------------------------------------------------------------------

def selftest() -> dict:
    from math import erfc as _m_erfc
    xs = np.array([-9.0, -6.0, -3.0, -1.0, 0.0, 0.5, 2.0, 5.0, 8.0])
    exact = np.array([_m_erfc(v) for v in xs])
    assert np.all(np.abs(_erfc(xs) / exact - 1) < 2e-7), _erfc(xs) / exact
    rng0 = np.random.default_rng(0)
    Fz, Kz = 100 * np.ones(200), rng0.uniform(70, 130, 200)
    Tz, sz, cz = np.full(200, 0.001), np.full(200, 0.8), rng0.random(200) < 0.5
    ref = op.black76_usd(Fz, Kz, Tz, sz, cz)
    mine = b76(Fz, Kz, Tz, sz, cz)
    big = ref > 1e-10
    assert np.all(np.abs(mine[big] / ref[big] - 1) < 1e-5), np.max(np.abs(mine[big] / ref[big] - 1))
    rng = np.random.default_rng(3)
    n = 400
    F = np.full(n, 100.0)
    K = rng.uniform(85, 115, n)
    T = np.full(n, 6 * 60 / YEAR_MIN)
    D = np.full(n, (0.01) ** 2)
    Lam = np.full(n, 0.3)
    sb = np.full(n, 0.002)
    nodes = np.tile(np.sort(rng.choice([-1, 1], 40) * (4 + rng.exponential(1.0, 40))), (n, 1))
    call = merton_price(F, K, T, D, Lam, sb, nodes, np.ones(n, bool))
    put = merton_price(F, K, T, D, Lam, sb, nodes, np.zeros(n, bool))
    # 1. put-call parity under the martingale-corrected model
    assert np.allclose(call - put, F - K, atol=2e-3), np.max(np.abs(call - put - (F - K)))
    # 2. no jumps -> Black-76
    c0 = merton_price(F, K, T, D, np.zeros(n), sb, nodes, np.ones(n, bool))
    assert np.allclose(c0, diffusion_price(F, K, T, D, np.ones(n, bool)), rtol=1e-10, atol=1e-12)
    # 3. Monte Carlo check of the semi-analytic price for one far-OTM call
    k = 106.0
    sims = 400000
    N = rng.poisson(0.3, sims)
    Jsum = np.zeros(sims)
    for j in range(1, N.max() + 1):
        Jsum += np.where(N >= j, rng.choice(nodes[0], sims) * 0.002, 0.0)
    m = np.mean(np.exp(nodes[0] * 0.002))
    logF = np.log(100.0) + Jsum - 0.3 * (m - 1) - 0.5 * D[0] + np.sqrt(D[0]) * rng.normal(size=sims)
    mc = np.mean(np.maximum(np.exp(logF) - k, 0.0))
    an = merton_price(F[:1], np.array([k]), T[:1], D[:1], Lam[:1], sb[:1], nodes[:1], np.ones(1, bool))[0]
    assert abs(an - mc) < 0.03 * max(mc, 1e-3) + 2e-4, (an, mc)
    # 4. jumps add wing value
    assert np.all(call[K > 108] >= c0[K > 108] - 1e-12)
    # 5. settlement weights
    w = settlement_weight(np.array([60.0, 30.0, 25.0, 0.0]))
    assert np.allclose(w, [1.0, 1.0, 27.5 / 30, 2.5 / 30])
    # 6. statistic recovers a planted log premium of +0.3 over sessions
    df = pd.DataFrame({"session": np.repeat([f"s{i}" for i in range(15)], 30),
                       "x": 0.3 + rng.normal(scale=0.2, size=450)})
    st = session_median_stat(df, "x", n_boot=500)
    assert st["verdict"] == "RICH" and abs(st["M"] - 0.3) < 0.15
    return {"fast_b76_matches_option_pricing": True, "put_call_parity": True, "zero_jump_is_black76": True, "monte_carlo_agrees": True,
            "jumps_add_wing_value": True, "settlement_weights": True, "session_statistic": True}
