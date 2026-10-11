"""
Volatility Dynamics Module 10b: model-free physical benchmark for 0DTE wings.

Why
---
Module 10 found 0DTE wings RICH against "ex-ante constant diffusion + Poisson
5-minute jumps", but that model omits variance uncertainty and multi-bar
moves, both of which fatten 1-20 h tails. Module 10b drops the model: each
quote is priced from the realized distribution of standardized moves to
settlement.

Standardized move to settlement
-------------------------------
For every asset and every 5-minute origin t with 60 <= minutes to the next
08:00 expiry <= 1200:

    u_t = ln(S_T / I_t) / sqrt(D_t)

I_t is the index at t. S_T is the approximate settlement (mean index of the bars
ending 07:35-08:00; Module 10). D_t is EXACTLY Module 10's ex-ante diffusive
variance: backward 4h bipower x Module 08 clock, with settlement-TWAP weights.
Because quotes and outcomes are standardized by the same D, any level bias
in D (e.g. 5m-noise inflation) cancels; only the SHAPE of the physical
distribution, relative to D, matters.

Prices (undiscounted USD; same rows as Module 10's primary sample)
-----------------------------------------------------------------
    P_emp = mean_i payoff(F exp(sqrt(D) u_i) / mbar, K),   mbar = mean_i exp(sqrt(D) u_i)

u_i is the LEAVE-ONE-SESSION-OUT pool of u, exact in the outer 1% on each
side (every raw outcome kept, so deep-OTM payoffs see each tail event) and
compressed into 500 equal-count bin means in the centre. Pooling covers all assets and TTEs, and the session being priced never
contributes. Dividing by mbar enforces the martingale condition per quote:
sample drift is removed, shape is kept.

Secondary price (reported): variance mixture,

    P_mix = mean_k B76(F, K, k D),   k = RV_to_settlement / D,

where RV = sum_j w_j^2 r_j^2 over the realized bars to expiry, from the same
LOSO pool. RV includes jumps, so no separate jump term is added (adding
Module 09 jumps would double count; this corrects VD 020 section 6.1).

Pre-registered (fixed before any real-data Module 10b output)
------------------------------------------------------------
Sample: Module 10's primary rows (10/10_priced_quotes.csv.gz; non-floor,
z_phys in [2, 5), no Module 09 P2 assets), unchanged.

Statistic: M_emp = median over sessions of the session median of ln(mid / P_emp).

Uncertainty: JOINT session bootstrap, B = 1000. Each replicate resamples
sessions with replacement and rebuilds BOTH the quote set and the LOSO
outcome pool from the drawn sessions, so the uncertainty of the physical tail
itself is included.

    RICH        iff the 2.5% bootstrap quantile of M_emp > 0
    CHEAP       iff the 97.5% quantile < 0
    CONSISTENT  otherwise
    Rows with P_emp = 0 are dropped, in the point estimate and in every
    replicate. P_emp = 0 means the strike lies beyond every LOSO outcome. The
    market's positive price is then compared with an unobserved event, and
    calling it rich would be a sample-size artifact. Dropping them is
    conservative against RICH.
    INSUFFICIENT_TAIL replaces the verdict if more than 25% of primary rows
    are dropped.

Reported, no decision:
- TTE-conditioned pool (same TTE bucket as the quote);
- variance mixture P_mix;
- per asset;
- all z x TTE buckets;
- tail frequencies P(|u| > x) vs normal;
- distribution of log k;
- the number of distinct (session, asset) outcomes beyond 2, 3 and 4.

Power disclosure: outcomes overlap heavily within a session (one S_T per
session and asset), so the tail of u rests on roughly sessions x assets
independent outcomes, with a common component. CONSISTENT is the likely
outcome on ~20 sessions and is an honest result, not a failure.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from data_engineering.modeling import volatility_vd_b_core as vb
from data_engineering.modeling import volatility_jump_hawkes as jh
from data_engineering.modeling import volatility_jump_option_wings as ow


VERSION = "0.1.1"   # 0.1.1: origin_outcomes also returns v (backward bipower); results unchanged
Q_NODES = 500
Q_NODES_MIX = 200
N_BOOT = 1000
MIN_BAR_COVERAGE = 0.9
TAIL_X = (1.0, 2.0, 3.0, 4.0, 5.0)
MAX_ZERO_PRICE_SHARE = 0.25
TAIL_RAW_SHARE = 0.01


# --------------------------------------------------------------------------
# Outcomes
# --------------------------------------------------------------------------

def settlement_matrix(I: np.ndarray, panel: jh.Panel, sessions_of_bar: np.ndarray) -> dict:
    """{(session, asset_idx): mean index of bars ending 07:35..08:00 of the session date}."""
    t = pd.to_datetime(panel.time_ns, utc=True)
    hm = t.hour * 60 + t.minute
    in_win = np.asarray((hm >= 7 * 60 + 35) & (hm <= 8 * 60))
    out = {}
    dates = np.asarray(t.strftime("%Y-%m-%d"))
    for s in np.unique(sessions_of_bar):
        rows = in_win & (dates == s)
        if not rows.any():
            continue
        for a in range(panel.N):
            v = I[rows, a]
            v = v[np.isfinite(v)]
            if len(v) >= 5:
                out[(s, a)] = float(v.mean())
    return out


def origin_outcomes(panel: jh.Panel, S: np.ndarray, BVb: np.ndarray, I: np.ndarray,
                    assets_used: Sequence[str]) -> pd.DataFrame:
    """One row per (asset, 5m origin) in the 60-1200 minute window: D, u, k."""
    sess = ow.session_of_bar(panel)
    settle = settlement_matrix(I, panel, sess)
    t = pd.to_datetime(panel.time_ns, utc=True)
    exp_t = pd.to_datetime(sess, utc=True) + pd.Timedelta(hours=8)
    mins = np.asarray((exp_t - t).total_seconds(), float) / 60.0
    window = (mins >= ow.MIN_TTE_MIN) & (mins <= ow.MAX_TTE_MIN)
    rows = []
    for a_name in assets_used:
        a = panel.assets.index(a_name)
        ti = np.flatnonzero(window & np.isfinite(BVb[:, a]) & np.isfinite(I[:, a]))
        if not len(ti):
            continue
        n_bars = np.round(mins[ti] / 5.0).astype(int)
        d_units, _, _ = ow.remaining_sums(panel.slot[ti], n_bars, S[:, a])
        D = BVb[ti, a] * d_units
        r = panel.R[:, a]
        for i, t0 in enumerate(ti):
            key = (sess[t0], a)
            if key not in settle or not (D[i] > 0):
                continue
            n = n_bars[i]
            seg = r[t0 + 1:t0 + 1 + n]
            if len(seg) < n:
                continue
            w = ow.settlement_weight(5.0 * (n - np.arange(1, n + 1)))
            fin = np.isfinite(seg)
            if fin.mean() < MIN_BAR_COVERAGE:
                continue
            w2 = w * w
            rv = np.sum(w2[fin] * seg[fin] ** 2) * (w2.sum() / w2[fin].sum())
            rows.append((a_name, sess[t0], int(t0), float(mins[t0]), float(D[i]),
                         float(np.log(settle[key] / I[t0, a]) / np.sqrt(D[i])), float(rv / D[i]),
                         float(BVb[t0, a])))
    out = pd.DataFrame(rows, columns=["underlying", "session", "bar_index", "minutes_to_expiry", "D", "u", "k", "v"])
    out["tte_bucket"] = pd.cut(out["minutes_to_expiry"], ow.TTE_EDGES_MIN, right=True).astype(str)
    return out


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------

def bin_means(x: np.ndarray, q: int) -> np.ndarray:
    """Equal-count bin means (preserve each bin's mean). Used for the CENTRE only."""
    x = np.sort(np.asarray(x, float)[np.isfinite(x)])
    if len(x) <= q:
        return x
    edges = np.linspace(0, len(x), q + 1).round().astype(int)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    return (cs[edges[1:]] - cs[edges[:-1]]) / np.diff(edges)


def compress(x: np.ndarray, q: int, tail: float = TAIL_RAW_SHARE) -> tuple[np.ndarray, np.ndarray]:
    """Weighted nodes reproducing the empirical distribution: every raw observation in
    the outer `tail` share on each side is kept (deep-OTM payoffs depend on them
    individually); the centre is compressed into q equal-count bin means."""
    x = np.sort(np.asarray(x, float)[np.isfinite(x)])
    n = len(x)
    if n == 0:
        return np.array([]), np.array([])
    k = int(np.floor(tail * n))
    centre = x[k:n - k]
    if len(centre) <= q:
        return x, np.full(n, 1.0 / n)
    edges = np.linspace(0, len(centre), q + 1).round().astype(int)
    cs = np.concatenate([[0.0], np.cumsum(centre)])
    cm = (cs[edges[1:]] - cs[edges[:-1]]) / np.diff(edges)
    nodes = np.concatenate([x[:k], cm, x[n - k:]])
    w = np.concatenate([np.full(k, 1.0 / n), np.diff(edges) / n, np.full(k, 1.0 / n)])
    return nodes, w


def _weights(nodes, weights):
    nodes = np.asarray(nodes, float)
    w = np.full(len(nodes), 1.0 / len(nodes)) if weights is None else np.asarray(weights, float)
    return nodes, w / w.sum()


def empirical_price(F, K, D, is_call, u_nodes, weights=None) -> np.ndarray:
    F, K, D = (np.asarray(v, float) for v in (F, K, D))
    is_call = np.asarray(is_call, bool)
    u, w = _weights(u_nodes, weights)
    e = np.exp(np.sqrt(D)[:, None] * u[None, :])
    ST = F[:, None] * e / (e @ w)[:, None]
    pay = np.where(is_call[:, None], np.maximum(ST - K[:, None], 0.0), np.maximum(K[:, None] - ST, 0.0))
    return pay @ w


def mixture_price(F, K, T, D, is_call, k_nodes, weights=None) -> np.ndarray:
    F, K, T, D = (np.asarray(v, float) for v in (F, K, T, D))
    is_call = np.asarray(is_call, bool)
    kn, w = _weights(k_nodes, weights)
    sig = np.sqrt(np.maximum(D[:, None] * kn[None, :], 1e-18) / T[:, None])
    return ow.b76(F[:, None], K[:, None], T[:, None], sig, is_call[:, None]) @ w


def _price_rows(rows: pd.DataFrame, nodes_w: tuple, kind: str) -> np.ndarray:
    F, K, D = (rows[c].to_numpy(float) for c in ("option_forward_price", "strike", "D"))
    nodes, w = nodes_w
    if kind == "emp":
        return empirical_price(F, K, D, rows["is_call"].to_numpy(), nodes, w)
    return mixture_price(F, K, rows["T"].to_numpy(float), D, rows["is_call"].to_numpy(), nodes, w)


def price_loso(q: pd.DataFrame, outcomes: pd.DataFrame, *, kind: str = "emp", by_tte: bool = False) -> np.ndarray:
    """ln(mid / P) for every row, using the LOSO outcome pool (optionally the same TTE bucket)."""
    col, Q = ("u", Q_NODES) if kind == "emp" else ("k", Q_NODES_MIX)
    out = np.full(len(q), np.nan)
    keys = ["session", "tte_bucket"] if by_tte else ["session"]
    for key, g in q.groupby(keys):
        s = key[0] if by_tte else (key[0] if isinstance(key, tuple) else key)
        pool = outcomes[outcomes["session"] != s]
        if by_tte:
            pool = pool[pool["tte_bucket"] == key[1]]
        if len(pool) < 50:
            continue
        nodes = compress(pool[col].to_numpy(), Q)
        idx = q.index.get_indexer(g.index)
        for s0 in range(0, len(g), 5000):
            gg = g.iloc[s0:s0 + 5000]
            with np.errstate(divide="ignore", invalid="ignore"):
                out[idx[s0:s0 + 5000]] = np.log(gg["mid_usd"].to_numpy(float) / _price_rows(gg, nodes, kind))
    return out


def joint_bootstrap(q: pd.DataFrame, outcomes: pd.DataFrame, *, kind: str = "emp", n_boot: int = N_BOOT,
                    seed: int = 1013) -> dict:
    """Resample sessions; rebuild quotes AND the LOSO outcome pool from the draw; return CI of M."""
    sessions = np.array(sorted(set(q["session"]) & set(outcomes["session"])))
    col, Q = ("u", Q_NODES) if kind == "emp" else ("k", Q_NODES_MIX)
    q_by = {s: g for s, g in q[q["session"].isin(sessions)].groupby("session")}
    o_by = {s: g[col].to_numpy(float) for s, g in outcomes[outcomes["session"].isin(sessions)].groupby("session")}
    rng = np.random.default_rng(int(seed))
    Ms = np.empty(int(n_boot))
    for b in range(int(n_boot)):
        draw = rng.choice(sessions, size=len(sessions), replace=True)
        uniq, mult = np.unique(draw, return_counts=True)
        med = {}
        for s in uniq:
            parts = [np.tile(o_by[x], m_) for x, m_ in zip(uniq, mult) if x != s and x in o_by]
            pool = np.concatenate(parts) if parts else np.array([])
            if len(pool) < 50:
                continue
            g = q_by[s]
            with np.errstate(divide="ignore", invalid="ignore"):
                lr = np.log(g["mid_usd"].to_numpy(float) / _price_rows(g, compress(pool, Q), kind))
            lr = lr[np.isfinite(lr)]
            if len(lr):
                med[s] = np.quantile(lr, 0.5, method="inverted_cdf")
        vals = np.concatenate([[med[s]] * m_ for s, m_ in zip(uniq, mult) if s in med])
        Ms[b] = np.quantile(vals, 0.5, method="inverted_cdf") if len(vals) else np.nan
    Ms = Ms[~np.isnan(Ms)]
    lo = float(np.quantile(Ms, 0.025, method="inverted_cdf"))        # no interpolation: +inf medians allowed
    hi = float(np.quantile(Ms, 0.975, method="inverted_cdf"))
    return {"ci_low": lo, "ci_high": hi, "n_boot": int(len(Ms)),
            "verdict": "RICH" if lo > 0 else ("CHEAP" if hi < 0 else "CONSISTENT")}


def primary(q: pd.DataFrame, outcomes: pd.DataFrame, *, kind: str = "emp", n_boot: int = N_BOOT) -> dict:
    x = q.copy()
    x["lr"] = price_loso(x, outcomes, kind=kind)
    zero_share = float(np.mean(np.isposinf(x["lr"])))
    finite_or_inf = x[np.isfinite(x["lr"])].copy()
    sm = finite_or_inf.groupby("session")["lr"].apply(lambda v: np.quantile(v.to_numpy(), 0.5, method="inverted_cdf"))
    jb = joint_bootstrap(x, outcomes, kind=kind, n_boot=n_boot)
    verdict = "INSUFFICIENT_TAIL" if zero_share > MAX_ZERO_PRICE_SHARE else jb["verdict"]
    return {"n_rows": int(len(finite_or_inf)), "n_sessions": int(len(sm)), "M": float(np.quantile(sm.to_numpy(), 0.5, method="inverted_cdf")),
            "ci_low": jb["ci_low"], "ci_high": jb["ci_high"], "verdict": verdict,
            "bootstrap_verdict": jb["verdict"], "share_model_price_zero": zero_share, "n_boot": jb["n_boot"]}


# --------------------------------------------------------------------------
# Descriptives
# --------------------------------------------------------------------------

def tail_table(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in [("ALL", outcomes)] + list(outcomes.groupby("underlying")) + \
            [(f"TTE {k}", v) for k, v in outcomes.groupby("tte_bucket")]:
        u = g["u"].to_numpy(float)
        r = {"group": name, "n_origins": int(len(u)), "n_session_assets": int(g.groupby(["session", "underlying"]).ngroups),
             "u_mean": float(u.mean()), "u_sd": float(u.std()), "log_k_mean": float(np.log(g["k"]).mean()),
             "log_k_sd": float(np.log(g["k"]).std())}
        for x in TAIL_X:
            r[f"P_abs_u_gt_{x:g}"] = float(np.mean(np.abs(u) > x))
            r[f"normal_{x:g}"] = float(ow._erfc(np.array([x / np.sqrt(2.0)]))[0])
            r[f"distinct_session_assets_gt_{x:g}"] = int(g.loc[np.abs(u) > x].groupby(["session", "underlying"]).ngroups)
        rows.append(r)
    return pd.DataFrame(rows)


def bucket_table(q: pd.DataFrame, col: str = "log_mid_over_emp", n_boot: int = 1000) -> pd.DataFrame:
    rows = []
    for (zb, tb), g in q[~q["floor_bound"]].groupby(["z_bucket", "tte_bucket"]):
        rows.append({"z_bucket": zb, "tte_bucket": tb, **ow.session_median_stat(g, col, n_boot=n_boot)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Self test
# --------------------------------------------------------------------------

def _fake_quotes(rng, n_sessions, per_session, true_sampler, noise=0.05):
    """Quotes whose mids are exact prices under the world `true_sampler`."""
    rows = []
    u_true, w_true = compress(true_sampler(200_000), 2000)
    for s in range(n_sessions):
        F = np.full(per_session, 100.0)
        D = np.full(per_session, 0.02 ** 2)
        z = rng.uniform(2.0, 3.5, per_session)
        call = rng.random(per_session) < 0.5
        K = np.where(call, F * np.exp(z * 0.02), F * np.exp(-z * 0.02))
        P = empirical_price(F, K, D, call, u_true, w_true)
        rows.append(pd.DataFrame({"session": f"s{s:02d}", "option_forward_price": F, "strike": K, "D": D,
                                  "T": 8 / 24 / 365, "is_call": call,
                                  "mid_usd": P * np.exp(rng.normal(0, noise, per_session)),
                                  "tte_bucket": "(240.0, 480.0]", "floor_bound": False}))
    return pd.concat(rows, ignore_index=True)


def _fake_outcomes(rng, n_sessions, per_session, sampler):
    return pd.DataFrame({"session": np.repeat([f"s{s:02d}" for s in range(n_sessions)], per_session),
                         "u": sampler(n_sessions * per_session), "k": np.exp(rng.normal(0, 0.5, n_sessions * per_session)),
                         "tte_bucket": "(240.0, 480.0]", "underlying": "X"})


def selftest() -> dict:
    out = {}
    rng = np.random.default_rng(7)
    # 1. Gaussian u reproduces Black-76.
    nodes = bin_means(rng.standard_normal(400_000), Q_NODES)
    F, D, T = np.full(2, 100.0), np.full(2, 0.02 ** 2), np.full(2, 8 / 24 / 365)
    K = np.array([100 * np.exp(2 * 0.02), 100 * np.exp(-2 * 0.02)])
    pe = empirical_price(F, K, D, np.array([True, False]), nodes)
    pb = ow.diffusion_price(F, K, T, D, np.array([True, False]))
    out["gaussian_matches_b76_z2"] = bool(np.all(np.abs(pe / pb - 1) < 0.03))
    # 2. k == 1 mixture equals Black-76.
    out["mixture_k1_is_b76"] = bool(np.allclose(mixture_price(F, K, T, D, np.array([True, False]), [1.0]), pb, rtol=1e-10))
    # 3. bin means preserve the mean.
    x = rng.standard_t(3, 10_000)
    out["bin_means_preserve_mean"] = bool(abs(bin_means(x, 100).mean() - x.mean()) < 1e-9)
    # 4. Fat-tailed world: correct benchmark -> CONSISTENT; Gaussian benchmark -> RICH.
    t3 = lambda n: rng.standard_t(3, n) / np.sqrt(3.0)
    gauss = lambda n: rng.standard_normal(n)
    q = _fake_quotes(rng, 16, 60, t3)
    r_ok = primary(q, _fake_outcomes(rng, 16, 3000, t3), n_boot=150)
    r_bad = primary(q, _fake_outcomes(rng, 16, 3000, gauss), n_boot=150)
    out["fat_world_correct_benchmark_consistent"] = r_ok["verdict"] == "CONSISTENT"
    out["fat_world_gaussian_benchmark_rich"] = r_bad["verdict"] == "RICH"
    # 5. LOSO: a session's own outcomes never price it.
    o = _fake_outcomes(rng, 4, 500, gauss)
    o.loc[o.session == "s00", "u"] = 50.0
    qq = _fake_quotes(rng, 4, 10, gauss)
    lr = price_loso(qq, o)
    out["loso_excludes_own_session"] = bool(np.nanmedian(lr[qq.session.eq("s00")]) > -1.0
                                            and np.nanmedian(lr[~qq.session.eq("s00")]) < -1.0)
    # 6. Outcome construction: constant-vol Gaussian world gives sd(u) ~ 1.
    n_days, N = 22, 2
    times = pd.date_range("2026-01-01 08:05", periods=n_days * 288, freq="5min", tz="UTC")
    fr = []
    for a in range(N):
        r = rng.normal(0, 0.002, len(times))
        fr.append(pd.DataFrame({"state_time": times, "underlying": f"A{a}", "return_5m": r,
                                "futures_index_price": 100 * np.exp(np.cumsum(r))}))
    g = vb.build_grid(pd.concat(fr, ignore_index=True))
    panel = jh.build_panel(g)
    S = np.ones((jh.BARS_PER_DAY, N))
    from data_engineering.modeling import volatility_jump_compound as jc
    I = jc.index_panel(g, panel)
    oc = origin_outcomes(panel, S, ow.backward_bipower(panel, S), I, panel.assets)
    out["outcome_u_sd_near_1"] = bool(0.8 < oc["u"].std() < 1.25)
    out["outcome_k_median_near_1"] = bool(0.8 < oc["k"].median() < 1.25)
    out["n_outcomes"] = int(len(oc))
    return out


if __name__ == "__main__":
    print(selftest())
