"""
Signed Jumps 03: do "bubbles" (aggressor activity) predict WHEN a jump happens, beyond what returns show?

Protocol: discussions/Signed_Jumps/005_claude_sj03_preregistration.md (fixed before any real-data output).

Heat analogy (Ivan, 2026-10-16): jumps are the boil; small aggressor events are the bubbles; a latent "heat"
drives both. SJ 01's strongest timing predictor, TV10, measures heat through returns only. This module adds
heat readings from order flow, all from the same Binance spot 1m klines and all strictly through bar t-1:

    AN_w  = log(mean trades over t-w..t-1 + 1) - log(mean trades over t-W..t-1 + 1)   trade-count acceleration
    AV_w  = same with quote volume                                                    volume acceleration
            (w, W) in {(10, 240), (60, 1440)}
    IMB10 = |sum(2 taker_buy_quote - quote)| / sum(quote) over t-10..t-1               unsigned aggressor imbalance
    All standardised on DEV.

Models (same jumps, same rows, SJ 01 kernel half-life fixed at its DEV estimate):
    (b)   SJ 01 engine-modulated Poisson
    (bf)  (b) + flow
    (e)   SJ 01 best model (signed Hawkes + TSK)
    (f)   (e) + flow
Gates on TEST (day-block bootstrap):
    F1  (f) vs (e): logLik/row delta > 0, q2.5 > 0          flow adds to the full SJ 01 model
    F2  15-minute occurrence log-loss (f) vs (e): delta > 0, q2.5 > 0
    F3  (f) vs (ev): same rule. (ev) = (e) + v_hat, where v_hat is the flow-implied forecast of the NEXT
        10 minutes' continuous-variance surprise (OLS on DEV of log mean Z^2, |Z| < c, over t..t+9 on the (b)
        covariates + flow; v_hat = flow part). Jumps are defined relative to the volatility forecast, so flow
        that merely nowcasts volatility better predicts more threshold crossings (found on synthetic data).
        F3 PASS = flow carries jump-specific "heat" beyond its volatility information.
Reported: (bf) vs (b); hazard-decile lift and top-1% realised jump rate; event-time profile of flow before jumps.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace

import numpy as np
import pandas as pd

from . import signed_jumps as sj

VERSION = "SJ03-0.1.0"
FLOW_COLS = ("trades", "quote_volume", "taker_buy_quote_volume")


@dataclass(frozen=True)
class FlowParams:
    accel_pairs: tuple = ((10, 240), (60, 1440))
    imb_window: int = 10
    min_cover: float = 0.75
    sj01_half_life: float = 1.0
    profile_window: tuple = (-30, 10)
    n_boot: int = 2000
    status: str = "PREREGISTERED_SJ005"


def registration(fp: FlowParams | None = None) -> dict:
    fp = fp or FlowParams()
    d = asdict(fp)
    d["version"] = VERSION
    d["gates"] = {"F1": "(f) vs (e) TEST logLik/row delta > 0 and day-bootstrap q2.5 > 0, >= 200 TEST jumps/sign",
                  "F2": "15-min signed occurrence log-loss (f) vs (e): delta > 0 and q2.5 > 0",
                  "F3": "(f) vs (e + flow-implied forward variance v_hat): same rule as F1"}
    return d


# ----------------------------------------------------------------------------- data

def load_spot_flow(path, grid: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["open_time", *FLOW_COLS])
    t = pd.to_datetime(df["open_time"], utc=True) + pd.Timedelta(minutes=1)
    out = df[list(FLOW_COLS)].astype(float).set_index(t)
    out = out[~out.index.duplicated()].sort_index()
    return out.reindex(grid)


def _lagged_mean(x: np.ndarray, w: int, cover: float) -> np.ndarray:
    m = pd.Series(x).rolling(w, min_periods=max(1, int(cover * w))).mean().to_numpy()
    return sj._lag(m)


def flow_covariates(flow: pd.DataFrame, fp: FlowParams) -> tuple[np.ndarray, list[str]]:
    """Raw (unstandardised) covariates on the asset's grid, strictly through t-1."""
    tr = flow["trades"].to_numpy()
    qv = flow["quote_volume"].to_numpy()
    tb = flow["taker_buy_quote_volume"].to_numpy()
    cols, names = [], []
    for w, W in fp.accel_pairs:
        for x, nm in ((tr, "AN"), (qv, "AV")):
            a = np.log(_lagged_mean(x, w, fp.min_cover) + 1) - np.log(_lagged_mean(x, W, fp.min_cover) + 1)
            cols.append(a)
            names.append(f"{nm}{w}")
    w = fp.imb_window
    num = pd.Series(2 * tb - qv).rolling(w, min_periods=int(fp.min_cover * w)).sum().to_numpy()
    den = pd.Series(qv).rolling(w, min_periods=int(fp.min_cover * w)).sum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        imb = sj._lag(np.abs(num) / den)
    cols.append(imb)
    names.append(f"IMB{w}")
    return np.column_stack(cols), names


def attach_flow(assets, pan: sj.Panel, flows: dict, fp: FlowParams) -> sj.Panel:
    """Standardise flow covariates on DEV, attach as pan.extra, and drop rows where any is missing.
    Returns a NEW panel (rows subset); excitation still comes from each asset's full grid."""
    raw, names = [], None
    for a, idx in zip(assets, pan._idx):
        F, names = flow_covariates(flows[a.name], fp)
        raw.append(F[idx])
    R = np.concatenate(raw)
    mu = np.nanmean(R[pan.dev], axis=0)
    sd = np.nanstd(R[pan.dev], axis=0)
    Z = (R - mu) / sd
    keep = np.isfinite(Z).all(axis=1)
    return subset_panel(pan, keep, Z.astype(np.float32), names, dict(mean=mu.tolist(), sd=sd.tolist()))


def subset_panel(pan: sj.Panel, keep: np.ndarray, extra: np.ndarray, names, scaler) -> sj.Panel:
    f = lambda x: x[keep]
    new = sj.Panel(f(pan.asset_idx), f(pan.day), f(pan.dev), f(pan.time), f(pan.clock), f(pan.L), f(pan.F),
                   f(pan.TSK), f(pan.TV), {s: f(v) for s, v in pan.J.items()}, f(pan.Z), f(pan.r),
                   pan.n_assets, pan.names)
    starts = np.cumsum([0] + [len(x) for x in pan._idx])
    new._idx = [pan._idx[i][keep[starts[i]:starts[i + 1]]] for i in range(len(pan._idx))]
    new.extra = extra[keep]
    new.extra_names = tuple(names)
    new.extra_scaler = scaler
    return new


# ----------------------------------------------------------------------------- fitting and gates

def fit_models(assets, pan: sj.Panel, p: sj.Params, fp: FlowParams, models=("b", "bf", "e", "f")) -> tuple[dict, dict]:
    hl = fp.sj01_half_life
    E = sj.panel_excitation(assets, pan, hl)
    Ecache = {hl: E}
    fits = {}
    dev = pan.dev
    for m in models:
        X = sj.design(pan, m)
        for s in sj.SIGNS:
            Ex = sj.excitation_design(E, m, s)
            fits[(m, s)] = sj.fit_sign(X[dev], None if Ex is None else Ex[dev], pan.J[s][dev], m, s,
                                       None if Ex is None else hl, pan.n_assets)
    return fits, Ecache


def compare(fits, pan, Ecache, p: sj.Params, seed: int = 31) -> tuple[pd.DataFrame, dict]:
    test = ~pan.dev
    ll = {m: sj.row_loglik(fits, pan, Ecache, m)[test] for m in ("b", "bf", "e", "f") if (m, "up") in fits}
    day = pan.day[test]
    nj = {s: int(pan.J[s][test].sum()) for s in sj.SIGNS}
    rows = []
    for name, m1, m0 in (("F1_f_vs_e", "f", "e"), ("bf_vs_b", "bf", "b"), ("e_vs_b", "e", "b"), ("f_vs_b", "f", "b")):
        if m1 in ll and m0 in ll:
            bt = sj.day_boot_delta(ll[m1] - ll[m0], day, p.n_boot, seed)
            bt.update(test=name, status=sj.gate(bt, nj, p))
            rows.append(bt)
    return pd.DataFrame(rows), nj


def horizon_compare(fits, assets, pan, p: sj.Params, seed: int = 41) -> pd.DataFrame:
    """F2: (f) vs (e) signed occurrence log-loss at each horizon; plus (bf) vs (b) (exact, no MC)."""
    rows_o = np.flatnonzero(~pan.dev)[:: p.origin_step]
    pr = {m: sj.horizon_forecast(fits, m, assets, pan, p, rows_o, seed=seed + i)
          for i, m in enumerate(("b", "bf", "e", "f"))}
    y = pr["b"][1]
    nj = {s: int(pan.J[s][~pan.dev].sum()) for s in sj.SIGNS}
    out = []
    for h in p.horizons:
        ok = np.isfinite(y[h]["up"]) & np.isfinite(y[h]["down"])
        yy = {s: y[h][s][ok] for s in sj.SIGNS}
        L = {m: sj.occurrence_logloss({s: pr[m][0][h][s][ok] for s in sj.SIGNS}, yy) for m in pr}
        for name, m1, m0 in (("F2_f_vs_e", "f", "e"), ("bf_vs_b", "bf", "b")):
            bt = sj.day_boot_delta(L[m0] - L[m1], pan.day[rows_o][ok], p.n_boot, seed + h)
            bt.update(test=name, horizon_min=h, status=sj.gate(bt, nj, p))
            out.append(bt)
    return pd.DataFrame(out), pr, rows_o


def lift_table(pr: dict, rows_o, pan, h: int) -> pd.DataFrame:
    """Realised P(any jump in next h min) by decile of predicted P(any jump), for (e) and (f); plus top 1%."""
    y = pr["b"][1][h]
    ok = np.isfinite(y["up"]) & np.isfinite(y["down"])
    anyj = ((y["up"] > 0) | (y["down"] > 0))[ok]
    out = []
    for m in ("b", "e", "f"):
        pu, pdn = pr[m][0][h]["up"][ok], pr[m][0][h]["down"][ok]
        pany = 1 - (1 - pu) * (1 - pdn)
        dec = pd.qcut(pany, 10, labels=False, duplicates="drop")
        g = pd.DataFrame({"dec": dec, "pred": pany, "real": anyj}).groupby("dec").agg(
            n=("pred", "size"), pred=("pred", "mean"), realised=("real", "mean")).reset_index()
        g["model"] = m
        top = pany >= np.quantile(pany, 0.99)
        g = pd.concat([g, pd.DataFrame([{"dec": "top1%", "n": int(top.sum()), "pred": float(pany[top].mean()),
                                         "realised": float(anyj[top].mean()), "model": m}])], ignore_index=True)
        out.append(g)
    t = pd.concat(out, ignore_index=True)
    t["base_rate"] = float(anyj.mean())
    t["lift"] = t.realised / t.base_rate
    t["horizon_min"] = h
    return t


def event_profile(assets, pan, flows: dict, fp: FlowParams, seed: int = 5) -> pd.DataFrame:
    """Mean log trade-count and log quote-volume deviation (vs the causal 1440-bar mean) at minutes k relative
    to cluster-starting jumps, TEST; placebo = random non-jump bars matched by asset and UTC hour."""
    lo, hi = fp.profile_window
    rng = np.random.default_rng(seed)
    rows = []
    for i_a, a in enumerate(assets):
        fl = flows[a.name]
        dev_tr = {}
        for col in ("trades", "quote_volume"):
            x = fl[col].to_numpy()
            dev_tr[col] = np.log(x + 1) - np.log(_lagged_mean(x, 1440, fp.min_cover) + 1)
        J = (a.cols["Jup"] + a.cols["Jdn"]) > 0
        cs = np.concatenate([[0], np.cumsum(J)])
        test = ~a.dev
        g = np.flatnonzero(J & test)
        g = g[(cs[g] - cs[np.maximum(g - 5, 0)]) == 0]
        g = g[(g + lo >= 0) & (g + hi < len(J))]
        hours = a.time.hour.to_numpy()
        cand = np.flatnonzero(~J & test)
        cand = cand[(cand + lo >= 0) & (cand + hi < len(J))]
        plac = []
        for hr in np.unique(hours[g]):
            need = int((hours[g] == hr).sum())
            pool = cand[hours[cand] == hr]
            if len(pool):
                plac.append(rng.choice(pool, size=need, replace=len(pool) < need))
        plac = np.concatenate(plac) if plac else np.array([], int)
        for k in range(lo, hi + 1):
            for col, d in dev_tr.items():
                rows.append({"asset": a.name, "k": k, "series": col, "jump_mean": float(np.nanmean(d[g + k])),
                             "placebo_mean": float(np.nanmean(d[plac + k])), "n_jumps": int(len(g))})
    t = pd.DataFrame(rows)
    t["excess"] = t.jump_mean - t.placebo_mean
    return t


def forward_cont_var(assets, pan: sj.Panel, p: sj.Params, w: int = 10) -> np.ndarray:
    """log mean Z^2 over t..t+w-1 with |Z| < c (>= 0.8 w valid): the forward continuous-variance surprise."""
    out = []
    for a, idx in zip(assets, pan._idx):
        Z = a.cols["Z"]
        keep = np.isfinite(Z) & (np.abs(Z) < p.c)
        x = np.where(keep, Z ** 2, 0.0)[::-1]
        k = keep.astype(float)[::-1]
        S = pd.Series(x).rolling(w, min_periods=1).sum().to_numpy()[::-1]
        K = pd.Series(k).rolling(w, min_periods=1).sum().to_numpy()[::-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            y = np.log(np.maximum(S / K, 1e-3))
        y[K < 0.8 * w] = np.nan
        out.append(y[idx])
    return np.concatenate(out)


def vhat_panel(assets, pf: sj.Panel, p: sj.Params) -> tuple[sj.Panel, dict]:
    """Panel identical to pf but with extra = v_hat (flow-implied forward variance surprise), fitted on DEV."""
    y = forward_cont_var(assets, pf, p)
    Xb = sj.design(pf, "b").astype(float)
    X = np.column_stack([Xb, pf.extra])
    ok = pf.dev & np.isfinite(y)
    # DEV targets must not reach into TEST: keep rows whose 10-minute forward window ends before the boundary
    tt = pd.DatetimeIndex(pf.time)
    test_start = tt[~pf.dev].min() if (~pf.dev).any() else tt.max()
    ok &= np.asarray(tt + pd.Timedelta(minutes=10) <= test_start)
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    nb = Xb.shape[1]
    v = (pf.extra.astype(float) @ beta[nb:])
    r2_full = 1 - np.var(y[ok] - X[ok] @ beta) / np.var(y[ok])
    bb, *_ = np.linalg.lstsq(Xb[ok], y[ok], rcond=None)
    r2_base = 1 - np.var(y[ok] - Xb[ok] @ bb) / np.var(y[ok])
    vp = subset_panel(pf, np.ones(len(pf.r), bool), ((v - v[pf.dev].mean()) / v[pf.dev].std())[:, None].astype(np.float32),
                      ("VHAT",), {})
    info = {"flow_beta_on_fwd_var": dict(zip(pf.extra_names, np.round(beta[nb:], 4).tolist())),
            "fwd_var_R2_base_dev": float(r2_base), "fwd_var_R2_with_flow_dev": float(r2_full)}
    return vp, info


def compare_f3(fits_f, Ef, vp: sj.Panel, assets, p: sj.Params, pf: sj.Panel, seed: int = 51) -> tuple[dict, dict]:
    hl = fits_f[("f", "up")].half_life
    fits_v, _ = fit_models(assets, vp, p, FlowParams(sj01_half_life=hl), models=("f",))
    Ev = {hl: sj.panel_excitation(assets, vp, hl)}
    test = ~pf.dev
    ll_f = sj.row_loglik(fits_f, pf, Ef, "f")[test]
    ll_v = sj.row_loglik(fits_v, vp, Ev, "f")[test]
    nj = {s: int(pf.J[s][test].sum()) for s in sj.SIGNS}
    bt = sj.day_boot_delta(ll_f - ll_v, pf.day[test], p.n_boot, seed)
    bt.update(test="F3_f_vs_ev", status=sj.gate(bt, nj, p))
    return bt, fits_v


def decide(cmp: pd.DataFrame, hz: pd.DataFrame, f3: dict, p: sj.Params) -> dict:
    st = dict(zip(cmp.test, cmp.status))
    f2 = hz[(hz.test == "F2_f_vs_e") & (hz.horizon_min == p.primary_horizon)].iloc[0]
    return {"version": VERSION, "F1_flow_adds_to_SJ01": st.get("F1_f_vs_e"), "F2_15min_forecast": f2["status"],
            "F3_jump_specific_heat": f3["status"], "flow_vs_engine_only_bf_vs_b": st.get("bf_vs_b")}
