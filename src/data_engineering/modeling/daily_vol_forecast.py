"""VD20: next-24h realised-variance forecasts - volatility engine vs daily HAR vs naive, and vs implied.

Protocol: discussions/Volatility_Dynamics/040_claude_vd20_next_24h_volatility_preregistration.md.

One origin per UTC day at ORIGIN_HOUR (bar ending at the origin is the last past bar). Target RV = sum of squared
5m returns over (t, t+h], strict complete window. ENGINE forecasts come from VD19's refit of VD18 HAR_4 and are
passed in as a column; this module fits the daily competitors and scores everything on the same origins.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

VERSION = "VD20-0.1.0"
ASSETS = ("AVAX", "BTC", "ETH", "SOL", "XRP")
OPTION_ASSETS = ("BTC", "ETH", "SOL", "XRP")
ORIGINS = {8: 1440, 12: 1200}           # UTC hour -> minutes to the next 08:00 expiry
PRIMARY_HOUR = 8
BARS_DAY = 288
DAILY = {"HAR_D": ["l1", "l5", "l22"],
         "HAR_DX": ["l1", "l5", "l22", "l30", "l60", "l240", "neg"]}
MODELS = ("NAIVE", "ENGINE", "HAR_D", "HAR_DX")


@dataclass(frozen=True)
class Params:
    version: str = VERSION
    forecast_end: str = "2026-10-15T00:00:00Z"
    min_frac_daily: float = 0.98       # 1d/5d/22d windows: >= 98% bars present, rescaled
    vol_quantile: float = 0.80
    tail_k: tuple = (1.0, 1.5, 2.0, 2.5)
    trade_ratio: float = 1.25          # VD19 economics: sigma_model >= 1.25 sigma_imp
    n_boot: int = 4000
    seed: int = 2010
    status: str = "HISTORICAL_EXPLORATORY_PREREGISTERED_VD040"


def registration(p: Params | None = None) -> dict:
    d = asdict(p or Params())
    d.update(origins=ORIGINS, primary_hour=PRIMARY_HOUR, assets=ASSETS, models=MODELS, daily_designs=DAILY)
    return d


# ---------------------------------------------------------------- origins and features

def daily_origins(raw: pd.DataFrame, hour: int, h: int, p: Params | None = None) -> pd.DataFrame:
    """raw: state_time (5m bar END), underlying, return_5m. One row per asset-day at `hour`:00 UTC."""
    p = p or Params()
    out = []
    for a, g in raw.groupby("underlying", sort=True):
        z = g[["state_time", "return_5m"]].copy()
        z["state_time"] = pd.to_datetime(z["state_time"], utc=True)
        z = z.drop_duplicates("state_time").set_index("state_time").sort_index()
        grid = pd.date_range(z.index.min(), z.index.max(), freq="5min")
        r = z["return_5m"].reindex(grid).to_numpy(float)
        fin = np.isfinite(r)
        rr = np.where(fin, r, 0.0)
        cv, cs, cn = (np.r_[0.0, np.cumsum(rr * rr)], np.r_[0.0, np.cumsum(rr)], np.r_[0, np.cumsum(fin)])
        days = pd.date_range(grid[0].normalize() + pd.Timedelta(days=23), grid[-1].normalize(), freq="D")
        t = days + pd.Timedelta(hours=hour)
        e = grid.searchsorted(t, "right")          # bars with end time <= t are indices < e

        def past(n, frac, what=cv):
            s = e - n
            ok = s >= 0
            v = np.full(len(e), np.nan)
            cnt = np.where(ok, cn[e] - cn[np.maximum(s, 0)], 0)
            good = ok & (cnt >= frac * n)
            v[good] = (what[e[good]] - what[s[good]]) * (n / cnt[good] if what is cv else 1.0)
            return v

        k = h // 5
        f_ok = (e + k <= len(r))
        rv_next = np.full(len(e), np.nan)
        r_next = np.full(len(e), np.nan)
        i1 = np.minimum(e + k, len(r))
        full = f_ok & (cn[i1] - cn[e] == k)
        rv_next[full] = cv[i1[full]] - cv[e[full]]
        r_next[full] = cs[i1[full]] - cs[e[full]]
        d = pd.DataFrame({"date": days, "state_time": t, "underlying": a,
                          "rv1": past(BARS_DAY, p.min_frac_daily),
                          "rv5": past(5 * BARS_DAY, p.min_frac_daily) / 5,
                          "rv22": past(22 * BARS_DAY, p.min_frac_daily) / 22,
                          "rv30": past(6, 1.0), "rv60": past(12, 1.0), "rv240": past(48, 1.0),
                          "r1": past(BARS_DAY, 1.0, cs), "rv_next": rv_next, "R_next": r_next})
        out.append(d)
    D = pd.concat(out, ignore_index=True)
    D["h"] = h
    D["hour"] = hour
    with np.errstate(divide="ignore", invalid="ignore"):
        for c, src in (("l1", "rv1"), ("l5", "rv5"), ("l22", "rv22"), ("l30", "rv30"), ("l60", "rv60"), ("l240", "rv240")):
            D[c] = np.log(np.maximum(D[src].to_numpy(float), 1e-14))
        D["neg"] = np.minimum(D["r1"] / np.sqrt(D["rv1"]), 0.0)
    return D


# ---------------------------------------------------------------- daily models

def _X(d: pd.DataFrame, cols: list, assets: list) -> np.ndarray:
    dummies = [(d["underlying"].to_numpy() == a).astype(float) for a in assets[1:]]
    return np.column_stack([np.ones(len(d)), d[cols].to_numpy(float), *dummies])


def fit_daily(train: pd.DataFrame) -> dict:
    """OLS log-RV HAR designs + NAIVE scale, all with per-asset TRAIN mean smearing (variance-unbiased on TRAIN)."""
    fits = {}
    ok = np.isfinite(train["rv_next"]) & (train["rv_next"] > 0) & np.isfinite(train["rv1"])
    tr = train[ok]
    assets = sorted(tr["underlying"].unique())
    fits["NAIVE"] = {"scale": {a: float(g.rv_next.mean() / g.rv1.mean()) for a, g in tr.groupby("underlying")}}
    for name, cols in DAILY.items():
        q = tr[np.isfinite(tr[cols]).all(axis=1)]
        X = _X(q, cols, assets)
        y = np.log(q["rv_next"].to_numpy(float))
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        base = np.exp(X @ b)
        smear = {a: float(q.rv_next[ix].mean() / base[ix.to_numpy()].mean())
                 for a in assets for ix in [q.underlying == a]}
        fits[name] = {"cols": cols, "assets": assets, "beta": b.tolist(), "smear": smear, "n_train": int(len(q))}
    return fits


def predict_daily(D: pd.DataFrame, fits: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=D.index)
    out["F_NAIVE"] = D["rv1"] * D["underlying"].map(fits["NAIVE"]["scale"])
    for name in DAILY:
        f = fits[name]
        X = _X(D, f["cols"], f["assets"])
        v = np.exp(np.clip(X @ np.asarray(f["beta"]), -40, 5)) * D["underlying"].map(f["smear"]).to_numpy(float)
        v[~np.isfinite(D[f["cols"]]).all(axis=1).to_numpy()] = np.nan
        out[f"F_{name}"] = v
    return out


# ---------------------------------------------------------------- scoring

def qlike(rv, f):
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.asarray(rv, float) / np.asarray(f, float)
        return x - np.log(x) - 1.0


def _date_weights(dates: np.ndarray, rng, n_boot: int):
    u, inv = np.unique(dates, return_inverse=True)
    for _ in range(n_boot):
        yield np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv].astype(float)


def boot_mean(x: np.ndarray, dates: np.ndarray, n_boot: int, seed: int) -> tuple:
    ok = np.isfinite(x)
    x, dates = x[ok], dates[ok]
    rng = np.random.default_rng(seed)
    b = np.array([np.sum(w * x) / np.sum(w) for w in _date_weights(dates, rng, n_boot)])
    return float(x.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975)), int(len(x))


def auc(score: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    o = np.argsort(score, kind="mergesort")
    return _auc_w(np.asarray(score)[o], np.asarray(y, bool)[o], w[o])


def _auc_w(s: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted Mann-Whitney AUC with ties counted 1/2 (s sorted ascending)."""
    df = pd.DataFrame({"s": s, "p": w * y, "n": w * ~y}).groupby("s", sort=True)[["p", "n"]].sum()
    n_below = np.r_[0.0, np.cumsum(df["n"].to_numpy())[:-1]]
    num = np.sum(df["p"].to_numpy() * (n_below + 0.5 * df["n"].to_numpy()))
    return float(num / (df["p"].sum() * df["n"].sum()))


def score_table(E: pd.DataFrame, models, p: Params | None = None) -> pd.DataFrame:
    """E: rows with rv_next, R_next, date and F_<model>; all models on identical rows."""
    p = p or Params()
    cols = [f"F_{m}" for m in models]
    E = E[np.isfinite(E[cols]).all(axis=1) & (E["rv_next"] > 0)].copy()
    dates = E["date"].to_numpy()
    rows = []
    for k, m in enumerate(models):
        f = E[f"F_{m}"].to_numpy(float)
        rv = E["rv_next"].to_numpy(float)
        ql = boot_mean(qlike(rv, f), dates, p.n_boot, p.seed + k)
        lx, ly = np.log(f), np.log(rv)
        X = np.column_stack([np.ones(len(lx)), lx])
        b, *_ = np.linalg.lstsq(X, ly, rcond=None)
        r2 = 1 - np.var(ly - X @ b) / np.var(ly)
        rows.append({"model": m, "n": ql[3], "days": int(len(np.unique(dates))), "qlike": ql[0], "qlike_lo": ql[1],
                     "qlike_hi": ql[2], "mse_log": float(np.mean((ly - lx) ** 2)), "mz_a": float(b[0]),
                     "mz_b": float(b[1]), "mz_r2": float(r2), "bias_mean_ratio": float(rv.mean() / f.mean())})
    return pd.DataFrame(rows), E


def pair_diffs(E: pd.DataFrame, pairs, p: Params | None = None) -> pd.DataFrame:
    p = p or Params()
    dates = E["date"].to_numpy()
    rows = []
    for k, (a, b) in enumerate(pairs):
        d = qlike(E.rv_next, E[f"F_{a}"]) - qlike(E.rv_next, E[f"F_{b}"])
        m, lo, hi, n = boot_mean(np.asarray(d, float), dates, p.n_boot, p.seed + 100 + k)
        rows.append({"model": a, "vs": b, "qlike_diff": m, "lo": lo, "hi": hi, "n": n,
                     "better": "model" if hi < 0 else "vs" if lo > 0 else "tie"})
    return pd.DataFrame(rows)


def classification(E: pd.DataFrame, models, p: Params | None = None) -> pd.DataFrame:
    """Volatile day = rv_next above the asset's EVAL-sample quantile (a fixed per-asset label threshold; a trailing
    threshold leaks its own sampling noise into the AUC). Score = log F - log threshold, pooled across assets."""
    p = p or Params()
    Q = E.copy()
    Q["thr_fixed"] = Q.groupby("underlying")["rv_next"].transform(lambda v: v.quantile(p.vol_quantile))
    y = (Q["rv_next"] > Q["thr_fixed"]).to_numpy()
    dates = Q["date"].to_numpy()
    rows = []
    for k, m in enumerate(models):
        s = np.log(Q[f"F_{m}"].to_numpy(float)) - np.log(Q["thr_fixed"].to_numpy(float))
        a0 = auc(s, y)
        rng = np.random.default_rng(p.seed + 200 + k)
        bs = []
        for w in _date_weights(dates, rng, max(500, p.n_boot // 4)):
            keep = w > 0
            bs.append(auc(s[keep], y[keep], w[keep]))
        top = s >= np.quantile(s, 0.9)
        bot = s <= np.quantile(s, 0.1)
        rows.append({"model": m, "n": int(len(y)), "base_rate": float(y.mean()), "auc": a0,
                     "auc_lo": float(np.quantile(bs, .025)), "auc_hi": float(np.quantile(bs, .975)),
                     "hit_top_decile": float(y[top].mean()), "lift_top_decile": float(y[top].mean() / y.mean()),
                     "hit_bottom_decile": float(y[bot].mean())})
    return pd.DataFrame(rows)


def tail_table(Etr: pd.DataFrame, E: pd.DataFrame, models, p: Params | None = None) -> pd.DataFrame:
    """Exceedance P(|R_next| > k sqrt(F)): observed (EVAL) vs Gaussian vs Student-t fitted on TRAIN z."""
    p = p or Params()
    rows = []
    for m in models:
        ztr = (Etr["R_next"] / np.sqrt(Etr[f"F_{m}"])).to_numpy(float)
        ztr = ztr[np.isfinite(ztr)]
        if len(ztr) < 50:
            continue
        df, _, sc = stats.t.fit(ztr, floc=0)
        z = (E["R_next"] / np.sqrt(E[f"F_{m}"])).to_numpy(float)
        z = z[np.isfinite(z)]
        for k in p.tail_k:
            rows.append({"model": m, "k": k, "n": int(len(z)), "observed": float(np.mean(np.abs(z) > k)),
                         "gaussian": float(2 * stats.norm.sf(k)), "student_t_train": float(2 * stats.t.sf(k / sc, df)),
                         "t_df": float(df), "t_scale": float(sc), "z_sd_eval": float(np.std(z))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- versus implied (VD19 12:00 rows)

def implied_check(rows19: pd.DataFrame, F12: pd.DataFrame, models, p: Params | None = None, mv=None) -> tuple:
    """rows19: VD19 19_rows.csv (entry_hour 12, session = expiry date, entry = session-1d 12:00).
    F12: 12:00-origin rows with date, underlying, F_<model> for h=1200."""
    p = p or Params()
    r = rows19[rows19["entry_hour"] == 12].copy()
    r["date"] = pd.to_datetime(r["session"], utc=True) - pd.Timedelta(days=1)
    f = F12.copy()
    f["date"] = pd.to_datetime(f["date"], utc=True)
    M = r.merge(f[["date", "underlying", *[f"F_{m}" for m in models]]], on=["date", "underlying"], how="inner")
    M["log_rv"] = np.log(M["rv_path"])
    M["log_var_imp"] = 2 * np.log(M["sigma_imp"])
    info, econ = [], []
    vp = mv.Params(n_boot=p.n_boot) if mv is not None else None
    for k, m in enumerate(models):
        M[f"log_F_{m}"] = np.log(M[f"F_{m}"])
        M[f"x_{m}"] = 0.5 * (M[f"log_F_{m}"] - M["log_var_imp"])
        M[f"yrv_{m}"] = M["log_rv"] - M["log_var_imp"]
        if mv is None:
            continue
        enc = mv.session_bootstrap(M, ["log_var_imp", f"log_F_{m}"], "log_rv", vp, seed_offset=300 + k)
        slope = mv.session_bootstrap(M, [f"x_{m}"], f"yrv_{m}", vp, seed_offset=400 + k)
        alone = mv.session_bootstrap(M, [f"log_F_{m}"], "log_rv", vp, seed_offset=500 + k)
        info.append({"model": m, **{f"enc_{a}": b for a, b in enc.items()},
                     **{f"beta_{a}": b for a, b in slope.items() if a.startswith("x_")},
                     "alone_r2_proxy_slope": alone.get(f"log_F_{m}")})
        sel = M[M[f"x_{m}"] >= np.log(p.trade_ratio)]
        econ.append({"model": m, "subset": "sigma_model>=1.25 sigma_imp", **mv.session_mean_boot(sel, "R_net", vp)})
    ia = mv.session_bootstrap(M, ["log_var_imp"], "log_rv", vp, seed_offset=600) if mv is not None else {}
    econ.append({"model": "-", "subset": "all 12:00 straddles", **(mv.session_mean_boot(M, "R_net", vp) if mv else {})})
    return M, pd.DataFrame(info), pd.DataFrame(econ), ia


def decide(pairs: pd.DataFrame, info: pd.DataFrame) -> dict:
    g1 = pairs[(pairs.vs == "NAIVE") & pairs.model.isin(["ENGINE", "HAR_D"])]
    G1 = "PASS" if (g1.hi < 0).any() else "FAIL"
    e = pairs[(pairs.model == "ENGINE") & (pairs.vs == "HAR_D")]
    G2 = e.iloc[0]["better"] if len(e) else "n/a"
    G2 = {"model": "ENGINE_BETTER", "vs": "HAR_D_BETTER", "tie": "TIE"}.get(G2, G2)
    G3 = "FAIL"
    best = None
    if len(info):
        for _, r in info.iterrows():
            lo = r.get(f"enc_log_F_{r.model}_lo", np.nan)
            if np.isfinite(lo) and lo > 0:
                G3, best = "PASS", r.model
    verdict = ("BUY_OTM_ON_FORECAST_SUPPORTED_PROSPECTIVE" if G1 == "PASS" and G3 == "PASS" else
               "FORECASTS_VOL_BUT_NOT_BEYOND_IMPLIED" if G1 == "PASS" else "NO_SKILL")
    return {"version": VERSION, "G1_skill_vs_naive": G1, "G2_engine_vs_daily_har": G2,
            "G3_beyond_implied": G3, "G3_model": best, "verdict": verdict}


# ---------------------------------------------------------------- synthetic data

def synthetic_returns(n_days=500, assets=("A", "B", "C"), phi=0.97, vol_of_vol=0.25, seed=0,
                      start="2024-01-01", persistent=True) -> pd.DataFrame:
    """5m returns with daily log-vol AR(1) (or constant vol) and an intraday U-shape. Bar END timestamps."""
    rng = np.random.default_rng(seed)
    t = pd.date_range(pd.Timestamp(start, tz="UTC") + pd.Timedelta(minutes=5), periods=n_days * BARS_DAY, freq="5min")
    slot = np.arange(len(t)) % BARS_DAY
    season = 1 + 0.4 * np.cos(2 * np.pi * (slot - 170) / BARS_DAY)
    season /= np.sqrt(np.mean(season ** 2))
    out = []
    for j, a in enumerate(assets):
        lv = np.zeros(n_days)
        if persistent:
            e = rng.normal(0, vol_of_vol, n_days)
            for d in range(1, n_days):
                lv[d] = phi * lv[d - 1] + e[d]
        sd = 0.03 * (1 + 0.3 * j) / np.sqrt(BARS_DAY) * np.exp(np.repeat(lv, BARS_DAY)) * season
        out.append(pd.DataFrame({"state_time": t, "underlying": a, "return_5m": rng.standard_normal(len(t)) * sd}))
    return pd.concat(out, ignore_index=True)
