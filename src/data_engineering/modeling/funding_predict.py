"""
Funding Carry FC 02, Stage A: can future Binance USD-M funding be predicted, by how much, and
does the AM-T-HAR-Blend volatility engine help?

Pre-registration: discussions/Funding_Carry/003_claude_fc02_stage_a_preregistration.md

Panel: one row per asset and day, origin b = 00:05 UTC (as FC 01). Everything in a row's features
uses data with timestamps <= b; targets use (b, b + h].

Targets (annualized funding, sum of prints * 365 / days):  y1, y7 (PRIMARY), y30.
Feature blocks (added in this order; each model contains asset fixed effects):
    P    persistence     f1, f7, f30            trailing annualized funding
    PR   premium         prem_now, prem_24h     log(perp / spot) at b and its 24h mean (5m grid)
    M    momentum        mom7, mom30            log spot return over 7d / 30d
    OI   positioning     oi_chg1, oi_chg7, ls_acct, ls_top, taker_ls   (Binance futures metrics)
    B    breadth         br_med, br_hi          cross-asset median f7, share of assets with f7 >= 10%
    V    naive vol       lrv1, lrv7             log trailing realized variance (spot 5m), 1d / 7d
    E    engine          leng1, leng7           log AM-T-HAR-Blend (VD18 HAR_4 level) variance forecast
                                                for the next 24h / 7d
Engine: VD18 HAR_4 refitted with VD18's own code (v18.fit_har / predict_har: log-RV lags 30/60/240/1440 min,
asset effects, time of day, ridge 0.02, TRAIN net-return scale) on Binance SPOT 5m returns of the 30 assets,
TRAIN bars only, at h = 1440 and 10080 minutes.
Split: TRAIN origins b < 2025-10-01 00:05 whose target window ends before it; TEST origins >= 2025-10-01 00:05.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

VERSION = "funding_predict 1.0.0"

SPLIT = pd.Timestamp("2025-10-01 00:05", tz="UTC")
BOUNDARY = pd.Timedelta(minutes=5)
TARGETS = {"y1": 1, "y7": 7, "y30": 30}
PRIMARY = "y7"
BLOCKS = {
    "P": ["f1", "f7", "f30"],
    "PR": ["prem_now", "prem_24h"],
    "M": ["mom7", "mom30"],
    "OI": ["oi_chg1", "oi_chg7", "ls_acct", "ls_top", "taker_ls"],
    "B": ["br_med", "br_hi"],
    "V": ["lrv1", "lrv7"],
    "E": ["leng1", "leng7"],
}
ORDER = ("P", "PR", "M", "OI", "B", "V", "E")
ENGINE_H = (1440, 10080)
ENGINE_TRAIN_STRIDE = 3        # fit on every 3rd TRAIN bar (memory on the PC); VD18 used every bar
HAR_D = ["lrv1", "lrv7", "lrv30"]
WINSOR_X = (0.01, 0.99)
WINSOR_Y = (0.005, 0.995)
MIN_PRINT_FRAC = 0.8
MIN_BAR_FRAC = 0.95
OI_MIN_SHARE = 0.80            # OI block used only if >= 80% of otherwise-complete rows have it
BOOT_B = 2000
BOOT_BLOCK = 14
SEED = 20261018
ALPHA = 0.05


@dataclass(frozen=True)
class Spec:
    split: str = str(SPLIT)
    primary: str = PRIMARY


def registration() -> dict:
    return {"version": VERSION, "split": str(SPLIT), "targets": TARGETS, "primary": PRIMARY,
            "blocks": BLOCKS, "order": list(ORDER), "engine_horizons_min": list(ENGINE_H),
            "engine": "VD18 HAR_4 (v18.fit_har) on Binance spot 5m, TRAIN bars only",
            "engine_train_stride": ENGINE_TRAIN_STRIDE,
            "winsor_x": WINSOR_X, "winsor_y": WINSOR_Y, "boot_B": BOOT_B, "boot_block_days": BOOT_BLOCK,
            "oi_min_share": OI_MIN_SHARE, "alpha": ALPHA}


def _ns(ix) -> np.ndarray:
    return pd.DatetimeIndex(ix).as_unit("ns").asi8


# ----------------------------------------------------------------------------- inputs
def load_metrics(path) -> pd.DataFrame:
    """Binance USD-M metrics (5-minute): create_time, sum_open_interest_value, count_long_short_ratio,
    sum_toptrader_long_short_ratio, sum_taker_long_short_vol_ratio. Index = create_time (UTC)."""
    m = pd.read_parquet(path)
    t = pd.to_datetime(m["create_time"], utc=True)
    keep = ["sum_open_interest_value", "count_long_short_ratio", "sum_toptrader_long_short_ratio",
            "sum_taker_long_short_vol_ratio"]
    out = m[[c for c in keep if c in m]].apply(pd.to_numeric, errors="coerce")
    out.index = t
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def grid5(close_1m: pd.Series) -> pd.Series:
    """5m closes on a dense grid (index = bar END), from 1m closes indexed by minute END."""
    s = close_1m[(close_1m.index.minute % 5 == 0) & (close_1m.index.second == 0)]
    g = pd.date_range(s.index.min().ceil("5min"), s.index.max(), freq="5min")
    return s.reindex(g)


def _trail_sum(cs: np.ndarray, cn: np.ndarray, e: np.ndarray, n: int, need: float):
    s = e - n
    ok = s >= 0
    v = np.full(len(e), np.nan)
    cnt = np.where(ok, cn[e] - cn[np.maximum(s, 0)], 0)
    good = ok & (cnt >= need * n)
    v[good] = (cs[e[good]] - cs[np.maximum(s, 0)[good]]) * n / cnt[good]
    return v


def asset_daily(spot1m: pd.Series, perp1m: pd.Series, funding: pd.Series,
                metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    s5, p5 = grid5(spot1m), grid5(perp1m)
    g = s5.index.union(p5.index)
    s5, p5 = s5.reindex(g), p5.reindex(g)
    ls = np.log(s5.to_numpy(float))
    lp = np.log(p5.to_numpy(float))
    r = np.r_[np.nan, np.diff(ls)]
    fin = np.isfinite(r)
    cv = np.r_[0.0, np.cumsum(np.where(fin, r * r, 0.0))]
    cn = np.r_[0, np.cumsum(fin)]
    prem = lp - ls
    pf = np.isfinite(prem)
    cp = np.r_[0.0, np.cumsum(np.where(pf, prem, 0.0))]
    cpn = np.r_[0, np.cumsum(pf)]
    start = max(g.min(), funding.index.min()) + pd.Timedelta(days=31)
    end = min(g.max(), funding.index.max())
    b = pd.date_range(start.floor("D") + BOUNDARY, end, freq="D")
    gi = _ns(g)
    e = np.searchsorted(gi, _ns(b), side="right")              # bars with end <= b are < e
    at = e - 1
    out = pd.DataFrame({"b": b})
    out["S"] = np.where(at >= 0, s5.to_numpy(float)[np.clip(at, 0, None)], np.nan)
    out["prem_now"] = 1e4 * np.where(at >= 0, prem[np.clip(at, 0, None)], np.nan)
    p24 = np.full(len(e), np.nan)
    s = e - 288
    ok = s >= 0
    cnt = np.where(ok, cpn[e] - cpn[np.maximum(s, 0)], 0)
    good = ok & (cnt >= 0.9 * 288)
    p24[good] = (cp[e[good]] - cp[np.maximum(s, 0)[good]]) / cnt[good]
    out["prem_24h"] = 1e4 * p24
    for n, name in ((288, "lrv1"), (2016, "lrv7"), (8640, "lrv30")):
        v = _trail_sum(cv, cn, e, n, MIN_BAR_FRAC)
        out[name] = np.log(np.maximum(v, 1e-12))
    # forward realized variance (for the engine check): bars in (b, b + h], all present
    for n, name in ((288, "rv_fwd1"), (2016, "rv_fwd7")):
        i1 = e + n
        okf = i1 <= len(r)
        v = np.full(len(e), np.nan)
        i1c = np.minimum(i1, len(r))
        full = okf & ((cn[i1c] - cn[e]) == n)
        v[full] = cv[i1c[full]] - cv[e[full]]
        out[name] = v
    for days, name in ((7, "mom7"), (30, "mom30")):
        k = np.searchsorted(gi, _ns(b - pd.Timedelta(days=days)), side="right") - 1
        past = np.where(k >= 0, ls[np.clip(k, 0, None)], np.nan)
        out[name] = 1e2 * (np.where(at >= 0, ls[np.clip(at, 0, None)], np.nan) - past)
    # funding: trailing and forward sums of prints, annualized
    ft = _ns(funding.index)
    fr = funding.to_numpy(float)
    csr = np.r_[0.0, np.cumsum(fr)]
    spacing = float(np.median(np.diff(ft)) / 3.6e12) if len(ft) > 2 else 8.0
    per_day = 24.0 / max(spacing, 1.0)
    bi = _ns(b)
    kb = np.searchsorted(ft, bi, side="right")
    fmax = ft.max()
    for days, name in ((1, "f1"), (7, "f7"), (30, "f30")):
        k0 = np.searchsorted(ft, bi - days * 86_400 * 10**9, side="right")
        n = kb - k0
        out[name] = np.where(n >= MIN_PRINT_FRAC * days * per_day, (csr[kb] - csr[k0]) * 365.0 / days, np.nan)
    for name, days in TARGETS.items():
        hi = bi + days * 86_400 * 10**9
        k1 = np.searchsorted(ft, hi, side="right")
        n = k1 - kb
        okt = (hi <= fmax) & (n >= MIN_PRINT_FRAC * days * per_day)
        out[name] = np.where(okt, (csr[k1] - csr[kb]) * 365.0 / days, np.nan)
    # positioning
    for c in BLOCKS["OI"]:
        out[c] = np.nan
    if metrics is not None and len(metrics):
        mt = _ns(metrics.index)

        def asof(col, times):
            k = np.searchsorted(mt, times, side="right") - 1
            v = np.where(k >= 0, metrics[col].to_numpy(float)[np.clip(k, 0, None)], np.nan)
            age = times - np.where(k >= 0, mt[np.clip(k, 0, None)], 0)
            return np.where((k >= 0) & (age <= 15 * 60 * 10**9), v, np.nan)

        def mean24(col):
            x = np.log(np.clip(metrics[col].to_numpy(float), 1e-9, None))
            fx = np.isfinite(x)
            c = np.r_[0.0, np.cumsum(np.where(fx, x, 0.0))]
            cc = np.r_[0, np.cumsum(fx)]
            k1 = np.searchsorted(mt, bi, side="right")
            k0 = np.searchsorted(mt, bi - 86_400 * 10**9, side="right")
            n = cc[k1] - cc[k0]
            return np.where(n >= 0.8 * 288, (c[k1] - c[k0]) / np.maximum(n, 1), np.nan)

        if "sum_open_interest_value" in metrics:
            oi0 = asof("sum_open_interest_value", bi)
            for days, name in ((1, "oi_chg1"), (7, "oi_chg7")):
                oik = asof("sum_open_interest_value", bi - days * 86_400 * 10**9)
                with np.errstate(divide="ignore", invalid="ignore"):
                    out[name] = 1e2 * np.log(oi0 / oik)
        for col, name in (("count_long_short_ratio", "ls_acct"), ("sum_toptrader_long_short_ratio", "ls_top"),
                          ("sum_taker_long_short_vol_ratio", "taker_ls")):
            if col in metrics:
                out[name] = mean24(col)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def add_breadth(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    g = p.groupby("b")["f7"]
    p["br_med"] = g.transform("median")
    p["br_hi"] = g.transform(lambda x: float((x >= 0.10).mean()) if x.notna().any() else np.nan)
    return p


# ----------------------------------------------------------------------------- engine (VD18 HAR_4)
def spot_returns_5m(spot1m: pd.Series, asset: str) -> pd.DataFrame:
    s5 = grid5(spot1m)
    r = np.log(s5).diff()
    return pd.DataFrame({"state_time": s5.index, "underlying": asset, "return_5m": r.to_numpy(float)})


def engine_forecasts(raw: pd.DataFrame, origins: pd.DataFrame, v18, split=SPLIT) -> tuple[pd.DataFrame, dict]:
    """Fit VD18 HAR_4 at each ENGINE_H on TRAIN bars (target window ends before `split`) and predict
    log variance at the origins (columns underlying, b). Origin bar = 5m bar ending at b."""
    p = raw.copy()
    p["state_time"] = pd.to_datetime(p["state_time"], utc=True)
    p = p.sort_values(["underlying", "state_time"]).reset_index(drop=True)
    p["slot"] = p.state_time.dt.hour * 12 + p.state_time.dt.minute // 5
    p = v18.add_har_lags(p)
    lag_cols = [f"har_logrv_{k}" for k in v18.ALL_FEATURES]
    out = origins[["underlying", "b"]].copy()
    fits = {}
    for h in ENGINE_H:
        n = h // 5
        RV = np.full(len(p), np.nan)
        R = np.full(len(p), np.nan)
        for a, g in p.groupby("underlying", sort=False):
            ix = g.index.to_numpy()
            r = g["return_5m"].to_numpy(float)
            fin = np.isfinite(r)
            cs = np.r_[0.0, np.cumsum(np.where(fin, r, 0.0))]
            cv = np.r_[0.0, np.cumsum(np.where(fin, r * r, 0.0))]
            cc = np.r_[0, np.cumsum(fin)]
            t = np.arange(len(r))
            e = np.minimum(t + n, len(r) - 1)
            ok = (t + n < len(r)) & (cc[e + 1] - cc[t + 1] == n)
            RV[ix] = np.where(ok, cv[e + 1] - cv[t + 1], np.nan)
            R[ix] = np.where(ok, cs[e + 1] - cs[t + 1], np.nan)
        p[f"RV_{h}"] = RV
        p[f"R_{h}"] = R
        end = p["state_time"] + pd.Timedelta(minutes=h)
        tr = p[(end < split) & np.isfinite(p[lag_cols].to_numpy(float)).all(axis=1) & np.isfinite(RV)]
        tr = tr.iloc[::ENGINE_TRAIN_STRIDE]
        fit = v18.fit_har(tr, h, v18.FEATURE_SETS["HAR_4"])
        fits[h] = {k: v for k, v in fit.items() if k != "design"} | {"asset_names": fit["design"]["asset_names"]}
        m = out.rename(columns={"b": "state_time"}).merge(p[["underlying", "state_time", "slot", *lag_cols]],
                                                         on=["underlying", "state_time"], how="left")
        good = (np.isfinite(m[lag_cols].to_numpy(float)).all(axis=1)
                & m["underlying"].isin(list(fit["net_return_scales"])).to_numpy())
        v = np.full(len(m), np.nan)
        if good.any():
            v[good] = v18.predict_har(m[good], fit)
        out[f"leng{1 if h == 1440 else 7}"] = np.log(np.maximum(v, 1e-14))
        p = p.drop(columns=[f"RV_{h}", f"R_{h}"])
    return out, fits


# ----------------------------------------------------------------------------- models
def _winsor_bounds(x: np.ndarray, q) -> tuple:
    return float(np.nanquantile(x, q[0])), float(np.nanquantile(x, q[1]))


def design(df: pd.DataFrame, cols: list, assets: list, bounds: dict) -> np.ndarray:
    X = [np.ones(len(df))]
    for a in assets[1:]:
        X.append((df["asset"].to_numpy() == a).astype(float))
    for c in cols:
        lo, hi = bounds[c]
        X.append(np.clip(df[c].to_numpy(float), lo, hi))
    return np.column_stack(X)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list, y: str) -> tuple[np.ndarray, np.ndarray, dict]:
    assets = sorted(train["asset"].unique())
    bounds = {c: _winsor_bounds(train[c].to_numpy(float), WINSOR_X) for c in cols}
    ylo, yhi = _winsor_bounds(train[y].to_numpy(float), WINSOR_Y)
    Xtr = design(train, cols, assets, bounds)
    ytr = np.clip(train[y].to_numpy(float), ylo, yhi)
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    Xte = design(test, cols, assets, bounds)
    return Xtr @ beta, Xte @ beta, {"beta": beta.tolist(), "cols": ["const", *[f"fe_{a}" for a in assets[1:]], *cols],
                                    "y_bounds": [ylo, yhi]}


def _date_blocks(dates: np.ndarray, block: int, rng, B: int):
    u = np.unique(dates)
    n = len(u)
    nb = int(math.ceil(n / block))
    st = rng.integers(0, max(n - block + 1, 1), size=(B, nb))
    return u, (st[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]


def boot_delta(loss_a: np.ndarray, loss_b: np.ndarray, dates: np.ndarray, B: int = BOOT_B,
               block: int = BOOT_BLOCK, seed: int = SEED) -> tuple:
    """Mean(loss_a - loss_b) with a date-block bootstrap (rows pooled within dates)."""
    d = pd.DataFrame({"t": dates, "x": loss_a - loss_b}).groupby("t")["x"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    _, idx = _date_blocks(d.index.to_numpy(), block, rng, B)
    s, c = d["sum"].to_numpy(), d["count"].to_numpy()
    m = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.quantile(m, ALPHA / 2)), float(np.quantile(m, 1 - ALPHA / 2))


def r2_oos(y: np.ndarray, f: np.ndarray, bench: np.ndarray) -> float:
    return float(1 - np.sum((y - f) ** 2) / np.sum((y - bench) ** 2))


def run_models(panel: pd.DataFrame, y: str, blocks_used: list) -> dict:
    """Nested sequence on common rows; returns predictions, OOS metrics and nested deltas."""
    cols_all = [c for b in blocks_used for c in BLOCKS[b]]
    d = panel.dropna(subset=cols_all + [y]).copy()
    tr = d[(d["b"] < SPLIT) & (d["b"] + pd.to_timedelta(TARGETS[y], unit="D") <= SPLIT)]
    te = d[d["b"] >= SPLIT]
    ylo, yhi = _winsor_bounds(tr[y].to_numpy(float), WINSOR_Y)
    yte = np.clip(te[y].to_numpy(float), ylo, yhi)
    _, fe0, _ = fit_predict(tr, te, [], y)                 # asset fixed effects only (benchmark)
    preds, fits, rows = {"FE": fe0}, {}, []
    cum = []
    for b in blocks_used:
        cum = cum + BLOCKS[b]
        name = "+".join(blocks_used[: blocks_used.index(b) + 1])
        _, f, fit = fit_predict(tr, te, cum, y)
        preds[name] = f
        fits[name] = fit
    dates = te["b"].to_numpy()
    names = list(preds)
    for i, n in enumerate(names):
        row = {"target": y, "model": n, "n_train": len(tr), "n_test": len(te),
               "r2_oos_vs_FE": r2_oos(yte, preds[n], fe0) if n != "FE" else 0.0,
               "rmse": float(np.sqrt(np.mean((yte - preds[n]) ** 2))),
               "corr": float(np.corrcoef(yte, preds[n])[0, 1]) if np.std(preds[n]) > 0 else np.nan}
        if i > 0:
            prev = names[i - 1]
            dm, lo, hi = boot_delta((yte - preds[prev]) ** 2, (yte - preds[n]) ** 2, dates)
            row.update(added=n.split("+")[-1], dmse=dm, dmse_lo=lo, dmse_hi=hi, adds=bool(lo > 0))
        rows.append(row)
    if "P" in preds:
        dm, lo, hi = boot_delta((yte - fe0) ** 2, (yte - preds["P"]) ** 2, dates)
        rows[1].update(dmse_vs_FE=dm, dmse_vs_FE_lo=lo, dmse_vs_FE_hi=hi)
    return {"table": pd.DataFrame(rows), "test": te.assign(y_w=yte, **{f"pred_{k}": v for k, v in preds.items()}),
            "train": tr, "fits": fits, "y_bounds": (ylo, yhi)}


def unique_contribution(panel: pd.DataFrame, y: str, blocks_used: list) -> pd.DataFrame:
    """Full model vs full-minus-block, same rows: how much each block adds given all others."""
    cols_all = [c for b in blocks_used for c in BLOCKS[b]]
    d = panel.dropna(subset=cols_all + [y]).copy()
    tr = d[(d["b"] < SPLIT) & (d["b"] + pd.to_timedelta(TARGETS[y], unit="D") <= SPLIT)]
    te = d[d["b"] >= SPLIT]
    ylo, yhi = _winsor_bounds(tr[y].to_numpy(float), WINSOR_Y)
    yte = np.clip(te[y].to_numpy(float), ylo, yhi)
    _, ff, _ = fit_predict(tr, te, cols_all, y)
    _, fe0, _ = fit_predict(tr, te, [], y)
    rows = []
    for b in blocks_used:
        cols = [c for c in cols_all if c not in BLOCKS[b]]
        _, fm, _ = fit_predict(tr, te, cols, y)
        dm, lo, hi = boot_delta((yte - fm) ** 2, (yte - ff) ** 2, te["b"].to_numpy())
        rows.append({"target": y, "block": b, "r2_full": r2_oos(yte, ff, fe0), "r2_without": r2_oos(yte, fm, fe0),
                     "dmse": dm, "dmse_lo": lo, "dmse_hi": hi, "unique_adds": bool(lo > 0)})
    return pd.DataFrame(rows)


def engine_vs_naive_for_funding(panel: pd.DataFrame, y: str, base_blocks: list) -> pd.DataFrame:
    """On common rows: base + V vs base + E vs base + V + E."""
    base = [c for b in base_blocks for c in BLOCKS[b]]
    d = panel.dropna(subset=base + BLOCKS["V"] + BLOCKS["E"] + [y]).copy()
    tr = d[(d["b"] < SPLIT) & (d["b"] + pd.to_timedelta(TARGETS[y], unit="D") <= SPLIT)]
    te = d[d["b"] >= SPLIT]
    ylo, yhi = _winsor_bounds(tr[y].to_numpy(float), WINSOR_Y)
    yte = np.clip(te[y].to_numpy(float), ylo, yhi)
    _, fe0, _ = fit_predict(tr, te, [], y)
    P = {k: fit_predict(tr, te, c, y)[1] for k, c in
         (("base", base), ("base+V", base + BLOCKS["V"]), ("base+E", base + BLOCKS["E"]),
          ("base+V+E", base + BLOCKS["V"] + BLOCKS["E"]))}
    rows = []
    t = te["b"].to_numpy()
    for a, b in (("base", "base+V"), ("base", "base+E"), ("base+V", "base+E"), ("base+V", "base+V+E")):
        dm, lo, hi = boot_delta((yte - P[a]) ** 2, (yte - P[b]) ** 2, t)
        rows.append({"target": y, "from": a, "to": b, "r2_from": r2_oos(yte, P[a], fe0), "r2_to": r2_oos(yte, P[b], fe0),
                     "dmse": dm, "dmse_lo": lo, "dmse_hi": hi, "improves": bool(lo > 0)})
    return pd.DataFrame(rows)


def quintiles(test: pd.DataFrame, pred_col: str, y: str = "y_w") -> pd.DataFrame:
    q = pd.qcut(test[pred_col].rank(method="first"), 5, labels=False)
    g = test.assign(q=q).groupby("q")
    return pd.DataFrame({"n": g.size(), "pred_mean": g[pred_col].mean(), "realized_mean": g[y].mean(),
                         "realized_neg_share": g[y].apply(lambda s: float((s < 0).mean()))}).reset_index()


# ----------------------------------------------------------------------------- volatility check
def qlike(rv: np.ndarray, f: np.ndarray) -> np.ndarray:
    x = rv / f
    return x - np.log(x) - 1


def vol_check(panel: pd.DataFrame) -> pd.DataFrame:
    """Next-24h and next-7d spot RV: ENGINE vs NAIVE (trailing RV, same horizon) vs HAR_D (daily HAR on
    lrv1/lrv7/lrv30 fitted on TRAIN, log target). QLIKE on TEST, date-block bootstrap of differences."""
    rows = []
    for h, rvc, eng, naive in ((1, "rv_fwd1", "leng1", "lrv1"), (7, "rv_fwd7", "leng7", "lrv7")):
        d = panel.dropna(subset=[rvc, eng, *HAR_D]).copy()
        d = d[d[rvc] > 0]
        tr = d[(d["b"] < SPLIT) & (d["b"] + pd.Timedelta(days=h) <= SPLIT)]
        te = d[d["b"] >= SPLIT]
        assets = sorted(tr["asset"].unique())
        bounds = {c: _winsor_bounds(tr[c].to_numpy(float), (0.0, 1.0)) for c in HAR_D}
        Xtr = design(tr, HAR_D, assets, bounds)
        beta, *_ = np.linalg.lstsq(Xtr, np.log(tr[rvc].to_numpy(float)), rcond=None)
        res = np.log(tr[rvc].to_numpy(float)) - Xtr @ beta
        smear = float(np.mean(np.exp(res)))
        F = {"ENGINE": np.exp(te[eng].to_numpy(float)),
             "NAIVE": np.exp(te[naive].to_numpy(float)),
             "HAR_D": smear * np.exp(design(te, HAR_D, assets, bounds) @ beta)}
        rv = te[rvc].to_numpy(float)
        L = {k: qlike(rv, v) for k, v in F.items()}
        t = te["b"].to_numpy()
        for k in F:
            rows.append({"horizon_days": h, "model": k, "qlike": float(np.mean(L[k])), "n": len(te),
                         "mz_slope": float(np.polyfit(np.log(F[k]), np.log(rv), 1)[0])})
        for a, b in (("NAIVE", "ENGINE"), ("HAR_D", "ENGINE")):
            dm, lo, hi = boot_delta(L[a], L[b], t)
            rows.append({"horizon_days": h, "model": f"{a}-minus-ENGINE", "qlike": dm, "lo": lo, "hi": hi,
                         "engine_better": bool(lo > 0), "n": len(te)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- verdict
def decide(seq7: pd.DataFrame, engine_fund: pd.DataFrame, vc: pd.DataFrame) -> dict:
    t = seq7.set_index("model")
    p = t.loc["P"]
    q1 = bool(p.get("dmse_vs_FE_lo", np.nan) > 0)
    adds = {r["added"]: bool(r["adds"]) for _, r in seq7.dropna(subset=["added"]).iterrows()}
    ev = engine_fund.set_index(["from", "to"])
    q3b = bool(ev.loc[("base+V", "base+V+E"), "improves"])
    vv = vc.dropna(subset=["engine_better"]).set_index(["horizon_days", "model"])["engine_better"]
    q3a = {f"{h}d": bool(vv.loc[(h, "NAIVE-minus-ENGINE")] and vv.loc[(h, "HAR_D-minus-ENGINE")]) for h in (1, 7)}
    best = seq7.iloc[seq7["r2_oos_vs_FE"].idxmax()] if len(seq7) else None
    return {"Q1_predictable_y7": q1, "Q1_r2_oos_persistence": float(p["r2_oos_vs_FE"]),
            "Q1_r2_oos_full_model": float(seq7["r2_oos_vs_FE"].iloc[-1]),
            "Q1_best_model_descriptive_selected_on_TEST": None if best is None else best["model"],
            "Q1_best_r2_oos": None if best is None else float(best["r2_oos_vs_FE"]),
            "Q2_block_adds_in_sequence": adds,
            "Q3a_engine_better_vol_forecaster": q3a,
            "Q3b_engine_adds_to_funding_beyond_naive_vol": q3b,
            "engine_useful_for_carry": bool(q3b)}


# ----------------------------------------------------------------------------- synthetic
def synthetic(n_days: int = 560, assets=("AAA", "BBB", "CCC", "DDD"), seed: int = 0, start: str = "2024-08-01"):
    """Spot/perp 1m and 8h funding where funding = persistent regime + premium; premium tracks the regime and
    a fast component; volatility rises with the regime (so vol has some information about funding)."""
    rng = np.random.default_rng(seed)
    n = n_days * 1440
    t = pd.date_range(pd.Timestamp(start, tz="UTC") + pd.Timedelta(minutes=1), periods=n, freq="1min")
    nf = n_days * 3
    tf = pd.date_range(pd.Timestamp(start, tz="UTC") + pd.Timedelta(hours=8), periods=nf, freq="8h")
    spot, perp, fund, met = {}, {}, {}, {}
    common = np.zeros(nf)
    for i in range(1, nf):
        common[i] = 0.995 * common[i - 1] + 0.1 * rng.standard_normal()
    for j, a in enumerate(assets):
        x = np.zeros(nf)
        for i in range(1, nf):
            x[i] = 0.99 * x[i - 1] + 0.12 * rng.standard_normal()
        reg = np.tanh(0.6 * x + 0.6 * common)
        f = 1e-4 + 2.5e-4 * reg + 0.5e-4 * rng.standard_normal(nf)
        regm = np.repeat(reg, 480)[:n]
        vol = 5e-4 * np.exp(0.5 * regm)
        lp = np.log(40.0 * (j + 1)) + np.cumsum(vol * rng.standard_normal(n))
        prem = 8e-4 * regm + 2e-4 * rng.standard_normal(n)
        spot[a] = pd.Series(np.exp(lp), index=t)
        perp[a] = pd.Series(np.exp(lp + prem), index=t)
        fund[a] = pd.Series(f, index=tf)
        tm = pd.date_range(t[0].ceil("5min"), t[-1], freq="5min")
        rm = np.repeat(reg, 96)[:len(tm)]
        met[a] = pd.DataFrame({"sum_open_interest_value": 1e8 * np.exp(np.cumsum(1e-3 * rng.standard_normal(len(tm)) + 2e-5 * rm)),
                               "count_long_short_ratio": np.exp(0.3 * rm + 0.05 * rng.standard_normal(len(tm))),
                               "sum_toptrader_long_short_ratio": np.exp(0.2 * rm + 0.05 * rng.standard_normal(len(tm))),
                               "sum_taker_long_short_vol_ratio": np.exp(0.1 * rng.standard_normal(len(tm)))}, index=tm)
    return spot, perp, fund, met
