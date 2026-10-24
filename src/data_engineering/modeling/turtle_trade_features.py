"""Turtle Breakout 04: what distinguishes winning from losing trades of the ORIGINAL turtle strategy?

Protocol: discussions/Turtle_Breakout/006_claude_tb04_preregistration.md (fixed before any real-data output).

Trades: the TB 01 original rules (turtle_breakout.Params() defaults: daily 17:00 NY, 20-day high entry,
max(entry - 2 ATR, 10-day low) exit, long only, fees + slippage + funding), run per asset.
    DISCOVERY     26 other Binance USD-M USDT perpetuals (Pairs fOU universe minus the four), full history
    CONFIRMATION  BTC/ETH/SOL/XRP from 2020-10-15 (the TB 01 primary trades)
Labels: R = net return per notional / (2 ATR / entry)  (R-multiple, net of costs); win = R > 0;
        big = R >= 3 (secondary).

ENTRY features: known before the entry bar (5m bars ending strictly before it; levels of completed days).
PATH features: measured during the trade - descriptive only, they explain, they cannot filter entries.
All features are dimensionless (ATR units, ratios, z-scores) so they pool across assets.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import turtle_breakout as tb

VERSION = "TB04-0.1.0"
CONFIRM_ASSETS = ("BTC", "ETH", "SOL", "XRP")
UNIVERSE = ("BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "LINK", "DOT", "LTC", "BCH", "UNI", "NEAR", "APT",
            "ARB", "OP", "SUI", "FIL", "ATOM", "ETC", "INJ", "AAVE", "TON", "WIF", "HBAR", "XLM", "SEI", "TIA",
            "ENA", "WLD")
DAY = 288                                           # 5m bars per day
BIG_R = 3.0

ENTRY_FEATURES = {
    "price": ["brk_gap_atr", "move_to_level_atr", "atr_pct", "atr_rel_100", "range20_atr", "ret20_atr", "ret60_log", "dist_ma50_atr",
              "dist_ma200_atr", "above_ma200", "days_since_prev_break", "entry_day_frac", "prev_trade_win",
              "consec_losses"],
    "flow": ["vol24_rel", "vol1h_rel", "tbr_1h", "tbr_24h", "cvd4h_norm", "trades24_rel", "trade_size_rel"],
    "funding": ["fund_last_bp", "fund_3d_bp", "fund_z30"],
    "engine": ["eng_sig1d_over_atr", "eng_sig60_rel30", "rv24_over_atr", "jump24_sig"],
    "oi": ["oi_chg_24h", "oi_chg_7d", "oi_over_vol", "top_ls", "top_ls_chg24", "global_ls", "taker_ls_24h"],
    "basis": ["basis_bp", "basis_24h_bp", "basis_z7d"],
    "market": ["btc_ret20_log", "btc_in_breakout", "breadth"],
}
PATH_FEATURES = ["mfe_R", "mae_R", "days_to_mfe", "R_1d", "R_3d", "R_5d", "tbr_first24h", "tbr_first72h",
                 "vol_first3d_rel", "oi_chg_first3d", "fund_during_bp", "hold_days"]


@dataclass(frozen=True)
class FParams:
    fdr_q: float = 0.10
    confirm_alpha: float = 0.05
    n_boot: int = 2000
    n_perm: int = 200
    ridge: float = 1.0
    min_feature_n: int = 100
    seed: int = 20261027
    status: str = "PREREGISTERED_TB006"


def registration(fp: FParams | None = None) -> dict:
    d = asdict(fp or FParams())
    d.update(version=VERSION, confirm_assets=list(CONFIRM_ASSETS),
             discovery_assets=[a for a in UNIVERSE if a not in CONFIRM_ASSETS],
             entry_features=ENTRY_FEATURES, path_features=PATH_FEATURES, label="win = R > 0; secondary big = R >= 3",
             gates={"univariate": "discovery: AUC month-cluster bootstrap p, BH-FDR q < 0.10 -> confirmation: same "
                                  "direction, one-sided p < 0.05 Holm over the discovery survivors",
                    "model": "ridge logistic on all entry features fitted on discovery; confirmation AUC 95% lower "
                             "bound > 0.5 AND label-permutation p < 0.05"})
    return d


# ---------------------------------------------------------------- data

def load_um(path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1m UM klines -> 5m OHLC (strategy) and 5m flow sums (volume, quote_volume, trades, taker_buy_volume)."""
    cols = ["open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_volume"]
    d = pd.read_parquet(path, columns=cols)
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d = d.drop_duplicates("open_time").sort_values("open_time")
    g = d.groupby(d["open_time"].dt.floor("5min"))
    b = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last()}).astype(float)
    x = g[["volume", "quote_volume", "trades", "taker_buy_volume"]].sum().astype(float)
    b.index = b.index + pd.Timedelta(minutes=5); x.index = b.index
    b.index.name = x.index.name = "end"
    return b, x


def load_spot_close(path) -> pd.Series | None:
    try:
        d = pd.read_parquet(path, columns=["open_time", "close"])
    except Exception:                                                     # noqa: BLE001
        return None
    t = pd.to_datetime(d["open_time"], utc=True)
    s = pd.Series(d["close"].astype(float).to_numpy(), index=t).sort_index()
    s = s[~s.index.duplicated()]
    s = s.groupby(s.index.floor("5min")).last()
    s.index = s.index + pd.Timedelta(minutes=5)
    return s


def load_metrics(path) -> pd.DataFrame | None:
    try:
        m = pd.read_parquet(path)
    except Exception:                                                     # noqa: BLE001
        return None
    m["create_time"] = pd.to_datetime(m["create_time"], utc=True)
    return m.drop_duplicates("create_time").set_index("create_time").sort_index()


# ---------------------------------------------------------------- engine for any asset

def engine_variances(end: pd.DatetimeIndex, close: np.ndarray, fits: dict, asset: str) -> np.ndarray:
    """Frozen VD18 HAR_4 variances (15/60/240) on 5m closes. Assets outside the VD18 fit use the baseline asset
    effect (all dummies 0); engine features are therefore used as ratios (to ATR, to the asset's own median)."""
    from . import vwap_drift as vd
    z = pd.DataFrame({"close": close}, index=end)
    r = np.log(z["close"]).diff()
    r[pd.Series(end, index=end).diff().ne(pd.Timedelta(minutes=5)).to_numpy()] = np.nan
    sq = r.pow(2)
    for L in (30, 60, 240, 1440):
        z[f"har_logrv_{L}"] = np.log(np.maximum(sq.rolling(L // 5, min_periods=L // 5).sum(), 1e-16))
    z["slot"] = (end.hour * 60 + end.minute) // 5
    out = []
    for h in vd.ENGINE_H:
        f = fits[h]; dsg = f["design"]
        x = (z[dsg["features"]].to_numpy(float) - np.asarray(dsg["center"])) / np.asarray(dsg["std"])
        ang = 2 * np.pi * z["slot"].to_numpy(float) / 288
        extras = [np.full(len(z), float(asset == a)) for a in dsg["asset_names"][1:]]
        X = np.column_stack([np.ones(len(z)), x, np.sin(ang), np.cos(ang), *extras])
        ok = np.isfinite(X).all(axis=1)
        v = np.full(len(z), np.nan)
        scale = f["net_return_scales"].get(asset, float(np.median(list(f["net_return_scales"].values()))))
        v[ok] = np.exp(np.clip(X[ok] @ np.asarray(f["coefficients"], float), -40, 5)) * scale
        out.append(v)
    return np.column_stack(out)


# ---------------------------------------------------------------- trades and features

def run_trades(d: tb.AssetData, p: tb.Params, start: pd.Timestamp, end=None) -> pd.DataFrame:
    _, T, _ = tb.simulate({d.name: d}, p, start=start, end=end)
    T = T[T.reason != "end"].reset_index(drop=True)
    if not len(T):
        return T
    k = tb.period_id(pd.DatetimeIndex(T.entry_time), p.period) - d.pid0
    atr = d.atr[k]
    T["k"] = k
    T["atr_entry"] = atr
    T["R"] = T.ret_notional / (p.stop_atr * atr / T.entry_px)
    T["win"] = (T.R > 0).astype(int)
    T["big"] = (T.R >= BIG_R).astype(int)
    return T


def _csum(x):
    return np.r_[0.0, np.cumsum(np.nan_to_num(np.asarray(x, float)))]


def _wsum(C, a, b):
    """sum over bars [a, b) from a cumulative array with C[0] = 0 (a, b arrays, clipped)."""
    a = np.clip(a, 0, len(C) - 1); b = np.clip(b, 0, len(C) - 1)
    return C[b] - C[a]


def entry_and_path_features(A: str, d: tb.AssetData, x5: pd.DataFrame, T: pd.DataFrame, fund: pd.DataFrame | None,
                            spot: pd.Series | None, met: pd.DataFrame | None, fits: dict | None) -> pd.DataFrame:
    if not len(T):
        return pd.DataFrame()
    je = T.je.to_numpy(); jx = T.jx.to_numpy(); k = T.k.to_numpy()
    E = T.entry_px.to_numpy(); atr = T.atr_entry.to_numpy(); Ru = 2 * atr
    n = len(d.c)
    j0 = je                                             # features use bars [.., j0) = strictly before the entry bar
    out = pd.DataFrame({"asset": A, "entry_time": T.entry_time.to_numpy(), "exit_time": T.exit_time.to_numpy(),
                        "R": T.R.to_numpy(), "win": T.win.to_numpy(), "big": T.big.to_numpy(),
                        "ret_notional": T.ret_notional.to_numpy(), "reason": T.reason.to_numpy()})
    # ---- price / trend (daily periods)
    pc = pd.Series(d.pc).ffill().to_numpy()
    km1 = np.maximum(k - 1, 0)
    out["brk_gap_atr"] = (E - d.hi_n[k]) / atr
    out["move_to_level_atr"] = (d.hi_n[k] - pc[km1]) / atr
    out["atr_pct"] = atr / E
    atr_med = pd.Series(d.atr).rolling(100, min_periods=50).median().shift(1).to_numpy()
    out["atr_rel_100"] = atr / atr_med[k]
    out["range20_atr"] = (d.hi_n[k] - d.lo_n[k]) / atr
    out["ret20_atr"] = (pc[km1] - pc[np.maximum(k - 21, 0)]) / atr
    out["ret60_log"] = np.log(pc[km1] / pc[np.maximum(k - 61, 0)])
    ma50 = pd.Series(pc).rolling(50, min_periods=45).mean().to_numpy()
    ma200 = pd.Series(pc).rolling(200, min_periods=180).mean().to_numpy()
    out["dist_ma50_atr"] = (E - ma50[km1]) / atr
    out["dist_ma200_atr"] = (E - ma200[km1]) / atr
    out["above_ma200"] = np.where(np.isfinite(ma200[km1]), (E > ma200[km1]).astype(float), np.nan)
    brk = np.where(np.isfinite(d.hi_n), d.ph >= d.hi_n, False)
    last_brk = pd.Series(np.where(brk, np.arange(len(brk)), np.nan)).ffill().shift(1).to_numpy()
    out["days_since_prev_break"] = k - last_brk[k]
    out["entry_day_frac"] = (je - d.s[k]) / max(1, (tb.PERIOD_MIN["day"] // 5))
    prev_win, consec = [], []
    for i in range(len(T)):
        prior = T.iloc[:i]
        prior = prior[prior.exit_time <= T.entry_time.iloc[i]]
        if not len(prior):
            prev_win.append(np.nan); consec.append(np.nan); continue
        prev_win.append(float(prior.win.iloc[-1]))
        c = 0
        for w in prior.win.to_numpy()[::-1]:
            if w:
                break
            c += 1
        consec.append(c)
    out["prev_trade_win"] = prev_win; out["consec_losses"] = consec
    # ---- flow (5m sums)
    xv = x5.reindex(d.end)
    V, Q, N, TB = (_csum(xv[c].to_numpy()) for c in ("volume", "quote_volume", "trades", "taker_buy_volume"))
    v24 = _wsum(V, j0 - DAY, j0); v20 = _wsum(V, j0 - 21 * DAY, j0 - DAY) / 20
    out["vol24_rel"] = v24 / v20
    out["vol1h_rel"] = _wsum(V, j0 - 12, j0) / (_wsum(V, j0 - 21 * DAY, j0 - DAY) / (20 * 24))
    with np.errstate(invalid="ignore", divide="ignore"):
        out["tbr_1h"] = _wsum(TB, j0 - 12, j0) / _wsum(V, j0 - 12, j0)
        out["tbr_24h"] = _wsum(TB, j0 - DAY, j0) / v24
        out["cvd4h_norm"] = (2 * _wsum(TB, j0 - 48, j0) - _wsum(V, j0 - 48, j0)) / _wsum(V, j0 - 48, j0)
        n24 = _wsum(N, j0 - DAY, j0); n20 = _wsum(N, j0 - 21 * DAY, j0 - DAY) / 20
        out["trades24_rel"] = n24 / n20
        out["trade_size_rel"] = (v24 / n24) / (_wsum(V, j0 - 21 * DAY, j0 - DAY) / _wsum(N, j0 - 21 * DAY, j0 - DAY))
    # ---- funding
    if fund is not None and len(fund):
        ft = pd.DatetimeIndex(fund["t"]); fr = fund["rate"].to_numpy()
        te = pd.DatetimeIndex(d.end[np.maximum(j0 - 1, 0)])
        i_last = ft.searchsorted(te, side="right") - 1
        fs = pd.Series(fr, index=ft)
        m3 = fs.rolling("3D").mean(); m30 = fs.rolling("30D").mean(); s30 = fs.rolling("30D").std()
        ok = i_last >= 0
        il = np.where(ok, i_last, 0)
        out["fund_last_bp"] = np.where(ok, 1e4 * fr[il], np.nan)
        out["fund_3d_bp"] = np.where(ok, 1e4 * m3.to_numpy()[il], np.nan)
        out["fund_z30"] = np.where(ok, (m3.to_numpy()[il] - m30.to_numpy()[il]) / s30.to_numpy()[il], np.nan)
        ent = pd.DatetimeIndex(d.end[je]); ext = pd.DatetimeIndex(d.end[jx])
        a_, b_ = ft.searchsorted(ent, side="right"), ft.searchsorted(ext, side="right")
        Cf = _csum(fr)
        cnt = b_ - a_
        out["fund_during_bp"] = np.where(cnt > 0, 1e4 * (Cf[b_] - Cf[a_]) / np.maximum(cnt, 1), np.nan)
    else:
        for c in ("fund_last_bp", "fund_3d_bp", "fund_z30", "fund_during_bp"):
            out[c] = np.nan
    # ---- engine
    r5 = np.r_[np.nan, np.diff(np.log(d.c))]
    rv24 = np.sqrt(pd.Series(r5 ** 2).rolling(DAY, min_periods=DAY // 2).mean().to_numpy() * DAY)
    out["rv24_over_atr"] = rv24[np.maximum(j0 - 1, 0)] * E / atr
    if fits is not None:
        Vg = engine_variances(d.end, d.c, fits, A)
        from . import vwap_drift as vd
        s1d = np.sqrt(vd.variance_to(np.full(n, 1440.0), np.where(np.isfinite(Vg), Vg, 1.0)))
        s1d[~np.isfinite(Vg).all(axis=1)] = np.nan
        s60 = np.sqrt(Vg[:, 1])
        med60 = pd.Series(s60).rolling(30 * DAY, min_periods=10 * DAY).median().to_numpy()
        jj = np.maximum(j0 - 1, 0)
        out["eng_sig1d_over_atr"] = s1d[jj] * d.c[jj] / atr
        out["eng_sig60_rel30"] = s60[jj] / med60[jj]
        mx = pd.Series(np.abs(r5)).rolling(DAY, min_periods=DAY // 2).max().to_numpy()
        out["jump24_sig"] = mx[jj] / (s1d[jj] / math.sqrt(DAY))
    else:
        for c in ("eng_sig1d_over_atr", "eng_sig60_rel30", "jump24_sig"):
            out[c] = np.nan
    # ---- open interest / positioning
    ent_prev = pd.DatetimeIndex(d.end[np.maximum(j0 - 1, 0)])
    if met is not None and len(met):
        def asof(col, t):
            s = met[col]
            i = s.index.searchsorted(t, side="right") - 1
            v = np.where(i >= 0, s.to_numpy()[np.maximum(i, 0)], np.nan)
            age = np.where(i >= 0, (t - s.index[np.maximum(i, 0)]).total_seconds(), np.inf)
            return np.where(age <= 3600, v, np.nan)
        oi0 = asof("sum_open_interest", ent_prev)
        out["oi_chg_24h"] = np.log(oi0 / asof("sum_open_interest", ent_prev - pd.Timedelta(days=1)))
        out["oi_chg_7d"] = np.log(oi0 / asof("sum_open_interest", ent_prev - pd.Timedelta(days=7)))
        out["oi_over_vol"] = asof("sum_open_interest_value", ent_prev) / _wsum(Q, j0 - DAY, j0)
        out["top_ls"] = asof("sum_toptrader_long_short_ratio", ent_prev)
        out["top_ls_chg24"] = out["top_ls"] - asof("sum_toptrader_long_short_ratio", ent_prev - pd.Timedelta(days=1))
        out["global_ls"] = asof("count_long_short_ratio", ent_prev)
        tk = met["sum_taker_long_short_vol_ratio"].rolling("1D").mean()
        i = tk.index.searchsorted(ent_prev, side="right") - 1
        out["taker_ls_24h"] = np.where(i >= 0, tk.to_numpy()[np.maximum(i, 0)], np.nan)
        out["oi_chg_first3d"] = np.log(asof("sum_open_interest", pd.DatetimeIndex(d.end[np.minimum(je + 3 * DAY, jx)])) / oi0)
    else:
        for c in ENTRY_FEATURES["oi"] + ["oi_chg_first3d"]:
            out[c] = np.nan
    # ---- basis (perp / spot - 1)
    if spot is not None and len(spot):
        sp = spot.reindex(d.end).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            b = d.c / sp - 1
        bs = pd.Series(b)
        jj = np.maximum(j0 - 1, 0)
        out["basis_bp"] = 1e4 * b[jj]
        out["basis_24h_bp"] = 1e4 * bs.rolling(DAY, min_periods=DAY // 2).mean().to_numpy()[jj]
        m7 = bs.rolling(7 * DAY, min_periods=3 * DAY).mean().to_numpy(); s7 = bs.rolling(7 * DAY, min_periods=3 * DAY).std().to_numpy()
        out["basis_z7d"] = (b[jj] - m7[jj]) / s7[jj]
    else:
        for c in ENTRY_FEATURES["basis"]:
            out[c] = np.nan
    # ---- path (descriptive)
    mfe, mae, tmfe, r1, r3, r5d, t24, t72, v3 = ([] for _ in range(9))
    for ti_, (a, b, e_, ru) in enumerate(zip(je, jx, E, Ru)):
        hh = d.h[a:b + 1]; ll = d.l[a:b + 1]
        mfe.append((np.nanmax(hh) - e_) / ru); mae.append((e_ - np.nanmin(ll)) / ru)
        tmfe.append(int(np.nanargmax(hh)) / DAY)
        for lst, dd in ((r1, 1), (r3, 3), (r5d, 5)):
            j = a + dd * DAY
            lst.append((d.c[j] - e_) / ru if j <= b else np.nan)
        for lst, w in ((t24, DAY), (t72, 3 * DAY)):
            vv = _wsum(V, np.array([a]), np.array([min(a + w, b + 1)]))[0]
            lst.append(_wsum(TB, np.array([a]), np.array([min(a + w, b + 1)]))[0] / vv if vv > 0 else np.nan)
        v3.append(_wsum(V, np.array([a]), np.array([min(a + 3 * DAY, b + 1)]))[0]
                  / max(1, min(3 * DAY, b + 1 - a)) * DAY / max(1e-12, v20[ti_]))
    out["mfe_R"], out["mae_R"], out["days_to_mfe"] = mfe, mae, tmfe
    out["R_1d"], out["R_3d"], out["R_5d"] = r1, r3, r5d
    out["tbr_first24h"], out["tbr_first72h"], out["vol_first3d_rel"] = t24, t72, v3
    out["hold_days"] = (jx - je) / DAY
    out["entry_day"] = pd.DatetimeIndex(d.end[je]).tz_convert(tb.NY).normalize().tz_localize(None)
    return out


def daily_states(A: str, d: tb.AssetData) -> pd.DataFrame:
    """Per completed day: close, prior 20-day high, breakout flag (for breadth and BTC context)."""
    pid = np.arange(len(d.s)) + d.pid0
    t = tb.period_end(pid, "day")
    pc = d.pc
    return pd.DataFrame({"asset": A, "t": t, "close": pc, "hi20": d.hi_n,
                         "in_breakout": np.where(np.isfinite(d.hi_n) & np.isfinite(pc), (pc >= d.hi_n).astype(float), np.nan)})


def add_market_features(F: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    """breadth = share of assets (with data) whose last completed daily close was at/above its prior 20-day high;
    BTC context from the last completed day before entry."""
    S = S.dropna(subset=["in_breakout"])
    br = S.groupby("t")["in_breakout"].mean().sort_index()
    btc = S[S.asset == "BTC"].set_index("t").sort_index()
    btc_r20 = np.log(btc.close / btc.close.shift(20))
    et = pd.DatetimeIndex(F.entry_time)
    def asof(s):
        i = s.index.searchsorted(et, side="right") - 1
        return np.where(i >= 0, s.to_numpy()[np.maximum(i, 0)], np.nan)
    F = F.copy()
    F["breadth"] = asof(br)
    F["btc_ret20_log"] = asof(btc_r20) if len(btc) else np.nan
    F["btc_in_breakout"] = asof(btc.in_breakout) if len(btc) else np.nan
    return F


# ---------------------------------------------------------------- statistics

def auc(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x)
    x, y = x[ok], y[ok].astype(int)
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(x).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cluster_ids(F: pd.DataFrame) -> np.ndarray:
    return pd.DatetimeIndex(F.entry_time).tz_convert("UTC").strftime("%Y-%m").to_numpy()


def boot_auc(x, y, cl, n_boot, seed) -> tuple[float, float, float, float]:
    """AUC with a calendar-month cluster bootstrap (trades in the same month share the market move)."""
    ok = np.isfinite(x)
    x, y, cl = x[ok], y[ok], cl[ok]
    a = auc(x, y)
    u, inv = np.unique(cl, return_inverse=True)
    groups = [np.flatnonzero(inv == g) for g in range(len(u))]
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(u), len(u))])
        bs[i] = auc(x[idx], y[idx])
    bs = bs[np.isfinite(bs)]
    lo, hi = np.quantile(bs, [0.025, 0.975])
    p = float(min(1.0, 2 * min((bs <= 0.5).mean(), (bs >= 0.5).mean()))) if len(bs) else np.nan
    return a, float(lo), float(hi), max(p, 1.0 / max(len(bs), 1))


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float); m = np.isfinite(p).sum()
    q = np.full(len(p), np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    o = idx[np.argsort(p[idx])]
    ranked = p[o] * m / np.arange(1, m + 1)
    q[o] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    return q


def univariate(F: pd.DataFrame, features: list, label: str, fp: FParams, seed0: int = 0) -> pd.DataFrame:
    cl = cluster_ids(F)
    rows = []
    for i, f in enumerate(features):
        x = F[f].to_numpy(float); y = F[label].to_numpy()
        n = int(np.isfinite(x).sum())
        if n < fp.min_feature_n or len(np.unique(x[np.isfinite(x)])) < 2:
            rows.append({"feature": f, "n": n, "auc": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan}); continue
        a, lo, hi, p = boot_auc(x, y, cl, fp.n_boot, fp.seed + seed0 + i)
        ok = np.isfinite(x)
        rho = pd.Series(x[ok]).corr(pd.Series(F.R.to_numpy()[ok]), method="spearman")
        rows.append({"feature": f, "n": n, "auc": a, "lo": lo, "hi": hi, "p": p, "spearman_R": rho,
                     "median_win": float(np.nanmedian(x[F[label].to_numpy() == 1])),
                     "median_loss": float(np.nanmedian(x[F[label].to_numpy() == 0]))})
    U = pd.DataFrame(rows)
    U["q_bh"] = bh(U.p.to_numpy())
    return U


def confirm(Ud: pd.DataFrame, Fc: pd.DataFrame, label: str, fp: FParams) -> pd.DataFrame:
    """Discovery survivors (q < fdr_q) tested on the confirmation set: same direction, one-sided p, Holm."""
    surv = Ud[Ud.q_bh < fp.fdr_q].copy()
    if not len(surv):
        return pd.DataFrame(columns=["feature", "auc_disc", "auc_conf", "p_one_sided", "holm_threshold", "confirmed"])
    cl = cluster_ids(Fc)
    rows = []
    for i, r in enumerate(surv.itertuples()):
        x = Fc[r.feature].to_numpy(float); y = Fc[label].to_numpy()
        if np.isfinite(x).sum() < 30:
            rows.append({"feature": r.feature, "auc_disc": r.auc, "auc_conf": np.nan, "p_one_sided": np.nan}); continue
        a, lo, hi, _ = boot_auc(x, y, cl, fp.n_boot, fp.seed + 500 + i)
        # one-sided p in the discovery direction from the cluster bootstrap
        ok = np.isfinite(x); xx, yy, cc = x[ok], y[ok], cl[ok]
        u, inv = np.unique(cc, return_inverse=True); groups = [np.flatnonzero(inv == g) for g in range(len(u))]
        rng = np.random.default_rng(fp.seed + 900 + i)
        bs = np.empty(fp.n_boot)
        for b in range(fp.n_boot):
            idx = np.concatenate([groups[g] for g in rng.integers(0, len(u), len(u))])
            bs[b] = auc(xx[idx], yy[idx])
        up = r.auc > 0.5
        p1 = float(((bs <= 0.5) if up else (bs >= 0.5)).mean())
        rows.append({"feature": r.feature, "auc_disc": r.auc, "auc_conf": a, "conf_lo": lo, "conf_hi": hi,
                     "same_direction": bool((a > 0.5) == up), "p_one_sided": max(p1, 1.0 / fp.n_boot), "n_conf": int(ok.sum())})
    C = pd.DataFrame(rows).sort_values("p_one_sided").reset_index(drop=True)
    m = len(C)
    C["holm_threshold"] = fp.confirm_alpha / (m - np.arange(m))
    passed = []
    stop = False
    for r in C.itertuples():
        ok = (not stop) and r.same_direction and r.p_one_sided < r.holm_threshold
        if not ok:
            stop = True
        passed.append(ok)
    C["confirmed"] = passed
    return C


# ---- ridge logistic (numpy Newton) on entry features

def _design(F: pd.DataFrame, cols: list, ref: dict | None = None):
    X = F[cols].to_numpy(float)
    if ref is None:
        med = np.nanmedian(X, axis=0)
        Xi = np.where(np.isfinite(X), X, med)
        lo, hi = np.nanquantile(Xi, [0.01, 0.99], axis=0)
        Xi = np.clip(Xi, lo, hi)
        mu, sd = Xi.mean(0), Xi.std(0)
        sd[sd == 0] = 1.0
        ref = {"med": med, "lo": lo, "hi": hi, "mu": mu, "sd": sd}
    Xi = np.where(np.isfinite(X), X, ref["med"])
    Xi = np.clip(Xi, ref["lo"], ref["hi"])
    Z = (Xi - ref["mu"]) / ref["sd"]
    miss = (~np.isfinite(X)).astype(float)
    return np.column_stack([np.ones(len(F)), Z, miss]), ref


def fit_logistic(X: np.ndarray, y: np.ndarray, lam: float, iters: int = 50) -> np.ndarray:
    w = np.zeros(X.shape[1])
    P = np.eye(X.shape[1]) * lam; P[0, 0] = 0.0
    for _ in range(iters):
        z = np.clip(X @ w, -30, 30); pr = 1 / (1 + np.exp(-z))
        g = X.T @ (pr - y) + P @ w
        H = (X * (pr * (1 - pr))[:, None]).T @ X + P
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def model_test(Fd: pd.DataFrame, Fc: pd.DataFrame, cols: list, label: str, fp: FParams) -> dict:
    cols = [c for c in cols if np.isfinite(Fd[c].to_numpy(float)).sum() >= fp.min_feature_n]
    Xd, ref = _design(Fd, cols); Xc, _ = _design(Fc, cols, ref)
    yd = Fd[label].to_numpy(float); yc = Fc[label].to_numpy(float)
    w = fit_logistic(Xd, yd, fp.ridge)
    sc = Xc @ w
    a, lo, hi, _ = boot_auc(sc, yc, cluster_ids(Fc), fp.n_boot, fp.seed + 7)
    rng = np.random.default_rng(fp.seed + 8)
    null = np.array([auc(Xc @ fit_logistic(Xd, rng.permutation(yd), fp.ridge), yc) for _ in range(fp.n_perm)])
    p_perm = float((np.sum(null >= a) + 1) / (len(null) + 1))
    coef = pd.Series(w[1:1 + len(cols)], index=cols).sort_values(key=np.abs, ascending=False)
    passed = bool(lo > 0.5 and p_perm < 0.05)
    return {"features": cols, "auc_discovery_insample": auc(Xd @ w, yd), "auc_confirm": a, "lo": lo, "hi": hi,
            "perm_p": p_perm, "perm_null_q95": float(np.quantile(null, 0.95)), "passed": passed,
            "coef": coef.to_dict(), "scores_confirm": sc}


def tercile_economics(Fc: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    """Reported only: confirmation trades by tercile of the discovery-fitted score."""
    q = pd.qcut(pd.Series(score).rank(method="first"), 3, labels=["low", "mid", "high"])
    g = Fc.assign(tercile=q.to_numpy()).groupby("tercile", observed=True)
    return g.agg(trades=("R", "size"), win_rate=("win", "mean"), mean_R=("R", "mean"), sum_R=("R", "sum"),
                 median_R=("R", "median"), big=("big", "sum"))


def early_path_table(F: pd.DataFrame) -> pd.DataFrame:
    """Reported only: P(final win) by the R reached after 1/3/5 days (trades still open at that day)."""
    rows = []
    for col in ("R_1d", "R_3d", "R_5d"):
        x = F[col]
        b = pd.cut(x, [-np.inf, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, np.inf])
        g = F.assign(bucket=b).dropna(subset=["bucket"]).groupby("bucket", observed=True)
        t = g.agg(trades=("win", "size"), win_rate=("win", "mean"), mean_final_R=("R", "mean")).reset_index()
        t.insert(0, "day", col)
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def path_table(F: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in PATH_FEATURES:
        x = F[f]
        rows.append({"feature": f, "n": int(x.notna().sum()), "median_win": x[F.win == 1].median(),
                     "median_loss": x[F.win == 0].median(), "auc_descriptive": auc(x.to_numpy(float), F.win.to_numpy())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- synthetic flow data for tests

def synthetic_flow(b5: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(b5)
    v = rng.gamma(2.0, 50.0, n)
    r = np.r_[0, np.diff(np.log(b5.close.to_numpy()))]
    tbr = np.clip(0.5 + 20 * r + rng.normal(0, 0.05, n), 0.05, 0.95)
    return pd.DataFrame({"volume": v, "quote_volume": v * b5.close.to_numpy(), "trades": np.maximum(1, v // 3),
                         "taker_buy_volume": v * tbr}, index=b5.index)
