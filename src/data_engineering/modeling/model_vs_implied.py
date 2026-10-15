"""VD19: does the AM-T-HAR-Blend variance level know something the 0DTE option price does not?

Protocol: discussions/Volatility_Dynamics/037_claude_vd19_model_vs_implied_preregistration.md.

Model side. The level of the VD18 AM-T-HAR-Blend forecast is the HAR_4 net-return variance (blend weight
w ~ 0.98 at 60-240 min; the Markov part only shapes the mixture). We refit VD18's HAR_4 exactly as VD18 does
(same frame, TRAIN split, ridge, net-return scale; data strictly before 2026-09-19) for the horizons that
match an option entry, and predict at collector-era entry times (all after the fit sample):

    entry 04:00 UTC -> 240 min to the 08:00 expiry   (PRIMARY: a native VD18 horizon)
    entry 06:00 UTC -> 120 min                        (native VD18 horizon)
    entry 12:00 UTC -> 1200 min                       (HAR_4 refit at 1200 min; extension, secondary)

Option side. The ATM straddle of the primary family at that entry (VRP 02 / RS 04 code): implied s.d. of
the log move to expiry sigma_imp = IV_straddle_mid * sqrt(T); realised move |log(S_T / F)|; R_net at the ask
plus fees.

    x    = log(sigma_model / sigma_imp)                  model-minus-market, known at entry
    y_rv = log(RV_(t,T] / sigma_imp^2)                   PRIMARY outcome: realised path variance (Binance 5m)
                                                          over implied variance; far less noisy than one |move|
    y    = log(|log(S_T/F)| / (sigma_imp * sqrt(2/pi)))  secondary: the settlement move the straddle pays on
    beta = OLS slope of y_rv on x, 95% session bootstrap.  beta > 0: the model adds information.

Encompassing (reported): log|move| = a + b_imp log sigma_imp + b_mod log sigma_model, session bootstrap.
Economics: mean R_net of straddles bought only when x >= log(1.25), session bootstrap.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

VERSION = "VD19-0.1.1"   # 0.1.1: UTC session fix (06/12 entries), Deribit-index RV diagnostic
ENTRY_HORIZON = {4: 240, 6: 120, 12: 1200}    # UTC entry hour -> minutes to 08:00 expiry
PRIMARY_ENTRY = 4
HAR_FAMILY = "HAR_4"                           # VD18's late-VALID choice at 120 and 240 min
FIT_ASSETS = ("AVAX", "BTC", "ETH", "HYPE", "SOL", "TRX", "XRP")   # VD18 fit universe (asset fixed effects)
TEST_ASSETS = ("BTC", "ETH", "SOL", "XRP")


@dataclass(frozen=True)
class Params:
    version: str = VERSION
    entry_horizon: tuple = tuple(ENTRY_HORIZON.items())
    primary_entry_utc_hour: int = PRIMARY_ENTRY
    har_family: str = HAR_FAMILY
    fit_sample_end: str = "2026-09-19T00:00:00Z"      # VD18 SAMPLE_END; collector options start after it
    forecast_end: str = "2026-10-15T00:00:00Z"
    trade_threshold_ratio: float = 1.25              # buy only when sigma_model >= 1.25 sigma_imp
    tte_tolerance_min: float = 10.0
    n_boot: int = 4000
    seed: int = 2609
    status: str = "HISTORICAL_EXPLORATORY_SMALL_SAMPLE"


def registration(p: Params | None = None) -> dict:
    return asdict(p or Params())


# ---------------------------------------------------------------- model side

def build_frame(raw: pd.DataFrame, horizons, v14, v18) -> pd.DataFrame:
    """VD18 frame: panelize -> EMA features -> forward targets -> HAR lags (VD14/VD18 code, unchanged)."""
    panel = v14.panelize(raw)
    feats = v14.make_features(panel)
    frame = v14.make_targets(panel, feats, tuple(horizons))
    return v18.add_har_lags(frame)


def fit_har_models(frame: pd.DataFrame, horizons, v18, family: str = HAR_FAMILY) -> dict:
    fits = {}
    for h in horizons:
        tr, _, _ = v18.complete(frame, h)
        fits[h] = v18.fit_har(tr, h, v18.FEATURE_SETS[family])
    return fits


def forecast_panel(raw: pd.DataFrame, v14, v18) -> pd.DataFrame:
    """HAR lag features on the dense 5m grid through the forecast period (no targets needed)."""
    p = v14.panelize(raw)
    return v18.add_har_lags(p)[["state_time", "underlying", "slot", "return_5m",
                                *[f"har_logrv_{k}" for k in v18.ALL_FEATURES]]]


def index_panel(idx: pd.DataFrame) -> pd.DataFrame:
    """Dense 5m panel of Deribit index log returns (columns state_time, underlying, return_5m) from
    time_series_feature_state.futures_index_price. Missing bars stay NaN (break the RV window)."""
    parts = []
    for a, g in idx.groupby("underlying"):
        z = g[["state_time", "futures_index_price"]].copy()
        z["state_time"] = pd.to_datetime(z["state_time"], utc=True)
        z = z.drop_duplicates("state_time").set_index("state_time").sort_index()
        grid = pd.date_range(z.index.min(), z.index.max(), freq="5min")
        pz = pd.to_numeric(z["futures_index_price"], errors="coerce").reindex(grid)
        r = np.log(pz.where(pz > 0)).diff()
        parts.append(pd.DataFrame({"state_time": grid, "underlying": a, "return_5m": r.to_numpy(float)}))
    return pd.concat(parts, ignore_index=True)


def realised_variance(panel: pd.DataFrame, origins: pd.DataFrame) -> pd.Series:
    """Sum of squared 5m returns over (t, t+h]; NaN unless every bar is present."""
    out = np.full(len(origins), np.nan)
    for a, g in panel.groupby("underlying"):
        g = g.sort_values("state_time")
        t = g["state_time"].to_numpy()
        r = g["return_5m"].to_numpy(float)
        cv = np.r_[0.0, np.cumsum(np.where(np.isfinite(r), r * r, 0.0))]
        cn = np.r_[0, np.cumsum(np.isfinite(r))]
        o = origins[origins["underlying"] == a]
        i0 = np.searchsorted(t, pd.to_datetime(o["state_time"], utc=True).to_numpy(), "right")
        k = (o["h"].to_numpy(int) // 5)
        i1 = i0 + k
        ok = i1 <= len(r)
        v = np.full(len(o), np.nan)
        v[ok] = np.where(cn[i1[ok]] - cn[i0[ok]] == k[ok], cv[i1[ok]] - cv[i0[ok]], np.nan)
        out[np.flatnonzero((origins["underlying"] == a).to_numpy())] = v
    return pd.Series(out, index=origins.index)


def model_sigma(panel: pd.DataFrame, origins: pd.DataFrame, fits: dict, v18) -> pd.Series:
    """sigma_model (log-return s.d. to expiry) for origins with columns underlying, state_time, h.
    The origin bar is the 5m bar ENDING at the entry time (price_5m state_time = bar end): no lookahead."""
    o = origins[["underlying", "state_time", "h"]].copy()
    o["state_time"] = pd.to_datetime(o["state_time"], utc=True)
    m = o.merge(panel, on=["underlying", "state_time"], how="left")
    out = np.full(len(m), np.nan)
    cols = [f"har_logrv_{k}" for k in v18.ALL_FEATURES]
    for h, fit in fits.items():
        ix = np.flatnonzero((m["h"] == h).to_numpy() & np.isfinite(m[cols].to_numpy(float)).all(axis=1)
                            & m["underlying"].isin(fit["net_return_scales"]).to_numpy())
        if len(ix):
            out[ix] = np.sqrt(v18.predict_har(m.iloc[ix], fit))
    return pd.Series(out, index=origins.index)


# ---------------------------------------------------------------- comparison

def entries(straddles: pd.DataFrame, p: Params | None = None) -> pd.DataFrame:
    """Straddles at the registered entry hours whose time to expiry matches the model horizon."""
    p = p or Params()
    s = straddles.copy()
    s["entry_hour"] = s["state_time"].dt.hour
    s = s[(s["state_time"].dt.minute == 0) & s["entry_hour"].isin(ENTRY_HORIZON)]
    s["h"] = s["entry_hour"].map(ENTRY_HORIZON)
    s = s[np.abs(s["minutes_to_expiry"] - s["h"]) <= p.tte_tolerance_min]
    return s.drop_duplicates(["session", "underlying", "entry_hour"]).reset_index(drop=True)


def comparison_frame(e: pd.DataFrame, sigma_model: pd.Series, rv: pd.Series | None = None,
                     rv_index: pd.Series | None = None) -> pd.DataFrame:
    d = e.copy()
    d["rv_index"] = rv_index.to_numpy(float) if rv_index is not None else np.nan
    d["sigma_model"] = sigma_model.to_numpy(float)
    d["rv_path"] = rv.to_numpy(float) if rv is not None else np.nan
    d["sigma_imp"] = d["iv_straddle_mid"].to_numpy(float) * np.sqrt(d["T"].to_numpy(float))
    move = np.abs(np.log(d["S_T"].to_numpy(float) / d["F"].to_numpy(float)))
    with np.errstate(divide="ignore", invalid="ignore"):
        d["x"] = np.log(d["sigma_model"] / d["sigma_imp"])
        d["log_move"] = np.log(move)
        d["y"] = np.log(move / (d["sigma_imp"] * np.sqrt(2 / np.pi)))
        d["y_model"] = np.log(move / (d["sigma_model"] * np.sqrt(2 / np.pi)))
        d["y_rv"] = np.log(d["rv_path"] / d["sigma_imp"] ** 2)
        d["y_rv_index"] = np.log(d["rv_index"] / d["sigma_imp"] ** 2)
        d["log_rv"] = np.log(d["rv_path"])
        d["log_sigma_imp"] = np.log(d["sigma_imp"])
        d["log_sigma_model"] = np.log(d["sigma_model"])
    ok = np.isfinite(d["x"]) & np.isfinite(d["y"])
    return d[ok].reset_index(drop=True)


def _ols(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = np.sqrt(w)
    b, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return b


def session_bootstrap(d: pd.DataFrame, cols: list, ycol: str, p: Params | None = None, seed_offset: int = 0) -> dict:
    """OLS of ycol on [1, cols], whole-session bootstrap. Returns estimates and 95% CIs."""
    p = p or Params()
    q = d[np.isfinite(d[ycol]) & np.isfinite(d[cols]).all(axis=1)]
    if q["session"].nunique() < 5:
        return {"n": int(len(q)), "sessions": int(q["session"].nunique())}
    X = np.column_stack([np.ones(len(q)), q[cols].to_numpy(float)])
    y = q[ycol].to_numpy(float)
    u, inv = np.unique(q["session"].to_numpy(), return_inverse=True)
    est = _ols(X, y, np.ones(len(y)))
    rng = np.random.default_rng(p.seed + seed_offset)
    bs = np.array([_ols(X, y, np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv].astype(float))
                   for _ in range(p.n_boot)])
    lo, hi = np.nanquantile(bs, [0.025, 0.975], axis=0)
    out = {"n": int(len(q)), "sessions": int(len(u))}
    for k, name in enumerate(["const", *cols]):
        out[name] = float(est[k]); out[f"{name}_lo"] = float(lo[k]); out[f"{name}_hi"] = float(hi[k])
    return out


def session_mean_boot(d: pd.DataFrame, col: str, p: Params | None = None) -> dict:
    p = p or Params()
    v = d.groupby("session")[col].mean().to_numpy(float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return {"n_sessions": int(len(v)), "n": int(len(d))}
    rng = np.random.default_rng(p.seed)
    b = rng.choice(v, size=(p.n_boot, len(v))).mean(axis=1)
    return {"n_sessions": int(len(v)), "n": int(len(d)), "mean": float(v.mean()),
            "lo": float(np.quantile(b, .025)), "hi": float(np.quantile(b, .975))}


def decision(primary: dict, econ_selected: dict, synthetic=False) -> dict:
    if synthetic:
        return {"verdict": "SYNTHETIC_NO_MARKET_INFERENCE"}
    if "x_lo" not in primary:
        return {"verdict": "INSUFFICIENT_SESSIONS", "primary": primary}
    info = primary["x_lo"] > 0
    neg = primary["x_hi"] < 0
    pays = "lo" in econ_selected and econ_selected["lo"] > 0
    v = ("MODEL_EDGE_CANDIDATE" if info and pays else "MODEL_ADDS_INFORMATION" if info else
         "MODEL_ANTI_INFORMATIVE" if neg else "NO_EVIDENCE")
    return {"verdict": v, "primary_entry_utc": PRIMARY_ENTRY, "beta": primary.get("x"),
            "beta_ci": [primary.get("x_lo"), primary.get("x_hi")], "selected_trades_R_net": econ_selected,
            "status": Params().status, "prospective_confirmation": False}


# ---------------------------------------------------------------- synthetic

def synthetic_comparison(n_sessions: int = 25, seed: int = 0, model_skill: float = 0.0, assets=TEST_ASSETS):
    """sigma_true varies by session; implied is sigma_true * noise; model is sigma_true^skill * noise.
    model_skill > 0 => the model carries information beyond implied."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sessions):
        for a in assets:
            true = 0.01 * np.exp(0.4 * rng.standard_normal())
            imp = true * np.exp(0.3 * rng.standard_normal()) * 1.1
            mod = imp * np.exp(model_skill * np.log(true / imp) + 0.2 * rng.standard_normal())
            move = abs(true * rng.standard_normal())
            rv = true ** 2 * rng.chisquare(48) / 48                 # 48 five-minute bars
            T = 240 / (365 * 24 * 60)
            prem = imp * np.sqrt(2 / np.pi)
            rows.append({"session": f"2026-09-{s + 1:02d}", "underlying": a, "T": T, "F": 100.0,
                         "S_T": 100.0 * np.exp(move * (1 if rng.random() < .5 else -1)),
                         "iv_straddle_mid": imp / np.sqrt(T), "R_net": move / (prem * 1.15) - 1,
                         "sigma_model_true": mod, "rv_true": rv})
    d = pd.DataFrame(rows)
    return d.drop(columns=["sigma_model_true", "rv_true"]), d["sigma_model_true"], d["rv_true"]
