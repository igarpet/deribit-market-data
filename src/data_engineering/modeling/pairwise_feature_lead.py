"""PFR 03: do any catalog features known at bar end t predict the PERPETUAL return AFTER t?

Protocol: discussions/Pairwise_Feature_Regression/007_claude_lead_screen_preregistration.md.

PFR 01 was same-bar (feature_t vs return_t), which cannot measure direction. Here, for every feature X_t in
the PFR 01 catalog (model_feature_frame) and horizon h in {5, 15, 60} min:

    y_{t,h} = 1e4 * sum_{j=1}^{h/5} perp_mid_return_{t+5j}      (bp; all future bars present, no gaps)

Primary statistic: PARTIAL information coefficient (Pearson correlation) of X and y after removing the
current perp return r_t from both (removes bounce / mechanical same-bar links). X is winsorised at
DISCOVERY 0.5/99.5% quantiles. Residualisation coefficients are fitted on DISCOVERY only.

Chronological split by UTC day: DISCOVERY first 60% of days, CONFIRMATION last 40%.
Inference: whole-UTC-day bootstrap of correlations built from per-day moment sums.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

VERSION = "PFR03-0.1.0"
STEP = pd.Timedelta(minutes=5)


@dataclass(frozen=True)
class Specification:
    version: str = VERSION
    assets: tuple = ("BTC", "ETH", "SOL", "XRP")
    horizons_min: tuple = (5, 15, 60)
    target_column: str = "perp_mid_return_5m"   # Deribit perpetual mid, tradeable instrument
    control_feature: str = "perp_ret_5m"       # current-bar perp return, partialled out
    discovery_fraction: float = 0.60
    winsor: tuple = (0.005, 0.995)
    min_obs: int = 500                         # per feature-split
    min_days: int = 6
    bootstrap_reps: int = 2000
    alpha_family: float = 0.05                 # Bonferroni over features x horizons, per asset
    confirm_alpha_one_sided: float = 0.05
    round_trip_cost_bp: float = 12.0           # taker 0.05% + 1 bp slippage per side
    seed: int = 2608
    status: str = "HISTORICAL_EXPLORATORY"


def registration() -> dict:
    return asdict(Specification())


# ---------------------------------------------------------------- targets

def forward_returns(state_time: pd.Series, r5: pd.Series, horizons_min=(5, 15, 60)) -> pd.DataFrame:
    """Forward cumulative return (bp) from the 5m bars strictly after t. NaN if any bar is missing or
    non-finite. Index aligned to the input rows."""
    t = pd.to_datetime(state_time, utc=True)
    s = pd.Series(pd.to_numeric(r5, errors="coerce").to_numpy(float), index=t)
    s = s[~s.index.duplicated()]
    grid = pd.date_range(s.index.min(), s.index.max(), freq=STEP)
    g = s.reindex(grid).to_numpy(float)
    out = {}
    for h in horizons_min:
        k = h // 5
        c = np.r_[0.0, np.cumsum(np.where(np.isfinite(g), g, 0.0))]
        bad = np.r_[0, np.cumsum(~np.isfinite(g))]
        n = len(g)
        idx = np.arange(n)
        hi = idx + k
        ok = hi < n
        val = np.full(n, np.nan)
        # sum of g[i+1 .. i+k] = c[i+k+1] - c[i+1]
        val[ok] = c[hi[ok] + 1] - c[idx[ok] + 1]
        nbad = np.full(n, 1)
        nbad[ok] = bad[hi[ok] + 1] - bad[idx[ok] + 1]
        val[nbad > 0] = np.nan
        out[f"y_{h}"] = pd.Series(1e4 * val, index=grid).reindex(t).to_numpy()
    return pd.DataFrame(out, index=state_time.index)


# ---------------------------------------------------------------- moments and bootstrap

MOMENTS = ("n", "sx", "sy", "sxx", "syy", "sxy")


def day_moments(x: np.ndarray, y: np.ndarray, day: np.ndarray, n_days: int) -> dict:
    """x: (N,F), y: (N,) -> per-day moment sums, each (D,F), using pairwise-complete rows per feature."""
    m = np.isfinite(x) & np.isfinite(y)[:, None]
    xv = np.where(m, x, 0.0)
    yv = np.where(m, y[:, None], 0.0)
    out = {}
    for name, v in (("n", m.astype(float)), ("sx", xv), ("sy", yv), ("sxx", xv * xv),
                    ("syy", yv * yv), ("sxy", xv * yv)):
        a = np.zeros((n_days, x.shape[1]))
        np.add.at(a, day, v)
        out[name] = a
    return out


def corr_from_sums(s: dict) -> np.ndarray:
    n = s["n"]
    with np.errstate(divide="ignore", invalid="ignore"):
        cxy = s["sxy"] - s["sx"] * s["sy"] / n
        cxx = s["sxx"] - s["sx"] ** 2 / n
        cyy = s["syy"] - s["sy"] ** 2 / n
        r = cxy / np.sqrt(cxx * cyy)
    return np.where((n >= 3) & (cxx > 0) & (cyy > 0), r, np.nan)


def bootstrap_corr(mom: dict, reps: int, seed: int) -> np.ndarray:
    """(reps, F) correlations from resampled whole days."""
    d = mom["n"].shape[0]
    rng = np.random.default_rng(seed)
    w = np.zeros((reps, d))
    draws = rng.integers(0, d, size=(reps, d))
    np.add.at(w, (np.repeat(np.arange(reps), d), draws.ravel()), 1.0)
    return corr_from_sums({k: w @ v for k, v in mom.items()})


# ---------------------------------------------------------------- preprocessing (discovery-fitted)

def winsor_bounds(x: np.ndarray, fit: np.ndarray, q=(0.005, 0.995)) -> tuple[np.ndarray, np.ndarray]:
    xf = np.where(fit[:, None], x, np.nan)
    with np.errstate(all="ignore"):
        lo, hi = np.nanquantile(xf, q[0], axis=0), np.nanquantile(xf, q[1], axis=0)
    return lo, hi


def residualize(v: np.ndarray, c: np.ndarray, fit: np.ndarray) -> np.ndarray:
    """Column-wise v - a - b*c with (a, b) fitted on rows fit & finite(v, c). v: (N,) or (N,F)."""
    two = v.ndim == 2
    V = v if two else v[:, None]
    out = np.full_like(V, np.nan)
    for j in range(V.shape[1]):
        m = fit & np.isfinite(V[:, j]) & np.isfinite(c)
        if m.sum() < 10 or np.var(c[m]) == 0:
            continue
        b = np.cov(c[m], V[m, j], bias=True)[0, 1] / np.var(c[m])
        a = V[m, j].mean() - b * c[m].mean()
        out[:, j] = np.where(np.isfinite(c), V[:, j] - a - b * c, np.nan)
    return out if two else out[:, 0]


def decile_spread(x: np.ndarray, y: np.ndarray, x_fit: np.ndarray, sign: float) -> dict:
    """Mean forward return (bp) when X is in the DISCOVERY top vs bottom decile, signed by discovery sign.
    Evaluated on the rows passed (x, y)."""
    xf = x_fit[np.isfinite(x_fit)]
    if len(xf) < 50:
        return {"top_bp": np.nan, "bottom_bp": np.nan, "long_short_bp": np.nan, "n_top": 0, "n_bottom": 0}
    lo, hi = np.quantile(xf, [0.1, 0.9])
    m = np.isfinite(x) & np.isfinite(y)
    top, bot = m & (x >= hi), m & (x <= lo)
    t = float(y[top].mean()) if top.any() else np.nan
    b = float(y[bot].mean()) if bot.any() else np.nan
    return {"top_bp": t, "bottom_bp": b, "long_short_bp": sign * (t - b) / 2.0,
            "n_top": int(top.sum()), "n_bottom": int(bot.sum())}


# ---------------------------------------------------------------- one asset

def screen_asset(features: pd.DataFrame, state_time: pd.Series, target_r5: pd.Series,
                 spec: Specification | None = None) -> tuple[pd.DataFrame, dict]:
    """Long table: one row per (feature, horizon) with discovery and confirmation statistics."""
    spec = spec or Specification()
    st = pd.to_datetime(state_time, utc=True).reset_index(drop=True)
    feats = features.reset_index(drop=True)
    names = [c for c in feats.columns if c != spec.control_feature]
    X = feats[names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    c = pd.to_numeric(feats[spec.control_feature], errors="coerce").to_numpy(float) \
        if spec.control_feature in feats else np.zeros(len(feats))
    Y = forward_returns(st, target_r5.reset_index(drop=True), spec.horizons_min)

    day = st.dt.floor("D")
    days = np.array(sorted(day.unique()))
    n_disc = int(np.floor(spec.discovery_fraction * len(days)))
    disc_days, conf_days = days[:n_disc], days[n_disc:]
    is_disc = day.isin(disc_days).to_numpy()
    is_conf = day.isin(conf_days).to_numpy()
    lo, hi = winsor_bounds(X, is_disc, spec.winsor)
    Xw = np.clip(X, lo, hi)
    Xr = residualize(Xw, c, is_disc)
    n_tests = len(names) * len(spec.horizons_min)
    a2 = spec.alpha_family / n_tests / 2
    rows = []
    for h in spec.horizons_min:
        y = Y[f"y_{h}"].to_numpy(float)
        yr = residualize(y, c, is_disc)
        res = {}
        for part, mask, part_days in (("disc", is_disc, disc_days), ("conf", is_conf, conf_days)):
            dcode = np.searchsorted(part_days, day[mask].to_numpy())
            for kind, xx, yy in (("raw", Xw, y), ("partial", Xr, yr)):
                mom = day_moments(xx[mask], yy[mask], dcode, len(part_days))
                tot = {k: v.sum(0, keepdims=True) for k, v in mom.items()}
                r = corr_from_sums(tot)[0]
                bs = bootstrap_corr(mom, spec.bootstrap_reps, spec.seed + h)
                res[(part, kind)] = (r, bs, tot["n"][0], (mom["n"] > 0).sum(0))
        for j, f in enumerate(names):
            row = {"feature": f, "horizon_min": h}
            for (part, kind), (r, bs, n, nd) in res.items():
                b = bs[:, j]
                row[f"{part}_{kind}_ic"] = r[j]
                row[f"{part}_{kind}_n"] = int(n[j])
                row[f"{part}_{kind}_days"] = int(nd[j])
                if kind == "partial":
                    lo_b = np.nanquantile(b, a2) if part == "disc" else np.nanquantile(b, spec.confirm_alpha_one_sided)
                    hi_b = np.nanquantile(b, 1 - a2) if part == "disc" else np.nanquantile(b, 1 - spec.confirm_alpha_one_sided)
                    row[f"{part}_partial_lo"], row[f"{part}_partial_hi"] = lo_b, hi_b
            enough = (row["disc_partial_n"] >= spec.min_obs and row["conf_partial_n"] >= spec.min_obs and
                      row["disc_partial_days"] >= spec.min_days and row["conf_partial_days"] >= spec.min_days)
            disc_pass = enough and (row["disc_partial_lo"] > 0 or row["disc_partial_hi"] < 0)
            sign = float(np.sign(row["disc_partial_ic"])) if np.isfinite(row["disc_partial_ic"]) else 0.0
            conf_pass = disc_pass and ((sign > 0 and row["conf_partial_lo"] > 0) or (sign < 0 and row["conf_partial_hi"] < 0))
            ds = decile_spread(Xw[is_conf, j], y[is_conf], Xw[is_disc, j], sign if sign else 1.0)
            row.update({"enough_data": bool(enough), "discovery_pass": bool(disc_pass), "confirmed": bool(conf_pass),
                        **{f"conf_{k}": v for k, v in ds.items()},
                        "covers_cost": bool(conf_pass and np.isfinite(ds["long_short_bp"]) and
                                            ds["long_short_bp"] > spec.round_trip_cost_bp)})
            rows.append(row)
    meta = {"days": len(days), "discovery_days": [str(d.date()) for d in (disc_days[0], disc_days[-1])] if n_disc else [],
            "confirmation_days": [str(d.date()) for d in (conf_days[0], conf_days[-1])] if len(conf_days) else [],
            "n_features": len(names), "tests_per_asset": n_tests, "discovery_two_sided_alpha_per_test": 2 * a2}
    return pd.DataFrame(rows), meta


def summary(table: pd.DataFrame, spec: Specification | None = None) -> pd.DataFrame:
    """Features confirmed (same sign) in >= 2 assets for the same horizon."""
    spec = spec or Specification()
    c = table[table.confirmed].copy()
    if c.empty:
        return pd.DataFrame(columns=["feature", "horizon_min", "assets", "signs", "mean_conf_partial_ic",
                                     "mean_long_short_bp", "any_covers_cost"])
    c["sign"] = np.sign(c.disc_partial_ic)
    g = c.groupby(["feature", "horizon_min"])
    out = g.agg(assets=("asset", lambda s: ",".join(sorted(s))), signs=("sign", lambda s: ",".join(f"{int(v):+d}" for v in s)),
                n_assets=("asset", "size"), mean_conf_partial_ic=("conf_partial_ic", "mean"),
                mean_long_short_bp=("conf_long_short_bp", "mean"), any_covers_cost=("covers_cost", "any")).reset_index()
    out["same_sign"] = out.signs.map(lambda s: len(set(s.split(","))) == 1)
    return out.sort_values(["n_assets", "mean_conf_partial_ic"], ascending=[False, False])


def decision(table: pd.DataFrame, spec: Specification | None = None, *, synthetic=False) -> dict:
    spec = spec or Specification()
    if synthetic:
        return {"verdict": "SYNTHETIC_NO_MARKET_INFERENCE"}
    s = summary(table, spec)
    multi = s[(s.n_assets >= 2) & s.same_sign] if len(s) else s
    worth = multi[multi.assets.str.contains("BTC|ETH")] if len(multi) else multi
    tradeable = table[table.covers_cost]
    return {"verdict": ("DIRECTIONAL_CANDIDATES_TRADEABLE" if len(tradeable) and len(worth) else
                        "DIRECTIONAL_CANDIDATES_BELOW_COST" if len(worth) else
                        "SINGLE_ASSET_ONLY" if table.confirmed.any() else "NO_DIRECTIONAL_FEATURE"),
            "n_confirmed_cells": int(table.confirmed.sum()),
            "worth_looking_at": worth[["feature", "horizon_min", "assets", "mean_conf_partial_ic", "mean_long_short_bp"]]
                                .to_dict("records") if len(worth) else [],
            "covers_cost_cells": tradeable[["asset", "feature", "horizon_min", "conf_long_short_bp"]].to_dict("records"),
            "status": spec.status, "prospective_confirmation": False}


# ---------------------------------------------------------------- synthetic

def synthetic_states(n_days: int = 30, seed: int = 0, beta_lead: float = 0.0, beta_same: float = 0.8,
                     bounce: float = 0.0):
    """Two features: 'lead' predicts the next perp return with strength beta_lead; 'same' is correlated
    only with the CURRENT return (mechanical). Plus noise features."""
    rng = np.random.default_rng(seed)
    n = n_days * 288
    t = pd.Series(pd.date_range("2026-09-01", periods=n, freq="5min", tz="UTC"))
    lead = rng.standard_normal(n)
    r = 8e-4 * rng.standard_normal(n)
    r[1:] += beta_lead * 8e-4 * lead[:-1]
    r[1:] += bounce * r[:-1]                    # first-order reversal (bid-ask bounce) if bounce < 0
    same = beta_same * r / r.std() + np.sqrt(1 - beta_same ** 2) * rng.standard_normal(n)
    f = pd.DataFrame({"lead": lead, "same": same, "noise1": rng.standard_normal(n),
                      "noise2": rng.standard_normal(n), "perp_ret_5m": r})
    drop = rng.random(n) < 0.01
    r_obs = pd.Series(np.where(drop, np.nan, r))
    return f, t, r_obs
