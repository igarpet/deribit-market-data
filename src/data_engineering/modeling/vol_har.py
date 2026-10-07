"""
Module 12: supervised volatility forecasts (discussions/001-004).

Target: y_{t,h} = log(RV_{t,h} + eps_h) of the Deribit index (vol_targets).

Models (all share the unpenalized HAR + clock block H):

    M0   persistence            y^ = log(RV_back30 + eps) + log(m / 6)
    M1   HAR + clock            OLS on H
    M1c  M1 with a training-only affine recalibration (out-of-fold, CV)
    M2a  M1 + log clock-allocated implied integrated variance
    M2b  M2a + ATM IV changes (15m, 30m)
    M3   M2b + returns, flow, L2, basis/OI, |D|, D, d|D|, perp ranges
    M4   M3 + option shape / dynamics / liquidity

H = [const, l5m, l30m, l1h, l4h, clock_fwd, clock_back, weekend] is never
penalized. The added block X is standardized on training rows and shrunk:

    min (1/n) |y - H phi - X gamma|^2 + lam |gamma|^2

solved by Frisch-Waugh-Lovell: y and X are residualized on H with a QR
projection, the ridge problem is solved on the residuals, and phi is
recovered by least squares on y - X gamma. lam = 0 gives OLS; gamma = 0 is
exactly M1, so every model nests the HAR null.

Every nuisance quantity (clock baseline, eps, standardization, smearing,
residual distribution, lam) is fitted on the training rows of the fold in
which it is used (discussion 004 §6). Forecast levels use Duan smearing:
F = exp(y^) * mean_train(exp(e)).

Scores per row: QLIKE (primary), squared log error, Gaussian and Student-t
log scores. Paired comparisons are session-blocked (session_eval).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from data_engineering.modeling import vol_targets as vt
from data_engineering.modeling.clock_baseline import ClockBaselineSet, FeatureSpec
from data_engineering.modeling.full_information import fit_linear
from data_engineering.modeling.hsmm.session_eval import paired_sessions


VOL_HAR_VERSION = "0.1"

LAMBDA_GRID = (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)
EN_ALPHA_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
EN_L1_RATIO = 0.5
CLIP_Z = 5.0
EPS_QUANTILE = 0.01
T_NU_GRID = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0)

H_NAMES = ["const", "l5m", "l30m", "l1h", "l4h", "clock_fwd", "clock_back", "weekend"]
DELTA_BASE = {"30m": "30m", "60m": "1h"}
CLOCK_SPEC = FeatureSpec("return_5m", "return_5m", transform="log_abs", scale="none")   # New York clock

MODEL_ORDER = ["M0", "M1", "M1c", "M2a", "M2b", "M3", "M4"]
HIERARCHY = [("M1", "M0"), ("M2a", "M1"), ("M2b", "M2a"), ("M3", "M2b"), ("M4", "M3")]


# --------------------------------------------------
# Nuisance fit and materialization
# --------------------------------------------------

def _pos_quantile(x: np.ndarray, q: float = EPS_QUANTILE) -> float:
    v = x[np.isfinite(x) & (x > 0)]
    if len(v) == 0:
        raise ValueError("No positive training values to set eps.")
    return float(np.quantile(v, q))


@dataclass
class Nuisance:
    asset: str
    clock: dict
    eps_target: dict
    eps_back: dict
    eps_settle: float
    train_sessions: list

    def clock_set(self) -> ClockBaselineSet:
        return ClockBaselineSet.from_dict(self.clock)

    def to_dict(self) -> dict:
        return asdict(self)


def fit_nuisance(g: pd.DataFrame, panel: vt.Panel, train_sessions: set) -> Nuisance:
    """Clock baseline (New York) and eps values from training sessions only."""
    tr = np.isin(panel.session_date, sorted(train_sessions))
    frame = g[["state_id", "state_time", "underlying", "is_weekend", "return_5m"]].reset_index(drop=True)
    cbs = ClockBaselineSet.fit(frame, pd.Series(tr), specs=(CLOCK_SPEC,))
    if not cbs.fitted:
        raise RuntimeError(f"{panel.asset}: clock baseline could not be fitted ({cbs.skipped}).")
    return Nuisance(
        asset=panel.asset,
        clock=cbs.to_dict(),
        eps_target={h: _pos_quantile(panel.fwd_rv[h][tr]) for h in vt.HORIZONS},
        eps_back={w: _pos_quantile(panel.back_rv[w][tr]) for w in vt.BACK_WINDOWS},
        eps_settle=_pos_quantile(panel.settle_rv[tr]) if np.isfinite(panel.settle_rv[tr]).any() else np.nan,
        train_sessions=sorted(train_sessions),
    )


@dataclass
class Material:
    """Model inputs for one asset under one nuisance fit."""

    asset: str
    sessions: np.ndarray
    tte_min: np.ndarray
    y: dict             # horizon -> log target
    level: dict         # horizon -> RV + eps (QLIKE scale)
    delta_base: dict    # horizon -> log backward RV of matching length (+ eps_h)
    persistence: dict   # horizon -> M0 log forecast (before smearing)
    H: dict             # horizon -> DataFrame (H_NAMES)
    X: dict             # horizon -> DataFrame of candidate added features
    implied: dict       # W_imp_total, W_imp_alloc_*, W_imp_settle
    settle: pd.DataFrame


def materialize(panel: vt.Panel, nuis: Nuisance, raw_X: pd.DataFrame) -> Material:
    """raw_X: candidate added features aligned with the panel rows, measured
    at the origin row (including |D|, D, d|D| and IV changes). The
    implied-variance regressor is added here because it depends on the
    fitted clock."""
    fb = nuis.clock_set().fitted[0]
    alloc = vt.clock_allocation(panel, fb)
    imp = vt.implied_variance(panel, alloc)
    eb = nuis.eps_back
    lb = {w: np.log(panel.back_rv[w] + eb[w]) for w in vt.BACK_WINDOWS}
    with np.errstate(divide="ignore", invalid="ignore"):
        clock_back = np.log(alloc.back_mean_30m)
        rng_feats = pd.DataFrame({
            "log_park30": np.log(panel.park_back30 + eb["30m"]),
            "log_gk30": np.log(panel.gk_back30 + eb["30m"]),
        })
    y, level, dbase, pers, H, X = {}, {}, {}, {}, {}, {}
    for h, m in vt.HORIZONS.items():
        e = nuis.eps_target[h]
        level[h] = panel.fwd_rv[h] + e
        y[h] = np.log(level[h])
        dbase[h] = np.log(panel.back_rv[DELTA_BASE[h]] + e)
        pers[h] = np.log(panel.back_rv["30m"] + e) + np.log(m / 6.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cf = np.log(alloc.fwd_mean[h])
            lW = np.log(imp[f"W_imp_alloc_{h}"])
        H[h] = pd.DataFrame({
            "const": 1.0, "l5m": lb["5m"], "l30m": lb["30m"], "l1h": lb["1h"], "l4h": lb["4h"],
            "clock_fwd": cf, "clock_back": clock_back, "weekend": panel.weekend.astype(float),
        })
        Xh = raw_X.reset_index(drop=True).copy()
        Xh.insert(0, "log_W_imp_alloc", np.where(np.isfinite(lW), lW, np.nan))
        X[h] = pd.concat([Xh, rng_feats], axis=1)
    settle = pd.DataFrame({
        "rv_settle": panel.settle_rv, "rv_settle_log": panel.settle_rv_log,
        "W_imp_settle": imp["W_imp_settle"], "tte_min": panel.tte_min,
    })
    return Material(panel.asset, panel.session_date, panel.tte_min, y, level, dbase, pers, H, X, imp, settle)


# --------------------------------------------------
# Estimator
# --------------------------------------------------

@dataclass
class FittedVol:
    model: str
    horizon: str
    kind: str                      # persistence | ridge | elasticnet
    h_names: list = field(default_factory=list)
    x_names: list = field(default_factory=list)
    x_mean: list = field(default_factory=list)
    x_scale: list = field(default_factory=list)
    phi: list = field(default_factory=list)
    gamma: list = field(default_factory=list)
    lam: float = 0.0
    l1_ratio: float = 0.0
    smear: float = 1.0
    sigma: float = 1.0
    t_nu: float = 30.0
    t_scale: float = 1.0
    n_train: int = 0
    calib: list | None = None      # [a, b] for M1c
    base: dict | None = None       # M1c: the M1 FittedVol

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FittedVol":
        d = dict(d)
        return cls(**d)


def complete(mat: Material, h: str, x_names: list, rows: np.ndarray | None = None) -> np.ndarray:
    ok = np.isfinite(mat.y[h]) & np.isfinite(mat.H[h][H_NAMES].to_numpy(float)).all(axis=1)
    if x_names:
        ok &= np.isfinite(mat.X[h][x_names].to_numpy(float)).all(axis=1)
    if rows is not None:
        ok &= rows
    return ok


def _t_fit(e: np.ndarray) -> tuple[float, float]:
    """Student-t (location 0) by profile likelihood over T_NU_GRID."""
    from math import lgamma, log, pi
    best = (-np.inf, 30.0, float(np.std(e)) or 1.0)
    for nu in T_NU_GRID:
        s2 = float(np.mean(e ** 2)) or 1e-8
        for _ in range(100):
            w = (nu + 1.0) / (nu + e ** 2 / s2)
            new = float(np.mean(w * e ** 2))
            if abs(new - s2) <= 1e-10 * max(s2, 1e-12):
                s2 = new
                break
            s2 = new
        s = np.sqrt(s2)
        ll = len(e) * (lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * log(nu * pi) - log(s)) \
            - (nu + 1) / 2 * float(np.sum(np.log1p(e ** 2 / (nu * s2))))
        if ll > best[0]:
            best = (ll, nu, s)
    return best[1], best[2]


def _finish(fv: FittedVol, resid: np.ndarray) -> FittedVol:
    fv.smear = float(np.mean(np.exp(resid)))
    fv.sigma = max(float(np.std(resid)), 1e-6)
    fv.t_nu, fv.t_scale = _t_fit(resid)
    fv.n_train = int(len(resid))
    return fv


def fit_vol(mat: Material, h: str, model: str, x_names: list, rows: np.ndarray,
            kind: str = "ridge", lam: float = 0.0, l1_ratio: float = EN_L1_RATIO) -> FittedVol:
    """Fit one model on the complete rows within `rows`."""
    if model == "M0":
        ok = rows & np.isfinite(mat.y[h]) & np.isfinite(mat.persistence[h])
        if ok.sum() < 20:
            raise ValueError(f"{mat.asset} {h} M0: too few rows ({ok.sum()}).")
        return _finish(FittedVol("M0", h, "persistence"), mat.y[h][ok] - mat.persistence[h][ok])

    ok = complete(mat, h, x_names, rows)
    Hdf = mat.H[h].loc[ok, H_NAMES]
    keep = [c for c in H_NAMES if c == "const" or float(Hdf[c].std()) > 1e-12]
    Hm = Hdf[keep].to_numpy(float)
    while np.linalg.matrix_rank(Hm) < Hm.shape[1]:      # drop trailing collinear columns
        keep = keep[:-1]
        Hm = Hdf[keep].to_numpy(float)
    y = mat.y[h][ok]
    n = len(y)
    if n <= Hm.shape[1] + len(x_names) + 5:
        raise ValueError(f"{mat.asset} {h} {model}: {n} complete training rows for {Hm.shape[1] + len(x_names)} terms.")

    fv = FittedVol(model, h, kind, h_names=keep, lam=float(lam), l1_ratio=float(l1_ratio if kind == "elasticnet" else 0.0))
    if x_names:
        Xraw = mat.X[h].loc[ok, x_names].to_numpy(float)
        mu, sd = Xraw.mean(axis=0), Xraw.std(axis=0)
        use = sd > 1e-12
        x_names = [c for c, u in zip(x_names, use) if u]
        mu, sd = mu[use], sd[use]
        Xs = np.clip((Xraw[:, use] - mu) / sd, -CLIP_Z, CLIP_Z)
        Q, _ = np.linalg.qr(Hm)
        yr = y - Q @ (Q.T @ y)
        Xr = Xs - Q @ (Q.T @ Xs)
        if kind == "elasticnet":
            gamma = fit_linear(Xr, yr, x_names, "elasticnet", alpha=lam, l1_ratio=l1_ratio).coef
        elif lam > 0:
            gamma = np.linalg.solve(Xr.T @ Xr / n + lam * np.eye(Xr.shape[1]), Xr.T @ yr / n)
        else:
            gamma, *_ = np.linalg.lstsq(Xr, yr, rcond=None)
        phi, *_ = np.linalg.lstsq(Hm, y - Xs @ gamma, rcond=None)
        fv.x_names, fv.x_mean, fv.x_scale, fv.gamma = list(x_names), mu.tolist(), sd.tolist(), np.asarray(gamma).tolist()
        resid = y - Hm @ phi - Xs @ gamma
    else:
        phi, *_ = np.linalg.lstsq(Hm, y, rcond=None)
        resid = y - Hm @ phi
    fv.phi = np.asarray(phi).tolist()
    return _finish(fv, resid)


def predict_log(fv: FittedVol, mat: Material, h: str) -> np.ndarray:
    """Log forecast for every row (NaN where an input is missing)."""
    if fv.kind == "persistence":
        return mat.persistence[h].copy()
    if fv.calib is not None:
        base = predict_log(FittedVol.from_dict(fv.base), mat, h)
        return fv.calib[0] + fv.calib[1] * base
    Hm = mat.H[h][fv.h_names].to_numpy(float)
    out = Hm @ np.asarray(fv.phi)
    if fv.x_names:
        Xs = np.clip((mat.X[h][fv.x_names].to_numpy(float) - np.asarray(fv.x_mean)) / np.asarray(fv.x_scale), -CLIP_Z, CLIP_Z)
        out = out + Xs @ np.asarray(fv.gamma)
    return out


def _t_logpdf(e: np.ndarray, nu: float, s: float) -> np.ndarray:
    from math import lgamma, log, pi
    c = lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * log(nu * pi) - log(s)
    return c - (nu + 1) / 2 * np.log1p((e / s) ** 2 / nu)


def row_scores(fv: FittedVol, mat: Material, h: str) -> pd.DataFrame:
    """Per-row scores; NaN where the forecast or target is missing."""
    yhat = predict_log(fv, mat, h)
    y = mat.y[h]
    e = y - yhat
    F = np.exp(yhat) * fv.smear
    ratio = mat.level[h] / F
    with np.errstate(invalid="ignore", divide="ignore"):
        qlike = ratio - np.log(ratio) - 1.0
    return pd.DataFrame({
        "yhat": yhat, "y": y, "qlike": qlike, "sqerr": e ** 2,
        "ls_gauss": -0.5 * np.log(2 * np.pi * fv.sigma ** 2) - 0.5 * (e / fv.sigma) ** 2,
        "ls_t": _t_logpdf(e, fv.t_nu, fv.t_scale),
    })


# --------------------------------------------------
# Model specifications
# --------------------------------------------------

M2B_EXTRA = ["atm_iv_d15", "atm_iv_d30"]
M3_EXTRA_FIXED = ["log_range_bps", "log_park30", "log_gk30", "absD", "D", "d_absD"]
M3_BLOCKS = ["returns", "flow", "l2", "basis_oi"]
M4_EXCLUDE = {"log_atm_iv", "log_fit_atm_iv", "atm_iv_d15", "atm_iv_d30"}


def model_specs(blocks: dict, available: set) -> dict:
    """X columns of each model; `blocks` maps full_information block -> eligible
    feature names, `available` the eligible extra columns."""
    def have(cols):
        return [c for c in cols if c in available]
    m2a = have(["log_W_imp_alloc"])
    m2b = m2a + have(M2B_EXTRA)
    m3 = m2b + [c for b in M3_BLOCKS for c in blocks.get(b, [])] + have(M3_EXTRA_FIXED)
    m4 = m3 + [c for c in blocks.get("options", []) if c not in M4_EXCLUDE]
    specs = {"M0": None, "M1": [], "M2a": m2a, "M2b": m2b, "M3": m3, "M4": m4}
    return {k: (list(dict.fromkeys(v)) if v is not None else None) for k, v in specs.items()}


# --------------------------------------------------
# Cross-validation (nuisance refitted per fold)
# --------------------------------------------------

def session_mean(values: np.ndarray, sessions: np.ndarray, rows: np.ndarray) -> pd.Series:
    ok = rows & np.isfinite(values)
    return pd.Series(values[ok]).groupby(sessions[ok]).mean()


def cross_validate(build_material: Callable[[set], Material], folds: list, h: str, specs: dict,
                   lam_grid=LAMBDA_GRID, en_grid=EN_ALPHA_GRID) -> tuple[pd.DataFrame, dict]:
    """Per (model, kind, lam): validation QLIKE by session, plus out-of-fold
    M1 predictions for the recalibration. build_material(train_sessions)
    returns a Material whose nuisance was fitted on those sessions only."""
    recs, oof = [], []
    for k, (tr_s, va_s) in enumerate(folds):
        mat = build_material(tr_s)
        tr = np.isin(mat.sessions, sorted(tr_s))
        va = np.isin(mat.sessions, sorted(va_s))
        for model, cols in specs.items():
            if model == "M1c":
                continue
            grid = [("persistence", 0.0)] if model == "M0" else (
                [("ridge", 0.0)] if not cols else [("ridge", l) for l in lam_grid])
            if model == "M4":
                grid += [("elasticnet", a) for a in en_grid]
            for kind, lam in grid:
                try:
                    fv = fit_vol(mat, h, model, cols or [], tr, kind=kind, lam=lam)
                except ValueError:
                    continue
                sc = row_scores(fv, mat, h)
                rows = va & complete(mat, h, cols or []) if model != "M0" else va
                for s, v in session_mean(sc["qlike"].to_numpy(), mat.sessions, rows).items():
                    recs.append({"fold": k, "model": model, "kind": kind, "lam": lam, "session": s, "qlike": v})
                if model == "M1":
                    r = rows & np.isfinite(sc["yhat"].to_numpy())
                    oof.append(pd.DataFrame({"y": mat.y[h][r], "yhat": sc["yhat"].to_numpy()[r], "session": mat.sessions[r]}))
    return pd.DataFrame(recs), {"oof_m1": pd.concat(oof, ignore_index=True) if oof else pd.DataFrame()}


def one_se_choice(cv: pd.DataFrame, model: str, kind: str = "ridge") -> tuple[float, pd.DataFrame]:
    """Largest penalty whose mean session QLIKE is within one session SE of the best."""
    d = cv[(cv.model == model) & (cv.kind == kind)]
    if d.empty:
        return np.nan, pd.DataFrame()
    tab = d.groupby("lam")["qlike"].agg(["mean", "std", "count"]).reset_index()
    tab["se"] = tab["std"] / np.sqrt(tab["count"].clip(lower=1))
    best = tab.loc[tab["mean"].idxmin()]
    within = tab[tab["mean"] <= best["mean"] + (best["se"] if np.isfinite(best["se"]) else 0.0)]
    return float(within["lam"].max()), tab


def calibrate_m1(m1: FittedVol, oof: pd.DataFrame) -> FittedVol:
    """M1c = a + b * M1 forecast, (a, b) from out-of-fold training forecasts."""
    if len(oof) < 30:
        raise ValueError("Too few out-of-fold rows to recalibrate M1.")
    A = np.column_stack([np.ones(len(oof)), oof["yhat"].to_numpy()])
    ab, *_ = np.linalg.lstsq(A, oof["y"].to_numpy(), rcond=None)
    resid = oof["y"].to_numpy() - A @ ab
    fv = FittedVol("M1c", m1.horizon, "calibrated", calib=ab.tolist(), base=m1.to_dict())
    return _finish(fv, resid)


# --------------------------------------------------
# Evaluation helpers
# --------------------------------------------------

def mz(y: np.ndarray, yhat: np.ndarray, sessions: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """Mincer-Zarnowitz y = a + b yhat with a session bootstrap for b."""
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat, sessions = y[ok], yhat[ok], sessions[ok]
    if len(y) < 10 or np.std(yhat) <= 0:
        return {"mz_a": np.nan, "mz_b": np.nan, "mz_b_lo": np.nan, "mz_b_hi": np.nan}
    A = np.column_stack([np.ones(len(y)), yhat])
    ab = np.linalg.lstsq(A, y, rcond=None)[0]
    us = np.unique(sessions)
    rng = np.random.default_rng(seed)
    idx = {s: np.flatnonzero(sessions == s) for s in us}
    bs = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[s] for s in rng.choice(us, size=len(us))])
        if np.std(yhat[pick]) > 0:
            bs.append(np.linalg.lstsq(A[pick], y[pick], rcond=None)[0][1])
    lo, hi = (np.quantile(bs, [0.025, 0.975]) if bs else (np.nan, np.nan))
    return {"mz_a": float(ab[0]), "mz_b": float(ab[1]), "mz_b_lo": float(lo), "mz_b_hi": float(hi)}


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(yhat)
    if ok.sum() < 3:
        return np.nan
    sst = float(((y[ok] - y[ok].mean()) ** 2).sum())
    return 1.0 - float(((y[ok] - yhat[ok]) ** 2).sum()) / sst if sst > 0 else np.nan


def non_overlapping(tte_min: np.ndarray, h: str) -> np.ndarray:
    """Session-aligned origins every h (from 08:00 UTC)."""
    step = vt.HORIZONS[h] * vt.BAR_MIN
    return np.isclose(np.mod(np.round(tte_min), step), 0.0)


def compare(scores_a: pd.DataFrame, scores_b: pd.DataFrame, sessions: np.ndarray, rows: np.ndarray,
            n_boot: int = 2000) -> dict:
    """Paired session comparison of a vs b on rows where both are scored.
    Positive = a better. QLIKE and squared log error enter with a minus sign."""
    ok = rows & np.isfinite(scores_a["qlike"].to_numpy()) & np.isfinite(scores_b["qlike"].to_numpy())
    out = {"n_rows": int(ok.sum())}
    for name, sign in (("qlike", -1.0), ("sqerr", -1.0), ("ls_t", 1.0)):
        p = paired_sessions(sign * scores_a[name].to_numpy()[ok], sign * scores_b[name].to_numpy()[ok],
                            sessions[ok], n_boot=n_boot)
        for k, v in p.items():
            out[f"{name}_{k}"] = v
    return out
