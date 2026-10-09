"""Volatility Regression 04/05: pre-registered prospective confirmation.

Pre-registered in VR 011 (approved by Ivan 2026-10-09). Everything below is
fixed BEFORE any confirmation session is scored. Do not edit these constants
after the frozen spec (04_frozen_spec.json) is committed; a change means a new
version and a new, later confirmation window.

Question: on sessions never inspected before, does adding ATM implied-vol
information (B0_ATM) to the past-volatility + clock regression (B0) lower
next-5m log-volatility forecast error?

Protocol:
  * Coefficients are fitted ONCE on expiries <= TRAIN_LAST_EXPIRY and frozen
    (no refitting during confirmation).
  * Feature admission (>= MIN_COVERAGE) is computed on TRAINING rows only.
  * Primary: Parkinson target, assets BTC/ETH/SOL (no zero-range bars).
    Secondary (reported, never decisive): index |r| target, other assets,
    mid-path RV target (10 s mid grid; trained on few sessions).
  * Comparison rows: paired-valid rows complete for BOTH models.
  * A session counts if it has >= MIN_SESSION_ROWS comparison rows.
  * Binding readout once >= READOUT_SESSIONS counted sessions exist.
  * Asset passes if B0_ATM has lower session log-RMSE in >= WIN_SHARE of
    counted sessions AND lower pooled log-RMSE. Confirmed if >= ASSETS_TO_PASS
    of the primary assets pass.
  * No deletion, no feature selection on confirmation data.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from data_engineering.modeling import vol_ols_5m as base
from data_engineering.modeling import vol_ols_5m_blocks as vb

VOL_OLS_CONFIRM_VERSION = "1.0.0"
TRAIN_LAST_EXPIRY = "2026-10-08"
CONFIRM_FIRST_EXPIRY = "2026-10-09"
PRIMARY_ASSETS = ("BTC", "ETH", "SOL")
PRIMARY_TARGET = "parkinson"
SECONDARY_TARGETS = ("index_abs", "mid_rv")
MIN_COVERAGE = 0.80
MIN_ROWS_PER_PARAM = 20
MIN_SESSION_ROWS = 100
READOUT_SESSIONS = 20
WIN_SHARE = 2.0 / 3.0
ASSETS_TO_PASS = 2
MID_MIN_INCREMENTS = 26          # of 29 within-bar 10 s increments

ATM_FEATURES = ["log_atm_iv", "atm_iv_d5", "atm_iv_d15", "atm_iv_d30"]
MODELS = {
    "B0": list(vb.BLOCKS["B0_returns_vol_clock"]),
    "B0_ATM": list(vb.BLOCKS["B0_returns_vol_clock"]) + ATM_FEATURES,
}
MID_COLUMNS = ("mid_path_realized_variance", "mid_path_increment_count")


def mid_sigma(g: pd.DataFrame) -> pd.Series:
    """Per-bar mid-path volatility, rescaled to 29 increments; NaN if sparse."""
    rv = pd.to_numeric(g["mid_path_realized_variance"], errors="coerce")
    n = pd.to_numeric(g["mid_path_increment_count"], errors="coerce")
    ok = (n >= MID_MIN_INCREMENTS) & (rv >= 0)
    return np.sqrt(rv * 29.0 / n).where(ok)


def panel(g: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Meta with three targets (next and current bar) and the model features."""
    meta, X = vb.panel(g)
    gs = g.sort_values("state_time").reset_index(drop=True)
    cur = mid_sigma(gs)
    t = pd.to_datetime(gs["state_time"], utc=True)
    link = ((t.shift(-1) - t) == base.BAR) & gs["session_id"].eq(gs["session_id"].shift(-1))
    meta["mid_sigma_curr_5m"] = cur.to_numpy()
    meta["mid_sigma_next_5m"] = cur.shift(-1).where(link).to_numpy()
    return meta, X


TARGET_COLS = {
    "parkinson": ("sigma_next_5m", "sigma_curr_5m"),
    "index_abs": ("index_sigma_next_5m", "index_sigma_curr_5m"),
    "mid_rv": ("mid_sigma_next_5m", "mid_sigma_curr_5m"),
}


def target_valid(meta: pd.DataFrame, target: str) -> np.ndarray:
    """Primary/index targets keep the 01-03 pairing; mid_rv stands alone."""
    if target in ("parkinson", "index_abs"):
        return vb.paired_valid(meta)
    nxt, cur = TARGET_COLS[target]
    return (np.isfinite(meta[nxt].to_numpy(float))
            & np.isfinite(meta[cur].to_numpy(float)))


def freeze_asset(meta: pd.DataFrame, X: pd.DataFrame, target: str) -> dict:
    """Fit both models once on training expiries; return a JSON-safe spec."""
    lab = base.session_labels(meta)
    valid = target_valid(meta, target)
    train = valid & (lab <= TRAIN_LAST_EXPIRY)
    nxt, cur = TARGET_COLS[target]
    sigma = meta[nxt].to_numpy(float)
    eps = base.fit_log_floor(sigma[train])
    y = np.log(sigma + eps)
    out = {"target": target, "epsilon": eps,
           "train_sessions": sorted(set(lab[train])), "models": {}}
    for model, feats in MODELS.items():
        cov = {f: float(X.loc[train, f].notna().mean()) for f in feats}
        admitted = [f for f in feats if cov[f] >= MIN_COVERAGE]
        complete = np.isfinite(X[admitted].to_numpy(float)).all(axis=1)
        rows = train & complete
        names = vb.drop_constant(X, rows, admitted) if rows.sum() > 0 else []
        rec = {"features_requested": feats, "train_coverage": cov,
               "features_admitted": admitted, "features_used": names,
               "n_train": int(rows.sum())}
        if rows.sum() < MIN_ROWS_PER_PARAM * (len(names) + 1):
            rec["fitted"] = False
            rec["reason"] = f"budget: n_train={int(rows.sum())} < {MIN_ROWS_PER_PARAM}*{len(names)+1}"
        else:
            fit = base.fit_ols(X, y, rows, names)
            rec.update({"fitted": True, "center": fit.center.tolist(),
                        "scale": fit.scale.tolist(), "beta": fit.beta.tolist(),
                        "train_mean": float(np.mean(y[rows])),
                        "condition_number": fit.condition_number})
        out["models"][model] = rec
    return out


def predict(rec: dict, X: pd.DataFrame) -> np.ndarray:
    names = rec["features_used"]
    z = (X[names].to_numpy(float) - np.array(rec["center"])) / np.array(rec["scale"])
    beta = np.array(rec["beta"])
    return beta[0] + z @ beta[1:]


def score_asset(meta: pd.DataFrame, X: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Per-session scores of the frozen models on confirmation sessions.

    Rows: target-valid, confirmation expiries, complete for BOTH fitted models.
    """
    target = spec["target"]
    lab = base.session_labels(meta)
    valid = target_valid(meta, target) & (lab >= CONFIRM_FIRST_EXPIRY)
    nxt, cur = TARGET_COLS[target]
    eps = spec["epsilon"]
    y = np.log(meta[nxt].to_numpy(float) + eps)
    pers = np.log(meta[cur].to_numpy(float) + eps)
    m = spec["models"]
    if not all(m[k].get("fitted") for k in MODELS):
        return pd.DataFrame()
    rows = valid & np.isfinite(pers)
    for k in MODELS:
        rows &= np.isfinite(X[m[k]["features_used"]].to_numpy(float)).all(axis=1)
    idx = np.flatnonzero(rows)
    preds = {k: predict(m[k], X.iloc[idx]) for k in MODELS}
    preds["persistence"] = pers[idx]
    preds["train_mean"] = np.full(len(idx), m["B0"]["train_mean"])
    out = []
    for s in sorted(set(lab[idx])):
        at = lab[idx] == s
        rec = {"session": s, "n": int(at.sum()), "counted": bool(at.sum() >= MIN_SESSION_ROWS)}
        for k, p in preds.items():
            rec[f"{k}_log_rmse"] = float(np.sqrt(np.mean((y[idx][at] - p[at]) ** 2)))
        out.append(rec)
    df = pd.DataFrame(out)
    if len(df):
        df.attrs["pooled"] = {k: float(np.sqrt(np.mean((y[idx] - p) ** 2)))
                              for k, p in preds.items()}
        df.attrs["n_pooled"] = int(len(idx))
    return df


def asset_verdict(sessions: pd.DataFrame) -> dict:
    """Apply the pre-registered rule to one asset's session table."""
    if sessions.empty:
        return {"status": "no_data"}
    c = sessions[sessions["counted"]]
    wins = int((c["B0_ATM_log_rmse"] < c["B0_log_rmse"]).sum())
    pooled = sessions.attrs.get("pooled", {})
    rec = {"counted_sessions": int(len(c)), "atm_wins": wins,
           "win_share": wins / len(c) if len(c) else np.nan,
           "pooled_B0": pooled.get("B0"), "pooled_B0_ATM": pooled.get("B0_ATM"),
           "pooled_persistence": pooled.get("persistence"),
           "pooled_train_mean": pooled.get("train_mean")}
    if len(c) < READOUT_SESSIONS:
        rec["status"] = f"interim ({len(c)}/{READOUT_SESSIONS} sessions; non-binding)"
        return rec
    passed = (wins / len(c) >= WIN_SHARE) and (rec["pooled_B0_ATM"] < rec["pooled_B0"])
    rec["status"] = "pass" if passed else "fail"
    return rec


def overall_verdict(primary: dict[str, dict]) -> str:
    st = [v.get("status", "") for v in primary.values()]
    if any(not s in ("pass", "fail") for s in st):
        return "interim (non-binding)"
    return "confirmed" if sum(s == "pass" for s in st) >= ASSETS_TO_PASS else "not confirmed"


def registration() -> dict:
    """The pre-registered protocol, serialised into the frozen spec."""
    return {"version": VOL_OLS_CONFIRM_VERSION,
            "train_last_expiry": TRAIN_LAST_EXPIRY,
            "confirm_first_expiry": CONFIRM_FIRST_EXPIRY,
            "primary_assets": list(PRIMARY_ASSETS), "primary_target": PRIMARY_TARGET,
            "secondary_targets": list(SECONDARY_TARGETS),
            "models": MODELS, "min_coverage_train": MIN_COVERAGE,
            "min_rows_per_param": MIN_ROWS_PER_PARAM,
            "min_session_rows": MIN_SESSION_ROWS, "readout_sessions": READOUT_SESSIONS,
            "win_share": WIN_SHARE, "assets_to_pass": ASSETS_TO_PASS,
            "mid_min_increments": MID_MIN_INCREMENTS,
            "rule": ("asset passes if B0_ATM session log-RMSE < B0 in >= 2/3 of counted "
                     "sessions AND pooled log-RMSE lower; confirmed if >= 2 of BTC/ETH/SOL "
                     "pass on the Parkinson target; binding at >= 20 counted sessions")}


def dumps(obj) -> str:
    return json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
