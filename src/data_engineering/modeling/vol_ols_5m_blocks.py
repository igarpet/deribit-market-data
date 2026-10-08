"""Volatility Regression 03: nested-block next-5m volatility OLS.

Design (VR 007, approved by Ivan 2026-10-09):

* Same two paired targets and causal alignment as `vol_ols_5m` (01).
* Option smile shape / fitted-smile / wing features are EXCLUDED BY DESIGN:
  on a 0DTE chain they do not exist in the hours before expiry, so requiring
  them turns the model into a 09:00-04:00 UTC model.
* Nested feature blocks, each fitted on its own complete rows:
    B0 returns + volatility + clock      (index and trade-range based)
    B1 B0 + trade flow + basis / OI
    B2 B1 + L2 time-weighted book state  (collector-derived)
    B3 B2 + ATM option level and short ATM IV changes
* Parameter budget: a fit is attempted only if n_train >= 20 p.
* Forward-chaining by session on the development sessions; final test
  sessions are scored once.
* Greedy backward deletion inside one block, accepted only when the mean
  fold RMSE improves AND the removal helps in a majority of folds.
* `log_rv_4h_tol`: branch-local 4h realized vol from >= 44 of 48 index
  returns (rescaled), replacing the shared strict 48/48 column in this branch.
* Collector vintage is attached per row from 02's data-derived boundaries;
  results are reported by vintage.

No HAR study; references are parameter-free (train mean, persistence).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_engineering.modeling import vol_ols_5m as base
from data_engineering.modeling.full_information import FEATURE_SPECS

VOL_OLS_BLOCKS_VERSION = "0.1.0"
MIN_ROWS_PER_PARAM = 20
FEATURE_MIN_COVERAGE = 0.80   # on development paired-valid rows, per asset
RV4H_MIN_RETURNS = 44
RV4H_WINDOW = 48

_SPEC = {s.name: s for s in FEATURE_SPECS}

EXCLUDED_BY_DESIGN = tuple(
    s.name for s in FEATURE_SPECS
    if s.block == "options" and s.scope in ("shape", "fit", "option")
) + ("atm_iv_d60", "log_rv_4h") + tuple(
    s.name for s in FEATURE_SPECS if s.scope == "instability")
# `option` scope = rr/bf wings, option spreads, OTM bid share: wing-dependent
# or quote-depth measures that also vanish near expiry.
# `instability` = L2 instability stats, absent before 2026-10-02 (VR 006).
# `log_rv_4h` is replaced by `log_rv_4h_tol`.

BLOCKS = {
    "B0_returns_vol_clock": [
        "return_5m", "ret_15m", "ret_30m", "ret_1h",
        "log_rv_30m", "log_rv_1h", "log_rv_4h_tol", "log_range_bps",
        "session_progress", "weekend", "clock_sin1", "clock_cos1",
        "clock_sin2", "clock_cos2", "log1p_tte_min",
    ],
    "B1_flow_basis": [
        "log_volume_usd", "log_trade_count", "tfi", "buy_share",
        "signed_volume", "log_avg_trade", "close_vs_vwap_bps",
        "perp_ret_5m", "perp_mid_basis_bps", "perp_mark_basis_bps",
        "microprice_mid_bps", "oi_change", "oi_return",
    ],
    "B2_l2_book": [
        "log_book_updates", "log_spread_bps", "obi_5", "obi_10", "obi_20",
        "obi_20_change", "log_bid_depth20", "log_ask_depth20",
        "microprice_ref_bps", "obi_20_min", "obi_20_max",
    ],
    "B3_atm_options": [
        "log_atm_iv", "atm_iv_d5", "atm_iv_d15", "atm_iv_d30",
    ],
}
BLOCK_ORDER = tuple(BLOCKS)


def nested_features(block: str, available: set[str] | None = None) -> list[str]:
    """Features of `block` and all earlier blocks, restricted to `available`."""
    out = []
    for b in BLOCK_ORDER:
        out += [n for n in BLOCKS[b] if available is None or n in available]
        if b == block:
            return out
    raise ValueError(block)


def admit_features(X: pd.DataFrame, dev_rows: np.ndarray,
                   min_coverage: float = FEATURE_MIN_COVERAGE) -> tuple[set[str], pd.DataFrame]:
    """Admit features with coverage >= min_coverage on development rows only.

    A feature absent for an asset would otherwise empty its whole block.
    Exclusions are recorded; they are not evidence of no predictive value.
    """
    rows, ok = [], set()
    for b in BLOCK_ORDER:
        for n in BLOCKS[b]:
            cov = float(X.loc[dev_rows, n].notna().mean()) if dev_rows.any() else 0.0
            admitted = cov >= min_coverage
            if admitted:
                ok.add(n)
            rows.append({"block": b, "feature": n, "dev_coverage": cov, "admitted": admitted})
    return ok, pd.DataFrame(rows)


def add_tolerant_rv4h(X: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """log of 4h realized vol from the last 48 index returns, >= 44 present.

    Time-based window (t-240min, t], so grid gaps are not bridged. The sum of
    squares is rescaled by 48 / count and annualized; the annualization
    constant only shifts the intercept of a log regressor.
    """
    r = pd.to_numeric(X["return_5m"], errors="coerce") if "return_5m" in X else None
    if r is None:
        X["log_rv_4h_tol"] = np.nan
        return X
    s = pd.Series(r.to_numpy(), index=pd.DatetimeIndex(meta["state_time"]))
    sq = (s ** 2).rolling("240min", closed="right").sum()
    n = s.rolling("240min", closed="right").count()
    sum48 = sq * RV4H_WINDOW / n
    ann = np.sqrt(sum48 * (365 * 288 / RV4H_WINDOW))
    ann = ann.where((n >= RV4H_MIN_RETURNS) & (ann > 0))
    X["log_rv_4h_tol"] = np.log(ann.to_numpy())
    return X


def panel(g: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Targets/meta and features for one underlying (all blocks)."""
    meta, X, _ = base.make_panel(g)
    X = add_tolerant_rv4h(X, meta)
    wanted = [c for b in BLOCK_ORDER for c in BLOCKS[b]]
    for c in wanted:
        if c not in X:
            X[c] = np.nan
    return meta, X[wanted]


def paired_valid(meta: pd.DataFrame) -> np.ndarray:
    """Both targets valid at t+1 and both current-bar estimators valid at t."""
    ok = np.ones(len(meta), dtype=bool)
    for t in base.TARGETS:
        ok &= np.isfinite(meta[base.TARGET_COLUMNS[t]].to_numpy(float))
        ok &= np.isfinite(meta[base.CURRENT_COLUMNS[t]].to_numpy(float))
    return ok


def session_plan(meta: pd.DataFrame, valid: np.ndarray, test_fraction: float = 0.25,
                 min_train_sessions: int = 6) -> dict:
    """Chronological sessions: development (forward-chained) then final test."""
    lab = base.session_labels(meta)
    sessions = np.array(sorted(set(lab[valid])))
    n = len(sessions)
    n_test = max(2, int(round(n * test_fraction)))
    n_dev = n - n_test
    if n_dev < min_train_sessions + 2:
        raise ValueError(f"Too few sessions: {n} (dev {n_dev}, need {min_train_sessions + 2})")
    dev, test = sessions[:n_dev], sessions[n_dev:]
    folds = [(dev[:k], dev[k]) for k in range(min_train_sessions, n_dev)]
    return {"labels": lab, "sessions": sessions, "dev": dev, "test": test, "folds": folds}


def target_y(meta: pd.DataFrame, target: str, train_rows: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray, float]:
    """(y, persistence, epsilon) with epsilon from the given training rows only."""
    sigma = meta[base.TARGET_COLUMNS[target]].to_numpy(float)
    cur = meta[base.CURRENT_COLUMNS[target]].to_numpy(float)
    eps = base.fit_log_floor(sigma[train_rows])
    y = np.log(sigma + eps)
    pers = np.log(cur + eps)
    return y, pers, eps


def drop_constant(X: pd.DataFrame, rows: np.ndarray, names: list[str]) -> list[str]:
    """Remove constants and exact aliases on training rows (QR)."""
    if not names:
        return []
    keep, _ = base.remove_aliases(X, rows, names)
    return keep


def fit_score(X: pd.DataFrame, y: np.ndarray, train: np.ndarray, score_rows: np.ndarray,
              names: list[str]) -> dict:
    """Fit on train rows complete for `names`; score on score_rows. Budget-checked."""
    complete = np.isfinite(X[names].to_numpy(float)).all(axis=1) if names else \
        np.ones(len(X), dtype=bool)
    tr = train & complete & np.isfinite(y)
    sc = score_rows & complete & np.isfinite(y)
    p = len(names) + 1
    if tr.sum() >= MIN_ROWS_PER_PARAM * p:
        names = drop_constant(X, tr, names)
        p = len(names) + 1
    if tr.sum() < MIN_ROWS_PER_PARAM * p:
        return {"ok": False, "reason": f"budget: n_train={int(tr.sum())} < {MIN_ROWS_PER_PARAM}*{p}",
                "names": names, "n_train": int(tr.sum()), "n_score": int(sc.sum())}
    fit = base.fit_ols(X, y, tr, names) if names else None
    mu = float(np.mean(y[tr]))
    idx = np.flatnonzero(sc)
    pred = fit.predict(X.iloc[idx]) if fit is not None else np.full(len(idx), mu)
    return {"ok": True, "fit": fit, "names": names, "train_mean": mu,
            "n_train": int(tr.sum()), "n_score": int(sc.sum()),
            "idx": idx, "pred": pred, **base.score(y[idx], pred, mu, "score")}


def evaluate_blocks(meta: pd.DataFrame, X: pd.DataFrame, valid: np.ndarray,
                    plan: dict, target: str, available: set[str] | None = None
                    ) -> tuple[pd.DataFrame, dict]:
    """Forward-chained fold scores and one final test score for every block.

    Two populations per block: its own complete rows ("own"), and rows
    complete for every block ("common"), so block increments can be read on
    identical rows.
    """
    lab = plan["labels"]
    all_names = nested_features(BLOCK_ORDER[-1], available)
    common_complete = np.isfinite(X[all_names].to_numpy(float)).all(axis=1)
    rows, final = [], {}
    stages = [("fold", i, tr, np.array([va])) for i, (tr, va) in enumerate(plan["folds"])]
    stages.append(("test", -1, plan["dev"], plan["test"]))
    for kind, k, tr_s, sc_s in stages:
        tr = np.isin(lab, tr_s) & valid
        sc = np.isin(lab, sc_s) & valid
        y, pers, eps = target_y(meta, target, tr)
        for population in ("own", "common"):
            extra = common_complete if population == "common" else np.ones(len(X), bool)
            for block in BLOCK_ORDER:
                r = fit_score(X, y, tr & extra, sc & extra, nested_features(block, available))
                rec = {"target": target, "stage": kind, "fold": k,
                       "validation_session": ",".join(map(str, sc_s)) if kind == "fold" else "test",
                       "population": population, "model": block,
                       "p": len(r["names"]), "n_train": r["n_train"], "n_score": r["n_score"],
                       "fitted": r["ok"], "reason": r.get("reason", "")}
                if r["ok"]:
                    rec.update({k2: r[k2] for k2 in ("log_rmse", "log_mae", "r2_vs_train_mean")})
                    # Persistence on exactly the scored rows of this block.
                    ps = base.score(y[r["idx"]], pers[r["idx"]], r["train_mean"], "score")
                    rec["persistence_log_rmse"] = ps["log_rmse"]
                    rec["rmse_ratio_vs_persistence"] = (
                        r["log_rmse"] / ps["log_rmse"] if ps["log_rmse"] else np.nan)
                    if kind == "test":
                        final[(population, block)] = {**r, "y": y, "eps": eps}
                rows.append(rec)
    return pd.DataFrame(rows), final


def fold_rmse(X, y_by_fold, folds_rows, names) -> list[float]:
    out = []
    for (tr, sc), y in zip(folds_rows, y_by_fold):
        r = fit_score(X, y, tr, sc, names)
        out.append(r["log_rmse"] if r["ok"] else np.nan)
    return out


def backward_by_folds(meta: pd.DataFrame, X: pd.DataFrame, valid: np.ndarray,
                      plan: dict, target: str, names: list[str], max_drops: int = 15
                      ) -> tuple[list[str], pd.DataFrame]:
    """Greedy deletion: accept a removal only if mean fold RMSE falls AND it
    improves the fold RMSE in a strict majority of folds. Rows are fixed to
    the starting feature set's complete rows, so comparisons are paired.
    Final test sessions are never used.
    """
    lab = plan["labels"]
    complete = np.isfinite(X[names].to_numpy(float)).all(axis=1)
    folds_rows, ys = [], []
    for tr_s, va in plan["folds"]:
        tr = np.isin(lab, tr_s) & valid & complete
        sc = (lab == va) & valid & complete
        y, _, _ = target_y(meta, target, tr)
        folds_rows.append((tr, sc))
        ys.append(y)
    keep = list(names)
    base_scores = np.array(fold_rmse(X, ys, folds_rows, keep))
    usable = np.isfinite(base_scores)
    if usable.sum() < 3:
        raise ValueError("Fewer than 3 usable folds for deletion")
    history = []
    for step in range(max_drops):
        best = None
        for col in keep:
            trial = [c for c in keep if c != col]
            s = np.array(fold_rmse(X, ys, folds_rows, trial))
            ok = usable & np.isfinite(s)
            if ok.sum() < 3:
                continue
            gain = base_scores[ok] - s[ok]
            mean_gain, wins = float(gain.mean()), int((gain > 0).sum())
            if mean_gain > 0 and wins > ok.sum() / 2:
                if best is None or mean_gain > best[0]:
                    best = (mean_gain, col, wins, int(ok.sum()), s)
        if best is None:
            break
        mean_gain, col, wins, nf, s = best
        history.append({"target": target, "step": step + 1, "removed": col,
                        "p_before": len(keep), "mean_fold_gain": mean_gain,
                        "folds_improved": wins, "folds": nf})
        keep.remove(col)
        base_scores = s
        usable = np.isfinite(base_scores)
    return keep, pd.DataFrame(history)


def assign_vintage(state_time: pd.Series, boundaries: pd.DataFrame | None,
                   underlying: str) -> np.ndarray:
    """Vintage label per row from 02_vintage_boundaries.csv (first bar per vintage)."""
    t = pd.to_datetime(state_time, utc=True).reset_index(drop=True)
    out = np.full(len(t), "unknown", dtype=object)
    if boundaries is None or boundaries.empty:
        return out
    b = boundaries[(boundaries["underlying"] == underlying)
                   & (boundaries["vintage"] != "<no_raw>")].copy()
    b["first"] = pd.to_datetime(b["first"], utc=True)
    for _, row in b.sort_values("first").iterrows():
        out[(t >= row["first"]).to_numpy()] = row["vintage"]
    return out


def clock_coverage(meta: pd.DataFrame, X: pd.DataFrame, valid: np.ndarray,
                   available: set[str] | None = None) -> pd.DataFrame:
    """Share of paired-valid rows complete for each nested block, by UTC hour."""
    hour = meta["state_time"].dt.hour.to_numpy()
    rows = []
    for block in BLOCK_ORDER:
        comp = np.isfinite(X[nested_features(block, available)].to_numpy(float)).all(axis=1)
        for h in range(24):
            at = valid & (hour == h)
            rows.append({"model": block, "utc_hour": h, "n_valid": int(at.sum()),
                         "n_complete": int((at & comp).sum())})
    out = pd.DataFrame(rows)
    out["share"] = out["n_complete"] / out["n_valid"].replace(0, np.nan)
    return out


def unique_loss(X: pd.DataFrame, valid: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Rows gained by removing ONE feature from `names` (marginal bottleneck)."""
    M = np.isfinite(X[names].to_numpy(float))
    full = int((valid & M.all(axis=1)).sum())
    rows = []
    for j, n in enumerate(names):
        rest = np.delete(M, j, axis=1).all(axis=1)
        rows.append({"feature": n, "coverage": float(M[valid, j].mean()) if valid.any() else np.nan,
                     "unique_loss_rows": int((valid & rest).sum()) - full})
    return pd.DataFrame(rows).sort_values("unique_loss_rows", ascending=False)
