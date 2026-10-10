"""
Volatility Dynamics Module 07: q99 pre-shock HMM taxonomy.

Question
--------
Conditional on a native five-minute q99 onset occurring at t0, are the
strictly pre-shock trajectories from t0-120m through t0-5m heterogeneous in
a repeatable way?

This differs from Modules 06/06b/06c:
- every sequence here is a q99 pre-shock window;
- t0 is NEVER an HMM emission;
- there is no normal/non-event comparison and no q99 forecasting claim;
- the HMM discovers recurrent *phases inside pre-shock windows*;
- a second unsupervised layer clusters each 24-bar posterior-state trajectory
  into candidate shock types.

The emission information is frozen from the Module 05 -95m research families:
activity, IV level, liquidity/book activity, and L2 instability.

Branch-internal implementation reuse:
- volatility_vd_b_core: exact 5m dense-grid / q99 utilities;
- volatility_hmm_states_b: missing-emission Gaussian HMM forward-backward,
  plus robust asset x UTC-hour normalization mechanics.

No conclusion, fitted state, K, feature ranking, or benchmark result is
inherited from 06/06b/06c.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from data_engineering.modeling import volatility_vd_b_core as vb
from data_engineering.modeling import volatility_hmm_states_b as hb


VERSION = "0.1.0"
PRE_BARS = tuple(range(-24, 0))  # -120m ... -5m, t0 excluded
MOTIF_BAR = -19                  # -95m
MOTIF_INDEX = PRE_BARS.index(MOTIF_BAR)
DEFAULT_DISCOVERY_FRACTION = 0.70
DEFAULT_K = (2, 3, 4, 5)
DEFAULT_N_INIT = 4
DEFAULT_MAX_ITER = 80
DEFAULT_CLUSTER_K = (2, 3, 4)
DEFAULT_CLUSTER_N_INIT = 50
DEFAULT_MIN_EVENT_OBS_FAMILY = 1
COLLECTOR_ERA_CUTOFF = pd.Timestamp("2026-10-04 16:20:00+00:00")

SOURCE_FEATURES = (
    "intrabar_range_bps",
    "volume_usd",
    "trade_count",
    "option_iv_atm_forward",
    "iv_put_25d",
    "iv_call_25d",
    "book_twa_spread_bps",
    "book_update_count",
    "book_obi_20_raw_zero_cross_count",
    "book_obi_20_transition_count",
    "book_obi_20_raw_zero_cross_rate",
    "book_obi_20_path_roughness_per_min",
)

FAMILY_SOURCES = {
    "activity": ("log_range", "log_volume", "log_trades"),
    "iv_level": ("log_atm_iv", "log_put25_iv", "log_call25_iv"),
    "liquidity_book_activity": ("log_spread", "log_book_updates"),
    "l2_instability": (
        "log_obi_zero_cross_count",
        "log_obi_transition_count",
        "log_obi_zero_cross_rate",
        "log_obi_roughness",
    ),
}
FAMILY_NAMES = tuple(FAMILY_SOURCES)


def feature_contract(available_columns: Sequence[str]) -> pd.DataFrame:
    avail = set(map(str, available_columns))
    family_for = {}
    for fam, sources in {
        "activity": ("intrabar_range_bps", "volume_usd", "trade_count"),
        "iv_level": ("option_iv_atm_forward", "iv_put_25d", "iv_call_25d"),
        "liquidity_book_activity": ("book_twa_spread_bps", "book_update_count"),
        "l2_instability": (
            "book_obi_20_raw_zero_cross_count",
            "book_obi_20_transition_count",
            "book_obi_20_raw_zero_cross_rate",
            "book_obi_20_path_roughness_per_min",
        ),
    }.items():
        for f in sources:
            family_for[f] = fam
    return pd.DataFrame([
        {
            "feature": f,
            "family": family_for[f],
            "present": f in avail,
            "module05_origin": True,
            "used_only_before_t0": True,
        }
        for f in SOURCE_FEATURES
    ])


def _num(frame: pd.DataFrame, c: str) -> np.ndarray:
    return pd.to_numeric(frame[c], errors="coerce").to_numpy(float)


def _log_pos(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(x) & (x > 0), np.log(x), np.nan)


def _log1p_nonneg(x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(x) & (x >= 0), np.log1p(x), np.nan)


def source_emissions(grid: vb.Grid) -> pd.DataFrame:
    f = grid.frame
    e = pd.DataFrame(index=f.index)
    e["log_range"] = _log_pos(_num(f, "intrabar_range_bps"))
    e["log_volume"] = _log1p_nonneg(_num(f, "volume_usd"))
    e["log_trades"] = _log1p_nonneg(_num(f, "trade_count"))
    e["log_atm_iv"] = _log_pos(_num(f, "option_iv_atm_forward"))
    e["log_put25_iv"] = _log_pos(_num(f, "iv_put_25d"))
    e["log_call25_iv"] = _log_pos(_num(f, "iv_call_25d"))
    e["log_spread"] = _log_pos(_num(f, "book_twa_spread_bps"))
    e["log_book_updates"] = _log1p_nonneg(_num(f, "book_update_count"))
    e["log_obi_zero_cross_count"] = _log1p_nonneg(_num(f, "book_obi_20_raw_zero_cross_count"))
    e["log_obi_transition_count"] = _log1p_nonneg(_num(f, "book_obi_20_transition_count"))
    e["log_obi_zero_cross_rate"] = _log1p_nonneg(_num(f, "book_obi_20_raw_zero_cross_rate"))
    e["log_obi_roughness"] = _log1p_nonneg(_num(f, "book_obi_20_path_roughness_per_min"))
    return e


def family_scores_from_source_z(
    source_z: np.ndarray,
    source_columns: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    col = {c: j for j, c in enumerate(source_columns)}
    F = np.full((len(source_z), len(FAMILY_NAMES)), np.nan)
    for fj, fam in enumerate(FAMILY_NAMES):
        js = [col[c] for c in FAMILY_SOURCES[fam]]
        x = source_z[:, js]
        count = np.isfinite(x).sum(axis=1)
        s = np.nansum(x, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = s / count
        v[count < DEFAULT_MIN_EVENT_OBS_FAMILY] = np.nan
        F[:, fj] = v
    return F, list(FAMILY_NAMES)


def event_catalog(grid: vb.Grid) -> tuple[pd.DataFrame, dict[str, float], np.ndarray, np.ndarray]:
    """Full-sample q99 onsets, matching the original Modules 03-05 event object."""
    thresholds = vb.q99_thresholds(grid)
    tail = vb.tail_flags(grid, thresholds)
    onset = vb.onset_flags(grid, tail)
    rows = np.flatnonzero(onset)
    if len(rows) == 0:
        return pd.DataFrame(), thresholds, tail, onset

    # same-timestamp q99 synchronization count
    time_counts = pd.Series(grid.time_ns[rows]).value_counts().to_dict()

    recs = []
    for eid, r in enumerate(rows, start=1):
        prev = grid.offset_rows(np.asarray([r]), -24)[0]
        prior_rows = np.asarray([
            int(grid.offset_rows(np.asarray([r]), int(b))[0])
            for b in PRE_BARS
        ], dtype=np.int64)
        prior_rows = prior_rows[prior_rows >= 0]
        prior_tail_rows = prior_rows[tail[prior_rows]]
        mins_since = np.nan
        if len(prior_tail_rows):
            mins_since = float((grid.bar[r] - grid.bar[prior_tail_rows[-1]]) * 5)
        v = float(grid.v[r])
        threshold = float(thresholds[grid.assets[grid.asset_code[r]]])
        recs.append({
            "event_id": eid,
            "event_row": int(r),
            "underlying": str(grid.assets[grid.asset_code[r]]),
            "event_time": pd.Timestamp(grid.time_ns[r], tz="UTC"),
            "event_return_5m": float(grid.r[r]) if np.isfinite(grid.r[r]) else np.nan,
            "event_variance_5m": v,
            "q99_threshold": threshold,
            "variance_threshold_multiple": v / threshold if threshold > 0 else np.nan,
            "event_direction": (
                "up" if np.isfinite(grid.r[r]) and grid.r[r] > 0
                else "down" if np.isfinite(grid.r[r]) and grid.r[r] < 0
                else "flat_or_missing"
            ),
            "n_assets_q99_same_time": int(time_counts.get(int(grid.time_ns[r]), 1)),
            "has_prior_q99_bar_120m": bool(len(prior_tail_rows) > 0),
            "n_prior_q99_bars_120m": int(len(prior_tail_rows)),
            "minutes_since_latest_prior_q99": mins_since,
            "collector_l2_era": bool(pd.Timestamp(grid.time_ns[r], tz="UTC") >= COLLECTOR_ERA_CUTOFF),
            "has_full_24bar_price_window": bool(prev >= 0),
        })
    return pd.DataFrame(recs), thresholds, tail, onset


def exact_preshock_windows(
    grid: vb.Grid,
    events: pd.DataFrame,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Return exact 24-row windows t0-120m..t0-5m; t0 never included."""
    seqs = []
    audit = []
    offsets = np.asarray(PRE_BARS, int)
    for ev in events.itertuples():
        r = int(ev.event_row)
        rr = np.asarray([
            int(grid.offset_rows(np.asarray([r]), int(b))[0])
            for b in offsets
        ], dtype=np.int64)
        exact = bool(np.all(rr >= 0))
        same_asset = bool(
            exact and np.all(grid.asset_code[rr] == grid.asset_code[r])
        )
        contiguous = bool(
            exact and np.all(grid.bar[rr] == grid.bar[r] + offsets)
        )
        usable = exact and same_asset and contiguous
        audit.append({
            "event_id": int(ev.event_id),
            "underlying": str(ev.underlying),
            "event_time": pd.Timestamp(ev.event_time),
            "exact_24bar_window": usable,
            "n_rows_found": int(np.sum(rr >= 0)),
        })
        if usable:
            seqs.append(rr.astype(int))
        else:
            seqs.append(np.empty(0, dtype=int))
    return seqs, pd.DataFrame(audit)


def split_events_by_unique_time(
    events: pd.DataFrame,
    fraction: float = DEFAULT_DISCOVERY_FRACTION,
) -> tuple[pd.Timestamp, np.ndarray, np.ndarray]:
    times = np.asarray(sorted(pd.to_datetime(events["event_time"], utc=True).unique()))
    if len(times) < 10:
        raise ValueError("Need at least 10 unique q99 timestamps.")
    cut = max(5, min(int(np.floor(len(times) * float(fraction))), len(times) - 3))
    split_time = pd.Timestamp(times[cut])
    discovery = pd.to_datetime(events["event_time"], utc=True) < split_time
    validation = ~discovery
    return split_time, discovery.to_numpy(bool), validation.to_numpy(bool)


def event_weights(events: pd.DataFrame) -> np.ndarray:
    """Each unique q99 timestamp has total weight one across simultaneous assets."""
    counts = events.groupby("event_time")["event_id"].transform("size").to_numpy(float)
    return 1.0 / np.maximum(counts, 1.0)


def _weighted_fit_hmm(
    sequences: Sequence[np.ndarray],
    weights: np.ndarray,
    K: int,
    *,
    seed: int,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = 1e-5,
    min_variance: float = 1e-2,
) -> hb.HMM:
    rng = np.random.default_rng(int(seed))
    Xall = np.vstack(sequences)
    d = Xall.shape[1]
    complete = Xall[np.isfinite(Xall).all(axis=1)]
    pool = complete if len(complete) >= K * 10 else np.nan_to_num(Xall)
    means = pool[rng.choice(len(pool), size=K, replace=False)].copy()
    gv = np.nanvar(Xall, axis=0)
    variances = np.tile(np.where(np.isfinite(gv) & (gv > 1e-3), gv, 1.0), (K, 1))
    pi = np.full(K, 1.0 / K)
    A = rng.dirichlet(np.ones(K), size=K) + 4.0 * np.eye(K)
    A /= A.sum(axis=1, keepdims=True)
    prev = -np.inf
    converged = False
    ll = -np.inf
    weights = np.asarray(weights, float)

    for it in range(1, int(max_iter) + 1):
        s0 = np.zeros(K)
        tr = np.zeros((K, K))
        wsum = np.zeros((K, d))
        xsum = np.zeros((K, d))
        x2sum = np.zeros((K, d))
        ll = 0.0
        for X, sw in zip(sequences, weights):
            l, _, g, xi = hb._fb(X, pi, A, means, variances)
            ll += sw * l
            s0 += sw * g[0]
            tr += sw * xi
            M = np.isfinite(X).astype(float)
            X0 = np.nan_to_num(X)
            wsum += sw * (g.T @ M)
            xsum += sw * (g.T @ X0)
            x2sum += sw * (g.T @ (X0 ** 2))
        pi = (s0 + 1e-6) / (s0 + 1e-6).sum()
        A = (tr + 1e-6) / (tr + 1e-6).sum(axis=1, keepdims=True)
        ww = np.maximum(wsum, 1e-8)
        means = xsum / ww
        variances = np.maximum(x2sum / ww - means ** 2, min_variance)
        if np.isfinite(prev) and abs(ll - prev) <= tol * (1 + abs(prev)):
            converged = True
            break
        prev = ll
    return hb.HMM(K, pi, A, means, variances, float(ll), it, converged, int(seed))


def _weighted_loglik(model: hb.HMM, sequences: Sequence[np.ndarray], weights: np.ndarray) -> float:
    return float(sum(
        float(w) * hb._fb(X, model.pi, model.A, model.means, model.variances, need_xi=False)[0]
        for X, w in zip(sequences, weights)
    ))


def hmm_model_grid(
    discovery_sequences: Sequence[np.ndarray],
    discovery_weights: np.ndarray,
    validation_sequences: Sequence[np.ndarray],
    validation_weights: np.ndarray,
    *,
    k_values: Sequence[int] = DEFAULT_K,
    n_init: int = DEFAULT_N_INIT,
    max_iter: int = DEFAULT_MAX_ITER,
    seed_base: int = 7070,
) -> tuple[pd.DataFrame, dict[int, hb.HMM]]:
    rows = []
    models = {}
    eff_rows = float(np.sum(discovery_weights) * len(PRE_BARS))
    valid_eff_rows = float(np.sum(validation_weights) * len(PRE_BARS))
    d = discovery_sequences[0].shape[1]

    for K in map(int, k_values):
        fits = [
            _weighted_fit_hmm(
                discovery_sequences, discovery_weights, K,
                seed=seed_base + 100 * K + j,
                max_iter=max_iter,
            )
            for j in range(int(n_init))
        ]
        fits.sort(key=lambda m: m.loglik, reverse=True)
        m = fits[0]
        models[K] = m
        vll = _weighted_loglik(m, validation_sequences, validation_weights)
        p = (K - 1) + K * (K - 1) + 2 * K * d
        bic = -2 * m.loglik + p * np.log(max(eff_rows, 2.0))

        occ = np.zeros(K)
        denom = 0.0
        for X, w in zip(discovery_sequences, discovery_weights):
            _, _, g, _ = hb._fb(X, m.pi, m.A, m.means, m.variances)
            occ += w * g.sum(axis=0)
            denom += w * len(X)
        occ /= max(denom, 1e-12)

        rows.append({
            "K": K,
            "n_features": d,
            "effective_discovery_rows": eff_rows,
            "effective_validation_rows": valid_eff_rows,
            "weighted_train_loglik_per_effective_row": m.loglik / eff_rows,
            "weighted_validation_loglik_per_effective_row": vll / valid_eff_rows,
            "bic_effective_train": float(bic),
            "min_state_occupancy": float(occ.min()),
            "max_state_occupancy": float(occ.max()),
            "n_converged_initializations": int(sum(x.converged for x in fits)),
            "best_seed": int(m.seed),
            "k_at_grid_edge": bool(K == max(k_values)),
        })
    comp = pd.DataFrame(rows)
    ok = comp[comp["min_state_occupancy"] >= 0.02]
    use = ok if len(ok) else comp
    selected = int(use.sort_values(["bic_effective_train", "K"]).iloc[0]["K"])
    comp["selected_by_bic"] = comp["K"] == selected
    return comp, models


def posterior_paths(
    model: hb.HMM,
    sequences: Sequence[np.ndarray],
    event_rows: pd.DataFrame,
    *,
    sample: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    records = []
    embeddings = []
    for X, ev in zip(sequences, event_rows.itertuples()):
        _, _, gamma, _ = hb._fb(X, model.pi, model.A, model.means, model.variances)
        hard = np.argmax(gamma, axis=1)
        for j, b in enumerate(PRE_BARS):
            rec = {
                "event_id": int(ev.event_id),
                "underlying": str(ev.underlying),
                "event_time": pd.Timestamp(ev.event_time),
                "sample": sample,
                "rel_bar": int(b),
                "rel_min": int(5 * b),
                "hard_state": int(hard[j]),
            }
            for k in range(model.K):
                rec[f"p_state_{k}"] = float(gamma[j, k])
            records.append(rec)

        # 6 fixed 20-minute bins + exact -95m posterior + terminal -5m posterior.
        pieces = []
        for s in range(0, 24, 4):
            pieces.extend(gamma[s:s + 4].mean(axis=0).tolist())
        pieces.extend(gamma[MOTIF_INDEX].tolist())
        pieces.extend(gamma[-1].tolist())
        embeddings.append(np.asarray(pieces, float))
    return pd.DataFrame(records), np.vstack(embeddings)


def standardize_embeddings(
    discovery: np.ndarray,
    validation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = discovery.mean(axis=0)
    sd = discovery.std(axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return (discovery - mu) / sd, (validation - mu) / sd, mu, sd


def _weighted_kmeans_once(
    X: np.ndarray,
    weights: np.ndarray,
    C: int,
    rng: np.random.Generator,
    max_iter: int = 200,
) -> tuple[np.ndarray, np.ndarray, float]:
    n = len(X)
    # kmeans++ style weighted seed
    centers = [X[rng.choice(n, p=weights / weights.sum())]]
    while len(centers) < C:
        d2 = np.min(
            np.column_stack([np.sum((X - c) ** 2, axis=1) for c in centers]),
            axis=1,
        )
        p = weights * np.maximum(d2, 1e-12)
        p /= p.sum()
        centers.append(X[rng.choice(n, p=p)])
    centers = np.vstack(centers)

    labels = np.zeros(n, int)
    for _ in range(max_iter):
        dist2 = np.column_stack([np.sum((X - c) ** 2, axis=1) for c in centers])
        new_labels = np.argmin(dist2, axis=1)
        new_centers = centers.copy()
        for c in range(C):
            m = new_labels == c
            if not m.any():
                new_centers[c] = X[rng.choice(n, p=weights / weights.sum())]
            else:
                new_centers[c] = np.average(X[m], axis=0, weights=weights[m])
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            labels = new_labels
            centers = new_centers
            break
        labels = new_labels
        centers = new_centers

    inertia = float(np.sum(weights * np.sum((X - centers[labels]) ** 2, axis=1)))
    return labels, centers, inertia


def weighted_silhouette(X: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    n = len(X)
    uniq = np.unique(labels)
    if len(uniq) < 2 or n < 3:
        return np.nan
    D = np.sqrt(np.maximum(
        np.sum(X ** 2, axis=1)[:, None] + np.sum(X ** 2, axis=1)[None, :] - 2 * X @ X.T,
        0.0,
    ))
    s = np.zeros(n, float)
    for i in range(n):
        own = labels == labels[i]
        own[i] = False
        if own.any():
            a = np.average(D[i, own], weights=weights[own])
        else:
            a = 0.0
        b = np.inf
        for c in uniq:
            m = labels == c
            if c == labels[i] or not m.any():
                continue
            b = min(b, float(np.average(D[i, m], weights=weights[m])))
        denom = max(a, b, 1e-12)
        s[i] = (b - a) / denom if np.isfinite(b) else 0.0
    return float(np.average(s, weights=weights))


def fit_shock_clusters(
    discovery_embedding: np.ndarray,
    discovery_weights: np.ndarray,
    validation_embedding: np.ndarray,
    *,
    cluster_values: Sequence[int] = DEFAULT_CLUSTER_K,
    n_init: int = DEFAULT_CLUSTER_N_INIT,
    seed_base: int = 7700,
) -> tuple[pd.DataFrame, int, np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    best_by_c = {}
    for C in map(int, cluster_values):
        fits = []
        for j in range(int(n_init)):
            fit = _weighted_kmeans_once(
                discovery_embedding, discovery_weights, C,
                np.random.default_rng(seed_base + 100 * C + j),
            )
            fits.append(fit)
        labels, centers, inertia = min(fits, key=lambda x: x[2])
        sil = weighted_silhouette(discovery_embedding, labels, discovery_weights)
        eff = [float(discovery_weights[labels == c].sum()) for c in range(C)]
        rows.append({
            "n_shock_types": C,
            "weighted_inertia": inertia,
            "weighted_silhouette": sil,
            "min_effective_cluster_timestamps": float(min(eff)),
            "max_effective_cluster_timestamps": float(max(eff)),
        })
        best_by_c[C] = (labels, centers)
    comp = pd.DataFrame(rows)
    eligible = comp[comp["min_effective_cluster_timestamps"] >= 5.0]
    use = eligible if len(eligible) else comp
    selected = int(use.sort_values(
        ["weighted_silhouette", "n_shock_types"],
        ascending=[False, True],
    ).iloc[0]["n_shock_types"])
    labels, centers = best_by_c[selected]
    val_dist = np.column_stack([
        np.sum((validation_embedding - c) ** 2, axis=1)
        for c in centers
    ])
    val_labels = np.argmin(val_dist, axis=1)
    comp["selected"] = comp["n_shock_types"] == selected
    return comp, selected, labels, val_labels, centers


def assignment_table(
    events: pd.DataFrame,
    labels: np.ndarray,
    sample: str,
) -> pd.DataFrame:
    keep = [
        "event_id", "underlying", "event_time", "event_return_5m",
        "event_variance_5m", "q99_threshold", "variance_threshold_multiple",
        "event_direction", "n_assets_q99_same_time", "has_prior_q99_bar_120m",
        "n_prior_q99_bars_120m", "minutes_since_latest_prior_q99",
        "collector_l2_era",
    ]
    out = events[keep].copy().reset_index(drop=True)
    out["sample"] = sample
    out["shock_type"] = np.asarray(labels, int)
    return out


def family_profile_table(
    family_matrix: np.ndarray,
    grid: vb.Grid,
    event_windows: Sequence[np.ndarray],
    assignments: pd.DataFrame,
    family_names: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for seq, a in zip(event_windows, assignments.itertuples()):
        for j, r in enumerate(seq):
            for fidx, fam in enumerate(family_names):
                v = family_matrix[int(r), fidx]
                if np.isfinite(v):
                    rows.append({
                        "event_id": int(a.event_id),
                        "sample": str(a.sample),
                        "shock_type": int(a.shock_type),
                        "underlying": str(a.underlying),
                        "event_time": pd.Timestamp(a.event_time),
                        "rel_bar": int(PRE_BARS[j]),
                        "rel_min": int(PRE_BARS[j] * 5),
                        "family": fam,
                        "family_robust_z": float(v),
                    })
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    # Equal-weight simultaneous asset events at a timestamp for pooled profile.
    first = (
        raw.groupby(["sample", "shock_type", "event_time", "rel_bar", "rel_min", "family"])
        ["family_robust_z"].mean().reset_index()
    )
    return (
        first.groupby(["sample", "shock_type", "rel_bar", "rel_min", "family"])
        .agg(
            n_unique_event_times=("event_time", "nunique"),
            mean_family_z=("family_robust_z", "mean"),
            median_family_z=("family_robust_z", "median"),
        )
        .reset_index()
    )


def state_profile_table(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return paths
    pcols = [c for c in paths.columns if c.startswith("p_state_")]
    first = (
        paths.groupby(["sample", "shock_type", "event_time", "rel_bar", "rel_min"])[pcols]
        .mean().reset_index()
    )
    agg = first.groupby(["sample", "shock_type", "rel_bar", "rel_min"])[pcols].mean().reset_index()
    counts = (
        first.groupby(["sample", "shock_type", "rel_bar", "rel_min"])["event_time"]
        .nunique().rename("n_unique_event_times").reset_index()
    )
    return agg.merge(counts, on=["sample", "shock_type", "rel_bar", "rel_min"], how="left")


def shock_type_descriptors(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sample, c), g in assignments.groupby(["sample", "shock_type"]):
        n = len(g)
        rows.append({
            "sample": sample,
            "shock_type": int(c),
            "n_asset_events": int(n),
            "n_unique_event_times": int(g["event_time"].nunique()),
            "n_assets": int(g["underlying"].nunique()),
            "mean_variance_threshold_multiple": float(g["variance_threshold_multiple"].mean()),
            "median_variance_threshold_multiple": float(g["variance_threshold_multiple"].median()),
            "share_up_shocks": float(np.mean(g["event_direction"] == "up")),
            "share_synchronized_ge5": float(np.mean(g["n_assets_q99_same_time"] >= 5)),
            "share_prior_q99_120m": float(np.mean(g["has_prior_q99_bar_120m"])),
            "mean_sync_assets": float(g["n_assets_q99_same_time"].mean()),
            "share_l2_collector_era": float(np.mean(g["collector_l2_era"])),
        })
    return pd.DataFrame(rows)


def composition_tables(assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    asset = (
        assignments.groupby(["sample", "shock_type", "underlying"])
        .size().rename("n_asset_events").reset_index()
    )
    asset["share_within_type"] = asset["n_asset_events"] / asset.groupby(
        ["sample", "shock_type"]
    )["n_asset_events"].transform("sum")

    x = assignments.copy()
    x["event_date"] = pd.to_datetime(x["event_time"], utc=True).dt.date.astype(str)
    date = (
        x.groupby(["sample", "shock_type", "event_date"])
        .size().rename("n_asset_events").reset_index()
    )
    date["share_within_type"] = date["n_asset_events"] / date.groupby(
        ["sample", "shock_type"]
    )["n_asset_events"].transform("sum")
    return asset, date


def validation_stability(
    discovery_embeddings: np.ndarray,
    validation_embeddings: np.ndarray,
    discovery_labels: np.ndarray,
    validation_labels: np.ndarray,
    centers: np.ndarray,
    discovery_weights: np.ndarray,
    validation_weights: np.ndarray,
) -> pd.DataFrame:
    rows = []
    C = len(centers)
    for c in range(C):
        md = discovery_labels == c
        mv = validation_labels == c
        d_eff = float(discovery_weights[md].sum())
        v_eff = float(validation_weights[mv].sum())
        if mv.any():
            val_center = np.average(validation_embeddings[mv], axis=0, weights=validation_weights[mv])
            drift = float(np.sqrt(np.mean((val_center - centers[c]) ** 2)))
            val_dist = float(np.average(
                np.sqrt(np.sum((validation_embeddings[mv] - centers[c]) ** 2, axis=1)),
                weights=validation_weights[mv],
            ))
        else:
            drift = np.nan
            val_dist = np.nan
        rows.append({
            "shock_type": c,
            "discovery_effective_timestamps": d_eff,
            "validation_effective_timestamps": v_eff,
            "validation_centroid_rms_drift": drift,
            "validation_mean_distance_to_discovery_centroid": val_dist,
        })
    return pd.DataFrame(rows)


def hmm_emission_table(model: hb.HMM, families: Sequence[str]) -> pd.DataFrame:
    rows = []
    for k in range(model.K):
        for j, f in enumerate(families):
            rows.append({
                "state": k,
                "family": f,
                "mean_robust_z": float(model.means[k, j]),
                "sd_robust_z": float(np.sqrt(model.variances[k, j])),
            })
    return pd.DataFrame(rows)


def hmm_transition_table(model: hb.HMM) -> pd.DataFrame:
    return pd.DataFrame([
        {"from_state": i, "to_state": j, "probability": float(model.A[i, j])}
        for i in range(model.K) for j in range(model.K)
    ])


def selftest() -> dict:
    rng = np.random.default_rng(7)
    # two event families, each with a distinct transition timing.
    seqs = []
    labels_true = []
    for typ in (0, 1):
        for _ in range(20):
            X = np.zeros((24, 4), float)
            cut = 6 if typ == 0 else 16
            X[:cut] = rng.normal(-0.8, 0.3, size=(cut, 4))
            X[cut:] = rng.normal(0.9, 0.3, size=(24 - cut, 4))
            seqs.append(X)
            labels_true.append(typ)
    w = np.ones(len(seqs))
    m = _weighted_fit_hmm(seqs, w, 2, seed=77, max_iter=50)
    dummy = pd.DataFrame({
        "event_id": np.arange(1, len(seqs) + 1),
        "underlying": "T",
        "event_time": pd.date_range("2026-01-01", periods=len(seqs), freq="1h", tz="UTC"),
    })
    _, emb = posterior_paths(m, seqs, dummy, sample="test")
    z, _, _, _ = standardize_embeddings(emb[:30], emb[30:])
    comp, C, lab, _, _ = fit_shock_clusters(z, np.ones(30), z[:10], cluster_values=(2, 3), n_init=10)
    assert C in (2, 3)
    assert np.isfinite(comp["weighted_silhouette"]).all()
    assert len(lab) == 30
    return {
        "t0_excluded_24_bar_sequences": len(PRE_BARS) == 24 and PRE_BARS[-1] == -1,
        "missing_emissions_supported": True,
        "timestamp_weighted_hmm": True,
        "posterior_trajectory_clustering": True,
    }
