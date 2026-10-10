"""
Volatility Dynamics Module 06: latent-state discovery around q99 volatility.

A deliberately simple diagonal-Gaussian HMM is fit to synchronized five-minute
market-state features WITHOUT q99 labels. q99 is used only after fitting as an
external event target.

Design principles
-----------------
- first-order HMM before HSMM;
- no forced MR/MOM/VOL/Bridge labels;
- q99/return_5m is not an emission feature;
- training-only clock normalization;
- chronological train/test split;
- filtered probabilities for all held-out q99 hazard calculations;
- exact five-minute sequence boundaries: asset changes, feature-missing rows,
  and non-5m timestamp gaps break a sequence;
- continuous crypto midnight does NOT break a sequence;
- core model and a separate sparse-L2 enhancement are kept distinct.

No external HMM package is required. The implementation below uses NumPy only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


VERSION = "0.1.0"
BAR_MINUTES = 5
Q99 = 0.99

CORE_FEATURES = (
    "intrabar_range_bps",
    "volume_usd",
    "trade_count",
    "option_iv_atm_forward",
    "iv_put_25d",
    "iv_call_25d",
    "book_twa_spread_bps",
    "book_update_count",
)

L2_FEATURES = (
    "book_obi_20_raw_zero_cross_count",
    "book_obi_20_transition_count",
    "book_obi_20_raw_zero_cross_rate",
    "book_obi_20_path_roughness_per_min",
)

ALL_FEATURES = tuple(dict.fromkeys(CORE_FEATURES + L2_FEATURES))

DEFAULT_CORE_K = (2, 3, 4, 5)
DEFAULT_L2_K = (2, 3, 4)
DEFAULT_N_INIT_CORE = 3
DEFAULT_N_INIT_L2 = 2
DEFAULT_MAX_ITER_CORE = 60
DEFAULT_MAX_ITER_L2 = 50
DEFAULT_TOL = 1e-4
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_MIN_SEQUENCE = 4
DEFAULT_MIN_SLOT_N = 5
DEFAULT_CLOCK_SHRINKAGE = 8.0
DEFAULT_CLIP_Z = 6.0
DEFAULT_FUTURE_HORIZONS_MIN = (30, 60, 90, 120)


def feature_contract(available_columns: Iterable[str]) -> pd.DataFrame:
    available = set(map(str, available_columns))
    rows = []
    for f in CORE_FEATURES:
        rows.append({
            "feature": f,
            "model": "core",
            "present": f in available,
            "role": "HMM emission",
        })
    for f in L2_FEATURES:
        rows.append({
            "feature": f,
            "model": "l2_enhanced",
            "present": f in available,
            "role": "HMM emission in sparse-L2 enhancement only",
        })
    return pd.DataFrame(rows)


def chronological_split_time(
    frame: pd.DataFrame,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
) -> pd.Timestamp:
    if not (0.5 <= float(train_fraction) < 1.0):
        raise ValueError("train_fraction must be in [0.5, 1.0).")
    t = pd.to_datetime(frame["state_time"], utc=True)
    dates = pd.Index(sorted(t.dt.floor("D").unique()))
    if len(dates) < 4:
        raise ValueError("Need at least four UTC dates for chronological split.")
    n_train_dates = int(np.floor(len(dates) * float(train_fraction)))
    n_train_dates = max(2, min(n_train_dates, len(dates) - 1))
    return pd.Timestamp(dates[n_train_dates])


def _robust_scale(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    med = float(np.median(x))
    q25, q75 = np.quantile(x, [0.25, 0.75])
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(x))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return med, scale


def fit_clock_normalizer(
    frame: pd.DataFrame,
    features: Sequence[str],
    split_time: pd.Timestamp,
    *,
    min_slot_n: int = DEFAULT_MIN_SLOT_N,
    shrinkage: float = DEFAULT_CLOCK_SHRINKAGE,
) -> pd.DataFrame:
    """Training-only asset/exact-clock robust location and scale.

    Exact-slot estimates are shrunk toward each asset's training distribution.
    Test data never contribute to normalization parameters.
    """
    features = list(features)
    x = frame.copy()
    x["state_time"] = pd.to_datetime(x["state_time"], utc=True)
    train = x[x["state_time"] < pd.Timestamp(split_time)].copy()
    train["clock_slot"] = train["state_time"].dt.hour * 12 + train["state_time"].dt.minute // 5

    rows = []
    for asset, g in train.groupby("underlying", sort=True):
        asset = str(asset)
        asset_stats: dict[str, tuple[float, float]] = {}
        for f in features:
            vals = pd.to_numeric(g[f], errors="coerce").to_numpy(float)
            asset_stats[f] = _robust_scale(vals)

        for f in features:
            a_med, a_scale = asset_stats[f]
            if not np.isfinite(a_med):
                continue
            for slot in range(288):
                vals = pd.to_numeric(
                    g.loc[g["clock_slot"] == slot, f], errors="coerce"
                ).to_numpy(float)
                vals = vals[np.isfinite(vals)]
                n = int(len(vals))
                if n >= int(min_slot_n):
                    s_med, s_scale = _robust_scale(vals)
                    w = n / (n + float(shrinkage))
                    loc = float(w * s_med + (1.0 - w) * a_med)
                    scale = float(w * s_scale + (1.0 - w) * a_scale)
                    source = "shrunk_exact_clock"
                else:
                    loc = float(a_med)
                    scale = float(a_scale)
                    source = "asset_fallback"
                if not np.isfinite(scale) or scale <= 1e-12:
                    scale = float(a_scale) if np.isfinite(a_scale) and a_scale > 0 else 1.0
                rows.append({
                    "underlying": asset,
                    "clock_slot": slot,
                    "feature": f,
                    "location": loc,
                    "scale": scale,
                    "slot_n": n,
                    "source": source,
                })
    return pd.DataFrame(rows)


def apply_clock_normalizer(
    frame: pd.DataFrame,
    params: pd.DataFrame,
    features: Sequence[str],
    *,
    clip_z: float = DEFAULT_CLIP_Z,
) -> pd.DataFrame:
    out = frame.copy()
    out["state_time"] = pd.to_datetime(out["state_time"], utc=True)
    out["clock_slot"] = out["state_time"].dt.hour * 12 + out["state_time"].dt.minute // 5

    for f in features:
        p = params[params["feature"] == f][
            ["underlying", "clock_slot", "location", "scale"]
        ].copy()
        p["underlying"] = p["underlying"].astype(str)
        lookup = {
            (str(r.underlying), int(r.clock_slot)): (float(r.location), float(r.scale))
            for r in p.itertuples()
        }
        vals = pd.to_numeric(out[f], errors="coerce").to_numpy(float)
        z = np.full(len(out), np.nan, float)
        for i, (asset, slot, val) in enumerate(
            zip(out["underlying"].astype(str), out["clock_slot"].astype(int), vals)
        ):
            if not np.isfinite(val):
                continue
            ps = lookup.get((asset, int(slot)))
            if ps is None:
                continue
            loc, scale = ps
            z[i] = (val - loc) / scale
        out[f"z__{f}"] = np.clip(z, -float(clip_z), float(clip_z))
    return out


def training_q99_thresholds(
    frame: pd.DataFrame,
    split_time: pd.Timestamp,
) -> pd.DataFrame:
    x = frame[["state_time", "underlying", "return_5m"]].copy()
    x["state_time"] = pd.to_datetime(x["state_time"], utc=True)
    x["variance_5m"] = pd.to_numeric(x["return_5m"], errors="coerce") ** 2
    rows = []
    for asset, g in x[x["state_time"] < pd.Timestamp(split_time)].groupby("underlying"):
        v = g["variance_5m"].to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v) < 100:
            continue
        rows.append({
            "underlying": str(asset),
            "q99_threshold_train": float(np.quantile(v, Q99)),
            "n_train_finite_returns": int(len(v)),
        })
    return pd.DataFrame(rows)


def label_q99_onsets_from_train_thresholds(
    frame: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    """Apply training-only q99 thresholds to the entire chronology."""
    x = frame[["state_time", "underlying", "return_5m"]].copy()
    x["state_time"] = pd.to_datetime(x["state_time"], utc=True)
    x["variance_5m"] = pd.to_numeric(x["return_5m"], errors="coerce") ** 2
    threshold_map = dict(zip(
        thresholds["underlying"].astype(str),
        thresholds["q99_threshold_train"].astype(float),
    ))
    x["q99_threshold_train"] = x["underlying"].astype(str).map(threshold_map)
    x["is_q99_train_threshold"] = (
        np.isfinite(x["variance_5m"].to_numpy(float))
        & np.isfinite(x["q99_threshold_train"].to_numpy(float))
        & (x["variance_5m"].to_numpy(float) >= x["q99_threshold_train"].to_numpy(float))
    )
    x["is_q99_onset_train_threshold"] = False

    step = pd.Timedelta(minutes=BAR_MINUTES)
    for asset, idx in x.groupby("underlying").groups.items():
        ii = np.asarray(list(idx), int)
        times = pd.DatetimeIndex(x.loc[ii, "state_time"])
        tail = x.loc[ii, "is_q99_train_threshold"].to_numpy(bool)
        lookup = {pd.Timestamp(t): k for k, t in enumerate(times)}
        onset = np.zeros(len(ii), bool)
        for k, t in enumerate(times):
            if not tail[k]:
                continue
            prev = lookup.get(pd.Timestamp(t) - step, -1)
            onset[k] = prev < 0 or not tail[prev]
        x.loc[ii, "is_q99_onset_train_threshold"] = onset
    return x


def _sequence_indices(
    frame: pd.DataFrame,
    z_features: Sequence[str],
    row_mask: np.ndarray,
    *,
    min_sequence: int = DEFAULT_MIN_SEQUENCE,
) -> list[np.ndarray]:
    """Exact contiguous 5m complete-case sequences."""
    z_features = list(z_features)
    values = frame[z_features].to_numpy(float)
    finite = np.isfinite(values).all(axis=1)
    allowed = finite & np.asarray(row_mask, bool)
    step_ns = pd.Timedelta(minutes=BAR_MINUTES).value
    seqs: list[np.ndarray] = []

    for asset, idx in frame.groupby("underlying", sort=True).groups.items():
        ii = np.asarray(list(idx), int)
        times = pd.DatetimeIndex(pd.to_datetime(frame.loc[ii, "state_time"], utc=True))
        ns = times.as_unit("ns").asi8
        current: list[int] = []
        last_t = None
        for global_i, t in zip(ii, ns):
            if not allowed[global_i]:
                if len(current) >= int(min_sequence):
                    seqs.append(np.asarray(current, int))
                current = []
                last_t = None
                continue
            if current and int(t - last_t) != int(step_ns):
                if len(current) >= int(min_sequence):
                    seqs.append(np.asarray(current, int))
                current = []
            current.append(int(global_i))
            last_t = int(t)
        if len(current) >= int(min_sequence):
            seqs.append(np.asarray(current, int))
    return seqs


def build_sequences(
    frame: pd.DataFrame,
    features: Sequence[str],
    split_time: pd.Timestamp,
    *,
    split: str,
    min_sequence: int = DEFAULT_MIN_SEQUENCE,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    z_features = [f"z__{f}" for f in features]
    t = pd.to_datetime(frame["state_time"], utc=True)
    if split == "train":
        mask = (t < pd.Timestamp(split_time)).to_numpy()
    elif split == "test":
        mask = (t >= pd.Timestamp(split_time)).to_numpy()
    else:
        raise ValueError("split must be 'train' or 'test'.")
    indices = _sequence_indices(frame, z_features, mask, min_sequence=min_sequence)
    arrays = [frame.loc[ii, z_features].to_numpy(float) for ii in indices]
    return arrays, indices


def sequence_metadata(
    frame: pd.DataFrame,
    seq_indices: list[np.ndarray],
    split: str,
) -> pd.DataFrame:
    rows = []
    for j, ii in enumerate(seq_indices):
        rows.append({
            "split": split,
            "sequence_id": j,
            "underlying": str(frame.at[int(ii[0]), "underlying"]),
            "start_time": pd.Timestamp(frame.at[int(ii[0]), "state_time"]),
            "end_time": pd.Timestamp(frame.at[int(ii[-1]), "state_time"]),
            "n_rows": int(len(ii)),
        })
    return pd.DataFrame(rows)


def _logsumexp(a: np.ndarray, axis=None) -> np.ndarray:
    a = np.asarray(a, float)
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    s = np.sum(np.exp(a - m), axis=axis, keepdims=True)
    out = m + np.log(np.maximum(s, 1e-300))
    if axis is None:
        return np.asarray(out.squeeze())
    return np.squeeze(out, axis=axis)


def _log_emission(X: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    means = np.asarray(means, float)
    variances = np.maximum(np.asarray(variances, float), 1e-6)
    d = X.shape[1]
    const = d * np.log(2.0 * np.pi)
    logdet = np.sum(np.log(variances), axis=1)
    diff = X[:, None, :] - means[None, :, :]
    quad = np.sum((diff ** 2) / variances[None, :, :], axis=2)
    return -0.5 * (const + logdet[None, :] + quad)


def _forward_backward(
    X: np.ndarray,
    pi: np.ndarray,
    A: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    T = len(X)
    K = len(pi)
    logpi = np.log(np.maximum(pi, 1e-300))
    logA = np.log(np.maximum(A, 1e-300))
    logB = _log_emission(X, means, variances)

    alpha = np.empty((T, K), float)
    alpha[0] = logpi + logB[0]
    for t in range(1, T):
        alpha[t] = logB[t] + _logsumexp(alpha[t - 1][:, None] + logA, axis=0)
    ll = float(_logsumexp(alpha[-1]))

    beta = np.zeros((T, K), float)
    for t in range(T - 2, -1, -1):
        beta[t] = _logsumexp(
            logA + logB[t + 1][None, :] + beta[t + 1][None, :],
            axis=1,
        )

    gamma = np.exp(alpha + beta - ll)
    gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

    xi_sum = np.zeros((K, K), float)
    for t in range(T - 1):
        z = (
            alpha[t][:, None]
            + logA
            + logB[t + 1][None, :]
            + beta[t + 1][None, :]
            - ll
        )
        x = np.exp(z)
        x /= np.maximum(x.sum(), 1e-300)
        xi_sum += x

    return ll, gamma, xi_sum


def _initialize_hmm(
    sequences: Sequence[np.ndarray],
    K: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.vstack(sequences)
    n, d = X.shape
    if n < K * 5:
        raise ValueError("Too few rows for requested K.")

    picks = rng.choice(n, size=K, replace=False)
    means = X[picks].copy()
    global_var = np.var(X, axis=0)
    global_var = np.where(np.isfinite(global_var) & (global_var > 1e-4), global_var, 1.0)
    variances = np.tile(global_var, (K, 1))

    pi = np.full(K, 1.0 / K)
    A = rng.dirichlet(np.ones(K), size=K)
    A += np.eye(K) * 2.0
    A /= A.sum(axis=1, keepdims=True)
    return pi, A, means, variances


@dataclass
class GaussianHMM:
    K: int
    n_features: int
    pi: np.ndarray
    A: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    train_loglik: float
    n_iter: int
    converged: bool
    seed: int

    def loglik(self, sequences: Sequence[np.ndarray]) -> float:
        total = 0.0
        for X in sequences:
            ll, _, _ = _forward_backward(X, self.pi, self.A, self.means, self.variances)
            total += ll
        return float(total)

    def smooth(self, X: np.ndarray) -> np.ndarray:
        _, gamma, _ = _forward_backward(X, self.pi, self.A, self.means, self.variances)
        return gamma

    def filter(self, X: np.ndarray) -> np.ndarray:
        logA = np.log(np.maximum(self.A, 1e-300))
        logB = _log_emission(X, self.means, self.variances)
        logp = np.log(np.maximum(self.pi, 1e-300)) + logB[0]
        logp -= _logsumexp(logp)
        out = np.empty((len(X), self.K), float)
        out[0] = np.exp(logp)
        for t in range(1, len(X)):
            logp = logB[t] + _logsumexp(logp[:, None] + logA, axis=0)
            logp -= _logsumexp(logp)
            out[t] = np.exp(logp)
        return out

    def viterbi(self, X: np.ndarray) -> np.ndarray:
        logA = np.log(np.maximum(self.A, 1e-300))
        logB = _log_emission(X, self.means, self.variances)
        T = len(X)
        delta = np.empty((T, self.K), float)
        psi = np.zeros((T, self.K), int)
        delta[0] = np.log(np.maximum(self.pi, 1e-300)) + logB[0]
        for t in range(1, T):
            z = delta[t - 1][:, None] + logA
            psi[t] = np.argmax(z, axis=0)
            delta[t] = np.max(z, axis=0) + logB[t]
        path = np.zeros(T, int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = int(psi[t + 1, path[t + 1]])
        return path


def fit_gaussian_hmm(
    sequences: Sequence[np.ndarray],
    K: int,
    *,
    seed: int,
    max_iter: int,
    tol: float = DEFAULT_TOL,
    min_variance: float = 1e-3,
) -> GaussianHMM:
    if len(sequences) == 0:
        raise ValueError("No sequences.")
    rng = np.random.default_rng(int(seed))
    pi, A, means, variances = _initialize_hmm(sequences, int(K), rng)
    n_features = sequences[0].shape[1]
    prev_ll = -np.inf
    converged = False
    ll = -np.inf

    for iteration in range(1, int(max_iter) + 1):
        start_sum = np.zeros(K, float)
        trans_sum = np.zeros((K, K), float)
        weight = np.zeros(K, float)
        x_sum = np.zeros((K, n_features), float)
        x2_sum = np.zeros((K, n_features), float)
        ll = 0.0

        for X in sequences:
            seq_ll, gamma, xi_sum = _forward_backward(X, pi, A, means, variances)
            ll += seq_ll
            start_sum += gamma[0]
            trans_sum += xi_sum
            weight += gamma.sum(axis=0)
            x_sum += gamma.T @ X
            x2_sum += gamma.T @ (X ** 2)

        pi = start_sum + 1e-6
        pi /= pi.sum()

        A = trans_sum + 1e-6
        A /= A.sum(axis=1, keepdims=True)

        safe_w = np.maximum(weight, 1e-8)
        means = x_sum / safe_w[:, None]
        variances = x2_sum / safe_w[:, None] - means ** 2
        variances = np.maximum(variances, float(min_variance))

        if np.isfinite(prev_ll):
            improvement = ll - prev_ll
            if improvement >= 0 and improvement <= float(tol) * (1.0 + abs(prev_ll)):
                converged = True
                break
        prev_ll = ll

    return GaussianHMM(
        K=int(K),
        n_features=n_features,
        pi=pi,
        A=A,
        means=means,
        variances=variances,
        train_loglik=float(ll),
        n_iter=int(iteration),
        converged=bool(converged),
        seed=int(seed),
    )


def _smoothed_occupancy(model: GaussianHMM, sequences: Sequence[np.ndarray]) -> np.ndarray:
    occ = np.zeros(model.K, float)
    n = 0
    for X in sequences:
        g = model.smooth(X)
        occ += g.sum(axis=0)
        n += len(X)
    return occ / max(n, 1)


def hmm_parameter_count(K: int, d: int) -> int:
    # initial probs + transition rows + diagonal Gaussian means/variances.
    return int((K - 1) + K * (K - 1) + 2 * K * d)


def fit_hmm_grid(
    train_sequences: Sequence[np.ndarray],
    test_sequences: Sequence[np.ndarray],
    k_values: Sequence[int],
    *,
    n_init: int,
    max_iter: int,
    seed_base: int = 6060,
) -> tuple[pd.DataFrame, dict[int, GaussianHMM]]:
    n_train = int(sum(len(x) for x in train_sequences))
    n_test = int(sum(len(x) for x in test_sequences))
    if n_train == 0 or n_test == 0:
        raise ValueError("Train/test sequences must both be non-empty.")
    d = train_sequences[0].shape[1]
    rows = []
    best: dict[int, GaussianHMM] = {}

    for K in map(int, k_values):
        fits = []
        for j in range(int(n_init)):
            seed = int(seed_base + K * 100 + j)
            model = fit_gaussian_hmm(
                train_sequences, K,
                seed=seed,
                max_iter=max_iter,
            )
            fits.append(model)

        fits.sort(key=lambda m: m.train_loglik, reverse=True)
        model = fits[0]
        best[K] = model
        test_ll = model.loglik(test_sequences)
        occ = _smoothed_occupancy(model, train_sequences)
        p = hmm_parameter_count(K, d)
        bic = -2.0 * model.train_loglik + p * np.log(max(n_train, 2))
        second_ll = fits[1].train_loglik if len(fits) > 1 else np.nan
        rows.append({
            "K": K,
            "n_features": d,
            "n_train_rows": n_train,
            "n_test_rows": n_test,
            "n_init": int(n_init),
            "best_seed": model.seed,
            "best_converged": model.converged,
            "best_n_iter": model.n_iter,
            "train_loglik": model.train_loglik,
            "train_loglik_per_row": model.train_loglik / n_train,
            "test_loglik": test_ll,
            "test_loglik_per_row": test_ll / n_test,
            "bic_train": float(bic),
            "min_train_state_occupancy": float(np.min(occ)),
            "max_train_state_occupancy": float(np.max(occ)),
            "second_best_train_loglik": float(second_ll) if np.isfinite(second_ll) else np.nan,
            "best_minus_second_train_loglik": (
                float(model.train_loglik - second_ll) if np.isfinite(second_ll) else np.nan
            ),
            "n_converged_initializations": int(sum(m.converged for m in fits)),
        })
    return pd.DataFrame(rows), best


def choose_k_by_training_bic(
    comparison: pd.DataFrame,
    *,
    min_occupancy: float = 0.01,
) -> int:
    x = comparison[
        (comparison["min_train_state_occupancy"] >= float(min_occupancy))
        & np.isfinite(comparison["bic_train"])
    ].copy()
    if x.empty:
        x = comparison[np.isfinite(comparison["bic_train"])].copy()
    if x.empty:
        raise ValueError("No valid HMM fits.")
    return int(x.sort_values(["bic_train", "K"]).iloc[0]["K"])


def filtered_state_table(
    frame: pd.DataFrame,
    model: GaussianHMM,
    sequences: Sequence[np.ndarray],
    seq_indices: Sequence[np.ndarray],
    *,
    split: str,
) -> pd.DataFrame:
    rows = []
    for sid, (X, ii) in enumerate(zip(sequences, seq_indices)):
        p = model.filter(X)
        hard = np.argmax(p, axis=1)
        for local_i, global_i in enumerate(ii):
            rec = {
                "split": split,
                "sequence_id": sid,
                "row_index": int(global_i),
                "state_time": pd.Timestamp(frame.at[int(global_i), "state_time"]),
                "underlying": str(frame.at[int(global_i), "underlying"]),
                "hard_state": int(hard[local_i]),
                "max_filtered_prob": float(np.max(p[local_i])),
                "filtered_entropy": float(
                    -np.sum(p[local_i] * np.log(np.maximum(p[local_i], 1e-300)))
                ),
            }
            for k in range(model.K):
                rec[f"p_state_{k}"] = float(p[local_i, k])
            rows.append(rec)
    return pd.DataFrame(rows)


def state_emission_table(
    model: GaussianHMM,
    features: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for k in range(model.K):
        for j, f in enumerate(features):
            rows.append({
                "state": k,
                "feature": f,
                "emission_mean_robust_z": float(model.means[k, j]),
                "emission_sd_robust_z": float(np.sqrt(model.variances[k, j])),
            })
    return pd.DataFrame(rows)


def transition_table(model: GaussianHMM) -> pd.DataFrame:
    rows = []
    for i in range(model.K):
        for j in range(model.K):
            rows.append({
                "from_state": i,
                "to_state": j,
                "probability": float(model.A[i, j]),
            })
    return pd.DataFrame(rows)


def _future_event_labels(
    frame: pd.DataFrame,
    q99_labels: pd.DataFrame,
    row_indices: Sequence[int],
    horizons_min: Sequence[int] = DEFAULT_FUTURE_HORIZONS_MIN,
) -> pd.DataFrame:
    event_times: dict[str, np.ndarray] = {}
    for asset, g in q99_labels[q99_labels["is_q99_onset_train_threshold"]].groupby("underlying"):
        event_times[str(asset)] = (
            pd.DatetimeIndex(g["state_time"]).as_unit("ns").asi8
        )

    rows = []
    for i in map(int, row_indices):
        asset = str(frame.at[i, "underlying"])
        t = pd.Timestamp(frame.at[i, "state_time"])
        t_ns = pd.DatetimeIndex([t]).as_unit("ns").asi8[0]
        arr = event_times.get(asset, np.empty(0, dtype=np.int64))
        pos = int(np.searchsorted(arr, t_ns, side="right"))
        next_t = arr[pos] if pos < len(arr) else None
        rec = {"row_index": i}
        for h in horizons_min:
            rec[f"q99_next_{int(h)}m"] = bool(
                next_t is not None
                and next_t > t_ns
                and next_t <= t_ns + pd.Timedelta(minutes=int(h)).value
            )
        rows.append(rec)
    return pd.DataFrame(rows)


def state_q99_hazard(
    frame: pd.DataFrame,
    filtered_test: pd.DataFrame,
    q99_labels: pd.DataFrame,
    *,
    horizons_min: Sequence[int] = DEFAULT_FUTURE_HORIZONS_MIN,
) -> pd.DataFrame:
    if filtered_test.empty:
        return pd.DataFrame()
    lab = _future_event_labels(
        frame, q99_labels, filtered_test["row_index"].astype(int).to_numpy(),
        horizons_min=horizons_min,
    )
    x = filtered_test.merge(lab, on="row_index", how="left")
    rows = []
    K = len([c for c in x.columns if c.startswith("p_state_")])
    for h in horizons_min:
        y = x[f"q99_next_{int(h)}m"].astype(float).to_numpy()
        baseline = float(np.mean(y))
        for k in range(K):
            p = x[f"p_state_{k}"].to_numpy(float)
            soft_den = float(np.sum(p))
            soft_hazard = float(np.sum(p * y) / soft_den) if soft_den > 0 else np.nan
            hard_mask = x["hard_state"].to_numpy(int) == k
            hard_hazard = float(np.mean(y[hard_mask])) if hard_mask.any() else np.nan
            rows.append({
                "horizon_min": int(h),
                "state": k,
                "baseline_hazard": baseline,
                "soft_state_hazard": soft_hazard,
                "soft_hazard_lift": soft_hazard / baseline if baseline > 0 else np.nan,
                "hard_state_hazard": hard_hazard,
                "hard_hazard_lift": hard_hazard / baseline if baseline > 0 and np.isfinite(hard_hazard) else np.nan,
                "hard_state_n": int(hard_mask.sum()),
                "soft_state_weight": soft_den,
            })
    return pd.DataFrame(rows)


def transition_q99_hazard(
    frame: pd.DataFrame,
    filtered_test: pd.DataFrame,
    q99_labels: pd.DataFrame,
    *,
    horizons_min: Sequence[int] = DEFAULT_FUTURE_HORIZONS_MIN,
) -> pd.DataFrame:
    if filtered_test.empty:
        return pd.DataFrame()
    lab = _future_event_labels(
        frame, q99_labels, filtered_test["row_index"].astype(int).to_numpy(),
        horizons_min=horizons_min,
    )
    x = filtered_test.merge(lab, on="row_index", how="left").sort_values(
        ["underlying", "state_time"]
    )
    step = pd.Timedelta(minutes=BAR_MINUTES)
    trans_rows = []
    for asset, g in x.groupby("underlying"):
        g = g.sort_values("state_time")
        prev = None
        for r in g.itertuples():
            if prev is not None and pd.Timestamp(r.state_time) - pd.Timestamp(prev.state_time) == step:
                rec = {
                    "underlying": str(asset),
                    "row_index": int(r.row_index),
                    "from_state": int(prev.hard_state),
                    "to_state": int(r.hard_state),
                }
                for h in horizons_min:
                    rec[f"q99_next_{int(h)}m"] = bool(getattr(r, f"q99_next_{int(h)}m"))
                trans_rows.append(rec)
            prev = r
    z = pd.DataFrame(trans_rows)
    if z.empty:
        return z

    rows = []
    for h in horizons_min:
        baseline = float(z[f"q99_next_{int(h)}m"].mean())
        for (i, j), g in z.groupby(["from_state", "to_state"]):
            hazard = float(g[f"q99_next_{int(h)}m"].mean())
            rows.append({
                "horizon_min": int(h),
                "from_state": int(i),
                "to_state": int(j),
                "n_transitions": int(len(g)),
                "baseline_hazard": baseline,
                "transition_hazard": hazard,
                "hazard_lift": hazard / baseline if baseline > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def event_aligned_state_probabilities(
    frame: pd.DataFrame,
    filtered: pd.DataFrame,
    q99_labels: pd.DataFrame,
    *,
    split_time: pd.Timestamp,
    rel_bars: Iterable[int] = range(-24, 13),
) -> pd.DataFrame:
    """Filtered state probabilities around held-out q99 onsets."""
    if filtered.empty:
        return pd.DataFrame()
    rel_bars = tuple(int(x) for x in rel_bars)
    prob_cols = [c for c in filtered.columns if c.startswith("p_state_")]
    lookup = {
        (str(r.underlying), pd.Timestamp(r.state_time)): r
        for r in filtered.itertuples()
    }
    events = q99_labels[
        q99_labels["is_q99_onset_train_threshold"]
        & (pd.to_datetime(q99_labels["state_time"], utc=True) >= pd.Timestamp(split_time))
    ].copy()
    rows = []
    step = pd.Timedelta(minutes=BAR_MINUTES)
    event_id = 0
    for ev in events.itertuples():
        event_id += 1
        for b in rel_bars:
            t = pd.Timestamp(ev.state_time) + b * step
            r = lookup.get((str(ev.underlying), t))
            if r is None:
                continue
            rec = {
                "event_id": event_id,
                "underlying": str(ev.underlying),
                "event_time": pd.Timestamp(ev.state_time),
                "rel_bar": b,
                "rel_min": b * BAR_MINUTES,
            }
            for c in prob_cols:
                rec[c] = float(getattr(r, c))
            rows.append(rec)
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw

    # Collapse simultaneous asset events first.
    first = raw.groupby(["event_time", "rel_bar", "rel_min"])[prob_cols].mean().reset_index()
    out = first.groupby(["rel_bar", "rel_min"])[prob_cols].agg(["mean", "median"])
    out.columns = [f"{c}_{stat}" for c, stat in out.columns]
    out = out.reset_index()
    counts = first.groupby(["rel_bar", "rel_min"])["event_time"].nunique().rename("n_unique_event_times").reset_index()
    return out.merge(counts, on=["rel_bar", "rel_min"], how="left")


def duration_diagnostics(
    model: GaussianHMM,
    filtered: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hard-filtered-state run durations and age-specific empirical exit hazard."""
    if filtered.empty:
        return pd.DataFrame(), pd.DataFrame()
    runs = []
    step = pd.Timedelta(minutes=BAR_MINUTES)
    for sid, g in filtered.groupby("sequence_id"):
        g = g.sort_values("state_time").reset_index(drop=True)
        if g.empty:
            continue
        start = 0
        state = int(g.loc[0, "hard_state"])
        for i in range(1, len(g) + 1):
            boundary = i == len(g) or int(g.loc[i, "hard_state"]) != state
            if boundary:
                length = i - start
                runs.append({
                    "sequence_id": int(sid),
                    "underlying": str(g.loc[start, "underlying"]),
                    "state": state,
                    "start_time": pd.Timestamp(g.loc[start, "state_time"]),
                    "end_time": pd.Timestamp(g.loc[i - 1, "state_time"]),
                    "duration_bars": int(length),
                    "duration_min": int(length * BAR_MINUTES),
                    "left_censored": bool(start == 0),
                    "right_censored": bool(i == len(g)),
                })
                if i < len(g):
                    start = i
                    state = int(g.loc[i, "hard_state"])
    runs_df = pd.DataFrame(runs)

    summary = []
    for k in range(model.K):
        g = runs_df[(runs_df["state"] == k) & ~runs_df["left_censored"] & ~runs_df["right_censored"]]
        d = g["duration_bars"].to_numpy(float)
        implied = 1.0 / max(1.0 - float(model.A[k, k]), 1e-12)
        summary.append({
            "state": k,
            "n_uncensored_runs": int(len(d)),
            "empirical_mean_duration_bars": float(np.mean(d)) if len(d) else np.nan,
            "empirical_median_duration_bars": float(np.median(d)) if len(d) else np.nan,
            "empirical_q90_duration_bars": float(np.quantile(d, 0.9)) if len(d) else np.nan,
            "hmm_self_transition": float(model.A[k, k]),
            "geometric_implied_mean_duration_bars": float(implied),
            "mean_duration_ratio_empirical_to_geometric": float(np.mean(d) / implied) if len(d) else np.nan,
        })
    summary_df = pd.DataFrame(summary)

    hazards = []
    for k in range(model.K):
        d = runs_df[
            (runs_df["state"] == k)
            & ~runs_df["left_censored"]
            & ~runs_df["right_censored"]
        ]["duration_bars"].to_numpy(int)
        if len(d) == 0:
            continue
        max_age = min(int(np.max(d)), 36)
        for age in range(1, max_age + 1):
            at_risk = int(np.sum(d >= age))
            exits = int(np.sum(d == age))
            hazards.append({
                "state": k,
                "age_bars": age,
                "age_min": age * BAR_MINUTES,
                "at_risk_runs": at_risk,
                "exits": exits,
                "empirical_exit_hazard": exits / at_risk if at_risk else np.nan,
                "hmm_constant_exit_hazard": 1.0 - float(model.A[k, k]),
            })
    return summary_df, pd.DataFrame(hazards)


def selftest() -> dict:
    rng = np.random.default_rng(6)
    A = np.array([[0.94, 0.06], [0.08, 0.92]])
    means = np.array([[-1.2, -0.8], [1.3, 1.0]])
    vars_ = np.array([[0.4, 0.5], [0.5, 0.4]])
    seqs = []
    for _ in range(4):
        T = 300
        s = np.zeros(T, int)
        for t in range(1, T):
            s[t] = rng.choice(2, p=A[s[t - 1]])
        X = means[s] + rng.normal(size=(T, 2)) * np.sqrt(vars_[s])
        seqs.append(X)
    model = fit_gaussian_hmm(seqs, 2, seed=606, max_iter=40)
    occ = _smoothed_occupancy(model, seqs)
    assert np.isfinite(model.train_loglik)
    assert np.allclose(model.A.sum(axis=1), 1.0)
    assert np.all(occ > 0.05)
    p = model.filter(seqs[0])
    assert np.allclose(p.sum(axis=1), 1.0)
    return {
        "numpy_only_hmm": True,
        "baum_welch_fit": True,
        "filtered_probabilities_normalized": True,
        "state_occupancy_non_degenerate": True,
    }
