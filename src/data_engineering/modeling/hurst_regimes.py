"""
Hurst Regimes branch, notebook 01 (HR 001): does the Hurst exponent of 5-minute
price paths vary from session to session, persist, and predict the next
session's terminal move per unit of path variance?

These are the three prerequisites for any HMM on the Hurst exponent. If H does
not vary beyond estimator noise, there are no states to find. If it does not
persist, a state cannot be acted on. If it does not move the terminal move
relative to path variance, it does not touch what a 0DTE straddle pays.

Data (shared base data, no other branch's outputs)
--------------------------------------------------
5-minute closes from DERIBIT_DATA_ROOT/history/price_5m/<ASSET>.csv.gz
(tools/download_price_history.py; columns state_time = bar END, close).

Block
-----
Session d = the 288 five-minute bars ending in (d-1 08:00, d 08:00] UTC (the
Deribit daily expiry cycle). A block is valid if at least MIN_BARS returns are
finite; missing returns are set to 0 (and to 0 volatility in the null).
Returns are deseasonalized by a per-asset 288-slot clock s(slot) (RMS return
per slot, development blocks only, normalized to mean square 1), and the path
X_t = cumsum(r_t / s(slot_t)).

Estimator (change of frequency, as in Ivan's thesis §5.1)
---------------------------------------------------------
    D1_t = X_t - 2X_{t-1} + X_{t-2},   D2_t = X_t - 2X_{t-2} + X_{t-4}
    H    = 0.5 log2( sum D2^2 / sum D1^2 ),  clipped to [0.01, 0.99]
Sensitivity (no decision): the same on the 10-minute subsample (H_10m), which
is less exposed to bid-ask bounce.

Outcome for gate 3
------------------
    VR_d = (sum_t r~_t)^2 / sum_t r~_t^2      (deseasonalized returns of block d)
the terminal 24h move per unit of path variance. Under H = 0.5, E[VR] = 1;
persistence (H > 0.5) raises it. A long straddle pays on the terminal move.

Null (constant H, real volatility path)
---------------------------------------
For each block, r*_t = sigma_t * e_t with e unit-variance fractional Gaussian
noise at the asset's constant H0 = median development H, and sigma_t the
centered 12-bar RMS of the block's own deseasonalized returns. The null keeps
the block's volatility clustering, jumps in magnitude and gaps, and removes
any block-to-block variation of H. N_REP replicates; every replicate of every
block is estimated with the same code as the data.

Pre-registered gates (development sessions only: DEV_END and before)
--------------------------------------------------------------------
G1 variation   p1 = MC p of Var(H_d) under the null (upper tail).
               Asset passes: Holm-adjusted p1 < 0.05 across eligible assets.
               Gate passes: >= half of eligible assets pass AND the median
               excess SD sqrt(max(Var_obs - mean Var_null, 0)) >= 0.02.
G2 persistence rho2 = Spearman(H_d, H_{d+1}) over consecutive valid blocks;
               p2 = MC p (upper tail) vs the same statistic in replicates.
               Asset passes: Holm p2 < 0.05. Gate passes: >= half of assets
               AND median (rho2_obs - mean rho2_null) >= 0.10.
G3 relevance   rho3 = Spearman(H_d, log VR_{d+1}); p3 = MC p (upper tail).
               Asset passes: Holm p3 < 0.05. Gate passes: >= half of assets
               AND median (rho3_obs - mean rho3_null) > 0.
Branch verdict: PROCEED_TO_HMM only if G1, G2 and G3 all pass on development
AND G3 passes on the confirmation sessions (CONF_START onward) with the same
rule, run once. Otherwise STOP.

Eligible asset: >= MIN_BLOCKS valid development blocks (>= CONF_MIN_BLOCKS in confirmation).
Confirmation uses the development clock and the development H0 (both frozen).
Why 0.02 for G1: with fBm scaling the 24h-to-5m variance ratio is 288^(2H-1),
so a change of H by 0.02 changes the terminal variance by 288^0.04 = 1.25x.
VRP 008 puts the long-straddle break-even at about 1.35x (move), i.e. a
variance factor of about 1.8 or dH of about 0.05.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


VERSION = "0.1.0"
BARS = 288
MIN_BARS = 280
MIN_BLOCKS = 200
CONF_MIN_BLOCKS = 100            # confirmation window is ~6 months
DEV_END = "2026-03-31"            # last development session (inclusive)
CONF_START = "2026-04-01"         # first confirmation session
N_REP = 400
SEED = 9001
VOL_WIN = 12                      # bars in the centered RMS volatility path of the null
H_CLIP = (0.01, 0.99)
G1_MIN_EXCESS_SD = 0.02
G2_MIN_EXCESS_RHO = 0.10
ALPHA = 0.05


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_history(directory: Path, assets: Sequence[str]) -> pd.DataFrame:
    frames = []
    for a in assets:
        p = Path(directory) / f"{a}.csv.gz"
        if p.exists():
            h = pd.read_csv(p, usecols=lambda c: c in ("state_time", "close", "source"))
            h["underlying"] = a
            frames.append(h)
    if not frames:
        raise FileNotFoundError(f"No history files in {directory}")
    h = pd.concat(frames, ignore_index=True)
    h["state_time"] = pd.to_datetime(h["state_time"], utc=True)
    h["close"] = pd.to_numeric(h["close"], errors="coerce")
    return h


def session_blocks(h: pd.DataFrame, asset: str) -> tuple[np.ndarray, np.ndarray]:
    """(sessions, R) with R[d, j] the log return of bar j of session d (NaN if missing).
    Bar j ends at session 08:00 - (287 - j) * 5 min."""
    g = h[(h["underlying"] == asset) & np.isfinite(h["close"]) & (h["close"] > 0)]
    g = g.drop_duplicates("state_time").set_index("state_time")["close"].sort_index()
    if g.empty:
        return np.array([], dtype=object), np.zeros((0, BARS))
    grid = pd.date_range(g.index.min().ceil("5min"), g.index.max().floor("5min"), freq="5min")
    lc = np.log(g.reindex(grid))
    r = lc.diff()
    r[~np.isfinite(r)] = np.nan
    anchor = (grid - pd.Timedelta(hours=8, minutes=5))      # bar ending 08:00 belongs to that day's session
    sess = (anchor.normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    pos = ((grid - pd.Timedelta(hours=8, minutes=5)) - anchor.normalize()) // pd.Timedelta(minutes=5)
    df = pd.DataFrame({"s": sess, "j": np.asarray(pos, int), "r": r.to_numpy()})
    wide = df.pivot_table(index="s", columns="j", values="r", aggfunc="first", dropna=False)
    wide = wide.reindex(columns=range(BARS))
    wide = wide[np.isfinite(wide.to_numpy(float)).any(axis=1)]
    return wide.index.to_numpy(), wide.to_numpy(float)


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------

def clock_profile(R: np.ndarray) -> np.ndarray:
    """288-slot RMS profile, normalized to mean square 1 (development rows only)."""
    ms = np.nanmean(R ** 2, axis=0)
    ms = np.where(np.isfinite(ms) & (ms > 0), ms, np.nanmean(ms))
    ms = pd.Series(np.r_[ms[-3:], ms, ms[:3]]).rolling(7, center=True).mean().to_numpy()[3:-3]   # light smoothing
    return np.sqrt(ms / ms.mean())


def cof_hurst(X: np.ndarray, step: int = 1) -> np.ndarray:
    """Change-of-frequency Hurst estimate per row of a path matrix X (n_rows, n_points)."""
    X = X[:, ::step]
    D1 = X[:, 4:] - 2 * X[:, 3:-1] + X[:, 2:-2]
    D2 = X[:, 4:] - 2 * X[:, 2:-2] + X[:, :-4]
    with np.errstate(divide="ignore", invalid="ignore"):
        h = 0.5 * np.log2((D2 ** 2).sum(1) / (D1 ** 2).sum(1))
    return np.clip(h, *H_CLIP)


def block_stats(Rt: np.ndarray) -> dict:
    """H, H_10m and log VR for deseasonalized return rows (NaN already set to 0)."""
    X = np.concatenate([np.zeros((len(Rt), 1)), np.cumsum(Rt, axis=1)], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vr = Rt.sum(1) ** 2 / (Rt ** 2).sum(1)
    return {"H": cof_hurst(X), "H_10m": cof_hurst(X, step=2), "log_vr": np.log(np.maximum(vr, 1e-12))}


def fgn_cholesky(H: float, n: int = BARS) -> np.ndarray:
    k = np.arange(n, dtype=float)
    g = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k - 1) ** (2 * H))
    C = g[np.abs(np.subtract.outer(np.arange(n), np.arange(n)))]
    return np.linalg.cholesky(C + 1e-12 * np.eye(n))


def vol_path(Rt: np.ndarray, valid: np.ndarray, win: int = VOL_WIN) -> np.ndarray:
    """Centered rolling RMS of each row; 0 where the return was missing."""
    sq = pd.DataFrame(Rt.T ** 2)
    v = np.sqrt(sq.rolling(win, center=True, min_periods=win // 2).mean().to_numpy().T)
    v = np.where(np.isfinite(v), v, np.nanmedian(v))
    return np.where(valid, v, 0.0)


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return np.nan
    return float(pd.Series(a[ok]).rank().corr(pd.Series(b[ok]).rank()))


def _lag_pairs(sessions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = pd.to_datetime(sessions)
    nxt = np.flatnonzero((d[1:] - d[:-1]) == pd.Timedelta(days=1))
    return nxt, nxt + 1


def asset_gates(sessions: np.ndarray, R: np.ndarray, clock: np.ndarray, *, n_rep: int = N_REP,
                seed: int = SEED, H0: float | None = None) -> dict:
    """Observed statistics and their constant-H null for one asset's blocks (rows of R)."""
    valid = np.isfinite(R)
    keep = valid.sum(1) >= MIN_BARS
    sessions, R, valid = sessions[keep], R[keep], valid[keep]
    Rt = np.where(valid, R / clock[None, :], 0.0)
    obs = block_stats(Rt)
    if H0 is None:
        H0 = float(np.median(obs["H"]))
    i0, i1 = _lag_pairs(sessions)
    o = {"n_blocks": int(len(sessions)), "H0": H0, "H_mean": float(np.mean(obs["H"])),
         "H_sd": float(np.std(obs["H"], ddof=1)), "H10_mean": float(np.mean(obs["H_10m"])),
         "var_H": float(np.var(obs["H"], ddof=1)),
         "rho2": _spearman(obs["H"][i0], obs["H"][i1]),
         "rho3": _spearman(obs["H"][i0], obs["log_vr"][i1]),
         "mean_log_vr": float(np.mean(obs["log_vr"])), "n_pairs": int(len(i0))}
    # Tercile read of G3 (economic size, reported): mean VR_{d+1} by tercile of H_d
    t = pd.qcut(pd.Series(obs["H"][i0]).rank(method="first"), 3, labels=["low", "mid", "high"])
    vr_next = np.exp(obs["log_vr"][i1])
    o.update({f"vr_next_{k}": float(vr_next[(t == k).to_numpy()].mean()) for k in ("low", "mid", "high")})
    sig = vol_path(Rt, valid)
    L = fgn_cholesky(H0)
    rng = np.random.default_rng(seed)
    var_n, rho2_n, rho3_n, hmean_n = (np.empty(n_rep) for _ in range(4))
    for k in range(n_rep):
        E = rng.standard_normal((len(Rt), BARS)) @ L.T
        s = block_stats(sig * E)
        var_n[k] = np.var(s["H"], ddof=1)
        rho2_n[k] = _spearman(s["H"][i0], s["H"][i1])
        rho3_n[k] = _spearman(s["H"][i0], s["log_vr"][i1])
        hmean_n[k] = s["H"].mean()
    p = lambda obs_v, null: float((1 + np.sum(null >= obs_v)) / (len(null) + 1))
    o.update({"p1": p(o["var_H"], var_n), "excess_sd": float(np.sqrt(max(o["var_H"] - var_n.mean(), 0.0))),
              "null_H_sd": float(np.sqrt(var_n.mean())), "null_H_mean": float(hmean_n.mean()),
              "p2": p(o["rho2"], rho2_n), "rho2_null": float(np.nanmean(rho2_n)),
              "p3": p(o["rho3"], rho3_n), "rho3_null": float(np.nanmean(rho3_n))})
    o["blocks"] = pd.DataFrame({"session": sessions, **obs})
    return o


def holm(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def decide(table: pd.DataFrame, min_blocks: int = MIN_BLOCKS) -> dict:
    """Gate verdicts from the per-asset table (eligible assets only)."""
    t = table[table["n_blocks"] >= min_blocks].copy()
    if t.empty:
        return {"eligible_assets": 0, "G1": False, "G2": False, "G3": False, "verdict": "INSUFFICIENT_DATA"}
    half = len(t) / 2.0
    out = {"eligible_assets": int(len(t))}
    for g, pcol, extra in (("G1", "p1", lambda d: float(d["excess_sd"].median()) >= G1_MIN_EXCESS_SD),
                           ("G2", "p2", lambda d: float((d["rho2"] - d["rho2_null"]).median()) >= G2_MIN_EXCESS_RHO),
                           ("G3", "p3", lambda d: float((d["rho3"] - d["rho3_null"]).median()) > 0)):
        t[f"{pcol}_holm"] = holm(t[pcol].to_numpy())
        n_pass = int((t[f"{pcol}_holm"] < ALPHA).sum())
        out[f"{g}_assets_passing"] = n_pass
        out[g] = bool(n_pass >= half and extra(t))
    out["median_excess_sd"] = float(t["excess_sd"].median())
    out["median_excess_rho2"] = float((t["rho2"] - t["rho2_null"]).median())
    out["median_excess_rho3"] = float((t["rho3"] - t["rho3_null"]).median())
    out["development_all_pass"] = bool(out["G1"] and out["G2"] and out["G3"])
    return out


def registration() -> dict:
    return {"version": VERSION, "bars": BARS, "min_bars": MIN_BARS, "min_blocks": MIN_BLOCKS,
            "conf_min_blocks": CONF_MIN_BLOCKS, "dev_end": DEV_END, "conf_start": CONF_START, "n_rep": N_REP, "seed": SEED, "vol_win": VOL_WIN,
            "g1_min_excess_sd": G1_MIN_EXCESS_SD, "g2_min_excess_rho": G2_MIN_EXCESS_RHO, "alpha_holm": ALPHA,
            "verdict_rule": "PROCEED_TO_HMM iff G1,G2,G3 pass on development and G3 passes on confirmation"}


# --------------------------------------------------------------------------
# Self test
# --------------------------------------------------------------------------

def _synthetic(n_blocks: int, H_of_block, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    cache = {}
    R = np.empty((n_blocks, BARS))
    for d in range(n_blocks):
        H = round(float(H_of_block(d)), 2)
        if H not in cache:
            cache[H] = fgn_cholesky(H)
        vol = 0.002 * np.exp(0.5 * rng.normal())                  # block volatility level
        R[d] = vol * (cache[H] @ rng.standard_normal(BARS))
    sessions = pd.date_range("2025-01-01", periods=n_blocks, freq="D").strftime("%Y-%m-%d").to_numpy()
    return sessions, R


def selftest(n_rep: int = 60) -> dict:
    ones = np.ones(BARS)
    # Estimator recovers H on long paths
    for H in (0.3, 0.5, 0.7):
        L = fgn_cholesky(H)
        E = np.random.default_rng(1).standard_normal((300, BARS)) @ L.T
        h = block_stats(E)["H"].mean()
        assert abs(h - H) < 0.03, (H, h)
    # Constant H: no gate passes
    s, R = _synthetic(240, lambda d: 0.5, seed=2)
    c = asset_gates(s, R, ones, n_rep=n_rep, seed=3)
    assert c["p1"] > 0.05 and c["p2"] > 0.05, c
    # Constant H with intrablock volatility bursts: raw H dispersion rises, the null absorbs it
    rng = np.random.default_rng(11)
    Rb = np.empty((240, BARS))
    for d in range(240):
        v = np.ones(BARS)
        ctr, w = rng.integers(0, BARS), rng.integers(5, 60)
        v[max(0, ctr - w):ctr + w] *= 1 + 4 * rng.random()
        Rb[d] = 0.002 * v * rng.standard_normal(BARS)
    b = asset_gates(s, Rb, ones, n_rep=n_rep, seed=12)
    assert b["H_sd"] > 0.1 and b["p1"] > 0.05, b
    # Persistent regimes in H (runs of 20 days at 0.35 / 0.65): G1 and G2 detect them
    s, R = _synthetic(240, lambda d: 0.35 if (d // 20) % 2 == 0 else 0.65, seed=4)
    v = asset_gates(s, R, ones, n_rep=n_rep, seed=5)
    assert v["p1"] < 0.05 and v["p2"] < 0.05 and v["excess_sd"] > 0.05, v
    # ... and G3: high H today -> larger terminal move tomorrow
    assert v["p3"] < 0.05 and v["vr_next_high"] > v["vr_next_low"], v
    return {"version": VERSION, "const_p1": round(c["p1"], 3), "regime_p1": round(v["p1"], 3),
            "regime_rho2": round(v["rho2"], 3), "regime_rho3": round(v["rho3"], 3), "ok": True}
