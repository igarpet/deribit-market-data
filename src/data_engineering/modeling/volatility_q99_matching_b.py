"""
Volatility Dynamics 04b: repaired volatility-conditioned q99 precursor test.

Why Module 04 has to be redone
------------------------------
Module 04 matched each q99 onset on six single-bar log r^2 values, choosing 8
controls out of the ~15 eligible rows at the same exact UTC slot (one per day
of data). Its own diagnostic showed the matching did not condition on the
volatility environment: realized_vol_30m retained 100% and realized_vol_1h 95%
of their clock-only deviation. Candidate "survival" was therefore expected
under the null of pure volatility clustering.

04b changes
-----------
1. Causal multi-horizon volatility environment, strictly before the origin:
     M1 (returns only): log mean r^2 over bars -3..-1, -12..-1, -48..-1,
        -288..-1 (15m, 1h, 4h, 24h rates);
     M2 = M1 + log mean intrabar range over -3..-1 and -12..-1.
2. Candidate controls: same asset, Module 03 eligibility, UTC clock within
   +/-30 min (circular), across ALL dates, >= 4h from the event (the two
   -120m windows never overlap): ~220-250 candidates per event instead of ~15.
   The 8 nearest in robust-scaled environment distance are kept.
3. Bias correction (Abadie-Imbens style). q99 onsets sit in the extreme
   upper tail of the environment, so even the nearest controls are calmer.
   For every asset, lag and feature, the feature (on a monotone log scale for
   non-negative fields) is regressed on the environment over all non-event
   origins; each control value is then shifted by beta'(env_event -
   env_control) before the event is ranked among the controls.
4. No best-lag selection. Pre-registered statistic W: window mean
   percentile over -60..-5 min (rel bars -12..-1): mean over lags per
   asset-event, mean within a timestamp, mean across timestamps.
5. Placebo null (day relabel, as in 03b) through the identical procedure.

Designs: clock_only (= Module 03 controls), M1_matched (no correction,
diagnostic), M1_bc, M2_bc.

Why M2_bc is primary (decided on synthetic data before any real 04b run)
------------------------------------------------------------------------
On synthetic GARCH-type panels where nothing but latent volatility drives
volume and range (``vb.synthetic_frame``), M1_bc removes the realized-vol
deviation (retention 0.02-0.13) yet volume and range still score W ~ 0.61-0.65:
returns are a noisier estimator of latent volatility than range, so
conditioning on return RV alone leaves volatility-proxy "survival" that is
pure confounding. Under M2_bc volume falls to ~0.51 (the noise-feature
level). Therefore:
- every candidate except intrabar_range_bps is judged under M2_bc;
- intrabar_range_bps is reported under M1_bc for information only and is
  labelled "volatility_estimator_not_testable": on the volatility-only
  synthetic world it also survives M1 (W ~ 0.61), because it is a more
  efficient volatility estimator than the return RV being conditioned on.

Pre-registered gate (fixed before any real-data 04b output exists)
-------------------------------------------------------------------
MATCHING GATE: retention = |W_design - 0.5| / |W_clock_only - 0.5| of
realized_vol_1h AND realized_vol_4h must be <= 0.20 under BOTH M1_bc and
M2_bc. These are near-mechanical (built from the same returns), so a failure
means the procedure does not condition on volatility and NO candidate result
may be interpreted.

Candidate reading rule: a field "survives volatility conditioning" only if
its W in its judging design differs from the placebo distribution of that
design with Holm-adjusted p < 0.05 (two-sided; normal approximation to the
placebo distribution; Holm across all judged fields) AND lies on the same
side of 0.5 as its clock-only W.

Survival is necessary, not sufficient, for a beyond-volatility precursor. On
a synthetic database where every activity/IV field is driven only by latent
volatility, fields that measure latent volatility MORE precisely than return
RV + range history (high-count activity, smooth IV level) also survive. A
surviving field must next be tested in a forecast comparison (06b style):
does it improve q99 forecasts more than it improves ordinary next-hour
volatility forecasts?
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from data_engineering.modeling import volatility_vd_b_core as vb


VERSION = "0.1.0"

PRIMARY_CANDIDATES = (
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
ENVIRONMENT_CHECKS = ("realized_vol_30m", "realized_vol_1h", "realized_vol_4h")
CONTRAST_CHECKS = (
    "trade_flow_imbalance",
    "signed_volume_usd",
    "open_interest_return_5m",
    "perpetual_mid_vs_index_bps",
)
GATE_FEATURES = ("realized_vol_1h", "realized_vol_4h")
GATE_MAX_RETENTION = 0.20
GATE_DESIGNS = ("M1_bc", "M2_bc")

WINDOW_BARS = tuple(range(-12, 0))          # -60 .. -5 minutes
PROFILE_BARS = tuple(range(-24, 25))        # -120 .. +120 minutes
M1_REQUIRED = ("env_lrv_15m", "env_lrv_1h", "env_lrv_4h")
M2_REQUIRED = M1_REQUIRED + ("env_lrange_15m", "env_lrange_1h")
DEFAULT_CLOCK_WINDOW_SLOTS = 6
DEFAULT_MIN_SEPARATION_BARS = 48
DEFAULT_MAX_CONTROLS = 8
DEFAULT_MIN_CONTROLS = 4


def analysis_features(available: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    available = set(map(str, available))
    rows = []
    for f in PRIMARY_CANDIDATES:
        rows.append({"feature": f, "role": "primary_candidate", "present": f in available})
    for f in ENVIRONMENT_CHECKS:
        rows.append({"feature": f, "role": "environment_check", "present": f in available})
    for f in CONTRAST_CHECKS:
        rows.append({"feature": f, "role": "contrast_check", "present": f in available})
    c = pd.DataFrame(rows)
    c["in_matching_set_M2"] = c["feature"].eq("intrabar_range_bps")
    return c, c.loc[c.present, "feature"].tolist()


def environment(grid: vb.Grid) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    rng_vals = (
        pd.to_numeric(grid.frame["intrabar_range_bps"], errors="coerce").to_numpy(float)
        if "intrabar_range_bps" in grid.frame else None
    )
    env, specs = vb.volatility_environment(grid, range_values=rng_vals)
    spec_df = pd.DataFrame([
        {"column": k, "start_bar": s, "end_bar": e, "min_finite_bars": m}
        for k, (s, e, m) in specs.items()
    ])
    return env.to_numpy(float), list(env.columns), spec_df


DESIGNS = ("clock_only", "M1_matched", "M1_bc", "M2_bc")
PRIMARY_DESIGN = "M2_bc"
RANGE_DESIGN = "M1_bc"


def _design_specs(env_cols: list[str]) -> dict[str, dict]:
    """Which environment columns each design matches on / adjusts with."""
    m1 = [c for c in env_cols if c.startswith("env_lrv_")]
    specs = {
        "clock_only": {"match": None, "adjust": None},
        "M1_matched": {"match": m1, "required": list(M1_REQUIRED), "adjust": None},
        "M1_bc": {"match": m1, "required": list(M1_REQUIRED), "adjust": list(M1_REQUIRED)},
    }
    if all(c in env_cols for c in M2_REQUIRED):
        specs["M2_bc"] = {"match": list(env_cols), "required": list(M2_REQUIRED), "adjust": list(M2_REQUIRED)}
    return specs


def prepare_adjustment(
    grid: vb.Grid,
    Z: np.ndarray,
    env: np.ndarray,
    env_cols: list[str],
    eligible: np.ndarray,
    rel_bars,
) -> dict[str, np.ndarray]:
    """Fit bias-correction slopes once per adjusting design (non-event origins)."""
    out = {}
    for name, sp in _design_specs(env_cols).items():
        if sp["adjust"]:
            idx = [env_cols.index(c) for c in sp["adjust"]]
            out[name] = vb.fit_adjustment(grid, Z, env[:, idx], eligible, list(rel_bars))
    return out


def control_sets(
    grid: vb.Grid,
    event_rows: np.ndarray,
    eligible: np.ndarray,
    env: np.ndarray,
    env_cols: list[str],
    *,
    max_controls: int = DEFAULT_MAX_CONTROLS,
    clock_window_slots: int = DEFAULT_CLOCK_WINDOW_SLOTS,
    min_separation_bars: int = DEFAULT_MIN_SEPARATION_BARS,
    scales: np.ndarray | None = None,
) -> dict[str, tuple[np.ndarray, pd.DataFrame | None]]:
    """Controls per design. M1_matched and M1_bc share the same control set."""
    sc_full = scales if scales is not None else vb.robust_asset_scales(grid, env)
    out: dict[str, tuple[np.ndarray, pd.DataFrame | None]] = {}
    cache: dict[tuple, tuple] = {}
    for name, sp in _design_specs(env_cols).items():
        if sp["match"] is None:
            out[name] = (vb.clock_controls(grid, event_rows, eligible, max_controls=max_controls), None)
            continue
        key = tuple(sp["match"])
        if key not in cache:
            idx = [env_cols.index(c) for c in sp["match"]]
            req = [sp["match"].index(c) for c in sp["required"]]
            cache[key] = vb.matched_controls(
                grid, event_rows, eligible, env[:, idx], required_dims=req,
                max_controls=max_controls, clock_window_slots=clock_window_slots,
                min_separation_bars=min_separation_bars, scales=sc_full[:, idx],
            )
        out[name] = cache[key]
    return out


def _tensor(grid, Z, event_rows, cc, rel_bars, env, env_cols, name, adj, min_controls, rel_index=None):
    sp = _design_specs(env_cols)[name]
    if sp["adjust"] and name in adj:
        idx = [env_cols.index(c) for c in sp["adjust"]]
        betas = adj[name] if rel_index is None else adj[name][rel_index]
        return vb.percentile_tensor(
            grid, Z, event_rows, cc, rel_bars, min_controls=min_controls,
            env=env[:, idx], betas=betas,
        )
    return vb.percentile_tensor(grid, Z, event_rows, cc, rel_bars, min_controls=min_controls)


def window_stat(grid: vb.Grid, P_window: np.ndarray, event_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per feature: mean over timestamps of (mean over window lags per asset-event)."""
    with np.errstate(invalid="ignore"):
        cnt = np.isfinite(P_window).sum(axis=1)
        ev_mean = np.nansum(np.where(np.isfinite(P_window), P_window, 0.0), axis=1) / np.maximum(cnt, 1)
    ev_mean[cnt == 0] = np.nan
    _, T = vb.collapse_by_time(grid.time_ns[event_rows], ev_mean[:, None, :])
    T = T[:, 0, :]
    n = np.isfinite(T).sum(axis=0)
    with np.errstate(invalid="ignore"):
        m = np.nansum(np.where(np.isfinite(T), T, 0.0), axis=0) / np.maximum(n, 1)
    m[n == 0] = np.nan
    return m, n


def observed_study(
    grid: vb.Grid,
    X: np.ndarray,
    features: list[str],
    event_rows: np.ndarray,
    eligible: np.ndarray,
    env: np.ndarray,
    env_cols: list[str],
    *,
    min_controls: int = DEFAULT_MIN_CONTROLS,
    **kw,
) -> dict:
    Z, scale_kinds = vb.natural_scale(X)
    adj = prepare_adjustment(grid, Z, env, env_cols, eligible, PROFILE_BARS)
    sets = control_sets(grid, event_rows, eligible, env, env_cols, **kw)
    win_idx = [PROFILE_BARS.index(b) for b in WINDOW_BARS]
    profiles, summary, balance = [], [], []
    for name, (cc, sup) in sets.items():
        P = _tensor(grid, Z, event_rows, cc, PROFILE_BARS, env, env_cols, name, adj, min_controls)
        uniq, T = vb.collapse_by_time(grid.time_ns[event_rows], P)
        n_ts = np.isfinite(T).sum(axis=0)
        with np.errstate(invalid="ignore"):
            mp = np.nansum(np.where(np.isfinite(T), T, 0.0), axis=0) / np.maximum(n_ts, 1)
        for j, b in enumerate(PROFILE_BARS):
            for fj, f in enumerate(features):
                profiles.append({
                    "control_type": name, "feature": f, "rel_bar": b, "rel_min": 5 * b,
                    "is_precursor": b < 0, "n_unique_event_times": int(n_ts[j, fj]),
                    "mean_percentile": float(mp[j, fj]) if n_ts[j, fj] else np.nan,
                })
        wm, wn = window_stat(grid, P[:, win_idx, :], event_rows)
        for fj, f in enumerate(features):
            summary.append({
                "control_type": name, "feature": f,
                "window_mean_percentile": float(wm[fj]) if wn[fj] else np.nan,
                "window_n_unique_event_times": int(wn[fj]),
            })
        for k, col in enumerate(env_cols):
            ev = env[event_rows, k]
            cv = np.where(cc >= 0, env[np.maximum(cc, 0), k], np.nan)
            cnt = np.isfinite(cv).sum(axis=1)
            cm = np.full(len(ev), np.nan)
            for i in np.flatnonzero(cnt > 0):
                cm[i] = np.median(cv[i][np.isfinite(cv[i])])
            dd = ev - cm
            dd = dd[np.isfinite(dd)]
            balance.append({
                "control_type": name, "env_column": col, "n_events": int(len(dd)),
                "median_event_minus_control": float(np.median(dd)) if len(dd) else np.nan,
                "mean_abs_event_minus_control": float(np.mean(np.abs(dd))) if len(dd) else np.nan,
                "note": "bias correction acts on features, not on this raw env balance" if name.endswith("_bc") else "",
            })
    s = pd.DataFrame(summary)
    base = s[s.control_type == "clock_only"].set_index("feature")["window_mean_percentile"]
    s["abs_dev_from_half"] = (s["window_mean_percentile"] - 0.5).abs()
    ret = []
    for r in s.itertuples():
        bv = base.get(r.feature, np.nan)
        ok = np.isfinite(bv) and abs(bv - 0.5) > 0 and np.isfinite(r.window_mean_percentile)
        ret.append(abs(r.window_mean_percentile - 0.5) / abs(bv - 0.5) if ok else np.nan)
    s["retention_vs_clock_only"] = ret
    supports = []
    seen = set()
    for name, (cc, sup) in sets.items():
        if sup is not None and id(sup) not in seen:
            seen.add(id(sup))
            supports.append(sup.assign(control_set=name.replace("_bc", "").replace("_matched", "")))
    return {
        "controls": {k: v[0] for k, v in sets.items()},
        "support": pd.concat(supports, ignore_index=True) if supports else pd.DataFrame(),
        "profiles": pd.DataFrame(profiles),
        "window_summary": s,
        "balance": pd.DataFrame(balance),
        "adjustment": adj,
        "Z": Z,
        "scale_kinds": dict(zip(features, scale_kinds)),
    }


def placebo_window_null(
    grid: vb.Grid,
    X: np.ndarray,
    event_rows: np.ndarray,
    eligible: np.ndarray,
    env: np.ndarray,
    env_cols: list[str],
    *,
    n_perm: int = 200,
    seed: int = 4040,
    min_controls: int = DEFAULT_MIN_CONTROLS,
    progress_every: int = 0,
    Z: np.ndarray | None = None,
    adjustment: dict | None = None,
    **kw,
) -> dict[str, np.ndarray]:
    """(n_perm, n_feat) placebo window means for each design."""
    rng = np.random.default_rng(int(seed))
    days = vb.valid_days(grid)
    scales = vb.robust_asset_scales(grid, env)
    if Z is None:
        Z, _ = vb.natural_scale(X)
    if adjustment is None:
        adjustment = prepare_adjustment(grid, Z, env, env_cols, eligible, PROFILE_BARS)
    rel_index = np.array([PROFILE_BARS.index(b) for b in WINDOW_BARS])
    out: dict[str, list] = {}
    for b in range(int(n_perm)):
        pl = vb.day_relabel_events(grid, event_rows, eligible, days, rng)
        sets = control_sets(grid, pl, eligible, env, env_cols, scales=scales, **kw)
        for name, (cc, _) in sets.items():
            P = _tensor(grid, Z, pl, cc, WINDOW_BARS, env, env_cols, name, adjustment, min_controls, rel_index=rel_index)
            wm, _ = window_stat(grid, P, pl)
            out.setdefault(name, []).append(wm)
        if progress_every and (b + 1) % int(progress_every) == 0:
            print(f"placebo {b + 1}/{n_perm}")
    return {k: np.vstack(v) for k, v in out.items()}


def attach_null(summary: pd.DataFrame, features: list[str], null: dict[str, np.ndarray]) -> pd.DataFrame:
    s = summary.copy()
    lo, hi, p = [], [], []
    for r in s.itertuples():
        arr = null.get(r.control_type)
        if arr is None:
            lo.append(np.nan); hi.append(np.nan); p.append(np.nan)
            continue
        x = arr[:, features.index(r.feature)]
        x = x[np.isfinite(x)]
        if not len(x) or not np.isfinite(r.window_mean_percentile):
            lo.append(np.nan); hi.append(np.nan); p.append(np.nan)
            continue
        lo.append(float(np.quantile(x, 0.025)))
        hi.append(float(np.quantile(x, 0.975)))
        obs = abs(r.window_mean_percentile - 0.5)
        p.append(float((1 + np.sum(np.abs(x - 0.5) >= obs)) / (len(x) + 1)))
    s["placebo_q025"] = lo
    s["placebo_q975"] = hi
    s["placebo_p_two_sided"] = p
    mu, sd = [], []
    for r in s.itertuples():
        arr = null.get(r.control_type)
        x = arr[:, features.index(r.feature)] if arr is not None else np.array([])
        x = x[np.isfinite(x)]
        mu.append(float(np.mean(x)) if len(x) > 2 else np.nan)
        sd.append(float(np.std(x, ddof=1)) if len(x) > 2 else np.nan)
    s["placebo_mean"] = mu
    s["placebo_sd"] = sd
    with np.errstate(invalid="ignore", divide="ignore"):
        s["placebo_z"] = (s["window_mean_percentile"] - s["placebo_mean"]) / s["placebo_sd"]
    s["placebo_p_normal"] = [_two_sided_normal_p(z) for z in s["placebo_z"]]
    return s


def _two_sided_normal_p(z: float) -> float:
    if not np.isfinite(z):
        return np.nan
    # erfc-based normal tail without scipy
    from math import erfc, sqrt
    return float(erfc(abs(z) / sqrt(2.0)))


def holm(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not len(ok):
        return out
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[j]))
        out[j] = running
    return out


def judging_design(feature: str) -> str:
    return RANGE_DESIGN if feature == "intrabar_range_bps" else PRIMARY_DESIGN


def matching_gate(summary: pd.DataFrame) -> dict:
    ret = {}
    ok = True
    for d in GATE_DESIGNS:
        m = summary[summary.control_type == d].set_index("feature")
        for f in GATE_FEATURES:
            v = float(m["retention_vs_clock_only"].get(f, np.nan)) if len(m) else np.nan
            ret[f"{d}:{f}"] = v
            ok = ok and np.isfinite(v) and v <= GATE_MAX_RETENTION
    return {
        "gate": f"retention of {list(GATE_FEATURES)} <= {GATE_MAX_RETENTION} under {list(GATE_DESIGNS)}",
        "retention": ret,
        "matching_gate_pass": bool(ok),
        "judging_design_default": PRIMARY_DESIGN,
        "judging_design_intrabar_range": RANGE_DESIGN,
    }


def candidate_verdicts(summary_with_null: pd.DataFrame, contract: pd.DataFrame, gate_pass: bool,
                       alpha: float = 0.05) -> pd.DataFrame:
    """Apply the pre-registered reading rule to every primary/contrast field.

    survives_vol_conditioning  iff  Holm-adjusted placebo p < alpha (two-sided,
    normal approximation to the placebo distribution of W, Holm across all
    judged fields) AND W on the same side of 0.5 as its clock-only W.

    "Survives" is necessary, not sufficient, for a beyond-volatility precursor:
    a field that measures latent volatility MORE PRECISELY than return RV and
    range history (e.g. a high-count activity measure or a smooth IV level)
    also survives. Separating the two needs a forecast test (06b style).
    """
    s = summary_with_null.set_index(["control_type", "feature"])
    rows = []
    for r in contract[contract.present & contract.role.isin(["primary_candidate", "contrast_check"])].itertuples():
        d = judging_design(r.feature)
        if (d, r.feature) not in s.index or ("clock_only", r.feature) not in s.index:
            continue
        x = s.loc[(d, r.feature)]
        c = s.loc[("clock_only", r.feature)]
        rows.append({
            "feature": r.feature, "role": r.role, "judging_design": d,
            "clock_only_W": c.window_mean_percentile, "judging_W": x.window_mean_percentile,
            "placebo_q025": x.placebo_q025, "placebo_q975": x.placebo_q975,
            "placebo_z": x.placebo_z, "placebo_p_normal": x.placebo_p_normal,
            "n_unique_event_times": x.window_n_unique_event_times,
            "retention_vs_clock_only": x.retention_vs_clock_only,
        })
    v = pd.DataFrame(rows)
    if v.empty:
        return v
    judged = v["feature"] != "intrabar_range_bps"
    v["holm_p"] = np.nan
    v.loc[judged, "holm_p"] = holm(v.loc[judged, "placebo_p_normal"].to_numpy(float))
    verdicts = []
    for r in v.itertuples():
        same_side = bool(np.isfinite(r.judging_W) and np.isfinite(r.clock_only_W)
                         and np.sign(r.judging_W - 0.5) == np.sign(r.clock_only_W - 0.5))
        if not gate_pass:
            verdicts.append("not_interpretable_gate_failed")
        elif r.feature == "intrabar_range_bps":
            # More efficient volatility estimator than return RV: survives M1 by
            # construction on volatility-only synthetic data. Not testable here.
            verdicts.append("volatility_estimator_not_testable")
        elif np.isfinite(r.holm_p) and r.holm_p < alpha and same_side:
            verdicts.append("survives_vol_conditioning")
        elif np.isfinite(r.holm_p) and r.holm_p < alpha:
            verdicts.append("opposite_sign_to_clock_only")
        else:
            verdicts.append("explained_by_volatility_or_noise")
    v["verdict"] = verdicts
    return v


def selftest() -> dict:
    """Synthetic panel where ONLY latent volatility drives range and volume."""
    f = vb.synthetic_frame(n_days=20, seed=21)
    out = []
    for a, g in f.groupby("underlying"):
        g = g.sort_values("state_time").copy()
        r2 = g["return_5m"] ** 2
        g["realized_vol_1h"] = np.sqrt(r2.rolling(12, min_periods=9).sum())
        g["realized_vol_4h"] = np.sqrt(r2.rolling(48, min_periods=36).sum())
        out.append(g)
    f = pd.concat(out, ignore_index=True)
    g = vb.build_grid(f)
    thr = vb.q99_thresholds(g)
    tail = vb.tail_flags(g, thr)
    ev = np.flatnonzero(vb.onset_flags(g, tail))
    elig = vb.eligible_origins(g, tail)
    env, cols, _ = environment(g)
    feats = ["realized_vol_1h", "realized_vol_4h", "intrabar_range_bps", "volume_usd", "noise_feature"]
    X = g.feature_matrix(feats)
    res = observed_study(g, X, feats, ev, elig, env, cols)
    s = res["window_summary"].set_index(["control_type", "feature"])
    assert s.loc[("clock_only", "volume_usd"), "window_mean_percentile"] > 0.75
    assert s.loc[("M2_bc", "volume_usd"), "window_mean_percentile"] < 0.56
    gate = matching_gate(res["window_summary"])
    assert gate["matching_gate_pass"]
    nul = placebo_window_null(g, X, ev, elig, env, cols, n_perm=30, seed=2, Z=res["Z"], adjustment=res["adjustment"])
    full = attach_null(res["window_summary"], feats, nul)
    contract = pd.DataFrame({"feature": ["volume_usd", "noise_feature"],
                             "role": ["primary_candidate", "contrast_check"], "present": [True, True]})
    v = candidate_verdicts(full, contract, gate["matching_gate_pass"]).set_index("feature")
    assert v.loc["volume_usd", "verdict"] != "survives_vol_conditioning"
    assert v.loc["noise_feature", "verdict"] == "explained_by_volatility_or_noise"
    sup = res["support"]
    assert sup[sup.control_set == "M1"]["n_candidate_pool"].median() > 50
    return {
        "bias_corrected_matching": True,
        "gate_passes_on_vol_only_world": True,
        "volume_not_flagged_when_only_vol_drives_it": True,
        "placebo_null": True,
    }
