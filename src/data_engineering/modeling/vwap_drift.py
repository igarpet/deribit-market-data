"""VWAP Drift 01: is there conditional drift of price TOWARD the anchored VWAP, once displacement is measured in
volatility-engine units and conditioned on the engine's volatility regime and on recent jumps?

Protocol: discussions/VWAP_Drift/001_claude_vwd01_preregistration.md (fixed before any real-data output).

Question (Ivan, 2026-10-24): buy one option in the direction of VWAP when price is displaced from it. A long
option only pays if price actually moves toward VWAP by more than premium + spread, so the prerequisite is
    E[ r_{t->t+h} * sign(-D_t) | state_t ] > costs,      D_t = log C_t - log VWAP_t.
This module measures that drift on the perpetual, before any option is involved.

Data: Binance USD-M perpetual 1-minute klines (as VWAP Events). Anchored VWAP = cumulative quote_volume /
volume since the anchor through bar t (vwap_events.anchored_vwap). Real anchors: session (08:00 UTC, Deribit
expiry) and 4H (00/04/../20 UTC). Placebo anchors: same period, phases nobody uses (vwap_touch_stats.SPECS).

Volatility engine: the frozen VD18 HAR_4 forecasts (notebooks/Volatility_Dynamics/18/18_manifest.json, no
refit), i.e. the variance level of AM-T-HAR-Blend (blend weight ~1 at 30-240 min). Inputs are 5-minute closes
built from the perpetual klines (VD18 was fitted on spot; spot-perp basis s.d. is ~1 bp, Pairs fOU 004).
    v_h  = net-return variance forecast over h minutes, h in {15, 60, 240}, as of the 5m bar ending at t
    EXP  : horizon to the next 08:00 UTC expiry, tau in [15, 1440] min; variance by log-linear interpolation
           of the per-minute rates v_h/h over h in {15, 60, 240}, flat beyond 240 (approximation, disclosed)

Origins: every 1m bar whose END is on the 5-minute grid, >= WARMUP minutes after its anchor.
    Z_h      = D_t / sqrt(v_h)                                   displacement in engine units for horizon h
    zb       = |Z_h| bucket: z0_1 (<1, descriptive only), z1_2, z2_3, z3p (>= 3)
    vol      = tercile of log v_60, cut-offs from the asset's DEV origins (low / mid / high)
    jump     = max |1m log return| over the last 60 bars >= 4 * sqrt(v_15 / 15)   (SJ 04b-style trigger)
Outcome (entry at the close of bar t+1, one-minute latency as VWAP Events):
    y_h = -sign(D_t) * (log C_exit - log C_{t+1}) * 1e4      bp, positive = price moved toward VWAP
    exit = close h minutes after entry (15/60/240) or the bar ending at the next 08:00 (EXP)
    hit  = price reached the decision-time VWAP level after entry (1m high/low), reported only
    p0   = 2 (1 - Phi(|log C_{t+1} - log VWAP_t| / sigma_h)), driftless reflection benchmark, reported only

Decision (pre-registered): 144 real cells = anchor(2) x horizon(4) x zb(z1_2,z2_3,z3p) x vol(3) x jump(2).
    DEV  (< 2026-04-01): select the cell with the highest day-bootstrap q2.5 of mean y (n_days >= 30,
         n >= 500 origins).
    TEST (>= 2026-04-01), selected cell only:
         G1 VWAP-specific   hour-matched real - placebo, q2.5 > 0
         G2 tradeable perp  real mean y, q2.5 > 7 bp
         verdict VWAP_DRIFT_EDGE (G1&G2) / VWAP_SPECIFIC_BUT_UNTRADEABLE / GENERIC_DRIFT_NOT_VWAP / NO_EDGE
    Options gate (reported): an ATM option (delta ~0.5) pays 2 x 0.03% of the underlying in fees per round
    trip, i.e. >= 12 bp per unit of delta before the spread. An options module is considered only if G1 and
    G2 pass AND the TEST mean exceeds 12 bp + the measured half-spreads / delta.
Secondary S1 (Ivan's volatility-regime hypothesis, fixed now): vol = high, zb in {z2_3, z3p}, both jump
states, each anchor x horizon (8 tests), TEST real mean and hour-matched real - placebo with Bonferroni
two-sided intervals at 1 - 0.05/8.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import vwap_events as ve
from . import vwap_touch_stats as vt

VERSION = "VWD01-0.1.0"
TEST_START = pd.Timestamp("2026-04-01", tz="UTC")
ASSETS = ("BTC", "ETH", "SOL", "XRP")
ENGINE_H = (15, 60, 240)
HORIZONS = ("15", "60", "240", "EXP")
EXPIRY_HOUR = 8
ZB = ("z0_1", "z1_2", "z2_3", "z3p")
ZB_CELLS = ("z1_2", "z2_3", "z3p")
VOL = ("low", "mid", "high")
JUMP = (0, 1)
ANCHORS = ("session", "h4")
KEYS = ["asset", "anchor", "kind", "h", "zb", "vol", "jump", "day", "hblk"]
CELL = ["anchor", "h", "zb", "vol", "jump"]
SUMS = ["n", "sy", "syy", "hit", "p0"]


@dataclass(frozen=True)
class Params:
    grid_min: int = 5
    warmup_min: int = 30
    entry_lag: int = 1
    min_exp_min: int = 15
    jump_k: float = 4.0
    jump_lookback: int = 60
    vol_h: int = 60
    tradeable_bp: float = 7.0
    option_fee_floor_bp: float = 12.0
    min_dev_days: int = 30
    min_dev_n: int = 500
    min_test_days: int = 20
    n_boot: int = 2000
    n_boot_table: int = 400
    s1_alpha: float = 0.05
    seed: int = 20261024
    status: str = "PREREGISTERED_VWD001"


def registration(p: Params | None = None) -> dict:
    p = p or Params()
    d = asdict(p)
    d.update(version=VERSION, test_start=str(TEST_START), assets=list(ASSETS), horizons=list(HORIZONS),
             anchors=[{"name": s.name, "kind": s.kind, "period_min": s.period_min, "phase_min": s.phase_min}
                      for s in vt.SPECS],
             cells="anchor(2) x horizon(4) x zb(z1_2,z2_3,z3p) x vol(3) x jump(2) = 144 real cells",
             selection="DEV real cell with max day-bootstrap q2.5 of mean y (n_days >= 30, n >= 500); "
                       "TEST needs >= 20 days or the verdict is INSUFFICIENT_TEST",
             gates={"G1": "TEST hour-matched real - placebo q2.5 > 0",
                    "G2": f"TEST real mean y q2.5 > {p.tradeable_bp} bp"},
             options_gate=f"G1 & G2 and TEST mean > {p.option_fee_floor_bp} bp + measured half-spreads/delta",
             secondary="S1: vol=high, zb in {z2_3,z3p}, 8 anchor x horizon tests, Bonferroni 1-0.05/8",
             engine="frozen VD18 HAR_4 (15/60/240), inputs = perp 5m closes; EXP by log-linear rate interpolation")
    return d


# ---------------------------------------------------------------- volatility engine (frozen VD18 HAR_4)

def load_engine(manifest_path: str | Path) -> dict:
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if data.get("parameters", {}).get("version") != "VD18-0.1.0":
        raise ValueError("VD18 artifact version drift")
    fits = {}
    for h in ENGINE_H:
        f = data["parameters"]["fitted_har"][str(h)]["har_fits"]["HAR_4"]
        if f["lookbacks"] != [30, 60, 240, 1440]:
            raise ValueError("unexpected HAR_4 lookbacks")
        if list(f["design"]["features"]) != [f"har_logrv_{k}" for k in (30, 60, 240, 1440)]:
            raise ValueError("unexpected HAR_4 features")
        fits[h] = f
    return fits


def engine_frame(fr: pd.DataFrame) -> pd.DataFrame:
    """5m closes at bar END from the 1m grid (fr indexed by 1m open time) and VD18 HAR features (strict
    complete windows: any missing 5m return -> NaN)."""
    end = fr.index + pd.Timedelta(minutes=1)
    m = np.asarray(end.minute % 5 == 0)
    z = pd.DataFrame({"close": fr["close"].to_numpy()[m]}, index=end[m])
    z.index.name = "state_time"
    r = np.log(z["close"]).diff()
    gap = pd.Series(z.index, index=z.index).diff().ne(pd.Timedelta(minutes=5))
    r[gap.to_numpy()] = np.nan
    sq = r.pow(2)
    for L in (30, 60, 240, 1440):
        z[f"har_logrv_{L}"] = np.log(np.maximum(sq.rolling(L // 5, min_periods=L // 5).sum(), 1e-16))
    z["slot"] = (z.index.hour * 60 + z.index.minute) // 5
    return z


def predict_har(fit: dict, z: pd.DataFrame, asset: str) -> np.ndarray:
    """VD18 markov_har_variance.predict_har algebra on the frozen coefficients; NaN where features are
    incomplete (no substitution)."""
    d = fit["design"]
    if asset not in d["asset_names"]:
        raise ValueError(f"{asset} not in the VD18 fit")
    x = (z[d["features"]].to_numpy(float) - np.asarray(d["center"])) / np.asarray(d["std"])
    ang = 2 * np.pi * z["slot"].to_numpy(float) / 288
    extras = [np.full(len(z), float(asset == a)) for a in d["asset_names"][1:]]
    X = np.column_stack([np.ones(len(z)), x, np.sin(ang), np.cos(ang), *extras])
    ok = np.isfinite(X).all(axis=1)
    out = np.full(len(z), np.nan)
    out[ok] = np.exp(np.clip(X[ok] @ np.asarray(fit["coefficients"], float), -40, 5)) * fit["net_return_scales"][asset]
    return np.maximum(out, 1e-12)


def variance_to(tau: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Variance over tau minutes from v[:, (15, 60, 240)]: log per-minute rate linear in log tau between the
    engine horizons, flat outside [15, 240]."""
    hs = np.array(ENGINE_H, float)
    lr = np.log(v / hs)
    lx = np.log(hs)
    lt = np.log(np.clip(np.asarray(tau, float), hs[0], hs[-1]))
    j = (lt > lx[1]).astype(int)
    a = (lt - lx[j]) / (lx[j + 1] - lx[j])
    rows = np.arange(len(v))
    lrate = lr[rows, j] * (1 - a) + lr[rows, j + 1] * a
    return np.asarray(tau, float) * np.exp(lrate)


def next_expiry(end: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First 08:00 UTC strictly after each time."""
    base = (end - pd.Timedelta(hours=EXPIRY_HOUR)).floor("1D") + pd.Timedelta(hours=EXPIRY_HOUR)
    return base + pd.Timedelta(days=1)


# ---------------------------------------------------------------- per-asset origins and outcomes

def _fwd_min(x: np.ndarray, h: int) -> np.ndarray:
    """min(x[i .. i+h-1]) (NaN if incomplete)."""
    return pd.Series(x[::-1]).rolling(h, min_periods=h).min().to_numpy()[::-1]


def _fwd_max(x: np.ndarray, h: int) -> np.ndarray:
    return pd.Series(x[::-1]).rolling(h, min_periods=h).max().to_numpy()[::-1]


def origin_states(fr: pd.DataFrame, asset: str, fits: dict, p: Params) -> pd.DataFrame:
    """Per-origin engine variances, vol tercile, jump flag, expiry horizon. Index = position in fr."""
    end = fr.index + pd.Timedelta(minutes=1)
    orig = np.flatnonzero(np.asarray(end.minute % p.grid_min == 0))
    z = engine_frame(fr)
    if len(z) != len(orig):
        raise AssertionError("engine grid misaligned")
    V = np.column_stack([predict_har(fits[h], z, asset) for h in ENGINE_H])
    tend = end[orig]
    exp = next_expiry(tend)
    tau = np.asarray((exp - tend) / pd.Timedelta(minutes=1), dtype=np.int64)
    absr = np.abs(fr["r"].to_numpy())
    rmax = pd.Series(absr).rolling(p.jump_lookback, min_periods=p.jump_lookback).max().to_numpy()[orig]
    sig1 = np.sqrt(V[:, 0] / 15.0)
    lv = np.log(V[:, ENGINE_H.index(p.vol_h)])
    dev = np.asarray(tend < TEST_START)
    okv = np.isfinite(lv)
    if (dev & okv).sum() >= 100:
        cut = np.quantile(lv[dev & okv], [1 / 3, 2 / 3])
    else:                                   # synthetic / short data: whole-sample cut-offs (flagged in info)
        cut = np.quantile(lv[okv], [1 / 3, 2 / 3]) if okv.any() else np.array([np.nan, np.nan])
    vol = np.where(okv, np.digitize(lv, cut), -1)
    return pd.DataFrame({"i": orig, "tend": tend, "tau": tau, "v15": V[:, 0], "v60": V[:, 1], "v240": V[:, 2],
                         "vol": vol, "jump": np.where(np.isfinite(rmax) & np.isfinite(sig1),
                                                      (rmax >= p.jump_k * sig1).astype(int), -1)}), cut


def run_asset(df: pd.DataFrame, asset: str, fits: dict, p: Params | None = None,
              specs=vt.SPECS, keep_rows: bool = False):
    """Day x hour-block aggregates for every (anchor spec, horizon, zb, vol, jump). keep_rows=True also
    returns the per-origin rows (tests only)."""
    p = p or Params()
    fr = ve.prepare(df, ve.Params())
    n = len(fr)
    logc = fr["logc"].to_numpy()
    lo, hi = fr["low"].to_numpy(), fr["high"].to_numpy()
    O, cut = origin_states(fr, asset, fits, p)
    i = O["i"].to_numpy()
    e = i + p.entry_lag
    # exits and hit windows (bars e+1 .. exit)
    exits, wmin, wmax, sig = {}, {}, {}, {}
    V = O[["v15", "v60", "v240"]].to_numpy()
    for hs in HORIZONS:
        if hs == "EXP":
            tau = O["tau"].to_numpy()
            x = i + tau
            ok = tau >= p.min_exp_min
            # suffix extremes within each expiry segment (bars whose END is in (E-1d, E])
            seg = np.asarray(next_expiry(fr.index + pd.Timedelta(minutes=1) - pd.Timedelta(seconds=1)).asi8)
            smin = pd.Series(lo[::-1]).groupby(seg[::-1]).cummin().to_numpy()[::-1]
            smax = pd.Series(hi[::-1]).groupby(seg[::-1]).cummax().to_numpy()[::-1]
            st = np.minimum(e + 1, n - 1)
            wmin[hs], wmax[hs] = smin[st], smax[st]
            sig[hs] = np.sqrt(variance_to(np.maximum(tau, 1), V))
        else:
            h = int(hs)
            x = e + h
            ok = np.ones(len(i), bool)
            fm, fM = _fwd_min(lo, h), _fwd_max(hi, h)
            st = np.minimum(e + 1, n - 1)
            wmin[hs], wmax[hs] = fm[st], fM[st]
            sig[hs] = np.sqrt(V[:, ENGINE_H.index(h)])
        ok &= x < n
        exits[hs] = (np.where(ok, x, 0), ok)
    day = np.asarray((O["tend"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.days, dtype=np.int32)
    hblk = (O["tend"].dt.hour.to_numpy() // 6).astype(np.int8)
    aggs, rows_out = [], []
    for sp in specs:
        vw, since, _ = ve.anchored_vwap(fr, sp.period_min, sp.phase_min)
        with np.errstate(invalid="ignore", divide="ignore"):
            lvw = np.log(vw[i])
        D = logc[i] - lvw
        s = np.sign(D)
        base_ok = (since[i] >= p.warmup_min) & np.isfinite(D) & (s != 0) & (O["vol"].to_numpy() >= 0) \
            & (O["jump"].to_numpy() >= 0) & np.isfinite(logc[np.minimum(e, n - 1)]) & (e < n)
        for hs in HORIZONS:
            x, okx = exits[hs]
            sg = sig[hs]
            y = -s * (logc[x] - logc[np.minimum(e, n - 1)]) * 1e4
            Z = D / sg
            ok = base_ok & okx & np.isfinite(y) & np.isfinite(Z)
            if not ok.any():
                continue
            lev = vw[i]
            crossed = np.sign(logc[np.minimum(e, n - 1)] - lvw) != s
            reach = np.where(s > 0, wmin[hs] <= lev, wmax[hs] >= lev)
            hit = (crossed | reach).astype(float)
            dist = np.abs(logc[np.minimum(e, n - 1)] - lvw)
            p0 = np.where(crossed, 1.0, np.minimum(1.0, 2 * (1 - norm.cdf(dist / sg))))
            az = np.abs(Z)
            zb = np.digitize(az, [1.0, 2.0, 3.0])
            R = pd.DataFrame({"zb": zb[ok].astype(np.int8), "vol": O["vol"].to_numpy()[ok].astype(np.int8),
                              "jump": O["jump"].to_numpy()[ok].astype(np.int8), "day": day[ok], "hblk": hblk[ok],
                              "n": 1, "sy": y[ok], "syy": y[ok] ** 2, "hit": hit[ok], "p0": p0[ok]})
            g = R.groupby(["zb", "vol", "jump", "day", "hblk"], sort=False)[SUMS].sum().reset_index()
            g["anchor"], g["kind"], g["h"] = sp.name, sp.kind, hs
            aggs.append(g)
            if keep_rows:
                rows_out.append(pd.DataFrame({"i": i[ok], "anchor": sp.name, "kind": sp.kind, "phase": sp.phase_min,
                                              "h": hs, "D": D[ok], "Z": Z[ok], "y": y[ok], "hit": hit[ok],
                                              "p0": p0[ok], "exit": x[ok]}))
    A = pd.concat(aggs, ignore_index=True)
    A["asset"] = asset
    A["zb"] = pd.Categorical.from_codes(A["zb"], ZB)
    A["vol"] = pd.Categorical.from_codes(A["vol"], VOL)
    A["test"] = A["day"] >= int((TEST_START - pd.Timestamp("1970-01-01", tz="UTC")).days)
    info = {"asset": asset, "first_bar": str(fr.index[0]), "last_bar": str(fr.index[-1]), "bars": n,
            "missing_bars": int(fr["close"].isna().sum()), "origins": int(len(O)),
            "engine_nan_share": float(np.isnan(V).any(axis=1).mean()),
            "vol_cutoffs_log_v60": [float(c) for c in cut], "dev_origins": int((O["tend"] < TEST_START).sum())}
    if keep_rows:
        return A, info, O, pd.concat(rows_out, ignore_index=True)
    return A, info


# ---------------------------------------------------------------- statistics (day-block bootstrap)

def _cube(A: pd.DataFrame, days: np.ndarray, col: str = "sy") -> tuple[np.ndarray, np.ndarray]:
    """(days x 4 hour blocks) sums of col and counts."""
    S = np.zeros((len(days), 4)); N = np.zeros((len(days), 4))
    if len(A):
        d = np.searchsorted(days, A["day"].to_numpy())
        b = A["hblk"].to_numpy().astype(int)
        np.add.at(S, (d, b), A[col].to_numpy(float))
        np.add.at(N, (d, b), A["n"].to_numpy(float))
    return S, N


def _weights(n_days: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_days, np.full(n_days, 1.0 / n_days), size=n_boot).astype(float)


def boot_mean(A: pd.DataFrame, n_boot: int, seed: int, q=(0.025, 0.975)) -> dict:
    days = np.unique(A["day"].to_numpy())
    if len(days) < 2:
        return {"n": int(A["n"].sum()) if len(A) else 0, "days": int(len(days)), "mean": np.nan, "lo": np.nan, "hi": np.nan}
    S, N = _cube(A, days)
    s, nn = S.sum(1), N.sum(1)
    W = _weights(len(days), n_boot, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = (W @ s) / (W @ nn)
    lo, hi = np.nanquantile(bs, q)
    return {"n": int(nn.sum()), "days": int(len(days)), "mean": float(s.sum() / nn.sum()), "lo": float(lo), "hi": float(hi)}


def boot_diff_hour(real: pd.DataFrame, plc: pd.DataFrame, n_boot: int, seed: int, q=(0.025, 0.975)) -> dict:
    """Hour-block-matched real - placebo: within each 6h block of the origin, mean(real) - mean(placebo),
    weighted by real counts; joint day bootstrap over the union of days."""
    days = np.unique(np.r_[real["day"].to_numpy(), plc["day"].to_numpy()])
    if len(days) < 2 or not len(real) or not len(plc):
        return {"diff": np.nan, "diff_lo": np.nan, "diff_hi": np.nan, "placebo_mean": np.nan}
    Sr, Nr = _cube(real, days); Sp, Np = _cube(plc, days)

    def stat(sr, nr, sp, np_):
        with np.errstate(invalid="ignore", divide="ignore"):
            mr, mp = sr / nr, sp / np_
        w = np.where(np.isfinite(mr) & np.isfinite(mp), nr, 0.0)
        tot = w.sum(-1)
        return np.where(tot > 0, (w * (np.nan_to_num(mr) - np.nan_to_num(mp))).sum(-1) / np.where(tot > 0, tot, 1), np.nan)

    est = float(stat(Sr.sum(0), Nr.sum(0), Sp.sum(0), Np.sum(0)))
    W = _weights(len(days), n_boot, seed)
    bs = stat(W @ Sr, W @ Nr, W @ Sp, W @ Np)
    lo, hi = np.nanquantile(bs, q)
    return {"diff": est, "diff_lo": float(lo), "diff_hi": float(hi),
            "placebo_mean": float(Sp.sum() / Np.sum()) if Np.sum() else np.nan}


def _sel(A: pd.DataFrame, **kw) -> pd.DataFrame:
    m = np.ones(len(A), bool)
    for k, v in kw.items():
        if isinstance(v, (list, tuple, set)):
            m &= A[k].isin(list(v)).to_numpy()
        else:
            m &= (A[k] == v).to_numpy()
    return A[m]


def dev_table(A: pd.DataFrame, p: Params) -> pd.DataFrame:
    R = A[(A["kind"] == "real") & ~A["test"] & A["zb"].isin(ZB_CELLS)]
    out = []
    for k, (key, g) in enumerate(R.groupby(CELL, observed=True)):
        b = boot_mean(g, p.n_boot, p.seed + k)
        out.append({**dict(zip(CELL, key)), "dev_mean": b["mean"], "dev_lo": b["lo"], "dev_hi": b["hi"],
                    "dev_n": b["n"], "dev_days": b["days"],
                    "dev_hit": float(g["hit"].sum() / g["n"].sum()), "dev_p0": float(g["p0"].sum() / g["n"].sum())})
    return pd.DataFrame(out)


def select(dev: pd.DataFrame, p: Params) -> dict:
    d = dev[(dev["dev_days"] >= p.min_dev_days) & (dev["dev_n"] >= p.min_dev_n)].sort_values("dev_lo", ascending=False)
    if not len(d):
        return {}
    r = d.iloc[0]
    return {c: (int(r[c]) if c == "jump" else str(r[c])) for c in CELL} | {
        "dev_mean": float(r["dev_mean"]), "dev_lo": float(r["dev_lo"]), "dev_n": int(r["dev_n"]),
        "dev_days": int(r["dev_days"])}


def evaluate(A: pd.DataFrame, cell: dict, p: Params) -> dict:
    if not cell:
        return {"version": VERSION, "status": p.status, "verdict": "NO_ELIGIBLE_CELL"}
    key = {c: cell[c] for c in CELL}
    T = _sel(A[A["test"]], **key)
    real, plc = T[T["kind"] == "real"], T[T["kind"] == "placebo"]
    b = boot_mean(real, p.n_boot, p.seed + 101)
    d = boot_diff_hour(real, plc, p.n_boot, p.seed + 102)
    g1 = "PASS" if (np.isfinite(d["diff_lo"]) and d["diff"] > 0 and d["diff_lo"] > 0) else "FAIL"
    g2 = "PASS" if (np.isfinite(b["lo"]) and b["lo"] > p.tradeable_bp) else "FAIL"
    if b["days"] < p.min_test_days:
        g1 = g2 = "INSUFFICIENT"
    verdict = ("INSUFFICIENT_TEST" if g1 == "INSUFFICIENT" else "VWAP_DRIFT_EDGE" if g1 == g2 == "PASS" else "VWAP_SPECIFIC_BUT_UNTRADEABLE" if g1 == "PASS"
               else "GENERIC_DRIFT_NOT_VWAP" if g2 == "PASS" else "NO_EDGE")
    nr = real["n"].sum()
    return {"version": VERSION, "status": p.status, "cell": cell, "test_n": b["n"], "test_days": b["days"],
            "test_mean_bp": b["mean"], "test_lo": b["lo"], "test_hi": b["hi"],
            "placebo_mean_bp": d["placebo_mean"], "real_minus_placebo_hour_matched": d["diff"],
            "diff_lo": d["diff_lo"], "diff_hi": d["diff_hi"],
            "test_hit_rate": float(real["hit"].sum() / nr) if nr else np.nan,
            "test_p0_mean": float(real["p0"].sum() / nr) if nr else np.nan,
            "G1_vwap_specific": g1, "G2_tradeable": g2,
            "options_fee_floor_bp": p.option_fee_floor_bp,
            "options_considered": bool(g1 == g2 == "PASS" and b["mean"] > p.option_fee_floor_bp),
            "verdict": verdict}


def secondary_s1(A: pd.DataFrame, p: Params) -> pd.DataFrame:
    a = p.s1_alpha / (len(ANCHORS) * len(HORIZONS))
    q = (a / 2, 1 - a / 2)
    T = A[A["test"] & (A["vol"] == "high") & A["zb"].isin(["z2_3", "z3p"])]
    out = []
    for k, (an, hs) in enumerate([(an, hs) for an in ANCHORS for hs in HORIZONS]):
        g = _sel(T, anchor=an, h=hs)
        real, plc = g[g["kind"] == "real"], g[g["kind"] == "placebo"]
        b = boot_mean(real, p.n_boot, p.seed + 300 + k, q)
        d = boot_diff_hour(real, plc, p.n_boot, p.seed + 400 + k, q)
        out.append({"anchor": an, "h": hs, "n": b["n"], "days": b["days"], "mean_bp": b["mean"], "lo": b["lo"],
                    "hi": b["hi"], **d, "ci_level": 1 - a,
                    "pass_vwap_specific": bool(np.isfinite(d["diff_lo"]) and d["diff_lo"] > 0),
                    "pass_tradeable": bool(np.isfinite(b["lo"]) and b["lo"] > p.tradeable_bp)})
    return pd.DataFrame(out)


def table(A: pd.DataFrame, by: list, p: Params, seed: int = 7) -> pd.DataFrame:
    """Descriptive: real and placebo means with day-bootstrap CI, hour-matched difference, hit vs p0."""
    out = []
    for k, (key, g) in enumerate(A.groupby(by + ["test"], observed=True)):
        key = key if isinstance(key, tuple) else (key,)
        real, plc = g[g["kind"] == "real"], g[g["kind"] == "placebo"]
        b = boot_mean(real, p.n_boot_table, seed + k)
        d = boot_diff_hour(real, plc, p.n_boot_table, seed + 5000 + k)
        nr, np_ = real["n"].sum(), plc["n"].sum()
        out.append({**dict(zip(by + ["test"], key)), "n": b["n"], "days": b["days"], "mean_bp": b["mean"],
                    "lo": b["lo"], "hi": b["hi"], "placebo_mean_bp": d["placebo_mean"], "diff": d["diff"],
                    "diff_lo": d["diff_lo"], "diff_hi": d["diff_hi"],
                    "sd_bp": float(np.sqrt(max(real["syy"].sum() / nr - (real["sy"].sum() / nr) ** 2, 0))) if nr else np.nan,
                    "hit": float(real["hit"].sum() / nr) if nr else np.nan,
                    "p0": float(real["p0"].sum() / nr) if nr else np.nan,
                    "hit_placebo": float(plc["hit"].sum() / np_) if np_ else np.nan})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- synthetic data

def synthetic_klines(n_days: int = 120, seed: int = 0, start: str = "2025-12-01", kappa: float = 0.0,
                     base_vol: float = 0.0006) -> pd.DataFrame:
    """1m klines with intraday seasonality and slow volatility clustering. kappa > 0 adds a per-minute pull
    toward the 08:00 session VWAP: r_t += -kappa * D_{t-1}. kappa = 0 is a martingale (no VWAP effect)."""
    rng = np.random.default_rng(seed)
    n = n_days * 1440
    t = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    h = t.hour.to_numpy() + t.minute.to_numpy() / 60
    season = 1 + 0.5 * np.exp(-((h - 14.5) ** 2) / 2) + 0.3 * np.exp(-((h - 8) ** 2) / 2)
    lv = np.zeros(n); ev = rng.standard_normal(n)
    for k in range(1, n):
        lv[k] = 0.999 * lv[k - 1] + 0.02 * ev[k]
    sig = base_vol * season * np.exp(lv)
    eps = rng.standard_t(5, n) / np.sqrt(5 / 3) * sig
    vol = rng.gamma(2.0, 50.0, n) * season
    sess = ((t - pd.Timedelta(hours=EXPIRY_HOUR)).floor("1D")).asi8
    c = np.empty(n); x = np.log(100.0); cq = cv = 0.0; prev = None
    for k in range(n):
        if sess[k] != prev:
            cq = cv = 0.0; prev = sess[k]
        drift = -kappa * (x - np.log(cq / cv)) if cv > 0 else 0.0
        x = x + drift + eps[k]
        c[k] = x
        cq += vol[k] * np.exp(x); cv += vol[k]
    o = np.r_[np.log(100.0), c[:-1]]
    wig = np.abs(rng.standard_normal((n, 2))) * sig[:, None] * 0.5
    hi = np.maximum(o, c) + wig[:, 0]
    lo = np.minimum(o, c) - wig[:, 1]
    return pd.DataFrame({"open_time": t, "open": np.exp(o), "high": np.exp(hi), "low": np.exp(lo),
                         "close": np.exp(c), "volume": vol, "quote_volume": vol * np.exp(c)})
