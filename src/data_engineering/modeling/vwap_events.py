"""VWAP Events 01 — does a touch of an anchored VWAP precede an unusually violent move?

Pre-registration: discussions/VWAP_Events/001_claude_branch_preregistration.md.

Data: Binance USD-M perpetual 1-minute klines (tools/download_binance_um_klines.py), one parquet per
symbol with open_time (UTC, bar open), open, high, low, close, volume, quote_volume, taker_buy_volume.

Pipeline per asset and anchor
-----------------------------
1. Anchored VWAP through bar t:  V_t = sum(quote_volume) / sum(volume) since the anchor (exact on klines).
2. Touch at bar t: low_t <= V_{t-1} <= high_t (the level known at the bar's open), after the previous
   M_APPROACH closes all sat on one side of the level and at least one was >= D_APPROACH * sigma * sqrt(M)
   away from it (log distance). sigma = causal EWMA of 1m log returns (half-life EWMA_HALFLIFE).
   Events need minutes-since-anchor >= warm-up; a COOLDOWN of bars follows each event.
3. Outcomes from the close of bar t over k bars:
      R_k = sqrt(sum_{i=1..k} r_{t+i}^2) / (sigma_{t-1} sqrt(k))    (primary, "violence" = realized vol)
      M_k = |log C_{t+k} - log C_t| / (sigma_{t-1} sqrt(k))
      cross_k = price ends on the opposite side of the touched level from the approach side
   y = log R_k minus the asset's median log R_k for that UTC hour (time-of-day normalisation).
4. Placebos, same detector and outcomes:
      offset — the real VWAP shifted by a constant log offset per anchor window,
               |delta| ~ U(OFFSET_MIN, OFFSET_MAX) x sigma_anchor x sqrt(60), random sign;
      phase  — a VWAP of the same period anchored at a random phase (a time nobody anchors on).
5. Statistic: Delta = mean y(real touches) - mean y(placebo touches), pooled over assets, with a
   bootstrap over UTC days. Gates are evaluated on the test window only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

VERSION = "vwap_events v1.0"

TEST_START = pd.Timestamp("2026-04-01", tz="UTC")
ASSETS = ("BTC", "ETH", "SOL", "XRP", "AVAX", "TRX", "HYPE")


@dataclass(frozen=True)
class Anchor:
    name: str
    period_min: int          # anchor period in minutes
    phase_min: int           # real anchor phase from 00:00 UTC
    warmup_min: int          # no events before this many minutes after the anchor
    phase_placebo: tuple     # (lo, hi) minutes: random phase drawn here (mod period)


ANCHORS = (
    Anchor("session_0800", 1440, 8 * 60, 60, (60, 1380)),   # Deribit daily expiry 08:00 UTC
    Anchor("h4", 240, 0, 30, (30, 210)),                    # 00/04/08/12/16/20 UTC
)


@dataclass(frozen=True)
class Params:
    m_approach: int = 15
    d_approach: float = 1.0
    cooldown: int = 15
    ewma_halflife: int = 60
    sigma_min_periods: int = 240
    horizons: tuple = (5, 15, 30, 60)
    primary_h: int = 15
    n_placebo: int = 20
    offset_min: float = 0.5
    offset_max: float = 3.0
    n_boot: int = 2000
    family_alpha: float = 0.05
    min_effect: float = 1.10          # exp(Delta) needed on top of significance
    seed: int = 20261013
    anchors: tuple = field(default=ANCHORS)

    @property
    def n_comparisons(self) -> int:
        return len(self.anchors) * 2          # anchors x placebo types

    @property
    def ci_level(self) -> float:
        return 1 - self.family_alpha / self.n_comparisons


def registration(p: Params | None = None) -> dict:
    p = p or Params()
    d = asdict(p)
    d["anchors"] = [asdict(a) for a in p.anchors]
    d.update(version=VERSION, test_start=str(TEST_START), ci_level=p.ci_level,
             primary="y at primary_h, real vs each placebo type, pooled assets, test window",
             gates={"G1": "two-sided CI at ci_level of Delta excludes 0 from above (lower bound > 0)",
                    "G2": "exp(Delta) >= min_effect",
                    "pass": "G1 and G2 vs BOTH placebo types, per anchor"})
    return d


# ---------------------------------------------------------------- data

def prepare(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Full 1-minute grid, log close, returns, causal EWMA sigma."""
    d = df.copy()
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d = d.drop_duplicates("open_time").set_index("open_time").sort_index()
    grid = pd.date_range(d.index[0], d.index[-1], freq="1min")
    d = d.reindex(grid)
    d.index.name = "t"
    out = pd.DataFrame(index=d.index)
    for c in ("high", "low", "close"):
        out[c] = d[c].astype(float)
    out["volume"] = d["volume"].astype(float).fillna(0.0)
    out["quote_volume"] = d["quote_volume"].astype(float).fillna(0.0)
    out["logc"] = np.log(out["close"])
    out["r"] = out["logc"].diff()
    out["sigma"] = np.sqrt(out["r"].pow(2).ewm(halflife=p.ewma_halflife, ignore_na=True,
                                              min_periods=p.sigma_min_periods).mean())
    return out


def anchored_vwap(fr: pd.DataFrame, period_min: int, phase_min: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """VWAP through each bar since its anchor; minutes since anchor; anchor id."""
    mins = (fr.index.as_unit("ns").asi8 // 60_000_000_000).astype(np.int64)   # minutes since epoch
    rel = mins - int(round(phase_min))
    aid = np.floor_divide(rel, period_min)
    since = rel - aid * period_min
    g = pd.Series(aid, index=fr.index)
    cq = fr["quote_volume"].groupby(g).cumsum().to_numpy()
    cv = fr["volume"].groupby(g).cumsum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(cv > 0, cq / cv, np.nan)
    return v, since, aid


def offset_levels(fr: pd.DataFrame, vwap: np.ndarray, aid: np.ndarray, rng: np.random.Generator,
                  p: Params) -> np.ndarray:
    """Real VWAP shifted by a constant log offset per anchor window (placebo 'offset')."""
    u, first = np.unique(aid, return_index=True)
    sig0 = fr["sigma"].to_numpy()[np.maximum(first - 1, 0)]
    mag = rng.uniform(p.offset_min, p.offset_max, len(u)) * rng.choice([-1.0, 1.0], len(u))
    off = mag * sig0 * np.sqrt(60.0)
    per_bar = off[np.searchsorted(u, aid)]
    return vwap * np.exp(per_bar)


# ---------------------------------------------------------------- events

def detect_touches(fr: pd.DataFrame, level: np.ndarray, since: np.ndarray, warmup: int,
                   p: Params) -> tuple[np.ndarray, np.ndarray]:
    """Indices t of touch events and the approach side s (+1 above, -1 below)."""
    m = p.m_approach
    if warmup <= m:
        raise ValueError("warm-up must exceed the approach window")
    hi, lo = fr["high"].to_numpy(), fr["low"].to_numpy()
    logc, sig = fr["logc"].to_numpy(), fr["sigma"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        loglev = np.log(level)
    lev_prev = np.r_[np.nan, level[:-1]]
    sig_prev = np.r_[np.nan, sig[:-1]]

    touch = (lo <= lev_prev) & (hi >= lev_prev)
    gap = logc - loglev
    side = np.sign(gap)
    side[~np.isfinite(gap)] = 0
    pos = pd.Series((side > 0).astype(float)).rolling(m).sum().shift(1).to_numpy()
    neg = pd.Series((side < 0).astype(float)).rolling(m).sum().shift(1).to_numpy()
    maxd = pd.Series(np.abs(gap)).rolling(m).max().shift(1).to_numpy()
    s = np.where(pos == m, 1.0, np.where(neg == m, -1.0, 0.0))
    with np.errstate(invalid="ignore"):
        far = maxd >= p.d_approach * sig_prev * np.sqrt(m)
    cand = touch & (s != 0) & far & (since >= warmup) & np.isfinite(sig_prev) & np.isfinite(lev_prev)
    idx = np.flatnonzero(cand)
    keep, last = [], -10**12
    for i in idx:                       # greedy cooldown
        if i - last > p.cooldown:
            keep.append(i)
            last = i
    keep = np.asarray(keep, dtype=np.int64)
    return keep, s[keep]


def outcome_arrays(fr: pd.DataFrame, p: Params) -> dict:
    """Precomputed cumulative sums for O(1) outcomes at any t."""
    r = fr["r"].to_numpy()
    miss = ~np.isfinite(r)
    return {"c2": np.r_[0.0, np.cumsum(np.where(miss, 0.0, r * r))],
            "cm": np.r_[0, np.cumsum(miss)],
            "logc": fr["logc"].to_numpy(),
            "sig_prev": np.r_[np.nan, fr["sigma"].to_numpy()[:-1]],
            "hour": fr.index.hour.to_numpy(),
            "n": len(fr)}


def outcomes_at(oa: dict, idx: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """log R_k and log M_k at event bars idx (NaN where the window is incomplete)."""
    n = oa["n"]
    ok = idx + k < n
    j = np.where(ok, idx + k, idx)
    nmiss = oa["cm"][j + 1] - oa["cm"][idx + 1]
    ok &= nmiss == 0
    rv = np.sqrt(oa["c2"][j + 1] - oa["c2"][idx + 1])
    den = oa["sig_prev"][idx] * np.sqrt(k)
    with np.errstate(invalid="ignore", divide="ignore"):
        logR = np.log(rv / den)
        logM = np.log(np.abs(oa["logc"][j] - oa["logc"][idx]) / den)
    logR[~ok] = np.nan
    logM[~ok] = np.nan
    return logR, logM


def hour_baseline(oa: dict, k: int) -> np.ndarray:
    """Median log R_k by UTC hour over all bars (24 values)."""
    idx = np.arange(oa["n"] - k - 1)
    idx = idx[np.isfinite(oa["sig_prev"][idx])]
    logR, _ = outcomes_at(oa, idx, k)
    s = pd.Series(logR).groupby(oa["hour"][idx]).median()
    return s.reindex(range(24)).to_numpy()


def event_rows(fr: pd.DataFrame, oa: dict, base: dict, idx: np.ndarray, side: np.ndarray,
               level: np.ndarray, p: Params, **tags) -> pd.DataFrame:
    if len(idx) == 0:
        return pd.DataFrame()
    t = fr.index[idx]
    rows = {"t": t, "side": side}
    lev = np.r_[np.nan, level[:-1]][idx]
    for k in p.horizons:
        logR, logM = outcomes_at(oa, idx, k)
        b = base[k][oa["hour"][idx]]
        rows[f"y{k}"] = logR - b
        rows[f"logM{k}"] = logM
        j = np.minimum(idx + k, oa["n"] - 1)
        end_side = np.sign(oa["logc"][j] - np.log(lev))
        cross = (end_side == -side).astype(float)
        cross[~np.isfinite(logR)] = np.nan
        rows[f"cross{k}"] = cross
    out = pd.DataFrame(rows)
    for kk, vv in tags.items():
        out[kk] = vv
    return out


def run_asset(df: pd.DataFrame, asset: str, p: Params) -> tuple[pd.DataFrame, dict]:
    """All real and placebo events for one asset; plus coverage info."""
    fr = prepare(df, p)
    oa = outcome_arrays(fr, p)
    base = {k: hour_baseline(oa, k) for k in p.horizons}
    rng = np.random.default_rng(p.seed + sum(map(ord, asset)))
    parts = []
    for a in p.anchors:
        v, since, aid = anchored_vwap(fr, a.period_min, a.phase_min)
        i, s = detect_touches(fr, v, since, a.warmup_min, p)
        parts.append(event_rows(fr, oa, base, i, s, v, p, anchor=a.name, kind="real", draw=-1))
        for dr in range(p.n_placebo):
            lv = offset_levels(fr, v, aid, rng, p)
            i, s = detect_touches(fr, lv, since, a.warmup_min, p)
            parts.append(event_rows(fr, oa, base, i, s, lv, p, anchor=a.name, kind="offset", draw=dr))
            ph = a.phase_min + rng.uniform(*a.phase_placebo)
            vp, sp, _ = anchored_vwap(fr, a.period_min, ph)
            i, s = detect_touches(fr, vp, sp, a.warmup_min, p)
            parts.append(event_rows(fr, oa, base, i, s, vp, p, anchor=a.name, kind="phase", draw=dr))
    ev = pd.concat([x for x in parts if len(x)], ignore_index=True)
    ev["asset"] = asset
    ev["day"] = ev["t"].dt.floor("1D")
    ev["window"] = np.where(ev["t"] >= TEST_START, "test", "dev")
    info = {"asset": asset, "first_bar": str(fr.index[0]), "last_bar": str(fr.index[-1]),
            "bars": int(len(fr)), "missing_bars": int(fr["close"].isna().sum()),
            "all_bar_mean_y_primary": float(np.nanmean(
                outcomes_at(oa, np.arange(oa["n"] - p.primary_h - 1), p.primary_h)[0]
                - base[p.primary_h][oa["hour"][:oa["n"] - p.primary_h - 1]]))}
    return ev, info


# ---------------------------------------------------------------- statistics

def boot_delta(real: pd.DataFrame, plac: pd.DataFrame, col: str, p: Params,
               rng: np.random.Generator) -> dict:
    """Delta = mean(real) - mean(placebo) with a UTC-day bootstrap (days resampled jointly)."""
    r = real[["day", col]].dropna()
    q = plac[["day", col]].dropna()
    if len(r) < 10 or len(q) < 10:
        return {"n_real": len(r), "n_placebo": len(q), "delta": np.nan, "lo": np.nan, "hi": np.nan}
    days = np.union1d(r["day"].unique(), q["day"].unique())
    def agg(x):
        g = x.groupby("day")[col].agg(["sum", "count"]).reindex(days).fillna(0.0)
        return g["sum"].to_numpy(), g["count"].to_numpy()
    sr, nr = agg(r)
    sq, nq = agg(q)
    delta = sr.sum() / nr.sum() - sq.sum() / nq.sum()
    D = len(days)
    W = rng.multinomial(D, np.full(D, 1.0 / D), size=p.n_boot).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        bd = (W @ sr) / (W @ nr) - (W @ sq) / (W @ nq)
    a = (1 - p.ci_level) / 2
    lo, hi = np.nanquantile(bd, [a, 1 - a])
    return {"n_real": int(nr.sum()), "n_placebo": int(nq.sum()), "days": D,
            "mean_real": sr.sum() / nr.sum(), "mean_placebo": sq.sum() / nq.sum(),
            "delta": delta, "lo": lo, "hi": hi, "ratio": float(np.exp(delta))}


def readout(ev: pd.DataFrame, p: Params, col: str | None = None,
            by: tuple = ("anchor", "window")) -> pd.DataFrame:
    col = col or f"y{p.primary_h}"
    rng = np.random.default_rng(p.seed)
    out = []
    for key, g in ev.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        real = g[g["kind"] == "real"]
        for kind in ("offset", "phase"):
            res = boot_delta(real, g[g["kind"] == kind], col, p, rng)
            out.append({**dict(zip(by, key)), "placebo": kind, "outcome": col, **res})
    df = pd.DataFrame(out)
    if col.startswith("y") or col.startswith("logM"):
        df["G1"] = df["lo"] > 0
        df["G2"] = df["ratio"] >= p.min_effect
    return df


def decision(primary: pd.DataFrame) -> pd.DataFrame:
    """Per anchor on the test window: pass iff G1 and G2 against both placebo types."""
    t = primary[primary["window"] == "test"]
    return (t.groupby("anchor")
             .apply(lambda g: pd.Series({"G1_both": bool(g["G1"].all()), "G2_both": bool(g["G2"].all()),
                                         "pass": bool((g["G1"] & g["G2"]).all())}),
                    include_groups=False)
             .reset_index())


# ---------------------------------------------------------------- synthetic data

def synthetic_klines(n_days: int = 60, seed: int = 0, start: str = "2026-01-01",
                     base_vol: float = 0.0006) -> pd.DataFrame:
    """1m klines with intraday volatility seasonality and slow vol clustering, no VWAP effect."""
    rng = np.random.default_rng(seed)
    n = n_days * 1440
    t = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    h = t.hour.to_numpy() + t.minute.to_numpy() / 60
    season = 1 + 0.5 * np.exp(-((h - 14.5) ** 2) / 2) + 0.3 * np.exp(-((h - 8) ** 2) / 2)
    lv = np.zeros(n)
    e = rng.standard_normal(n)
    for i in range(1, n):
        lv[i] = 0.999 * lv[i - 1] + 0.02 * e[i]
    sig = base_vol * season * np.exp(lv)
    r = sig * rng.standard_t(5, n) / np.sqrt(5 / 3)
    return klines_from_returns(t, r, rng, sig / base_vol)


def klines_from_returns(t, r, rng, vol_scale=None) -> pd.DataFrame:
    n = len(r)
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[100.0, c[:-1]]
    wig = np.abs(rng.standard_normal((n, 2))) * np.abs(r).mean() * 0.5
    hi = np.maximum(o, c) * np.exp(wig[:, 0])
    lo = np.minimum(o, c) * np.exp(-wig[:, 1])
    vs = np.ones(n) if vol_scale is None else vol_scale
    vol = rng.gamma(2.0, 50.0, n) * vs
    typ = (hi + lo + c) / 3
    return pd.DataFrame({"open_time": t, "open": o, "high": hi, "low": lo, "close": c,
                         "volume": vol, "quote_volume": vol * typ,
                         "taker_buy_volume": vol * rng.uniform(0.3, 0.7, n)})

