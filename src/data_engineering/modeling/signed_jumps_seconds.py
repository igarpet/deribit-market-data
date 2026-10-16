"""
Signed Jumps 04: what does a 1-minute jump look like at 1-second resolution?

Protocol: discussions/Signed_Jumps/007_claude_sj04_preregistration.md (fixed before any real-data output).

Jumps: SJ 01 engine on 1m spot, unchanged (c = 4, 5 bp floor). Seconds data: Binance spot 1s klines.
A jump minute is the bar ending at T, i.e. seconds (T-60s, T]. All times are bar-END times.

Onset second
    Signed cumulative return inside the jump minute, C(u) = s * (log P(T-60+u) - log P(T-60)), u = 1..60.
    onset = first u with C(u) >= 0.5 * C(60)   (the second by which half of the minute's move is done).

P1 (gate) early build-up
    a(t) = log(trades_t + 1) - log(mean trades over the previous 3600 s + 1)  (strictly before t).
    Event statistic: mean a over seconds [onset-120, onset-31]. Placebo: random seconds in minutes with no jump
    within +-10 minutes, matched by asset and UTC hour (3 per event). PASS iff mean(event) - mean(placebo) > 0
    with day-block bootstrap q2.5 > 0. A PASS means activity is elevated more than 30 s before the move.

P2 (reported, threshold stated) split events
    For pairs of consecutive same-sign jump minutes (t, t+1): the seconds of the largest signed 1s return in
    each minute. Split event iff the two are <= 30 s apart (one move straddling a minute boundary).
    Share of split pairs > 50% -> SJ 01's diagonal excitation is mostly a bar-boundary artefact.

P3 (gate) seconds-scale continuation
    Signed spot return from onset + 5 s to onset + 65 s (bp), cluster-starting jumps.
    PASS iff gross day-bootstrap q2.5 > 0. Net at 12 bp (perp-equivalent) and 22 bp (spot taker) reported.
    Full grid delta in {1, 5, 10, 30} s x H in {10, 30, 60, 300} s reported, never used for the decision.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

VERSION = "SJ04-0.1.0"
COLS = ("close", "trades", "quote_volume", "taker_buy_quote_volume")


@dataclass(frozen=True)
class SecParams:
    baseline_s: int = 3600
    profile: tuple = (-600, 300)
    p1_window: tuple = (-120, -31)
    split_gap_s: int = 30
    split_share_threshold: float = 0.5
    p3_delta_s: int = 5
    p3_h_s: int = 60
    grid_delta: tuple = (1, 5, 10, 30)
    grid_h: tuple = (10, 30, 60, 300)
    placebo_per_event: int = 3
    placebo_exclusion_min: int = 10
    cluster_gap_min: int = 5
    costs_bp: tuple = (12.0, 22.0)
    n_boot: int = 2000
    status: str = "PREREGISTERED_SJ007"


def registration(sp: SecParams | None = None) -> dict:
    sp = sp or SecParams()
    d = asdict(sp)
    d["version"] = VERSION
    d["gates"] = {"P1": "event-minus-placebo activity over [onset-120, onset-31] s: q2.5 > 0",
                  "P2": "share of consecutive same-sign jump minutes whose peak seconds are <= 30 s apart; > 0.5 => artefact",
                  "P3": "signed spot return onset+5 s -> onset+65 s: gross q2.5 > 0"}
    return d


# ----------------------------------------------------------------------------- data

def load_spot_1s(files) -> pd.DataFrame:
    """Concatenate monthly 1s files; index = bar END time (open_time + 1 s) on a full second grid."""
    parts = [pd.read_parquet(f, columns=["open_time", *COLS]) for f in files]
    df = pd.concat(parts, ignore_index=True)
    t = pd.to_datetime(df["open_time"], utc=True) + pd.Timedelta(seconds=1)
    out = df[list(COLS)].astype(float).set_index(t)
    out = out[~out.index.duplicated()].sort_index()
    grid = pd.date_range(out.index.min(), out.index.max(), freq="1s")
    out = out.reindex(grid)
    out["trades"] = out["trades"].fillna(0.0)          # no kline row = no trade in that second
    out["logp"] = np.log(out["close"].ffill())
    return out


def activity(sec: pd.DataFrame, baseline_s: int) -> np.ndarray:
    tr = sec["trades"].to_numpy()
    base = pd.Series(tr).rolling(baseline_s, min_periods=int(0.75 * baseline_s)).mean().to_numpy()
    base = np.concatenate([[np.nan], base[:-1]])        # strictly before t
    return np.log(tr + 1) - np.log(base + 1)


# ----------------------------------------------------------------------------- events

def jump_minutes(assets, pan, cluster_gap: int) -> pd.DataFrame:
    """All SJ 01 jumps with sign, bar-end time and cluster-starter flag."""
    rows = []
    time = pd.DatetimeIndex(pan.time)
    if time.tz is None:
        time = time.tz_localize("UTC")
    up, dn = pan.J["up"] > 0.5, pan.J["down"] > 0.5
    for i_a in range(pan.n_assets):
        idx = np.flatnonzero(pan.asset_idx == i_a)
        anyj = (up | dn)[idx]
        cs = np.concatenate([[0], np.cumsum(anyj)])
        for k in np.flatnonzero(anyj):
            i = idx[k]
            rows.append({"asset": pan.names[i_a], "T": time[i], "sign": 1 if up[i] else -1,
                         "starter": bool(cs[k] - cs[max(0, k - cluster_gap)] == 0), "minute_bp": float(pan.r[i] * 1e4)})
    return pd.DataFrame(rows)


def onset_seconds(ev: pd.DataFrame, secs: dict) -> pd.DataFrame:
    """Adds onset (Timestamp), onset_u (1..60), share_in_peak_second, peak_u."""
    ev = ev.copy()
    for col in ("onset_u", "peak_u", "peak_share"):
        ev[col] = np.nan
    onset = {}
    for a, g in ev.groupby("asset"):
        if a not in secs:
            continue
        lp = secs[a]["logp"]
        idx = lp.index
        v = lp.to_numpy()
        start = idx.get_indexer(pd.DatetimeIndex(g["T"]) - pd.Timedelta(seconds=60))
        for r, s0, sgn in zip(g.index, start, g["sign"]):
            if s0 < 0 or s0 + 60 >= len(v):
                continue
            path = sgn * (v[s0 + 1: s0 + 61] - v[s0])
            if not np.isfinite(path).all() or path[-1] <= 0:
                continue
            u = int(np.argmax(path >= 0.5 * path[-1])) + 1
            r1 = np.diff(np.concatenate([[0.0], path]))
            ev.loc[r, "onset_u"] = u
            onset[r] = idx[s0 + u]
            ev.loc[r, "peak_u"] = int(np.argmax(r1)) + 1
            ev.loc[r, "peak_share"] = float(r1.max() / path[-1])
    ev["onset"] = pd.Series(onset, dtype="datetime64[ns, UTC]").reindex(ev.index)
    return ev


def placebo_onsets(ev: pd.DataFrame, secs: dict, sp: SecParams, seed: int = 7) -> pd.DataFrame:
    """Random seconds in minutes with no jump within +-exclusion minutes, matched by asset and UTC hour."""
    rng = np.random.default_rng(seed)
    out = []
    for a, g in ev[ev.onset.notna()].groupby("asset"):
        if a not in secs:
            continue
        idx = secs[a].index
        jm = pd.DatetimeIndex(ev[ev.asset == a]["T"]).floor("min")
        bad = set()
        for k in range(-sp.placebo_exclusion_min, sp.placebo_exclusion_min + 2):
            bad.update((jm + pd.Timedelta(minutes=k)).asi8.tolist())
        lo, hi = -sp.profile[0] + sp.baseline_s + 1, len(idx) - sp.profile[1] - 400
        cand_pos = rng.integers(lo, hi, size=40 * len(g) * sp.placebo_per_event)
        cand_t = idx[cand_pos]
        ok = ~np.isin(cand_t.floor("min").asi8, np.fromiter(bad, dtype=np.int64))
        cand_pos, cand_t = cand_pos[ok], cand_t[ok]
        hrs = cand_t.hour
        for hr, n in pd.DatetimeIndex(g["onset"]).hour.value_counts().items():
            pool = cand_pos[hrs == hr]
            take = rng.choice(pool, size=min(len(pool), n * sp.placebo_per_event), replace=False) if len(pool) else []
            for pos in take:
                out.append({"asset": a, "onset": idx[pos]})
    return pd.DataFrame(out)


def _window_stat(secs: dict, act: dict, frame: pd.DataFrame, lo: int, hi: int) -> np.ndarray:
    vals = np.full(len(frame), np.nan)
    for a, g in frame.groupby("asset"):
        idx = secs[a].index
        pos = idx.get_indexer(pd.DatetimeIndex(g["onset"]))
        A = act[a]
        for j, (r, p) in enumerate(zip(g.index, pos)):
            if p + lo < 0 or p + hi >= len(A):
                continue
            w = A[p + lo: p + hi + 1]
            vals[frame.index.get_loc(r)] = np.nanmean(w)
    return vals


def p1_test(ev: pd.DataFrame, plc: pd.DataFrame, secs, act, sp: SecParams, seed: int = 11) -> dict:
    e = ev[ev.starter & ev.onset.notna()].reset_index(drop=True)
    x = _window_stat(secs, act, e, *sp.p1_window)
    y = _window_stat(secs, act, plc.reset_index(drop=True), *sp.p1_window)
    de = pd.DatetimeIndex(e["onset"]).normalize().asi8
    dp = pd.DatetimeIndex(plc["onset"]).normalize().asi8
    days = np.unique(np.concatenate([de, dp]))
    ie, ip = np.searchsorted(days, de), np.searchsorted(days, dp)
    okx, oky = np.isfinite(x), np.isfinite(y)
    Sx = np.bincount(ie[okx], weights=x[okx], minlength=len(days)); Nx = np.bincount(ie[okx], minlength=len(days))
    Sy = np.bincount(ip[oky], weights=y[oky], minlength=len(days)); Ny = np.bincount(ip[oky], minlength=len(days))
    rng = np.random.default_rng(seed)
    d = rng.integers(0, len(days), size=(sp.n_boot, len(days)))
    bs = Sx[d].sum(1) / np.maximum(Nx[d].sum(1), 1) - Sy[d].sum(1) / np.maximum(Ny[d].sum(1), 1)
    diff = float(np.nanmean(x) - np.nanmean(y))
    lo = float(np.quantile(bs, 0.025))
    return {"event_mean": float(np.nanmean(x)), "placebo_mean": float(np.nanmean(y)), "excess": diff,
            "q025": lo, "q975": float(np.quantile(bs, 0.975)), "n_events": int(okx.sum()), "n_placebo": int(oky.sum()),
            "status": "PASS" if diff > 0 and lo > 0 else "FAIL"}


def profile(ev: pd.DataFrame, plc: pd.DataFrame, secs, act, sp: SecParams, bin_s: int = 10) -> pd.DataFrame:
    """Mean activity by bin of seconds relative to onset: events (starters) vs placebo."""
    lo, hi = sp.profile
    rows = []
    for name, fr in (("event", ev[ev.starter & ev.onset.notna()]), ("placebo", plc)):
        acc = np.zeros(hi - lo + 1)
        cnt = np.zeros(hi - lo + 1)
        for a, g in fr.groupby("asset"):
            idx = secs[a].index
            A = act[a]
            pos = idx.get_indexer(pd.DatetimeIndex(g["onset"]))
            pos = pos[(pos + lo >= 0) & (pos + hi < len(A))]
            if not len(pos):
                continue
            M = A[pos[:, None] + np.arange(lo, hi + 1)[None, :]]
            acc += np.nansum(M, axis=0)
            cnt += np.isfinite(M).sum(axis=0)
        mean = acc / np.maximum(cnt, 1)
        for k0 in range(lo, hi + 1, bin_s):
            sl = slice(k0 - lo, min(k0 - lo + bin_s, hi - lo + 1))
            rows.append({"who": name, "bin_start_s": k0, "mean_activity": float(np.mean(mean[sl]))})
    t = pd.DataFrame(rows).pivot(index="bin_start_s", columns="who", values="mean_activity")
    t["excess"] = t["event"] - t["placebo"]
    return t.reset_index()


def split_pairs(ev: pd.DataFrame, sp: SecParams) -> tuple[pd.DataFrame, dict]:
    """Consecutive jump minutes on the same asset: gap (s) between their peak seconds."""
    e = ev[ev.peak_u.notna()].sort_values(["asset", "T"])
    rows = []
    for a, g in e.groupby("asset"):
        T = pd.DatetimeIndex(g["T"])
        for i in range(len(g) - 1):
            if T[i + 1] - T[i] == pd.Timedelta(minutes=1):
                gap = (60 - g.peak_u.iloc[i]) + g.peak_u.iloc[i + 1]
                rows.append({"asset": a, "T": T[i], "same_sign": bool(g.sign.iloc[i] == g.sign.iloc[i + 1]),
                             "gap_s": float(gap), "split": bool(gap <= sp.split_gap_s)})
    t = pd.DataFrame(rows)
    same = t[t.same_sign] if len(t) else t
    summ = {"pairs_same_sign": int(len(same)), "pairs_opposite": int(len(t) - len(same)),
            "split_share_same_sign": float(same.split.mean()) if len(same) else float("nan"),
            "split_share_opposite": float(t[~t.same_sign].split.mean()) if len(t) and (~t.same_sign).any() else float("nan"),
            "median_gap_same_sign_s": float(same.gap_s.median()) if len(same) else float("nan")}
    summ["verdict"] = ("MOSTLY_ARTEFACT" if summ["split_share_same_sign"] > sp.split_share_threshold
                       else "MOSTLY_DISTINCT_EVENTS")
    return t, summ


def post_onset(ev: pd.DataFrame, secs, sp: SecParams) -> pd.DataFrame:
    """Signed spot return (bp) from onset+delta to onset+delta+H for the grid; cluster starters."""
    e = ev[ev.starter & ev.onset.notna()].copy()
    for a, g in e.groupby("asset"):
        idx = secs[a].index
        v = secs[a]["logp"].to_numpy()
        pos = idx.get_indexer(pd.DatetimeIndex(g["onset"]))
        for d in sp.grid_delta:
            for h in sp.grid_h:
                p0, p1 = pos + d, pos + d + h
                ok = (pos >= 0) & (p1 < len(v))
                val = np.full(len(g), np.nan)
                val[ok] = g["sign"].to_numpy()[ok] * (v[p1[ok]] - v[p0[ok]]) * 1e4
                e.loc[g.index, f"r_{d}_{h}"] = val
    return e


def boot_mean(x, day, n_boot, seed):
    ok = np.isfinite(x)
    x, day = x[ok], day[ok]
    ud, inv = np.unique(day, return_inverse=True)
    S, N = np.bincount(inv, weights=x), np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    d = rng.integers(0, len(ud), size=(n_boot, len(ud)))
    bs = S[d].sum(1) / N[d].sum(1)
    return float(x.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), int(len(x))


def p3_table(e: pd.DataFrame, sp: SecParams) -> tuple[pd.DataFrame, dict]:
    day = pd.DatetimeIndex(e["onset"]).normalize().asi8
    rows = []
    for d in sp.grid_delta:
        for h in sp.grid_h:
            m, lo, hi, n = boot_mean(e[f"r_{d}_{h}"].to_numpy(), day, sp.n_boot, 3 + d + h)
            rows.append({"delta_s": d, "h_s": h, "gross_bp": m, "lo": lo, "hi": hi, "n": n,
                         "share_positive": float((e[f"r_{d}_{h}"] > 0).mean())})
    t = pd.DataFrame(rows)
    prim = t[(t.delta_s == sp.p3_delta_s) & (t.h_s == sp.p3_h_s)].iloc[0]
    res = {"gross_bp": prim.gross_bp, "q025": prim.lo, "q975": prim.hi, "n": int(prim.n),
           "status": "PASS" if prim.lo > 0 else "FAIL"}
    for c in sp.costs_bp:
        res[f"net_{c:g}bp"] = prim.gross_bp - c
    return t, res


# ----------------------------------------------------------------------------- synthetic

def simulate_seconds(n_days=12, n_assets=2, jump_rate_per_day=6.0, jump_bp=60.0, precursor=0.0,
                     precursor_s=120, drift_bp=0.0, drift_s=60, straddle_share=0.0, sigma_bp=0.8, seed=0,
                     start="2026-07-01", pair_share=0.0, pair_gap_s=90):
    """1s synthetic: Gaussian 1s returns, Poisson trades (precursor multiplies the trade rate by
    exp(precursor) over the precursor_s seconds before each jump), jumps of jump_bp concentrated in 3 seconds,
    optional post-jump drift. straddle_share of jumps are placed so their 3 seconds straddle a minute boundary.
    pair_share of jumps get a distinct second same-sign jump pair_gap_s seconds later (genuine excitation).
    Returns (secs: name -> DataFrame like load_spot_1s output, one_min: name -> 1m log close Series (bar end))."""
    rng = np.random.default_rng(seed)
    n = n_days * 86400
    t = pd.date_range(start, periods=n, freq="1s", tz="UTC") + pd.Timedelta(seconds=1)
    secs, one = {}, {}
    for a in range(n_assets):
        r = rng.normal(0, sigma_bp / 1e4, n)
        lam = np.full(n, 3.0)
        nj = rng.poisson(jump_rate_per_day * n_days)
        pos = np.sort(rng.integers(5000, n - 2000, nj))
        sgn = rng.choice([-1, 1], nj)
        for p, s in zip(pos, sgn):
            if rng.random() < straddle_share:
                p = p - (p % 60) + 59          # second 59 of a minute (bar-end index): straddles the boundary
            r[p: p + 3] += s * jump_bp / 3 / 1e4
            if pair_share and rng.random() < pair_share:
                q = p + pair_gap_s
                r[q: q + 3] += s * jump_bp / 3 / 1e4
            if drift_bp:
                r[p + 3: p + 3 + drift_s] += s * drift_bp / drift_s / 1e4
            if precursor:
                lam[max(0, p - precursor_s): p] *= np.exp(precursor)
        trades = rng.poisson(lam).astype(float)
        lp = np.log(100) + np.cumsum(r)
        df = pd.DataFrame({"close": np.exp(lp), "trades": trades, "quote_volume": trades * 1e3,
                           "taker_buy_quote_volume": trades * 5e2}, index=t)
        df["logp"] = lp
        secs[f"S{a}"] = df
        one[f"S{a}"] = df["logp"].iloc[59::60]
    return secs, one
