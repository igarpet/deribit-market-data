"""Regime Structure 01 — do Ivan's regimes exist on crypto as persistent states beyond volatility?

Pre-registration: discussions/Regime_Structure/001_claude_branch_preregistration.md.

Data: Binance USD-M perpetual 1-minute klines (tools/download_binance_um_klines.py) for BTC, ETH, SOL, XRP,
aggregated to 5-minute bars; labels are computed on clock-hour windows of n = 12 bars.

Hourly quantities (window = bars t-11..t of one clock hour; sigma measured before the window)
    net   = sum r5                     path = sum |r5|            ER = |net| / path
    s     = sigma_pre * sqrt(12)       sigma_pre: EWMA of r5^2 (half-life 24 bars) at the bar before the window
    shock = max |r5| / sigma_pre       rvol = volume / dev median volume of that UTC hour
    rrv   = sqrt(sum r5^2) / dev median of the same for that UTC hour
    delta = sum (2 * taker_buy_volume - volume)   (window CVD change)

Labels, first match wins:
    SHOCK  shock >= 3
    DISP   ER >= q67(ER) and |net| >= 1.0 s and sign(delta) == sign(net)
    NOISE  rvol <= q25(rvol) and rrv <= q33(rrv)
    RANGE  ER <= q33(ER) and |net| < 1.0 s
    OTHER  otherwise
Quantiles are taken per asset on the development window of the real data and frozen (applied unchanged to
the null).

Null: every 1-minute log return and its taker delta are multiplied by an independent random sign. This keeps
1m |returns|, volume and time of day exactly (so volatility and activity clustering survive) and destroys
all directional dependence (trend, reversion, CVD-price persistence).

Tests (test window): lift(X -> Y) = P(Y_{w+1} | X_w) / P(Y_{w+1}) between consecutive non-overlapping windows
(needs >= 30 'from' windows; a null draw below that counts as lift 1 = no persistence), pooled as the
geometric mean of per-asset lifts, real vs N_NULL null draws (same pooling draw by draw).
    E1 RANGE -> RANGE     E2 DISP -> DISP
    C1 RANGE -> DISP      C2 DISP -> NOISE      C3 upDISP -> SHOCK
Pass: one-sided permutation p <= alpha and lift_real / median(lift_null) >= 1.10.

Window scales (v1.2, RS 003): the same labels and tests on clock-aligned windows of 15 min, 30 min, 1 h, 2 h and
4 h (3, 6, 12, 24, 48 five-minute bars). Two families:
    primary  1 h (as pre-registered in RS 001)          alpha = 0.05 / 5
    sweep    15 min, 30 min, 2 h, 4 h                    alpha = 0.05 / 20
Quantile thresholds are computed separately per scale (development window, real data).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

VERSION = "regime_structure v1.2"
# v1.2 (RS 003, before any v1.1 run): window-scale sweep 15m/30m/1h/2h/4h in one pass; 399 null draws.
# v1.1 (after the first real run, RS 002): null signs are SHARED across assets (one random sign per UTC minute),
# so the null keeps the cross-asset dependence of the real data; v1.0 drew independent signs per asset, which
# makes the pooled null too narrow and the pooled p-values anti-conservative. Adds shock direction (descriptive).
NULL_EPOCH = pd.Timestamp("2024-01-01", tz="UTC")
TEST_START = pd.Timestamp("2026-04-01", tz="UTC")
ASSETS = ("BTC", "ETH", "SOL", "XRP")
LABELS = ("SHOCK", "DISP", "NOISE", "RANGE", "OTHER")
TESTS = (("E1", "RANGE", "RANGE"), ("E2", "DISP", "DISP"), ("C1", "RANGE", "DISP"),
         ("C2", "DISP", "NOISE"), ("C3", "upDISP", "SHOCK"))


@dataclass(frozen=True)
class Params:
    bar_min: int = 5
    n_bars: int = 12                 # primary scale: one clock hour
    scales: tuple = (3, 6, 12, 24, 48)   # window lengths in 5m bars: 15m, 30m, 1h, 2h, 4h
    ewma_halflife: int = 24          # 5m bars (2 h)
    sigma_min_periods: int = 96
    shock_k: float = 3.0
    move_k: float = 1.0
    er_lo_q: float = 0.33
    er_hi_q: float = 0.67
    rvol_q: float = 0.25
    rrv_q: float = 0.33
    n_null: int = 399
    family_alpha: float = 0.05
    min_effect: float = 1.10
    seed: int = 20261013

    @property
    def alpha(self) -> float:
        return self.family_alpha / len(TESTS)

    @property
    def sweep_scales(self) -> tuple:
        return tuple(n for n in self.scales if n != self.n_bars)

    @property
    def alpha_sweep(self) -> float:
        return self.family_alpha / (len(TESTS) * max(len(self.sweep_scales), 1))


def registration(p: Params | None = None) -> dict:
    p = p or Params()
    d = asdict(p)
    d.update(version=VERSION, test_start=str(TEST_START), alpha_per_test=p.alpha, alpha_sweep=p.alpha_sweep,
             scale_minutes={n: n * p.bar_min for n in p.scales}, labels=list(LABELS),
             tests=[{"id": i, "from": x, "to": y} for i, x, y in TESTS],
             rule="pass iff p <= alpha_per_test and lift_real / median(lift_null) >= min_effect, test window, pooled")
    return d


# ---------------------------------------------------------------- minutes -> hours

@dataclass
class Minutes:
    t0: pd.Timestamp        # first minute (aligned to a clock hour)
    r1: np.ndarray          # 1m log returns (0 where missing)
    d1: np.ndarray          # 1m taker delta
    v1: np.ndarray          # 1m volume
    n_missing: int


def prepare_minutes(df: pd.DataFrame) -> Minutes:
    d = df.copy()
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d = d.drop_duplicates("open_time").set_index("open_time").sort_index()
    start = d.index[0].ceil("1h")
    end = d.index[-1].floor("1h")                      # keep whole hours only
    grid = pd.date_range(start, end - pd.Timedelta(minutes=1), freq="1min")
    d = d.reindex(grid)
    logc = np.log(d["close"].astype(float).to_numpy())
    r = np.diff(logc, prepend=np.nan)
    miss = ~np.isfinite(r)
    vol = d["volume"].astype(float).fillna(0.0).to_numpy()
    tb = d["taker_buy_volume"].astype(float).fillna(0.0).to_numpy()
    return Minutes(t0=grid[0], r1=np.where(miss, 0.0, r), d1=2 * tb - vol, v1=vol, n_missing=int(miss.sum()) - 1)


def bars5(m: Minutes, p: Params, signs: np.ndarray | None = None) -> dict:
    """5-minute bars and causal EWMA sigma; `signs` (+-1 per minute) applies the null."""
    r1, d1 = (m.r1, m.d1) if signs is None else (m.r1 * signs, m.d1 * signs)
    k = p.bar_min
    nb = len(r1) // k
    r5 = r1[:nb * k].reshape(nb, k).sum(1)
    sig = np.sqrt(pd.Series(r5 * r5).ewm(halflife=p.ewma_halflife, min_periods=p.sigma_min_periods).mean().to_numpy())
    return {"t0": m.t0, "r5": r5, "d5": d1[:nb * k].reshape(nb, k).sum(1),
            "v5": m.v1[:nb * k].reshape(nb, k).sum(1), "sig": sig}


def windows(b: dict, n: int, p: Params) -> pd.DataFrame:
    """Window quantities for clock-aligned windows of n five-minute bars (sigma measured before the window)."""
    r5, d5, v5, sig = b["r5"], b["d5"], b["v5"], b["sig"]
    nw = len(r5) // n
    R = r5[:nw * n].reshape(nw, n)
    sig_pre = np.r_[np.nan, sig[n - 1:nw * n - 1:n]][:nw]       # sigma at the last bar before each window
    net = R.sum(1)
    path = np.abs(R).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        er = np.where(path > 0, np.abs(net) / path, np.nan)
        shock = np.abs(R).max(1) / sig_pre
    shock_up = R[np.arange(nw), np.abs(R).argmax(1)] > 0
    t = b["t0"] + pd.to_timedelta(np.arange(nw) * n * p.bar_min, unit="min")
    return pd.DataFrame({"t": t, "hour": t.hour, "net": net, "er": er, "s": sig_pre * np.sqrt(n),
                         "shock": shock, "shock_up": shock_up, "rv": np.sqrt((R * R).sum(1)),
                         "vol": v5[:nw * n].reshape(nw, n).sum(1),
                         "delta": d5[:nw * n].reshape(nw, n).sum(1)})


def hourly(m: Minutes, p: Params, signs: np.ndarray | None = None) -> pd.DataFrame:
    """Primary-scale (1 h) windows."""
    return windows(bars5(m, p, signs), p.n_bars, p)


# ---------------------------------------------------------------- thresholds and labels

def thresholds(h: pd.DataFrame, p: Params) -> dict:
    """Frozen per-asset thresholds from the development window of the real data."""
    dev = h[(h["t"] < TEST_START) & np.isfinite(h["s"])]
    return {"er_lo": float(dev["er"].quantile(p.er_lo_q)), "er_hi": float(dev["er"].quantile(p.er_hi_q)),
            "vol_med": dev.groupby("hour")["vol"].median().reindex(range(24)).to_numpy(),
            "rv_med": dev.groupby("hour")["rv"].median().reindex(range(24)).to_numpy(),
            "rvol_q": None, "rrv_q": None, "_dev": dev}


def finalize_thresholds(th: dict, p: Params) -> dict:
    dev = th.pop("_dev")
    rvol = dev["vol"].to_numpy() / th["vol_med"][dev["hour"].to_numpy()]
    rrv = dev["rv"].to_numpy() / th["rv_med"][dev["hour"].to_numpy()]
    th["rvol_q"] = float(np.nanquantile(rvol, p.rvol_q))
    th["rrv_q"] = float(np.nanquantile(rrv, p.rrv_q))
    return th


def label(h: pd.DataFrame, th: dict, p: Params) -> pd.DataFrame:
    hr = h["hour"].to_numpy()
    rvol = h["vol"].to_numpy() / th["vol_med"][hr]
    rrv = h["rv"].to_numpy() / th["rv_med"][hr]
    net, s, er = h["net"].to_numpy(), h["s"].to_numpy(), h["er"].to_numpy()
    valid = np.isfinite(s) & np.isfinite(er) & (s > 0)
    with np.errstate(invalid="ignore"):
        shock = h["shock"].to_numpy() >= p.shock_k
        disp = (er >= th["er_hi"]) & (np.abs(net) >= p.move_k * s) & (np.sign(h["delta"].to_numpy()) == np.sign(net))
        noise = (rvol <= th["rvol_q"]) & (rrv <= th["rrv_q"])
        rng = (er <= th["er_lo"]) & (np.abs(net) < p.move_k * s)
    lab = np.select([shock, disp, noise, rng], ["SHOCK", "DISP", "NOISE", "RANGE"], "OTHER").astype(object)
    lab[~valid] = None
    out = h[["t", "hour"]].copy()
    out["label"] = lab
    out["up"] = net > 0
    out["shock_up"] = h["shock_up"].to_numpy() if "shock_up" in h else np.nan
    out["window"] = np.where(out["t"] >= TEST_START, "test", "dev")
    return out


# ---------------------------------------------------------------- statistics

def _pairs(lab: pd.DataFrame) -> pd.DataFrame:
    a = lab.iloc[:-1].reset_index(drop=True)
    b = lab.iloc[1:].reset_index(drop=True)
    ok = a["label"].notna() & b["label"].notna()
    return pd.DataFrame({"x": a["label"][ok].to_numpy(), "x_up": a["up"][ok].to_numpy(),
                         "y": b["label"][ok].to_numpy(), "window": b["window"][ok].to_numpy()})


def counts(lab: pd.DataFrame) -> dict:
    """Sufficient statistics for every test, by window: n, n_x, n_y, n_xy."""
    pr = _pairs(lab)
    out = {}
    for w in ("dev", "test"):
        q = pr[pr["window"] == w]
        for tid, xf, yt in TESTS:
            x = (q["x"] == "DISP") & q["x_up"] if xf == "upDISP" else (q["x"] == xf)
            y = q["y"] == yt
            out[(w, tid)] = np.array([len(q), x.sum(), y.sum(), (x & y).sum()], float)
    return out


MIN_FROM = 30   # a conditional probability needs >= 30 'from' hours to be estimable


def lift(c: np.ndarray, neutral_if_rare: bool = False) -> float:
    """P(Y_{h+1} | X_h) / P(Y_{h+1}); +0.5 on the joint cell avoids log(0).
    Fewer than MIN_FROM 'from' hours: NaN, or 1.0 (no evidence of persistence) for null draws."""
    n, nx, ny, nxy = c
    if nx < MIN_FROM or ny == 0 or n == 0:
        return 1.0 if neutral_if_rare else np.nan
    return ((nxy + 0.5) / (nx + 0.5)) / (ny / n)


def pooled_lift(counts_by_asset: list, key, null: bool = False) -> float:
    """Geometric mean of per-asset lifts (pooling raw counts across assets with different base rates
    creates spurious lift)."""
    v = np.array([lift(c[key], neutral_if_rare=null) for c in counts_by_asset])
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.exp(np.log(v).mean())) if len(v) else np.nan


def transition_matrix(lab: pd.DataFrame, window: str) -> pd.DataFrame:
    pr = _pairs(lab)
    q = pr[pr["window"] == window]
    m = pd.crosstab(q["x"], q["y"]).reindex(index=LABELS, columns=LABELS).fillna(0)
    return m.div(m.sum(1).replace(0, np.nan), axis=0)


def run_lengths(lab: pd.DataFrame, window: str) -> pd.Series:
    q = lab[lab["window"] == window]["label"]
    grp = (q != q.shift()).cumsum()
    runs = q.groupby(grp).agg(["first", "size"])
    return runs.groupby("first")["size"].mean().reindex(LABELS)


def evaluate(real_by_asset: list, null_by_asset: list, p: Params, alpha: float | None = None) -> pd.DataFrame:
    """real_by_asset: [counts dict per asset]; null_by_asset: [[counts dict per draw] per asset].
    Lifts are pooled as the geometric mean over assets, draw by draw for the null."""
    n_draws = min(len(x) for x in null_by_asset) if null_by_asset else 0
    rows = []
    for key in real_by_asset[0]:
        w, tid = key
        c = sum(r[key] for r in real_by_asset)
        lr = pooled_lift(real_by_asset, key)
        ln = np.array([pooled_lift([nb[j] for nb in null_by_asset], key, null=True) for j in range(n_draws)])
        ln = ln[np.isfinite(ln)]
        med = float(np.median(ln)) if len(ln) else np.nan
        pval = (1 + np.sum(ln >= lr)) / (1 + len(ln)) if len(ln) else np.nan
        share_null = float(np.median([sum(nb[j][key][1] for nb in null_by_asset) /
                                      sum(nb[j][key][0] for nb in null_by_asset) for j in range(n_draws)])) if n_draws else np.nan
        xf, yt = next((x, y) for i, x, y in TESTS if i == tid)
        rows.append({"window": w, "test": tid, "from": xf, "to": yt, "pairs": int(c[0]), "n_from": int(c[1]),
                     "n_from_to": int(c[3]), "share_from_real": c[1] / c[0] if c[0] else np.nan,
                     "share_from_null": share_null,
                     "p_to_given_from": c[3] / c[1] if c[1] else np.nan, "p_to": c[2] / c[0] if c[0] else np.nan,
                     "lift_real": lr, "lift_null_median": med,
                     "lift_null_q95": float(np.quantile(ln, 0.95)) if len(ln) else np.nan,
                     "ratio_to_null": lr / med if med else np.nan, "p_value": pval})
    df = pd.DataFrame(rows).sort_values(["window", "test"]).reset_index(drop=True)
    df["alpha"] = p.alpha if alpha is None else alpha
    df["pass"] = (df["p_value"] <= df["alpha"]) & (df["ratio_to_null"] >= p.min_effect)
    return df


def add_counts(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b[k] for k in b}


# ---------------------------------------------------------------- per-asset driver

def null_signs(p: Params, draw: int, t0: pd.Timestamp, n: int) -> np.ndarray:
    """Sign per UTC minute for null draw `draw`, identical for every asset at the same minute."""
    off = int((t0 - NULL_EPOCH) / pd.Timedelta(minutes=1))
    if off < 0:
        raise ValueError("data start before NULL_EPOCH")
    rng = np.random.default_rng([p.seed, draw])
    return rng.choice(np.array([-1.0, 1.0]), size=off + n)[off:]


def run_asset(df: pd.DataFrame, p: Params, rng: np.random.Generator | None = None) -> dict:
    """All scales in one pass. Results are dicts keyed by scale n (bars): labels, thresholds, counts,
    null_counts (list per draw); null transition matrix and run lengths (test window) for every scale."""
    m = prepare_minutes(df)
    b = bars5(m, p)
    labs, ths, cnt = {}, {}, {}
    for n in p.scales:
        h = windows(b, n, p)
        ths[n] = finalize_thresholds(thresholds(h, p), p)
        labs[n] = label(h, ths[n], p)
        cnt[n] = counts(labs[n])
    ncnt = {n: [] for n in p.scales}
    ntr = {n: [] for n in p.scales}
    nrun = {n: [] for n in p.scales}
    for j in range(p.n_null):
        bn = bars5(m, p, null_signs(p, j, m.t0, len(m.r1)))
        for n in p.scales:
            ln = label(windows(bn, n, p), ths[n], p)
            ncnt[n].append(counts(ln))
            if n == p.n_bars:
                ntr[n].append(transition_matrix(ln, "test"))
                nrun[n].append(run_lengths(ln, "test"))
    jsonable = lambda th: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in th.items()}
    n0 = p.n_bars
    return {"labels": labs, "thresholds": {n: jsonable(t) for n, t in ths.items()},
            "counts": cnt, "null_counts": ncnt,
            "null_trans_mean": (pd.concat(ntr[n0]).groupby(level=0).mean().reindex(index=LABELS, columns=LABELS)
                                if ntr[n0] else None),
            "null_runs_mean": pd.concat(nrun[n0], axis=1).mean(1) if nrun[n0] else None,
            "minutes": len(m.r1), "missing_minutes": m.n_missing}


# ---------------------------------------------------------------- synthetic data

def synthetic_klines(n_days: int = 120, seed: int = 0, start: str = "2025-12-01",
                     regime: bool = True, drift: float = 0.6) -> pd.DataFrame:
    """1m klines. regime=True: persistent hourly trend/range states (Markov, p_stay=0.8) on top of
    volatility clustering; regime=False: same volatility, no directional structure."""
    rng = np.random.default_rng(seed)
    n = n_days * 1440
    t = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    lv = np.zeros(n)
    e = rng.standard_normal(n)
    for i in range(1, n):
        lv[i] = 0.999 * lv[i - 1] + 0.02 * e[i]
    sig = 6e-4 * np.exp(lv)
    z = rng.standard_normal(n)
    if regime:
        nh = n // 60
        st = np.zeros(nh, int)                      # 0 range, 1 up-trend, 2 down-trend
        for i in range(1, nh):
            st[i] = st[i - 1] if rng.random() < 0.8 else rng.integers(0, 3)
        s = np.repeat(st, 60)[:n]
        mu = np.where(s == 1, drift, np.where(s == 2, -drift, 0.0))
        # range state: mean reversion of the hour's cumulative move
        r = np.empty(n)
        cum = 0.0
        for i in range(n):
            if i % 60 == 0:
                cum = 0.0
            pull = -0.15 * cum if s[i] == 0 else 0.0
            r[i] = sig[i] * (mu[i] + z[i]) + pull
            cum += r[i]
    else:
        r = sig * z
    c = 100 * np.exp(np.cumsum(r))
    vol = rng.gamma(2.0, 50.0, n) * np.exp(lv)
    tb = vol * np.clip(0.5 + 0.5 * np.tanh(r / (sig + 1e-12)), 0.05, 0.95)
    return pd.DataFrame({"open_time": t, "close": c, "volume": vol, "taker_buy_volume": tb})
