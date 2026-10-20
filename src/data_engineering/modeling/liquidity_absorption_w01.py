"""
Liquidity Absorption W01: anatomy of the A03b forward 1:1 trades (EXT, Deribit-native) -- what separates winners.

Protocol: discussions/Liquidity_Absorption/027_claude_w01_winner_anatomy_preregistration.md

Population: every closed EXT A03b forward 1:1 taker trade (TARGET / STOP / TIMEOUT), BTC- and ETH-PERPETUAL,
2024-01-01 -> latest day. Splits (frozen):
    MINING    signal < 2025-07-01                          patterns may be searched here only
    CONFIRM   2025-07-01 <= signal < 2026-07-01, and >= 2026-10-01
    INSAMPLE  Jul-Sep 2026 (inspected since A02), reported separately, never used to confirm

Layers per trade (all signed TOWARD THE TARGET, R = the trade's initial risk in bp):
    trade record / A02 setup      EXT 03b row (+ VD18 sigma60 from 03d)
    session volume profile        Deribit perpetual prints from the last 18:00 New York to entry (LA V2 profile code)
    perpetual tape per minute     price, high/low, taker flow, volume, large-print flow (>= day's 99th pct print),
                                  print count                                              (this module)
    Deribit context per minute    liquidations with/against, premium, blocks, dated-future basis, ATM IV, skew,
                                  option delta flow / volume / blocks                        (A06 build_minutes)

Units. F60 = mean traded USD per minute over the 60 minutes before entry. flow_n = signed taker flow / F60 per minute
("net aggressive volume, in minutes of normal volume"). f3 = 3-minute sum of flow_n / 3.

Landmark logic (the only legitimate way to use post-entry information): at L minutes after entry, among trades
still open, use data up to entry + L only. Holding vs exiting at L differs by remaining_R = final_R - progress_R(L),
gross, because both choices pay the same entry and exit costs.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "LA-W01-0.1.0"
MIN_MS = 60_000
DAY_MS = 86_400_000
WINDOW = 240
LANDMARKS = (5, 15, 30, 60)
PRIMARY_L = 15
MINING_END = pd.Timestamp("2025-07-01", tz="UTC")
INSAMPLE = (pd.Timestamp("2026-07-01", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC"))
PRE_PHASES = {"context": (-240, -61), "approach": (-60, -16), "formation": (-15, -1)}
CLOSED = ("TARGET", "STOP", "TIMEOUT")
SEED = 20261021
CUR = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}
PANEL_COLS = ("px_R", "fav_R", "adv_R", "flow_n", "vol_n", "large_n", "liq_with_n", "liq_against_n", "premium_bp",
              "fut_basis", "atm_iv", "skew", "opt_flow_n", "opt_amt_n")


@dataclass(frozen=True)
class Spec:
    large_q: float = 0.99          # large print = at or above this quantile of that day's print sizes (USD)
    burst_q: float = 0.95          # burst threshold = this quantile of f3 in MINING pre-entry context minutes
    rule_tercile: float = 1 / 3    # landmark rule: exit at L if cvd_L is in the MINING bottom tercile
    fdr_q: float = 0.10
    confirm_alpha: float = 0.05
    boot_b: int = 2000
    seed: int = SEED


def registration(spec: Spec = Spec()) -> dict:
    return {"version": VERSION, **asdict(spec), "landmarks": LANDMARKS, "primary_landmark": PRIMARY_L,
            "mining_end": str(MINING_END.date()), "insample": [str(x.date()) for x in INSAMPLE],
            "pre_phases": PRE_PHASES,
            "H1": "at L=15, cvd_15 has a positive partial rank correlation with remaining_R given progress_15",
            "status": "PRE-REGISTERED_EXPLORATORY_MINING_WITH_CONFIRMATION"}


def split_label(sig: pd.Series) -> pd.Series:
    t = pd.to_datetime(sig, utc=True)
    lab = np.where(t < MINING_END, "MINING", "CONFIRM")
    lab = np.where((t >= INSAMPLE[0]) & (t < INSAMPLE[1]), "INSAMPLE", lab)
    return pd.Series(lab, index=sig.index)


# ----------------------------------------------------------------------------- trades
def load_trades(ext: Path) -> pd.DataFrame:
    from src.data_engineering.modeling import liquidity_absorption_extended as ex
    ext = Path(ext)
    b = pd.read_csv(ext / "03b" / "03b_1to1_events.csv", low_memory=False)
    b = b[b.reason.isin(CLOSED) & b.entry_time_ms.notna()].copy()
    b["net_deribit_bp"] = ex.deribit_net(b)
    d3 = ext / "03d" / "03d_risk_execution.csv"
    if d3.exists():
        s = pd.read_csv(d3, usecols=["a02_candidate_row", "vol_sigma_bp_60"], low_memory=False)
        s = s.dropna().drop_duplicates("a02_candidate_row")
        b = b.merge(s, on="a02_candidate_row", how="left")
    else:
        b["vol_sigma_bp_60"] = np.nan
    b["sign"] = np.where(b.side.eq("long"), 1.0, -1.0)
    b["currency"] = b.symbol.map(CUR)
    b["entry_ms"] = b.entry_time_ms.astype("int64")
    b["exit_ms"] = b.exit_time_ms.astype("int64")
    b["R_bp"] = b.initial_risk_bps.astype(float)
    b["final_R"] = b.gross_return_bp / b.R_bp
    b["cost_bp"] = b.gross_return_bp - b.net_deribit_bp
    b["net_R"] = b.net_deribit_bp / b.R_bp
    b["cost_R"] = b.cost_bp / b.R_bp
    b["win"] = b.reason.eq("TARGET").astype(int)
    b["split"] = split_label(b.signal_time_utc)
    b["day"] = pd.to_datetime(b.signal_time_utc, utc=True).dt.floor("D")
    b["dur_min"] = (b.exit_ms - b.entry_ms) / MIN_MS
    b = b[np.isfinite(b.R_bp) & (b.R_bp > 0)]
    return b.sort_values("entry_ms").reset_index(drop=True)


# ----------------------------------------------------------------------------- tape extras and profile
def tape_minutes(pr: pd.DataFrame, large_q: float = 0.99) -> pd.DataFrame:
    """Per-minute perpetual tape extras from one day of prints (time_ms, price, usd, buy)."""
    if pr.empty:
        return pd.DataFrame()
    thr = float(np.quantile(pr.usd.to_numpy(float), large_q))
    sg = np.where(pr.buy.to_numpy(bool), 1.0, -1.0)
    big = pr.usd.to_numpy(float) >= thr
    z = pd.DataFrame({"minute": pr.time_ms.to_numpy("int64") // MIN_MS, "price": pr.price.to_numpy(float),
                      "usd": pr.usd.to_numpy(float), "s_usd": sg * pr.usd.to_numpy(float),
                      "l_usd": np.where(big, sg * pr.usd.to_numpy(float), 0.0)})
    g = z.groupby("minute")
    return pd.DataFrame({"high": g.price.max(), "low": g.price.min(), "t_usd": g.usd.sum(), "t_signed": g.s_usd.sum(),
                         "t_large": g.l_usd.sum(), "t_n": g.size()})


def profile_features(px: np.ndarray, usd: np.ndarray, entry: float, target: float, risk_bp: float, side: str,
                     vspec=None) -> dict:
    """Session-profile geometry relative to the trade, R units, signed toward the target."""
    from src.data_engineering.modeling import liquidity_absorption_v2 as v2
    vspec = vspec or v2.Spec()
    nan = {"pf_poc_R": np.nan, "pf_path_mass": np.nan, "pf_min_frac_path": np.nan, "pf_hvn_on_path": np.nan,
           "pf_frac_at_entry": np.nan, "pf_frac_at_target": np.nan, "pf_target_outside_range": np.nan,
           "pf_vah_val_width_R": np.nan}
    if len(px) < 100 or not np.isfinite(target):
        return nan
    bins, vol = v2.profile_from_prints(px, usd, vspec.bin_bp)
    if len(vol) < 5 or vol.max() <= 0:
        return nan
    w = math.log1p(vspec.bin_bp / 1e4)
    sm = pd.Series(vol).rolling(vspec.smooth_bins, center=True, min_periods=1).mean().to_numpy()
    frac = sm / sm.max()
    s = 1.0 if side == "long" else -1.0
    k = lambda p: int(math.floor(math.log(p) / w)) - int(bins[0])
    ke, kt = k(entry), k(target)
    poc = math.exp((bins[int(np.argmax(vol))] + 0.5) * w)
    lo_k, hi_k = sorted((ke, kt))
    inside = lambda i: 0 <= i < len(frac)
    path = [i for i in range(lo_k + 1, hi_k) if inside(i)]
    tot = vol.sum()
    # value area (70%) width around the POC
    o = np.argsort(vol)[::-1]
    cum = np.cumsum(vol[o]) / tot
    va = o[: int(np.searchsorted(cum, 0.70)) + 1]
    va_w_bp = (va.max() - va.min() + 1) * vspec.bin_bp
    return {"pf_poc_R": s * (poc / entry - 1) * 1e4 / risk_bp,
            "pf_path_mass": float(vol[path].sum() / tot) if path else 0.0,
            "pf_min_frac_path": float(frac[path].min()) if path else np.nan,
            "pf_hvn_on_path": float(any(frac[i] >= vspec.hvn_frac for i in path)) if path else 0.0,
            "pf_frac_at_entry": float(frac[ke]) if inside(ke) else 0.0,
            "pf_frac_at_target": float(frac[kt]) if inside(kt) else 0.0,
            "pf_target_outside_range": float(not inside(kt)),
            "pf_vah_val_width_R": va_w_bp / risk_bp}


# ----------------------------------------------------------------------------- build
def _prints(path: Path, perp: str) -> pd.DataFrame:
    """Perpetual prints of one day file: time_ms, price, usd (Deribit perp amount is USD), buy (taker bought)."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    cols = [c for c in ("timestamp", "instrument_name", "price", "amount", "direction") if c in pf.schema.names]
    d = pf.read(columns=cols).to_pandas()
    d = d[d.instrument_name.astype(str).eq(perp)]
    t = d.timestamp
    tm = (pd.to_numeric(t).to_numpy("int64") if pd.api.types.is_numeric_dtype(t)
          else pd.to_datetime(t, utc=True).dt.as_unit("ms").astype("int64").to_numpy())
    out = pd.DataFrame({"time_ms": tm, "price": pd.to_numeric(d.price, errors="coerce").to_numpy(float),
                        "usd": pd.to_numeric(d.amount, errors="coerce").to_numpy(float),
                        "buy": d.direction.astype(str).eq("buy").to_numpy()})
    out = out[np.isfinite(out.price) & (out.price > 0) & np.isfinite(out.usd) & (out.usd > 0)]
    return out.sort_values("time_ms", kind="mergesort").reset_index(drop=True)


def _days_for(entry_ms: int) -> list[str]:
    d0 = pd.Timestamp(int(entry_ms), unit="ms", tz="UTC").normalize()
    return [str((d0 + pd.Timedelta(days=k)).date()) for k in (-1, 0, 1)]


def build(ext: Path, history: Path, local_out: Path, spec: Spec = Spec(), log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-trade feature table and the per-minute panel (rel -240..+240). Writes both to `local_out` (parquet)."""
    from src.data_engineering.modeling import liquidity_absorption_stage_a06 as a06
    from src.data_engineering.modeling import liquidity_absorption_v2 as v2
    history, local_out = Path(history), Path(local_out)
    local_out.mkdir(parents=True, exist_ok=True)
    trades = load_trades(ext)
    feats, panels = [], []
    rel = np.arange(-WINDOW, WINDOW + 1)
    for sym, T in trades.groupby("symbol"):
        cur = CUR[sym]
        perp = f"{cur}-PERPETUAL"
        days = sorted({d for e in T.entry_ms for d in _days_for(e)})
        days = [d for d in days if (history / "trades" / cur / "future" / d[:4] / f"{d}.parquet").is_file()]
        if log:
            log(sym, "A06 minutes over", len(days), "days")
        m, _ = a06.build_minutes(history, cur, days)
        cache, extras, done = {}, {}, set()
        prof = {}
        for _, e in T.iterrows():
            need = [d for d in _days_for(e.entry_ms) if d in days]
            for d in need:
                if d not in cache:
                    p = history / "trades" / cur / "future" / d[:4] / f"{d}.parquet"
                    cache[d] = _prints(p, perp)
                    if d not in done:
                        extras[d] = tape_minutes(cache[d], spec.large_q)
                        done.add(d)
            for d in [k for k in cache if k < need[0]] if need else []:
                del cache[d]
            # session profile: prints in [last 18:00 NY, entry)
            s0 = v2.session_start_ms(int(e.entry_ms))
            parts = [cache[d] for d in need if d in cache]
            if parts:
                pr = pd.concat(parts, ignore_index=True)
                sel = (pr.time_ms >= s0) & (pr.time_ms < int(e.entry_ms))
                pf = profile_features(pr.price.to_numpy(float)[sel], pr.usd.to_numpy(float)[sel], float(e.entry_print),
                                      float(e.target), float(e.R_bp), e.side)
            else:
                pf = profile_features(np.array([]), np.array([]), 1.0, np.nan, 1.0, e.side)
            pf["session_age_min"] = (int(e.entry_ms) - s0) / MIN_MS
            prof[e.a02_candidate_row] = pf
        if m.empty:
            continue
        X = pd.concat(extras.values()).sort_index() if extras else pd.DataFrame()
        X = X[~X.index.duplicated(keep="first")]
        m = m.join(X, how="left")
        for c in ("t_usd", "t_signed", "t_large", "t_n"):
            m[c] = m[c].fillna(0.0)
        m["high"] = m.high.fillna(m.close)
        m["low"] = m.low.fillna(m.close)
        for _, e in T.iterrows():
            r = trade_panel(e, m, rel)
            if r is None:
                continue
            P, f = r
            f.update(prof.get(e.a02_candidate_row, {}))
            feats.append(f)
            panels.append(P)
        if log:
            log(sym, "trades with panel", sum(1 for f in feats if f["symbol"] == sym))
    F = pd.DataFrame(feats)
    P = pd.concat(panels, ignore_index=True) if panels else pd.DataFrame()
    F.to_parquet(local_out / "W01_trade_features_raw.parquet", index=False)
    P.to_parquet(local_out / "W01_panel.parquet", index=False)
    return F, P


def trade_panel(e: pd.Series, m: pd.DataFrame, rel: np.ndarray) -> tuple[pd.DataFrame, dict] | None:
    m0 = int(e.entry_ms) // MIN_MS
    idx = m0 + rel
    if idx[0] < m.index[0] or idx[-1] > m.index[-1]:
        return None
    w = m.loc[idx]
    vol = (w.buy_amt + w.sell_amt).to_numpy(float)
    F60 = float(np.mean(vol[WINDOW - 60:WINDOW]))
    if not np.isfinite(F60) or F60 <= 0:
        return None
    s, ent, R = float(e.sign), float(e.entry_print), float(e.R_bp)
    toR = lambda p: s * (np.asarray(p, float) / ent - 1) * 1e4 / R
    hi, lo = toR(w.high), toR(w.low)
    opt_amt = w.opt_amt.to_numpy(float) if "opt_amt" in w else np.full(len(w), np.nan)
    opt_ref = float(np.nanmean(opt_amt[WINDOW - 60:WINDOW])) if np.isfinite(opt_amt).any() else np.nan
    opt_ref = opt_ref if np.isfinite(opt_ref) and opt_ref > 0 else np.nan
    sk = (w.put_iv - w.call_iv).ffill(limit=60).to_numpy(float) if "put_iv" in w else np.full(len(w), np.nan)
    d = {"a02_candidate_row": e.a02_candidate_row, "rel": rel.astype(np.int16),
         "px_R": toR(w.close), "fav_R": np.maximum(hi, lo), "adv_R": np.minimum(hi, lo),
         "flow_n": s * (w.buy_amt - w.sell_amt).to_numpy(float) / F60,
         "vol_n": vol / F60,
         "large_n": s * w.t_large.to_numpy(float) / F60,
         "liq_with_n": (w.liq_long if s > 0 else w.liq_short).to_numpy(float) / F60,
         "liq_against_n": (w.liq_short if s > 0 else w.liq_long).to_numpy(float) / F60,
         "premium_bp": s * w.prem.to_numpy(float),
         "fut_basis": s * w.fut_basis_lvl.to_numpy(float) if "fut_basis_lvl" in w else np.full(len(w), np.nan),
         "atm_iv": w.atm_iv_lvl.to_numpy(float) if "atm_iv_lvl" in w else np.full(len(w), np.nan),
         "skew": -s * sk,
         "opt_flow_n": s * w.opt_delta_flow.to_numpy(float) / opt_ref if "opt_delta_flow" in w else np.full(len(w), np.nan),
         "opt_amt_n": opt_amt / opt_ref}
    P = pd.DataFrame(d)
    for c in PANEL_COLS:
        P[c] = P[c].astype(np.float32)
    keep = ["a02_candidate_row", "symbol", "side", "signal_time_utc", "reason", "level_type", "target_source",
            "zone_halfwidth", "zone_quote_touch_ratio", "opposing_zone_share_change", "zone_share_touch_change",
            "absorption_like_exploratory", "entry_ms", "exit_ms", "R_bp", "vol_sigma_bp_60", "final_R", "net_R",
            "cost_R", "win", "split", "day", "dur_min", "gross_return_bp", "net_deribit_bp", "mfe_print_initial_r",
            "mae_print_initial_r"]
    f = {k: e[k] for k in keep if k in e.index}
    f["F60_usd"] = F60
    f["exit_rel_min"] = (int(e.exit_ms) - m0 * MIN_MS) / MIN_MS
    return P, f


# ----------------------------------------------------------------------------- features
def _seg(P: pd.DataFrame, a: int, b: int) -> pd.DataFrame:
    return P[(P.rel >= a) & (P.rel <= b)]


def pre_entry_features(F: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    """Entry-time features only (no post-entry data)."""
    out = F.copy()
    t = pd.to_datetime(out.signal_time_utc, utc=True)
    ny = t.dt.tz_convert("America/New_York")
    out["x_hour_ny"] = ny.dt.hour + ny.dt.minute / 60
    out["x_weekday"] = ny.dt.weekday
    out["x_risk_over_sigma60"] = out.R_bp / out.vol_sigma_bp_60
    out["x_R_bp"] = out.R_bp
    out["x_vwap_target"] = out.target_source.eq("SESSION_VWAP").astype(float)
    out["x_level_prior_day"] = out.level_type.astype(str).str.contains("prior").astype(float)
    G = P.groupby("a02_candidate_row")
    rows = {}
    for aid, g in G:
        g = g.set_index("rel")
        r = {}
        for ph, (a, b) in PRE_PHASES.items():
            w = g.loc[a:b]
            prev = g.loc[a - 1] if (a - 1) in g.index else g.loc[a]
            n = len(w)
            r[f"{ph}_px_R"] = float(w.px_R.iloc[-1] - prev.px_R)
            r[f"{ph}_cvd"] = float(w.flow_n.sum() / n)
            r[f"{ph}_vol"] = float(w.vol_n.mean())
            r[f"{ph}_large"] = float(w.large_n.sum() / n)
            r[f"{ph}_liq_with"] = float(w.liq_with_n.sum())
            r[f"{ph}_liq_against"] = float(w.liq_against_n.sum())
            r[f"{ph}_premium"] = float(np.nanmean(w.premium_bp)) if np.isfinite(w.premium_bp).any() else np.nan
            r[f"{ph}_opt_flow"] = float(np.nansum(w.opt_flow_n)) if np.isfinite(w.opt_flow_n).any() else np.nan
            r[f"{ph}_opt_amt"] = float(np.nanmean(w.opt_amt_n)) if np.isfinite(w.opt_amt_n).any() else np.nan
            for c in ("fut_basis", "atm_iv", "skew"):
                v = w[c].to_numpy(float)
                ok = v[np.isfinite(v)]
                r[f"{ph}_{c}_chg"] = float(ok[-1] - ok[0]) if len(ok) >= 2 else np.nan
            r[f"{ph}_range_R"] = float(w.fav_R.max() - w.adv_R.min())
        # price-flow divergence in the formation: flow against the trade while price held / moved toward target
        r["formation_div"] = r["formation_px_R"] - r["formation_cvd"]
        rows[aid] = r
    X = pd.DataFrame.from_dict(rows, orient="index")
    X.index.name = "a02_candidate_row"
    return out.merge(X.reset_index(), on="a02_candidate_row", how="left")


def landmark_features(F: pd.DataFrame, P: pd.DataFrame, L: int) -> pd.DataFrame:
    """Trades still open after L minutes; data from minutes rel 0..L-1 (all completed by entry + L)."""
    open_ = F[F.exit_ms > (F.entry_ms // MIN_MS + L) * MIN_MS][["a02_candidate_row"]]
    w = _seg(P[P.a02_candidate_row.isin(open_.a02_candidate_row)], 0, L - 1)
    g = w.groupby("a02_candidate_row")
    f3 = w.assign(f3=w.groupby("a02_candidate_row").flow_n.transform(lambda x: x.rolling(3, min_periods=3).sum() / 3))
    last = g.tail(1).set_index("a02_candidate_row")
    first = g.head(1).set_index("a02_candidate_row")
    X = pd.DataFrame({
        f"L{L}_prog_R": last.px_R,
        f"L{L}_mfe_R": g.fav_R.max(), f"L{L}_mae_R": g.adv_R.min(),
        f"L{L}_cvd": g.flow_n.sum() / L,
        f"L{L}_burst": f3.groupby("a02_candidate_row").f3.max(),
        f"L{L}_vol": g.vol_n.mean(),
        f"L{L}_large": g.large_n.sum() / L,
        f"L{L}_liq_with": g.liq_with_n.sum(), f"L{L}_liq_against": g.liq_against_n.sum(),
        f"L{L}_premium_chg": last.premium_bp - first.premium_bp,
        f"L{L}_opt_flow": g.opt_flow_n.sum(min_count=1),
        f"L{L}_atm_iv_chg": last.atm_iv - first.atm_iv,
        f"L{L}_skew_chg": last["skew"] - first["skew"],
        f"L{L}_basis_chg": last.fut_basis - first.fut_basis,
    })
    X.index.name = "a02_candidate_row"
    X = X.reset_index().merge(F[["a02_candidate_row", "final_R", "net_R", "cost_R", "win", "reason", "split", "day",
                                 "symbol"]], on="a02_candidate_row")
    X[f"L{L}_remaining_R"] = X.final_R - X[f"L{L}_prog_R"]
    return X


# ----------------------------------------------------------------------------- statistics
def _rank(x):
    return pd.Series(x).rank().to_numpy(float)


def partial_spearman(x, y, z) -> float:
    """Rank correlation of x and y after removing a linear effect of rank(z) from both ranks."""
    rx, ry, rz = _rank(x), _rank(y), _rank(z)
    A = np.c_[np.ones(len(rz)), rz]
    ex_ = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    d = np.sqrt((ex_ ** 2).sum() * (ey ** 2).sum())
    return float((ex_ * ey).sum() / d) if d > 0 else np.nan


def day_boot(df: pd.DataFrame, stat, B: int, seed: int) -> tuple[float, float, float]:
    """(statistic, one-sided 95% lower bound, bootstrap P(stat <= 0)) resampling signal days."""
    df = df.reset_index(drop=True)
    est = stat(df)
    groups = df.groupby("day").indices
    keys = list(groups)
    if len(keys) < 10 or not np.isfinite(est):
        return est, np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        ix = np.concatenate([groups[keys[i]] for i in pick])
        vals.append(stat(df.iloc[ix]))
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    return est, float(np.quantile(v, 0.05)), float((v <= 0).mean())


def h1_test(X: pd.DataFrame, L: int, spec: Spec) -> pd.DataFrame:
    rows = []
    c, y, z = f"L{L}_cvd", f"L{L}_remaining_R", f"L{L}_prog_R"
    for sp in ("MINING", "CONFIRM", "INSAMPLE"):
        d = X[X.split.eq(sp)].dropna(subset=[c, y, z])
        if len(d) < 30:
            continue
        est, lo, p = day_boot(d, lambda q: partial_spearman(q[c], q[y], q[z]), spec.boot_b, spec.seed)
        raw = float(pd.Series(d[c].to_numpy()).corr(pd.Series(d[y].to_numpy()), method="spearman"))
        rows.append({"landmark": L, "split": sp, "n_open": len(d), "partial_rho_cvd_remaining": est,
                     "partial_rho_lo95": lo, "p_boot_le0": p, "raw_rho_cvd_remaining": raw,
                     "raw_rho_cvd_prog": float(pd.Series(d[c].to_numpy()).corr(pd.Series(d[z].to_numpy()), method="spearman"))})
    return pd.DataFrame(rows)


def rule_eval(F: pd.DataFrame, X: pd.DataFrame, L: int, flag_col: str, thr: float | None, name: str,
              spec: Spec, below: bool = True) -> pd.DataFrame:
    """Exit at L when flag is triggered, else hold. Net R per trade vs baseline; trades closed before L unchanged."""
    base = F[["a02_candidate_row", "net_R", "split", "day", "symbol"]].copy()
    x = X.set_index("a02_candidate_row")
    trig = (x[flag_col] < thr) if below else (x[flag_col] >= thr)
    early = x.loc[trig, f"L{L}_prog_R"] - x.loc[trig, "cost_R"]
    base["rule_net_R"] = base.net_R
    base.loc[base.a02_candidate_row.isin(early.index), "rule_net_R"] = base.a02_candidate_row.map(early)
    base["diff_R"] = base.rule_net_R - base.net_R
    base["triggered"] = base.a02_candidate_row.isin(early.index)
    rows = []
    for sp in ("MINING", "CONFIRM", "INSAMPLE"):
        d = base[base.split.eq(sp)]
        if len(d) < 30:
            continue
        m, lo, _ = day_boot(d, lambda q: float(q.rule_net_R.mean()), spec.boot_b, spec.seed)
        dm, dlo, _ = day_boot(d, lambda q: float(q.diff_R.mean()), spec.boot_b, spec.seed)
        rows.append({"rule": name, "landmark": L, "threshold": thr, "split": sp, "n": len(d),
                     "triggered_share": float(d.triggered.mean()), "baseline_net_R": float(d.net_R.mean()),
                     "rule_net_R": m, "rule_net_lo95": lo, "improvement_R": dm, "improvement_lo95": dlo})
    return pd.DataFrame(rows)


def burst_study(F: pd.DataFrame, P: pd.DataFrame, thr: float, spec: Spec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First 3-minute burst of taker flow toward the target while the trade is open. Remaining R from the end of the
    burst minute = the gross outcome of holding (or of adding a new 1:1-sized unit) from there."""
    w = P[P.rel >= 0].copy()
    w["f3"] = w.groupby("a02_candidate_row").flow_n.transform(lambda x: x.rolling(3, min_periods=3).sum() / 3)
    w = w.merge(F[["a02_candidate_row", "exit_rel_min", "final_R", "cost_R", "reason", "split", "day", "dur_min"]],
                on="a02_candidate_row")
    w = w[w.rel + 1 <= w.exit_rel_min]          # minute completed while still open
    hit = w[w.f3 >= thr].groupby("a02_candidate_row").head(1)
    hit = hit.assign(onset_min=hit.rel, prog_at_onset_R=hit.px_R, remaining_R=hit.final_R - hit.px_R,
                     addon_net_R=hit.final_R - hit.px_R - hit.cost_R,
                     onset_to_exit_min=hit.exit_rel_min - hit.rel)
    per = F[["a02_candidate_row", "reason", "split"]].merge(
        hit[["a02_candidate_row", "onset_min", "prog_at_onset_R", "remaining_R", "addon_net_R", "onset_to_exit_min"]],
        on="a02_candidate_row", how="left")
    per["had_burst"] = per.onset_min.notna()
    summ = []
    for sp in ("MINING", "CONFIRM", "INSAMPLE"):
        for oc in ("TARGET", "STOP", "TIMEOUT", "ALL"):
            d = per[per.split.eq(sp) & (per.reason.eq(oc) if oc != "ALL" else True)]
            if d.empty:
                continue
            h = d[d.had_burst]
            summ.append({"split": sp, "outcome": oc, "n": len(d), "burst_share": float(d.had_burst.mean()),
                         "median_onset_min": float(h.onset_min.median()) if len(h) else np.nan,
                         "median_prog_at_onset_R": float(h.prog_at_onset_R.median()) if len(h) else np.nan,
                         "median_onset_to_exit_min": float(h.onset_to_exit_min.median()) if len(h) else np.nan})
        d = hit[hit.split.eq(sp)]
        if len(d) >= 30:
            m, lo, _ = day_boot(d, lambda q: float(q.addon_net_R.mean()), spec.boot_b, spec.seed)
            summ.append({"split": sp, "outcome": "ADD_ON_AT_BURST", "n": len(d), "addon_mean_net_R": m,
                         "addon_net_lo95": lo, "addon_mean_remaining_R": float(d.remaining_R.mean()),
                         "addon_win_share": float((d.remaining_R > 0).mean())})
    return per, pd.DataFrame(summ)


def burst_threshold(F: pd.DataFrame, P: pd.DataFrame, q: float) -> float:
    mine = F.loc[F.split.eq("MINING"), "a02_candidate_row"]
    c = P[P.a02_candidate_row.isin(mine) & (P.rel >= PRE_PHASES["context"][0]) & (P.rel <= PRE_PHASES["context"][1])]
    f3 = c.groupby("a02_candidate_row").flow_n.transform(lambda x: x.rolling(3, min_periods=3).sum() / 3)
    return float(np.nanquantile(f3.to_numpy(float), q))


def screen(D: pd.DataFrame, features: list[str], y_rank: str, spec: Spec, family: str) -> pd.DataFrame:
    """MINING: TARGET vs STOP Mann-Whitney + Spearman with y_rank (all closed); BH within family.
    CONFIRM: survivors (q <= fdr_q on the Mann-Whitney) must keep their sign with one-sided p < alpha, Holm."""
    from scipy.stats import mannwhitneyu, spearmanr
    from src.data_engineering.modeling.liquidity_absorption_stage_a06 import bh
    rows = []
    M = D[D.split.eq("MINING")]
    C = D[D.split.eq("CONFIRM")]
    for f in features:
        x = M.loc[M.reason.eq("TARGET"), f].dropna().to_numpy(float)
        y = M.loc[M.reason.eq("STOP"), f].dropna().to_numpy(float)
        r = {"family": family, "feature": f, "n_target": len(x), "n_stop": len(y)}
        if len(x) < 20 or len(y) < 20 or np.ptp(np.r_[x, y]) == 0:
            rows.append({**r, "p": np.nan})
            continue
        u, p = mannwhitneyu(x, y, alternative="two-sided")
        mm = M[[f, y_rank]].dropna()
        rho = spearmanr(mm[f], mm[y_rank]).statistic if len(mm) > 20 else np.nan
        rows.append({**r, "med_target": float(np.median(x)), "med_stop": float(np.median(y)),
                     "rank_biserial": float(2 * u / (len(x) * len(y)) - 1), "p": float(p), "rho_with_y": float(rho)})
    T = pd.DataFrame(rows)
    T["q"] = bh(T.p.to_numpy(float))
    T["survivor"] = T.q <= spec.fdr_q
    T["confirm_rank_biserial"] = np.nan
    T["confirm_p_one_sided"] = np.nan
    for i, r in T[T.survivor].iterrows():
        x = C.loc[C.reason.eq("TARGET"), r.feature].dropna().to_numpy(float)
        y = C.loc[C.reason.eq("STOP"), r.feature].dropna().to_numpy(float)
        if len(x) < 20 or len(y) < 20:
            continue
        alt = "greater" if r.rank_biserial > 0 else "less"
        u, p = mannwhitneyu(x, y, alternative=alt)
        T.at[i, "confirm_rank_biserial"] = float(2 * u / (len(x) * len(y)) - 1)
        T.at[i, "confirm_p_one_sided"] = float(p)
    s = T[T.survivor & T.confirm_p_one_sided.notna()].sort_values("confirm_p_one_sided")
    k = len(s)
    holm = {idx: min(1.0, float(p) * (k - j)) for j, (idx, p) in enumerate(s.confirm_p_one_sided.items())}
    T["confirm_holm"] = pd.Series(holm, dtype=float)
    T["confirmed"] = T.confirm_holm < spec.confirm_alpha
    return T.sort_values(["survivor", "q"], ascending=[False, True])


def curves(F: pd.DataFrame, P: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Median curves by outcome: entry-aligned (rel -240..240, CVD cumulated from entry, in minutes of normal volume)
    and exit-aligned for TARGET/STOP (-60..0 minutes before exit)."""
    z = P.merge(F[["a02_candidate_row", "reason", "split", "exit_rel_min"]], on="a02_candidate_row")
    z = z.sort_values(["a02_candidate_row", "rel"])
    post = z.rel >= 0
    z["cvd_from_entry"] = np.where(post, z.flow_n.where(post, 0).groupby(z.a02_candidate_row).cumsum(), np.nan)
    z["f3"] = z.groupby("a02_candidate_row").flow_n.transform(lambda x: x.rolling(3, min_periods=3).sum() / 3)
    cols = ["px_R", "cvd_from_entry", "f3", "vol_n", "large_n", "liq_with_n", "liq_against_n"]
    e = z.groupby(["split", "reason", "rel"])[cols].median().reset_index()
    z["to_exit"] = np.floor(z.rel - z.exit_rel_min).astype(int)
    x = z[z.reason.isin(["TARGET", "STOP"]) & z.to_exit.between(-60, 0)]
    xa = x.groupby(["split", "reason", "to_exit"])[cols].median().reset_index()
    return e, xa


def run_analysis(F: pd.DataFrame, P: pd.DataFrame, out: Path, spec: Spec = Spec()) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    D = pre_entry_features(F, P)
    pre_cols = [c for c in D.columns if c.startswith(("context_", "approach_", "formation_", "x_", "pf_"))] + \
               [c for c in ("zone_halfwidth", "zone_quote_touch_ratio", "opposing_zone_share_change",
                            "zone_share_touch_change", "session_age_min") if c in D.columns]
    D.to_csv(out / "W01_trade_features.csv.gz", index=False)
    res = {"version": VERSION, "trades": int(len(D)), "by_split": D.groupby(["split", "reason"]).size().unstack(fill_value=0).to_dict("index")}
    # H1 at all landmarks, primary L=15
    H, LM, RULES = [], {}, []
    for L in LANDMARKS:
        X = landmark_features(D, P, L)
        LM[L] = X
        H.append(h1_test(X, L, spec))
        thr = float(np.nanquantile(X.loc[X.split.eq("MINING"), f"L{L}_cvd"], spec.rule_tercile))
        RULES.append(rule_eval(D, X, L, f"L{L}_cvd", thr, f"exit_if_cvd{L}_bottom_tercile", spec))
        RULES.append(rule_eval(D, X, L, f"L{L}_prog_R", 0.0, f"exit_if_losing_at_{L}", spec))
        X.to_csv(out / f"W01_landmark_{L}.csv.gz", index=False)
    H = pd.concat(H, ignore_index=True)
    RULES = pd.concat(RULES, ignore_index=True)
    H.to_csv(out / "W01_h1_cvd_landmark.csv", index=False)
    RULES.to_csv(out / "W01_management_rules.csv", index=False)
    # bursts
    thr = burst_threshold(D, P, spec.burst_q)
    per, bs = burst_study(D, P, thr, spec)
    per.to_csv(out / "W01_burst_per_trade.csv.gz", index=False)
    bs.to_csv(out / "W01_burst_summary.csv", index=False)
    # screens
    S1 = screen(D, pre_cols, "final_R", spec, "PRE_ENTRY")
    X15 = LM[PRIMARY_L]
    lcols = [c for c in X15.columns if c.startswith(f"L{PRIMARY_L}_") and not c.endswith("remaining_R")]
    S2 = screen(X15, lcols, f"L{PRIMARY_L}_remaining_R", spec, f"LANDMARK_{PRIMARY_L}")
    S = pd.concat([S1, S2], ignore_index=True)
    S.to_csv(out / "W01_feature_screen.csv", index=False)
    e, xa = curves(D, P)
    e.to_csv(out / "W01_curves_entry.csv.gz", index=False)
    xa.to_csv(out / "W01_curves_exit.csv", index=False)
    h15 = H[(H.landmark == PRIMARY_L)].set_index("split")
    c = h15.loc["CONFIRM"] if "CONFIRM" in h15.index else None
    m = h15.loc["MINING"] if "MINING" in h15.index else None
    res["H1_primary"] = {
        "mining_partial_rho": None if m is None else float(m.partial_rho_cvd_remaining),
        "confirm_partial_rho": None if c is None else float(c.partial_rho_cvd_remaining),
        "confirm_lo95": None if c is None else float(c.partial_rho_lo95),
        "supported": bool(c is not None and m is not None and m.partial_rho_cvd_remaining > 0
                          and c.partial_rho_lo95 > 0)}
    rc = RULES[RULES.split.eq("CONFIRM")]
    res["rules_confirm"] = rc[["rule", "triggered_share", "baseline_net_R", "rule_net_R", "rule_net_lo95",
                               "improvement_R", "improvement_lo95"]].round(4).to_dict("records")
    res["rule_with_positive_net_lo95"] = rc.loc[rc.rule_net_lo95 > 0, "rule"].tolist()
    res["burst_threshold_f3"] = thr
    res["screen_survivors"] = int(S.survivor.sum())
    res["screen_confirmed"] = S.loc[S.confirmed.fillna(False).astype(bool), ["family", "feature", "rank_biserial",
                                                                             "confirm_rank_biserial"]].to_dict("records")
    (out / "W01_decision.json").write_text(json.dumps(res, indent=1, default=str))
    return res


# ----------------------------------------------------------------------------- synthetic
def synthetic_ext(ext: Path, history: Path, n: int = 120, seed: int = 0, planted: bool = True) -> None:
    """Fake 03b trades on the synthetic A06 history. With `planted`, winners get a burst of taker flow toward the
    target in minutes 3..8 after entry (so H1 / burst code paths can be checked)."""
    from src.data_engineering.modeling import liquidity_absorption_stage_a06 as a06
    rng = np.random.default_rng(seed)
    days = [str(d.date()) for d in pd.date_range("2025-06-20", periods=24, freq="D")]
    t0 = int(pd.Timestamp(days[2], tz="UTC").value // 10**6)
    ent = np.sort(t0 + rng.integers(0, 19 * DAY_MS, n))
    win = rng.random(n) < 0.5
    side = np.where(rng.random(n) < 0.5, "long", "short")
    for cur in ("BTC", "ETH"):
        a06.synthetic_history(history, days, currency=cur, seed=seed + (cur == "ETH"))
    if planted:
        # rewrite prints: add aggressive prints toward the target after winners' entries
        for cur, sym in (("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")):
            for d in days:
                p = Path(history) / "trades" / cur / "future" / d[:4] / f"{d}.parquet"
                df = pd.read_parquet(p)
                add = []
                for i in range(n):
                    if (i % 2 == 0) != (cur == "BTC") or not win[i]:
                        continue
                    if pd.Timestamp(int(ent[i]), unit="ms", tz="UTC").strftime("%Y-%m-%d") != d:
                        continue
                    ts = ent[i] + np.arange(3 * 60_000, 8 * 60_000, 2_000)
                    add.append(pd.DataFrame({"timestamp": pd.to_datetime(ts, unit="ms", utc=True),
                                             "instrument_name": f"{cur}-PERPETUAL", "price": float(df.price.median()),
                                             "amount": 50_000.0, "direction": "buy" if side[i] == "long" else "sell",
                                             "mark_price": float(df.price.median()), "index_price": float(df.price.median()),
                                             "liquidation": pd.array([None] * len(ts), dtype="string"),
                                             "block_trade_id": pd.array([None] * len(ts), dtype="string")}))
                if add:
                    pd.concat([df] + add, ignore_index=True).sort_values("timestamp").to_parquet(p, index=False)
    rows = []
    for i in range(n):
        sym = "BTCUSDT" if i % 2 == 0 else "ETHUSDT"
        reason = "TARGET" if win[i] else ("STOP" if rng.random() < 0.7 else "TIMEOUT")
        dur = rng.uniform(20, 200) if reason != "TIMEOUT" else 240.0
        R = rng.uniform(30, 120)
        g = R if reason == "TARGET" else (-R if reason == "STOP" else rng.normal(0, R / 2))
        px = 60000.0
        rows.append({"a02_candidate_row": i, "symbol": sym, "side": side[i],
                     "signal_time_utc": str(pd.Timestamp(int(ent[i]) - 1000, unit="ms", tz="UTC")),
                     "reason": reason, "level_type": "pivot_5m", "target_source": "SESSION_VWAP",
                     "zone_halfwidth": 5.0, "zone_quote_touch_ratio": rng.random(), "opposing_zone_share_change": rng.normal(),
                     "zone_share_touch_change": rng.normal(), "absorption_like_exploratory": False,
                     "entry_time_ms": float(ent[i]), "exit_time_ms": float(ent[i] + dur * 60_000),
                     "entry_print": px, "entry_fill_assumed": px, "exit_fill_assumed": px, "target": px * (1 + R / 1e4),
                     "initial_risk_bps": R, "gross_return_bp": g, "net_return_bp_assumed_costs": g - 13.0,
                     "entry_fee_bp": 6.0, "exit_fee_bp": 6.0, "mfe_print_initial_r": 0.5, "mae_print_initial_r": -0.5})
    b = pd.DataFrame(rows)
    (Path(ext) / "03b").mkdir(parents=True, exist_ok=True)
    b.to_csv(Path(ext) / "03b" / "03b_1to1_events.csv", index=False)
