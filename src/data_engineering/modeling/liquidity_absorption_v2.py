"""
Liquidity Absorption V2: Ivan's own rules added to the frozen A02 second-touch detector, on the EXT venue inputs.

Protocol: discussions/Liquidity_Absorption/025_claude_v2_cvd_profile_preregistration.md

Rules (fixed before any V2 output was seen; written before the EXT results were opened):
    signals      the EXT A02 second-touch candidates (unchanged detector, Deribit perpetual 1m bars)
    CVD filter   absorption = aggressors push into the level but price holds. Signed quote flow
                 (taker buy - taker sell, USD) summed over the minutes AFTER the first-touch candle up to and
                 including the second-touch candle: long needs flow < 0 and second-touch low >= first-touch low
                 - 1 zone width; short needs flow > 0 and second-touch high <= first-touch high + 1 zone width.
    profile      traded volume by price from the session start (18:00 America/New_York, DST-aware, reset daily)
                 to the signal close, from the print tape, log-price bins of BIN_BP; 3-bin moving average.
                 No profile (NO_PROFILE) if the session is younger than MIN_SESSION_MIN at the signal.
    take-profit  walking from the signal close in the trade direction, beyond MIN_TARGET_BP: the first LVN
                 (contiguous bins < LVN_FRAC x POC), then the first bin after it with >= HVN_FRAC x POC (the
                 next HVN). TP = the near edge of that HVN (the far side of the LVN). Leaving the profile range
                 before finding it = NO_PROFILE_TARGET (no trade).
    stop         A03d risk_stop: the farthest of beyond the zone edge, beyond the second-touch wick (1 tick) and
                 SIGMA_MULT x the frozen VD18 60-minute sigma from the entry.
    management   FIXED / BREAKEVEN (+1R -> entry) / TRAIL_1M (A03 rules), 240-minute time stop, A03 entry
                 (first print after the signal close within 60 s) and A03 fill/adverse assumptions.
    CUT15        secondary hypothesis from Ivan's A04 observation: at the first print >= 15 min after entry,
                 exit (taker) if the position is losing.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

VERSION = "LA-V2-0.1.0"
NY = ZoneInfo("America/New_York")
MIN_MS = 60_000
DAY_MS = 86_400_000
CLOSED = {"STOP", "TARGET", "TIMEOUT", "CUT15"}


@dataclass(frozen=True)
class Spec:
    bin_bp: float = 5.0
    smooth_bins: int = 3
    min_session_min: int = 120
    lvn_frac: float = 0.20
    hvn_frac: float = 0.50
    min_target_bp: float = 10.0
    sigma_mult: float = 1.0
    cut_minutes: int = 15
    session_hour_ny: int = 18


def registration(spec: Spec = Spec()) -> dict:
    return {"version": VERSION, **asdict(spec),
            "primary": "CVD filter ON, profile LVN->HVN target, VD18 risk stop, TRAIL_1M, no CUT15, Deribit fees",
            "gate": "as LA 024: mean net > 0 and one-sided 95% day-bootstrap lower bound > 0 in BTC/ETH x H1/H2 holdout"}


# ----------------------------------------------------------------------------- session clock
def session_start_ms(t_ms: int, hour: int = 18) -> int:
    """UTC ms of the most recent 18:00 America/New_York wall-clock time at or before t (DST-aware)."""
    t = pd.Timestamp(int(t_ms), unit="ms", tz="UTC").tz_convert(NY)
    d = t.date()
    s = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=hour).tz_localize(NY)
    if s > t:
        d = d - pd.Timedelta(days=1)
        s = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=hour).tz_localize(NY)
    return int(s.tz_convert("UTC").value // 10**6)


# ----------------------------------------------------------------------------- CVD divergence
def cvd_divergence(bars: pd.DataFrame, ev: dict) -> tuple[bool, float]:
    """bars indexed by close_ms with quote_volume and taker_buy_quote_volume."""
    t1 = int(pd.Timestamp(ev["first_touch_close"]).value // 10**6)
    t2 = int(pd.Timestamp(ev["second_touch_close"]).value // 10**6)
    w = bars.loc[(bars.index > t1) & (bars.index <= t2)]
    flow = float((2 * w.taker_buy_quote_volume - w.quote_volume).sum())
    width = float(ev["zone_halfwidth"])
    if ev["side"] == "long":
        ok = flow < 0 and float(ev["bar_low"]) >= float(ev["first_touch_low"]) - width
    else:
        ok = flow > 0 and float(ev["bar_high"]) <= float(ev["first_touch_high"]) + width
    return bool(ok), flow


# ----------------------------------------------------------------------------- volume profile target
def profile_from_prints(px: np.ndarray, qty_usd: np.ndarray, bin_bp: float) -> tuple[np.ndarray, np.ndarray]:
    """(bin index array k, volume per bin) on a log grid: bin k covers [exp(k w), exp((k+1) w))."""
    w = math.log1p(bin_bp / 1e4)
    k = np.floor(np.log(px) / w).astype(np.int64)
    if len(k) == 0:
        return np.array([], np.int64), np.array([])
    lo = int(k.min())
    vol = np.bincount(k - lo, weights=qty_usd)
    return np.arange(lo, lo + len(vol)), vol


def lvn_hvn_target(bins: np.ndarray, vol: np.ndarray, ref: float, side: str, spec: Spec = Spec()) -> tuple[float, str]:
    if len(vol) < 5 or vol.max() <= 0:
        return np.nan, "NO_PROFILE"
    w = math.log1p(spec.bin_bp / 1e4)
    sm = pd.Series(vol).rolling(spec.smooth_bins, center=True, min_periods=1).mean().to_numpy()
    frac = sm / sm.max()
    d = 1 if side == "long" else -1
    k_ref = int(math.floor(math.log(ref) / w))
    k_min = int(math.floor(math.log(ref * (1 + d * spec.min_target_bp / 1e4)) / w))
    i = k_min - bins[0]
    in_lvn = False
    while 0 <= i < len(frac):
        if not in_lvn:
            if frac[i] < spec.lvn_frac:
                in_lvn = True
        elif frac[i] >= spec.hvn_frac:
            k = bins[0] + i
            edge = math.exp(k * w) if d == 1 else math.exp((k + 1) * w)     # near edge of the HVN bin
            if d * (edge - ref) <= 0:
                return np.nan, "NO_PROFILE_TARGET"
            return float(edge), "PROFILE_LVN_TO_HVN"
        i += d
    return np.nan, "NO_PROFILE_TARGET"


def attach_targets(events: pd.DataFrame, tape_root: Path, spec: Spec = Spec()) -> pd.DataFrame:
    """Session profile (prints with time in [session start, signal close)) and the LVN->HVN target per event."""
    import pyarrow.parquet as pq
    out = events.copy()
    out["session_start_ms"] = [session_start_ms(int(pd.Timestamp(t).value // 10**6), spec.session_hour_ny)
                               for t in out.signal_time_utc]
    out["signal_ms"] = [int(pd.Timestamp(t).value // 10**6) for t in out.signal_time_utc]
    out["session_age_min"] = (out.signal_ms - out.session_start_ms) / MIN_MS
    out["v2_target"] = np.nan
    out["v2_target_source"] = "NOT_COMPUTED"
    out["profile_poc"] = np.nan
    for (sym, s0), g in out.groupby(["symbol", "session_start_ms"]):
        last = int(g.signal_ms.max())
        days = sorted({pd.Timestamp(x, unit="ms", tz="UTC").strftime("%Y-%m-%d") for x in range(s0, last + 1, DAY_MS // 4)}
                      | {pd.Timestamp(last, unit="ms", tz="UTC").strftime("%Y-%m-%d")})
        parts = []
        for d in days:
            f = Path(tape_root) / "aggTrades" / sym / f"{sym}-aggTrades-{d}.parquet"
            if f.exists():
                t = pq.read_table(f, columns=["time_ms", "price", "qty"]).to_pandas()
                parts.append(t[(t.time_ms >= s0) & (t.time_ms < last)])
        tape = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["time_ms", "price", "qty"])
        tm = tape.time_ms.to_numpy("int64")
        for idx, e in g.iterrows():
            if e.session_age_min < spec.min_session_min:
                out.at[idx, "v2_target_source"] = "NO_PROFILE_SESSION_TOO_YOUNG"
                continue
            sel = tm < int(e.signal_ms)
            if sel.sum() < 100:
                out.at[idx, "v2_target_source"] = "NO_PROFILE"
                continue
            px = tape.price.to_numpy(float)[sel]
            usd = px * tape.qty.to_numpy(float)[sel]
            bins, vol = profile_from_prints(px, usd, spec.bin_bp)
            tgt, src = lvn_hvn_target(bins, vol, float(e.bar_close), e.side, spec)
            out.at[idx, "v2_target"] = tgt
            out.at[idx, "v2_target_source"] = src
            out.at[idx, "profile_poc"] = math.exp((bins[int(np.argmax(vol))] + 0.5) * math.log1p(spec.bin_bp / 1e4))
    return out


# ----------------------------------------------------------------------------- simulation
def make_sim_class():
    from src.data_engineering.modeling import liquidity_absorption_stage_a03 as a03
    from src.data_engineering.modeling import liquidity_absorption_stage_a03d as a03d

    class V2Sim(a03.EventSimulation):
        """A03 EventSimulation with the V2 stop/target; optional CUT15 losing-trade exit."""
        def __init__(self, event, bars, cfg, spec: Spec, cut: bool):
            self.v2 = dict(event)
            self.spec = spec
            self.cut = cut
            self.cut_checked = False
            super().__init__(event, bars, cfg)

        def _start(self, tm, px):
            if self.entry_ms is not None or self.entry_skip_reason:
                return
            def skip(reason):
                self.entry_skip_reason = reason
                for st in self.states.values():
                    self._mark_closed(st, reason)
            if tm > self.entry_limit:
                return skip("NO_PROMPT_ENTRY_PRINT")
            tgt = float(self.v2.get("v2_target", np.nan))
            if not np.isfinite(tgt):
                return skip(str(self.v2.get("v2_target_source", "NO_PROFILE_TARGET")))
            sig = float(self.v2.get("sigma_bp_60", np.nan))
            if not np.isfinite(sig):
                return skip("NO_VOL_FORECAST")
            try:
                stop = a03d.risk_stop(self.side, px, self.level, self.width,
                                      self.signal_low if self.d == 1 else self.signal_high, sig * self.spec.sigma_mult,
                                      self.tick, a03d.RiskConfig())
            except ValueError:
                return skip("INVALID_ENTRY_STOP_GEOMETRY")
            risk = self.d * (px - stop)
            if risk <= 0 or self.d * (tgt - px) <= 0:
                return skip("TARGET_NOT_BEYOND_ENTRY" if risk > 0 else "INVALID_ENTRY_STOP_GEOMETRY")
            self.initial_stop = stop
            self.entry_ms, self.entry_print, self.risk = tm, float(px), float(risk)
            self.target, self.target_source = tgt, "PROFILE_LVN_TO_HVN"
            for st in self.states.values():
                st.stop = stop
                st.entered = True
                st.entry_ms = tm
                st.entry_print = px
                st.entry_fill = px * (1 + self.d * self.cfg.entry_adverse_bp / 10000)
                st.initial_risk_price = risk
                st.trigger = tgt
                st.max_favorable_px = px
                st.max_adverse_px = px

        def feed(self, tm, px):
            if (self.cut and self.entry_ms is not None and not self.cut_checked
                    and tm >= self.entry_ms + self.spec.cut_minutes * MIN_MS and tm < self.deadline):
                self.cut_checked = True
                if self.d * (px - self.entry_print) < 0:
                    for st in self.states.values():
                        if not st.closed:
                            self._mark_closed(st, "CUT15", tm, px)
                    return
            super().feed(tm, px)

        def records(self):
            out = super().records()
            for r in out:
                r["v2_variant"] = ("CUT15_" if self.cut else "") + r["mode"]
                r["cvd_ok"] = bool(self.v2.get("cvd_ok", False))
                r["cvd_flow_between_touches"] = self.v2.get("cvd_flow", np.nan)
                r["v2_target_source"] = self.v2.get("v2_target_source")
                r["sigma_bp_60"] = self.v2.get("sigma_bp_60")
                r["session_age_min"] = self.v2.get("session_age_min")
                r["reward_to_risk"] = (self.d * (self.target - self.entry_print) / self.risk
                                       if self.entry_ms is not None else np.nan)
            return out

    return V2Sim


def run_v2(ext: Path, root: Path, repo: Path, out: Path, spec: Spec = Spec()) -> pd.DataFrame:
    from src.data_engineering.modeling import liquidity_absorption_extended as ex
    from src.data_engineering.modeling import liquidity_absorption_stage_a03 as a03
    from src.data_engineering.modeling import liquidity_absorption_stage_a03d as a03d
    cands = ex.load_candidates_any(ext)
    bars = {s: a03.prepare_bars(Path(root), s) for s in ex.SYMBOLS}
    flow = {}
    for s in ex.SYMBOLS:
        f = pd.read_parquet(Path(root) / f"{s}_um_1m.parquet", columns=["open_time", "quote_volume", "taker_buy_quote_volume"])
        f.index = pd.to_datetime(f.open_time, utc=True).dt.as_unit("ms").astype("int64") + MIN_MS
        flow[s] = f
    cv = [cvd_divergence(flow[r["symbol"]], r) for r in cands.to_dict("records")]
    cands["cvd_ok"] = [c[0] for c in cv]
    cands["cvd_flow"] = [c[1] for c in cv]
    fits, _ = a03d.load_vd18_fits(Path(repo))
    fc = a03d.forecast_origins(cands, Path(root) / "price_5m", fits)
    cands = cands.merge(fc[["a02_candidate_row", "sigma_bp_60", "forecast_status"]], on="a02_candidate_row", how="left")
    cands = attach_targets(cands, Path(root), spec)
    out.mkdir(parents=True, exist_ok=True)
    cands.to_csv(out / "V2_signals.csv", index=False)
    V2Sim = make_sim_class()
    sims = []
    for sym, g in cands.groupby("symbol"):
        for ev in g.to_dict("records"):
            for cut in (False, True):
                sims.append(V2Sim(ev, bars[sym], a03.Config(), spec, cut))
    a03.scan_event_tapes(Path(root), sims)
    res = pd.DataFrame([r for s in sims for r in s.records()])
    res.to_csv(out / "V2_events.csv", index=False)
    return res


# ----------------------------------------------------------------------------- evaluation and Kelly
def kelly_growth(r: np.ndarray) -> tuple[float, float]:
    """f* maximizing mean log(1 + f R) over f in [0, 1) for per-trade R multiples (net, in units of risk)."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) < 10 or r.mean() <= 0:
        return 0.0, 0.0
    fs = np.linspace(0, 0.99, 991)
    lo = -1.0 / r.min() if r.min() < 0 else 1.0
    best_f, best_g = 0.0, 0.0
    for f in fs:
        if f >= lo * 0.999:
            break
        g = float(np.mean(np.log1p(f * r)))
        if g > best_g:
            best_f, best_g = float(f), g
    return best_f, best_g


def evaluate(out: Path) -> tuple[pd.DataFrame, dict]:
    from src.data_engineering.modeling import liquidity_absorption_extended as ex
    ev = pd.read_csv(Path(out) / "V2_events.csv", low_memory=False)
    ev["net_deribit"] = ex.deribit_net(ev)
    risk_bp = ev["initial_risk_bps"].astype(float)
    ev["net_R_deribit"] = ev["net_deribit"] / risk_bp
    rows = []
    for (cvd_on, variant), g0 in [((True, v), g[g.cvd_ok]) for v, g in ev.groupby("v2_variant")] + \
                                  [((False, v), g) for v, g in ev.groupby("v2_variant")]:
        z = g0[g0.reason.isin(CLOSED)].copy()
        if z.empty:
            continue
        z["period"] = ex.period_label(z.signal_time_utc)
        z["day"] = pd.to_datetime(z.signal_time_utc, utc=True).dt.floor("D")
        hold = z[z.period != "INSAMPLE_JUL_SEP_2026"]
        groups = [((s, p), g) for (s, p), g in z.groupby(["symbol", "period"])] + \
                 [((s, "HOLDOUT"), g) for s, g in hold.groupby("symbol")] + [(("ALL", "HOLDOUT"), hold)]
        for (sym, per), g in groups:
            m, lo = ex.cluster_lower(g.net_deribit, g.day)
            mg, glo = ex.cluster_lower(g.gross_return_bp, g.day)
            f, gr = kelly_growth(g.net_R_deribit.to_numpy())
            rows.append({"cvd_filter": cvd_on, "variant": variant, "symbol": sym, "period": per, "closed": len(g),
                         "target": int((g.reason == "TARGET").sum()), "stop": int((g.reason == "STOP").sum()),
                         "timeout": int((g.reason == "TIMEOUT").sum()), "cut15": int((g.reason == "CUT15").sum()),
                         "win_rate_net": float((g.net_deribit > 0).mean()),
                         "median_reward_to_risk": float(g.reward_to_risk.median()),
                         "mean_gross_bp": mg, "gross_lo95": glo, "mean_net_bp": m, "net_lo95": lo,
                         "mean_net_R": float(g.net_R_deribit.mean()), "kelly_f": f, "kelly_log_growth": gr})
    T = pd.DataFrame(rows)
    p = T[T.cvd_filter & T.variant.eq("TRAIL_1M")].set_index(["symbol", "period"])
    cells = {f"{s}_{h}": (bool(p.loc[(s, h), "mean_net_bp"] > 0 and p.loc[(s, h), "net_lo95"] > 0)
                          if (s, h) in p.index else False) for s in ex.SYMBOLS for h in ("H1", "H2")}
    hold = {s: (bool(p.loc[(s, "HOLDOUT"), "net_lo95"] > 0) if (s, "HOLDOUT") in p.index else False) for s in ex.SYMBOLS}
    sig = pd.read_csv(Path(out) / "V2_signals.csv", low_memory=False)
    dec = {"version": VERSION, "primary": "CVD on, TRAIL_1M, profile target, Deribit fees", "cells": cells,
           "holdout_by_symbol_lo95_positive": hold,
           "verdict": "PASS" if all(cells.values()) and all(hold.values()) else "FAIL",
           "signals": int(len(sig)), "cvd_ok_share": float(sig.cvd_ok.mean()) if len(sig) else None,
           "target_sources": sig.v2_target_source.value_counts().to_dict()}
    return T, dec
