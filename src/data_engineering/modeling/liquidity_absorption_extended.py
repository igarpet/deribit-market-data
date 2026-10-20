"""
Liquidity Absorption EXT: the whole LA research module (A02 -> A02b -> A03 -> A03b -> A03c -> A03d -> A04 -> A05a
-> A06) re-run, venue-native on Deribit, from 2024-01-01 to the latest downloaded day.

Protocol: discussions/Liquidity_Absorption/024_claude_extended_deribit_full_period_preregistration.md

Why venue-native: the A02 zone is +-2 bp around a level; Binance-derived levels cannot be executed on Deribit prints
(cross-venue price differences are of the same order), and Binance aggTrades for 2024-2026 are not on disk.

Inputs built here from the downloaded Deribit public trade history (history/trades/<CUR>/future/YYYY/day.parquet):
    <root>/aggTrades/<SYM>/<SYM>-aggTrades-<day>.parquet   time_ms, price, qty (coin = USD amount / price),
                                                           is_buyer_maker (= taker sold)   [A02/A02b/A03/A03c tape]
    <root>/<SYM>_um_1m.parquet                              1m OHLCV from perpetual prints in the Binance kline schema
                                                           (open_time = minute OPEN; empty minutes = previous close,
                                                           zero volume)                    [A02 detector, bars]
    <root>/price_5m/<CUR>.csv.gz                            5m close of the Deribit INDEX (last print's index_price
                                                           in each bar), state_time = bar END   [VD18 spot input]
Symbols keep the engines' labels BTCUSDT / ETHUSDT (they mean Deribit BTC-PERPETUAL / ETH-PERPETUAL here).

The stage engines are imported unchanged. Their run() wrappers assert the original 452-signal quarter, so this module
re-implements only the thin loaders/orchestration without those count checks; every simulation, feature and
target/stop rule is the engines' own code.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "LA-EXT-0.1.0"
SYMBOLS = {"BTCUSDT": ("BTC", "BTC-PERPETUAL"), "ETHUSDT": ("ETH", "ETH-PERPETUAL")}
MIN_MS = 60_000
DAY_MS = 86_400_000
INSAMPLE = (pd.Timestamp("2026-07-01", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC"))   # inspected since A02
HALF_SPLIT = pd.Timestamp("2025-04-01", tz="UTC")
DERIBIT_TAKER_BP = 5.0
DERIBIT_MAKER_BP = 0.0
BOOT_B = 2000
SEED = 20261020
CLOSED = {"STOP", "TARGET", "TIMEOUT"}


@dataclass(frozen=True)
class Spec:
    start: str = "2024-01-01"
    end: str | None = None          # inclusive; default = last complete Deribit day on disk


def registration(spec: Spec = Spec()) -> dict:
    return {"version": VERSION, **asdict(spec), "venue": "Deribit BTC-PERPETUAL / ETH-PERPETUAL prints",
            "in_sample_window": [str(INSAMPLE[0]), str(INSAMPLE[1])], "half_split": str(HALF_SPLIT),
            "deribit_fees_bp": {"taker": DERIBIT_TAKER_BP, "maker": DERIBIT_MAKER_BP}, "boot_B": BOOT_B,
            "primary": "A03b forward 1:1 taker (FIXED), closed trades, Deribit fees",
            "gate": "mean net bp per closed trade > 0 with one-sided 95% day-cluster bootstrap lower bound > 0, in BTC "
                    "and ETH separately, and in both holdout halves (before / from 2025-04-01), holdout = all days "
                    "outside 2026-07-01..2026-09-30"}


# ----------------------------------------------------------------------------- 1. venue inputs from Deribit prints
def _day_list(history: Path, cur: str, start: str, end: str | None) -> list[str]:
    files = sorted((Path(history) / "trades" / cur / "future").glob("*/*.parquet"))
    days = [f.stem for f in files if f.stem >= start and (end is None or f.stem <= end)]
    return days


def _perp_prints(path: Path, perp: str) -> pd.DataFrame:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    cols = [c for c in ("timestamp", "instrument_name", "trade_seq", "price", "amount", "direction", "index_price")
            if c in pf.schema.names]
    d = pf.read(columns=cols).to_pandas()
    d = d[d.instrument_name.astype(str).eq(perp)]
    t = d.timestamp
    if pd.api.types.is_numeric_dtype(t):
        tm = pd.to_numeric(t).to_numpy("int64")
    else:
        tm = pd.to_datetime(t, utc=True).dt.as_unit("ms").astype("int64").to_numpy()
    out = pd.DataFrame({"time_ms": tm, "seq": pd.to_numeric(d.trade_seq, errors="coerce").to_numpy(),
                        "price": pd.to_numeric(d.price, errors="coerce").to_numpy(float),
                        "usd": pd.to_numeric(d.amount, errors="coerce").to_numpy(float),
                        "buy": d.direction.astype(str).eq("buy").to_numpy(),
                        "index": pd.to_numeric(d.get("index_price"), errors="coerce").to_numpy(float)
                        if "index_price" in d else np.nan})
    out = out[np.isfinite(out.price) & (out.price > 0) & np.isfinite(out.usd) & (out.usd > 0)]
    return out.sort_values(["time_ms", "seq"], kind="mergesort").reset_index(drop=True)


def day_inputs(pr: pd.DataFrame, day: str, prev_close: float | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(aggTrades frame, 1440 minute bars, 288 index 5m closes) for one UTC day of perpetual prints."""
    t0 = int(pd.Timestamp(day, tz="UTC").value // 10**6)
    agg = pd.DataFrame({"time_ms": pr.time_ms.astype("int64"), "price": pr.price, "qty": pr.usd / pr.price,
                        "is_buyer_maker": ~pr.buy.astype(bool)})
    m = (pr.time_ms - t0) // MIN_MS
    g = pd.DataFrame({"m": m, "p": pr.price, "usd": pr.usd, "qty": pr.usd / pr.price,
                      "bq": np.where(pr.buy, pr.usd, 0.0)}).groupby("m")
    bars = pd.DataFrame({"open": g.p.first(), "high": g.p.max(), "low": g.p.min(), "close": g.p.last(),
                         "volume": g.qty.sum(), "quote_volume": g.usd.sum(), "taker_buy_quote_volume": g.bq.sum(),
                         "trades": g.p.size()}).reindex(range(1440))
    close = bars.close.ffill()
    if prev_close is not None and np.isfinite(prev_close):
        close = close.fillna(prev_close)
    else:
        close = close.bfill()
    empty = bars.open.isna()
    for c in ("open", "high", "low", "close"):
        bars.loc[empty, c] = close[empty]
    for c in ("volume", "quote_volume", "taker_buy_quote_volume", "trades"):
        bars[c] = bars[c].fillna(0.0)
    bars["open_time"] = pd.to_datetime(t0 + bars.index.to_numpy() * MIN_MS, unit="ms", utc=True)
    bars = bars.dropna(subset=["close"])
    # index 5m close: last index price with time in (end-5m, end] -> state_time = bar END
    idx = pr[np.isfinite(pr["index"])]
    b5 = (idx.time_ms - t0 - 1) // (5 * MIN_MS)
    ix = pd.DataFrame({"b": b5, "close": idx["index"]}).groupby("b").close.last()
    ix = ix[(ix.index >= 0) & (ix.index < 288)]
    ix5 = pd.DataFrame({"state_time": pd.to_datetime(t0 + (ix.index.to_numpy() + 1) * 5 * MIN_MS, unit="ms", utc=True),
                        "close": ix.to_numpy()})
    return agg, bars[["open_time", "open", "high", "low", "close", "volume", "quote_volume",
                      "taker_buy_quote_volume", "trades"]], ix5


def prepare_venue_root(history: Path, root: Path, spec: Spec = Spec(), log=print) -> pd.DataFrame:
    """Writes the venue inputs; resumable per day. Returns per-day coverage."""
    root = Path(root)
    cov = []
    for sym, (cur, perp) in SYMBOLS.items():
        days = _day_list(history, cur, spec.start, spec.end)
        (root / "aggTrades" / sym).mkdir(parents=True, exist_ok=True)
        (root / "_bars1m" / sym).mkdir(parents=True, exist_ok=True)
        (root / "_index5m" / cur).mkdir(parents=True, exist_ok=True)
        prev_close = None
        prev_day = None
        for k, day in enumerate(days):
            fa = root / "aggTrades" / sym / f"{sym}-aggTrades-{day}.parquet"
            fb = root / "_bars1m" / sym / f"{day}.parquet"
            fi = root / "_index5m" / cur / f"{day}.parquet"
            contiguous = prev_day is not None and (pd.Timestamp(day) - pd.Timestamp(prev_day)).days == 1
            if fa.exists() and fb.exists() and fi.exists():
                b = pd.read_parquet(fb, columns=["close"])
                prev_close = float(b.close.iloc[-1]) if len(b) else prev_close
                n = None
            else:
                pr = _perp_prints(history / "trades" / cur / "future" / day[:4] / f"{day}.parquet", perp)
                agg, bars, ix5 = day_inputs(pr, day, prev_close if contiguous else None)
                agg.to_parquet(fa, index=False)
                bars.to_parquet(fb, index=False)
                ix5.to_parquet(fi, index=False)
                prev_close = float(bars.close.iloc[-1]) if len(bars) else prev_close
                n = len(agg)
            prev_day = day
            cov.append({"symbol": sym, "day": day, "prints": n})
            if log and (k % 50 == 0 or k == len(days) - 1):
                log(f"  {sym} {day} ({k + 1}/{len(days)})")
        bars = pd.concat([pd.read_parquet(root / "_bars1m" / sym / f"{d}.parquet") for d in days], ignore_index=True)
        bars = bars.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
        bars.to_parquet(root / f"{sym}_um_1m.parquet", index=False)
        ix = pd.concat([pd.read_parquet(root / "_index5m" / cur / f"{d}.parquet") for d in days], ignore_index=True)
        ix = ix.drop_duplicates("state_time").sort_values("state_time").reset_index(drop=True)
        (root / "price_5m").mkdir(exist_ok=True)
        ix.to_csv(root / "price_5m" / f"{cur}.csv.gz", index=False)
    return pd.DataFrame(cov)


# ----------------------------------------------------------------------------- 2. stages (engines unchanged)
def _base(shadow: Path) -> Path:
    """Output tree of the extended run: <shadow>/{02,02b,03,03b,03c,03d,04,05a,06}."""
    return Path(shadow)


def stage_a02(root: Path, shadow: Path, spec: Spec) -> dict:
    from src.data_engineering.modeling import liquidity_absorption_stage_a02 as a02
    end = spec.end or max(p.stem.split("-aggTrades-")[1] for p in (Path(root) / "aggTrades" / "BTCUSDT").glob("*.parquet"))
    cfg = a02.Config(start=spec.start, end=end)
    return a02.run_a02(Path(root), _base(shadow) / "02", cfg)


def stage_a02b(root: Path, shadow: Path) -> dict:
    """A02b.run_a02b's work (read A02 candidates, enrich from the tape) without its repo-path coupling."""
    from src.data_engineering.modeling import liquidity_absorption_stage_a02b as a02b
    cands = a02b.read_candidates(_base(shadow) / "02" / "02_price_only_candidates.csv")
    feats = a02b.enrich_a02_candidates(cands, Path(root), a02b.Config())
    out = _base(shadow) / "02b"
    out.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out / "02b_touch_features.csv", index=False)
    dec = {"version": a02b.LA_A02B_VERSION, "candidate_count": int(len(feats)),
           "touch_status": feats.touch_status.value_counts().to_dict()}
    (out / "02b_decision.json").write_text(json.dumps(dec, indent=1))
    return dec


def load_candidates_any(shadow: Path) -> pd.DataFrame:
    """A03.load_candidates without the 452-count assertion (identity checks kept)."""
    from src.data_engineering.modeling.liquidity_absorption_stage_a03 import millis
    base = _base(shadow)
    a = pd.read_csv(base / "02b" / "02b_touch_features.csv")
    flow = pd.read_csv(base / "02" / "02_causal_flow_features.csv")
    if len(flow) != len(a) or not a.a02_candidate_row.is_unique:
        raise ValueError("A02/A02b candidate identity/count mismatch")
    q = flow.iloc[a.a02_candidate_row.to_numpy(int)]
    if not (q.symbol.to_numpy() == a.symbol.to_numpy()).all() or not (q.side.to_numpy() == a.side.to_numpy()).all():
        raise ValueError("A02/A02b event identity drift")
    if not np.array_equal([millis(x) for x in q.signal_time_utc], [millis(x) for x in a.signal_time_utc]):
        raise ValueError("A02/A02b signal time drift")
    a["vwap_utc_session"] = q["vwap_utc_session"].to_numpy()
    return a


def stage_a03(root: Path, shadow: Path) -> pd.DataFrame:
    from src.data_engineering.modeling import liquidity_absorption_stage_a03 as a03
    a = load_candidates_any(shadow)
    sims = []
    for sym, sub in a.groupby("symbol"):
        bars = a03.prepare_bars(Path(root), sym)
        for ev in sub.to_dict("records"):
            sims.append(a03.EventSimulation(ev, bars, a03.Config()))
    tape = a03.scan_event_tapes(Path(root), sims)
    res = pd.DataFrame([r for s in sims for r in s.records()])
    out = _base(shadow) / "03"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "03_execution_events.csv", index=False)
    (out / "03_decision.json").write_text(json.dumps({"version": "LA-A03-0.1.0", "source_events": int(len(a)),
                                                      "tape": {k: (v if k != "missing_trade_day_files" else len(v))
                                                               for k, v in tape.items()}}, indent=1))
    return res


def stage_a03b(root: Path, shadow: Path) -> pd.DataFrame:
    from src.data_engineering.modeling import liquidity_absorption_stage_a03 as a03
    from src.data_engineering.modeling import liquidity_absorption_stage_a03b as a03b
    a = load_candidates_any(shadow)
    sims = []
    for sym, sub in a.groupby("symbol"):
        bars = a03.prepare_bars(Path(root), sym)
        for ev in sub.to_dict("records"):
            sims.append(a03b.MirroredStopSimulation(ev, bars, a03.Config()))
    a03.scan_event_tapes(Path(root), sims)
    new = pd.DataFrame([s.fixed_record() for s in sims])
    out = _base(shadow) / "03b"
    out.mkdir(parents=True, exist_ok=True)
    new.to_csv(out / "03b_1to1_events.csv", index=False)
    (out / "03b_decision.json").write_text(json.dumps({"version": "LA-A03b-0.1.0", "source_events": int(len(new))}))
    return new


def validate_inputs_any(shadow: Path) -> pd.DataFrame:
    """A03c.validate_inputs without the 452-count assertions."""
    base = _base(shadow)
    src = pd.read_csv(base / "03b" / "03b_1to1_events.csv")
    obs = pd.read_csv(base / "02b" / "02b_touch_features.csv",
                      usecols=["a02_candidate_row", "symbol", "side", "signal_time_utc", "level_price", "bar_close"])
    j = src.merge(obs, on="a02_candidate_row", validate="one_to_one", how="left", suffixes=("", "_a02b"))
    if j.bar_close.isna().any() or not j.symbol.eq(j.symbol_a02b).all() or not j.side.eq(j.side_a02b).all():
        raise ValueError("A02b/A03b identity mismatch")
    return j.drop(columns=[f"{n}_a02b" for n in ("symbol", "side", "signal_time_utc", "level_price")])


def stage_a03c(root: Path, shadow: Path) -> pd.DataFrame:
    from src.data_engineering.modeling import liquidity_absorption_stage_a03c as a03c
    src = validate_inputs_any(shadow)
    cfg = a03c.Config()
    sims = [a03c.TapeStudy(dict(r), mode, cfg) for r in src.to_dict("records") for mode in a03c.MODES]
    a03c.scan_tapes(Path(root), sims, cfg.batch_rows)
    tr = pd.DataFrame.from_records([s.record() for s in sims])
    out = _base(shadow) / "03c"
    out.mkdir(parents=True, exist_ok=True)
    tr.to_csv(out / "03c_events.csv", index=False)
    return tr


def stage_a03d(root: Path, shadow: Path, repo: Path) -> pd.DataFrame:
    """A03d.build_results without 452 assertions; VD18 fits from the real repo, index 5m from the venue root."""
    from src.data_engineering.modeling import liquidity_absorption_stage_a03 as a03
    from src.data_engineering.modeling import liquidity_absorption_stage_a03c as a03c
    from src.data_engineering.modeling import liquidity_absorption_stage_a03d as a03d
    cfg = a03d.RiskConfig()
    candidates = load_candidates_any(shadow)
    base = validate_inputs_any(shadow).merge(candidates[["a02_candidate_row", "bar_low", "bar_high"]],
                                             on="a02_candidate_row", validate="one_to_one")
    fits, prov = a03d.load_vd18_fits(Path(repo))
    origins = a03d.forecast_origins(base, Path(root) / "price_5m", fits)
    wf = base.merge(origins.drop(columns=["symbol", "signal_time_utc"]), on="a02_candidate_row", validate="one_to_one")
    out = _base(shadow) / "03d"
    out.mkdir(parents=True, exist_ok=True)
    wf.to_csv(out / "03d_event_forecasts.csv", index=False)
    usable = wf[wf.forecast_status == "PRESENT"].copy()
    smap = {int(z.a02_candidate_row): z for z in candidates.itertuples(index=False)}

    class VolTaker(a03.EventSimulation):
        def __init__(self, event, bars, c, source):
            self.vol_source = source
            super().__init__(event, bars, c)

        def _start(self, tm, px):
            super()._start(tm, px)
            if self.entry_ms is None:
                return
            ex = self.vol_source
            if (not np.isclose(px, float(ex["entry_print"]), rtol=1e-10, atol=1e-7)
                    or int(tm) != int(float(ex["entry_time_ms"]))
                    or not np.isclose(self.target, float(ex["target"]), rtol=1e-10, atol=1e-7)):
                raise AssertionError("A03b entry/target drift")
            st = a03d.risk_stop(self.side, px, self.level, self.width,
                                self.signal_low if self.d == 1 else self.signal_high,
                                float(ex["sigma_bp_60"]), self.tick, cfg)
            self.initial_stop = st
            self.risk = self.d * (px - st)
            for s in self.states.values():
                s.stop = st
                s.initial_risk_price = self.risk

        def records(self):
            o = super().records()
            for r in o:
                r.update(entry_kind="TAKER_FIRST_POST_SIGNAL", target_frozen_a03b=True,
                         vol_sigma_bp_60=self.vol_source["sigma_bp_60"], vd18_origin_utc=self.vol_source["vd18_origin_utc"],
                         hold_minutes=self.cfg.hold_minutes, model_mode="VD18_VOL_" + r["mode"],
                         initial_risk_bp=r["initial_risk_bps"], maker_eligibility_not_fill=False)
            return o

    class VolMaker(a03c.TapeStudy):
        def __post_init__(self):
            super().__post_init__()
            if self.reason:
                return
            self.stop = a03d.risk_stop(self.side, self.level, self.level, float(self.source["zone_halfwidth"]),
                                       float(self.source["bar_low"] if self.d == 1 else self.source["bar_high"]),
                                       float(self.source["sigma_bp_60"]), a03.tick_for(self.symbol, a03.Config()), cfg)

        def _entry(self, tm, entry_price, print_price):
            if not (self.d * (entry_price - self.stop) > 0 and self.d * (self.target - entry_price) > 0):
                raise ValueError("Invalid passive volatility stop/structural target")
            self.entry_time, self.entry_px, self.qualifying_print = tm, entry_price, print_price
            self.max_favorable = self.max_adverse = entry_price

        def record(self):
            r = super().record()
            r.update(entry_kind="MAKER_ELIGIBILITY_ONLY", target_frozen_a03b=True,
                     vol_sigma_bp_60=self.source["sigma_bp_60"], vd18_origin_utc=self.source["vd18_origin_utc"],
                     hold_minutes=self.cfg.hold_minutes, model_mode="VD18_VOL_" + self.mode,
                     maker_eligibility_not_fill=True)
            return r

    takers, makers = [], []
    for sym, g in usable.groupby("symbol"):
        bars = a03.prepare_bars(Path(root), sym)
        for rec in g.to_dict("records"):
            raw = smap[int(rec["a02_candidate_row"])]._asdict()
            for hold in cfg.hold_windows:
                takers.append(VolTaker(raw, bars, a03.Config(hold_minutes=hold), rec))
                for mode in ("MAKER_TOUCH_3RD", "MAKER_THROUGH_3RD"):
                    makers.append(VolMaker(rec, mode, a03c.Config(hold_minutes=hold)))
    if takers:
        a03.scan_event_tapes(Path(root), takers)
    if makers:
        a03c.scan_tapes(Path(root), makers)
    fr = pd.DataFrame.from_records([r for s in takers for r in s.records()] + [s.record() for s in makers])
    fr.to_csv(out / "03d_risk_execution.csv", index=False)
    (out / "03d_contract.json").write_text(json.dumps({"version": "LA-A03d-0.1.0", "source_events": int(len(wf)),
                                                       "forecast_present": int(len(usable)), "vd18": prov}, default=str))
    return fr


def stage_a04(root: Path, shadow: Path) -> pd.DataFrame:
    from src.data_engineering.modeling import liquidity_absorption_stage_a04 as a04
    data = pd.read_csv(_base(shadow) / "03d" / "03d_risk_execution.csv", low_memory=False)
    cfg = a04.Config()
    z = data.loc[(data.entry_kind == cfg.entry_kind) & (data.model_mode == cfg.model_mode)
                 & (pd.to_numeric(data.hold_minutes) == cfg.hold_minutes)].copy()
    z["signal_ms"] = pd.to_datetime(z.signal_time_utc, utc=True).dt.as_unit("ms").astype("int64")
    sgn = np.where(z.side == "long", 1, -1)
    z["target_distance_bp"] = sgn * (pd.to_numeric(z.target) - pd.to_numeric(z.entry_print)) / pd.to_numeric(z.entry_print) * 1e4
    z["risk_distance_bp"] = (pd.to_numeric(z.entry_print) - pd.to_numeric(z.initial_stop)).abs() / pd.to_numeric(z.entry_print) * 1e4
    z["target_to_sigma60"] = z.target_distance_bp / pd.to_numeric(z.vol_sigma_bp_60)
    z["time_to_exit_min"] = (pd.to_numeric(z.exit_time_ms) - pd.to_numeric(z.entry_time_ms)) / MIN_MS
    z = z.sort_values("a02_candidate_row").reset_index(drop=True)
    out = _base(shadow) / "04"
    out.mkdir(parents=True, exist_ok=True)
    z.to_csv(out / "04_primary_event_outcomes.csv", index=False)
    try:
        minutes = {s: a04.load_minute_bars(Path(root) / f"{s}_um_1m.parquet") for s in SYMBOLS}
        lm = a04.build_landmarks(z, minutes, cfg)
        a04.summarize_landmarks(lm).to_csv(out / "04_landmark_summary.csv", index=False)
        a04.cumulative_competing_exits(z).to_csv(out / "04_competing_exits.csv", index=False)
    except Exception as e:  # descriptive stage must not block the verdict
        (out / "04_error.txt").write_text(repr(e))
    return z


def stage_a05a(root: Path, shadow: Path) -> dict:
    from src.data_engineering.modeling import liquidity_absorption_stage_a05a as a05a
    base = _base(shadow)
    z = pd.read_csv(base / "04" / "04_primary_event_outcomes.csv", low_memory=False)
    b = pd.read_csv(base / "02b" / "02b_touch_features.csv", usecols=[
        "a02_candidate_row", "bar_close", "first_touch_close", "second_touch_close", "level_source_time", "level_type"])
    j = z.drop(columns=[c for c in ("level_type",) if c in z]).merge(b, on="a02_candidate_row", validate="one_to_one")
    j["signal_ms"] = a05a.ts_millis(j.signal_time_utc)
    j["first_touch_ms"] = a05a.ts_millis(j.first_touch_close)
    j["level_source_ms"] = a05a.ts_millis(j.level_source_time)
    j["eventual_outcome"] = j.reason
    j["feasibility_band"] = np.where(j.reason.isin(a05a.OUTCOMES) & pd.to_numeric(j.target_to_sigma60).le(1.0),
                                     "TARGET_LE_1SIGMA60", "TARGET_GT_1SIGMA60")
    j.loc[~j.reason.isin(a05a.OUTCOMES), "feasibility_band"] = "NO_CLOSED_OUTCOME"
    cfg = a05a.Config()
    out = base / "05a"
    out.mkdir(parents=True, exist_ok=True)
    try:
        candles = {s: a05a.load_context_minutes(Path(root) / f"{s}_um_1m.parquet", cfg) for s in SYMBOLS}
        panel = a05a.build_panel(j, candles, cfg)
        curves, checkpoints, qc = a05a.summaries(panel, j)
        phases, ps = a05a.phase_panels(panel)
        curves.to_csv(out / "05a_median_context_curves.csv", index=False)
        phases.to_csv(out / "05a_phase_panel.csv", index=False)
        ps.to_csv(out / "05a_phase_summary.csv", index=False)
        return {"signals": int(len(j)), "minute_rows": int(len(panel))}
    except Exception as e:
        (out / "05a_error.txt").write_text(repr(e))
        return {"error": repr(e)}


# ----------------------------------------------------------------------------- 3. evaluation
def deribit_net(df: pd.DataFrame, entry_maker: bool = False) -> pd.Series:
    """Reprice the engines' 6 bp taker (2 bp maker) fees at Deribit's schedule; fills/adverse unchanged."""
    ent = df["entry_fill_assumed"].astype(float)
    exi = df["exit_fill_assumed"].astype(float)
    ref = (df["entry_price"] if "entry_price" in df and entry_maker else df["entry_print"]).astype(float)
    old = (df["entry_fee_bp"].astype(float) * ent + df["exit_fee_bp"].astype(float) * exi) / ref
    new = ((DERIBIT_MAKER_BP if entry_maker else DERIBIT_TAKER_BP) * ent + DERIBIT_TAKER_BP * exi) / ref
    return df["net_return_bp_assumed_costs"].astype(float) + old - new


def period_label(sig: pd.Series) -> pd.Series:
    t = pd.to_datetime(sig, utc=True)
    lab = np.where(t < HALF_SPLIT, "H1", "H2")
    lab = np.where((t >= INSAMPLE[0]) & (t < INSAMPLE[1]), "INSAMPLE_JUL_SEP_2026", lab)
    return pd.Series(lab, index=sig.index)


def cluster_lower(x: pd.Series, day: pd.Series, B: int = BOOT_B, seed: int = SEED) -> tuple[float, float]:
    """Mean and one-sided 95% lower bound, bootstrap over signal days (overlapping trades share risk)."""
    d = pd.DataFrame({"x": x.to_numpy(float), "d": day.to_numpy()}).dropna()
    if len(d) < 5:
        return float("nan"), float("nan")
    g = d.groupby("d").x.agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(B, len(g)))
    means = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.quantile(means, 0.05))


def evaluate(shadow: Path) -> tuple[pd.DataFrame, dict]:
    base = _base(shadow)
    rows = []

    def add(name, df, net_col, gross_col="gross_return_bp"):
        z = df[df.reason.isin(CLOSED)].copy()
        if z.empty:
            return
        z["period"] = period_label(z.signal_time_utc)
        z["day"] = pd.to_datetime(z.signal_time_utc, utc=True).dt.floor("D")
        for (sym, per), g in list(z.groupby(["symbol", "period"])) + [(("ALL", p), g) for p, g in z.groupby("period")] + \
                [((s, "HOLDOUT"), g) for s, g in z[z.period != "INSAMPLE_JUL_SEP_2026"].groupby("symbol")] + \
                [(("ALL", "HOLDOUT"), z[z.period != "INSAMPLE_JUL_SEP_2026"])]:
            m, lo = cluster_lower(g[net_col], g.day)
            mg, log_ = cluster_lower(g[gross_col], g.day)
            rows.append({"variant": name, "symbol": sym, "period": per, "closed": int(len(g)),
                         "target": int((g.reason == "TARGET").sum()), "stop": int((g.reason == "STOP").sum()),
                         "timeout": int((g.reason == "TIMEOUT").sum()), "mean_gross_bp": mg, "gross_lo95": log_,
                         "mean_net_bp": m, "net_lo95": lo, "net_col": net_col})

    b = pd.read_csv(base / "03b" / "03b_1to1_events.csv", low_memory=False)
    b["net_deribit"] = deribit_net(b)
    add("A03b_forward_1to1_taker", b, "net_deribit")
    add("A03b_forward_1to1_taker_frozen_costs", b, "net_return_bp_assumed_costs")
    a3 = pd.read_csv(base / "03" / "03_execution_events.csv", low_memory=False)
    a3["net_deribit"] = deribit_net(a3)
    for mode, g in a3.groupby("mode"):
        add(f"A03_{mode}", g, "net_deribit")
    c = pd.read_csv(base / "03c" / "03c_events.csv", low_memory=False)
    for mode, g in c.groupby("mode"):
        g = g.copy()
        g["net_deribit"] = deribit_net(g, entry_maker=mode != "REVERSE_TAKER_1TO1")
        add(f"A03c_{mode}", g, "net_deribit")
    if (base / "03d" / "03d_risk_execution.csv").exists():
        d = pd.read_csv(base / "03d" / "03d_risk_execution.csv", low_memory=False)
        for (mm, h), g in d.groupby(["model_mode", "hold_minutes"]):
            g = g.copy()
            g["net_deribit"] = deribit_net(g, entry_maker=str(mm).find("MAKER") >= 0)
            add(f"A03d_{mm}_{int(h)}m", g, "net_deribit")
    T = pd.DataFrame(rows)
    p = T[T.variant.eq("A03b_forward_1to1_taker")].set_index(["symbol", "period"])
    cells = [("BTCUSDT", "H1"), ("BTCUSDT", "H2"), ("ETHUSDT", "H1"), ("ETHUSDT", "H2")]
    checks = {f"{s}_{h}": (bool(p.loc[(s, h), "mean_net_bp"] > 0 and p.loc[(s, h), "net_lo95"] > 0)
                           if (s, h) in p.index else False) for s, h in cells}
    sym_hold = {s: (bool(p.loc[(s, "HOLDOUT"), "net_lo95"] > 0) if (s, "HOLDOUT") in p.index else False)
                for s in SYMBOLS}
    verdict = "PASS" if all(checks.values()) and all(sym_hold.values()) else "FAIL_CLOSE_LA"
    dec = {"version": VERSION, "primary": "A03b_forward_1to1_taker (Deribit fees)", "cells": checks,
           "holdout_by_symbol_lo95_positive": sym_hold, "verdict": verdict}
    return T, dec


def bridge_vs_binance(shadow: Path, repo: Path, tol_min: int = 5, tol_bp: float = 10.0) -> pd.DataFrame:
    """Jul-Sep 2026: how many of the original 452 Binance signals have a Deribit-native counterpart (same symbol and
    side, signal within +-tol_min, level within tol_bp), and do their A03b outcomes agree. Descriptive."""
    new = pd.read_csv(_base(shadow) / "03b" / "03b_1to1_events.csv", low_memory=False)
    old = pd.read_csv(Path(repo) / "notebooks" / "Liquidity_Absorption" / "03b" / "03b_1to1_events.csv", low_memory=False)
    for z in (new, old):
        z["t"] = pd.to_datetime(z.signal_time_utc, utc=True)
    new = new[(new.t >= INSAMPLE[0]) & (new.t < INSAMPLE[1])]
    rows = []
    for r in old.itertuples():
        c = new[(new.symbol == r.symbol) & (new.side == r.side) & ((new.t - r.t).abs() <= pd.Timedelta(minutes=tol_min))]
        c = c[(np.abs(c.level_price / r.level_price - 1) * 1e4) <= tol_bp]
        if len(c):
            k = c.iloc[int(np.argmin(np.abs((c.t - r.t).dt.total_seconds())))]
            rows.append({"a02_candidate_row": r.a02_candidate_row, "symbol": r.symbol, "matched": True,
                         "binance_reason": r.reason, "deribit_reason": k.reason,
                         "agree": r.reason == k.reason, "dt_min": (k.t - r.t).total_seconds() / 60,
                         "level_diff_bp": (k.level_price / r.level_price - 1) * 1e4})
        else:
            rows.append({"a02_candidate_row": r.a02_candidate_row, "symbol": r.symbol, "matched": False,
                         "binance_reason": r.reason})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- synthetic
def synthetic_history(root: Path, days: list[str], seed: int = 0) -> None:
    """Fake Deribit future-kind day files for BTC and ETH: random-walk perpetual prints with index prices."""
    rng = np.random.default_rng(seed)
    for cur, px0 in (("BTC", 60000.0), ("ETH", 3000.0)):
        px = px0
        for d in days:
            t0 = int(pd.Timestamp(d, tz="UTC").value // 10**6)
            n = 1440 * 6
            ts = np.sort(t0 + rng.integers(0, DAY_MS, n))
            p = px * np.exp(np.cumsum(rng.normal(0, 2.5e-4, n)))
            px = float(p[-1])
            df = pd.DataFrame({"timestamp": pd.to_datetime(ts, unit="ms", utc=True),
                               "instrument_name": f"{cur}-PERPETUAL", "trade_seq": np.arange(n) + 1,
                               "price": np.round(p, 1 if cur == "BTC" else 2), "amount": rng.uniform(10, 5000, n).round(0),
                               "direction": np.where(rng.random(n) < 0.5, "buy", "sell"), "index_price": p * 0.9999})
            f = Path(root) / "trades" / cur / "future" / d[:4]
            f.mkdir(parents=True, exist_ok=True)
            df.to_parquet(f / f"{d}.parquet", index=False)
