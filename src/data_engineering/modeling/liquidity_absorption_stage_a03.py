"""LA-A03 v0.1: exploratory, causal, tape-driven execution research.

All position/event decisions use only available 1m closed candles and trade prints
at or after the A02 signal timestamp. Existing A02/A02b events are unchanged.
Historical print prices are NOT order-book executable bid/ask quotes.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_engineering.modeling.liquidity_absorption_stage_a import (
    binance_root, parse_kline_open_time, repo_root,
)

VERSION = "LA-A03-0.1.0"
MIN_MS = 60_000
DAY_MS = 86_400_000
MODES = ("FIXED", "BREAKEVEN", "TRAIL_1M")


@dataclass(frozen=True)
class Config:
    # Hypotheses/implementation proxies, NOT an agreed MNQ->crypto translation.
    stop_buffer_zone_halfwidths: float = 1.0
    break_even_initial_r: float = 1.0
    trailing_buffer_ticks: int = 2
    assumed_tick_btc: float = 0.1
    assumed_tick_eth: float = 0.01
    hold_minutes: int = 240
    max_entry_wait_seconds: int = 60
    # Explicit structural target priority (no LVNs until a causal LVN contract exists).
    target_priority: str = "vwap_if_displaced_else_opposing_prior_day"
    vwap_displacement_initial_r: float = 2.0
    # Cost assumptions for an aggressive hypothetical round trip, in bp per side.
    taker_fee_bp_per_side: float = 6.0
    entry_adverse_bp: float = 0.5
    target_exit_adverse_bp: float = 0.5
    stop_exit_adverse_bp: float = 1.5
    timeout_exit_adverse_bp: float = 0.5
    # QC flag, established from A02b observations prior to any post-entry PnL.
    flow_reconcile_flag_above_fraction: float = 0.01


@dataclass
class ModeState:
    mode: str
    stop: float = np.nan
    entered: bool = False
    closed: bool = False
    reason: str = ""
    entry_ms: int | None = None
    exit_ms: int | None = None
    entry_print: float = np.nan
    exit_print: float = np.nan
    exit_reference_px: float = np.nan
    entry_fill: float = np.nan
    exit_fill: float = np.nan
    initial_risk_price: float = np.nan
    trigger: float = np.nan
    be_enabled: bool = False
    be_at_ms: int | None = None
    stop_updates: int = 0
    trail_updates: int = 0
    last_ref_high: float = np.nan
    last_ref_low: float = np.nan
    high_candidate_price: float = np.nan
    high_candidate_index: int | None = None
    low_candidate_price: float = np.nan
    low_candidate_index: int | None = None
    max_favorable_px: float = 0.0
    max_adverse_px: float = 0.0


def millis(x: object) -> int:
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        raise ValueError(f"Naive timestamp {x}")
    return int(t.tz_convert("UTC").value // 1_000_000)


def utc_day(ms: int) -> str:
    return pd.Timestamp(ms, unit="ms", tz="UTC").strftime("%Y-%m-%d")


def tick_for(sym: str, cfg: Config) -> float:
    if sym == "BTCUSDT":
        return cfg.assumed_tick_btc
    if sym == "ETHUSDT":
        return cfg.assumed_tick_eth
    raise ValueError(f"Unsupported tick-size assumption for {sym}")


def round_stop_adverse(x: float, side: str, tick: float) -> float:
    """Long stops round down; short stops round up, never tighter via rounding."""
    units = x / tick
    return float((np.floor(units + 1e-9) if side == "long" else np.ceil(units - 1e-9)) * tick)


def select_target(side: str, entry: float, risk: float, prior_high: float,
                  prior_low: float, vwap: float, cfg: Config) -> tuple[float, str]:
    """Pre-entry completed-day references and signal-close VWAP only; no future path."""
    d = 1 if side == "long" else -1
    if cfg.target_priority != "vwap_if_displaced_else_opposing_prior_day":
        raise ValueError("Unregistered target priority")
    if np.isfinite(vwap) and d * (vwap - entry) >= cfg.vwap_displacement_initial_r * risk:
        return float(vwap), "SESSION_VWAP"
    other = prior_high if side == "long" else prior_low
    if np.isfinite(other) and d * (other - entry) > 0:
        return float(other), "OPPOSING_PRIOR_DAY"
    return np.nan, "NO_STRUCTURAL_TARGET"


def prepare_bars(data_root: Path, symbol: str) -> pd.DataFrame:
    path = data_root / f"{symbol}_um_1m.parquet"
    df = pd.read_parquet(path, columns=["open_time", "open", "high", "low", "close", "quote_volume", "volume"])
    t = parse_kline_open_time(df.open_time)
    if t.isna().any() or t.duplicated().any():
        raise ValueError(f"Bad 1m timestamps: {symbol}")
    df["open_ms"] = t.dt.as_unit("ms").astype("int64")
    df["close_ms"] = df.open_ms + MIN_MS
    df["utc_day"] = t.dt.strftime("%Y-%m-%d")
    daily = df.groupby("utc_day").agg(high=("high", "max"), low=("low", "min"), n=("open", "size"))
    daily.loc[daily.n != 1440, ["high", "low"]] = np.nan
    daily["prior_high"] = daily.high.shift(1)
    daily["prior_low"] = daily.low.shift(1)
    df = df.join(daily[["prior_high", "prior_low"]], on="utc_day")
    return df.set_index("close_ms", drop=False, verify_integrity=True).sort_index()


class EventSimulation:
    """Three counterfactual exit policies, same ex-ante candidate and first print.

    Feed observed prints in nondecreasing (timestamp, archival) order; before each
    print incorporate only 1m bars whose CLOSE timestamp is <= the print time.
    """
    def __init__(self, event: dict, bars: pd.DataFrame, cfg: Config):
        self.event = dict(event)
        self.cfg = cfg
        self.symbol = str(event["symbol"])
        self.side = str(event["side"])
        if self.side not in ("long", "short"):
            raise ValueError("Invalid position side")
        self.d = 1 if self.side == "long" else -1
        self.tick = tick_for(self.symbol, cfg)
        self.signal = millis(event["signal_time_utc"])
        self.deadline = self.signal + cfg.hold_minutes * MIN_MS
        self.entry_limit = self.signal + cfg.max_entry_wait_seconds * 1000
        self.level = float(event["level_price"])
        self.width = float(event["zone_halfwidth"])
        self.initial_stop = round_stop_adverse(self.level - self.d * self.width * cfg.stop_buffer_zone_halfwidths,
                                                self.side, self.tick)
        self.vwap = float(event.get("vwap_utc_session", np.nan))
        signal_bar = bars.loc[self.signal] if self.signal in bars.index else None
        if signal_bar is None:
            raise ValueError(f"Signal candle missing {self.symbol} {event['signal_time_utc']}")
        self.prior_high, self.prior_low = float(signal_bar.prior_high), float(signal_bar.prior_low)
        self.signal_low, self.signal_high = float(signal_bar.low), float(signal_bar.high)
        self.bar_frame = bars.loc[(bars.index > self.signal) & (bars.index <= self.deadline)]
        self.bar_records = [dict(close_ms=int(t), high=float(r.high), low=float(r.low))
                            for t, r in self.bar_frame.iterrows()]
        self.bar_pointer = 0
        self.closed_bars: list[dict] = []
        self.states = {mode: ModeState(mode, stop=self.initial_stop,
                                      last_ref_high=self.signal_high, last_ref_low=self.signal_low)
                       for mode in MODES}
        self.entry_print = np.nan
        self.entry_ms: int | None = None
        self.target = np.nan
        self.target_source = "NOT_ENTERED"
        self.risk = np.nan
        self.entry_skip_reason = ""
        self.last_t: int | None = None

    def _mark_closed(self, mode: ModeState, reason: str, tm: int | None = None, print_px: float = np.nan) -> None:
        mode.closed = True
        mode.reason = reason
        if tm is not None:
            mode.exit_ms = tm
            mode.exit_print = print_px
            if reason == "STOP":
                conservative_px = min(print_px, mode.stop) if self.d == 1 else max(print_px, mode.stop)
                adverse = self.cfg.stop_exit_adverse_bp
            elif reason == "TARGET":
                conservative_px = mode.trigger  # no better-than-target fills claimed
                adverse = self.cfg.target_exit_adverse_bp
            else:
                conservative_px = print_px
                adverse = self.cfg.timeout_exit_adverse_bp
            mode.exit_reference_px = conservative_px
            mode.exit_fill = conservative_px * (1 - self.d * adverse / 10000)

    def _start(self, tm: int, px: float) -> None:
        if self.entry_ms is not None or self.entry_skip_reason:
            return
        if tm > self.entry_limit:
            self.entry_skip_reason = "NO_PROMPT_ENTRY_PRINT"
            for state in self.states.values():
                self._mark_closed(state, self.entry_skip_reason)
            return
        risk = self.d * (px - self.initial_stop)
        if risk <= 0 or not np.isfinite(risk):
            self.entry_skip_reason = "INVALID_ENTRY_STOP_GEOMETRY"
            for state in self.states.values():
                self._mark_closed(state, self.entry_skip_reason)
            return
        target, source = select_target(self.side, px, risk, self.prior_high, self.prior_low, self.vwap, self.cfg)
        if not np.isfinite(target) or self.d * (target - px) <= 0:
            self.entry_skip_reason = "NO_STRUCTURAL_TARGET"
            for state in self.states.values():
                self._mark_closed(state, self.entry_skip_reason)
            return
        self.entry_ms, self.entry_print, self.risk = tm, float(px), float(risk)
        self.target, self.target_source = target, source
        for state in self.states.values():
            state.entered = True
            state.entry_ms = tm
            state.entry_print = px
            state.entry_fill = px * (1 + self.d * self.cfg.entry_adverse_bp / 10000)
            state.initial_risk_price = risk
            state.trigger = target
            state.max_favorable_px = px
            state.max_adverse_px = px

    def _on_completed_bar(self, bar: dict) -> None:
        """Close-time updates NEVER inspect the next or an unclosed candle."""
        if self.entry_ms is None or bar["close_ms"] <= self.entry_ms:
            return
        self.closed_bars.append(bar)
        state = self.states["TRAIL_1M"]
        if state.closed or len(self.closed_bars) < 3:
            return
        previous, center, next_bar = self.closed_bars[-3:]
        center_index = len(self.closed_bars) - 2
        if self.d == 1:
            high_pivot = center["high"] > previous["high"] and center["high"] >= next_bar["high"]
            low_pivot = center["low"] < previous["low"] and center["low"] <= next_bar["low"]
            if high_pivot and center["high"] > state.last_ref_high:
                state.high_candidate_price = center["high"]
                state.high_candidate_index = center_index
            if (low_pivot and state.high_candidate_index is not None
                    and center_index > state.high_candidate_index and center["low"] > state.last_ref_low):
                proposal = round_stop_adverse(center["low"] - self.cfg.trailing_buffer_ticks * self.tick,
                                               self.side, self.tick)
                if state.be_enabled and proposal > state.stop:
                    state.stop = proposal
                    state.stop_updates += 1
                    state.trail_updates += 1
                    state.last_ref_high = state.high_candidate_price
                    state.last_ref_low = center["low"]
                    state.high_candidate_index = None
        else:
            low_pivot = center["low"] < previous["low"] and center["low"] <= next_bar["low"]
            high_pivot = center["high"] > previous["high"] and center["high"] >= next_bar["high"]
            if low_pivot and center["low"] < state.last_ref_low:
                state.low_candidate_price = center["low"]
                state.low_candidate_index = center_index
            if (high_pivot and state.low_candidate_index is not None
                    and center_index > state.low_candidate_index and center["high"] < state.last_ref_high):
                proposal = round_stop_adverse(center["high"] + self.cfg.trailing_buffer_ticks * self.tick,
                                               self.side, self.tick)
                if state.be_enabled and proposal < state.stop:
                    state.stop = proposal
                    state.stop_updates += 1
                    state.trail_updates += 1
                    state.last_ref_low = state.low_candidate_price
                    state.last_ref_high = center["high"]
                    state.low_candidate_index = None

    def feed(self, tm: int, px: float) -> None:
        if self.last_t is not None and tm < self.last_t:
            raise ValueError("Tape not timestamp ordered for event")
        self.last_t = tm
        if tm < self.signal or (self.all_closed() and self.entry_skip_reason):
            return
        if self.entry_ms is None:
            self._start(tm, px)
            if self.entry_ms is None:
                return
            return  # first print defines entry; never exit at same print
        while self.bar_pointer < len(self.bar_records) and self.bar_records[self.bar_pointer]["close_ms"] <= tm:
            self._on_completed_bar(self.bar_records[self.bar_pointer])
            self.bar_pointer += 1
        for state in self.states.values():
            if state.closed:
                continue
            state.max_favorable_px = max(state.max_favorable_px, px) if self.d == 1 else min(state.max_favorable_px, px)
            state.max_adverse_px = min(state.max_adverse_px, px) if self.d == 1 else max(state.max_adverse_px, px)
            if tm >= self.deadline:
                self._mark_closed(state, "TIMEOUT", tm, px)
                continue
            if state.mode != "FIXED" and not state.be_enabled and self.d * (px - self.entry_print) >= self.cfg.break_even_initial_r * self.risk:
                state.be_enabled = True
                state.be_at_ms = tm
                candidate = self.entry_print
                if self.d * (candidate - state.stop) > 0:
                    state.stop = candidate
                    state.stop_updates += 1
            if self.d * (px - state.stop) <= 0:
                self._mark_closed(state, "STOP", tm, px)
            elif self.d * (px - self.target) >= 0:
                self._mark_closed(state, "TARGET", tm, px)

    def all_closed(self) -> bool:
        return all(m.closed for m in self.states.values())

    def records(self) -> list[dict]:
        recon_values = [float(self.event.get(name, np.nan)) for name in (
            "first_touch_trade_kline_quote_error", "second_touch_trade_kline_quote_error")]
        finite_recon = [x for x in recon_values if np.isfinite(x)]
        flag = max(finite_recon) if finite_recon else np.nan
        suspect = np.isfinite(flag) and flag > self.cfg.flow_reconcile_flag_above_fraction
        # Mechanically defined prior to looking at A03 returns; already explored
        # descriptively in A02b and therefore NOT a confirmatory entry filter.
        abs_like = (float(self.event.get("zone_quote_touch_ratio", np.nan)) > 1
                    and float(self.event.get("opposing_zone_share_change", np.nan)) > 0
                    and float(self.event.get("zone_share_touch_change", np.nan)) > 0)
        records = []
        for mode, state in self.states.items():
            terminal = state.reason if state.closed else (
                "CENSORED_TAPE_END" if state.entered else "NO_ENTRY_TAPE")
            out = {
                "a02_candidate_row": int(self.event["a02_candidate_row"]),
                "symbol": self.symbol, "side": self.side,
                "signal_time_utc": self.event["signal_time_utc"],
                "mode": mode, "reason": terminal,
                "level_type": self.event.get("level_type", ""),
                "level_price": self.level, "zone_halfwidth": self.width,
                "signal_close": self.signal, "entry_time_ms": state.entry_ms,
                "exit_time_ms": state.exit_ms,
                "entry_print": state.entry_print, "entry_fill_assumed": state.entry_fill,
                "initial_stop": self.initial_stop, "target": self.target,
                "target_source": self.target_source,
                "target_rr_initial": self.d * (self.target - self.entry_print) / self.risk if self.entry_ms is not None else np.nan,
                "initial_risk_bps": self.risk / self.entry_print * 10000 if self.entry_ms is not None else np.nan,
                "exit_print": state.exit_print, "exit_reference_px": state.exit_reference_px,
                "exit_fill_assumed": state.exit_fill,
                "final_stop": state.stop, "stop_updates": state.stop_updates,
                "trail_updates": state.trail_updates, "break_even_triggered": state.be_enabled,
                "first_touch_quote_reconciliation_error": self.event.get("first_touch_trade_kline_quote_error", np.nan),
                "second_touch_quote_reconciliation_error": self.event.get("second_touch_trade_kline_quote_error", np.nan),
                "flow_integrity_flag": "RECONCILIATION_OUTLIER" if suspect else ("UNASSESSED" if not finite_recon else "OK"),
                "absorption_like_exploratory": bool(abs_like),
                "zone_quote_touch_ratio": self.event.get("zone_quote_touch_ratio", np.nan),
                "opposing_zone_share_change": self.event.get("opposing_zone_share_change", np.nan),
                "zone_share_touch_change": self.event.get("zone_share_touch_change", np.nan),
                "mfe_print_initial_r": self.d * (state.max_favorable_px - self.entry_print) / self.risk if self.entry_ms is not None else np.nan,
                "mae_print_initial_r": self.d * (state.max_adverse_px - self.entry_print) / self.risk if self.entry_ms is not None else np.nan,
                "gross_r": np.nan, "net_r_assumed_costs": np.nan,
                "gross_return_bp": np.nan, "net_return_bp_assumed_costs": np.nan,
                "entry_fee_bp": self.cfg.taker_fee_bp_per_side, "exit_fee_bp": self.cfg.taker_fee_bp_per_side,
            }
            if state.entered and state.closed and state.exit_ms is not None:
                pnl_gross = self.d * (state.exit_reference_px - self.entry_print)
                costed_price_pnl = self.d * (state.exit_fill - state.entry_fill)
                fees_px = self.cfg.taker_fee_bp_per_side / 10000 * (state.entry_fill + state.exit_fill)
                out["gross_r"] = pnl_gross / self.risk
                out["net_r_assumed_costs"] = (costed_price_pnl - fees_px) / self.risk
                out["gross_return_bp"] = pnl_gross / self.entry_print * 10000
                out["net_return_bp_assumed_costs"] = (costed_price_pnl - fees_px) / self.entry_print * 10000
            records.append(out)
        return records


def load_candidates(repo: Path) -> pd.DataFrame:
    base = repo / "notebooks" / "Liquidity_Absorption"
    for subdir, file, version in (("02", "02_decision.json", "LA-A02-0.1.0"),
                                  ("02b", "02b_decision.json", "LA-A02b-0.1.0")):
        meta = json.loads((base / subdir / file).read_text(encoding="utf-8"))
        if meta.get("version") != version:
            raise ValueError(f"Unexpected input version {subdir}: {meta.get('version')}")
    a = pd.read_csv(base / "02b" / "02b_touch_features.csv")
    flow = pd.read_csv(base / "02" / "02_causal_flow_features.csv")
    if len(flow) != len(a) or len(a) != 452 or not a.a02_candidate_row.is_unique:
        raise ValueError("A02/A02b candidate identity/count mismatch; do not silently change source universe")
    for i, v in a.iterrows():
        q = flow.iloc[int(v.a02_candidate_row)]
        if (v.symbol != q.symbol or v.side != q.side or
            millis(v.signal_time_utc) != millis(q.signal_time_utc) or
            abs(float(v.level_price) - float(q.level_price)) > 1e-7):
            raise ValueError(f"A02/A02b event identity drift at {i}")
    a["vwap_utc_session"] = flow.iloc[a.a02_candidate_row.to_numpy(int)]["vwap_utc_session"].to_numpy()
    return a


def scan_event_tapes(data_root: Path, sims: list[EventSimulation], batch_rows: int = 250_000) -> dict:
    """Read each needed per-symbol per-day file once, in bounded batches."""
    import pyarrow.parquet as pq
    jobs: dict[tuple[str, str], list[EventSimulation]] = defaultdict(list)
    for sim in sims:
        days = range(sim.signal // DAY_MS, sim.deadline // DAY_MS + 1)
        for day_idx in days:
            jobs[(sim.symbol, utc_day(day_idx * DAY_MS))].append(sim)
    total_files = 0
    missing_files = []
    for (sym, day), group in sorted(jobs.items()):
        path = data_root / "aggTrades" / sym / f"{sym}-aggTrades-{day}.parquet"
        if not path.is_file():
            missing_files.append(f"{sym}:{day}")
            continue
        total_files += 1
        pf = pq.ParquetFile(path)
        if not {"time_ms", "price"}.issubset(pf.schema.names):
            raise ValueError(f"Missing print timestamps/prices in {path}")
        earliest = min(e.signal for e in group)
        latest = max(e.deadline + MIN_MS for e in group)
        prev = None
        for batch in pf.iter_batches(columns=["time_ms", "price"], batch_size=batch_rows):
            chunk = batch.to_pandas()
            t = pd.to_numeric(chunk.time_ms, errors="coerce").to_numpy(dtype="float64")
            px = pd.to_numeric(chunk.price, errors="coerce").to_numpy(dtype="float64")
            if not np.isfinite(t).all() or not np.isfinite(px).all() or (px <= 0).any():
                raise ValueError(f"Invalid execution tape {path}")
            t = t.astype("int64")
            if (len(t) and (np.any(np.diff(t) < 0) or (prev is not None and t[0] < prev))):
                raise ValueError(f"Out-of-order aggregate trade tape {path}")
            if len(t):
                prev = int(t[-1])
            if not len(t) or t[-1] < earliest or t[0] >= latest:
                continue
            for e in group:
                if e.all_closed():
                    continue
                lo = np.searchsorted(t, e.signal, side="left")
                hi = np.searchsorted(t, e.deadline + MIN_MS, side="left")
                for i in range(lo, hi):
                    if e.all_closed():
                        break
                    e.feed(int(t[i]), float(px[i]))
    return {"read_daily_parquet_files": total_files, "missing_trade_day_files": missing_files}


def run_a03(repo: Path, data_root: Path, out: Path, cfg: Config = Config()) -> dict:
    if cfg.hold_minutes < 1 or cfg.stop_buffer_zone_halfwidths <= 0 or cfg.break_even_initial_r <= 0 or cfg.taker_fee_bp_per_side < 0:
        raise ValueError("Invalid config")
    out.mkdir(parents=True, exist_ok=True)
    a = load_candidates(repo)
    sims = []
    for sym, sub in a.groupby("symbol"):
        candles = prepare_bars(data_root, sym)
        for event in sub.to_dict("records"):
            sims.append(EventSimulation(event, candles, cfg))
    tape = scan_event_tapes(data_root, sims)
    results = pd.DataFrame([record for sim in sims for record in sim.records()])
    if len(results) != len(a) * len(MODES):
        raise AssertionError("Expected exactly three policies per unchanged candidate")
    results.to_csv(out / "03_execution_events.csv", index=False)
    coverage = []
    for (sym, mode), sub in results.groupby(["symbol", "mode"]):
        valid = sub.net_r_assumed_costs.notna()
        coverage.append({"symbol": sym, "mode": mode, "events": len(sub),
                         "closed_trades": int(valid.sum()), "no_entry_or_no_target": int(sub.reason.isin(["NO_PROMPT_ENTRY_PRINT", "INVALID_ENTRY_STOP_GEOMETRY", "NO_STRUCTURAL_TARGET"]).sum()),
                         "censored": int((sub.reason == "CENSORED_TAPE_END").sum()),
                         "no_entry_tape": int((sub.reason == "NO_ENTRY_TAPE").sum()),
                         "stop_exits": int((sub.reason == "STOP").sum()),
                         "target_exits": int((sub.reason == "TARGET").sum()),
                         "timeouts": int((sub.reason == "TIMEOUT").sum()),
                         "flow_quality_flagged": int((sub.flow_integrity_flag == "RECONCILIATION_OUTLIER").sum()),
                         "mean_net_r_descriptive": float(sub.loc[valid, "net_r_assumed_costs"].mean()) if valid.any() else None,
                         "median_net_r_descriptive": float(sub.loc[valid, "net_r_assumed_costs"].median()) if valid.any() else None,
                         "positive_net_r_share_descriptive": float((sub.loc[valid, "net_r_assumed_costs"] > 0).mean()) if valid.any() else None,
                         "mean_gross_r_descriptive": float(sub.loc[valid, "gross_r"].mean()) if valid.any() else None})
    pd.DataFrame(coverage).to_csv(out / "03_execution_coverage.csv", index=False)
    contract = {"version": VERSION, "config": asdict(cfg), "input_candidates": len(a),
                "target_LVN": "NOT_IMPLEMENTED: requires prior-time causal LVN definition; exclude from claims",
                "entry_rule": "first aggregated trade print at/after second 1m close, within 60 seconds; assumed entry adverse cost",
                "initial_stop": "level plus/minus 1 zone halfwidth (2bp level width in A02), rounded outward to assumed tick",
                "break_even": "upon intratrade favorable move >= 1 times initial risk; stop to unadjusted entry print",
                "trail_1m": "after BE: 1-left/1-right confirmed 1m swing high then higher low (long), or lower low then lower high (short); 2 assumed ticks beyond low/high; never loosen; only after confirming bar CLOSE",
                "targets": "VWAP if displacement >= 2 initial R in favorable direction, otherwise opposing fully completed UTC prior-day extreme; no LVN",
                "tape_model": "first crossing print, no executable bid/ask or liquidity/queue model; timestamp ties keep stable parquet order",
                "costs": "per-side fee plus entry/exit adverse bps; target fill capped at level; stop not improved over stop price",
                "flow_quality": "A02b >1% first or second 1m quote-volume discrepancy flagged; not outcome-selected exclusions",
                "flow_group": "Exploratory A02b-descriptive intersection: second/first zone volume >1, opposing zone share increases, near-zone share of candle increases; NOT an eligible confirmatory classifier",
                "funding": "NOT MODELED; net_r_assumed_costs represents transaction execution costs only, before funding",
                "statistical_status": "EXPLORATORY_ONLY_NO_EDGE_OR_PROSPECTIVE_CONFIRMATION"}
    (out / "03_execution_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    report = {"version": VERSION, "status": "EXPLORATORY_EXECUTION_SIMULATION_NOT_CONFIRMED_STRATEGY",
              "source_events": len(a), "simulated_policy_rows": len(results), "policy_modes": list(MODES),
              "coverage": coverage, "tape": tape,
              "caution": "Historical prints do not prove executable fills; crypto-normalized stop/BE/targets are research proxies, not user-approved MNQ equivalents. Funding is omitted; net PnL is after ASSUMED EXECUTION COSTS ONLY. LVN targets excluded. No prospective validation."}
    (out / "03_decision.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path)
    p.add_argument("--binance-root")
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    root = a.repo or repo_root()
    data = binance_root(root, a.binance_root)
    out = a.out or root / "notebooks" / "Liquidity_Absorption" / "03"
    print(json.dumps(run_a03(root, data, out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
