"""LA-A03c: full tape replay of reversed 1:1 and third-touch post-only maker proxies.

EXPLORATORY, never a proof of executable passive fills. Preserves the A03b
first print, frozen A03 structural target and A02/A02b 452 signal universe.
Historical Binance aggTrades reveal aggressor side and print price, not queue.
No capital accounting, funding, bid/ask or realized spread is available.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "LA-A03c-0.1.1"
MIN_MS = 60_000
DAY_MS = 86_400_000
NOTIONAL_USDT = 10_000.0
MODES = ("REVERSE_TAKER_1TO1", "MAKER_TOUCH_3RD", "MAKER_THROUGH_3RD")
CLOSED = {"STOP", "TARGET", "TIMEOUT"}


@dataclass(frozen=True)
class Config:
    limit_order_ttl_min: int = 90
    hold_minutes: int = 240
    placement_latency_ms: int = 1000
    cancel_on_pre_entry_target_or_stop: bool = True
    through_ticks: int = 1
    tick_btc: float = 0.1  # project A03 assumptions, not independently validated exchange tick
    tick_eth: float = 0.01
    maker_fee_bp: float = 2.0  # indicative regular-user USDT-M, NOT verified user's fee tier
    taker_fee_bp: float = 6.0  # preserved conservative A03 taker assumption
    maker_entry_adverse_bp: float = 0.0  # assumed LIMIT price, NOT proof of fill
    taker_entry_adverse_bp: float = 0.5
    stop_exit_adverse_bp: float = 1.5
    target_exit_adverse_bp: float = 0.5
    timeout_exit_adverse_bp: float = 0.5
    batch_rows: int = 250_000


def utc_ms(value) -> int:
    t = pd.Timestamp(value)
    if t.tzinfo is None:
        raise ValueError("Event timestamps must be timezone-aware")
    return int(t.tz_convert("UTC").value // 1_000_000)


def utc_day(ms: int) -> str:
    return pd.Timestamp(ms, unit="ms", tz="UTC").strftime("%Y-%m-%d")


def tick_for(symbol: str, cfg: Config) -> float:
    if symbol == "BTCUSDT":
        return cfg.tick_btc
    if symbol == "ETHUSDT":
        return cfg.tick_eth
    raise ValueError(f"Unsupported symbol {symbol}")


def is_present(r: dict, key: str) -> bool:
    val = r.get(key)
    return val is not None and str(val) != "" and pd.notna(val)


@dataclass
class TapeStudy:
    """A single counterfactual order; chronological print feed via shared daily scanner."""
    source: dict
    mode: str
    cfg: Config
    symbol: str = field(init=False)
    side: str = field(init=False)
    d: int = field(init=False)
    signal: int = field(init=False)
    live_from: int = field(init=False)
    order_expiry: int = field(init=False)
    deadline: int = field(init=False)
    level: float = field(init=False)
    entry_px: float = field(default=np.nan)
    entry_time: int | None = field(default=None)
    exit_time: int | None = field(default=None)
    exit_print: float = field(default=np.nan)
    exit_reference: float = field(default=np.nan)
    exit_fill: float = field(default=np.nan)
    target: float = field(default=np.nan)
    stop: float = field(default=np.nan)
    reason: str = field(default="")
    observed_touch: bool = field(default=False)
    side_eligible_print: bool = field(default=False)
    third_touch_time: int | None = field(default=None)
    qualifying_print: float = field(default=np.nan)
    last_seen: int | None = field(default=None)
    missing_days: list[str] = field(default_factory=list)
    max_favorable: float = field(default=np.nan)
    max_adverse: float = field(default=np.nan)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError("Unknown A03c variant")
        self.symbol = str(self.source["symbol"])
        original_side = str(self.source["side"])
        if original_side not in ("long", "short"):
            raise ValueError("Unknown source direction")
        self.side = ({"long": "short", "short": "long"}[original_side]
                     if self.mode == "REVERSE_TAKER_1TO1" else original_side)
        self.d = 1 if self.side == "long" else -1
        self.level = float(self.source["level_price"])
        self.signal = utc_ms(self.source["signal_time_utc"])
        self.live_from = self.signal if self.mode == "REVERSE_TAKER_1TO1" else self.signal + self.cfg.placement_latency_ms
        self.order_expiry = (self.signal + 60_000 if self.mode == "REVERSE_TAKER_1TO1"
                             else self.signal + self.cfg.limit_order_ttl_min * MIN_MS)
        self.deadline = (self.signal + self.cfg.hold_minutes * MIN_MS if self.mode == "REVERSE_TAKER_1TO1"
                         else self.order_expiry + self.cfg.hold_minutes * MIN_MS)
        # Frozen A03/A03b structural target is known before third-touch placement.
        if not is_present(self.source, "target"):
            self.reason = "NO_STRUCTURAL_TARGET"
            return
        original_target = float(self.source["target"])
        if self.mode == "REVERSE_TAKER_1TO1":
            # Exact reversed two price barriers, same historical first entry print.
            self.stop = original_target
            self.target = float(self.source["initial_stop"])
            if not is_present(self.source, "entry_print") or not is_present(self.source, "entry_time_ms"):
                self.reason = "NO_SOURCE_ENTRY"
                return
            e = float(self.source["entry_print"])
            if not np.isclose(self.d * (self.target - e), self.d * (e - self.stop), rtol=0, atol=1e-6):
                raise AssertionError("A03b did not provide identical 1:1 reverse barriers")
            if self.d * (self.target - e) <= 0:
                raise ValueError("Reversed target does not have positive distance")
        else:
            close = float(self.source["bar_close"])
            orig_d = 1 if original_side == "long" else -1
            if not orig_d * (close - self.level) > 0:
                self.reason = "NOT_POST_ONLY_AT_SIGNAL_CLOSE_PROXY"
                return
            if not orig_d * (original_target - self.level) > 0:
                self.reason = "INVALID_MAKER_TARGET_SIDE"
                return
            # Retain the original target and put risk at equal distance from the NEW limit entry.
            self.target = original_target
            self.stop = 2 * self.level - original_target
            if self.stop <= 0:
                self.reason = "NONPOSITIVE_MIRRORED_STOP"
                return
        if any(not np.isfinite(v) for v in (self.target, self.stop)):
            raise ValueError("Nonfinite barriers")

    def all_closed(self) -> bool:
        return bool(self.reason)

    def _entry(self, tm: int, entry_price: float, print_price: float) -> None:
        risk = self.d * (entry_price - self.stop)
        target_dist = self.d * (self.target - entry_price)
        if not (risk > 0 and target_dist > 0 and np.isclose(risk, target_dist, atol=1e-7, rtol=1e-10)):
            raise AssertionError("Invalid positive symmetric 1:1 geometry")
        self.entry_time = tm
        self.entry_px = entry_price
        self.qualifying_print = print_price
        self.max_favorable = entry_price
        self.max_adverse = entry_price

    def _maker_fill_allowed(self, px: float, is_buyer_maker: bool) -> bool:
        # Sell aggressor hits passive bid on long; buy aggressor lifts passive ask on short.
        correct_side = is_buyer_maker if self.side == "long" else not is_buyer_maker
        price_touches = self.d * (px - self.level) <= 0
        if price_touches:
            self.observed_touch = True
            if self.third_touch_time is None:
                self.third_touch_time = self.last_seen
        if price_touches and correct_side:
            self.side_eligible_print = True
        if not (price_touches and correct_side):
            return False
        if self.mode == "MAKER_TOUCH_3RD":
            return True
        # Trade-through provides MORE evidence of crossing the resting limit, but
        # still cannot prove our queue position or whether POST_ONLY was accepted.
        dist = self.cfg.through_ticks * tick_for(self.symbol, self.cfg)
        return self.d * (px - self.level) <= -dist + 1e-9

    def feed(self, tm: int, px: float, is_buyer_maker: bool) -> None:
        if self.reason or tm < self.signal:
            return
        if self.last_seen is not None and tm < self.last_seen:
            raise ValueError("Out-of-order historical print")
        self.last_seen = tm
        if not (np.isfinite(px) and px > 0):
            raise ValueError("Invalid traded price")
        if self.entry_time is None:
            if tm < self.live_from:
                return
            if self.mode == "REVERSE_TAKER_1TO1":
                original_ms = int(float(self.source["entry_time_ms"]))
                if tm < original_ms:
                    return
                if tm > self.order_expiry:
                    self.reason = "ENTRY_TIME_CHANGED"
                    return
                first_px = float(self.source["entry_print"])
                if tm != original_ms or not np.isclose(px, first_px, atol=1e-7, rtol=1e-10):
                    raise AssertionError("Reverse entry print differs from baseline A03b tape")
                self._entry(tm, px, px)
            else:
                if tm >= self.order_expiry:
                    self.reason = ("NO_THIRD_TOUCH" if not self.observed_touch else
                                   "NO_OPPOSITE_AGGRESSOR_PRINT" if not self.side_eligible_print else
                                   "NO_TRADE_THROUGH" if self.mode == "MAKER_THROUGH_3RD" else "NO_MAKER_FILL")
                    return
                # The setup becomes stale if its destination was achieved while
                # our limit remains unfilled. A move beyond the proposed stop
                # without an eligible print likewise invalidates queue inference.
                if self.cfg.cancel_on_pre_entry_target_or_stop:
                    if self.d * (px - self.target) >= 0:
                        self.reason = "TARGET_REACHED_BEFORE_MAKER_FILL"
                        return
                    if self.d * (px - self.stop) <= 0:
                        self.reason = "STOP_BREACHED_BEFORE_MAKER_FILL"
                        return
                if self._maker_fill_allowed(px, is_buyer_maker):
                    self._entry(tm, self.level, px)
            return  # Entry print cannot also trigger an exit.
        if tm >= self.entry_time + self.cfg.hold_minutes * MIN_MS:
            self._exit(tm, px, "TIMEOUT", px)
            return
        self.max_favorable = max(self.max_favorable, px) if self.d == 1 else min(self.max_favorable, px)
        self.max_adverse = min(self.max_adverse, px) if self.d == 1 else max(self.max_adverse, px)
        if self.d * (px - self.stop) <= 0:
            ref = min(px, self.stop) if self.d == 1 else max(px, self.stop)
            self._exit(tm, px, "STOP", ref)
        elif self.d * (px - self.target) >= 0:
            self._exit(tm, px, "TARGET", self.target)

    def _exit(self, tm: int, print_price: float, reason: str, reference: float):
        self.exit_time, self.exit_print, self.reason = tm, print_price, reason
        self.exit_reference = reference
        adverse = (self.cfg.stop_exit_adverse_bp if reason == "STOP" else
                   self.cfg.target_exit_adverse_bp if reason == "TARGET" else
                   self.cfg.timeout_exit_adverse_bp)
        self.exit_fill = reference * (1 - self.d * adverse / 10000)

    def finish(self, available_through: dict[str, int]):
        # An intervening absent daily tape invalidates even a later recorded fill
        # or exit: stop/target/qualifying maker prints could have happened in gap.
        # Do not silently promote an incomplete path into a closed outcome.
        if self.reason in {"NO_STRUCTURAL_TARGET", "NOT_POST_ONLY_AT_SIGNAL_CLOSE_PROXY",
                           "INVALID_MAKER_TARGET_SIDE", "NONPOSITIVE_MIRRORED_STOP", "NO_SOURCE_ENTRY"}:
            return
        for day in set(self.missing_days):
            gap_start = utc_ms(pd.Timestamp(day, tz="UTC"))
            gap_end = gap_start + DAY_MS
            active_from = (self.live_from if self.mode != "REVERSE_TAKER_1TO1" else
                           self.entry_time if self.entry_time is not None else self.live_from)
            active_until = (self.exit_time if self.exit_time is not None else
                            self.entry_time + self.cfg.hold_minutes * MIN_MS if self.entry_time is not None
                            else self.order_expiry)
            if gap_start < active_until and gap_end > active_from:
                self.reason = ("CENSORED_TAPE_GAP" if self.entry_time is not None else "CENSORED_ORDER_WINDOW")
                self.exit_time = None
                self.exit_print = self.exit_reference = self.exit_fill = np.nan
                return
        if self.reason:
            return
        if self.entry_time is None:
            end = self.order_expiry
            if self.last_seen is not None and self.last_seen >= end:
                self.reason = ("NO_THIRD_TOUCH" if not self.observed_touch else
                               "NO_OPPOSITE_AGGRESSOR_PRINT" if not self.side_eligible_print else
                               "NO_TRADE_THROUGH" if self.mode == "MAKER_THROUGH_3RD" else "NO_MAKER_FILL")
            else:
                self.reason = "CENSORED_ORDER_WINDOW"
        else:
            self.reason = "CENSORED_TAPE_END"

    def record(self) -> dict:
        entry_fee = (self.cfg.taker_fee_bp if self.mode == "REVERSE_TAKER_1TO1"
                     else self.cfg.maker_fee_bp)
        entry_adverse = (self.cfg.taker_entry_adverse_bp if self.mode == "REVERSE_TAKER_1TO1"
                         else self.cfg.maker_entry_adverse_bp)
        entry_fill = (self.entry_px * (1 + self.d * entry_adverse / 10000)
                      if self.entry_time is not None else np.nan)
        exit_fee = self.cfg.taker_fee_bp
        risk = self.d * (self.entry_px - self.stop) if self.entry_time is not None else np.nan
        valid = self.reason in CLOSED and self.exit_time is not None
        gross = self.d * (self.exit_reference - self.entry_px) if valid else np.nan
        fee = (entry_fee * entry_fill + exit_fee * self.exit_fill) / 10000 if valid else np.nan
        net = self.d * (self.exit_fill - entry_fill) - fee if valid else np.nan
        return {
            "a02_candidate_row": int(self.source["a02_candidate_row"]),
            "symbol": self.symbol, "source_side": self.source["side"], "side": self.side,
            "signal_time_utc": self.source["signal_time_utc"], "mode": self.mode,
            "reason": self.reason, "level_price": self.level, "limit_entry_price": self.level if self.mode != "REVERSE_TAKER_1TO1" else np.nan,
            "target": self.target, "stop": self.stop, "is_inferred_maker": self.mode != "REVERSE_TAKER_1TO1",
            "order_live_from_ms": self.live_from, "order_expiry_ms": self.order_expiry,
            "third_touch_time_ms": self.third_touch_time, "observed_zone_touch": self.observed_touch,
            "opposite_aggressor_seen": self.side_eligible_print,
            "entry_time_ms": self.entry_time, "entry_print": self.qualifying_print,
            "entry_price": self.entry_px, "entry_fill_assumed": entry_fill,
            "exit_time_ms": self.exit_time, "exit_print": self.exit_print,
            "exit_reference": self.exit_reference, "exit_fill_assumed": self.exit_fill,
            "initial_risk_bp": risk / self.entry_px * 10000 if self.entry_time is not None else np.nan,
            "gross_r": gross / risk if valid else np.nan,
            "net_r_assumed_costs": net / risk if valid else np.nan,
            "gross_return_bp": gross / self.entry_px * 10000 if valid else np.nan,
            "net_return_bp_assumed_costs": net / self.entry_px * 10000 if valid else np.nan,
            "entry_fee_bp": entry_fee, "exit_fee_bp": exit_fee,
            "missing_trade_days_in_horizon": ";".join(sorted(set(self.missing_days))),
            "absorption_like_exploratory": self.source.get("absorption_like_exploratory", False),
            "flow_integrity_flag": self.source.get("flow_integrity_flag", "UNKNOWN"),
        }


def scan_tapes(root: Path, sims: list[TapeStudy], batch_rows: int = 250_000) -> dict:
    """Read each symbol/day once, stream Parquet batches, dispatch in tape order."""
    import pyarrow.parquet as pq
    jobs: dict[tuple[str, str], list[TapeStudy]] = defaultdict(list)
    for e in sims:
        if e.all_closed():
            continue
        for i in range(e.signal // DAY_MS, e.deadline // DAY_MS + 1):
            jobs[e.symbol, utc_day(i * DAY_MS)].append(e)
    missing, files = set(), 0
    seen_last: dict[str, int] = {}
    for (sym, day), group in sorted(jobs.items()):
        fp = root / "aggTrades" / sym / f"{sym}-aggTrades-{day}.parquet"
        if not fp.is_file():
            missing.add((sym, day))
            for e in group:
                e.missing_days.append(day)
            continue
        files += 1
        pf = pq.ParquetFile(fp)
        keys = ["time_ms", "price", "is_buyer_maker"]
        if not set(keys).issubset(set(pf.schema.names)):
            raise ValueError(f"Missing trade fields in {fp}")
        earliest = min(e.signal for e in group)
        latest = max(e.deadline for e in group) + MIN_MS
        prev_t = None
        for batch in pf.iter_batches(batch_size=batch_rows, columns=keys):
            d = batch.to_pandas()
            if d[keys].isna().any().any() or not pd.api.types.is_bool_dtype(d.is_buyer_maker.dtype):
                raise ValueError(f"Missing/invalid aggressor side or print in {fp}")
            t = pd.to_numeric(d.time_ms, errors="raise").to_numpy(dtype="int64")
            p = pd.to_numeric(d.price, errors="raise").to_numpy(dtype="float64")
            side = d.is_buyer_maker.to_numpy(dtype="bool")
            if not np.isfinite(p).all() or (p <= 0).any() or (len(t) > 1 and np.any(np.diff(t) < 0)) or (prev_t is not None and len(t) and t[0] < prev_t):
                raise ValueError(f"Invalid/out-of-order tape {fp}")
            if len(t):
                prev_t = int(t[-1])
                seen_last[sym] = max(seen_last.get(sym, -1), prev_t)
            if not len(t) or t[-1] < earliest or t[0] >= latest:
                continue
            for e in group:
                if e.all_closed():
                    continue
                lo = np.searchsorted(t, e.signal, side="left")
                hi = np.searchsorted(t, e.deadline + MIN_MS, side="left")
                for j in range(lo, hi):
                    if e.all_closed():
                        break
                    e.feed(int(t[j]), float(p[j]), bool(side[j]))
    for e in sims:
        e.finish(seen_last)
    return {"trade_daily_files_read": files, "missing_daily_files": [f"{s}:{d}" for s, d in sorted(missing)]}


def validate_inputs(repo: Path) -> pd.DataFrame:
    a03b = repo / "notebooks/Liquidity_Absorption/03b"
    meta = json.loads((a03b / "03b_decision.json").read_text(encoding="utf-8"))
    if meta.get("version") != "LA-A03b-0.1.0" or meta.get("source_events") != 452:
        raise ValueError("Unexpected/changed A03b source")
    src = pd.read_csv(a03b / "03b_1to1_events.csv")
    if len(src) != 452 or not src.a02_candidate_row.is_unique:
        raise ValueError("A03b source ID/count changed")
    if not src.loc[src.target.notna(), "target_rr_initial"].dropna().between(1 - 1e-8, 1 + 1e-8).all():
        raise ValueError("A03b symmetric target contract changed")
    if not src.loc[src.reason == "NO_STRUCTURAL_TARGET", "target"].isna().all():
        raise ValueError("Bad A03b target-state contract")
    return src


def _summaries(trades: pd.DataFrame, cfg: Config) -> list[dict]:
    out = []
    for (sym, mode), a in trades.groupby(["symbol", "mode"]):
        closed = a[a.reason.isin(CLOSED)]
        unresolved = a.reason.str.startswith("CENSORED")
        out.append({
            "symbol": sym, "mode": mode, "signals": len(a), "entered": int(a.entry_time_ms.notna().sum()),
            "closed": len(closed), "unresolved": int(unresolved.sum()),
            "no_structural_target": int((a.reason == "NO_STRUCTURAL_TARGET").sum()),
            "unfilled_or_ineligible": int((~a.reason.isin(CLOSED) & ~unresolved & (a.reason != "NO_STRUCTURAL_TARGET")).sum()),
            "stop": int((a.reason == "STOP").sum()), "target": int((a.reason == "TARGET").sum()),
            "timeout": int((a.reason == "TIMEOUT").sum()),
            "mean_net_bp_per_closed": float(closed.net_return_bp_assumed_costs.mean()) if len(closed) else None,
            "mean_net_r_per_closed": float(closed.net_r_assumed_costs.mean()) if len(closed) else None,
            "mean_net_bp_per_signal_zero_if_no_trade_PARTIAL": float(closed.net_return_bp_assumed_costs.sum() / len(a)) if len(a) else None,
            "net_positive_closed_rate": float((closed.net_return_bp_assumed_costs > 0).mean()) if len(closed) else None,
            "unresolved_prevents_final_ev": bool(unresolved.any()),
        })
    return out


def equity(trades: pd.DataFrame, notional: float = NOTIONAL_USDT) -> pd.DataFrame:
    if notional <= 0:
        raise ValueError("notional must be positive")
    a = trades[trades.reason.isin(CLOSED)].copy()
    if a.empty:
        return pd.DataFrame()
    results = []
    for universe in ("BTCUSDT", "ETHUSDT", "COMBINED"):
        for mode in MODES:
            x = a.loc[(a["mode"] == mode) & ((a.symbol == universe) if universe != "COMBINED" else True)].copy()
            if x.empty:
                continue
            x = x.sort_values(["exit_time_ms", "a02_candidate_row"]).reset_index(drop=True)
            x["time_utc"] = pd.to_datetime(x.exit_time_ms, unit="ms", utc=True)
            x["cumulative_net_usdt"] = (x.net_return_bp_assumed_costs / 10000 * notional).cumsum()
            x["cumulative_gross_usdt"] = (x.gross_return_bp / 10000 * notional).cumsum()
            x["cumulative_net_r"] = x.net_r_assumed_costs.cumsum()
            x["universe"] = universe
            x["position_number"] = np.arange(1, len(x) + 1)
            results.append(x[["a02_candidate_row", "time_utc", "mode", "universe", "position_number", "cumulative_net_usdt", "cumulative_gross_usdt", "cumulative_net_r"]])
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def plot_curves(frame: pd.DataFrame, out: Path):
    if frame.empty:
        return
    import matplotlib.pyplot as plt
    for metric, filename in [("cumulative_net_usdt", "03c_net_equity.png"), ("cumulative_gross_usdt", "03c_gross_equity.png")]:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        for ax, universe in zip(axes, ("BTCUSDT", "ETHUSDT", "COMBINED")):
            for mode in MODES:
                x = frame[(frame.universe == universe) & (frame["mode"] == mode)]
                if len(x):
                    ax.step(x.time_utc, x[metric], where="post", label=mode)
            ax.set_ylabel(f"{universe}\nUSDT")
            ax.axhline(0, linewidth=.8)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
        fig.suptitle(f"A03c historical {metric}: $10,000 notional/trade (not an investable equity curve)\nMaker entries are unverified fills; funding, capital and queue omitted")
        axes[-1].set_xlabel("UTC exit time")
        fig.tight_layout(rect=[0, 0, 1, .91])
        fig.savefig(out / filename, dpi=140, bbox_inches="tight")
        plt.close(fig)


def _fee_sensitivity(trades: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Reprice FEES only, without changing signals/print crossing/fills; not full economic sensitivity."""
    rows = []
    for maker_fee, taker_fee in ((2., 6.), (2., 5.), (0., 5.), (0., 4.), (4., 6.)):
        for mode in MODES:
            a = trades[(trades["mode"] == mode) & trades.reason.isin(CLOSED)]
            if not len(a):
                continue
            # Original stored fee rates are subtracted back from P&L in basis points.
            current_fee = a.entry_fee_bp * a.entry_fill_assumed / a.entry_price + a.exit_fee_bp * a.exit_fill_assumed / a.entry_price
            e_new_fee = taker_fee if mode == "REVERSE_TAKER_1TO1" else maker_fee
            new_fee = e_new_fee * a.entry_fill_assumed / a.entry_price + taker_fee * a.exit_fill_assumed / a.entry_price
            repriced_net = a.net_return_bp_assumed_costs + current_fee - new_fee
            rows.append({"mode": mode, "maker_fee_bp_assumed": maker_fee, "taker_fee_bp_assumed": taker_fee,
                         "closed": len(a), "mean_net_bp_per_closed": float(repriced_net.mean()),
                         "mean_net_bp_per_452_signals_PARTIAL": float(repriced_net.sum() / 452),
                         "funding": "OMITTED", "fills": "UNCHANGED_UNVERIFIED_MAKER_QUEUE"})
    return pd.DataFrame(rows)


def run(repo: Path, data_root: Path, out: Path, cfg: Config = Config()) -> dict:
    if cfg.limit_order_ttl_min < 1 or cfg.hold_minutes < 1 or cfg.through_ticks < 1 or cfg.placement_latency_ms < 0 or cfg.batch_rows < 1:
        raise ValueError("Invalid time, tick or batch config")
    if min(cfg.tick_btc, cfg.tick_eth) <= 0 or min(cfg.maker_fee_bp, cfg.taker_fee_bp) < 0:
        raise ValueError("Invalid fees/tick values")
    src = validate_inputs(repo)
    sims = [TapeStudy(dict(row), mode, cfg) for row in src.to_dict("records") for mode in MODES]
    tape = scan_tapes(data_root, sims, cfg.batch_rows)
    trade = pd.DataFrame.from_records([sim.record() for sim in sims])
    if len(trade) != 452 * 3 or not trade.groupby("a02_candidate_row").size().eq(3).all():
        raise AssertionError("Unexpected candidate universe drift")
    out.mkdir(parents=True, exist_ok=True)
    trade.to_csv(out / "03c_events.csv", index=False)
    summary = pd.DataFrame(_summaries(trade, cfg))
    summary.to_csv(out / "03c_coverage.csv", index=False)
    sen = _fee_sensitivity(trade, cfg)
    sen.to_csv(out / "03c_fee_sensitivity.csv", index=False)
    curves = equity(trade)
    curves.to_csv(out / "03c_equity_paths.csv", index=False)
    plot_curves(curves, out)
    contract = {
        "version": VERSION, "input": "A03b 452 candidates, first post-signal entries, and frozen A03-selected targets",
        "config": asdict(cfg),
        "reversed": "Original A03b first entry print; REVERSE side; swapped original A03b stop and TP. Independent aggTrade replay, 1:1 before costs.",
        "passive": "Third revisit to ORIGINAL A02 liquidity level after second bar close, order placed +latency, expires 90m, original A03 target frozen, SL mirrored from LIMIT PRICE; maker exit is TAKer.",
        "maker_touch": "Buyer rests bid / seller rests ask. Requires OPPOSING-aggressor prints at/through limit, NO QUEUE PROOF; optimistic eligibility.",
        "maker_through": "Requires same opposing-aggressor trade >=1 assumed tick through order limit, MORE conservative eligibility, but STILL NO QUEUE PROOF.",
        "price_at_placement": "Second 1m close only is the post-only nonmarketable proxy; actual historical BBO unknown.",
        "pre_entry_cancellation": "Cancel a maker order (no assumed fill) if preselected target OR mirrored stop is crossed first while waiting, using prints observed after placement.",
        "fees": "2bp assumed maker entry, 6bp assumed taker exit (A03 continuity). Sensitivities reprice fees only; funding omitted.",
        "equity": "Exit-ordered $10k notional per trade, overlapping hypothetical positions permitted, not capital equity. No filled => zero realized P&L but denominator 452; censored cause incomplete EV.",
        "status": "EXPLORATORY_NO_REAL_LIMIT_ORDER_FILL_OR_CONFIRMED_EDGE",
    }
    (out / "03c_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    decision = {"version": VERSION, "status": contract["status"], "candidates": 452,
                "policy_rows": len(trade), "coverage": summary.to_dict("records"), "tape": tape,
                "caution": "AggTrades cannot reconstruct resting queue or post-only order acceptance; maker results are optimistic/strict proxies, NOT actual fill likelihood. No prospective data, funding, bid/ask or portfolio risk."}
    (out / "03c_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    return decision


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path)
    ap.add_argument("--binance-root", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    from src.data_engineering.modeling.liquidity_absorption_stage_a import repo_root, binance_root
    repo = args.repo or repo_root()
    root = args.binance_root or binance_root(repo)
    dest = args.out or repo / "notebooks" / "Liquidity_Absorption" / "03c"
    print(json.dumps(run(repo, root, dest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
