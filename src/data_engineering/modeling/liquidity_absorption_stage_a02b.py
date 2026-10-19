"""LA-A02b: reconstruct both second-touch encounters from Binance USD-M aggTrades.

Feature extraction ONLY; consumes A02's frozen-for-this-run candidate rows without
changing their timestamps, sides or prices. All trade intervals end <= signal
close. This module neither simulates entries nor examines subsequent returns.

Actual trade prices, rather than approximate logarithmic-bin centers, define
whether volume is near the event zone. A 1bp bin is used only to describe the
historical pre-first-touch point of control (POC); POC is NOT an entry/level.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_engineering.modeling.liquidity_absorption_stage_a import binance_root, parse_kline_open_time, repo_root

LA_A02B_VERSION = "LA-A02b-0.1.0"
MS_PER_MINUTE = 60_000
MS_PER_DAY = 86_400_000


@dataclass(frozen=True)
class Config:
    pre_first_profile_min: int = 30
    poc_log_bin_bp: float = 1.0
    batch_rows: int = 250_000
    expected_a02_version: str = "LA-A02-0.1.0"


def _ms(value: object) -> int:
    t = pd.Timestamp(value)
    if t.tzinfo is None:
        raise ValueError(f"Timezone-naive event timestamp: {value}")
    return int(t.tz_convert("UTC").value // 1_000_000)


def _utc_days(start_ms: int, end_exclusive_ms: int) -> set[str]:
    if start_ms >= end_exclusive_ms:
        return set()
    days = range(start_ms // MS_PER_DAY, (end_exclusive_ms - 1) // MS_PER_DAY + 1)
    return {pd.Timestamp(d * MS_PER_DAY, unit="ms", tz="UTC").strftime("%Y-%m-%d") for d in days}


@dataclass
class Accumulator:
    row: int
    symbol: str
    side: str
    first_ms: int
    second_ms: int
    level: float
    halfwidth: float
    config: Config
    first_start: int = field(init=False)
    base_start: int = field(init=False)
    bars: int = field(init=False)
    all_quote: np.ndarray = field(init=False)
    zone_quote: np.ndarray = field(init=False)
    zone_buy: np.ndarray = field(init=False)
    zone_sell: np.ndarray = field(init=False)
    active_zone_seconds: set[int] = field(default_factory=set)
    pre_first_quote: float = 0.0
    pre_first_zone_quote: float = 0.0
    profile: dict[int, float] = field(default_factory=dict)
    missing_touch_days: list[str] = field(default_factory=list)
    missing_pre_days: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.side not in {"long", "short"}:
            raise ValueError(f"Bad side: {self.side}")
        if not (np.isfinite(self.level) and np.isfinite(self.halfwidth) and self.level > self.halfwidth > 0):
            raise ValueError("Invalid zone/level")
        if self.first_ms >= self.second_ms or self.first_ms % MS_PER_MINUTE or self.second_ms % MS_PER_MINUTE:
            raise ValueError("Touch closes must be ordered and aligned to whole minutes")
        self.first_start = self.first_ms - MS_PER_MINUTE
        self.base_start = self.first_start - self.config.pre_first_profile_min * MS_PER_MINUTE
        self.bars = (self.second_ms - self.first_ms) // MS_PER_MINUTE + 1
        self.all_quote = np.zeros(self.bars, dtype="float64")
        self.zone_quote = np.zeros(self.bars, dtype="float64")
        self.zone_buy = np.zeros(self.bars, dtype="float64")
        self.zone_sell = np.zeros(self.bars, dtype="float64")

    def accept_batch(self, tm: np.ndarray, px: np.ndarray, qty: np.ndarray, buyer_maker: np.ndarray) -> None:
        """Consume already validated, sorted-or-unsorted trades; no future tape is used."""
        width = np.log1p(self.config.poc_log_bin_bp / 10_000)
        in_pre = (tm >= self.base_start) & (tm < self.first_start)
        if np.any(in_pre):
            pt, pq = px[in_pre], qty[in_pre]
            quote = pt * pq
            self.pre_first_quote += float(quote.sum())
            self.pre_first_zone_quote += float(quote[(pt >= self.level - self.halfwidth) & (pt <= self.level + self.halfwidth)].sum())
            unique, inverse = np.unique(np.floor(np.log(pt) / width).astype("int64"), return_inverse=True)
            summed = np.bincount(inverse, weights=quote)
            for k, value in zip(unique, summed):
                self.profile[int(k)] = self.profile.get(int(k), 0.0) + float(value)
        in_touch = (tm >= self.first_start) & (tm < self.second_ms)
        if not np.any(in_touch):
            return
        tt = tm[in_touch]
        prices, quantity, sides = px[in_touch], qty[in_touch], buyer_maker[in_touch]
        idx = ((tt - self.first_start) // MS_PER_MINUTE).astype("int64")
        q = prices * quantity
        self.all_quote += np.bincount(idx, weights=q, minlength=self.bars)
        near = (prices >= self.level - self.halfwidth) & (prices <= self.level + self.halfwidth)
        if np.any(near):
            at = idx[near]
            near_q = q[near]
            near_sides = sides[near]
            self.zone_quote += np.bincount(at, weights=near_q, minlength=self.bars)
            self.zone_buy += np.bincount(at, weights=np.where(near_sides, 0.0, near_q), minlength=self.bars)
            self.zone_sell += np.bincount(at, weights=np.where(near_sides, near_q, 0.0), minlength=self.bars)
            self.active_zone_seconds.update(np.unique(tt[near] // 1000).tolist())


def read_candidates(candidate_path: Path) -> pd.DataFrame:
    c = pd.read_csv(candidate_path)
    required = {"symbol", "side", "level_price", "zone_halfwidth", "first_touch_close", "second_touch_close", "signal_time_utc"}
    if not required.issubset(c.columns):
        raise ValueError(f"Missing A02 candidate columns: {sorted(required - set(c))}")
    for field_name in ("first_touch_close", "second_touch_close", "signal_time_utc"):
        c[field_name] = pd.to_datetime(c[field_name], utc=True, errors="raise")
    if not (c.second_touch_close == c.signal_time_utc).all():
        raise ValueError("A02 signal_time does not match second touch close")
    if c.duplicated(["symbol", "side", "signal_time_utc", "level_price"]).any():
        raise ValueError("Repeated A02 candidate identity")
    c.insert(0, "a02_candidate_row", np.arange(len(c), dtype="int64"))
    return c


def _load_minute_context(data_root: Path, symbol: str) -> pd.DataFrame:
    path = data_root / f"{symbol}_um_1m.parquet"
    fields = ["open_time", "open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"]
    df = pd.read_parquet(path, columns=fields)
    t = parse_kline_open_time(df.open_time)
    if t.isna().any() or t.duplicated().any():
        raise ValueError(f"Bad candle timestamps for {symbol}")
    df["close_ms"] = t.dt.as_unit("ms").astype("int64") + MS_PER_MINUTE
    df = df.set_index("close_ms", verify_integrity=True).sort_index()
    return df


def _context_features(acc: Accumulator, candles: pd.DataFrame) -> dict:
    times = np.arange(acc.first_ms, acc.second_ms + MS_PER_MINUTE, MS_PER_MINUTE, dtype="int64")
    selected = candles.reindex(times)
    if selected[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError(f"Missing 1m candle within event: {acc.symbol} row {acc.row}")
    low = selected.low.to_numpy(dtype="float64")
    high = selected.high.to_numpy(dtype="float64")
    near = (high >= acc.level - acc.halfwidth) & (low <= acc.level + acc.halfwidth)
    half = max(1, acc.bars // 2)
    early_count, late_count = int(near[:half].sum()), int(near[half:].sum())
    early_quote, late_quote = float(acc.zone_quote[:half].sum()), float(acc.zone_quote[half:].sum())
    early_rate = early_quote / early_count if early_count else np.nan
    late_rate = late_quote / late_count if late_count else np.nan
    c_first, c_last = selected.iloc[0], selected.iloc[-1]
    delta = 2 * selected.taker_buy_quote_volume.astype(float) - selected.quote_volume.astype(float)
    return {
        "intersecting_1m_candles": int(near.sum()), "early_intersecting_1m": early_count,
        "late_intersecting_1m": late_count, "zone_quote_early": early_quote, "zone_quote_late": late_quote,
        "zone_quote_per_intersecting_1m": float(acc.zone_quote.sum()) / int(near.sum()) if near.any() else np.nan,
        "early_zone_quote_per_intersecting_1m": early_rate,
        "late_zone_quote_per_intersecting_1m": late_rate,
        "late_early_zone_rate_ratio": late_rate / early_rate if early_count and late_count and early_rate > 0 else np.nan,
        "delta_between_touch_candles": float(delta.sum()),
        "delta_first_touch_1m": float(delta.iloc[0]), "delta_second_touch_1m": float(delta.iloc[-1]),
        "delta_touch_change_1m": float(delta.iloc[-1] - delta.iloc[0]),
        "first_touch_open_close_bp": float((c_first.close - c_first.open) / c_first.open * 1e4),
        "second_touch_open_close_bp": float((c_last.close - c_last.open) / c_last.open * 1e4),
        "first_to_second_close_bp": float((c_last.close - c_first.close) / c_first.close * 1e4),
        "first_touch_kline_quote": float(c_first.quote_volume),
        "second_touch_kline_quote": float(c_last.quote_volume),
    }


def finalize_features(acc: Accumulator, candles: pd.DataFrame) -> dict:
    result = {"a02_candidate_row": acc.row, "symbol": acc.symbol,
              "touch_status": "MISSING_TRADE_DAY" if acc.missing_touch_days else "PRESENT",
              "missing_touch_days": ",".join(acc.missing_touch_days),
              "pre_first_profile_status": "MISSING_TRADE_DAY" if acc.missing_pre_days else ("PRESENT" if acc.pre_first_quote > 0 else "NO_TRADES"),
              "missing_profile_days": ",".join(acc.missing_pre_days),
              "minutes_between_touches": acc.bars - 1,
              "first_zone_quote": np.nan, "second_zone_quote": np.nan,
              "first_zone_buy_quote": np.nan, "second_zone_buy_quote": np.nan,
              "first_zone_sell_quote": np.nan, "second_zone_sell_quote": np.nan,
              "first_zone_signed_quote": np.nan, "second_zone_signed_quote": np.nan,
              "zone_quote_touch_change": np.nan, "zone_quote_touch_ratio": np.nan,
              "first_zone_share_of_minute_quote": np.nan, "second_zone_share_of_minute_quote": np.nan,
              "zone_share_touch_change": np.nan,
              "first_opposing_zone_share": np.nan, "second_opposing_zone_share": np.nan,
              "opposing_zone_share_change": np.nan,
              "zone_quote_between_touches": np.nan,
              "zone_active_print_seconds_between_touches": np.nan,
              "zone_quote_per_active_print_second": np.nan,
              "first_touch_trade_kline_quote_error": np.nan,
              "second_touch_trade_kline_quote_error": np.nan,
              "pre_first_profile_quote": np.nan,
              "pre_first_profile_zone_share": np.nan,
              "pre_first_poc_price": np.nan,
              "pre_first_poc_distance_bp": np.nan,
              "pre_first_poc_bin_share": np.nan,
              }
    result.update(_context_features(acc, candles))
    if acc.missing_touch_days:
        # Candle-derived visitation/delta remain valid, but no partial historical
        # trade-volume estimate may appear as complete touch/zone evidence.
        for key in ("zone_quote_early", "zone_quote_late", "zone_quote_per_intersecting_1m",
                    "early_zone_quote_per_intersecting_1m", "late_zone_quote_per_intersecting_1m",
                    "late_early_zone_rate_ratio"):
            result[key] = np.nan
    if not acc.missing_touch_days:
        a, b = 0, acc.bars - 1
        first, second = float(acc.zone_quote[a]), float(acc.zone_quote[b])
        first_buy, second_buy = float(acc.zone_buy[a]), float(acc.zone_buy[b])
        first_sell, second_sell = float(acc.zone_sell[a]), float(acc.zone_sell[b])
        first_share = first / acc.all_quote[a] if acc.all_quote[a] > 0 else np.nan
        second_share = second / acc.all_quote[b] if acc.all_quote[b] > 0 else np.nan
        opposing_first = first_sell if acc.side == "long" else first_buy
        opposing_second = second_sell if acc.side == "long" else second_buy
        opposing_share_first = opposing_first / first if first else np.nan
        opposing_share_second = opposing_second / second if second else np.nan
        secs = len(acc.active_zone_seconds)
        result.update({
            "first_zone_quote": first, "second_zone_quote": second,
            "first_zone_buy_quote": first_buy, "second_zone_buy_quote": second_buy,
            "first_zone_sell_quote": first_sell, "second_zone_sell_quote": second_sell,
            "first_zone_signed_quote": first_buy - first_sell,
            "second_zone_signed_quote": second_buy - second_sell,
            "zone_quote_touch_change": second - first,
            "zone_quote_touch_ratio": second / first if first > 0 else np.nan,
            "first_zone_share_of_minute_quote": first_share,
            "second_zone_share_of_minute_quote": second_share,
            "zone_share_touch_change": second_share - first_share,
            "first_opposing_zone_share": opposing_share_first,
            "second_opposing_zone_share": opposing_share_second,
            "opposing_zone_share_change": opposing_share_second - opposing_share_first,
            "zone_quote_between_touches": float(acc.zone_quote.sum()),
            "zone_active_print_seconds_between_touches": secs,
            "zone_quote_per_active_print_second": float(acc.zone_quote.sum()) / secs if secs else np.nan,
            "first_touch_trade_kline_quote_error": abs(float(acc.all_quote[a]) - result["first_touch_kline_quote"]) / max(result["first_touch_kline_quote"], 1.0),
            "second_touch_trade_kline_quote_error": abs(float(acc.all_quote[b]) - result["second_touch_kline_quote"]) / max(result["second_touch_kline_quote"], 1.0),
        })
    if not acc.missing_pre_days and acc.pre_first_quote > 0 and acc.profile:
        width = np.log1p(acc.config.poc_log_bin_bp / 10_000)
        poc_bin = min(acc.profile, key=lambda k: (-acc.profile[k], k))
        poc_px = float(np.exp((poc_bin + 0.5) * width))
        result.update({"pre_first_profile_quote": acc.pre_first_quote,
                       "pre_first_profile_zone_share": acc.pre_first_zone_quote / acc.pre_first_quote,
                       "pre_first_poc_price": poc_px,
                       "pre_first_poc_distance_bp": (poc_px / acc.level - 1) * 1e4,
                       "pre_first_poc_bin_share": acc.profile[poc_bin] / acc.pre_first_quote})
    return result


def enrich_a02_candidates(candidates: pd.DataFrame, data_root: Path, cfg: Config = Config()) -> pd.DataFrame:
    """Read-only trade scans. Every event uses only prints available through signal close."""
    if candidates.empty:
        return candidates.copy()
    required = {"a02_candidate_row", "symbol", "side", "first_touch_close", "second_touch_close", "level_price", "zone_halfwidth"}
    if missing := required - set(candidates):
        raise ValueError(f"Missing candidate input columns: {sorted(missing)}")
    accs = [Accumulator(row=int(r.a02_candidate_row), symbol=str(r.symbol), side=str(r.side),
                        first_ms=_ms(r.first_touch_close), second_ms=_ms(r.second_touch_close),
                        level=float(r.level_price), halfwidth=float(r.zone_halfwidth), config=cfg)
            for r in candidates.itertuples(index=False)]
    candles = {s: _load_minute_context(data_root, s) for s in sorted(candidates.symbol.unique())}
    days: dict[tuple[str, str], list[Accumulator]] = defaultdict(list)
    for a in accs:
        for day in _utc_days(a.base_start, a.second_ms):
            days[(a.symbol, day)].append(a)
            path = data_root / "aggTrades" / a.symbol / f"{a.symbol}-aggTrades-{day}.parquet"
            if not path.is_file():
                if day in _utc_days(a.first_start, a.second_ms):
                    a.missing_touch_days.append(day)
                if day in _utc_days(a.base_start, a.first_start):
                    a.missing_pre_days.append(day)
    import pyarrow.parquet as pq
    for (symbol, day), day_events in sorted(days.items()):
        path = data_root / "aggTrades" / symbol / f"{symbol}-aggTrades-{day}.parquet"
        if not path.exists():
            continue
        pf = pq.ParquetFile(path)
        needed = ["time_ms", "price", "qty", "is_buyer_maker"]
        if not set(needed).issubset(set(pf.schema.names)):
            raise ValueError(f"Missing columns in {path}")
        earliest = min(a.base_start for a in day_events)
        latest = max(a.second_ms for a in day_events)
        for batch in pf.iter_batches(batch_size=cfg.batch_rows, columns=needed):
            v = batch.to_pandas()
            tm = pd.to_numeric(v.time_ms, errors="coerce").to_numpy(dtype="float64")
            px = pd.to_numeric(v.price, errors="coerce").to_numpy(dtype="float64")
            qty = pd.to_numeric(v.qty, errors="coerce").to_numpy(dtype="float64")
            side = v.is_buyer_maker
            mask = np.isfinite(tm) & np.isfinite(px) & np.isfinite(qty) & (px > 0) & (qty > 0) & side.notna().to_numpy()
            if not mask.all():
                raise ValueError(f"Invalid or missing trade row in {path}")
            if not pd.api.types.is_bool_dtype(side.dtype):
                raise ValueError(f"Aggressor side is not bool in {path}")
            tm = tm.astype("int64")
            if tm.max(initial=-1) < earliest or tm.min(initial=2**63 - 1) >= latest:
                continue
            s = side.to_numpy(dtype="bool")
            for a in day_events:
                if not ((tm >= a.base_start) & (tm < a.second_ms)).any():
                    continue
                a.accept_batch(tm, px, qty, s)
    features = pd.DataFrame.from_records([finalize_features(a, candles[a.symbol]) for a in accs])
    if features.a02_candidate_row.duplicated().any():
        raise AssertionError("Feature row identity repeated")
    enriched = candidates.merge(features.drop(columns="symbol"), on="a02_candidate_row", how="left", validate="one_to_one", sort=False)
    if len(enriched) != len(candidates):
        raise AssertionError("Candidate counts changed during enrichment")
    return enriched


def run_a02b(data_root: Path, repo: Path, output: Path, cfg: Config = Config()) -> dict:
    a02_dir = repo / "notebooks" / "Liquidity_Absorption" / "02"
    with (a02_dir / "02_decision.json").open(encoding="utf-8") as fh:
        predecessor = json.load(fh)
    if predecessor.get("version") != cfg.expected_a02_version or predecessor.get("status") != "FEATURE_EXTRACTION_ONLY":
        raise RuntimeError("Expected original LA-A02-0.1.0 candidate extraction; do not silently change event source")
    candidates = read_candidates(a02_dir / "02_price_only_candidates.csv")
    output.mkdir(parents=True, exist_ok=True)
    features = enrich_a02_candidates(candidates, data_root, cfg)
    out_csv = output / "02b_touch_features.csv"
    features.to_csv(out_csv, index=False)
    coverage = []
    for symbol, sub in features.groupby("symbol"):
        coverage.append({"symbol": symbol, "events": int(len(sub)),
                         "touch_flow_present": int((sub.touch_status == "PRESENT").sum()),
                         "missing_touch_day_events": int((sub.touch_status == "MISSING_TRADE_DAY").sum()),
                         "pre_first_profile_present": int((sub.pre_first_profile_status == "PRESENT").sum()),
                         "pre_first_profile_missing_day": int((sub.pre_first_profile_status == "MISSING_TRADE_DAY").sum()),
                         "zero_first_zone_quote": int((sub.first_zone_quote == 0).sum()),
                         "zero_second_zone_quote": int((sub.second_zone_quote == 0).sum()),
                         "undefined_touch_volume_ratio": int(sub.zone_quote_touch_ratio.isna().sum()),
                         "undefined_late_early_dwell_ratio": int(sub.late_early_zone_rate_ratio.isna().sum()),
                         "median_second_touch_reconciliation_error": float(sub.second_touch_trade_kline_quote_error.median()),
                         "p95_second_touch_reconciliation_error": float(sub.second_touch_trade_kline_quote_error.quantile(.95)),
                         })
    pd.DataFrame(coverage).to_csv(output / "02b_coverage.csv", index=False)
    (output / "02b_feature_contract.json").write_text(json.dumps({"version": LA_A02B_VERSION,
        "parent_version": cfg.expected_a02_version, "config": asdict(cfg),
        "event_selection": "Original A02 price-only candidate rows; unchanged",
        "touch_intervals": "[first_touch_close-1m, first_touch_close) and [second_touch_close-1m, second_touch_close)",
        "between_touch_interval": "[first_touch_close-1m, second_touch_close), no trades at or after signal close",
        "pre_first_profile_interval": "[first_touch_close-1m-pre_first_profile_min, first_touch_close-1m)",
        "zone_inclusion": "actual print price within [level-zone_halfwidth, level+zone_halfwidth], inclusive",
        "dwell_definition": "1m OHLC intersects zone; proxy for visiting zone, NOT elapsed seconds physically at level",
        "active_second_definition": "distinct seconds with at least one trade print inside zone; NOT passive order exposure",
        "profile_node": "POC from pre-first trades only, 1bp log-price bins; descriptive, not candidate-level selection",
        "growth_undefined": "ratio NaN when first-touch zone quote is zero; do not impute",
        "status": "EXPLORATORY_FEATURE_ONLY_NO_PNL"}, indent=2) + "\n", encoding="utf-8")
    decision = {"version": LA_A02B_VERSION, "status": "FEATURE_EXTRACTION_ONLY", "candidate_count": int(len(features)),
                "source_a02_candidate_count": int(len(candidates)), "coverage": coverage,
                "ready_for_a03_design": bool(len(features) == len(candidates) and len(features) > 0 and (features.touch_status == "PRESENT").all()),
                "caution": "Feature audit only; no absorption edge, returns, trade simulation, holdout, or final entry/stop/target specification."}
    (output / "02b_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    return decision


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binance-root", default=None)
    p.add_argument("--repo", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    repo = a.repo or repo_root()
    root = binance_root(repo, a.binance_root)
    out = a.out or repo / "notebooks" / "Liquidity_Absorption" / "02b"
    print(json.dumps(run_a02b(root, repo, out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
