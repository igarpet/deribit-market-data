"""LA-A02: causally timestamped second-touch candidate events and flow features.

No forward returns, trade exits, scores, or fitted thresholds. Read-only Binance data.
The 1bp volume-at-price aggregation approximates zone volume and is explicitly
labeled as such; raw trade prices remain the authoritative input.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_engineering.modeling.liquidity_absorption_stage_a import (
    binance_root, parse_kline_open_time, repo_root,
)

LA_A02_VERSION = "LA-A02-0.1.0"
MINUTE_MS = 60_000


@dataclass(frozen=True)
class Config:
    start: str = "2026-07-01"
    end: str = "2026-09-30"  # inclusive UTC days; A01 verified trade coverage
    warmup_days: int = 3
    pivot_left_5m: int = 2
    pivot_right_5m: int = 2
    pivot_max_age_min: int = 720
    zone_halfwidth_bp: float = 2.0  # exploration-design placeholder, not MNQ tick tolerance
    wick_fraction_min: float = 0.35
    max_penetration_zone_widths: float = 2.0
    leave_zone_multiple: float = 3.0
    min_touch_separation_min: int = 3
    max_retest_gap_min: int = 90
    cooldown_after_event_min: int = 30
    flow_5m: int = 5
    flow_15m: int = 15
    profile_window_min: int = 5
    profile_bin_bp: float = 1.0  # storage/diagnostic approximation only


def get_minutes(kline: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Parquet must contain full USD-M perp OHLCV and taker-buy-quote fields."""
    required = {"open_time", "open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote_volume"}
    missing = required - set(kline)
    if missing:
        raise ValueError(f"Missing perp kline columns: {sorted(missing)}")
    t = parse_kline_open_time(kline["open_time"])
    if t.isna().any():
        raise ValueError("Null or invalid minute open_time")
    keep = (t >= pd.Timestamp(cfg.start, tz="UTC") - pd.Timedelta(days=cfg.warmup_days)) & (t < pd.Timestamp(cfg.end, tz="UTC") + pd.Timedelta(days=1))
    x = kline.loc[keep, list(required - {"open_time"})].copy()
    x["open_time"] = t.loc[keep]
    x = x.sort_values("open_time").reset_index(drop=True)
    if x.empty:
        return x
    if x.open_time.duplicated().any() or (x.open_time.diff().dropna() != pd.Timedelta(minutes=1)).any():
        raise ValueError("One-minute input is not a contiguous unique 1m series")
    for name in required - {"open_time"}:
        x[name] = pd.to_numeric(x[name], errors="coerce")
    if x[list(required - {"open_time"})].isna().any().any():
        raise ValueError("Null numeric candle fields")
    if (x[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Nonpositive OHLC")
    x["close_time"] = x["open_time"] + pd.Timedelta(minutes=1)
    x["signed_quote_1m"] = 2 * x["taker_buy_quote_volume"] - x["quote_volume"]
    for n in (cfg.flow_5m, cfg.flow_15m):
        x[f"delta_{n}m"] = x.signed_quote_1m.rolling(n, min_periods=n).sum()
        x[f"quote_{n}m"] = x.quote_volume.rolling(n, min_periods=n).sum()
    # The observed session anchor is UTC midnight, not the 08:00 Deribit option clock.
    x["cvd_utc_session"] = x.groupby(x.open_time.dt.floor("D"))["signed_quote_1m"].cumsum()
    x["vwap_utc_session"] = x.quote_volume.groupby(x.open_time.dt.floor("D")).cumsum() / x.volume.groupby(x.open_time.dt.floor("D")).cumsum().replace(0, np.nan)
    return x


def get_reference_levels(x: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Previous *completed* UTC day H/L and last confirmed 5m swing pivot.

    Candidate pivots are detected offline using right bars, but only published
    after those right bars are *complete* (availability timestamp). This is not
    backdated to the pivot itself.
    """
    z = x[["open_time", "close_time", "open", "high", "low", "close"]].copy()
    z["utc_day"] = z.open_time.dt.floor("D")
    daily = z.groupby("utc_day", sort=True).agg(day_high=("high", "max"), day_low=("low", "min"))
    # Discard incomplete boundary days, because they cannot be used as prior-day levels.
    counts = z.groupby("utc_day").size()
    daily.loc[counts != 1440, ["day_high", "day_low"]] = np.nan
    daily[["prior_day_high", "prior_day_low"]] = daily[["day_high", "day_low"]].shift(1).to_numpy()
    z = z.join(daily[["prior_day_high", "prior_day_low"]], on="utc_day")

    five = z.groupby(z.open_time.dt.floor("5min"), sort=True).agg(
        high=("high", "max"), low=("low", "min"), count=("close", "count"))
    five = five.loc[five["count"] == 5].copy()
    five["bar_close"] = five.index + pd.Timedelta(minutes=5)
    for side, field, op in (("support", "low", "lt"), ("resistance", "high", "gt")):
        vals = five[field]
        good = pd.Series(True, index=five.index)
        for offset in list(range(-cfg.pivot_left_5m, 0)) + list(range(1, cfg.pivot_right_5m + 1)):
            other = vals.shift(offset)
            good &= vals.lt(other) if op == "lt" else vals.gt(other)
        # This candidate is only visible at the end of the two *following* bars.
        ready = five.loc[good, [field, "bar_close"]].copy()
        ready["available_at"] = ready.bar_close + pd.Timedelta(minutes=5 * cfg.pivot_right_5m)
        ready["source_time"] = ready.bar_close
        ready = ready.rename(columns={field: f"pivot_{side}_price"})
        lookup = z[["close_time"]].sort_values("close_time").copy()
        ready = ready.sort_values("available_at")
        if len(ready):
            lookup = pd.merge_asof(lookup, ready[["available_at", "source_time", f"pivot_{side}_price"]],
                                   left_on="close_time", right_on="available_at", direction="backward")
            age_min = (lookup.close_time - lookup.available_at).dt.total_seconds() / 60
            valid = age_min.le(cfg.pivot_max_age_min)
            z[f"pivot_{side}_price"] = lookup[f"pivot_{side}_price"].where(valid).to_numpy()
            z[f"pivot_{side}_source_time"] = lookup["source_time"].where(valid).to_numpy()
        else:
            z[f"pivot_{side}_price"] = np.nan
            z[f"pivot_{side}_source_time"] = pd.NaT
    z["prior_day_source_time"] = z.utc_day - pd.Timedelta(days=1)
    return z[["open_time", "close_time", "prior_day_low", "prior_day_high", "prior_day_source_time",
              "pivot_support_price", "pivot_support_source_time", "pivot_resistance_price",
              "pivot_resistance_source_time"]]


def detect_price_only_events(x: pd.DataFrame, cfg: Config, symbol: str) -> pd.DataFrame:
    """Deterministic candidate set; does not inspect forward price paths or delta.

    Each stream tracks the most recent *confirmed* 5m pivot or prior full UTC
    day extremum. Retest requires an intermediate completed candle closing away
    from the level, not two adjacent candles' touches.
    """
    columns = ["symbol", "side", "level_type", "level_source_time", "level_price", "zone_halfwidth",
               "first_touch_close", "second_touch_close", "signal_time_utc", "bar_close", "bar_low", "bar_high",
               "bar_open", "lower_wick_fraction", "upper_wick_fraction", "first_touch_low", "first_touch_high",
               "minutes_between_touches", "delta_5m", "delta_15m", "quote_5m", "quote_15m",
               "cvd_utc_session", "vwap_utc_session"]
    if x.empty:
        return pd.DataFrame(columns=columns)
    if any(c not in x for c in ("signed_quote_1m", "delta_5m", "delta_15m")):
        raise ValueError("Call get_minutes before event detection")
    ref = get_reference_levels(x, cfg)
    a = x.merge(ref.drop(columns="close_time"), on="open_time", validate="one_to_one", how="left")
    a = a.sort_values("open_time").reset_index(drop=True)
    start = pd.Timestamp(cfg.start, tz="UTC")
    end = pd.Timestamp(cfg.end, tz="UTC") + pd.Timedelta(days=1)
    streams = [("support", "prior_day", "prior_day_low", "prior_day_source_time"),
               ("resistance", "prior_day", "prior_day_high", "prior_day_source_time"),
               ("support", "pivot_5m", "pivot_support_price", "pivot_support_source_time"),
               ("resistance", "pivot_5m", "pivot_resistance_price", "pivot_resistance_source_time")]
    states = {t[:2]: {"id": None, "first": None, "away": False, "last_signal": None} for t in streams}
    events = []
    # to_dict records with pd.Timestamp and float values: avoids iterrows dtype coercion.
    for row in a.to_dict("records"):
        when = row["close_time"]
        if pd.isna(when):
            continue
        for side, level_type, col, source in streams:
            state = states[(side, level_type)]
            level = row[col]
            stamp = row[source]
            if pd.isna(level) or pd.isna(stamp):
                state.update(id=None, first=None, away=False)
                continue
            level = float(level)
            width = level * cfg.zone_halfwidth_bp / 10_000
            if not np.isfinite(width) or width <= 0:
                continue
            identifier = (stamp, level)
            if state["id"] != identifier:
                state.update(id=identifier, first=None, away=False)
            o, h, l, c = (float(row[k]) for k in ("open", "high", "low", "close"))
            candle_range = max(h - l, 1e-12)
            low_wick = (min(o, c) - l) / candle_range
            high_wick = (h - max(o, c)) / candle_range
            if side == "support":
                invalid_close = c < level - 2 * width
                departed = c >= level + cfg.leave_zone_multiple * width
                valid_touch = (level - cfg.max_penetration_zone_widths * width <= l <= level + width
                               and c > level + width and low_wick >= cfg.wick_fraction_min)
            else:
                invalid_close = c > level + 2 * width
                departed = c <= level - cfg.leave_zone_multiple * width
                valid_touch = (level - width <= h <= level + cfg.max_penetration_zone_widths * width
                               and c < level - width and high_wick >= cfg.wick_fraction_min)
            if invalid_close:
                state.update(first=None, away=False)
                continue
            first = state["first"]
            if first is not None:
                elapsed = (when - first["close_time"]).total_seconds() / 60
                if elapsed > cfg.max_retest_gap_min:
                    state.update(first=None, away=False)
                    first = None
            cooldown = state["last_signal"] is not None and (when - state["last_signal"]).total_seconds()/60 < cfg.cooldown_after_event_min
            if not valid_touch or cooldown:
                if first is not None and departed and when > first["close_time"]:
                    state["away"] = True
                continue
            if first is not None:
                elapsed = (when - first["close_time"]).total_seconds()/60
                if (elapsed >= cfg.min_touch_separation_min and state["away"] and start <= when < end):
                    events.append({"symbol": symbol, "side": "long" if side == "support" else "short",
                                   "level_type": level_type, "level_source_time": stamp,
                                   "level_price": level, "zone_halfwidth": width,
                                   "first_touch_close": first["close_time"],
                                   "second_touch_close": when, "signal_time_utc": when,
                                   "bar_close": c, "bar_low": l, "bar_high": h, "bar_open": o,
                                   "lower_wick_fraction": low_wick, "upper_wick_fraction": high_wick,
                                   "first_touch_low": first["low"], "first_touch_high": first["high"],
                                   "minutes_between_touches": elapsed,
                                   **{k: float(row[k]) for k in ("delta_5m", "delta_15m", "quote_5m",
                                                                        "quote_15m", "cvd_utc_session", "vwap_utc_session")}})
                    state.update(first=None, away=False, last_signal=when)
                    continue
            if first is None:
                state.update(first={"close_time": when, "low": l, "high": h}, away=False)
            elif departed and when > first["close_time"]:
                # A *later* retest may use this completed candle as an excursion,
                # never the current retest candle itself.
                state["away"] = True
    return pd.DataFrame.from_records(events, columns=columns)


def trade_profile_window_features(events: pd.DataFrame, trade_root: Path, symbol: str, cfg: Config) -> pd.DataFrame:
    """Profile in 5m and preceding 5m ending at the signal *close*.

    Scan needed UTC-day trade files in bounded Parquet batches. Trades exactly
    at signal_time are excluded. Counts and volume around a level are approximate
    because the price axis is binned at 1bp and compared using bin centers.
    """
    result = events.copy().reset_index(drop=True)
    extra = ["profile_status", "zone_quote_last5m", "zone_quote_prev5m", "zone_buy_quote_last5m",
             "zone_sell_quote_last5m", "all_quote_last5m", "near_zone_share_last5m",
             "zone_growth_quote_ratio", "zone_delta_last5m"]
    for c in extra:
        result[c] = np.nan if c != "profile_status" else "NOT_COMPUTED"
    if result.empty:
        return result
    # Collect only minutes that can contribute to a pre-signal 10-minute window.
    wants: dict[str, set[int]] = {}
    times = result.signal_time_utc.dt.as_unit("ms").astype("int64").to_numpy()
    for t in times:
        for m in range(1, cfg.profile_window_min * 2 + 1):
            v = int(t - m * MINUTE_MS)
            day = pd.Timestamp(v, unit="ms", tz="UTC").strftime("%Y-%m-%d")
            wants.setdefault(day, set()).add(v)
    width = np.log1p(cfg.profile_bin_bp / 10_000)
    pieces = []
    found = set()
    import pyarrow.parquet as pq
    for day in sorted(wants):
        fp = trade_root / "aggTrades" / symbol / f"{symbol}-aggTrades-{day}.parquet"
        if not fp.exists():
            continue
        found.add(day)
        needed = np.array(sorted(wants[day]), dtype=np.int64)
        pf = pq.ParquetFile(fp)
        for batch in pf.iter_batches(batch_size=250_000, columns=["time_ms", "price", "qty", "is_buyer_maker"]):
            d = batch.to_pandas()
            if d[["time_ms", "price", "qty", "is_buyer_maker"]].isna().any().any():
                raise ValueError(f"Null trade inputs in {fp}")
            t = d.time_ms.to_numpy(dtype="int64")
            minute = (t // MINUTE_MS) * MINUTE_MS
            selected = np.isin(minute, needed)
            if not np.any(selected):
                continue
            px = d.price.to_numpy(dtype="float64")[selected]
            qty = d.qty.to_numpy(dtype="float64")[selected]
            side = d.is_buyer_maker.to_numpy(dtype="bool")[selected]
            valid = np.isfinite(px) & np.isfinite(qty) & (px > 0) & (qty > 0)
            if not np.all(valid):
                raise ValueError(f"Invalid prices or quantities in {fp}")
            quote = px * qty
            v = pd.DataFrame({"minute_ms": minute[selected],
                              "bin": np.floor(np.log(px) / width).astype("int64"),
                              "quote": quote, "buy": np.where(side, 0.0, quote),
                              "sell": np.where(side, quote, 0.0)})
            pieces.append(v.groupby(["minute_ms", "bin"], sort=False, as_index=False).sum())
    binned = (pd.concat(pieces).groupby(["minute_ms", "bin"], sort=False, as_index=False).sum()
              if pieces else pd.DataFrame(columns=["minute_ms", "bin", "quote", "buy", "sell"]))
    binned = binned.sort_values(["minute_ms", "bin"]).reset_index(drop=True)
    bin_times = binned["minute_ms"].to_numpy(dtype="int64")
    for idx, event in result.iterrows():
        end_ms = int(times[idx])
        window = [end_ms - m * MINUTE_MS for m in range(1, 2 * cfg.profile_window_min + 1)]
        missing_days = {pd.Timestamp(v, unit="ms", tz="UTC").strftime("%Y-%m-%d") for v in window} - found
        if missing_days:
            result.at[idx, "profile_status"] = "MISSING_TRADE_DAY"
            continue
        if binned.empty:
            result.at[idx, "profile_status"] = "NO_TRADE_BINS"
            continue
        first_index = np.searchsorted(bin_times, end_ms - 2 * cfg.profile_window_min * MINUTE_MS, side="left")
        after_index = np.searchsorted(bin_times, end_ms, side="left")
        a = binned.iloc[first_index:after_index].copy()
        if len(np.unique(a.minute_ms)) != 2 * cfg.profile_window_min:
            result.at[idx, "profile_status"] = "MISSING_TRADE_MINUTE"
            continue
        # Grouped center lies within one half-bin of each actual trade price.
        midpoint = np.exp((a["bin"].to_numpy(float) + 0.5) * width)
        near = (midpoint >= event.level_price - event.zone_halfwidth) & (midpoint <= event.level_price + event.zone_halfwidth)
        a = a.assign(near=near)
        recent = a.loc[a.minute_ms >= end_ms - cfg.profile_window_min * MINUTE_MS]
        previous = a.loc[a.minute_ms < end_ms - cfg.profile_window_min * MINUTE_MS]
        recent_near = recent.loc[recent.near]
        previous_near = previous.loc[previous.near]
        last5 = float(recent_near.quote.sum())
        prev5 = float(previous_near.quote.sum())
        buy5 = float(recent_near.buy.sum())
        sell5 = float(recent_near.sell.sum())
        all5 = float(recent.quote.sum())
        result.at[idx, "profile_status"] = "PRESENT" if all5 > 0 else "NO_TRADES_IN_WINDOW"
        result.at[idx, "zone_quote_last5m"] = last5
        result.at[idx, "zone_quote_prev5m"] = prev5
        result.at[idx, "zone_buy_quote_last5m"] = buy5
        result.at[idx, "zone_sell_quote_last5m"] = sell5
        result.at[idx, "all_quote_last5m"] = all5
        result.at[idx, "near_zone_share_last5m"] = last5 / all5 if all5 > 0 else np.nan
        result.at[idx, "zone_growth_quote_ratio"] = last5 / prev5 if prev5 > 0 else np.nan
        result.at[idx, "zone_delta_last5m"] = buy5 - sell5
    return result


def run_a02(data_root: Path, out: Path, cfg: Config = Config(), symbols=("BTCUSDT", "ETHUSDT")) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    summary, prices, enriched = [], [], []
    for sym in symbols:
        fp = data_root / f"{sym}_um_1m.parquet"
        if not fp.exists():
            raise FileNotFoundError(fp)
        k = pd.read_parquet(fp)
        minute = get_minutes(k, cfg)
        event = detect_price_only_events(minute, cfg, sym)
        extra = trade_profile_window_features(event, data_root, sym, cfg)
        prices.append(event)
        enriched.append(extra)
        summary.append({"symbol": sym, "minute_rows_including_warmup": len(minute),
                        "price_only_candidates": len(event), "profile_present": int((extra.profile_status == "PRESENT").sum()),
                        "missing_trade_day_events": int((extra.profile_status == "MISSING_TRADE_DAY").sum()),
                        "first_signal": str(event.signal_time_utc.min()) if len(event) else "",
                        "last_signal": str(event.signal_time_utc.max()) if len(event) else ""})
    p = pd.concat(prices, ignore_index=True)
    e = pd.concat(enriched, ignore_index=True)
    # Strictly separate the price-only baseline from contemporaneous flow features.
    # Selection of candidates is price-only in both files.
    price_only_drop = ["delta_5m", "delta_15m", "quote_5m", "quote_15m",
                       "cvd_utc_session", "vwap_utc_session"]
    p.drop(columns=price_only_drop).to_csv(out / "02_price_only_candidates.csv", index=False)
    e.to_csv(out / "02_causal_flow_features.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "02_coverage.csv", index=False)
    spec = asdict(cfg)
    (out / "02_feature_contract.json").write_text(json.dumps({"version": LA_A02_VERSION, "config": spec,
        "design_status": "EXPLORATORY_FEATURE_CONTRACT_NOT_FROZEN_FOR_PNL",
        "signals_are_after_closed_1m_bar": True, "features_include_future_returns": False,
        "profile_method": "1bp midpoint-binned trade quote volume, approximate level volume",
        "note": "Context thresholds are hypotheses, NOT fitted to returns. No actual trade entries/exits are executed."}, indent=2))
    decision = {"version": LA_A02_VERSION, "status": "FEATURE_EXTRACTION_ONLY", "symbols": list(symbols),
                "summary": summary,
                "caution": "No backtested P&L, economic edge, stop simulation, prospective holdout or proof of absorption."}
    (out / "02_decision.json").write_text(json.dumps(decision, indent=2))
    return decision


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binance-root")
    p.add_argument("--out", type=Path)
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    a = p.parse_args(argv)
    base = repo_root()
    root = binance_root(base, a.binance_root)
    out = a.out or base / "notebooks" / "Liquidity_Absorption" / "02"
    print(json.dumps(run_a02(root, out, symbols=a.symbols), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
