"""Liquidity Absorption Stage A/01: audit locally held Binance research data.

Research-only, read-only inputs. No signals, strategy selection, or trading orders.
Dependencies: numpy, pandas, pyarrow (for Parquet). All outputs under an explicit --out.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "LA-A01-0.1.2"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
MINUTE_MS = 60_000
TRADE_RE = re.compile(r"-aggTrades-(\d{4}-\d{2}-\d{2})\.parquet$")
MINUTE_COLS = (
    "open_time", "open", "high", "low", "close", "volume", "quote_volume",
    "trades", "taker_buy_volume", "taker_buy_quote_volume",
)
# The basis/spot 1m downloader intentionally retains no OHLC; only close.
SPOT_COLS = tuple(c for c in MINUTE_COLS if c not in ("open", "high", "low"))


def repo_root() -> Path:
    p = Path(__file__).resolve()
    for d in p.parents:
        if (d / "src" / "data_engineering").is_dir():
            return d
    raise RuntimeError("Cannot locate repository root")


def binance_root(root: Path, override: str | None = None) -> Path:
    if override:
        return Path(override)
    values = dict(os.environ)
    for env_path in (root / ".env",):
        if env_path.is_file():
            for raw in env_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    values.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if values.get("BINANCE_UM_DIR"):
        return Path(values["BINANCE_UM_DIR"])
    if values.get("DERIBIT_DATA_ROOT"):
        return Path(values["DERIBIT_DATA_ROOT"]).parent / "Binance"
    raise ValueError("Provide --binance-root or set BINANCE_UM_DIR / DERIBIT_DATA_ROOT")


def parse_kline_open_time(values: pd.Series) -> pd.Series:
    """UTC Binance candle open times: accept datetimes or Unix s/ms/us/ns.

    Binance archive files can store the timestamp as numeric milliseconds or
    microseconds, whereas previously prepared parquet files may store proper
    pandas datetimes. pd.to_datetime(numeric) defaults to nanoseconds, which
    collapses many distinct minute bars onto the same join key if misapplied.
    """
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        finite = numeric[np.isfinite(numeric)]
        if finite.empty:
            return pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
        size = float(finite.abs().median())
        if 1e17 <= size < 1e20:
            unit = "ns"
        elif 1e14 <= size < 1e17:
            unit = "us"
        elif 1e11 <= size < 1e14:
            unit = "ms"
        elif 1e8 <= size < 1e11:
            unit = "s"
        else:
            raise ValueError(f"Unrecognized numeric Binance open_time epoch scale: median={size:g}")
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(values, utc=True, errors="coerce")


def minute_quality(df: pd.DataFrame, symbol: str, venue: str) -> dict:
    missing = set(SPOT_COLS if venue == "spot" else MINUTE_COLS) - set(df.columns)
    if missing:
        return {"symbol": symbol, "venue": venue, "status": "BAD_SCHEMA", "missing_columns": ",".join(sorted(missing))}
    t = parse_kline_open_time(df["open_time"])
    valid = t.dropna().sort_values()
    step_s = valid.diff().dt.total_seconds().dropna()
    price_cols = ["close"] if venue == "spot" else ["open", "high", "low", "close"]
    numeric = df[price_cols + ["volume", "quote_volume", "taker_buy_quote_volume"]].apply(pd.to_numeric, errors="coerce")
    bad_price = (~np.isfinite(numeric[price_cols].to_numpy()) | (numeric[price_cols].to_numpy() <= 0)).any(axis=1).sum()
    neg_volume = (numeric[["volume", "quote_volume", "taker_buy_quote_volume"]] < 0).any(axis=1).sum()
    invalid_taker = (numeric["taker_buy_quote_volume"] > numeric["quote_volume"] + np.maximum(1e-8, numeric["quote_volume"].abs() * 1e-8)).sum()
    return {
        "symbol": symbol, "venue": venue, "status": "PRESENT", "rows": len(df),
        "first_utc": str(valid.iloc[0]) if len(valid) else "",
        "last_utc": str(valid.iloc[-1]) if len(valid) else "",
        "null_times": int(t.isna().sum()), "duplicate_minutes": int(t.duplicated().sum()),
        "out_of_order_minutes": int((t.diff().dt.total_seconds() < 0).sum()),
        "missing_minutes_inside_span": int(((step_s / 60).round() - 1).clip(lower=0).sum()),
        "non_one_minute_intervals": int((step_s != 60).sum()),
        "bad_price_rows": int(bad_price), "negative_volume_rows": int(neg_volume),
        "taker_quote_exceeds_total_rows": int(invalid_taker),
    }


def daily_trade_paths(root: Path, symbol: str) -> list[Path]:
    d = root / "aggTrades" / symbol
    return sorted(p for p in d.glob(f"{symbol}-aggTrades-????-??-??.parquet") if TRADE_RE.search(p.name))


def coverage(symbol: str, paths: list[Path]) -> dict:
    dates = pd.DatetimeIndex([pd.Timestamp(TRADE_RE.search(p.name).group(1), tz="UTC") for p in paths])
    full = pd.date_range(dates.min(), dates.max(), freq="D", tz="UTC") if len(dates) else pd.DatetimeIndex([], tz="UTC")
    return {"symbol": symbol, "daily_files": len(paths),
            "first_day": str(dates.min().date()) if len(dates) else "",
            "last_day": str(dates.max().date()) if len(dates) else "",
            "missing_days_within_span": int(len(full.difference(dates))) if len(dates) else None,
            "missing_day_examples": ",".join(str(d.date()) for d in full.difference(dates)[:10]) if len(dates) else ""}


def choose_probe_days(paths: list[Path]) -> list[Path]:
    """Fixed temporal positions, not selected from trade outcomes."""
    return [paths[i] for i in sorted({0, len(paths) // 2, len(paths) - 1})] if paths else []


def aggregate_trade_chunk(df: pd.DataFrame, profile_bin_bps: float = 2.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Batchwise minute/price bins. Binning is diagnostic, NOT a strategy parameter."""
    cols = ["time_ms", "price", "qty", "is_buyer_maker"]
    if any(c not in df for c in cols):
        raise ValueError("Missing aggTrades columns")
    x = df[cols].copy()
    tm = pd.to_numeric(x["time_ms"], errors="coerce").to_numpy(dtype=float)
    px = pd.to_numeric(x["price"], errors="coerce").to_numpy(dtype=float)
    qty = pd.to_numeric(x["qty"], errors="coerce").to_numpy(dtype=float)
    side = x["is_buyer_maker"]
    good = np.isfinite(tm) & np.isfinite(px) & np.isfinite(qty) & (px > 0) & (qty > 0) & side.notna().to_numpy()
    if not np.any(good):
        return pd.DataFrame(columns=["minute_ms", "quote", "buy_quote", "sell_quote", "trade_rows"]), pd.DataFrame(columns=["log_price_bin", "quote", "trade_rows"])
    b = side[good].astype(bool).to_numpy()
    quote = px[good] * qty[good]
    m = pd.DataFrame({"minute_ms": (tm[good].astype("int64") // MINUTE_MS) * MINUTE_MS,
                      "quote": quote, "buy_quote": np.where(b, 0.0, quote),
                      "sell_quote": np.where(b, quote, 0.0), "trade_rows": 1})
    width = np.log1p(profile_bin_bps / 10000)
    p = pd.DataFrame({"log_price_bin": np.floor(np.log(px[good]) / width).astype("int64"),
                      "quote": quote, "trade_rows": 1})
    return (m.groupby("minute_ms", as_index=False).sum(), p.groupby("log_price_bin", as_index=False).sum())


def probe_trades(path: Path, profile_bin_bps: float = 2.0, batch_size: int = 200_000):
    """Stream one trade day to avoid retaining a multi-million-row file in memory."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    must = {"time_ms", "price", "qty", "is_buyer_maker"}
    if not must.issubset(set(pf.schema.names)):
        return ({"file": path.name, "status": "BAD_SCHEMA", "missing_columns": ",".join(sorted(must - set(pf.schema.names)))}, pd.DataFrame(), pd.DataFrame())
    mparts, pparts = [], []
    n = bad = descents = null_side = 0
    first = last = prev = None
    for batch in pf.iter_batches(batch_size=batch_size, columns=sorted(must)):
        df = batch.to_pandas()
        t = pd.to_numeric(df["time_ms"], errors="coerce").to_numpy(dtype=float)
        px = pd.to_numeric(df["price"], errors="coerce").to_numpy(dtype=float)
        q = pd.to_numeric(df["qty"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(t) & np.isfinite(px) & np.isfinite(q) & (px > 0) & (q > 0) & df["is_buyer_maker"].notna().to_numpy()
        bad += int((~mask).sum())
        null_side += int(df["is_buyer_maker"].isna().sum())
        valid_t = t[np.isfinite(t)]
        if len(valid_t):
            if prev is not None and valid_t[0] < prev:
                descents += 1
            descents += int((np.diff(valid_t) < 0).sum())
            if first is None:
                first = int(valid_t[0])
            last = int(valid_t[-1]); prev = last
        mn, profile = aggregate_trade_chunk(df, profile_bin_bps)
        mparts.append(mn); pparts.append(profile)
        n += len(df)
    mins = pd.concat(mparts).groupby("minute_ms", as_index=False).sum() if mparts else pd.DataFrame()
    bins = pd.concat(pparts).groupby("log_price_bin", as_index=False).sum() if pparts else pd.DataFrame()
    info = {"file": path.name, "status": "PRESENT", "rows": n, "invalid_trade_rows": bad,
            "null_aggressor_side": null_side, "timestamp_descents": descents,
            "first_timestamp_utc": str(pd.to_datetime(first, unit="ms", utc=True)) if first is not None else "",
            "last_timestamp_utc": str(pd.to_datetime(last, unit="ms", utc=True)) if last is not None else "",
            "minute_count": len(mins), "profile_bin_count": len(bins)}
    return info, mins, bins


def compare_minute_flow(kline: pd.DataFrame, traded: pd.DataFrame, day: str, symbol: str) -> dict:
    timestamp = parse_kline_open_time(kline["open_time"])
    selected = timestamp.dt.strftime("%Y-%m-%d") == day
    k = kline.loc[selected].copy()
    # Never assume that astype("int64") is nanoseconds: pandas preserves
    # datetime64[ms]/[us] resolution on some environments. Normalize explicitly.
    k["minute_ms"] = timestamp.loc[selected].dt.as_unit("ms").astype("int64")
    if k["minute_ms"].duplicated().any():
        return {"symbol": symbol, "day": day, "matched_minutes": 0,
                "duplicate_kline_minutes": int(k["minute_ms"].duplicated().sum()),
                "status": "DUPLICATE_KLINE_MINUTES"}
    pairs = k.merge(traded, on="minute_ms", how="inner", validate="one_to_one")
    if pairs.empty:
        return {"symbol": symbol, "day": day, "matched_minutes": 0, "status": "NO_OVERLAP"}
    kquote = pd.to_numeric(pairs["quote_volume"], errors="coerce").to_numpy(float)
    kdelta = 2 * pd.to_numeric(pairs["taker_buy_quote_volume"], errors="coerce").to_numpy(float) - kquote
    tradequote = pairs["quote"].to_numpy(float)
    tradedelta = pairs["buy_quote"].to_numpy(float) - pairs["sell_quote"].to_numpy(float)
    denom = np.maximum(np.abs(kquote), 1.0)
    return {"symbol": symbol, "day": day, "status": "PRESENT", "matched_minutes": len(pairs),
            "median_quote_relative_error": float(np.median(np.abs(kquote - tradequote) / denom)),
            "p95_quote_relative_error": float(np.quantile(np.abs(kquote - tradequote) / denom, .95)),
            "median_delta_error_as_quote_fraction": float(np.median(np.abs(kdelta - tradedelta) / denom)),
            "p95_delta_error_as_quote_fraction": float(np.quantile(np.abs(kdelta - tradedelta) / denom, .95))}


def run_audit(data_root: Path, out: Path, symbols=SYMBOLS) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    inventory, cov, probes, flow, profiles = [], [], [], [], []
    statuses = []
    for symbol in symbols:
        perp = data_root / f"{symbol}_um_1m.parquet"
        spot = data_root / "basis" / "spot" / f"{symbol}_spot_1m.parquet"
        if not perp.exists():
            inventory.append({"symbol": symbol, "venue": "perp", "status": "MISSING_FILE", "file": str(perp)})
            statuses.append(False)
            continue
        df = pd.read_parquet(perp)
        quality = minute_quality(df, symbol, "perp")
        inventory.append({**quality, "file": str(perp)})
        statuses.append(quality.get("status") == "PRESENT" and all(quality.get(k, 1) == 0 for k in ("null_times", "duplicate_minutes", "out_of_order_minutes", "missing_minutes_inside_span", "bad_price_rows", "negative_volume_rows")))
        if spot.exists():
            sq = minute_quality(pd.read_parquet(spot), symbol, "spot")
            inventory.append({**sq, "file": str(spot)})
            statuses.append(sq.get("status") == "PRESENT" and all(sq.get(k, 1) == 0 for k in
                            ("null_times", "duplicate_minutes", "out_of_order_minutes", "missing_minutes_inside_span", "bad_price_rows", "negative_volume_rows")))
        else:
            inventory.append({"symbol": symbol, "venue": "spot", "status": "MISSING_FILE", "file": str(spot)})
        paths = daily_trade_paths(data_root, symbol)
        cov.append(coverage(symbol, paths))
        statuses.append(bool(paths) and cov[-1]["missing_days_within_span"] == 0)
        for path in choose_probe_days(paths):
            info, minutes, bins = probe_trades(path)
            day = TRADE_RE.search(path.name).group(1)
            probes.append({"symbol": symbol, "day": day, **info})
            statuses.append(info.get("status") == "PRESENT" and info.get("invalid_trade_rows") == 0 and info.get("timestamp_descents") == 0)
            if not minutes.empty:
                comparison = compare_minute_flow(df, minutes, day, symbol)
                flow.append(comparison)
                # Do not declare readiness until the sampled aggTrades actually
                # reconcile to the correct candle timestamps. Numerical flow
                # differences remain descriptive rather than a tuned threshold.
                statuses.append(comparison.get("status") == "PRESENT" and
                                comparison.get("matched_minutes", 0) == info.get("minute_count"))
            else:
                statuses.append(False)
            if not bins.empty:
                largest = bins.nlargest(50, "quote")
                profiles.extend({"symbol": symbol, "day": day, "log_price_bin": int(row.log_price_bin),
                                 "quote": float(row.quote), "trade_rows": int(row.trade_rows)}
                                for row in largest.itertuples(index=False))
        del df
    for filename, rows in (("01_minute_quality.csv", inventory), ("01_aggtrade_coverage.csv", cov),
                           ("01_trade_probe_quality.csv", probes), ("01_trade_kline_flow_comparison.csv", flow),
                           ("01_sample_volume_profile.csv", profiles)):
        pd.DataFrame(rows).to_csv(out / filename, index=False)
    decision = {"module": VERSION, "purpose": "readiness and consistency audit only, not strategy validation",
                "symbols": list(symbols), "binance_root": str(data_root), "ready_for_signal_design": bool(statuses and all(statuses)),
                "checks": {"all_present_sources_and_sample_integrity_and_flow_alignment": bool(statuses and all(statuses))},
                "warning": "Flow differences are descriptive; all-day aggTrades completeness is NOT proven by three-day probes. No edge is claimed.",
                "files": ["01_minute_quality.csv", "01_aggtrade_coverage.csv", "01_trade_probe_quality.csv",
                          "01_trade_kline_flow_comparison.csv", "01_sample_volume_profile.csv"]}
    (out / "01_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binance-root", help="Override the local Binance data directory")
    p.add_argument("--out", type=Path)
    p.add_argument("--symbols", nargs="+", default=list(SYMBOLS))
    a = p.parse_args(argv)
    root = repo_root()
    data_root = binance_root(root, a.binance_root)
    out = a.out or root / "notebooks" / "Liquidity_Absorption" / "01"
    print(json.dumps(run_audit(data_root, out, a.symbols), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
