"""Download Binance USD-M futures 'metrics' (free public data): 5-minute open interest and long/short ratios.

Source: https://data.binance.vision/data/futures/um/daily/metrics/<SYMBOL>/<SYMBOL>-metrics-YYYY-MM-DD.zip
(daily files only). Each zip is verified against its SHA256 .CHECKSUM file when one is published.
Used by Funding Carry FC 02 (OI / positioning block). Downloads run in parallel threads; days already in the
output parquet are skipped on re-runs; zips are not kept.

Usage (PC, Windows CMD):
    python tools\\download_binance_um_metrics.py --start 2024-07-01
Optional:
    --end 2026-10-17  --symbols BTCUSDT ETHUSDT ...  --workers 8  --out D:\\path
Default output: <BINANCE_UM_DIR or parent of DERIBIT_DATA_ROOT\\Binance>\\basis\\metrics\\<SYMBOL>_metrics.parquet

Columns kept: create_time (UTC), sum_open_interest, sum_open_interest_value, count_toptrader_long_short_ratio,
sum_toptrader_long_short_ratio, count_long_short_ratio, sum_taker_long_short_vol_ratio.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

BASE = "https://data.binance.vision/data/futures/um/daily/metrics"
KEEP = ["create_time", "sum_open_interest", "sum_open_interest_value", "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]
UNIVERSE = ("BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "LINK", "DOT", "LTC", "BCH", "UNI", "NEAR", "APT",
            "ARB", "OP", "SUI", "FIL", "ATOM", "ETC", "INJ", "AAVE", "TON", "WIF", "HBAR", "XLM", "SEI", "TIA",
            "ENA", "WLD")


def fetch(url: str, tries: int = 3) -> bytes | None:
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if k == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if k == tries - 1:
                raise
    return None


def one_day(symbol: str, d: date) -> pd.DataFrame | None:
    url = f"{BASE}/{symbol}/{symbol}-metrics-{d:%Y-%m-%d}.zip"
    data = fetch(url)
    if data is None:
        return None
    chk = fetch(url + ".CHECKSUM")
    if chk is not None:
        if hashlib.sha256(data).hexdigest() != chk.decode().split()[0].strip().lower():
            raise ValueError(f"Checksum mismatch for {url}")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw))
    if "create_time" not in df.columns:                      # header-less fallback
        df = pd.read_csv(io.BytesIO(raw), header=None)
        df.columns = ["create_time", "symbol", *KEEP[1:]][:df.shape[1]]
    return df[[c for c in KEEP if c in df.columns]]


def build(symbol: str, start: date, end: date, out: Path, workers: int) -> None:
    target = out / f"{symbol}_metrics.parquet"
    old = pd.read_parquet(target) if target.exists() else None
    have = set()
    if old is not None and len(old):
        have = set(pd.to_datetime(old["create_time"], utc=True).dt.date)
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    todo = [d for d in days if d not in have]
    print(f"== {symbol}: {len(todo)} days to fetch ({len(days) - len(todo)} already present)")
    frames, missing = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for d, df in zip(todo, ex.map(lambda x: one_day(symbol, x), todo)):
            if df is None:
                missing += 1
            else:
                frames.append(df)
    if old is not None:
        frames.append(old)
    if not frames:
        print(f"  no data for {symbol}")
        return
    df = pd.concat(frames, ignore_index=True)
    df["create_time"] = pd.to_datetime(df["create_time"], utc=True)
    df = df.drop_duplicates("create_time").sort_values("create_time").reset_index(drop=True)
    df.to_parquet(target, index=False)
    print(f"  {len(df):,} rows {df['create_time'].min()} -> {df['create_time'].max()} | missing days {missing} | {target}")


def default_out() -> Path | None:
    v = os.getenv("BINANCE_UM_DIR")
    root = None
    if not v:
        root = os.getenv("DERIBIT_DATA_ROOT")
        if not root:
            env = Path(__file__).resolve().parents[1] / ".env"
            if env.exists():
                for line in env.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("DERIBIT_DATA_ROOT="):
                        root = line.split("=", 1)[1].strip().strip('"').strip("'")
    base = Path(v) if v else (Path(root).parent / "Binance" if root else None)
    return base / "basis" / "metrics" if base else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=default_out())
    ap.add_argument("--start", default="2024-07-01")
    ap.add_argument("--end", default=str(date.today() - timedelta(days=1)))
    ap.add_argument("--symbols", nargs="+", default=[f"{x}USDT" for x in UNIVERSE])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.out is None:
        ap.error("--out is required when DERIBIT_DATA_ROOT / BINANCE_UM_DIR are not set")
    a.out.mkdir(parents=True, exist_ok=True)
    s, e = date.fromisoformat(a.start), date.fromisoformat(a.end)
    for sym in a.symbols:
        build(sym, s, e, a.out, a.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
