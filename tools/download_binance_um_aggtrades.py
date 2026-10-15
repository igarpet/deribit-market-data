"""Download Binance USD-M perpetual aggTrades (free public data) as one parquet per symbol and UTC day.

Source: https://data.binance.vision/data/futures/um/daily/aggTrades/<SYMBOL>/<SYMBOL>-aggTrades-YYYY-MM-DD.zip
Each zip is verified against its SHA256 .CHECKSUM file, converted, and deleted (unless --keep-zips).

Used by Candle Streaks CS03e (passive-fill test). Daily files keep memory small (BTC ~1-3M rows/day).

Usage (PC, Windows CMD):
    python tools\\download_binance_um_aggtrades.py --symbols BTCUSDT ETHUSDT --start 2026-07-01 --end 2026-09-30
Default output: <BINANCE_UM_DIR or parent of DERIBIT_DATA_ROOT\\Binance>\\aggTrades\\<SYMBOL>\\<SYMBOL>-aggTrades-<day>.parquet

Columns kept: time_ms (int64, trade time UTC ms), price (float64), qty (float64),
is_buyer_maker (bool; True = the aggressor SOLD).
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
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

BASE = "https://data.binance.vision/data/futures/um/daily/aggTrades"
COLS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker"]


def fetch(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=300) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def parse_csv(raw: bytes) -> pd.DataFrame:
    first = raw.split(b"\n", 1)[0]
    has_header = not first[:1].isdigit()
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None,
                     usecols=[1, 2, 5, 6], names=COLS if not has_header else None)
    df.columns = ["price", "qty", "time", "is_buyer_maker"]
    t = df["time"].astype("int64")
    if t.max() > 10**14:                     # microsecond guard
        t = t // 1000
    ibm = df["is_buyer_maker"]
    if ibm.dtype != bool:
        ibm = ibm.astype(str).str.lower().eq("true")
    out = pd.DataFrame({"time_ms": t.to_numpy("int64"), "price": df["price"].astype("float64"),
                        "qty": df["qty"].astype("float64"), "is_buyer_maker": ibm.to_numpy(bool)})
    return out.sort_values("time_ms", kind="stable").reset_index(drop=True)


def one_day(symbol: str, d: date, out: Path, keep_zips: bool) -> str:
    target = out / symbol / f"{symbol}-aggTrades-{d:%Y-%m-%d}.parquet"
    if target.exists():
        return "exists"
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{symbol}/{symbol}-aggTrades-{d:%Y-%m-%d}.zip"
    data = fetch(url)
    if data is None:
        return "missing"
    chk = fetch(url + ".CHECKSUM")
    if chk is not None and hashlib.sha256(data).hexdigest() != chk.decode().split()[0].strip().lower():
        raise ValueError(f"Checksum mismatch for {url}")
    if keep_zips:
        (out / "zips").mkdir(parents=True, exist_ok=True)
        (out / "zips" / url.rsplit("/", 1)[-1]).write_bytes(data)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        raw = z.read(z.namelist()[0])
    df = parse_csv(raw)
    df.to_parquet(target, index=False)
    return f"{len(df):,} trades"


def default_out() -> Path | None:
    v = os.getenv("BINANCE_UM_DIR")
    if v:
        return Path(v) / "aggTrades"
    root = os.getenv("DERIBIT_DATA_ROOT")
    if not root:
        env = Path(__file__).resolve().parents[1] / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("DERIBIT_DATA_ROOT="):
                    root = line.split("=", 1)[1].strip().strip('"').strip("'")
    return Path(root).parent / "Binance" / "aggTrades" if root else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    ap.add_argument("--start", default="2026-07-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep-zips", action="store_true")
    a = ap.parse_args()
    out = Path(a.out) if a.out else default_out()
    if out is None:
        print("Set --out, BINANCE_UM_DIR or DERIBIT_DATA_ROOT")
        return 1
    start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
    for s in a.symbols:
        d = start
        while d <= end:
            print(f"{s} {d}: {one_day(s, d, out, a.keep_zips)}", flush=True)
            d += timedelta(days=1)
    print("Output:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
