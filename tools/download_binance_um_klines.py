"""Download Binance USD-M perpetual 1m klines (free public data) and build one parquet per symbol.

Source: https://data.binance.vision (monthly files for complete months, daily files for the current month).
Each zip is verified against its SHA256 .CHECKSUM file.

Used by the VWAP Events branch (VWAP 001). Default symbols: the project assets with a Binance USD-M
perpetual (<ASSET>USDT); a symbol or month that does not exist is reported as missing and skipped.
Default output: <parent of DERIBIT_DATA_ROOT>/Binance (e.g. D:\\MarketData\\Binance).

Usage (PC, Windows CMD):
    python tools\\download_binance_um_klines.py --months 24
Optional:
    --out D:\\path  --symbols BTCUSDT ETHUSDT ...

Columns kept: open_time (UTC), open, high, low, close, volume, quote_volume, trades,
taker_buy_volume, taker_buy_quote_volume.
    VWAP over any window  = sum(quote_volume) / sum(volume)
    Per-bar delta         = 2 * taker_buy_volume - volume   (CVD = cumulative sum)
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

BASE = "https://data.binance.vision/data/futures/um"
COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
KEEP = [
    "open_time", "open", "high", "low", "close", "volume",
    "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume",
]


def fetch(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def get_verified_zip(url: str, cache: Path) -> Path | None:
    dest = cache / url.rsplit("/", 1)[-1]
    if dest.exists():
        return dest
    data = fetch(url)
    if data is None:
        return None
    chk = fetch(url + ".CHECKSUM")
    if chk is not None:
        expected = chk.decode().split()[0].strip().lower()
        got = hashlib.sha256(data).hexdigest()
        if got != expected:
            raise ValueError(f"Checksum mismatch for {url}")
    dest.write_bytes(data)
    return dest


def read_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0])
    first = raw.split(b"\n", 1)[0]
    has_header = not first[:1].isdigit()
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)
    df.columns = COLS
    return df[KEEP]


def month_list(n: int, today: date) -> list[tuple[int, int]]:
    y, m = today.year, today.month
    out = []
    for _ in range(n):
        m -= 1
        if m == 0:
            y, m = y - 1, 12
        out.append((y, m))
    return sorted(out)


def build(symbol: str, months: int, out: Path) -> None:
    cache = out / "zips" / symbol
    cache.mkdir(parents=True, exist_ok=True)
    today = date.today()
    urls = [
        f"{BASE}/monthly/klines/{symbol}/1m/{symbol}-1m-{y}-{m:02d}.zip"
        for y, m in month_list(months, today)
    ]
    d = today.replace(day=1)
    while d < today:  # current month: daily files up to yesterday
        urls.append(f"{BASE}/daily/klines/{symbol}/1m/{symbol}-1m-{d:%Y-%m-%d}.zip")
        d += timedelta(days=1)

    frames = []
    for url in urls:
        p = get_verified_zip(url, cache)
        if p is None:
            print(f"  missing (skipped): {url.rsplit('/', 1)[-1]}")
            continue
        frames.append(read_zip(p))
        print(f"  ok: {p.name}")

    if not frames:
        print(f"No data for {symbol}")
        return
    df = pd.concat(frames, ignore_index=True)
    ts = df["open_time"].astype("int64")
    unit = "us" if ts.max() > 10**14 else "ms"  # guard for microsecond timestamps
    df["open_time"] = pd.to_datetime(ts, unit=unit, utc=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

    gaps = df["open_time"].diff().dt.total_seconds().gt(60).sum()
    target = out / f"{symbol}_um_1m.parquet"
    df.to_parquet(target, index=False)
    print(f"{symbol}: {len(df):,} rows, {df['open_time'].min()} -> {df['open_time'].max()}, "
          f"gaps>1m: {gaps}, saved {target}")


ASSETS = ("BTC", "ETH", "SOL", "XRP", "AVAX", "TRX", "HYPE")


def default_out() -> Path | None:
    v = os.getenv("BINANCE_UM_DIR")
    if v:
        return Path(v)
    root = os.getenv("DERIBIT_DATA_ROOT")
    if not root:
        env = Path(__file__).resolve().parents[1] / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("DERIBIT_DATA_ROOT="):
                    root = line.split("=", 1)[1].strip().strip('"').strip("'")
    return Path(root).parent / "Binance" if root else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=default_out())
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--symbols", nargs="+", default=[f"{x}USDT" for x in ASSETS])
    a = ap.parse_args()
    if a.out is None:
        ap.error("--out is required when DERIBIT_DATA_ROOT / BINANCE_UM_DIR are not set")
    a.out.mkdir(parents=True, exist_ok=True)
    for s in a.symbols:
        print(f"== {s}")
        build(s, a.months, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
