"""Download Binance SPOT 1-second klines (free public data), one parquet per symbol and month.

Source: https://data.binance.vision/data/spot/monthly/klines/<SYM>/1s/<SYM>-1s-YYYY-MM.zip
        (daily files for the current, incomplete month: .../spot/daily/klines/<SYM>/1s/<SYM>-1s-YYYY-MM-DD.zip)
Each zip is verified against its SHA256 .CHECKSUM file. Existing output months are skipped (re-runs resume).

Usage (PC, Windows CMD):
    python tools\\download_binance_spot_1s.py --months 3
    python tools\\download_binance_spot_1s.py --symbols BTCUSDT ETHUSDT --months 2 --include-current

Output: <BINANCE_UM_DIR or parent of DERIBIT_DATA_ROOT\\Binance>\\spot_1s\\<SYM>\\<SYM>_spot_1s_<YYYY-MM>.parquet
        (current month, if requested: <SYM>_spot_1s_<YYYY-MM>_partial.parquet, rewritten on each run)
Columns: open_time (UTC), close, volume, quote_volume, trades, taker_buy_volume, taker_buy_quote_volume
Size: about 2.6 M rows (~60-120 MB parquet) per symbol-month.
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

BASE = "https://data.binance.vision/data/spot"
KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
         "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
KEEP = ["open_time", "close", "volume", "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume"]
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")


def fetch(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=300) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def verified(url: str) -> bytes | None:
    data = fetch(url)
    if data is None:
        return None
    chk = fetch(url + ".CHECKSUM")
    if chk is not None and hashlib.sha256(data).hexdigest() != chk.decode().split()[0].strip().lower():
        raise ValueError(f"Checksum mismatch: {url}")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return z.read(z.namelist()[0])


def parse_klines(raw: bytes) -> pd.DataFrame:
    has_header = not raw.split(b"\n", 1)[0][:1].isdigit()
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)
    df.columns = KCOLS
    t = df["open_time"].astype("int64")
    df["open_time"] = pd.to_datetime(t, unit="us" if t.max() > 10 ** 14 else "ms", utc=True)  # spot: us since 2025
    out = df[KEEP].copy()
    for c in KEEP[1:]:
        out[c] = out[c].astype("float64")
    return out


def months_back(n: int, today: date) -> list[tuple[int, int]]:
    y, m, out = today.year, today.month, []
    for _ in range(n):
        m -= 1
        if m == 0:
            y, m = y - 1, 12
        out.append((y, m))
    return sorted(out)


def default_out() -> Path | None:
    v = os.getenv("BINANCE_UM_DIR")
    if v:
        return Path(v)
    root = os.getenv("DERIBIT_DATA_ROOT")
    if not root:
        env = Path(__file__).resolve().parents[1] / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s.startswith("BINANCE_UM_DIR="):
                    return Path(s.split("=", 1)[1].strip().strip('"').strip("'"))
                if s.startswith("DERIBIT_DATA_ROOT="):
                    root = s.split("=", 1)[1].strip().strip('"').strip("'")
    return Path(root).parent / "Binance" if root else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    ap.add_argument("--months", type=int, default=3, help="complete calendar months back from today")
    ap.add_argument("--include-current", action="store_true", help="also fetch daily files of the current month")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    base_out = a.out or default_out()
    if base_out is None:
        print("Set BINANCE_UM_DIR or DERIBIT_DATA_ROOT in .env, or pass --out", file=sys.stderr)
        return 2
    today = date.today()
    months = months_back(a.months, today)
    for sym in a.symbols:
        d = base_out / "spot_1s" / sym
        d.mkdir(parents=True, exist_ok=True)
        for y, m in months:
            p = d / f"{sym}_spot_1s_{y}-{m:02d}.parquet"
            if p.exists():
                print(f"{sym} {y}-{m:02d}: exists, skipped", flush=True)
                continue
            raw = verified(f"{BASE}/monthly/klines/{sym}/1s/{sym}-1s-{y}-{m:02d}.zip")
            if raw is None:
                print(f"{sym} {y}-{m:02d}: not published", flush=True)
                continue
            df = parse_klines(raw)
            df.to_parquet(p, index=False)
            print(f"{sym} {y}-{m:02d}: {len(df):,} rows -> {p}", flush=True)
        if a.include_current:
            parts, day = [], today.replace(day=1)
            while day < today:
                raw = verified(f"{BASE}/daily/klines/{sym}/1s/{sym}-1s-{day:%Y-%m-%d}.zip")
                if raw is not None:
                    parts.append(parse_klines(raw))
                day += timedelta(days=1)
            if parts:
                df = pd.concat(parts, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
                p = d / f"{sym}_spot_1s_{today:%Y-%m}_partial.parquet"
                df.to_parquet(p, index=False)
                print(f"{sym} {today:%Y-%m} (partial): {len(df):,} rows -> {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
