"""Download the Binance data for Pairs fOU 02 (construction-tied spreads). Free public data, data.binance.vision.

For each asset A in the PF02 universe (or --assets):
    spot 1m klines           -> <OUT>/basis/spot/<A>USDT_spot_1m.parquet
    USD-M perp 1m klines     -> <OUT>/<A>USDT_um_1m.parquet          (skipped if it already exists)
    USD-M funding rates      -> <OUT>/basis/funding/<A>USDT_funding.parquet
    USD-M quarterly 1m       -> <OUT>/basis/quarterly/<A>USDT_<YYMMDD>_1m.parquet  (only where the contract exists)
Monthly files for complete months, daily files for the current month. Each zip is SHA256-verified.
A symbol/month that does not exist is reported and skipped. Default OUT: BINANCE_UM_DIR or <parent of DERIBIT_DATA_ROOT>/Binance.

Usage (PC, Windows CMD):
    python tools\\download_binance_basis_data.py --months 26
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

BASE = "https://data.binance.vision/data"
KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
         "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
KEEP = ["open_time", "close", "volume", "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume"]
UNIVERSE = ("BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "LINK", "DOT", "LTC", "BCH", "UNI", "NEAR", "APT",
            "ARB", "OP", "SUI", "FIL", "ATOM", "ETC", "INJ", "AAVE", "TON", "WIF", "HBAR", "XLM", "SEI", "TIA",
            "ENA", "WLD")


def fetch(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
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
    df["open_time"] = pd.to_datetime(t, unit="us" if t.max() > 10 ** 14 else "ms", utc=True)   # spot uses us since 2025
    return df[KEEP]


def parse_funding(raw: bytes) -> pd.DataFrame:
    has_header = not raw.split(b"\n", 1)[0][:1].isdigit()
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)
    if not has_header:
        df.columns = ["calc_time", "funding_interval_hours", "last_funding_rate"][:df.shape[1]]
    return df


def months_back(n: int, today: date) -> list[tuple[int, int]]:
    y, m, out = today.year, today.month, []
    for _ in range(n):
        m -= 1
        if m == 0:
            y, m = y - 1, 12
        out.append((y, m))
    return sorted(out)


def current_month_days(today: date) -> list[date]:
    d, out = today.replace(day=1), []
    while d < today:
        out.append(d); d += timedelta(days=1)
    return out


def klines(market: str, sym: str, months, days) -> pd.DataFrame | None:
    root = f"{BASE}/spot" if market == "spot" else f"{BASE}/futures/um"
    parts = []
    for y, m in months:
        raw = verified(f"{root}/monthly/klines/{sym}/1m/{sym}-1m-{y}-{m:02d}.zip")
        if raw is not None:
            parts.append(parse_klines(raw))
    for d in days:
        raw = verified(f"{root}/daily/klines/{sym}/1m/{sym}-1m-{d:%Y-%m-%d}.zip")
        if raw is not None:
            parts.append(parse_klines(raw))
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def last_fridays(start: date, end: date) -> list[date]:
    out = []
    for y in range(start.year, end.year + 2):
        for m in (3, 6, 9, 12):
            d = (date(y + (m == 12), (m % 12) + 1, 1) - timedelta(days=1))
            while d.weekday() != 4:
                d -= timedelta(days=1)
            if start <= d <= end + timedelta(days=200):
                out.append(d)
    return out


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
    ap.add_argument("--months", type=int, default=26)
    ap.add_argument("--assets", nargs="+", default=list(UNIVERSE))
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip", nargs="*", default=[], choices=["spot", "perp", "funding", "quarterly"])
    a = ap.parse_args()
    out = Path(a.out) if a.out else default_out()
    if out is None:
        print("Set --out, BINANCE_UM_DIR or DERIBIT_DATA_ROOT"); return 1
    for sub in ("basis/spot", "basis/funding", "basis/quarterly"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    today = date.today()
    months, days = months_back(a.months, today), current_month_days(today)
    start = date(*months[0], 1)
    for A in a.assets:
        sym = f"{A}USDT"
        if "spot" not in a.skip:
            p = out / "basis" / "spot" / f"{sym}_spot_1m.parquet"
            if not p.exists():
                df = klines("spot", sym, months, days)
                print(f"{sym} spot: {'missing' if df is None else f'{len(df):,} rows'}", flush=True)
                if df is not None: df.to_parquet(p, index=False)
        if "perp" not in a.skip:
            p = out / f"{sym}_um_1m.parquet"
            if not p.exists():
                df = klines("um", sym, months, days)
                print(f"{sym} perp: {'missing' if df is None else f'{len(df):,} rows'}", flush=True)
                if df is not None: df.to_parquet(p, index=False)
        if "funding" not in a.skip:
            p = out / "basis" / "funding" / f"{sym}_funding.parquet"
            if not p.exists():
                parts = [parse_funding(r) for y, m in months
                         if (r := verified(f"{BASE}/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{y}-{m:02d}.zip")) is not None]
                print(f"{sym} funding: {len(parts)} months", flush=True)
                if parts: pd.concat(parts, ignore_index=True).to_parquet(p, index=False)
        if "quarterly" not in a.skip:
            for exp in last_fridays(start, today):
                q = f"{sym}_{exp:%y%m%d}"
                p = out / "basis" / "quarterly" / f"{q}_1m.parquet"
                if p.exists():
                    continue
                life = [(y, m) for (y, m) in months_back(a.months + 8, today)
                        if date(y, m, 1) >= (exp - timedelta(days=200)).replace(day=1) and date(y, m, 1) <= exp]
                probe = verified(f"{BASE}/futures/um/monthly/klines/{q}/1m/{q}-1m-{exp.year}-{exp.month:02d}.zip") \
                    if (exp.year, exp.month) < (today.year, today.month) else b"live"
                if probe is None:
                    continue                                  # contract never listed for this asset
                df = klines("um", q, [ym for ym in life if (ym[0], ym[1]) < (today.year, today.month)],
                            days if exp >= today.replace(day=1) else [])
                print(f"{q}: {'missing' if df is None else f'{len(df):,} rows'}", flush=True)
                if df is not None: df.to_parquet(p, index=False)
    print("Output:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
