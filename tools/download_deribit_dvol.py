"""Download Deribit's DVOL implied-volatility index (30-day ATM implied volatility, annualised %) for BTC and ETH.

Source (public, no login): https://www.deribit.com/api/v2/public/get_volatility_index_data
    currency = BTC | ETH, resolution = 3600 (1 hour), start_timestamp / end_timestamp in ms.
DVOL starts in March 2021. Used by Turtle Breakout 06 to price ~38-day ATM options on historical trade dates.

Output: <DERIBIT_DATA_ROOT>\\history\\dvol\\<CURRENCY>_dvol_1h.csv   columns: t (UTC, bar start), open, high, low, close
Re-runs resume from the last stored hour.

Usage (PC, Windows CMD):
    python tools\\download_deribit_dvol.py
Optional: --currencies BTC ETH  --start 2021-03-01  --out D:\\path
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
HOUR_MS = 3600 * 1000
CHUNK_H = 900                     # hours per request (the API returns at most ~1000 points)


def _root() -> Path | None:
    v = os.getenv("DERIBIT_DATA_ROOT")
    if not v:
        env = Path(__file__).resolve().parents[1] / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("DERIBIT_DATA_ROOT="):
                    v = line.split("=", 1)[1].strip().strip('"').strip("'")
    return Path(v) if v else None


def _get(params, session=requests, tries=6):
    for k in range(tries):
        try:
            r = session.get(URL, params=params, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                raise RuntimeError(f"HTTP {r.status_code}")
            r.raise_for_status()
            js = r.json()
            if "error" in js:
                raise RuntimeError(js["error"])
            return js["result"]
        except Exception:                                          # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def download(currency: str, start_ms: int, end_ms: int, session=requests) -> pd.DataFrame:
    rows, t = [], start_ms
    while t < end_ms:
        e = min(t + CHUNK_H * HOUR_MS, end_ms)
        res = _get({"currency": currency, "start_timestamp": t, "end_timestamp": e, "resolution": "3600"}, session)
        rows.extend(res.get("data") or [])
        t = e
        time.sleep(0.15)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"])
    if df.empty:
        return pd.DataFrame(columns=["t", "open", "high", "low", "close"])
    df["t"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop(columns="ts").drop_duplicates("t").sort_values("t")[["t", "open", "high", "low", "close"]]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--currencies", nargs="+", default=["BTC", "ETH"])
    ap.add_argument("--start", default="2021-03-01")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    root = Path(a.out) if a.out else ((_root() / "history" / "dvol") if _root() else None)
    if root is None:
        sys.exit("Set DERIBIT_DATA_ROOT (or .env) or pass --out.")
    root.mkdir(parents=True, exist_ok=True)
    end_ms = int(pd.Timestamp.now("UTC").floor("h").value // 10 ** 6)
    for cur in a.currencies:
        path = root / f"{cur}_dvol_1h.csv"
        old = pd.read_csv(path, parse_dates=["t"]) if path.exists() else None
        start = pd.Timestamp(a.start, tz="UTC") if old is None or not len(old) else pd.Timestamp(old.t.max())
        df = download(cur, int(start.value // 10 ** 6), end_ms)
        if old is not None and len(old):
            old["t"] = pd.to_datetime(old["t"], utc=True)
            df = pd.concat([old, df], ignore_index=True).drop_duplicates("t", keep="last").sort_values("t")
        df.to_csv(path, index=False)
        print(f"{cur}: {len(df):,} hours, {df.t.min()} -> {df.t.max()} -> {path}")


if __name__ == "__main__":
    main()
