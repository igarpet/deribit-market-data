"""Turtle Breakout 17: download BTC-USD daily OHLC from Yahoo Finance (from 2014-09-17).

Output: <parent of DERIBIT_DATA_ROOT>/Stocks/tb17/btc_daily.parquet (date, ticker, open, high, low, close, volume).
Usage (PC, Windows CMD; needs yfinance):  python tools\\download_tb17_btc.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import download_tb14_stocks as base  # noqa: E402


def download(out: Path, start: str = "2014-01-01", end: str | None = None) -> Path:
    base.tickers = lambda: ["BTC-USD"]
    p = base.download(out, start, end, 1)
    target = out / "btc_daily.parquet"
    p.replace(target)
    print("renamed ->", target)
    return target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=(base.default_out().parent / "tb17") if base.default_out() else None)
    a = ap.parse_args()
    if a.out is None:
        ap.error("--out is required when DERIBIT_DATA_ROOT is not set")
    download(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
