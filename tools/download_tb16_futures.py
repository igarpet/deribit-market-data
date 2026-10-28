"""Turtle Breakout 16: download daily OHLC from Yahoo Finance for the futures and futures-proxy ETF universe.

Tickers: notebooks/Turtle_Breakout/16/16_universe.csv (continuous front-month futures '=F' and ETFs).
Output:  <parent of DERIBIT_DATA_ROOT>/Stocks/tb16/futures_daily.parquet (date, ticker, open, high, low, close, volume)
         and download_report.csv. ETFs are adjusted for splits and dividends; Yahoo's '=F' series are the
         front contract as quoted (not back-adjusted at rolls).

Usage (PC, Windows CMD; needs yfinance):  python tools\\download_tb16_futures.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import download_tb14_stocks as base  # noqa: E402

LIST = ROOT / "notebooks" / "Turtle_Breakout" / "16" / "16_universe.csv"


def tickers() -> list[str]:
    return pd.read_csv(LIST)["ticker"].astype(str).tolist()


def download(out: Path, start: str = "1999-10-01", end: str | None = None, batch: int = 20) -> Path:
    base.tickers = tickers                                   # same downloader, different list and file name
    p = base.download(out, start, end, batch)
    target = out / "futures_daily.parquet"
    p.replace(target)
    print("renamed ->", target)
    return target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=(base.default_out().parent / "tb16") if base.default_out() else None)
    ap.add_argument("--start", default="1999-10-01")
    a = ap.parse_args()
    if a.out is None:
        ap.error("--out is required when DERIBIT_DATA_ROOT is not set")
    download(a.out, a.start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
