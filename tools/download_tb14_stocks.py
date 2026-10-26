"""Turtle Breakout 14: download daily adjusted OHLC from Yahoo Finance for the thesis S&P 500 list and index ETFs.

Tickers: notebooks/Turtle_Breakout/14/14_constituents_thesis.csv (column Symbol; '.' -> '-', as in the thesis)
         + SPY, QQQ, DIA, IWM.
Output:  <BINANCE_UM_DIR parent or DERIBIT_DATA_ROOT parent>/Stocks/tb14/stocks_daily.parquet
         (long format: date, ticker, open, high, low, close, volume; prices adjusted for splits and dividends)
         and download_report.csv (rows per ticker, first and last date).

Usage (PC, Windows CMD; needs `pip install yfinance` in the environment):
    python tools\\download_tb14_stocks.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "notebooks" / "Turtle_Breakout" / "14" / "14_constituents_thesis.csv"
ETFS = ["SPY", "QQQ", "DIA", "IWM"]


def default_out() -> Path | None:
    root = os.getenv("DERIBIT_DATA_ROOT")
    if not root:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("DERIBIT_DATA_ROOT="):
                    root = line.split("=", 1)[1].strip().strip('"').strip("'")
    return Path(root).parent / "Stocks" / "tb14" if root else None


def tickers() -> list[str]:
    c = pd.read_csv(LIST)
    return sorted(set(c["Symbol"].astype(str).str.replace(".", "-", regex=False))) + ETFS


def fetch(batch: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    for k in range(3):
        try:
            d = yf.download(batch, start=start, end=end, auto_adjust=True, actions=False, progress=False,
                            group_by="ticker", threads=True)
            break
        except Exception as e:                                          # noqa: BLE001
            print("  retry after error:", e); time.sleep(5 * (k + 1))
    else:
        return pd.DataFrame()
    rows = []
    for t in batch:
        if isinstance(d.columns, pd.MultiIndex):
            if t not in d.columns.get_level_values(0):
                continue
            x = d[t]
        else:
            x = d
        x = x.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        if len(x):
            rows.append(x.assign(ticker=t).rename_axis("date").reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=default_out())
    ap.add_argument("--start", default="1999-10-01")
    ap.add_argument("--end", default=None, help="exclusive; default = today")
    ap.add_argument("--batch", type=int, default=50)
    a = ap.parse_args()
    if a.out is None:
        ap.error("--out is required when DERIBIT_DATA_ROOT is not set")
    a.out.mkdir(parents=True, exist_ok=True)
    end = a.end or str(pd.Timestamp.today().date())
    T = tickers()
    print(f"{len(T)} tickers, {a.start} -> {end}")
    parts = []
    for i in range(0, len(T), a.batch):
        b = T[i:i + a.batch]
        df = fetch(b, a.start, end)
        print(f"  {i + len(b)}/{len(T)}: {df.ticker.nunique() if len(df) else 0} tickers with data", flush=True)
        parts.append(df)
    D = pd.concat(parts, ignore_index=True)
    D["date"] = pd.to_datetime(D["date"]).dt.tz_localize(None)
    D.to_parquet(a.out / "stocks_daily.parquet", index=False)
    rep = D.groupby("ticker").agg(rows=("close", "size"), first=("date", "min"), last=("date", "max")).reindex(T)
    rep.to_csv(a.out / "download_report.csv")
    print(f"saved {len(D):,} rows, {D.ticker.nunique()} tickers; missing: {rep['rows'].isna().sum()} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
