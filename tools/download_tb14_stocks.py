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
        x = x.rename(columns=str.lower)
        if not {"open", "high", "low", "close"} <= set(x.columns):
            continue
        x = x[[c for c in ("open", "high", "low", "close", "volume") if c in x.columns]].dropna(subset=["close"])
        if len(x):
            rows.append(x.assign(ticker=t).rename_axis("date").reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def download(out: Path, start: str = "1999-10-01", end: str | None = None, batch: int = 50) -> Path:
    """Download all tickers and write <out>/stocks_daily.parquet; returns the path. Raises if nothing came back."""
    try:
        import yfinance  # noqa: F401
    except ImportError as e:
        raise ImportError("yfinance is not installed in this environment: run  pip install yfinance") from e
    out.mkdir(parents=True, exist_ok=True)
    end = end or str(pd.Timestamp.today().date())
    T = tickers()
    print(f"{len(T)} tickers, {start} -> {end}", flush=True)
    parts = []
    for i in range(0, len(T), batch):
        b = T[i:i + batch]
        df = fetch(b, start, end)
        print(f"  {i + len(b)}/{len(T)}: {df.ticker.nunique() if len(df) else 0} tickers with data", flush=True)
        if len(df):
            parts.append(df)
    if not parts:
        raise RuntimeError("Yahoo Finance returned no data (network, rate limit or yfinance version); nothing saved")
    D = pd.concat(parts, ignore_index=True)
    D["date"] = pd.to_datetime(D["date"]).dt.tz_localize(None)
    target = out / "stocks_daily.parquet"
    D.to_parquet(target, index=False)
    rep = D.groupby("ticker").agg(rows=("close", "size"), first=("date", "min"), last=("date", "max")).reindex(T)
    rep.to_csv(out / "download_report.csv")
    print(f"saved {len(D):,} rows, {D.ticker.nunique()} tickers; missing: {int(rep['rows'].isna().sum())} -> {target}", flush=True)
    return target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=default_out())
    ap.add_argument("--start", default="1999-10-01")
    ap.add_argument("--end", default=None, help="exclusive; default = today")
    ap.add_argument("--batch", type=int, default=50)
    a = ap.parse_args()
    if a.out is None:
        ap.error("--out is required when DERIBIT_DATA_ROOT is not set")
    download(a.out, a.start, a.end, a.batch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
