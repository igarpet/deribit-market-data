"""
Download long 5-minute price history for Volatility Dynamics Module 10c.

Primary source: Binance spot <ASSET>USDT klines. These are liquid
constituents of the Deribit index and the best long proxy for it.
Fallback: Deribit perpetual trade candles
(public/get_tradingview_chart_data). Used only for an asset with no Binance
spot pair, or when --source deribit is given.

Output: one gzip CSV per asset under OUT_DIR:
    <ASSET>.csv.gz   columns: state_time (bar END, UTC), source, symbol, close, volume
plus OUT_DIR/manifest.json with the coverage of each asset.

Re-running resumes: an existing file is extended from its last bar.

Usage (PC, Windows CMD):
    python tools\\download_price_history.py --start 2024-09-01 --end 2026-10-11
Optional:
    --assets BTC ETH ...   --source auto|binance|deribit   --out D:\\path
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ASSETS = ("AVAX", "BTC", "ETH", "HYPE", "SOL", "TRX", "XRP")
BINANCE = "https://api.binance.com/api/v3/klines"
DERIBIT = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
DERIBIT_PERPS = {
    "BTC": ("BTC-PERPETUAL", "BTC_USDC-PERPETUAL"),
    "ETH": ("ETH-PERPETUAL", "ETH_USDC-PERPETUAL"),
}
BAR_MS = 5 * 60 * 1000


def _data_root() -> Path | None:
    v = os.getenv("DERIBIT_DATA_ROOT")
    if not v:
        for env in (Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env",
                    Path(__file__).resolve().parents[1] / "notebooks" / "Volatility_Dynamics" / ".env"):
            if env.exists():
                for line in env.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("DERIBIT_DATA_ROOT="):
                        v = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
            if v:
                break
    return Path(v) if v else None


def _get(url, params, tries=6):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                raise RuntimeError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()
        except Exception as e:                                    # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(2 ** k)
    return None


def binance_history(asset: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    sym = f"{asset}USDT"
    rows, t = [], start_ms
    while t < end_ms:
        data = _get(BINANCE, {"symbol": sym, "interval": "5m", "startTime": t, "endTime": end_ms, "limit": 1000})
        if isinstance(data, dict):                                # error payload, e.g. invalid symbol
            raise RuntimeError(f"Binance {sym}: {data}")
        if not data:
            break
        rows.extend((int(k[0]) + BAR_MS, float(k[4]), float(k[5])) for k in data)
        nxt = int(data[-1][0]) + BAR_MS
        if nxt <= t:
            break
        t = nxt
        time.sleep(0.12)
    df = pd.DataFrame(rows, columns=["end_ms", "close", "volume"])
    df["source"], df["symbol"] = "binance_spot", sym
    return df


def deribit_history(asset: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    last_err = None
    for inst in DERIBIT_PERPS.get(asset, (f"{asset}_USDC-PERPETUAL",)):
        rows, t = [], start_ms
        chunk = 3 * 24 * 3600 * 1000
        try:
            while t < end_ms:
                e = min(t + chunk, end_ms)
                res = _get(DERIBIT, {"instrument_name": inst, "start_timestamp": t, "end_timestamp": e, "resolution": "5"})
                if "error" in res:
                    raise RuntimeError(res["error"])
                r = res["result"]
                if r.get("status") == "ok" and r.get("ticks"):
                    rows.extend((int(tk) + BAR_MS, float(c), float(v))
                                for tk, c, v in zip(r["ticks"], r["close"], r["volume"]))
                t = e
                time.sleep(0.12)
        except Exception as ex:                                   # noqa: BLE001
            last_err = ex
            continue
        if rows:
            df = pd.DataFrame(rows, columns=["end_ms", "close", "volume"]).drop_duplicates("end_ms")
            df["source"], df["symbol"] = "deribit_perp", inst
            return df
    raise RuntimeError(f"Deribit perpetual history unavailable for {asset}: {last_err}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-09-01")
    ap.add_argument("--end", default=pd.Timestamp.now("UTC").strftime("%Y-%m-%d"))
    ap.add_argument("--assets", nargs="*", default=list(ASSETS))
    ap.add_argument("--source", choices=("auto", "binance", "deribit"), default="auto")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    root = Path(a.out) if a.out else (_data_root() / "history" / "price_5m" if _data_root() else None)
    if root is None:
        sys.exit("Set DERIBIT_DATA_ROOT (or .env) or pass --out.")
    root.mkdir(parents=True, exist_ok=True)
    s_ms = int(pd.Timestamp(a.start, tz="UTC").value // 10 ** 6)
    e_ms = int(pd.Timestamp(a.end, tz="UTC").value // 10 ** 6)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for asset in a.assets:
        path = root / f"{asset}.csv.gz"
        old = pd.read_csv(path) if path.exists() else None
        start = s_ms
        if old is not None and len(old):
            start = int(pd.Timestamp(old["state_time"].max()).value // 10 ** 6)   # resume (end time -> next open)
        t0 = time.time()
        try:
            if a.source == "deribit":
                df = deribit_history(asset, start, e_ms)
            else:
                try:
                    df = binance_history(asset, start, e_ms)
                    if df.empty and a.source == "auto":
                        raise RuntimeError("empty")
                except Exception as ex:                           # noqa: BLE001
                    if a.source == "binance":
                        raise
                    print(f"{asset}: Binance unavailable ({ex}); falling back to Deribit perpetual")
                    df = deribit_history(asset, start, e_ms)
        except Exception as ex:                                   # noqa: BLE001
            print(f"{asset}: FAILED {ex}")
            manifest[asset] = {"status": "failed", "error": str(ex)}
            continue
        df["state_time"] = pd.to_datetime(df["end_ms"], unit="ms", utc=True)
        df = df[["state_time", "source", "symbol", "close", "volume"]]
        if old is not None and len(old):
            old["state_time"] = pd.to_datetime(old["state_time"], utc=True)
            if set(old["source"]) != set(df["source"]) and len(df):
                print(f"{asset}: WARNING source changed {set(old['source'])} -> {set(df['source'])}")
            df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates("state_time", keep="last").sort_values("state_time")
        df.to_csv(path, index=False, compression="gzip")
        manifest[asset] = {"status": "ok", "source": sorted(set(df["source"])), "symbol": sorted(set(df["symbol"])),
                           "first": str(df["state_time"].min()), "last": str(df["state_time"].max()),
                           "n_bars": int(len(df))}
        print(f"{asset}: {manifest[asset]['source']} {manifest[asset]['first']} -> {manifest[asset]['last']} "
              f"({len(df):,} bars, {time.time() - t0:.0f}s)")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()
