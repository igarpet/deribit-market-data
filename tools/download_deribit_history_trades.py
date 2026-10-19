"""Download Deribit historical TRADES (free, public, no login) from history.deribit.com — options, futures and
perpetuals, including expired instruments — plus optional 1-minute perpetual OHLC.

Source (public API v2):
    https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time
        currency = BTC | ETH | USDC (USDC = linear SOL/XRP/... options and perps), kind = option | future
        ('future' includes perpetuals and dated futures). Up to 1000 trades per request, ascending by trade id.
    https://www.deribit.com/api/v2/public/get_tradingview_chart_data   (--perp-ohlc only; 1-minute OHLC)

Granularity: every trade (tick level). There are NO historical bid/ask quotes here; each trade carries Deribit's
mark price, index price and (options) implied volatility at the time of the trade.

Output (one parquet per currency / kind / UTC day; a day is written only once complete, so re-runs resume):
    <DERIBIT_DATA_ROOT>\\history\\trades\\<CURRENCY>\\<kind>\\<YYYY>\\<YYYY-MM-DD>.parquet
    <DERIBIT_DATA_ROOT>\\history\\perp_ohlc_1m\\<INSTRUMENT>\\<YYYY-MM>.parquet          (--perp-ohlc)
Columns: trade_id, trade_seq, instrument_name, timestamp (UTC), direction, tick_direction, price, amount, contracts,
mark_price, index_price, iv, liquidation, block_trade_id, combo_id, plus parsed kind / underlying / expiry / strike /
option_type (C/P) / is_perpetual.

Usage (PC, Windows CMD):
    python tools\\download_deribit_history_trades.py --currencies BTC ETH --kinds option --start 2024-01-01
    python tools\\download_deribit_history_trades.py --currencies BTC ETH --kinds future --start 2024-01-01
    python tools\\download_deribit_history_trades.py --perp-ohlc BTC-PERPETUAL ETH-PERPETUAL --start 2024-01-01
Options: --end YYYY-MM-DD (default yesterday), --rps 8 (requests per second), --out <dir>.
Volume warning: 'future' includes BTC-PERPETUAL, by far the most traded instrument (expect many hours for 2+ years);
run options first. Ctrl+C is safe: the day in progress is discarded and redone on the next run.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

HISTORY = "https://history.deribit.com/api/v2"
MAIN = "https://www.deribit.com/api/v2"
PAGE = 1000
COLS = ["trade_id", "trade_seq", "instrument_name", "timestamp", "direction", "tick_direction", "price", "amount",
        "contracts", "mark_price", "index_price", "iv", "liquidation", "block_trade_id", "combo_id"]
MONTHS = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV",
                                      "DEC"], start=1)}
OPT_RE = re.compile(r"^(?P<u>[A-Z0-9_]+)-(?P<d>\d{1,2})(?P<m>[A-Z]{3})(?P<y>\d{2})-(?P<k>[\dd.]+)-(?P<cp>[CP])$")
FUT_RE = re.compile(r"^(?P<u>[A-Z0-9_]+)-(?P<d>\d{1,2})(?P<m>[A-Z]{3})(?P<y>\d{2})$")


# ----------------------------------------------------------------------------- http
class Client:
    def __init__(self, rps: float = 8.0, retries: int = 8):
        if requests is None:
            raise SystemExit("pip install requests")
        self.s = requests.Session()
        self.min_gap = 1.0 / max(rps, 0.1)
        self.last = 0.0
        self.retries = retries
        self.calls = 0

    def get(self, base: str, method: str, params: dict) -> dict:
        url = f"{base}/public/{method}"
        delay = 1.0
        for k in range(self.retries):
            wait = self.min_gap - (time.monotonic() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()
            try:
                r = self.s.get(url, params=params, timeout=60)
                self.calls += 1
                js = r.json() if r.content else {}
            except Exception as e:  # network / JSON
                print(f"    network error ({e!r}); retry in {delay:.0f}s", flush=True)
                time.sleep(delay); delay = min(delay * 2, 60); continue
            err = js.get("error") if isinstance(js, dict) else None
            if r.status_code == 429 or (err and err.get("code") in (10028, 10040)):
                print(f"    rate-limited; backing off {delay:.0f}s", flush=True)
                time.sleep(delay); delay = min(delay * 2, 120); continue
            if err:
                raise RuntimeError(f"{method} {params}: {err}")
            if r.status_code >= 500:
                time.sleep(delay); delay = min(delay * 2, 60); continue
            return js["result"]
        raise RuntimeError(f"{method}: too many retries")


# ----------------------------------------------------------------------------- trades
def paginate_day(fetch, start_ms: int, end_ms: int) -> list[dict]:
    """All trades with start_ms <= timestamp <= end_ms. `fetch(start, end)` returns (trades asc, has_more).
    Next page starts at the last timestamp (inclusive) and duplicates are dropped by trade_id. If a whole page
    shares one millisecond (> PAGE trades in 1 ms), the start advances by 1 ms and a warning is printed."""
    out, seen = [], set()
    s = start_ms
    while True:
        trades, more = fetch(s, end_ms)
        new = [t for t in trades if t["trade_id"] not in seen]
        for t in new:
            seen.add(t["trade_id"])
        out.extend(new)
        if not trades or not more:
            break
        last = max(t["timestamp"] for t in trades)
        if last <= s:
            if len(trades) >= PAGE:
                print(f"    warning: >{PAGE} trades at {s} ms; some may be skipped", flush=True)
            s += 1
        else:
            s = last
        if s > end_ms:
            break
    return out


def parse_instruments(df: pd.DataFrame) -> pd.DataFrame:
    names = df["instrument_name"].astype(str)
    und, exp, strike, cp, perp, kind = [], [], [], [], [], []
    for n in names:
        m = OPT_RE.match(n)
        if m:
            und.append(m["u"]); kind.append("option"); perp.append(False); cp.append(m["cp"])
            strike.append(float(m["k"].replace("d", ".")))
            exp.append(pd.Timestamp(2000 + int(m["y"]), MONTHS[m["m"]], int(m["d"]), 8, tz="UTC"))
            continue
        und.append(n.split("-")[0]); strike.append(float("nan")); cp.append(None)
        if n.endswith("PERPETUAL"):
            kind.append("perpetual"); perp.append(True); exp.append(pd.NaT)
            continue
        f = FUT_RE.match(n)
        if f:
            kind.append("future"); perp.append(False)
            exp.append(pd.Timestamp(2000 + int(f["y"]), MONTHS[f["m"]], int(f["d"]), 8, tz="UTC"))
        else:
            kind.append("other"); perp.append(False); exp.append(pd.NaT)
    out = df.copy()
    out["underlying"] = und
    out["inst_kind"] = kind
    out["is_perpetual"] = perp
    out["expiry"] = pd.to_datetime(pd.Series(exp, index=df.index), utc=True)
    out["strike"] = strike
    out["option_type"] = cp
    return out


def to_frame(trades: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(trades)
    for c in COLS:
        if c not in df:
            df[c] = None
    df = df[COLS].copy()
    df["trade_id"] = df["trade_id"].astype(str)
    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms", utc=True)
    for c in ("price", "amount", "contracts", "mark_price", "index_price", "iv", "trade_seq", "tick_direction"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("liquidation", "block_trade_id", "combo_id", "direction"):
        df[c] = df[c].astype("string")
    return parse_instruments(df).sort_values(["timestamp", "trade_id"]).reset_index(drop=True)


def day_trades(cli: Client, currency: str, kind: str, d: date) -> pd.DataFrame:
    s = int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)
    e = s + 86_400_000 - 1

    def fetch(a, b):
        r = cli.get(HISTORY, "get_last_trades_by_currency_and_time",
                    {"currency": currency, "kind": kind, "start_timestamp": a, "end_timestamp": b,
                     "count": PAGE, "sorting": "asc"})
        return r.get("trades", []), bool(r.get("has_more", False))

    tr = paginate_day(fetch, s, e)
    return to_frame(tr) if tr else pd.DataFrame(columns=COLS)


def run_trades(cli: Client, out: Path, currencies, kinds, start: date, end: date) -> None:
    for cur in currencies:
        for kind in kinds:
            d = start
            t0 = time.time()
            while d <= end:
                f = out / "trades" / cur / kind / f"{d:%Y}" / f"{d:%Y-%m-%d}.parquet"
                if f.exists():
                    d += timedelta(days=1); continue
                f.parent.mkdir(parents=True, exist_ok=True)
                c0 = cli.calls
                df = day_trades(cli, cur, kind, d)
                tmp = f.with_suffix(".tmp")
                df.to_parquet(tmp, index=False)
                os.replace(tmp, f)
                print(f"{cur} {kind} {d}: {len(df):>8,} trades | {cli.calls - c0} requests | "
                      f"{time.time() - t0:,.0f}s elapsed", flush=True)
                d += timedelta(days=1)


# ----------------------------------------------------------------------------- perp OHLC
def run_ohlc(cli: Client, out: Path, instruments, start: date, end: date) -> None:
    for inst in instruments:
        m = date(start.year, start.month, 1)
        while m <= end:
            nxt = date(m.year + (m.month == 12), m.month % 12 + 1, 1)
            f = out / "perp_ohlc_1m" / inst / f"{m:%Y-%m}.parquet"
            complete = nxt <= end + timedelta(days=1)
            if f.exists() and complete:
                m = nxt; continue
            f.parent.mkdir(parents=True, exist_ok=True)
            s = int(datetime(m.year, m.month, m.day, tzinfo=timezone.utc).timestamp() * 1000)
            e_d = min(nxt, end + timedelta(days=1))
            e = int(datetime(e_d.year, e_d.month, e_d.day, tzinfo=timezone.utc).timestamp() * 1000) - 1
            parts = []
            a = s
            while a <= e:                                       # chunks of 5000 minutes
                b = min(a + 5000 * 60_000 - 1, e)
                r = cli.get(MAIN, "get_tradingview_chart_data",
                            {"instrument_name": inst, "start_timestamp": a, "end_timestamp": b, "resolution": "1"})
                if r.get("status") == "ok" and r.get("ticks"):
                    parts.append(pd.DataFrame({"open_time": pd.to_datetime(r["ticks"], unit="ms", utc=True),
                                               "open": r["open"], "high": r["high"], "low": r["low"],
                                               "close": r["close"], "volume": r.get("volume"), "cost": r.get("cost")}))
                a = b + 1
            df = (pd.concat(parts).drop_duplicates("open_time").sort_values("open_time")
                  if parts else pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume", "cost"]))
            df.to_parquet(f, index=False)
            print(f"{inst} {m:%Y-%m}: {len(df):,} minutes", flush=True)
            m = nxt


# ----------------------------------------------------------------------------- cli
def default_out() -> Path | None:
    root = os.getenv("DERIBIT_DATA_ROOT")
    if not root:
        env = Path(__file__).resolve().parents[1] / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("DERIBIT_DATA_ROOT="):
                    root = line.split("=", 1)[1].strip().strip('"').strip("'")
    return Path(root) / "history" if root else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=default_out())
    ap.add_argument("--currencies", nargs="+", default=["BTC", "ETH"])
    ap.add_argument("--kinds", nargs="+", default=[], choices=["option", "future"])
    ap.add_argument("--perp-ohlc", nargs="*", default=None, metavar="INSTRUMENT")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default=str(date.today() - timedelta(days=1)))
    ap.add_argument("--rps", type=float, default=8.0)
    a = ap.parse_args(argv)
    if a.out is None:
        ap.error("--out is required when DERIBIT_DATA_ROOT is not set")
    if not a.kinds and a.perp_ohlc is None:
        ap.error("nothing to do: pass --kinds option/future and/or --perp-ohlc")
    s, e = date.fromisoformat(a.start), date.fromisoformat(a.end)
    cli = Client(a.rps)
    print(f"output {a.out} | {s} -> {e} | {a.rps} req/s", flush=True)
    if a.kinds:
        run_trades(cli, a.out, a.currencies, a.kinds, s, e)
    if a.perp_ohlc is not None:
        run_ohlc(cli, a.out, a.perp_ohlc or ["BTC-PERPETUAL", "ETH-PERPETUAL"], s, e)
    print(f"done | {cli.calls:,} requests", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
