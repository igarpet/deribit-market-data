import importlib.util
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dh", ROOT / "tools" / "download_deribit_history_trades.py")
dh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dh)


def _trades(n, t0, step=1, inst="BTC-27DEC24-100000-C", id0=0):
    return [{"trade_id": f"T{id0 + i}", "trade_seq": i, "instrument_name": inst, "timestamp": t0 + i * step,
             "direction": "buy", "tick_direction": 0, "price": 0.01, "amount": 1.0, "mark_price": 0.0101,
             "index_price": 95000.0, "iv": 55.0} for i in range(n)]


class FakeServer:
    """Answers get_last_trades_by_currency_and_time from an in-memory list (asc, inclusive bounds, count)."""
    def __init__(self, trades):
        self.t = sorted(trades, key=lambda x: (x["timestamp"], x["trade_id"]))
        self.calls = 0

    def fetch(self, a, b, count=dh.PAGE):
        self.calls += 1
        sel = [x for x in self.t if a <= x["timestamp"] <= b]
        return sel[:count], len(sel) > count


class TestHistoryDownloader(unittest.TestCase):
    def test_pagination_complete_and_deduplicated(self):
        tr = _trades(2500, 1_000, step=1) + _trades(30, 3_499, step=0, id0=10_000)   # 30 trades in one ms
        srv = FakeServer(tr)
        got = dh.paginate_day(srv.fetch, 0, 10_000)
        self.assertEqual(len(got), len(tr))
        self.assertEqual(len({g["trade_id"] for g in got}), len(tr))

    def test_crowded_millisecond_aborts_instead_of_silently_skipping_trades(self):
        tr = _trades(1500, 5_000, step=0) + _trades(5, 5_001, step=0, id0=9_000)
        srv = FakeServer(tr)
        with self.assertRaisesRegex(RuntimeError, "Ambiguous Deribit history pagination"):
            dh.paginate_day(srv.fetch, 0, 10_000)

    def test_nonadvancing_has_more_does_not_persist_incomplete_day(self):
        day = date(2026, 7, 1)
        day0 = int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)
        srv = FakeServer(_trades(1500, day0 + 500, step=0))
        class Cli:
            calls = 0
            def get(self, base, method, params):
                tr, more = srv.fetch(params["start_timestamp"], params["end_timestamp"], params["count"])
                return {"trades": tr, "has_more": more}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            with self.assertRaisesRegex(RuntimeError, "stopping without writing"):
                dh.run_trades(Cli(), out, ["BTC"], ["future"], day, day)
            target = out / "trades" / "BTC" / "future" / "2026" / "2026-07-01.parquet"
            self.assertFalse(target.exists())
            self.assertFalse(target.with_suffix(".tmp").exists())

    def test_parse_instruments(self):
        df = pd.DataFrame({"instrument_name": ["BTC-27DEC24-100000-C", "ETH-3JAN25-3500-P", "BTC-PERPETUAL",
                                               "ETH-28MAR25", "XRP_USDC-30OCT26-2d4-C", "SOL_USDC-PERPETUAL"]})
        p = dh.parse_instruments(df)
        self.assertEqual(list(p.inst_kind), ["option", "option", "perpetual", "future", "option", "perpetual"])
        self.assertEqual(p.expiry.iloc[0], pd.Timestamp("2024-12-27 08:00", tz="UTC"))
        self.assertEqual(p.strike.iloc[4], 2.4)
        self.assertEqual(p.option_type.iloc[1], "P")
        self.assertEqual(p.underlying.iloc[4], "XRP_USDC")

    def test_run_trades_writes_and_resumes(self):
        day0 = int(pd.Timestamp("2025-01-01", tz="UTC").value // 10**6)
        srv = FakeServer(_trades(1200, day0 + 1000, step=30_000))      # spans ~10 h of the first day
        calls = []

        class Cli:
            calls = 0

            def get(self, base, method, params):
                calls.append(params)
                t, more = srv.fetch(params["start_timestamp"], params["end_timestamp"], params["count"])
                return {"trades": t, "has_more": more}

        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            dh.run_trades(Cli(), out, ["BTC"], ["option"], date(2025, 1, 1), date(2025, 1, 2))
            f = out / "trades" / "BTC" / "option" / "2025" / "2025-01-01.parquet"
            df = pd.read_parquet(f)
            self.assertEqual(len(df), 1200)
            self.assertEqual(str(df.timestamp.dt.tz), "UTC")
            self.assertTrue((out / "trades" / "BTC" / "option" / "2025" / "2025-01-02.parquet").exists())
            n = len(calls)
            dh.run_trades(Cli(), out, ["BTC"], ["option"], date(2025, 1, 1), date(2025, 1, 2))
            self.assertEqual(len(calls), n)                         # resumed: nothing re-downloaded


if __name__ == "__main__":
    unittest.main()
