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

    def test_invalid_params_day_falls_back_to_hours_then_instruments(self):
        day = date(2026, 10, 30)
        d0 = int(pd.Timestamp(day, tz="UTC").value // 10**6)
        perp = _trades(300, d0 + 1000, step=200_000, inst="BTC-PERPETUAL")            # whole day
        fut = _trades(50, d0 + 5 * 3_600_000, step=1000, inst="BTC-27NOV26", id0=5000)
        bad = _trades(1, d0 + 5 * 3_600_000 + 10, inst="BTC-WEIRD", id0=9000)          # breaks the currency call
        allt = sorted(perp + fut + bad, key=lambda x: (x["timestamp"], x["trade_id"]))
        err = "{'message': 'Invalid params', 'data': {'reason': 'wrong format', 'param': 'instrument_name'}, 'code': -32602}"

        class Cli:
            calls = 0

            def get(self, base, method, params):
                self.calls += 1
                if method == "get_instruments":
                    if params["expired"] == "true":
                        return [{"instrument_name": "BTC-PERPETUAL", "creation_timestamp": 0,
                                 "expiration_timestamp": 32503680000000},
                                {"instrument_name": "BTC-27NOV26", "creation_timestamp": 0,
                                 "expiration_timestamp": d0 + 28 * 86_400_000},
                                {"instrument_name": "BTC-WEIRD", "creation_timestamp": 0,
                                 "expiration_timestamp": d0 + 86_400_000},
                                {"instrument_name": "BTC-1OCT26", "creation_timestamp": 0,
                                 "expiration_timestamp": d0 - 86_400_000}]
                    return []
                a, b = params["start_timestamp"], params["end_timestamp"]
                if method == "get_last_trades_by_currency_and_time":
                    if any(a <= t["timestamp"] <= b for t in bad):
                        raise RuntimeError(f"{method} {params}: {err}")
                    sel = [t for t in allt if a <= t["timestamp"] <= b]
                else:
                    if params["instrument_name"] == "BTC-WEIRD":
                        raise RuntimeError(f"{method} {params}: {err}")
                    sel = [t for t in allt if a <= t["timestamp"] <= b and t["instrument_name"] == params["instrument_name"]]
                return {"trades": sel[: params["count"]], "has_more": len(sel) > params["count"]}

        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            dh.run_trades(Cli(), out, ["BTC"], ["future"], day, day)
            f = out / "trades" / "BTC" / "future" / "2026" / "2026-10-30.parquet"
            df = pd.read_parquet(f)
            self.assertEqual(len(df), len(perp) + len(fut))                 # everything but the rejected instrument
            self.assertEqual(df.trade_id.nunique(), len(df))
            side = f.with_name("2026-10-30.skipped.json")
            self.assertTrue(side.exists())
            self.assertIn("BTC-WEIRD", side.read_text())

    def test_invalid_params_on_options_is_not_silently_skipped(self):
        class Cli:
            calls = 0

            def get(self, base, method, params):
                raise RuntimeError("get_last_trades_by_currency_and_time {}: {'code': -32602, 'message': 'Invalid params'}")

        with self.assertRaises(dh.InvalidParams):
            dh.day_trades(Cli(), "BTC", "option", date(2026, 10, 30))


if __name__ == "__main__":
    unittest.main()
