"""Synthetic tests for LA-A01. Run: python -m unittest tests.test_liquidity_absorption_stage_a"""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from src.data_engineering.modeling.liquidity_absorption_stage_a import (
    aggregate_trade_chunk, choose_probe_days, compare_minute_flow, minute_quality, parse_kline_open_time, run_audit,
)


class TestStageA01(unittest.TestCase):
    def test_minute_quality_counts_gaps_and_duplicates(self):
        x = pd.DataFrame({"open_time": pd.to_datetime(["2026-07-01 00:00Z", "2026-07-01 00:02Z", "2026-07-01 00:02Z"]),
                          "open": [100, 101, 101], "high": [102, 103, 103], "low": [99, 100, 100],
                          "close": [101, 102, 102], "volume": [2., 3., 3.], "quote_volume": [200., 300., 300.],
                          "trades": [2, 3, 3], "taker_buy_volume": [1, 2, 2],
                          "taker_buy_quote_volume": [100., 200., 200.]})
        r = minute_quality(x, "BTCUSDT", "perp")
        self.assertEqual(r["missing_minutes_inside_span"], 1)
        self.assertEqual(r["duplicate_minutes"], 1)
        self.assertEqual(r["bad_price_rows"], 0)

    def test_aggregate_trade_side_and_volume(self):
        x = pd.DataFrame({"time_ms": [60_000, 60_010, 120_000], "price": [100., 100., 101.],
                          "qty": [2., 3., 1.], "is_buyer_maker": [True, False, False]})
        minute, bins = aggregate_trade_chunk(x)
        self.assertEqual(minute["quote"].sum(), 601.)
        self.assertEqual(minute["sell_quote"].sum(), 200.)
        self.assertEqual(minute["buy_quote"].sum(), 401.)
        self.assertEqual(bins["trade_rows"].sum(), 3)

    def test_kline_vs_aggtrade_consistency(self):
        x = pd.DataFrame({"open_time": pd.to_datetime(["2026-07-01T00:01:00Z"]),
                          "quote_volume": [500.], "taker_buy_quote_volume": [300.]})
        y = pd.DataFrame({"minute_ms": [int(pd.Timestamp("2026-07-01T00:01Z").timestamp()*1000)],
                          "quote": [500.], "buy_quote": [300.], "sell_quote": [200.]})
        r = compare_minute_flow(x, y, "2026-07-01", "BTCUSDT")
        self.assertEqual(r["matched_minutes"], 1)
        self.assertAlmostEqual(r["p95_delta_error_as_quote_fraction"], 0.)

    def test_numeric_binance_millisecond_epoch_preserves_unique_minutes(self):
        times = pd.to_datetime(["2026-07-01 00:00Z", "2026-07-01 00:01Z"])
        ms = (times.astype("int64") // 1_000_000).astype("int64")
        k = pd.DataFrame({"open_time": ms, "quote_volume": [500., 600.],
                          "taker_buy_quote_volume": [300., 250.]})
        audit_input = k.assign(open=[100., 101.], high=[101., 102.], low=[99., 100.],
                               close=[100., 101.], volume=[5., 6.], trades=[2, 3],
                               taker_buy_volume=[3., 2.5])
        q = minute_quality(audit_input, "BTCUSDT", "perp")
        self.assertEqual(q["duplicate_minutes"], 0)
        self.assertEqual(q["missing_minutes_inside_span"], 0)
        self.assertEqual(q["first_utc"], "2026-07-01 00:00:00+00:00")
        traded = pd.DataFrame({"minute_ms": ms, "quote": [500., 600.],
                               "buy_quote": [300., 250.], "sell_quote": [200., 350.]})
        result = compare_minute_flow(k, traded, "2026-07-01", "BTCUSDT")
        self.assertEqual(result["matched_minutes"], 2)
        self.assertAlmostEqual(result["median_quote_relative_error"], 0.)

    def test_microsecond_and_datetime_epoch_are_consistent(self):
        ms = pd.Series([1782864000000, 1782864060000])
        expected = pd.to_datetime(ms, unit="ms", utc=True)
        for stamp in (ms * 1000, ms * 1_000_000, expected):
            pd.testing.assert_series_equal(parse_kline_open_time(pd.Series(stamp)), expected)

    def test_duplicate_minute_data_reports_error_without_silent_deduplication(self):
        k = pd.DataFrame({"open_time": [1782864000000, 1782864000000],
                          "quote_volume": [500., 500.], "taker_buy_quote_volume": [300., 300.]})
        traded = pd.DataFrame({"minute_ms": [1782864000000], "quote": [500.],
                               "buy_quote": [300.], "sell_quote": [200.]})
        result = compare_minute_flow(k, traded, "2026-07-01", "BTCUSDT")
        self.assertEqual(result["status"], "DUPLICATE_KLINE_MINUTES")
        self.assertEqual(result["duplicate_kline_minutes"], 1)

    def test_millisecond_resolution_datetime_preserves_one_minute_keys(self):
        # pandas 2.x/3.x can preserve datetime64[ms] (not datetime64[ns]);
        # astype(int64)//1_000_000 would collapse ~17 minutes into one key.
        stamps = pd.to_datetime([1782864000000 + n * 60000 for n in range(1440)], unit="ms", utc=True)
        low_resolution = pd.Series(stamps).dt.as_unit("ms")
        k = pd.DataFrame({"open_time": low_resolution,
                          "quote_volume": [500.] * 1440,
                          "taker_buy_quote_volume": [300.] * 1440})
        m = low_resolution.dt.as_unit("ms").astype("int64")
        traded = pd.DataFrame({"minute_ms": m, "quote": [500.] * 1440,
                               "buy_quote": [300.] * 1440, "sell_quote": [200.] * 1440})
        result = compare_minute_flow(k, traded, "2026-07-01", "BTCUSDT")
        self.assertEqual(result["status"], "PRESENT")
        self.assertEqual(result["matched_minutes"], 1440)

    def test_spot_close_only_schema_matches_downloader(self):
        ms = [1782864000000, 1782864060000]
        compact = pd.DataFrame({"open_time": ms, "close": [100., 101.],
                                "volume": [5., 6.], "quote_volume": [500., 600.],
                                "trades": [2, 3], "taker_buy_volume": [3., 2.],
                                "taker_buy_quote_volume": [300., 200.]})
        quality = minute_quality(compact, "BTCUSDT", "spot")
        self.assertEqual(quality["status"], "PRESENT")
        self.assertEqual(quality["missing_minutes_inside_span"], 0)
        self.assertEqual(quality["bad_price_rows"], 0)

    def test_audit_readiness_does_not_ignore_flow_join(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            symbol = "BTCUSDT"
            perp = pd.DataFrame({"open_time": [1782864000000, 1782864060000],
                                 "open": [100., 101.], "high": [102., 103.],
                                 "low": [99., 100.], "close": [101., 102.],
                                 "volume": [5., 6.], "quote_volume": [500., 600.],
                                 "trades": [10, 12], "taker_buy_volume": [3., 4.],
                                 "taker_buy_quote_volume": [300., 400.]})
            (root / f"{symbol}_um_1m.parquet").touch()
            f = root / "aggTrades" / symbol / f"{symbol}-aggTrades-2026-07-01.parquet"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.touch()
            minutes = pd.DataFrame({"minute_ms": [1782864000000, 1782864060000],
                                    "quote": [500., 600.], "buy_quote": [300., 400.],
                                    "sell_quote": [200., 200.]})
            info = {"status": "PRESENT", "invalid_trade_rows": 0,
                    "timestamp_descents": 0, "minute_count": 2}
            bins = pd.DataFrame({"log_price_bin": [1], "quote": [100.], "trade_rows": [1]})
            with patch("src.data_engineering.modeling.liquidity_absorption_stage_a.probe_trades",
                       return_value=(info, minutes, bins)), patch(
                           "src.data_engineering.modeling.liquidity_absorption_stage_a.pd.read_parquet",
                           return_value=perp):
                ok = run_audit(root, root / "out", symbols=(symbol,))
                self.assertTrue(ok["ready_for_signal_design"])
                # Simulate the former 1,353-duplicate key error: the audit
                # must not signal readiness if reconciliation fails.
                with patch("src.data_engineering.modeling.liquidity_absorption_stage_a.compare_minute_flow",
                           return_value={"status": "DUPLICATE_KLINE_MINUTES", "matched_minutes": 0}):
                    blocked = run_audit(root, root / "out", symbols=(symbol,))
                    self.assertFalse(blocked["ready_for_signal_design"])

    def test_probe_selection_has_no_outcome_dependence(self):
        x = ["a", "b", "c", "d", "e"]
        self.assertEqual(choose_probe_days(x), ["a", "c", "e"])


if __name__ == "__main__":
    unittest.main()
