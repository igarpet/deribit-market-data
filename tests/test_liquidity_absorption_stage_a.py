"""Synthetic tests for LA-A01. Run: python -m unittest tests.test_liquidity_absorption_stage_a"""
import unittest
import pandas as pd
from src.data_engineering.modeling.liquidity_absorption_stage_a import (
    aggregate_trade_chunk, choose_probe_days, compare_minute_flow, minute_quality,
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

    def test_probe_selection_has_no_outcome_dependence(self):
        x = ["a", "b", "c", "d", "e"]
        self.assertEqual(choose_probe_days(x), ["a", "c", "e"])


if __name__ == "__main__":
    unittest.main()
