"""Synthetic tests for VRP 02 (vrp_long_straddle).

Run from repo root: python -m unittest discover -s tests -p "test_vrp_long_straddle.py" -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_engineering.modeling import vrp_long_straddle as v

EMPTY_DELIVERY = pd.DataFrame(columns=["index_name", "session", "delivery_price"])


class LongStraddleTests(unittest.TestCase):
    def test_selftest(self):
        self.assertTrue(v.selftest()["ok"])

    def test_payoff_arithmetic(self):
        x = pd.DataFrame({"S_T": [95.0, 100.0], "K": [100.0, 100.0], "F": [100.0, 100.0],
                          "inverse": [False, False], "ask_usd": [4.0, 4.0], "fee_usd": [1.0, 1.0],
                          "mid_usd": [3.5, 3.5], "bid_usd": [3.0, 3.0], "fee_bid_usd": [0.0, 0.0],
                          "iv_straddle_mid": [0.5, 0.5], "T": [0.001, 0.001]})
        o = v.straddle_pnl(x)
        np.testing.assert_allclose(o["R_net"], [5 / 5 - 1, 0 / 5 - 1])
        np.testing.assert_allclose(o["R_gross"], [5 / 4 - 1, -1.0])
        np.testing.assert_allclose(o["pnl_long_net_bps"], [0.0, -500.0])

    def test_straddle_selection_nearest_two_sided(self):
        t = pd.Timestamp("2026-10-01 20:00", tz="UTC")
        rows = []
        for K, cbid, pbid in [(99.0, 1.0, 1.0), (100.0, 0.0, 1.0), (101.0, 1.0, 1.0)]:
            for cp, b in (("call", cbid), ("put", pbid)):
                rows.append({"state_time": t, "underlying": "SOL", "option_family_id": "SOL|f", "option_type": cp,
                             "strike": K, "best_bid_price": b, "best_ask_price": 1.2, "option_forward_price": 100.2,
                             "minutes_to_expiry": 720.0, "instrument_type": "linear", "settlement_currency": "USDC"})
        s = v.select_straddles(v.prepare_legs(pd.DataFrame(rows)))
        self.assertEqual(len(s), 1)
        self.assertEqual(float(s["K"].iloc[0]), 101.0)   # 100 has a one-sided call; 101 is nearer than 99
        self.assertEqual(s["session"].iloc[0], "2026-10-02")

    def test_confirmation_window_excluded(self):
        t = pd.Timestamp("2026-11-07 06:00", tz="UTC")     # expires 2026-11-08 08:00
        rows = [{"state_time": t, "underlying": "BTC", "option_family_id": "BTC|f", "option_type": cp,
                 "strike": 100.0, "best_bid_price": 1.0, "best_ask_price": 1.1, "option_forward_price": 100.0,
                 "minutes_to_expiry": 1560.0 - 600, "instrument_type": "linear", "settlement_currency": "USDC"}
                for cp in ("call", "put")]
        self.assertTrue(v.prepare_legs(pd.DataFrame(rows)).empty)

    def test_settlement_prefers_exchange(self):
        st = pd.DataFrame({"underlying": ["BTC", "ETH"], "session": ["2026-10-02"] * 2,
                           "settlement_currency": ["BTC", "USDC"]})
        dl = pd.DataFrame({"index_name": ["btc_usd"], "session": ["2026-10-02"], "delivery_price": [101.0]})
        ap = pd.DataFrame({"session": ["2026-10-02"] * 2, "underlying": ["BTC", "ETH"], "settle_approx": [100.0, 50.0]})
        x = v.attach_settlement(st, dl, ap).set_index("underlying")
        self.assertEqual(x.loc["BTC", "S_T"], 101.0)
        self.assertEqual(x.loc["BTC", "settle_source"], "exchange")
        self.assertEqual(x.loc["ETH", "settle_source"], "approx_twap")
        self.assertEqual(v.index_name("ETH", "USDC"), "eth_usdc")

    def test_verdicts(self):
        self.assertEqual(v.verdict({"ci_low": 0.1, "ci_high": 0.3}), "LONG_EDGE_CANDIDATE")
        self.assertEqual(v.verdict({"ci_low": -0.3, "ci_high": -0.1}), "NO_LONG_EDGE")
        self.assertEqual(v.verdict({"ci_low": -0.1, "ci_high": 0.1}), "INCONCLUSIVE")
        self.assertEqual(v.verdict({"n_sessions": 2}), "INSUFFICIENT_SESSIONS")


if __name__ == "__main__":
    unittest.main()
