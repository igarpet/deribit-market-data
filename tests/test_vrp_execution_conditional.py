"""Tests for VRP 03 (vrp_execution) and VRP 04 (vrp_conditional_short).

Run from repo root: python -m unittest discover -s tests -p "test_vrp_execution_conditional.py" -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_engineering.modeling import vrp_conditional_short as cs
from data_engineering.modeling import vrp_execution as ve


class ExecutionTests(unittest.TestCase):
    def test_selftest(self):
        self.assertTrue(ve.selftest()["ok"])

    def test_family_tick(self):
        p = np.round(np.arange(1, 200) * 0.0005, 6)
        snap = pd.DataFrame({"option_family_id": "f", "best_bid_price": p, "best_ask_price": p + 0.0005})
        self.assertAlmostEqual(float(ve.family_ticks(snap)["f"]), 0.0005)

    def test_missing_snapshot_never_fills(self):
        paths = {"best_bid_price": np.array([[np.nan]]), "best_ask_price": np.array([[np.nan]]),
                 "last_price": np.array([[0.0]]), "traded": np.array([[1.0]])}
        self.assertEqual(ve.first_fill(paths, np.array([1.0]), "buy", 1)[0], 0)

    def test_fee_native_inverse(self):
        f = ve.fee_native(np.array([0.01]), np.array([100000.0]), np.array([True]))
        self.assertAlmostEqual(float(f[0]), min(0.0003, 0.125 * 0.01))

    def test_long_return_matches_vrp02_convention(self):
        st = pd.DataFrame({"S_T": [110.0], "K": [100.0], "inverse": [True]})
        r = ve.returns(st, np.array([0.08]), np.array([0.0]), "buy")[0]
        self.assertAlmostEqual(r, (10 / 110) / 0.08 - 1)


class ConditionalShortTests(unittest.TestCase):
    def test_selftest(self):
        self.assertTrue(cs.selftest()["ok"])

    def test_prepare_legs_windows(self):
        t = [pd.Timestamp("2026-11-05 20:00", tz="UTC"), pd.Timestamp("2026-11-07 20:00", tz="UTC")]
        ch = pd.DataFrame({"state_time": t, "underlying": "BTC", "option_family_id": "f", "option_type": "call",
                           "strike": 100.0, "best_bid_price": 1.0, "best_ask_price": 1.1, "option_forward_price": 100.0,
                           "minutes_to_expiry": 720.0, "instrument_type": "linear", "settlement_currency": "USDC"})
        self.assertEqual(list(cs.prepare_legs(ch, confirm=False).session), ["2026-11-06"])
        self.assertEqual(list(cs.prepare_legs(ch, confirm=True).session), ["2026-11-08"])

    def test_freeze_never_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "spec.json"
            cs.freeze(p, {"BTC": np.ones(288)}, [-0.5, 0.1], "2026-11-06")
            with self.assertRaises(FileExistsError):
                cs.freeze(p, {"BTC": np.ones(288)}, [0.0, 1.0], "2026-11-06")
            spec = cs.load_frozen(p)
            self.assertEqual(spec["thresholds"], [-0.5, 0.1])

    def test_decide_needs_sessions(self):
        x = pd.DataFrame({"tercile": "high", "ironfly_ror": [0.1] * 10, "session": [f"s{i}" for i in range(10)]})
        self.assertEqual(cs.decide(x, n_boot=100)["verdict"], "INSUFFICIENT_SESSIONS")
        x = pd.DataFrame({"tercile": "high", "ironfly_ror": np.linspace(0.05, 0.15, 30), "session": [f"s{i}" for i in range(30)]})
        self.assertEqual(cs.decide(x, n_boot=200)["verdict"], "CONDITIONAL_PREMIUM_CONFIRMED")


if __name__ == "__main__":
    unittest.main()
