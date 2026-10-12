"""Tests for VD 12 (volatility_lifted_memory_long).

Run from repo root: python -m unittest discover -s tests -p "test_volatility_lifted_memory_long.py" -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_engineering.modeling import volatility_lifted_memory as lm
from data_engineering.modeling import volatility_lifted_memory_long as ll


class LongMemoryTests(unittest.TestCase):
    def test_selftest(self):
        self.assertTrue(ll.selftest()["ok"])

    def test_load_returns_gaps_end_and_min_days(self):
        with tempfile.TemporaryDirectory() as d:
            t = pd.date_range("2025-01-01 00:05", "2026-09-20", freq="5min", tz="UTC")
            c = np.exp(np.arange(len(t)) * 1e-6)
            h = pd.DataFrame({"state_time": t, "close": c}).drop(index=[10])
            h.to_csv(Path(d) / "BTC.csv.gz", index=False, compression="gzip")
            pd.DataFrame({"state_time": t[:288 * 30], "close": 1.0}).to_csv(Path(d) / "HYPE.csv.gz", index=False,
                                                                           compression="gzip")
            raw, excl = ll.load_returns(Path(d), ["BTC", "HYPE", "ETH"])
            self.assertIn("HYPE", excl)
            self.assertIn("ETH", excl)
            self.assertLess(raw.state_time.max(), ll.SAMPLE_END)
            r = raw.set_index("state_time").return_5m
            self.assertTrue(np.isnan(r.iloc[10]))                 # bar after the gap: no return across it
            self.assertAlmostEqual(float(r.iloc[5]), 1e-6, places=9)

    def test_unrestricted_columns(self):
        f = pd.DataFrame(index=range(3))
        cols = ll.add_unrestricted(f, np.arange(12, dtype=float).reshape(3, 4), 4)
        self.assertEqual(cols, ["u_n4_m0", "u_n4_m1", "u_n4_m2", "u_n4_m3"])
        self.assertEqual(float(f.loc[2, "u_n4_m3"]), 11.0)

    def test_climatology_design_has_no_volatility_inputs(self):
        tr = pd.DataFrame({"underlying": ["A", "B"] * 50, "slot": np.arange(100) % 288})
        Xtr, Xte = lm.design(tr, tr.iloc[:10], [])
        self.assertEqual(Xtr.shape[1], 1 + 1 + 2)                  # intercept, one asset dummy, two harmonics


if __name__ == "__main__":
    unittest.main()
