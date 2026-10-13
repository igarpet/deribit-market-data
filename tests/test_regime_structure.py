"""Synthetic tests for Regime Structure 01 (regime_structure).

Run from repo root: python -m unittest discover -s tests -p "test_regime_structure.py" -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_engineering.modeling import regime_structure as rs


class TestAggregation(unittest.TestCase):
    def test_hours_aligned_and_sums(self):
        df = rs.synthetic_klines(n_days=3, seed=1, regime=False)
        df = df.iloc[7:].reset_index(drop=True)             # start mid-hour
        m = rs.prepare_minutes(df)
        self.assertEqual(m.t0.minute, 0)
        h = rs.hourly(m, rs.Params())
        self.assertTrue((h["t"].dt.minute == 0).all())
        self.assertAlmostEqual(h["net"].iloc[3], m.r1[180:240].sum(), places=12)
        self.assertAlmostEqual(h["vol"].iloc[3], m.v1[180:240].sum(), places=6)

    def test_null_preserves_volume_and_abs_minute_returns(self):
        df = rs.synthetic_klines(n_days=3, seed=2, regime=True)
        m = rs.prepare_minutes(df)
        s = np.random.default_rng(0).choice([-1.0, 1.0], len(m.r1))
        h0, h1 = rs.hourly(m, rs.Params()), rs.hourly(m, rs.Params(), s)
        self.assertTrue(np.allclose(h0["vol"], h1["vol"]))
        self.assertFalse(np.allclose(h0["net"], h1["net"]))

    def test_shock_has_priority(self):
        h = pd.DataFrame({"t": pd.date_range("2026-01-01", periods=1, freq="h", tz="UTC"), "hour": [0],
                          "net": [5.0], "er": [1.0], "s": [1.0], "shock": [4.0], "rv": [1.0],
                          "vol": [1.0], "delta": [1.0]})
        th = {"er_lo": 0.2, "er_hi": 0.5, "vol_med": np.ones(24), "rv_med": np.ones(24),
              "rvol_q": 0.5, "rrv_q": 0.5}
        self.assertEqual(rs.label(h, th, rs.Params())["label"].iloc[0], "SHOCK")
        h["shock"] = 1.0
        self.assertEqual(rs.label(h, th, rs.Params())["label"].iloc[0], "DISP")
        h["delta"] = -1.0                                   # CVD disagrees -> not a displacement
        self.assertEqual(rs.label(h, th, rs.Params())["label"].iloc[0], "OTHER")


class TestPower(unittest.TestCase):
    def run_one(self, regime):
        p = rs.Params(n_null=99)
        out = rs.run_asset(rs.synthetic_klines(n_days=200, seed=3, regime=regime), p, np.random.default_rng(0))
        ev = rs.evaluate([out["counts"]], [out["null_counts"]], p)
        return ev[ev["window"] == "test"].set_index("test")

    def test_planted_regimes_detected(self):
        ev = self.run_one(True)
        self.assertTrue(ev.loc["E1", "pass"])
        self.assertTrue(ev.loc["E2", "pass"])

    def test_no_regimes_no_pass(self):
        ev = self.run_one(False)
        self.assertFalse(ev["pass"].any())


class TestPooling(unittest.TestCase):
    def test_heterogeneous_base_rates_do_not_create_lift(self):
        # two assets, each with NO persistence (lift 1) but different base rates of X=Y
        def c(n, px):
            nx = n * px
            return {("test", "E1"): np.array([n, nx, nx, nx * px])}
        real = [c(1000, 0.1), c(1000, 0.6)]
        summed = sum(r[("test", "E1")] for r in real)
        self.assertGreater(rs.lift(summed), 1.2)                          # naive pooling is biased
        self.assertAlmostEqual(rs.pooled_lift(real, ("test", "E1")), 1.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
