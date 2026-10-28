"""Tests for Turtle Breakout 18 (turtle_flat_diag).

Run from repo root: python -m unittest discover -s tests -p "test_turtle_flat_diag.py" -v
"""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_engineering.modeling import turtle_flat_diag as fd

DP = replace(fd.DParams(), n_perm=300)


def ar1(n, phi, seed):
    rng = np.random.default_rng(seed); e = rng.normal(0, 0.03, n); r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + e[t]
    return r


class Tests(unittest.TestCase):
    def test_sums(self):
        r = np.arange(1.0, 7.0)
        np.testing.assert_allclose(fd.fwd_sum(r, 2)[:4], [5, 7, 9, 11]); self.assertTrue(np.isnan(fd.fwd_sum(r, 2)[-1]))
        np.testing.assert_allclose(fd.past_sum(r, 3)[2:], [6, 9, 12, 15])

    def test_leading_nan(self):
        r = ar1(2000, -0.25, 9); r[0] = np.nan; m = np.ones(2000, bool)
        s = fd.state_stats(r, m, fd.permutation(r, m, DP))
        self.assertTrue(np.isfinite(s["VR5"])); self.assertTrue(np.isfinite(s["REV55"])); self.assertEqual(fd.decide(s, DP)["verdict"], "MR_PRESENT")
        self.assertTrue(np.isnan(fd.fwd_sum(r, 3)[-1])); self.assertTrue(np.isfinite(fd.fwd_sum(r, 3)[0]))

    def test_random_walk_no_mr(self):
        r = ar1(3000, 0.0, 1); m = np.ones(3000, bool)
        s = fd.state_stats(r, m, fd.permutation(r, m, DP))
        self.assertEqual(fd.decide(s, DP)["verdict"], "NO_MEAN_REVERSION")
        self.assertAlmostEqual(s["VR5"], 1.0, delta=0.15)

    def test_mean_reversion_detected(self):
        r = ar1(3000, -0.25, 2); m = np.ones(3000, bool)
        s = fd.state_stats(r, m, fd.permutation(r, m, DP))
        d = fd.decide(s, DP)
        self.assertLess(s["VR5"], 0.85); self.assertEqual(d["verdict"], "MR_PRESENT")

    def test_trend_not_mr(self):
        r = ar1(3000, 0.25, 3); m = np.ones(3000, bool)
        s = fd.state_stats(r, m, fd.permutation(r, m, DP))
        self.assertGreater(s["VR5"], 1.15); self.assertEqual(fd.decide(s, DP)["verdict"], "NO_MEAN_REVERSION")

    def test_mask_conditions(self):
        r = np.r_[ar1(1500, -0.3, 4), ar1(1500, 0.3, 5)]; m = np.r_[np.ones(1500, bool), np.zeros(1500, bool)]
        self.assertLess(fd.vr(r, m, 5), 0.9); self.assertGreater(fd.vr(r, ~m, 5), 1.1)

    def test_state_and_trend(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        tr = pd.DataFrame({"entry": [dates[40]], "exit": [dates[50]]})
        s = fd.turtle_state(dates, tr, dates[30])
        self.assertTrue(s.iloc[:30].isna().all()); self.assertEqual((s == "in").sum(), 10); self.assertEqual(s.iloc[50], "flat")
        c = pd.Series(np.r_[np.linspace(100, 50, 60), np.linspace(50, 120, 40)], index=dates)
        lab = fd.trend_label(c)
        self.assertEqual(lab.iloc[59], "down"); self.assertEqual(lab.iloc[99], "up")

    def test_tables(self):
        r = ar1(2000, -0.2, 6); m = np.ones(2000, bool)
        s, T = fd.rev(r, m, 5, 5); self.assertEqual(len(T), 5); self.assertLess(s, 0)
        c = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.date_range("2015-01-01", periods=2000, freq="D"))
        Z = fd.z_table(c, r, m, 20, 5); self.assertEqual(int(Z.n.sum()), int(np.isfinite(fd.fwd_sum(r, 5)[19:]).sum()))


if __name__ == "__main__":
    unittest.main()
