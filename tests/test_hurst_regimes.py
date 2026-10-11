"""Synthetic tests for Hurst Regimes 01 (hurst_regimes).

Run from repo root: python -m unittest discover -s tests -p "test_hurst_regimes.py" -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_engineering.modeling import hurst_regimes as hr


class HurstRegimeTests(unittest.TestCase):
    def test_selftest(self):
        self.assertTrue(hr.selftest()["ok"])

    def test_estimator_unbiased_on_fgn(self):
        for H in (0.4, 0.6):
            E = np.random.default_rng(7).standard_normal((800, hr.BARS)) @ hr.fgn_cholesky(H).T
            self.assertAlmostEqual(float(hr.block_stats(E)["H"].mean()), H, delta=0.02)

    def test_estimator_scale_invariant(self):
        E = np.random.default_rng(8).standard_normal((50, hr.BARS))
        np.testing.assert_allclose(hr.block_stats(E)["H"], hr.block_stats(1e-4 * E)["H"])

    def test_session_blocks_alignment(self):
        t = pd.date_range("2026-01-01 08:00", "2026-01-03 08:00", freq="5min", tz="UTC")
        h = pd.DataFrame({"state_time": t, "close": np.exp(np.arange(len(t)) * 1e-4), "underlying": "BTC"})
        s, R = hr.session_blocks(h, "BTC")
        self.assertEqual(list(s), ["2026-01-02", "2026-01-03"])
        self.assertTrue(np.isfinite(R).all())                          # bar ending 08:05 opens, 08:00 closes
        np.testing.assert_allclose(R, 1e-4)

    def test_missing_bars_and_minimum(self):
        t = pd.date_range("2026-01-01 08:00", "2026-01-02 08:00", freq="5min", tz="UTC")
        h = pd.DataFrame({"state_time": t, "close": 100.0, "underlying": "ETH"}).drop(index=range(50, 70))
        s, R = hr.session_blocks(h, "ETH")
        self.assertLess(np.isfinite(R).sum(), hr.MIN_BARS)              # 21 missing returns -> block dropped later

    def test_holm(self):
        np.testing.assert_allclose(hr.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_decide_requires_half_and_size(self):
        base = {"n_blocks": 300, "excess_sd": 0.05, "rho2": 0.3, "rho2_null": 0.0, "rho3": 0.1, "rho3_null": 0.0}
        t = pd.DataFrame([{**base, "p1": 0.001, "p2": 0.001, "p3": 0.001},
                          {**base, "p1": 0.001, "p2": 0.5, "p3": 0.5},
                          {**base, "p1": 0.5, "p2": 0.5, "p3": 0.5}])
        d = hr.decide(t)
        self.assertTrue(d["G1"])
        self.assertFalse(d["G2"])
        self.assertFalse(d["development_all_pass"])


if __name__ == "__main__":
    unittest.main()
