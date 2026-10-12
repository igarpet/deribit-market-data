"""Synthetic tests for Hurst Regimes 03 (rough_vol_smm).

Run from repo root: python -m unittest discover -s tests -p "test_rough_vol_smm.py" -v
The recovery test takes about a minute.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_engineering.modeling import rough_vol_smm as sm


class RoughVolSMMTests(unittest.TestCase):
    def test_selftest(self):
        self.assertTrue(sm.selftest()["ok"])

    def test_day_grid_alignment(self):
        t = pd.date_range("2026-01-05 00:00", "2026-01-08 03:00", freq="5min", tz="UTC")   # Monday 00:00
        lp = pd.Series(np.cumsum(np.full(len(t), 1e-4)), index=t)
        r, slot, idx = sm.day_grid(lp)
        self.assertEqual(len(r) % sm.BARS_DAY, 0)
        self.assertEqual(idx[0], pd.Timestamp("2026-01-05 00:05", tz="UTC"))
        self.assertEqual(slot[0], 0)                       # Monday, first slot
        self.assertEqual(slot[sm.BARS_DAY], sm.BARS_DAY)   # Tuesday, first slot

    def test_fgn_fast(self):
        x = sm.fgn_fast(0.3, 200_000, np.random.default_rng(3))
        self.assertAlmostEqual(float(x.var()), 1.0, delta=0.02)
        self.assertAlmostEqual(float(np.corrcoef(x[1:], x[:-1])[0, 1]), 2 ** 0.6 / 2 - 1, delta=0.01)

    def test_t_innovations(self):
        x = sm.t_innovations(np.random.default_rng(4).random(1_000_000), 8.0)
        self.assertAlmostEqual(float(x.var()), 1.0, delta=0.02)
        self.assertAlmostEqual(float(np.mean(x ** 4)), 3 + 6 / 4, delta=0.3)

    def test_fou_rho_limits(self):
        x = sm.fou_one_minus_rho(0.5, 0.05, [1, 3, 10])
        np.testing.assert_allclose(x, 1 - 0.95 ** np.array([1, 3, 10]), atol=1e-6)
        y = sm.fou_one_minus_rho(0.1, 0.01, [1, 10, 100])
        self.assertTrue(np.all(np.diff(y) > 0) and y[0] > 0.3)     # rough: large jump at lag 1

    def test_param_fit_recovers_ar1(self):
        rng = np.random.default_rng(5)
        e = rng.standard_normal(60_000)
        y = np.zeros_like(e)
        for i in range(1, len(e)):
            y[i] = 0.98 * y[i - 1] + e[i]
        f = sm.param_fit_1h(y + 0.5 * rng.standard_normal(len(y)), h_grid=(0.3, 0.4, 0.5, 0.6),
                            kappa_grid_h=(0.005, 0.01, 0.02, 0.04, 0.08))
        self.assertEqual(f["H"], 0.5)
        self.assertAlmostEqual(f["kappa_per_hour"], 0.02, delta=0.01)
        self.assertGreater(f["noise_eta2"], 0.1)

    def test_optimizer_beats_truth_short_sample(self):
        """Optimizer check (not a statistical one): on a short sample the fitted point must
        fit the moments at least as well as the true parameters. Statistical recovery at
        full length is in notebook 03 section 3."""
        slot, _ = sm._layout(120)
        mask = np.zeros(len(slot), bool)
        theta0 = (0.05, 0.05, 0.8, 5.0)
        r = sm.simulate(theta0, slot, mask, np.ones(7 * sm.BARS_DAY), 99_001)
        f = sm.fit(r, slot, h_grid=(0.01, 0.03, 0.07, 0.15, 0.3), kappa_grid=(0.01, 0.05, 0.3),
                   r_final=6, r_cov=12, maxfev=150)
        j_hat = sm.objective(f["theta"], f, slot, R=6)
        j_true = sm.objective(theta0, f, slot, R=6)
        self.assertLessEqual(j_hat, 1.05 * j_true + 0.5)

if __name__ == "__main__":
    unittest.main()
