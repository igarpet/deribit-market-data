"""Synthetic tests for Hurst Regimes 02 (hurst_roughness).

Run from repo root: python -m unittest discover -s tests -p "test_hurst_roughness.py" -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_engineering.modeling import hurst_roughness as rg


class RoughnessTests(unittest.TestCase):
    def test_selftest(self):
        self.assertTrue(rg.selftest()["ok"])

    def test_fgn_variance_and_correlation(self):
        x = rg.fgn(0.7, 100_000, np.random.default_rng(1))
        self.assertAlmostEqual(float(np.var(x)), 1.0, delta=0.05)
        self.assertAlmostEqual(float(np.corrcoef(x[1:], x[:-1])[0, 1]), 2 ** 0.4 - 1, delta=0.02)

    def test_log_vol_blocks(self):
        t = pd.date_range("2026-01-01 00:05", periods=288 * 3, freq="5min", tz="UTC")
        r = np.tile([0.01, -0.01], 288 * 3 // 2)
        lv = rg.log_vol_series(pd.Series(np.cumsum(r), index=t), 288, deseason=False)
        self.assertEqual(int(np.isfinite(lv).sum()), 3)            # first day loses one return (diff), still >= 90%
        np.testing.assert_allclose(lv.dropna().to_numpy(), 0.5 * np.log(288 * 1e-4), rtol=1e-6)

    def test_missing_block_dropped(self):
        t = pd.date_range("2026-01-01 00:05", periods=288 * 2, freq="5min", tz="UTC")
        lp = pd.Series(np.cumsum(np.full(len(t), 0.001)), index=t)
        lp.iloc[300:400] = np.nan
        lv = rg.log_vol_series(lp, 288, deseason=False)
        self.assertTrue(np.isnan(lv.iloc[1]))

    def test_decay_fit_prefers_power_for_power_acf(self):
        k = np.arange(0, 61, dtype=float)
        rho = np.r_[1.0, 0.8 * k[1:] ** -0.4]
        self.assertEqual(rg.decay_fits(rho, 60)["better"], "power")

    def test_invert(self):
        cal = pd.DataFrame({"H_true": [0.1, 0.2, 0.3], "block": "1d", "kind": "corr", "H_est": [0.2, 0.25, 0.33]})
        self.assertAlmostEqual(rg.invert(cal, "1d", "corr", 0.25), 0.2)
        self.assertTrue(np.isnan(rg.invert(cal, "1d", "corr", 0.5)))


if __name__ == "__main__":
    unittest.main()
