"""Synthetic tests for Regime Structure 03 (big_move_days).

Run from repo root: python -m unittest discover -s tests -p "test_big_move_days.py" -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_engineering.modeling import big_move_days as bm

P = bm.Params(n_boot=200)


class TestSessions(unittest.TestCase):
    def setUp(self):
        self.df = bm.synthetic_klines(n_days=60, seed=1, start="2026-01-01 03:17")
        self.b = bm.session_bars(self.df)

    def test_sessions_start_0800_and_complete(self):
        self.assertEqual(self.b["r5"].shape[1], 288)
        self.assertTrue((self.b["session"].hour == 0).all())
        # first session starts at the first 08:00 after the data start, ends next day
        self.assertEqual(str(self.b["session"][0].date()), "2026-01-02")

    def test_session_return_matches_prices(self):
        d = self.df.set_index(pd.to_datetime(self.df["open_time"], utc=True))
        s = self.b["session"][3]
        p0 = np.log(d.loc[s - pd.Timedelta(days=1) + pd.Timedelta(hours=8) - pd.Timedelta(minutes=1), "close"])
        p1 = np.log(d.loc[s + pd.Timedelta(hours=8) - pd.Timedelta(minutes=1), "close"])
        self.assertAlmostEqual(self.b["r5"][3].sum(), p1 - p0, places=10)

    def test_forecast_is_causal(self):
        t, _, _ = bm.session_table(self.b, P)
        rv = t["rv"].to_numpy()
        k = 20
        self.assertAlmostEqual(t["sigma_hat"].iloc[k], np.sqrt(np.mean(rv[k - 7:k] ** 2)), places=12)
        self.assertTrue(np.isnan(t["sigma_hat"].iloc[6]))


class TestStats(unittest.TestCase):
    def test_auc_matches_bruteforce_and_weights(self):
        rng = np.random.default_rng(0)
        x = rng.integers(0, 5, 200).astype(float)
        y = rng.random(200) < 0.3
        brute = np.mean([(a > b) + 0.5 * (a == b) for a in x[y] for b in x[~y]])
        self.assertAlmostEqual(bm.auc(x, y), brute, places=12)
        w = rng.integers(0, 3, 200).astype(float)
        xr, yr = np.repeat(x, w.astype(int)), np.repeat(y, w.astype(int))
        self.assertAlmostEqual(bm.auc(x, y, w), bm.auc(xr, yr), places=12)

    def test_planted_precursor_found_and_null_not(self):
        for planted in (True, False):
            r = bm.run_asset(bm.synthetic_klines(n_days=700, seed=3, planted=planted, start="2025-01-01"), "SYN", P)
            t = r["table"]
            t["window"] = np.where(t["session"] >= pd.Timestamp("2025-08-01", tz="UTC"), "test", "dev")
            out = bm.screen(t, bm.PRE, P).set_index("feature")
            if planted:
                self.assertTrue(out.loc["pre_last4h_vol_rel", "confirmed"])
                self.assertLess(out.loc["pre_last4h_vol_rel", "auc_dev"], 0.5)
            else:
                self.assertLessEqual(int(out["confirmed"].sum()), 1)


class TestEarlyTarget(unittest.TestCase):
    def test_rest_excludes_first_two_hours(self):
        b = bm.session_bars(bm.synthetic_klines(n_days=40, seed=2))
        t, _, _ = bm.session_table(b, P)
        k = 15
        rest = abs(b["r5"][k, 24:].sum()) / (t["sigma_hat"].iloc[k] * np.sqrt(264 / 288))
        self.assertAlmostEqual(t["z_rest"].iloc[k], rest, places=12)
        t2, _ = bm.label_big(t, P)
        self.assertIn("big_rest", t2)


if __name__ == "__main__":
    unittest.main()
