"""Synthetic smoke/regression tests; no research DuckDB required.

Run from repo root: python -m unittest discover -s tests -p 'test_vol_ols_5m.py' -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_engineering.modeling import vol_ols_5m as m


class VolOLS5MTests(unittest.TestCase):
    def toy(self):
        t = pd.date_range("2026-09-22 00:00", periods=6, freq="5min", tz="UTC")
        df = pd.DataFrame({
            "underlying": ["BTC"] * 6,
            "state_time": t,
            "session_id": ["a", "a", "a", "b", "b", "b"],
            "expiry_date": ["2026-09-22"] * 3 + ["2026-09-23"] * 3,
            "high": [101., 102., 103., 104., 105., 106.],
            "low": [99., 98., 97., 96., 95., 94.],
            "perp_integrity_ok": [True] * 6,
            "utc_minute_of_day": t.hour * 60 + t.minute,
            "session_progress_fraction": np.linspace(0.1, 0.9, 6),
            "settlement_locked_fraction": np.zeros(6),
            "is_weekend": [False] * 6,
            "minutes_to_expiry": [30., 25., 20., 15., 10., 5.],
            "return_5m": [0.001] * 6,
        })
        return df

    def test_target_next_bar_only_and_session_boundary(self):
        g = self.toy()
        meta, X, _ = m.make_panel(g)
        expected = np.log(102/98) / np.sqrt(4*np.log(2))
        self.assertAlmostEqual(meta.loc[0, "sigma_next_5m"], expected)
        self.assertTrue(pd.isna(meta.loc[2, "sigma_next_5m"]))
        self.assertTrue(pd.isna(meta.loc[5, "sigma_next_5m"]))
        self.assertEqual(X.loc[0, "return_5m"], 0.001)
        g.loc[1, "high"] = 120
        meta2, X2, _ = m.make_panel(g)
        self.assertNotEqual(meta2.loc[0, "sigma_next_5m"], meta.loc[0, "sigma_next_5m"])
        self.assertEqual(X.loc[0, "return_5m"], X2.loc[0, "return_5m"])

    def test_gap_and_target_integrity(self):
        g = self.toy()
        g.loc[1, "state_time"] += pd.Timedelta(minutes=1)
        meta, _, _ = m.make_panel(g)
        self.assertTrue(pd.isna(meta.loc[0, "sigma_next_5m"]))
        g = self.toy()
        g.loc[1, "perp_integrity_ok"] = False
        meta, _, _ = m.make_panel(g)
        self.assertTrue(pd.isna(meta.loc[0, "sigma_next_5m"]))

    def test_train_only_epsilon(self):
        train = np.array([0.0, 0.01, 0.02, 0.03] * 10)
        eps = m.fit_log_floor(train)
        self.assertAlmostEqual(eps, 0.0002)
        self.assertEqual(eps, m.fit_log_floor(np.r_[train, [100000.0]]))

    def test_ols_and_collinear_pruning(self):
        rng = np.random.default_rng(42)
        n = 300
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        X = pd.DataFrame({"a": x1, "b": x2, "a_clone": x1.copy()})
        y = 2. + 3*x1 - 2*x2
        train = np.arange(n) < 200
        names, drops = m.remove_aliases(X, train, list(X.columns))
        self.assertEqual(len(names), 2)
        self.assertEqual(len(drops), 1)
        fit = m.fit_ols(X, y, train, names)
        result = m.metrics(fit, X, y, ~train, "test")
        self.assertLess(result["log_rmse"], 1e-10)
        self.assertGreater(result["r2"], 0.999999999)

    def test_frozen_common_rows_in_backward_search(self):
        rng = np.random.default_rng(43)
        n = 400
        X = pd.DataFrame({"signal": rng.normal(size=n),
                          "junk": rng.normal(size=n)})
        y = X["signal"].to_numpy() + rng.normal(size=n) * 0.01
        tr = np.arange(n) < 280
        va = ~tr
        kept, history = m.backward_validation(X, y, tr, va,
                                               ["signal", "junk"], max_drops=1)
        self.assertIn("signal", kept)
        self.assertTrue(set(history.columns).issubset({
            "step", "removed", "p_before", "rmse_before",
            "rmse_after", "improvement"
        }))


if __name__ == "__main__":
    unittest.main()
