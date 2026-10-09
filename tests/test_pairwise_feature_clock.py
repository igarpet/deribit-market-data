"""Synthetic tests: exactly 288 same-clock 5-minute matrices, no hourly pooling.

Run from repo root: python -m unittest discover -s tests -p "test_pairwise_feature_clock.py" -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_engineering.modeling.pairwise_feature_clock import (
    clock_slot_indices, slot_labels, fit_clock_matrices,
    pair_clock_trajectory, save_clock_matrices, load_clock_matrix,
)


class ExactClockTests(unittest.TestCase):
    def sample(self, n_days=12):
        start = pd.Timestamp("2026-09-21 00:00:00", tz="UTC")
        t, x, y, z = [], [], [], []
        for k in range(n_days):
            date = start + pd.Timedelta(days=k)
            for minute in (0, 5, 10):
                t.append(date + pd.Timedelta(minutes=minute))
                # Unique nonconstant values across different UTC days.
                v = float(k) + 0.01 * minute
                x.append(v)
                y.append(5 + (minute // 5 + 2) * v)
                z.append(v if minute != 5 or k < 7 else np.nan)
        return pd.DataFrame({"x": x, "y": y, "z": z}), pd.Series(t)

    def test_exact_288_labels(self):
        labels = slot_labels()
        self.assertEqual(len(labels), 288)
        self.assertEqual(len(set(labels)), 288)
        self.assertEqual(labels[0], "00:00")
        self.assertEqual(labels[1], "00:05")
        self.assertEqual(labels[132], "11:00")
        self.assertEqual(labels[287], "23:55")

    def test_utc_time_conversion_not_local_clock(self):
        dt = pd.Series(pd.to_datetime(
            ["2026-10-01T12:00:00+02:00", "2026-10-01T11:05:00+00:00"]
        ))
        self.assertEqual(clock_slot_indices(dt).tolist(), [120, 133])

    def test_never_joins_neighboring_five_minute_slots(self):
        features, times = self.sample()
        m = fit_clock_matrices(features, times, min_pair_rows=6)
        self.assertEqual(m.beta_raw.shape, (288, 3, 3))
        self.assertEqual(int(m.summary.loc[0, "n_dates"]), 12)
        self.assertEqual(int(m.summary.loc[1, "n_dates"]), 12)
        self.assertEqual(int(m.summary.loc[2, "n_dates"]), 12)
        self.assertEqual(int(m.summary.loc[3, "n_dates"]), 0)
        self.assertAlmostEqual(m.view("00:00").loc["y", "x"], 2.0, places=10)
        self.assertAlmostEqual(m.view("00:05").loc["y", "x"], 3.0, places=10)
        self.assertAlmostEqual(m.view("00:10").loc["y", "x"], 4.0, places=10)
        self.assertTrue(np.isnan(m.view("00:15").loc["y", "x"]))
        self.assertEqual(int(m.view("00:05", "pair_counts").loc["y", "x"]), 12)

    def test_pairwise_complete_missingness_per_exact_slot(self):
        features, times = self.sample()
        m = fit_clock_matrices(features, times, min_pair_rows=6)
        self.assertEqual(int(m.view("00:05", "pair_counts").loc["x", "z"]), 7)
        self.assertEqual(int(m.view("00:05", "pair_counts").loc["y", "x"]), 12)
        self.assertEqual(int(m.view("00:00", "pair_counts").loc["x", "z"]), 12)
        self.assertTrue(np.isnan(m.view("00:05").loc["x", "x"]))
        self.assertEqual(int(m.summary.loc[1, "n_fitted_below_caution_n"]), 4)

    def test_no_fit_at_or_under_minimum(self):
        features, times = self.sample(n_days=5)
        m = fit_clock_matrices(features, times, min_pair_rows=6)
        self.assertEqual(int(m.summary.n_fitted_directed_pairs.sum()), 0)
        self.assertTrue(np.isnan(m.view("00:00").loc["y", "x"]))

    def test_duplicate_clock_date_rejected(self):
        features, times = self.sample()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            fit_clock_matrices(pd.concat([features, features.iloc[[0]]], ignore_index=True),
                               pd.concat([times, times.iloc[[0]]], ignore_index=True))

    def test_off_grid_timestamp_rejected(self):
        with self.assertRaisesRegex(ValueError, "five-minute"):
            clock_slot_indices(pd.Series([pd.Timestamp("2026-10-01T11:02:00Z")]))

    def test_npz_roundtrip_and_full_288_trajectory(self):
        features, times = self.sample()
        m = fit_clock_matrices(features, times)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clock_matrices.npz"
            save_clock_matrices(m, path)
            betas = load_clock_matrix(path, "00:10", "beta_raw")
            self.assertAlmostEqual(betas.loc["y", "x"], 4.0, places=10)
            n = load_clock_matrix(path, "00:05", "pair_counts")
            self.assertEqual(int(n.loc["y", "z"]), 7)
        trajectory = pair_clock_trajectory(m, "y", "x")
        self.assertEqual(len(trajectory), 288)
        self.assertAlmostEqual(trajectory.loc[1, "beta"], 3.0, places=10)
        self.assertTrue(np.isnan(trajectory.loc[10, "beta"]))

    def test_no_invalid_thresholds(self):
        f, t = self.sample()
        with self.assertRaisesRegex(ValueError, "min_pair_rows"):
            fit_clock_matrices(f, t, min_pair_rows=2)
        with self.assertRaisesRegex(ValueError, "caution_below"):
            fit_clock_matrices(f, t, min_pair_rows=10, caution_below=10)


if __name__ == "__main__":
    unittest.main()
