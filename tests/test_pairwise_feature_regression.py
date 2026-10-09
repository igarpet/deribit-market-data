"""Synthetic checks for the directed pairwise OLS matrix engine.

Run: python -m unittest discover -s tests -p "test_pairwise_feature_regression.py" -v
No DuckDB, external data or API access required.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_engineering.modeling.pairwise_feature_regression import (
    pairwise_ols, summarize_pairs, model_feature_frame,
)


class PairwiseFeatureRegressionTests(unittest.TestCase):
    def test_exact_linear_relationship_both_directions(self):
        x = np.arange(200, dtype=float)
        y = 5.0 + 2.0 * x
        out = pairwise_ols(pd.DataFrame({"x": x, "y": y}), min_n=100)
        self.assertAlmostEqual(out.beta.loc["y", "x"], 2.0, places=10)
        self.assertAlmostEqual(out.beta.loc["x", "y"], 0.5, places=10)
        self.assertAlmostEqual(out.intercept.loc["y", "x"], 5.0, places=10)
        self.assertLess(out.rmse.loc["y", "x"], 1e-6)
        self.assertAlmostEqual(out.standardized_beta.loc["y", "x"], 1.0)
        self.assertAlmostEqual(out.r_squared.loc["y", "x"], 1.0)
        self.assertTrue(np.isnan(out.beta.loc["x", "x"]))
        self.assertEqual(out.n_pairs.loc["x", "x"], 200)

    def test_negative_relationship(self):
        x = np.arange(200, dtype=float)
        out = pairwise_ols(pd.DataFrame({"x": x, "y": 12 - 3*x}), 100)
        self.assertAlmostEqual(out.beta.loc["y", "x"], -3.0, places=10)
        self.assertAlmostEqual(out.standardized_beta.loc["x", "y"], -1.0)

    def test_joint_missingness_is_pairwise_not_global(self):
        x = np.arange(150, dtype=float)
        y = 1 + 4*x
        z = x.copy()
        y[:20] = np.nan
        z[100:] = np.nan
        result = pairwise_ols(pd.DataFrame({"x": x, "y": y, "z": z}), min_n=60)
        self.assertEqual(result.n_pairs.loc["y", "x"], 130)
        self.assertEqual(result.n_pairs.loc["y", "z"], 80)
        self.assertEqual(result.n_pairs.loc["x", "z"], 100)
        self.assertAlmostEqual(result.beta.loc["y", "x"], 4, places=10)
        self.assertEqual(summarize_pairs(result)["possible_directed_pairs"], 6)

    def test_below_minimum_does_not_fit(self):
        x = np.arange(80, dtype=float)
        out = pairwise_ols(pd.DataFrame({"x": x, "y": x*2}), min_n=100)
        self.assertEqual(out.n_pairs.loc["x", "y"], 80)
        self.assertTrue(np.isnan(out.rmse.loc["x", "y"]))

    def test_constant_column_returns_nan_fits(self):
        x = np.arange(200, dtype=float)
        out = pairwise_ols(pd.DataFrame({"x": x, "const": np.full(200, 4.0)}))
        self.assertTrue(np.isnan(out.beta.loc["x", "const"]))
        self.assertTrue(np.isnan(out.beta.loc["const", "x"]))

    def test_scale_dependent_error_vs_symmetric_relative_error(self):
        rng = np.random.default_rng(12)
        x = rng.standard_normal(500)
        y = 2*x + rng.standard_normal(500)
        out = pairwise_ols(pd.DataFrame({"x": x, "y": y}))
        self.assertNotAlmostEqual(out.rmse.loc["x", "y"], out.rmse.loc["y", "x"])
        self.assertAlmostEqual(out.relative_rmse.loc["x", "y"],
                               out.relative_rmse.loc["y", "x"], places=10)
        self.assertAlmostEqual(out.standardized_beta.loc["x", "y"],
                               out.standardized_beta.loc["y", "x"], places=10)

    def test_long_form_directed_and_no_identity(self):
        x = np.arange(120, dtype=float)
        f = pd.DataFrame({"a": x, "b": 2*x+1, "c": 7-x})
        out = pairwise_ols(f, min_n=30)
        long = out.long_form()
        self.assertEqual(len(long), 6)
        self.assertEqual(len(long[long.response == long.predictor]), 0)
        self.assertEqual(int(long.n_pairs.min()), 120)

    def test_invalid_arguments(self):
        with self.assertRaisesRegex(ValueError, "at least 3"):
            pairwise_ols(pd.DataFrame({"x": [1,2]}), min_n=2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            pairwise_ols(pd.DataFrame([[1,2],[2,4]], columns=["x","x"]))

    def test_catalog_missing_features_are_visible_and_masked(self):
        # A deliberately sparse source state: optional catalog fields are
        # explicitly represented as unavailable rather than dropped.
        df = pd.DataFrame({"return_5m": [0.01, np.nan, -0.02],
                           "minutes_to_expiry": [120, 115, 110],
                           "utc_minute_of_day": [600, 605, 610],
                           "session_progress_fraction": [0.2, 0.21, 0.22],
                           "settlement_locked_fraction": [0, 0, 0],
                           "is_weekend": [False, False, True]})
        X, inventory = model_feature_frame(df)
        self.assertIn("log_spread_bps", X)
        self.assertTrue(X["log_spread_bps"].isna().all())
        self.assertFalse(bool(inventory.set_index("feature").loc[
            "log_spread_bps", "source_present"]))
        self.assertEqual(int(X["return_5m"].notna().sum()), 2)
        self.assertTrue(np.isfinite(X["clock_cos1"]).all())


if __name__ == "__main__":
    unittest.main()
