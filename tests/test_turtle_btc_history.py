"""Tests for Turtle Breakout 17 (turtle_btc_history).

Run from repo root: python -m unittest discover -s tests -p "test_turtle_btc_history.py" -v
"""
import sys
import unittest
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_engineering.modeling import turtle_breakout as tb
from data_engineering.modeling import turtle_btc_history as bh
from data_engineering.modeling import turtle_reverse as tr
from data_engineering.modeling import turtle_stock_accounts as ta
from data_engineering.modeling import turtle_stocks as tk

warnings.filterwarnings("ignore", category=RuntimeWarning)
P = replace(tb.Params(), min_bar_share=0.0, funding=False)


def btc_like(n=3000, seed=0, drift=0.0):
    d, _ = tk.synthetic_daily(n, seed=seed, vol=0.04, gap=0.0, steps=48, drift=drift)
    d.index = pd.date_range("2014-09-17", periods=n, freq="D")
    return d


class Tests(unittest.TestCase):
    def test_regimes_cover_and_order(self):
        self.assertEqual(bh.regime_of("2018-06-01"), "bear 2018"); self.assertEqual(bh.regime_of("2020-03-12"), "bull 2019-21")
        self.assertEqual(bh.regime_of("2014-09-17"), "bear 2014"); self.assertEqual(bh.regime_of("2026-10-01"), "after the 2025 top")
        ends = [pd.Timestamp(b) for _, b, _ in bh.REGIMES]; starts = [pd.Timestamp(a) for a, _, _ in bh.REGIMES]
        for e, s in zip(ends[:-1], starts[1:]):
            self.assertEqual(s - e, pd.Timedelta(days=1))

    def test_run_window_pre_is_truncated(self):
        d = btc_like(4000)
        r = bh.run_window(d, "2014-09-17", "2019-12-31", P, tr.RParams(), ta.AParams(), 20, np.random.default_rng(0))
        self.assertLessEqual(r["long"].index[-1], pd.Timestamp("2019-12-31"))
        self.assertTrue((r["trades"].entry <= pd.Timestamp("2019-12-31")).all())
        self.assertEqual(len(r["null_means"]), 20); self.assertGreater(len(r["R_long"]), 10)
        self.assertTrue(set(r["trades"].regime) <= {lab for _, _, lab in bh.REGIMES})

    def test_cost_override(self):
        d = btc_like(1500, seed=2)
        a = bh.run_window(d, "2014-09-17", None, P, tr.RParams(), ta.AParams(), 0, np.random.default_rng(0))
        b = bh.run_window(d, "2014-09-17", None, P, tr.RParams(), ta.AParams(), 0, np.random.default_rng(0), cost=0.0015)
        self.assertLess(b["long"].iloc[-1], a["long"].iloc[-1])

    def test_random_by_regime(self):
        d = btc_like(2000, seed=3)
        r = bh.run_window(d, "2014-09-17", None, P, tr.RParams(), ta.AParams(), 0, np.random.default_rng(0))
        X = bh.random_by_regime(r["D"], P, r["t0"], r["p_entry"], 5, np.random.default_rng(1), r["sessions"])
        self.assertIn("random_mean_R", X); self.assertGreater(X.random_trades.sum(), 0)

    def test_threshold_controls_false_positives(self):
        hits = 0
        for s in range(12):
            d = btc_like(1932, seed=900 + s)
            r = bh.run_window(d, "2014-09-17", None, P, tr.RParams(), ta.AParams(), 30, np.random.default_rng(s))
            hits += (r["R_long"].mean() - r["null_means"].mean()) >= bh.THRESHOLD_R["PRE"]
        self.assertLessEqual(hits, 2)


if __name__ == "__main__":
    unittest.main()
