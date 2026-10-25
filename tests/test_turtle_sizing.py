"""Synthetic tests for Turtle Breakout 08 (turtle_sizing).

Run from repo root: python -m unittest discover -s tests -p "test_turtle_sizing.py" -v
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
from data_engineering.modeling import turtle_sizing as ts

warnings.filterwarnings("ignore", category=RuntimeWarning)
P = tb.Params()
SP = ts.SParams(n_boot=400, n_kelly_boot=300)
FR = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


def data(trend=0.01, seed=11, days=1500):
    b = tb.synthetic_5m(days, seed=seed, trend=trend, start="2020-01-01")
    return b, tb.prepare_asset("BTC", b, tb.synthetic_funding(b, 0.0), P)


class SweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b, cls.d = data()
        cls.start = cls.b.index[0] + pd.Timedelta(days=30)
        cls.tab, cls.C = ts.sweep(cls.d, P, cls.start, 125.0, fracs=FR)

    def test_defaults_reproduce_tb01(self):
        E0, T0, _ = tb.simulate({"BTC": self.d}, P, start=self.start)
        E1, T1, _ = ts.run_f(self.d, P, 0.02, self.start, 3.0)
        pd.testing.assert_series_equal(E0, E1); self.assertEqual(len(T0), len(T1))

    def test_same_entries_scale_with_f(self):
        _, Ta, _ = ts.run_f(self.d, P, 0.01, self.start, 125.0)
        _, Tb, _ = ts.run_f(self.d, P, 0.02, self.start, 125.0)
        np.testing.assert_array_equal(Ta.je.to_numpy(), Tb.je.to_numpy())
        np.testing.assert_allclose(Ta.ret_notional.to_numpy(), Tb.ret_notional.to_numpy(), atol=1e-12)
        self.assertGreater(Tb.units.iloc[0], 1.9 * Ta.units.iloc[0])

    def test_leverage_and_drawdown_grow_with_f(self):
        t = self.tab.set_index("f")
        self.assertTrue(np.all(np.diff(t.mean_gross_lev.loc[[0.01, 0.02, 0.05, 0.1]].to_numpy()) > 0))
        self.assertTrue(np.all(np.diff(t.max_dd.loc[[0.01, 0.02, 0.05, 0.1]].to_numpy()) < 0))

    def test_ruin_is_absorbing_and_floored(self):
        t = self.tab.set_index("f")
        self.assertTrue(t.loc[1.0, "ruined"])
        c = self.C[1.0]
        self.assertAlmostEqual(c.min(), ts.FLOOR * P.start_equity)
        first = int(np.argmax((c <= ts.FLOOR * P.start_equity * 1.0001).to_numpy()))
        self.assertTrue((c.iloc[first:] == c.iloc[first]).all())
        self.assertEqual(t.loc[1.0, "cagr"], -1.0)

    def test_cap_binds(self):
        tab3, _ = ts.sweep(self.d, P, self.start, 3.0, fracs=(0.5, 1.0))
        self.assertGreater(tab3.capped_entries.min(), 0)
        np.testing.assert_allclose(tab3.max_gross_lev.max(), 3.0, rtol=0.25)

    def test_equity_sizing_differs(self):
        a, _ = ts.sweep(self.d, P, self.start, 125.0, "hwm", fracs=(0.1,))
        e, _ = ts.sweep(self.d, P, self.start, 125.0, "equity", fracs=(0.1,))
        self.assertNotAlmostEqual(a.final_equity[0], e.final_equity[0])


class StatTests(unittest.TestCase):
    def test_holm(self):
        p = np.array([0.01, 0.04, 0.03, 0.2])
        np.testing.assert_allclose(ts.holm(p), [0.04, 0.09, 0.09, 0.2])

    def test_block_index_circular(self):
        I = ts.block_index(100, 21, 50, 0)
        self.assertEqual(I.shape, (50, 100)); self.assertTrue((I >= 0).all() and (I < 100).all())

    def test_no_edge_no_significant_growth(self):
        b, d = data(trend=0.0, seed=13)
        tab, C = ts.sweep(d, P, b.index[0] + pd.Timedelta(days=30), 125.0, fracs=FR)
        R = ts.compare(C, SP)
        self.assertEqual(len(R), len(FR) - 1)
        self.assertFalse((R.growth_result == "HIGHER_GROWTH").any())

    def test_strong_edge_detected(self):
        b, d = data(trend=0.06, seed=11)
        tab, C = ts.sweep(d, P, b.index[0] + pd.Timedelta(days=30), 125.0, fracs=FR)
        R = ts.compare(C, SP).set_index("f")
        self.assertEqual(R.loc[0.05, "growth_result"], "HIGHER_GROWTH")
        self.assertEqual(R.loc[0.01, "growth_result"], "LOWER_GROWTH")

    def test_kelly(self):
        R = np.r_[np.full(60, -0.8), np.full(40, 2.0)]          # p = 0.4, b = 2.5 per 0.8 -> f* = (0.4*2 - 0.6*0.8)/(0.8*2) = 0.2
        k = ts.kelly_curve(R)
        self.assertAlmostEqual(k.idxmax(), 0.2, delta=0.011)
        q = np.quantile(ts.kelly_boot(R, SP), [0.05, 0.95])
        self.assertLess(q[0], 0.2); self.assertGreater(q[1], 0.2)


if __name__ == "__main__":
    unittest.main()
