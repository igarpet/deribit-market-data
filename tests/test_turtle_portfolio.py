"""Synthetic tests for Turtle Breakout 11 (turtle_portfolio).

Run from repo root: python -m unittest discover -s tests -p "test_turtle_portfolio.py" -v
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
from data_engineering.modeling import turtle_portfolio as tp
from data_engineering.modeling import turtle_sizing as ts

warnings.filterwarnings("ignore", category=RuntimeWarning)
PP = tp.PParams(n_boot=300)
P = tb.Params()


class SingleCoinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b = tb.synthetic_5m(1300, seed=11, trend=0.01, start="2021-01-01")
        cls.d = tb.prepare_asset("BTC", b, tb.synthetic_funding(b, 0.0001), P)
        cls.start = b.index[0] + pd.Timedelta(days=30)
        cls.L = ts.trade_list(cls.d, P, cls.start, with_reverse=True)
        cls.books = {"BTC": tp.make_book(cls.d, cls.L, P)}

    def test_r0_equals_single_coin_engine(self):
        o = tp.simulate(self.books, "R0", PP, self.start)
        E2, _, _ = ts.equity_from_trades(self.d, self.L, P, PP.f, self.start, PP.pos_cap, "hwm")
        a = tb.daily(o["equity"]); c = tb.daily(E2)
        i = a.index.intersection(c.index)
        self.assertGreater(len(i), 1000)
        np.testing.assert_allclose(a[i].to_numpy()[:-1], c[i].to_numpy()[:-1], rtol=1e-9)

    def test_equity_sizing_equals_single_coin_engine(self):
        pe = replace(PP, size_base="equity", f=0.01)
        o = tp.simulate(self.books, "R0", pe, self.start)
        E2, _, _ = ts.equity_from_trades(self.d, self.L, P, 0.01, self.start, PP.pos_cap, "equity")
        a = tb.daily(o["equity"]); c = tb.daily(E2)
        i = a.index.intersection(c.index)
        np.testing.assert_allclose(a[i].to_numpy()[:-1], c[i].to_numpy()[:-1], rtol=1e-9)

    def test_rules_reduce_to_r0_when_slack(self):
        base = tb.daily(tp.simulate(self.books, "R0", PP, self.start)["equity"])
        for rule, kw in (("R1", {"H": 10.0}), ("R2", {}), ("R3", {"C": 10.0}), ("R4", {})):
            o = tb.daily(tp.simulate(self.books, rule, replace(PP, r4_min_trades=10_000), self.start, **kw)["equity"])
            np.testing.assert_allclose(o.to_numpy(), base.to_numpy(), rtol=1e-12, err_msg=rule)

    def test_unit_R_matches_tb07(self):
        T = self.books["BTC"].trades
        self.assertTrue(np.isfinite(T.R).all())
        self.assertTrue(((T.kind == "reverse") == (T.dir == -1)).all())

    def test_window_excludes_orphan_reverses(self):
        T = self.books["BTC"].trades
        rv = T[T.kind == "reverse"].iloc[3]
        t0 = rv.parent_t + pd.Timedelta(minutes=5)          # the long is before t0, its reverse after
        o = tp.simulate(self.books, "R0", PP, t0)
        n_in = int(((T.t_in >= t0) & (T.parent_t >= t0)).sum())
        self.assertEqual(o["stats"]["entries"], n_in)


class PortfolioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.B = tp.synthetic_books(8, 1200, seed=1, with_reverse=True)
        cls.R = tp.returns_panel(cls.B)
        cls.t0 = pd.Timestamp("2021-07-15", tz="UTC")

    def test_heat_cap_binds_equity_base(self):
        pe = replace(PP, size_base="equity", f=0.005)
        o1 = tp.simulate(self.B, "R1", pe, self.t0, H=0.01)
        self.assertLessEqual(o1["max_open_risk"], 0.01 * 1.2)       # cap on the entry-day equity; marks move it slightly
        self.assertGreater(o1["stats"]["skipped"], 0)

    def test_heat_cap_binds(self):
        o0 = tp.simulate(self.B, "R0", PP, self.t0)
        o1 = tp.simulate(self.B, "R1", PP, self.t0, H=0.06)
        self.assertGreater(o0["max_open_risk"], 0.06)
        self.assertLessEqual(o1["max_open_risk"], 0.06 * 1.0001)
        self.assertGreater(o1["stats"]["skipped"], 0)

    def test_corr_scaling_reduces_risk(self):
        o0 = tp.simulate(self.B, "R0", PP, self.t0); o2 = tp.simulate(self.B, "R2", PP, self.t0, R=self.R)
        self.assertLess(o2["mean_open_risk"], o0["mean_open_risk"])
        self.assertEqual(o2["stats"]["entries"], o0["stats"]["entries"])

    def test_cluster_cap(self):
        o3 = tp.simulate(self.B, "R3", PP, self.t0, C=0.02, R=self.R)
        self.assertGreater(o3["stats"]["skipped"], 0)

    def test_average_linkage_two_blocks(self):
        names = list("abcdef")
        C = pd.DataFrame(0.05, index=names, columns=names)
        for g in ("abc", "def"):
            for x in g:
                for y in g:
                    C.at[x, y] = 0.9
        for x in names:
            C.at[x, x] = 1.0
        cl = tp.average_linkage(C, 2)
        self.assertEqual(len({cl[x] for x in "abc"}), 1); self.assertEqual(len({cl[x] for x in "def"}), 1)
        self.assertNotEqual(cl["a"], cl["d"])

    def test_r4_multipliers(self):
        m = tp.r4_multipliers(self.B, pd.Timestamp("2030-01-01", tz="UTC"), PP)
        self.assertEqual(set(m), set(self.B)); self.assertTrue(all(0 <= v <= PP.r4_max for v in m.values()))
        m0 = tp.r4_multipliers(self.B, pd.Timestamp("2021-01-01", tz="UTC"), PP)
        self.assertTrue(all(v == 1.0 for v in m0.values()))


class StatTests(unittest.TestCase):
    def test_calmar_paths(self):
        L = np.full(365, np.log(1.1) / 365)
        L[100] -= np.log(2)                      # one -50% day
        c = tp.calmar_paths(L[None])[0]
        cagr = np.exp(L.sum() * 365.25 / 365) - 1
        mdd = 1 - np.exp(-np.log(2) + np.log(1.1) / 365)
        self.assertAlmostEqual(c, cagr / mdd, places=9)

    def test_calmar_diff_identical(self):
        idx = pd.date_range("2024-01-01", periods=500, freq="D", tz="UTC")
        rng = np.random.default_rng(0)
        D = pd.Series(1e5 * np.exp(np.cumsum(rng.normal(0.001, 0.02, 500))), index=idx)
        r = tp.calmar_diff(D, D, PP, 1)
        self.assertAlmostEqual(r["calmar_diff"], 0.0); self.assertGreater(r["p"], 0.5)

    def test_select_tie_and_ruin(self):
        g = pd.DataFrame({"H": [0.04, 0.06, 0.08], "calmar": [1.0, 1.0, 2.0], "ruined": [False, False, True]})
        self.assertEqual(tp.select(g, "H"), 0.04)

    def test_holm(self):
        np.testing.assert_allclose(tp.holm([0.01, 0.04, 0.03, 0.2]), [0.04, 0.09, 0.09, 0.2])


if __name__ == "__main__":
    unittest.main()
