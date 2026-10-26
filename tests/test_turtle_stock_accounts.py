"""Tests for Turtle Breakout 15 (turtle_stock_accounts).

Run from repo root: python -m unittest discover -s tests -p "test_turtle_stock_accounts.py" -v
"""
import sys
import unittest
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests"))
from data_engineering.modeling import turtle_breakout as tb
from data_engineering.modeling import turtle_portfolio as tp
from data_engineering.modeling import turtle_reverse as tr
from data_engineering.modeling import turtle_sizing as ts
from data_engineering.modeling import turtle_stock_accounts as ta
from data_engineering.modeling import turtle_stocks as tk
from test_turtle_stocks import daily_ohlc, P

warnings.filterwarnings("ignore", category=RuntimeWarning)
AP = ta.AParams(n_null=20)
RP = replace(tr.RParams(), side_cost=0.0005)


def stock(seed, n=2500, drift=0.0003):
    d, ses = daily_ohlc(n, seed=seed, drift=drift)
    D = tb.prepare_asset(f"S{seed}", tk.pseudo_bars(d, ses), None, P)
    return D, ses, D.end[0] + pd.Timedelta(days=30)


class AccountTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.D, cls.ses, cls.start = stock(1)
        cls.L0 = ts.trade_list(cls.D, P, cls.start, RP, with_reverse=False)
        cls.L1 = ts.trade_list(cls.D, P, cls.start, RP, with_reverse=True)

    def test_trade_R_matches_book(self):
        b = tp.make_book(self.D, self.L1, P)
        np.testing.assert_allclose(ta.trade_R(self.L1, self.D).to_numpy(), b.trades.R.to_numpy(), rtol=1e-12)

    def test_borrow_only_shorts(self):
        B = ta.borrow_list(self.L1, self.D, 0.05, self.ses)
        lg = (self.L1.dir == 1).to_numpy()
        np.testing.assert_allclose(B.px_out[lg], self.L1.px_out[lg])
        self.assertTrue((B.px_out[~lg].to_numpy() > self.L1.px_out[~lg].to_numpy()).all())

    def test_account_is_tb01_single_asset(self):
        Dd, T, st = ta.account(self.D, self.L0, P, AP, self.start, self.ses)
        E, _, _ = tb.simulate({self.D.name: self.D}, P, start=self.start)
        ref = tk.real_daily(tp.daily_curve(E, P.start_equity), self.ses)
        i = Dd.index.intersection(ref.index)
        np.testing.assert_allclose(Dd[i].to_numpy()[:-1], ref[i].to_numpy()[:-1], rtol=1e-9)
        self.assertTrue(set(Dd.index) <= set(self.ses))

    def test_random_long_R(self):
        _, T, st = tb.simulate({self.D.name: self.D}, P, start=self.start)
        out = ta.random_long_R(self.D, P, self.start, st["p_entry"][self.D.name], 5, np.random.default_rng(0))
        self.assertEqual(out.shape, (5, 2)); self.assertTrue((out[:, 1] > 0).all())


class DecisionTests(unittest.TestCase):
    def test_sign_and_z(self):
        self.assertAlmostEqual(ta.sign_test(10, 10), 1 / 1024)
        z, p = ta.z_test(1.0, np.array([0.0, 0.1, -0.1, 0.05, -0.05]))
        self.assertGreater(z, 5); self.assertLess(p, 1e-6)

    def test_decide(self):
        d = ta.decide(100.0, 200, np.full(50, 50.0) + np.linspace(-5, 5, 50), np.full(50, 200), np.zeros(30), np.ones(30), AP)
        self.assertEqual(d["G1"], "PASS"); self.assertEqual(d["G2"], "PASS"); self.assertEqual(d["verdict"], "EDGE_AND_REVERSAL")
        d = ta.decide(55.0, 200, np.full(50, 50.0) + np.linspace(-0.5, 0.5, 50), np.full(50, 200), np.zeros(30), np.ones(30), AP)
        self.assertEqual(d["G1"], "FAIL")                                  # significant but below the 0.10R effect floor
        d = ta.decide(10.0, 200, np.full(50, 50.0) + np.linspace(-5, 5, 50), np.full(50, 200), np.ones(30), np.zeros(30), AP)
        self.assertEqual(d["verdict"], "NO_EDGE")

    def test_martingale_does_not_pass_G1(self):
        rs, rn, ns, nn = 0.0, 0, np.zeros(AP.n_null), np.zeros(AP.n_null)
        rng = np.random.default_rng(1)
        for s in range(8):
            d, ses = tk.synthetic_daily(2500, seed=500 + s)
            D = tb.prepare_asset("X", tk.pseudo_bars(d, ses), None, P); start = D.end[0] + pd.Timedelta(days=30)
            L0 = ts.trade_list(D, P, start, RP, with_reverse=False); L0 = L0[~L0.end]
            R = ta.trade_R(L0, D); rs += R.sum(); rn += len(R)
            _, T, st = tb.simulate({D.name: D}, P, start=start)
            o = ta.random_long_R(D, P, start, st["p_entry"][D.name], AP.n_null, rng); ns += o[:, 0]; nn += o[:, 1]
        d = ta.decide(rs, rn, ns, nn, np.zeros(10), np.ones(10), AP)
        self.assertEqual(d["G1"], "FAIL"); self.assertLess(d["diff_R"], AP.min_effect_R)

    def _unused(self):
        rs, rn, ns, nn = 0.0, 0, np.zeros(AP.n_null), np.zeros(AP.n_null)
        rng = np.random.default_rng(1)
        for s in range(6):
            D, ses, start = stock(100 + s, n=2000, drift=0.0)
            L0 = ts.trade_list(D, P, start, RP, with_reverse=False); L0 = L0[~L0.end]
            R = ta.trade_R(L0, D); rs += R.sum(); rn += len(R)
            _, T, st = tb.simulate({D.name: D}, P, start=start)
            o = ta.random_long_R(D, P, start, st["p_entry"][D.name], AP.n_null, rng); ns += o[:, 0]; nn += o[:, 1]
        z, p = ta.z_test(rs / rn, ns / nn)
        self.assertGreater(p, 0.01)


if __name__ == "__main__":
    unittest.main()
