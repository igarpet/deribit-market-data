"""Tests for Turtle Breakout 20 (turtle_flat_legs).

Run from repo root: python -m unittest discover -s tests -p "test_turtle_flat_legs.py" -v
"""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_engineering.modeling import turtle_flat_legs as fl

LP = replace(fl.LParams(), n_null=2000)


def bars(C, spread=0.01):
    C = np.asarray(C, float); O = np.r_[C[0], C[:-1]]
    return pd.DataFrame({"open": O, "high": np.maximum(O, C) * (1 + spread), "low": np.minimum(O, C) * (1 - spread), "close": C},
                        index=pd.date_range("2020-01-01", periods=len(C), freq="D"))


def rw(n, seed, drift=0.0):
    rng = np.random.default_rng(seed)
    return bars(100 * np.exp(np.cumsum(rng.normal(drift, 0.035, n))), 0.015)


class Tests(unittest.TestCase):
    def test_no_time_limit_and_trailing(self):
        d = bars(np.r_[np.full(30, 100.0), 100 * 1.01 ** np.arange(200)], 0.002)    # steady uptrend: the long never stops out
        arrs = fl._arrays(d, None); atr = fl.wilder_atr(d, 20)
        j, fill, why = fl.walk(*arrs[:4], atr, arrs[4], 60, 1, LP)
        self.assertEqual(why, "end"); self.assertEqual(j, len(d) - 1)
        C = np.r_[np.full(30, 100.0), np.linspace(100, 150, 50), np.linspace(150, 100, 50)]
        d2 = bars(C, 0.002); arrs = fl._arrays(d2, None); atr = fl.wilder_atr(d2, 20)
        j, fill, why = fl.walk(*arrs[:4], atr, arrs[4], 40, 1, LP)
        self.assertEqual(why, "stop"); self.assertGreater(j, 80); self.assertGreater(fill, 130)    # trailing 10-day low locks in the gain

    def test_fixed_stop_and_gap(self):
        C = np.r_[np.full(40, 100.0), [70.0]]
        d = bars(C, 0.005); d.iloc[-1, d.columns.get_loc("open")] = 70.0; d.iloc[-1, d.columns.get_loc("high")] = 70.5
        arrs = fl._arrays(d, None); atr = fl.wilder_atr(d, 20)
        j, fill, why = fl.walk(*arrs[:4], atr, arrs[4], 39, 1, LP)
        self.assertEqual(why, "gap"); self.assertEqual(fill, 70.0)
        j, fill, why = fl.walk(*arrs[:4], atr, arrs[4], 39, -1, LP)
        self.assertEqual(why, "end")                                               # the short is not stopped by a fall

    def test_short_funding_sign_and_cost(self):
        d = bars(np.r_[np.full(40, 100.0), np.linspace(100, 80, 10), np.full(15, 95.0)], 0.002)
        f = pd.Series(0.0003, index=d.index)
        arrs = fl._arrays(d, f); atr = fl.wilder_atr(d, 20)
        r = fl.trade_record(d, arrs, atr, 39, -1, LP)
        self.assertLess(r["fund"], 0)                                              # side * funding = -1 * positive: received
        self.assertAlmostEqual(r["ret"], -(r["px_out"] / r["px_in"] - 1) - 0.0006 * (1 + r["px_out"] / r["px_in"]) - r["fund"])

    def test_signals_b(self):
        rng = np.random.default_rng(0)
        C = np.r_[100 * np.exp(np.cumsum(rng.normal(0.0015, 0.02, 450))), np.zeros(60)]
        C[450:] = C[449] * (1 + 0.002 * np.sin(np.arange(60)))                   # stalls at its high
        d = bars(C, 0.003); flat = pd.Series(True, index=d.index)
        s = fl.signals_b(d, flat, LP)
        self.assertTrue(s.iloc[-10:].all()); self.assertFalse(s.iloc[:100].any())  # needs 100 days of return history
        self.assertFalse(fl.signals_b(d, pd.Series(False, index=d.index), LP).any())

    def test_one_position_and_sizing(self):
        d = rw(1500, 2); sig = pd.Series(True, index=d.index)
        T, E = fl.simulate_leg(d, sig, 1, LP)
        self.assertTrue((T.i.to_numpy()[1:] >= T.j.to_numpy()[:-1]).all())
        self.assertEqual(len(E), len(d)); self.assertTrue(np.isfinite(E.iloc[-1]))

    def test_random_walk_does_not_pass(self):
        d = rw(2500, 3); flat = pd.Series(np.random.default_rng(3).random(len(d)) < 0.6, index=d.index)
        r = np.log(d.close).diff(); q = r.rolling(365, min_periods=100).quantile(0.2).shift(1)
        sigA = (r <= q) & flat
        T, _ = fl.simulate_leg(d, sigA, 1, LP); P = fl.pool(d, flat, 1, LP)
        self.assertFalse(fl.gate(T, P, LP, 1)["pass"])
        sigB = fl.signals_b(d, flat, LP)
        T, _ = fl.simulate_leg(d, sigB, -1, LP); P = fl.pool(d, flat, -1, LP)
        self.assertFalse(fl.gate(T, P, LP, 2)["pass"])


if __name__ == "__main__":
    unittest.main()
