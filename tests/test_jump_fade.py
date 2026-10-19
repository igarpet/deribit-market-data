import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_engineering.modeling import jump_fade as jf  # noqa: E402


def _flat(n=40):
    c = np.full(n, 100.0)
    return c.copy(), c + 0.01, c - 0.01, c.copy()


def _sig(n=40):
    s = np.full(n, np.nan)
    s[4::5] = 0.01
    return s


class TestJumpFade(unittest.TestCase):
    def test_selftest_take_profit(self):
        self.assertIn("ok", jf.selftest())

    def test_short_side_stop(self):
        o, h, l, c = _flat()
        h[5], c[5] = 101.6, 101.55                 # sell fills at 100*e^0.015 = 101.51
        o[6], h[6], l[6], c[6] = 101.6, 103.5, 101.5, 103.4   # through SL = 100*e^0.03 = 103.05
        d, f, e, s, E, X, ty, am = jf.simulate_kernel(o, h, l, c, _sig(), np.ones(40, np.bool_), 1.5, 3.0, 60)
        self.assertEqual((s[0], ty[0], e[0]), (-1, 1, 6))
        self.assertAlmostEqual(X[0], 100 * math.exp(0.03), places=9)

    def test_gap_through_stop_uses_open(self):
        o, h, l, c = _flat()
        l[5], c[5] = 98.4, 98.45
        o[6], h[6], l[6], c[6] = 96.0, 96.5, 95.5, 96.2      # opens below SL = 97.04
        d, f, e, s, E, X, ty, am = jf.simulate_kernel(o, h, l, c, _sig(), np.ones(40, np.bool_), 1.5, 3.0, 60)
        self.assertEqual(ty[0], 1)
        self.assertAlmostEqual(X[0], 96.0)

    def test_time_stop_and_no_fill(self):
        o, h, l, c = _flat(200)
        l[5], c[5] = 98.4, 98.45
        o[6:], h[6:], l[6:], c[6:] = 98.45, 98.5, 98.4, 98.45
        d, f, e, s, E, X, ty, am = jf.simulate_kernel(o, h, l, c, _sig(200), np.ones(200, np.bool_), 1.5, 3.0, 60)
        self.assertEqual((ty[0], e[0] - f[0]), (2, 60))
        o, h, l, c = _flat()
        d, *_ = jf.simulate_kernel(o, h, l, c, _sig(), np.ones(40, np.bool_), 1.5, 3.0, 60)
        self.assertEqual(len(d), 0)

    def test_ambiguous_books_worse_side(self):
        o, h, l, c = _flat(80)
        h[5], l[5], c[5] = 101.6, 98.4, 100.0       # both cross in the same minute
        o[6:], h[6:], l[6:], c[6:] = 100.0, 103.5, 99.9, 103.4   # up move: short side is the loser
        d, f, e, s, E, X, ty, am = jf.simulate_kernel(o, h, l, c, _sig(80), np.ones(80, np.bool_), 1.5, 3.0, 60)
        self.assertEqual((am[0], s[0]), (1, -1))

    def test_costs(self):
        tr = pd.DataFrame({"gross": [0.0015, -0.003, 0.0], "type": [0, 1, 2]})
        x = jf.add_costs(tr, "deribit")
        np.testing.assert_allclose(x["cost"], [0.0, 6e-4, 6e-4])
        x = jf.add_costs(tr, "binance")
        np.testing.assert_allclose(x["cost"], [4e-4, 8e-4, 8e-4])

    def test_sign_flip_preserves_abs_returns(self):
        M = jf.synthetic_minutes(3, assets=("A", "B"), seed=2)
        F = jf.sign_flip(M, 5)
        for a in M:
            r0 = np.diff(np.log(M[a]["close"]))
            r1 = np.diff(np.log(F[a]["close"]))
            np.testing.assert_allclose(np.abs(r0), np.abs(r1), atol=1e-10)
            self.assertTrue((F[a]["high"] >= F[a][["open", "close"]].max(axis=1) - 1e-9).all())
            self.assertTrue((F[a]["low"] <= F[a][["open", "close"]].min(axis=1) + 1e-9).all())

    def test_detects_reversion_vs_null(self):
        M = jf.synthetic_minutes(500, seed=1, revert=0.3)
        inp = {a: jf.decision_inputs(m) for a, m in M.items()}
        thr = {a: jf.burst_threshold(i) for a, i in inp.items()}
        g = jf.pooled_gross_bp(jf.run_split(inp, "all", thr), "test")
        n = jf.run_null(M, 3, splits=("all",))
        self.assertGreater(g, n["all_test"].max() + 3)
        M0 = jf.synthetic_minutes(500, seed=3, revert=0.0)
        inp0 = {a: jf.decision_inputs(m) for a, m in M0.items()}
        g0 = jf.pooled_gross_bp(jf.run_split(inp0, "all", {a: 1.0 for a in inp0}), "test")
        self.assertLess(g0, 0)


if __name__ == "__main__":
    unittest.main()
