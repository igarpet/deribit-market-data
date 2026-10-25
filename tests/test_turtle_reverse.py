"""Synthetic tests for Turtle Breakout 07 (turtle_reverse).

Run from repo root: python -m unittest discover -s tests -p "test_turtle_reverse.py" -v
"""
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_engineering.modeling import turtle_breakout as tb
from data_engineering.modeling import turtle_reverse as tr
from data_engineering.modeling import turtle_trade_features as tf

warnings.filterwarnings("ignore", category=RuntimeWarning)
P = tb.Params()
RP = tr.RParams(n_null=500, n_boot=300)


def setup(trend=0.0, seed=3, days=1200, rate=0.0):
    b = tb.synthetic_5m(days, seed=seed, trend=trend, start="2021-01-01")
    d = tb.prepare_asset("X", b, tb.synthetic_funding(b, rate), P)
    T = tf.run_trades(d, P, b.index[0] + pd.Timedelta(days=40))
    return b, d, T


class MechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b, cls.d, cls.T = setup()
        cls.R = tr.reverse_trades(cls.d, cls.T, RP)

    def test_one_reverse_per_losing_stop_exit(self):
        L = tr.losing_exits(self.T)
        self.assertGreater(len(L), 5)
        self.assertLessEqual(len(self.R), len(L)); self.assertGreaterEqual(len(self.R), len(L) - 1)
        self.assertTrue(set(L.reason) <= set(tr.STOP_REASONS)); self.assertTrue((L.R < 0).all())

    def test_short_exits_are_valid(self):
        d, R = self.d, self.R
        for r in R.itertuples():
            self.assertGreater(r.rev_exit_time, r.rev_time)
            self.assertIn(r.rev_reason, ("stop", "next_long"))
        nl = R[R.rev_reason == "next_long"]
        self.assertTrue(nl.rev_exit_time.isin(pd.DatetimeIndex(self.T.entry_time)).all())

    def test_stop_never_beyond_fixed_without_gap(self):
        R = self.R[self.R.rev_reason == "stop"]
        # without a gap the loss is at most the fixed stop (1R gross)
        self.assertTrue((R.R_rev_gross >= -1.0 - 1e-9).mean() > 0.9)

    def test_costs_and_funding_sign(self):
        _, d, T = setup(rate=0.0003)
        R = tr.reverse_trades(d, T, RP)
        self.assertTrue((R.funding_bp >= -1e-9).all())               # positive funding is received by a short
        self.assertTrue(((R.R_rev - R.R_rev_gross) < 0).sum() + ((R.funding_bp > 0)).sum() > 0)

    def test_flat_days_exclude_open_longs(self):
        d, T = self.d, self.T
        fd = set(tr.flat_days(d, T, 0))
        for r in T.itertuples():
            ks, ke = tr._k_of(d, int(r.je)), tr._k_of(d, int(r.jx))
            for k in range(ks + 1, ke + 1):
                self.assertNotIn(k, fd)

    def test_curves_end_at_sums(self):
        C = tr.curves(self.T, self.R)
        np.testing.assert_allclose(C.long_only.iloc[-1], self.T.R.sum(), atol=1e-9)
        np.testing.assert_allclose(C.reverse_only.iloc[-1], self.R.R_rev.sum(), atol=1e-9)
        self.assertLessEqual(tr.max_dd(C.long_plus_reverse), 0)


class StatisticsTests(unittest.TestCase):
    def run_set(self, trend, seeds):
        out = []
        for s in seeds:
            _, d, T = setup(trend=trend, seed=s)
            R = tr.reverse_trades(d, T, RP)
            pool = tr.random_short_pool(d, T, tr._k_of(d, int(T.je.min())), RP)
            nm = tr.null_means({"X": pool}, {"X": len(R)}, RP, s)
            out.append((tr.evaluate(R, nm, RP, s), tr.event_study(d, T, pool, RP)))
        return out

    def test_martingale_no_edge(self):
        res = self.run_set(0.0, range(8))
        ps = [ev["p"] for ev, _ in res]
        self.assertLessEqual(sum(ev["verdict"] == "REVERSE_PAYS" for ev, _ in res), 1)
        self.assertGreater(np.mean(ps), 0.25)
        diff = np.mean([ev["mean_R_rev"] - ev["null_mean_R"] for ev, _ in res])
        self.assertLess(abs(diff), 0.15)

    def test_event_table_shape(self):
        _, E = self.run_set(0.0, [1])[0]
        et = tr.event_table(E, RP)
        self.assertEqual(len(et), 3 * len(tr.HORIZONS_D))
        self.assertTrue((et.n > 0).all())
        r = et[et.kind == "random_flat"]
        self.assertTrue((r.short_R_net < 0.3).all())                 # no drift: shorts do not pay at random

    def test_downtrend_shorts_pay(self):
        res = self.run_set(-0.02, [5])
        ev, _ = res[0]
        self.assertGreaterEqual(ev["reversals"], 5)
        self.assertGreater(ev["mean_R_rev"], 0)

    def test_empirical_p(self):
        nm = np.linspace(-1, 1, 999)
        R = pd.DataFrame({"R_rev": [0.95] * 10, "rev_time": pd.date_range("2022-01-01", periods=10, freq="40D", tz="UTC"),
                          "R_rev_gross": 1.0, "hold_days": 3.0, "funding_bp": 0.0, "rev_reason": "stop"})
        ev = tr.evaluate(R, nm, RP, 0)
        self.assertLess(ev["p"], 0.05); self.assertEqual(ev["G1"], "PASS"); self.assertEqual(ev["verdict"], "REVERSE_PAYS")


if __name__ == "__main__":
    unittest.main()
