"""Synthetic tests for Turtle Breakout 06 (turtle_options).

Run from repo root: python -m unittest discover -s tests -p "test_turtle_options.py" -v
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
from data_engineering.modeling import turtle_options as to
from data_engineering.modeling import turtle_trade_features as tf

warnings.filterwarnings("ignore", category=RuntimeWarning)
P = tb.Params()


def setup(trend=0.01, seed=7, days=900):
    b = tb.synthetic_5m(days, seed=seed, trend=trend, start="2021-01-01")
    d = tb.prepare_asset("BTC", b, tb.synthetic_funding(b), P)
    dvol = to.synthetic_dvol(b.index[0], b.index[-1], 0.6)
    T = tf.run_trades(d, P, b.index[0] + pd.Timedelta(days=40))
    return b, d, dvol, T


class HelperTests(unittest.TestCase):
    def test_expiry_is_friday_0800_after_target(self):
        for s in ("2024-01-01 10:00", "2024-03-07 09:00", "2024-03-08 07:00"):
            t = pd.Timestamp(s, tz="UTC")
            e = to.expiry_for(t, 38)
            self.assertEqual(e.weekday(), 4); self.assertEqual((e.hour, e.minute), (8, 0))
            self.assertGreaterEqual(e, t + pd.Timedelta(days=38)); self.assertLess(e, t + pd.Timedelta(days=45))

    def test_sigma_is_causal(self):
        s = pd.Series([0.5, 0.7], index=pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 01:00"], tz="UTC"))
        # the 01:00 bar closes at 02:00: at 01:30 only the 00:00 bar is complete
        self.assertAlmostEqual(to.sigma_at(s, pd.Timestamp("2024-01-01 01:30", tz="UTC"))[0], 0.5)
        self.assertAlmostEqual(to.sigma_at(s, pd.Timestamp("2024-01-01 02:00", tz="UTC"))[0], 0.7)
        self.assertTrue(np.isnan(to.sigma_at(s, pd.Timestamp("2024-01-02 12:00", tz="UTC"))[0]))


class ParityTests(unittest.TestCase):
    def test_perp_plus_put_equals_call_without_costs(self):
        b, d, dvol, T = setup()
        o = replace(to.OParams(), fee_rate=0.0, delivery_rate=0.0)
        tr = T.iloc[3]
        te, tx = pd.Timestamp(tr.entry_time), pd.Timestamp(tr.exit_time)
        F0, _ = to._price_at(d, te); F1, _ = to._price_at(d, tx)
        p = to.option_leg(d, dvol, te, tx, F0, False, o, 0.0)
        c = to.option_leg(d, dvol, te, tx, F0, True, o, 0.0)
        self.assertAlmostEqual(p["pnl"] + (F1 - F0), c["pnl"], places=6)

    def test_spread_and_fees_cost_money(self):
        b, d, dvol, T = setup()
        tr = T.iloc[2]; te, tx = pd.Timestamp(tr.entry_time), pd.Timestamp(tr.exit_time)
        F0, _ = to._price_at(d, te)
        a = to.option_leg(d, dvol, te, tx, F0, True, replace(to.OParams(), fee_rate=0.0, delivery_rate=0.0), 0.0)
        z = to.option_leg(d, dvol, te, tx, F0, True, to.OParams(), 0.05)
        self.assertLess(z["pnl"], a["pnl"])


class StrategyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b, cls.d, cls.dvol, cls.T = setup()
        cls.X = to.run_asset(cls.d, cls.dvol, cls.T, to.OParams(), 0.02)

    def test_columns_and_rolls(self):
        X = self.X
        for c in ("perp_R", "put_R", "perp_put_R", "callATM_R", "callOTM_R", "put_hold_R", "callATM_noroll_R"):
            self.assertIn(c, X)
        long_trades = X[(pd.to_datetime(X.exit_time) - pd.to_datetime(X.entry_time)) > pd.Timedelta(days=46)]
        if len(long_trades):
            self.assertTrue((long_trades.callATM_legs >= 2).all())
        short = X[(pd.to_datetime(X.exit_time) - pd.to_datetime(X.entry_time)) < pd.Timedelta(days=30)]
        self.assertTrue((short.callATM_legs == 1).all())
        self.assertTrue((short.put_how == "sold").all())

    def test_put_bounds_losses(self):
        X = self.X
        losers = X[X.perp_R < -0.5]
        self.assertGreater(len(losers), 3)
        self.assertGreater(losers.put_R.mean(), 0)           # puts pay on the trades that lose
        winners = X[X.perp_R > 2]
        self.assertLess(winners.put_R.mean(), 0)               # and cost premium on the winners

    def test_call_loss_bounded_by_premium(self):
        X = self.X
        self.assertTrue((X.callATM_R >= -(X.callATM_premium_R * X.callATM_legs) * 1.2 - 0.05).all())

    def test_curves_and_stats(self):
        o = replace(to.OParams(), n_boot=200)
        days = to.daily_days(pd.Timestamp(self.X.entry_time.min()), self.b.index[-1])
        C = {k: to.daily_curve(self.d, self.dvol, self.X, k, o, 0.02, days) for k in ("perp", "perp_put", "callATM")}
        np.testing.assert_allclose(C["perp"].iloc[-1], self.X.perp_R.sum(), atol=1e-6)
        np.testing.assert_allclose(C["perp_put"].iloc[-1], self.X.perp_put_R.sum(), atol=1e-6)
        r = to.sharpe_diff_R(C["perp_put"], C["perp"], o, 0.025, 1)
        self.assertIn(r["result"], ("IMPROVES", "WORSENS", "NO_DIFFERENCE"))
        m = to.mean_diff_month(self.X.perp_put_R.to_numpy(), self.X.perp_R.to_numpy(),
                               pd.DatetimeIndex(self.X.entry_time), o, 0.025, 1)
        self.assertLessEqual(m["lo"], m["mean_diff_R"]); self.assertLessEqual(m["mean_diff_R"], m["hi"])
        self.assertLessEqual(to.max_dd(C["perp"]), 0)


if __name__ == "__main__":
    unittest.main()
