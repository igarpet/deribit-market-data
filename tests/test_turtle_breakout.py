"""Synthetic tests for Turtle Breakout 01 (turtle_breakout).

Run from repo root: python -m unittest discover -s tests -p "test_turtle_breakout.py" -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_engineering.modeling import turtle_breakout as tb

S0 = pd.Timestamp("2020-03-01", tz="UTC")


def build(p, n_days=500, trend=0.0, seeds=(5, 6), rate=0.0001):
    data = {}
    for k, a in enumerate(("BTC", "ETH")[:len(seeds)]):
        b = tb.synthetic_5m(n_days, seed=seeds[k], trend=trend, start="2020-01-01")
        data[a] = tb.prepare_asset(a, b, tb.synthetic_funding(b, rate), p)
    return data


class PeriodTests(unittest.TestCase):
    def test_day_boundary_is_1700_new_york(self):
        # EDT (summer): 17:00 NY = 21:00 UTC ; EST (winter): 22:00 UTC
        t = pd.DatetimeIndex(["2025-07-01 21:00", "2025-07-01 21:05", "2025-01-15 22:00", "2025-01-15 22:05"], tz="UTC")
        pid = tb.period_id(t, "day")
        self.assertEqual(pid[1] - pid[0], 1)
        self.assertEqual(pid[3] - pid[2], 1)
        pe = tb.period_end(pid[[0, 2]], "day")
        self.assertEqual(list(pe), [t[0], t[2]])

    def test_hour_period(self):
        t = pd.DatetimeIndex(["2025-07-01 10:05", "2025-07-01 11:00", "2025-07-01 11:05"], tz="UTC")
        pid = tb.period_id(t, "hour")
        self.assertEqual(pid[0], pid[1]); self.assertEqual(pid[2] - pid[1], 1)
        self.assertEqual(tb.period_end(pid[:1], "hour")[0], pd.Timestamp("2025-07-01 11:00", tz="UTC"))

    def test_bars_5m_end_label(self):
        t = pd.date_range("2025-01-01 00:00", periods=10, freq="1min", tz="UTC")
        df = pd.DataFrame({"open_time": t, "open": np.arange(10.0), "high": np.arange(10.0) + 1,
                           "low": np.arange(10.0) - 1, "close": np.arange(10.0) + 0.5})
        b = tb.bars_5m(df)
        self.assertEqual(list(b.index), list(pd.DatetimeIndex(["2025-01-01 00:05", "2025-01-01 00:10"], tz="UTC")))
        self.assertEqual(b.iloc[0].tolist(), [0.0, 5.0, -1.0, 4.5])


class LevelTests(unittest.TestCase):
    def test_levels_use_completed_periods_only(self):
        p = tb.Params()
        b = tb.synthetic_5m(120, seed=1)
        d = tb.prepare_asset("BTC", b, None, p)
        cut = d.s[80]                                   # remove everything from period 80 on
        d2 = tb.prepare_asset("BTC", b.iloc[:cut + 1], None, p)
        for f in ("hi_n", "lo_n", "lo_x", "hi_x", "atr"):
            np.testing.assert_allclose(getattr(d, f)[:81], getattr(d2, f)[:81], equal_nan=True, err_msg=f)

    def test_level_definitions(self):
        p = tb.Params()
        d = tb.prepare_asset("BTC", tb.synthetic_5m(80, seed=2), None, p)
        k = 50
        self.assertAlmostEqual(d.hi_n[k], np.nanmax(d.ph[k - 20:k]))
        self.assertAlmostEqual(d.lo_x[k], np.nanmin(d.pl[k - 10:k]))
        fu = d.first_up[k]
        if fu >= 0:
            self.assertGreaterEqual(d.h[fu], d.hi_n[k])
            self.assertTrue((d.h[d.s[k]:fu] < d.hi_n[k]).all())


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = tb.Params()
        cls.data = build(cls.p, 600, trend=0.01)
        cls.E, cls.T, cls.st = tb.simulate(cls.data, cls.p, start=S0)

    def test_accounting_closes(self):
        T, E, p = self.T, self.E, self.p
        end = T[T.reason == "end"]
        exit_fees = ((p.fee + p.slip) * end.units * end.exit_px).sum()
        self.assertAlmostEqual(p.start_equity + T.pnl.sum() + exit_fees, E.iloc[-1], places=4)

    def test_fills_and_stops(self):
        for r in self.T[self.T.kind == "breakout"].itertuples():
            d = self.data[r.asset]
            self.assertGreaterEqual(r.entry_px, d.o[r.je] - 1e-9)          # stop order: never better than the open
            self.assertLessEqual(r.entry_px, d.h[r.je] + 1e-9)
            if r.reason in ("trail", "fixed_stop", "stop_same_period"):
                self.assertLessEqual(r.exit_px, d.h[r.jx] + 1e-9)
                self.assertGreaterEqual(r.exit_px, d.l[r.jx] - 1e-9)
        self.assertTrue((self.T.dir == 1).all())

    def test_sizing_is_two_percent_of_hwm(self):
        r = self.T.iloc[0]
        d = self.data[r.asset]
        k = int(tb.period_id(pd.DatetimeIndex([d.end[r.je]]), "day")[0]) - d.pid0
        self.assertAlmostEqual(r.units * self.p.stop_atr * d.atr[k], self.p.risk_frac * self.p.start_equity, places=6)

    def test_funding_sign(self):
        self.assertTrue((self.T.funding[self.T.hold_bars > 300] > 0).all())     # long pays positive funding
        ps = tb.Params(side="short")
        _, Ts, _ = tb.simulate(build(ps, 400, trend=0.01), ps, start=S0)
        self.assertTrue((Ts.dir == -1).all())
        self.assertTrue((Ts.funding[Ts.hold_bars > 300] < 0).all())          # short receives it

    def test_no_entries_when_random_probability_zero(self):
        E, T, _ = tb.simulate(self.data, self.p, start=S0, mode="random", p_entry={"BTC": 0.0, "ETH": 0.0},
                              rng=np.random.default_rng(0))
        self.assertEqual(len(T), 0)
        self.assertTrue(np.allclose(E.to_numpy(), self.p.start_equity))

    def test_causal_simulation(self):
        cut = pd.Timestamp("2021-01-01", tz="UTC")
        E2, T2, _ = tb.simulate(self.data, self.p, start=S0, end=cut)
        T1 = self.T[self.T.exit_time < cut - pd.Timedelta(days=2)]
        T2c = T2[T2.reason != "end"].iloc[:len(T1)]
        np.testing.assert_allclose(T1.pnl.to_numpy(), T2c.pnl.to_numpy())

    def test_metrics_and_benchmark(self):
        m = tb.metrics(self.E, self.T)
        for k in ("cagr", "sharpe", "max_dd", "profit_factor", "trades"):
            self.assertIn(k, m)
        bh = tb.buy_and_hold(self.data, self.p, S0, None, 1.0)
        self.assertGreater(len(bh), 300)
        sd = tb.sharpe_diff(self.E, bh, tb.Params(n_boot=200))
        self.assertLessEqual(sd["lo"], sd["diff"]); self.assertLessEqual(sd["diff"], sd["hi"])


class NullTests(unittest.TestCase):
    def test_nulls_run_and_random_walk_is_not_flagged(self):
        p = tb.Params(n_twin=400, n_null=40, n_boot=200)
        data = build(p, 700, trend=0.0, rate=0.0)
        E, T, st = tb.simulate(data, p, start=S0)
        tw = tb.trade_twins(data, T, p, seed=1)
        self.assertEqual(list(tw.asset), ["BTC", "ETH", "ALL"])
        self.assertTrue(((tw.share_null_ge_real >= 0) & (tw.share_null_ge_real <= 1)).all())
        N = tb.random_portfolios(data, p, st["p_entry"], S0, n=40, seed=2)
        dec = tb.decide(tb.metrics(E, T), N, tb.sharpe_diff(E, tb.buy_and_hold(data, p, S0, None, 0.3), p), p)
        self.assertIn(dec["verdict"], ("NO_EDGE", "RISK_PROFILE_ONLY", "TIMING_ONLY", "TREND_EDGE"))
        self.assertEqual(dec["G1_beats_luck"], "FAIL")

    def test_hour_mode_and_leverage_cap(self):
        p = tb.Params(period="hour")
        data = build(p, 120, trend=0.0)
        E, T, st = tb.simulate(data, p, start=S0)
        self.assertGreater(len(T), 50)
        self.assertLessEqual(st["max_gross_leverage"], 2 * p.max_pos_lev + 1e-6 + 2.0)
        self.assertTrue((E > -1e-6).all() or E.min() > -p.start_equity)


if __name__ == "__main__":
    unittest.main()
