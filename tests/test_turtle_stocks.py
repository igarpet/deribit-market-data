"""Tests for Turtle Breakout 14 (turtle_stocks) and the portfolio gross cap.

Run from repo root: python -m unittest discover -s tests -p "test_turtle_stocks.py" -v
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
from data_engineering.modeling import turtle_reverse as tr
from data_engineering.modeling import turtle_sizing as ts
from data_engineering.modeling import turtle_stocks as tk

warnings.filterwarnings("ignore", category=RuntimeWarning)
P = replace(tb.Params(), min_bar_share=0.0, funding=False, fee=0.0004)


def daily_ohlc(n=1500, seed=0, drift=0.0004, vol=0.02, start="2000-01-03"):
    rng = np.random.default_rng(seed)
    ses = pd.bdate_range(start, periods=n)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    o = c * np.exp(rng.normal(0, vol / 3, n))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, vol / 2, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, vol / 2, n)))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=ses), ses


class CalendarTests(unittest.TestCase):
    def test_pseudo_bars_shape_and_order(self):
        d, ses = daily_ohlc(50)
        b = tk.pseudo_bars(d, ses)
        self.assertEqual(len(b), 150)
        first = b.iloc[:3]
        self.assertEqual(first.open.iloc[0], d.open.iloc[0]); self.assertEqual(first.close.iloc[0], d.high.iloc[0])
        self.assertEqual(first.low.iloc[1], d.low.iloc[0]); self.assertEqual(first.close.iloc[2], d.close.iloc[0])
        pid = tb.period_id(pd.DatetimeIndex(b.index), "day")
        self.assertEqual(len(np.unique(pid)), 50)                                   # one period per session
        self.assertTrue((np.diff(np.unique(pid)) == 1).all())                       # consecutive: no weekends

    def test_period_ohlc_and_atr_equal_daily(self):
        d, ses = daily_ohlc(300, seed=1)
        D = tb.prepare_asset("X", tk.pseudo_bars(d, ses), None, P)
        np.testing.assert_allclose(D.ph, d.high.to_numpy()); np.testing.assert_allclose(D.pl, d.low.to_numpy())
        np.testing.assert_allclose(D.pc, d.close.to_numpy())
        prev = d.close.shift(1)
        trr = np.nanmax(np.c_[(d.high - d.low).to_numpy(), (d.high - prev).abs().to_numpy(), (d.low - prev).abs().to_numpy()], axis=1)
        trr[0] = d.high.iloc[0] - d.low.iloc[0]
        np.testing.assert_allclose(D.atr[1:], tb._rma(trr, 20)[:-1], rtol=1e-9, equal_nan=True)

    def test_to_real_roundtrip(self):
        d, ses = daily_ohlc(40)
        b = tk.pseudo_bars(d, ses)
        r = tk.to_real(pd.DatetimeIndex(b.index), ses)
        self.assertEqual(r[0], ses[0]); self.assertEqual(r[-1], ses[-1])
        self.assertTrue(set(r) <= set(ses))

    def test_missing_session_and_bad_bars(self):
        d, ses = daily_ohlc(60)
        d2 = d.drop(d.index[10]); d2.iloc[5, 1] = d2.iloc[5, 3] * 0.5           # high below close is repaired
        b = tk.pseudo_bars(d2, ses)
        self.assertEqual(len(b), 59 * 3)
        self.assertTrue((b.high >= b[["open", "close"]].max(axis=1) - 1e-12).all())


class StrategyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.books, cls.ses = {}, None
        for i in range(5):
            d, ses = daily_ohlc(2000, seed=10 + i, drift=0.0006)
            cls.ses = ses
            D = tb.prepare_asset(f"S{i}", tk.pseudo_bars(d, ses), None, P)
            start = D.end[0] + pd.Timedelta(days=30)
            L1 = ts.trade_list(D, P, start, replace(tr.RParams(), side_cost=0.0005), with_reverse=True)
            cls.books[f"S{i}"] = tp.make_book(D, L1, P)

    def test_same_day_entry_then_stop_is_adverse(self):
        T = pd.concat([b.trades for b in self.books.values()])
        same = T[(T.kind == "long") & (T.pid_in == T.pid_out)]
        if len(same):
            self.assertTrue((same.px_out <= same.px_in).all())

    def test_ticker_without_trades(self):
        d, ses = daily_ohlc(200, seed=3, drift=-0.002, vol=0.001)              # steady decline: no 20-day high breakout
        D = tb.prepare_asset("Z", tk.pseudo_bars(d, ses), None, P)
        start = D.end[0] + pd.Timedelta(days=30)
        for rev in (False, True):
            L = ts.trade_list(D, P, start, replace(tr.RParams(), side_cost=0.0005), with_reverse=rev)
            self.assertEqual(len(L), 0)
            bk = tk.apply_borrow(tp.make_book(D, L, P), 0.005, ses)
            self.assertEqual(len(bk.trades), 0)
        o = tp.simulate({**self.books, "Z": bk}, "R0", replace(tp.PParams(), size_base="equity", f=0.005), min(b.trades.t_in.min() for b in self.books.values()))
        self.assertGreater(o["stats"]["entries"], 0)

    def test_borrow_charges_shorts_only(self):
        b = self.books["S0"]
        nb = tk.apply_borrow(b, 0.05, self.ses)
        L = b.trades.kind == "long"
        np.testing.assert_allclose(nb.trades.R[L], b.trades.R[L])
        self.assertTrue((nb.trades.R[~L] < b.trades.R[~L]).all())

    def test_gross_cap(self):
        pp = replace(tp.PParams(), size_base="equity", f=0.02, gross_cap=0.5)
        t0 = min(b.trades.t_in.min() for b in self.books.values())
        o = tp.simulate(self.books, "R0", pp, t0)
        o0 = tp.simulate(self.books, "R0", replace(pp, gross_cap=None), t0)
        self.assertGreater(o["stats"]["skipped"] + o["stats"]["scaled"], 0)
        self.assertLess(o["max_gross_lev"], o0["max_gross_lev"])
        self.assertLess(o["max_gross_lev"], 0.5 * 1.6)                              # entry notional cap; marks drift

    def test_real_daily_and_metrics(self):
        pp = replace(tp.PParams(), size_base="equity", f=0.005, gross_cap=2.0)
        t0 = min(b.trades.t_in.min() for b in self.books.values())
        D = tp.daily_curve(tp.simulate(self.books, "R0", pp, t0)["equity"], pp.start_equity)
        R = tk.real_daily(D, self.ses)
        self.assertTrue(set(R.index) <= set(self.ses))
        m = tk.metrics(R, pp.start_equity)
        self.assertGreater(m["years"], 5); self.assertTrue(np.isfinite(m["sharpe"]))


if __name__ == "__main__":
    unittest.main()
