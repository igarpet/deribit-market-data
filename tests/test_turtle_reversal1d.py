"""Tests for Turtle Breakout 19 (turtle_reversal1d).

Run from repo root: python -m unittest discover -s tests -p "test_turtle_reversal1d.py" -v
"""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_engineering.modeling import turtle_reversal1d as rv

RP = replace(rv.RParams(), n_null=2000)


def series(n=2000, phi=0.0, seed=0, flat_share=0.6):
    rng = np.random.default_rng(seed); e = rng.normal(0, 0.035, n); r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + e[t]
    days = pd.date_range("2015-01-01", periods=n, freq="D")
    C = 100 * np.exp(np.cumsum(r)); O = np.r_[100, C[:-1]]
    d = pd.DataFrame({"open": O, "high": np.maximum(O, C) * 1.01, "low": np.minimum(O, C) * 0.99, "close": C}, index=days)
    flat = pd.Series(rng.random(n) < flat_share, index=days)
    return d, pd.Series(np.log(d.close).diff().to_numpy(), index=days), flat


class Tests(unittest.TestCase):
    def test_signals_causal_and_share(self):
        d, r, flat = series()
        s1 = rv.signals(r, flat, RP)
        r2 = r.copy(); r2.iloc[1500:] = r2.iloc[1500:] * 10 + 1
        s2 = rv.signals(r2, flat, RP)
        pd.testing.assert_series_equal(s1.iloc[:1500], s2.iloc[:1500])          # future data cannot change past signals
        share = s1[flat].iloc[400:].mean()
        self.assertAlmostEqual(share, 0.2, delta=0.05)
        self.assertFalse(s1[~flat].any())

    def test_trade_math_and_funding(self):
        days = pd.date_range("2020-01-01", periods=40, freq="D")
        C = np.full(40, 100.0); C[30] = 100.0; C[31] = 102.0
        d = pd.DataFrame({"open": C, "high": C * 1.01, "low": C * 0.99, "close": C}, index=days)
        sig = pd.Series(False, index=days); sig.iloc[30] = True
        T = rv.trades(d, sig, RP, funding=pd.Series(0.0001, index=days))
        self.assertEqual(len(T), 1)
        self.assertAlmostEqual(T.gross.iloc[0], 0.02); self.assertAlmostEqual(T.ret.iloc[0], 0.02 - 0.0006 * 2.02 - 0.0001)
        Ts = rv.trades(d, sig, RP, side="short", funding=pd.Series(0.0001, index=days))
        self.assertAlmostEqual(Ts.ret.iloc[0], -0.02 - 0.0006 * 2.02 + 0.0001)

    def test_random_walk_fails(self):
        d, r, flat = series(phi=0.0, seed=3)
        T = rv.trades(d, rv.signals(r, flat, RP), RP); pool = rv.trades(d, flat & r.notna(), RP)
        self.assertFalse(rv.gate(T, pool, RP, 1)["pass"])

    def test_reversion_passes(self):
        d, r, flat = series(n=3000, phi=-0.25, seed=4)
        T = rv.trades(d, rv.signals(r, flat, RP), RP); pool = rv.trades(d, flat & r.notna(), RP)
        g = rv.gate(T, pool, RP, 1)
        self.assertTrue(g["pass"]); self.assertGreater(g["mean_R"], g["null_mean_R"])

    def test_account_and_daily(self):
        d, r, flat = series(n=800, phi=-0.25, seed=5)
        T = rv.trades(d, rv.signals(r, flat, RP), RP)
        E = rv.account(T, d.index)
        self.assertEqual(len(E), len(d)); self.assertGreater(E.iloc[-1], 0)
        df1m = pd.DataFrame({"open_time": pd.date_range("2021-01-01", periods=3 * 1440, freq="min", tz="UTC")})
        df1m["open"] = df1m["high"] = df1m["low"] = df1m["close"] = np.arange(len(df1m), dtype=float) + 1
        D = rv.daily_from_1m(df1m)
        self.assertEqual(len(D), 3); self.assertEqual(D.close.iloc[0], 1440.0); self.assertEqual(D.open.iloc[1], 1441.0)
        f = pd.DataFrame({"t": pd.to_datetime(["2021-01-01 08:00", "2021-01-01 16:00", "2021-01-02 00:00"], utc=True), "rate": [1e-4, 2e-4, 3e-4]})
        fd = rv.daily_funding(f, D.index); self.assertAlmostEqual(fd.iloc[0], 3e-4); self.assertAlmostEqual(fd.iloc[1], 3e-4)


if __name__ == "__main__":
    unittest.main()
