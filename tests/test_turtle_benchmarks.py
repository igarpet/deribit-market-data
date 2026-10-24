"""Synthetic tests for Turtle Breakout 02 (turtle_benchmarks).

Run from repo root: python -m unittest discover -s tests -p "test_turtle_benchmarks.py" -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_engineering.modeling import turtle_breakout as tb
from data_engineering.modeling import turtle_benchmarks as bm

P = tb.Params()
COST = P.fee + P.slip


def build(trend, rate=0.0001, days=1500, seed0=3):
    return {a: tb.prepare_asset(a, b, tb.synthetic_funding(b, rate), P) for a, b in
            ((a, tb.synthetic_5m(days, seed=seed0 + k, trend=trend, start="2020-01-01"))
             for k, a in enumerate(("BTC", "ETH", "SOL", "XRP")))}


class MixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build(0.0)
        cls.C, cls.F = bm.daily_panel(cls.data, tb.COMMON_START, None)

    def test_panel_is_17h_new_york(self):
        loc = self.C.index.tz_convert(tb.NY)
        self.assertTrue((loc.hour == 17).all() and (loc.minute == 0).all())
        self.assertGreaterEqual(self.C.index[0], tb.COMMON_START)

    def test_single_asset_daily_no_cost_tracks_price(self):
        s = bm.constant_mix(self.C[["BTC"]], self.F[["BTC"]], 1.0, "D", False, 0.0)
        np.testing.assert_allclose(s.to_numpy() / s.iloc[0], self.C.BTC.to_numpy() / self.C.BTC.iloc[0], rtol=1e-10)

    def test_zero_leverage_is_flat(self):
        s = bm.constant_mix(self.C, self.F, 0.0, "M", True, COST)
        self.assertTrue(np.allclose(s.to_numpy(), 100_000.0))

    def test_funding_costs_longs(self):
        perp = bm.constant_mix(self.C, self.F, 1.0, "M", True, COST)
        spot = bm.constant_mix(self.C, self.F, 1.0, "M", False, COST)
        self.assertLess(perp.iloc[-1], spot.iloc[-1])

    def test_rebalancing_costs(self):
        m = bm.constant_mix(self.C, self.F, 1.0, "M", False, 0.0)
        mc = bm.constant_mix(self.C, self.F, 1.0, "M", False, COST)
        d = bm.constant_mix(self.C, self.F, 1.0, "D", False, COST)
        self.assertLess(mc.iloc[-1], m.iloc[-1])
        nm = self.C.index.tz_convert(tb.NY).month
        self.assertGreater(len(set(zip(self.C.index.year, nm))), 30)
        self.assertTrue(np.isfinite(d.iloc[-1]))

    def test_inverse_vol_weights_run(self):
        s = bm.constant_mix(self.C, self.F, 1.0, "M", True, COST, weights="invvol")
        self.assertTrue(np.isfinite(s).all())

    def test_vol_matching(self):
        S, L = bm.vol_matched(self.C, self.F, 0.25, COST)
        self.assertAlmostEqual(bm.realised_vol(S), 0.25, delta=0.005)


class StatTests(unittest.TestCase):
    def test_alpha_beta_recovers_beta(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2021-01-01", periods=1500, freq="1D", tz="UTC")
        x = rng.normal(0, 0.03, len(idx)); y = 0.002 + 0.5 * x + rng.normal(0, 0.01, len(idx))
        Eb = pd.Series(100 * np.cumprod(1 + x), index=idx); Es = pd.Series(100 * np.cumprod(1 + y), index=idx)
        ab = bm.alpha_beta(Es, Eb)
        self.assertAlmostEqual(ab["beta"], 0.5, delta=0.03)
        self.assertGreater(ab["t_alpha_nw"], 3.0); self.assertAlmostEqual(ab["alpha_daily"], 0.002, delta=0.0008)

    def test_sharpe_diff_identical_is_zero(self):
        idx = pd.date_range("2021-01-01", periods=800, freq="1D", tz="UTC")
        E = pd.Series(100 * np.cumprod(1 + np.random.default_rng(1).normal(0, 0.02, 800)), index=idx)
        r = bm.sharpe_diff(E, E, bm.BParams(n_boot=200), 0)
        self.assertAlmostEqual(r["diff"], 0.0); self.assertFalse(r["pass"])


class EndToEnd(unittest.TestCase):
    def _verdict(self, trend):
        data = build(trend)
        E, T, st = tb.simulate(data, P, start=tb.COMMON_START)
        C, F = bm.daily_panel(data, tb.COMMON_START, None)
        L = st["mean_gross_leverage"]
        B = {"B1": bm.constant_mix(C, F, L, "M", True, COST), "B2": bm.constant_mix(C, F, L, "D", True, COST),
             "B3": bm.constant_mix(C, F, L, "M", False, COST)}
        bp = bm.BParams(n_boot=300)
        return bm.decide({k: bm.sharpe_diff(E, B[k], bp, i) for i, k in enumerate(B)})["verdict"]

    def test_trending_market_beats(self):
        self.assertEqual(self._verdict(0.01), "BEATS_ALL_BENCHMARKS")

    def test_random_walk_does_not_beat(self):
        self.assertNotEqual(self._verdict(0.0), "BEATS_ALL_BENCHMARKS")


if __name__ == "__main__":
    unittest.main()
