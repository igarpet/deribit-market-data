"""Synthetic tests for Turtle Breakout 05 (turtle_fractal).

Run from repo root: python -m unittest discover -s tests -p "test_turtle_fractal.py" -v
"""
import math
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_engineering.modeling import turtle_breakout as tb
from data_engineering.modeling import turtle_fractal as fr

warnings.filterwarnings("ignore", category=RuntimeWarning)
FP = fr.FrParams()


def martingale_bars(days, seed, base_min=5, sub=30, vol=0.6, start="2023-01-01"):
    """Bars from a fine log-Brownian path (true highs/lows): no edge for any stopping rule."""
    rng = np.random.default_rng(seed)
    per_day = 1440 // base_min
    n = days * per_day
    sig = vol / math.sqrt(365 * per_day)
    path = math.log(100) + np.cumsum(rng.normal(0, sig / math.sqrt(sub), (n, sub)).ravel()).reshape(n, sub)
    o = np.r_[math.log(100), path[:-1, -1]]
    t = pd.date_range(pd.Timestamp(start, tz="UTC") + pd.Timedelta(minutes=base_min), periods=n, freq=f"{base_min}min")
    return pd.DataFrame({"open": np.exp(o), "high": np.exp(np.maximum(path.max(1), o)),
                         "low": np.exp(np.minimum(path.min(1), o)), "close": np.exp(path[:, -1])},
                        index=pd.DatetimeIndex(t, name="end"))


class CandleTests(unittest.TestCase):
    def test_candle_ids(self):
        t = pd.DatetimeIndex(["2025-01-01 00:01", "2025-01-01 00:02", "2025-01-01 00:03", "2025-01-01 00:04"], tz="UTC")
        np.testing.assert_array_equal(np.diff(fr.candle_id(t, 1)), [1, 1, 1])
        c3 = fr.candle_id(t, 3)
        self.assertEqual(c3[0], c3[2]); self.assertEqual(c3[3] - c3[2], 1)
        self.assertEqual(fr.base_for(3), 1); self.assertEqual(fr.base_for(5), 5)

    def test_levels_use_completed_candles(self):
        F = fr.prepare(martingale_bars(30, 1), None, 15, FP)
        k = 500
        self.assertAlmostEqual(F.hi_n[k], F.ph[k - 20:k].max())
        self.assertAlmostEqual(F.lo_x[k], F.pl[k - 10:k].min())
        fu = F.first_up[k]
        if fu >= 0:
            self.assertGreaterEqual(F.h[fu], F.hi_n[k])
            self.assertTrue((F.h[F.s[k]:fu] < F.hi_n[k]).all())


class SimulationTests(unittest.TestCase):
    def test_daily_matches_tb01_trade_bars(self):
        b = tb.synthetic_5m(1200, seed=7, trend=0.01, start="2020-01-01")
        p = tb.Params()
        _, T1, _ = tb.simulate({"BTC": tb.prepare_asset("BTC", b, None, p)}, p, start=tb.COMMON_START)
        T1 = T1[T1.reason != "end"]
        F = fr.prepare(b, None, 1440, FP)
        T2, _ = fr.simulate(F, FP, fr.k_start_for(F, tb.COMMON_START))
        self.assertGreater(len(T1), 10)
        # same entry bars; exits agree on most trades (fills now at bar close, so a later stop can differ)
        self.assertEqual(list(T1.je), list(T2.je[:len(T1)]))
        self.assertGreater((T1.jx.to_numpy() == T2.jx[:len(T1)].to_numpy()).mean(), 0.8)

    def test_fills_and_costs(self):
        F = fr.prepare(martingale_bars(60, 2), None, 15, FP)
        T, st = fr.simulate(F, FP, 100)
        np.testing.assert_allclose(T.entry_px, F.c[T.je]); np.testing.assert_allclose(T.exit_px, F.c[T.jx])
        self.assertTrue((T.jx > T.je).all())
        np.testing.assert_allclose(T.bp_gross - T.bp_net, 1e4 * FP.side_cost * (1 + T.exit_px / T.entry_px) + T.bp_funding)
        self.assertGreater(st["p_entry"], 0)

    def test_martingale_has_no_edge(self):
        m, se = [], []
        for seed in range(4):
            F = fr.prepare(martingale_bars(300, 100 + seed), None, 5, FP)
            T, _ = fr.simulate(F, FP, 100)
            m.append(T.R_gross.mean()); se.append(T.R_gross.std() / math.sqrt(len(T)))
        self.assertLess(abs(np.mean(m)), 3 * np.mean(se) / 2)

    def test_one_minute_base(self):
        F = fr.prepare(martingale_bars(20, 3, base_min=1, sub=12), None, 2, FP)
        self.assertEqual(F.base, 1)
        T, _ = fr.simulate(F, FP, 50)
        self.assertGreater(len(T), 100)

    def test_trend_beats_null_daily(self):
        frames = {a: fr.prepare(tb.synthetic_5m(1500, seed=s, trend=0.01, start="2020-01-01"), None, 1440, FP)
                  for a, s in (("A", 3), ("B", 4), ("C", 5))}
        k0 = {a: 60 for a in frames}
        tot, cnt, pe = 0.0, 0, {}
        for a, F in frames.items():
            T, st = fr.simulate(F, FP, k0[a]); tot += T.R_gross.sum(); cnt += len(T); pe[a] = st["p_entry"]
        null = fr.null_means(frames, FP, pe, k0, 40, 1)
        self.assertGreater(tot / cnt, np.quantile(null, 0.9))

    def test_summary(self):
        F = fr.prepare(martingale_bars(60, 4), None, 15, FP)
        T, st = fr.simulate(F, FP, 100)
        null = fr.null_means({"X": F}, FP, {"X": st["p_entry"]}, {"X": 100}, 10, 2)
        s = fr.summarize(15, T, null, fr.FrParams(n_boot=200), 60.0)
        for k in ("mean_R_gross", "edge_R", "breakeven_rt_cost_bp", "mean_bp_net", "G_edge", "G_net", "verdict"):
            self.assertIn(k, s)
        self.assertEqual(s["verdict"], "NO_EDGE")


if __name__ == "__main__":
    unittest.main()
