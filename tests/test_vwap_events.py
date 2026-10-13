"""Synthetic tests for VWAP Events 01 (vwap_events).

Run from repo root: python -m unittest discover -s tests -p "test_vwap_events.py" -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_engineering.modeling import vwap_events as ve

P = ve.Params(n_placebo=2, n_boot=300, sigma_min_periods=30)


def bars(close, high=None, low=None, vol=None, start="2026-01-05 08:00"):
    close = np.asarray(close, float)
    n = len(close)
    vol = np.ones(n) if vol is None else np.asarray(vol, float)
    return pd.DataFrame({"open_time": pd.date_range(start, periods=n, freq="1min", tz="UTC"),
                         "open": close, "high": close if high is None else np.asarray(high, float),
                         "low": close if low is None else np.asarray(low, float), "close": close,
                         "volume": vol, "quote_volume": vol * close})


class TestVWAP(unittest.TestCase):
    def test_anchored_vwap_exact_and_resets(self):
        close = np.r_[np.full(3, 100.0), np.full(3, 110.0)]
        df = bars(close, vol=[1, 1, 2, 1, 1, 1], start="2026-01-05 11:57")
        fr = ve.prepare(df, P)
        v, since, aid = ve.anchored_vwap(fr, 240, 0)        # anchor at 12:00
        self.assertAlmostEqual(v[2], 100.0)
        self.assertEqual(since[3], 0)                       # 12:00 is a new anchor
        self.assertAlmostEqual(v[3], 110.0)                 # reset, not a blend with 100
        self.assertNotEqual(aid[2], aid[3])

    def test_missing_minutes_are_reindexed(self):
        df = bars(np.linspace(100, 101, 10)).drop(index=[4, 5])
        fr = ve.prepare(df, P)
        self.assertEqual(len(fr), 10)
        self.assertTrue(fr["close"].isna().sum() == 2)


class TestDetector(unittest.TestCase):
    def make(self):
        # 200 quiet bars around 100, then 40 bars well above, then one bar whose low dips to the level.
        rng = np.random.default_rng(0)
        c = 100 * np.exp(np.cumsum(rng.normal(0, 2e-4, 200)))
        c = np.r_[c, np.full(40, c[-1] * 1.01), c[-1] * 1.01]
        hi, lo = c.copy(), c.copy()
        df = bars(c, hi, lo, vol=np.r_[np.full(200, 100.0), np.full(41, 0.01)])
        fr = ve.prepare(df, P)
        v, since, _ = ve.anchored_vwap(fr, 1440, 8 * 60)
        return df, fr, v, since

    def test_no_touch_without_reaching_level(self):
        _, fr, v, since = self.make()
        idx, _ = ve.detect_touches(fr, v, since, 60, P)
        self.assertEqual(len(idx), 0)

    def test_touch_from_above_detected_with_known_level_only(self):
        df, fr, v, since = self.make()
        lev_prev = v[-2]
        df.loc[df.index[-1], "low"] = lev_prev * 0.9999       # touches the level known at the bar's open
        fr2 = ve.prepare(df, P)
        v2, s2, _ = ve.anchored_vwap(fr2, 1440, 8 * 60)
        idx, side = ve.detect_touches(fr2, v2, s2, 60, P)
        self.assertEqual(list(idx), [len(fr2) - 1])
        self.assertEqual(side[0], 1.0)

    def test_cooldown(self):
        p = ve.Params(cooldown=15)
        n = 40
        fr = pd.DataFrame({"high": np.full(n, 2.0), "low": np.zeros(n), "logc": np.zeros(n),
                           "sigma": np.full(n, 1e-6)},
                          index=pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC"))
        level = np.full(n, 1.0)                                # closes at 1.0 -> log gap 0 -> side 0
        fr["logc"] = np.log(np.full(n, 1.5))                   # always above the level, far away
        idx, _ = ve.detect_touches(fr, level, np.arange(n) + 100, 60, p)
        self.assertTrue(np.all(np.diff(idx) > p.cooldown))

    def test_offset_placebo_constant_within_window(self):
        df = ve.synthetic_klines(n_days=3, seed=1)
        fr = ve.prepare(df, P)
        v, since, aid = ve.anchored_vwap(fr, 240, 0)
        lv = ve.offset_levels(fr, v, aid, np.random.default_rng(0), P)
        ratio = pd.Series(np.round(np.log(lv / v), 10)).groupby(aid).nunique(dropna=True)
        self.assertTrue((ratio <= 1).all())


class TestStatistics(unittest.TestCase):
    def table(self, effect, n_days=120, seed=0):
        rng = np.random.default_rng(seed)
        days = pd.date_range("2026-01-01", periods=n_days, freq="1D", tz="UTC")
        rows = []
        for d in days:
            dv = rng.normal(0, 0.1)                           # shared day shock
            for _ in range(6):
                rows.append({"day": d, "kind": "real", "y15": dv + effect + rng.normal(0, 0.4)})
            for kind in ("offset", "phase"):
                for _ in range(20):
                    rows.append({"day": d, "kind": kind, "y15": dv + rng.normal(0, 0.4)})
        ev = pd.DataFrame(rows)
        ev["anchor"], ev["window"] = "h4", "test"
        return ev

    def test_detects_known_effect(self):
        out = ve.readout(self.table(np.log(1.25)), P)
        self.assertTrue(out["G1"].all() and out["G2"].all())
        self.assertTrue(np.allclose(out["ratio"], 1.25, atol=0.06))

    def test_null_does_not_pass(self):
        out = ve.readout(self.table(0.0, seed=3), P)
        self.assertFalse((out["G1"] & out["G2"]).all())


class TestPipelineSmoke(unittest.TestCase):
    def test_null_synthetic_end_to_end(self):
        df = ve.synthetic_klines(n_days=20, seed=4)
        p = ve.Params(n_placebo=3, n_boot=200)
        ev, info = ve.run_asset(df, "SYN", p)
        self.assertIn("real", set(ev["kind"]))
        ev["window"] = "test"
        dec = ve.decision(ve.readout(ev, p))
        self.assertEqual(len(dec), len(p.anchors))
        self.assertGreater(info["bars"], 20 * 1440 - 5)


if __name__ == "__main__":
    unittest.main()
