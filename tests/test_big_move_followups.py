"""Synthetic tests for Regime Structure 04/05 (big_move_followups).

Run from repo root: python -m unittest discover -s tests -p "test_big_move_followups.py" -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_engineering.modeling import big_move_days as bm
from data_engineering.modeling import big_move_followups as bf

P = bf.Params(n_boot=400, n_null=199)   # two-sided p can reach 0.01 only with >= 199 draws


class TestRS04(unittest.TestCase):
    def frame(self, beta, seed=0, n_sess=40):
        rng = np.random.default_rng(seed)
        rows = []
        for s in range(n_sess):
            day_shock = rng.normal(0, 0.3)
            for a in ("BTC", "ETH", "SOL", "XRP"):
                x = rng.normal(0, 0.3)
                rows.append({"session": f"2026-09-{s:02d}", "underlying": a, "x": x,
                             "y": beta * x + day_shock + rng.normal(0, 0.4)})
        return pd.DataFrame(rows)

    def test_slope_recovered_and_verdicts(self):
        d = self.frame(0.8)
        s = bf.session_boot_slope(d, "y", P)
        self.assertAlmostEqual(s["beta"], 0.8, delta=0.2)
        self.assertEqual(bf.rs04_verdict(s), "UNDERREACTION")
        self.assertEqual(bf.rs04_verdict(bf.session_boot_slope(self.frame(0.0, seed=2), "y", P)), "NO_EVIDENCE")
        self.assertEqual(bf.rs04_verdict(bf.session_boot_slope(self.frame(-0.8, seed=3), "y", P)), "OVERREACTION")

    def test_rs04_frame_units(self):
        ent = pd.DataFrame({"session": ["2026-09-21"], "underlying": ["BTC"], "S_T": [101.0], "F": [100.0],
                            "T": [20 * 60 / ls_year()], "iv_straddle_mid": [0.5]})
        pred = pd.DataFrame({"session": ["2026-09-21"], "underlying": ["BTC"], "x": [0.1], "x_raw": [np.exp(0.1)],
                             "sigma_hat": [0.03], "window_rs03": ["test"]})
        d = bf.rs04_frame(ent, pred, (-0.1, 0.1))
        T = 20 * 60 / ls_year()
        impl = 0.5 * np.sqrt(T) * np.sqrt(2 / np.pi)
        self.assertAlmostEqual(d["y"].iloc[0], np.log(np.log(1.01) / impl), places=12)
        self.assertAlmostEqual(d["y_impl"].iloc[0], np.log(impl / (0.03 * np.sqrt(20 / 24))), places=10)
        self.assertEqual(d["tercile"].iloc[0], "mid")


def ls_year():
    from data_engineering.modeling import vrp_long_straddle as ls
    return ls.YEAR_MIN


class TestRS05(unittest.TestCase):
    def table(self, df):
        b = bm.session_bars(df)
        t, _, _ = bm.session_table(b, bm.Params())
        t, thr = bm.label_big(t, bm.Params())
        t["window"] = np.where(t["session"] >= pd.Timestamp("2025-08-01", tz="UTC"), "test", "dev")
        return b, t, thr

    def test_signs_shared_and_preserve_abs(self):
        s = pd.DatetimeIndex(pd.date_range("2025-01-02", periods=5, freq="D", tz="UTC"))
        a = bf.null_signs(bf.day_index(s), 4, P)
        b = bf.null_signs(bf.day_index(s[2:]), 4, P)
        self.assertTrue(np.array_equal(a[2:], b))

    def test_random_walk_not_significant_trending_is(self):
        b, t, thr = self.table(bm.synthetic_klines(n_days=420, seed=5, planted=False, start="2025-01-01"))
        ev = bf.rs05_evaluate([bf.rs05_asset(b, t, thr, P)], P).set_index(["window", "stat"])
        self.assertFalse(ev.loc[("test", "S1_VR_all"), "significant"])
        # plant intraday trending: AR(1) on 5m returns, phi = 0.3
        r = b["r5"].copy()
        for i in range(1, r.shape[1]):
            r[:, i] = r[:, i] + 0.3 * r[:, i - 1]
        b2 = dict(b, r5=r)
        ev2 = bf.rs05_evaluate([bf.rs05_asset(b2, t, thr, P)], P).set_index(["window", "stat"])
        self.assertTrue(ev2.loc[("test", "S1_VR_all"), "significant"])
        self.assertGreater(ev2.loc[("test", "S1_VR_all"), "real_over_null"], 1.3)


if __name__ == "__main__":
    unittest.main()
