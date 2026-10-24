"""Synthetic tests for VWAP Drift 01 (vwap_drift).

Run from repo root: python -m unittest discover -s tests -p "test_vwap_drift.py" -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_engineering.modeling import vwap_drift as vd
from data_engineering.modeling import vwap_events as ve
from data_engineering.modeling import markov_har_variance as v18

MANIFEST = ROOT / "notebooks" / "Volatility_Dynamics" / "18" / "18_manifest.json"
FITS = vd.load_engine(MANIFEST)


class EngineTests(unittest.TestCase):
    def test_predict_matches_vd18_algebra(self):
        df = vd.synthetic_klines(10, seed=1)
        fr = ve.prepare(df, ve.Params())
        z = vd.engine_frame(fr).dropna().copy()
        z["underlying"] = "BTC"
        for h in vd.ENGINE_H:
            a = vd.predict_har(FITS[h], z, "BTC")
            b = v18.predict_har(z, FITS[h])
            np.testing.assert_allclose(a, b, rtol=1e-12)

    def test_engine_features_are_causal(self):
        df = vd.synthetic_klines(8, seed=2)
        fr = ve.prepare(df, ve.Params())
        z = vd.engine_frame(fr)
        cut = len(fr) // 2
        z2 = vd.engine_frame(fr.iloc[:cut])
        common = z2.index
        cols = [f"har_logrv_{k}" for k in (30, 60, 240, 1440)]
        np.testing.assert_allclose(z.loc[common, cols].to_numpy(), z2[cols].to_numpy(), equal_nan=True)

    def test_engine_grid_is_bar_end(self):
        df = vd.synthetic_klines(3, seed=3)
        fr = ve.prepare(df, ve.Params())
        z = vd.engine_frame(fr)
        self.assertTrue((z.index.minute % 5 == 0).all())
        # the 5m close at 00:05 is the close of the 1m bar opening at 00:04
        self.assertEqual(z.loc[pd.Timestamp("2025-12-01 00:05", tz="UTC"), "close"],
                         fr.loc[pd.Timestamp("2025-12-01 00:04", tz="UTC"), "close"])

    def test_variance_interpolation(self):
        v = np.array([[15.0, 60.0, 240.0], [30.0, 60.0, 120.0]])     # rates 1,1,1 and 2,1,0.5 per minute
        np.testing.assert_allclose(vd.variance_to(np.array([60.0, 60.0]), v), [60.0, 60.0])
        np.testing.assert_allclose(vd.variance_to(np.array([1440.0, 1440.0]), v), [1440.0, 720.0])
        np.testing.assert_allclose(vd.variance_to(np.array([30.0, 30.0]), v)[0], 30.0)
        self.assertTrue(1.0 * 30 < vd.variance_to(np.array([30.0]), v[1:])[0] < 2.0 * 30)

    def test_next_expiry(self):
        t = pd.DatetimeIndex(["2026-01-01 07:55", "2026-01-01 08:00", "2026-01-01 08:05"], tz="UTC")
        e = vd.next_expiry(t)
        self.assertEqual(list(e), [pd.Timestamp("2026-01-01 08:00", tz="UTC"),
                                   pd.Timestamp("2026-01-02 08:00", tz="UTC"),
                                   pd.Timestamp("2026-01-02 08:00", tz="UTC")])


class OriginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = vd.synthetic_klines(20, seed=4)
        cls.A, cls.info, cls.O, cls.R = vd.run_asset(cls.df, "BTC", FITS, vd.Params(), keep_rows=True)
        cls.fr = ve.prepare(cls.df, ve.Params())

    def test_outcome_definition(self):
        logc = self.fr["logc"].to_numpy()
        r = self.R[(self.R.h == "60") & (self.R.kind == "real")].iloc[100]
        i = int(r.i)
        y = -np.sign(r.D) * (logc[i + 1 + 60] - logc[i + 1]) * 1e4
        self.assertAlmostEqual(r.y, y, places=8)
        self.assertEqual(int(r.exit), i + 61)

    def test_exp_exit_is_0800_bar(self):
        rr = self.R[self.R.h == "EXP"]
        end_exit = self.fr.index[rr.exit.to_numpy()] + pd.Timedelta(minutes=1)
        self.assertTrue(((end_exit.hour == 8) & (end_exit.minute == 0)).all())
        end_o = self.fr.index[rr.i.to_numpy()] + pd.Timedelta(minutes=1)
        tau = (end_exit - end_o) / pd.Timedelta(minutes=1)
        self.assertTrue(((tau >= 15) & (tau <= 1440)).all())

    def test_states_are_causal(self):
        cut = len(self.fr) * 2 // 3
        O2, _ = vd.origin_states(self.fr.iloc[:cut], "BTC", FITS, vd.Params())
        m = self.O.set_index("i").loc[O2["i"]]
        np.testing.assert_allclose(m[["v15", "v60", "v240"]].to_numpy(), O2[["v15", "v60", "v240"]].to_numpy(),
                                   equal_nan=True)
        np.testing.assert_array_equal(m["jump"].to_numpy(), O2["jump"].to_numpy())

    def test_warmup_and_side(self):
        self.assertTrue(np.isfinite(self.R.Z).all())
        self.assertTrue((self.R.D != 0).all())
        hit = self.R.hit.to_numpy()
        self.assertTrue(((hit == 0) | (hit == 1)).all())
        self.assertTrue(((self.R.p0 >= 0) & (self.R.p0 <= 1)).all())

    def test_aggregates_match_rows(self):
        a = self.A[(self.A.kind == "real") & (self.A.h == "240")]
        r = self.R[(self.R.kind == "real") & (self.R.h == "240")]
        self.assertEqual(int(a.n.sum()), len(r))
        self.assertAlmostEqual(float(a.sy.sum()), float(r.y.sum()), places=4)

    def test_martingale_hit_matches_reflection_benchmark(self):
        r = self.R[self.R.h == "60"]
        self.assertLess(abs(r.hit.mean() - r.p0.mean()), 0.05)


class DecisionTests(unittest.TestCase):
    def _run(self, kappa, seed):
        p = vd.Params(n_boot=300, n_boot_table=100)
        parts = []
        for k, a in enumerate(("BTC", "ETH")):
            df = vd.synthetic_klines(260, seed=seed + k, start="2025-10-01", kappa=kappa)
            parts.append(vd.run_asset(df, a, FITS, p)[0])
        A = pd.concat(parts, ignore_index=True)
        return A, p

    def test_null_gives_no_edge(self):
        A, p = self._run(0.0, 50)
        ev = vd.evaluate(A, vd.select(vd.dev_table(A, p), p), p)
        self.assertEqual(ev["verdict"], "NO_EDGE")
        t = vd.table(A, ["h"], p)
        self.assertTrue(((t.lo < 0) | (t.lo < 3)).all())

    def test_injected_pull_is_detected(self):
        A, p = self._run(0.01, 70)
        t = vd.table(A[A.anchor == "session"], ["h"], p)
        tt = t[t.test]
        self.assertTrue((tt.lo > 0).all())
        self.assertTrue((tt.diff_lo > 0).any())          # real session VWAP beats placebo phases
        s1 = vd.secondary_s1(A, p)
        self.assertEqual(len(s1), 8)

    def test_boot_diff_hour_zero_for_identical(self):
        A, p = self._run(0.0, 90)
        g = A[(A.kind == "real") & (A.h == "60")]
        d = vd.boot_diff_hour(g, g, 200, 1)
        self.assertAlmostEqual(d["diff"], 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
