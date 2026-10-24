"""Synthetic tests for Turtle Breakout 04 (turtle_trade_features).

Run from repo root: python -m unittest discover -s tests -p "test_turtle_trade_features.py" -v
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
from data_engineering.modeling import turtle_trade_features as tf
from data_engineering.modeling import vwap_drift as vd

warnings.filterwarnings("ignore", category=RuntimeWarning)
FITS = vd.load_engine(ROOT / "notebooks" / "Volatility_Dynamics" / "18" / "18_manifest.json")
P = tb.Params()
ENTRY = [c for g in tf.ENTRY_FEATURES.values() for c in g if c not in ("btc_ret20_log", "btc_in_breakout", "breadth")]


def asset(name="BNB", seed=11, days=900):
    b = tb.synthetic_5m(days, seed=seed, trend=0.01, start="2020-01-01")
    x = tf.synthetic_flow(b, seed)
    f = tb.synthetic_funding(b, 0.0001)
    f["rate"] = f["rate"] * (1 + 0.5 * np.sin(np.arange(len(f)) / 30))
    rng = np.random.default_rng(seed)
    met = pd.DataFrame({"sum_open_interest": np.exp(np.cumsum(rng.normal(0, .01, len(b) // 12))) * 1e6},
                       index=b.index[::12][:len(b) // 12])
    met["sum_open_interest_value"] = met.sum_open_interest * 100
    met["sum_toptrader_long_short_ratio"] = 1.2 + rng.normal(0, .05, len(met))
    met["count_long_short_ratio"] = 1.1 + rng.normal(0, .05, len(met))
    met["sum_taker_long_short_vol_ratio"] = 1.0 + rng.normal(0, .05, len(met))
    spot = b.close * (1 - 0.0002 * (1 + rng.normal(0, 0.5, len(b))))
    d = tb.prepare_asset(name, b, f, P)
    T = tf.run_trades(d, P, b.index[0] + pd.Timedelta(days=30))
    return b, x, f, met, spot, d, T


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b, cls.x, cls.f, cls.met, cls.spot, cls.d, cls.T = asset()
        cls.F = tf.entry_and_path_features("BNB", cls.d, cls.x, cls.T, cls.f, cls.spot, cls.met, FITS)

    def test_r_multiple(self):
        T = self.T
        np.testing.assert_allclose(T.R * 2 * T.atr_entry / T.entry_px, T.ret_notional)
        self.assertTrue(((T.win == 1) == (T.ret_notional > 0)).all())

    def test_entry_features_do_not_see_the_entry_bar_or_later(self):
        i = len(self.T) // 2
        je = int(self.T.je.iloc[i])
        cut = self.d.end[je - 1]
        x2 = self.x.copy(); x2.loc[x2.index > cut] = x2.loc[x2.index > cut] * 7.0
        sp2 = self.spot.copy(); sp2[sp2.index > cut] *= 1.05
        m2 = self.met.copy(); m2.loc[m2.index > cut] *= 3.0
        f2 = self.f.copy(); f2.loc[f2.t > cut, "rate"] *= -5
        F2 = tf.entry_and_path_features("BNB", self.d, x2, self.T, f2, sp2, m2, FITS)
        a = self.F.iloc[i][ENTRY].astype(float).to_numpy(); b = F2.iloc[i][ENTRY].astype(float).to_numpy()
        np.testing.assert_allclose(a, b, equal_nan=True, err_msg="entry features changed with post-entry data")

    def test_entry_features_mostly_present(self):
        cov = self.F[ENTRY].notna().mean()
        self.assertGreater(cov.drop(["fund_z30", "basis_z7d", "dist_ma200_atr", "above_ma200"], errors="ignore").min(), 0.8)

    def test_path_features(self):
        self.assertTrue((self.F.mfe_R >= -1e-9).all() and (self.F.mae_R >= -1e-9).all())
        w = self.F[self.F.win == 1]; l = self.F[self.F.win == 0]
        self.assertGreater(w.mfe_R.median(), l.mfe_R.median())

    def test_market_features(self):
        S = pd.concat([tf.daily_states("BTC", self.d), tf.daily_states("BNB", self.d)])
        G = tf.add_market_features(self.F, S)
        self.assertTrue(G.breadth.dropna().between(0, 1).all())
        self.assertTrue(G.btc_in_breakout.dropna().isin([0.0, 1.0]).all())


class StatTests(unittest.TestCase):
    def test_auc_and_bh(self):
        self.assertAlmostEqual(tf.auc(np.array([1., 2, 3, 4]), np.array([0, 0, 1, 1])), 1.0)
        self.assertAlmostEqual(tf.auc(np.array([1., 1, 1, 1]), np.array([0, 1, 0, 1])), 0.5)
        q = tf.bh(np.array([0.01, 0.02, 0.03, 0.5]))
        np.testing.assert_allclose(q, [0.04, 0.04, 0.04, 0.5])

    def _frame(self, n, effect, seed):
        rng = np.random.default_rng(seed)
        t = pd.date_range("2021-01-01", periods=n, freq="5D", tz="UTC")
        sig = rng.normal(0, 1, n); noise = rng.normal(0, 1, (n, 5))
        p = 1 / (1 + np.exp(-(-0.4 + effect * sig)))
        win = (rng.random(n) < p).astype(int)
        F = pd.DataFrame({"entry_time": t, "win": win, "R": np.where(win, 2.0, -0.6) + rng.normal(0, .1, n),
                          "sig": sig, **{f"n{i}": noise[:, i] for i in range(5)}})
        F["big"] = (F.R > 1.5).astype(int)
        return F

    def test_univariate_and_confirmation(self):
        fp = tf.FParams(n_boot=400, min_feature_n=50)
        Fd = self._frame(800, 1.0, 1); Fc = self._frame(200, 1.0, 2)
        cols = ["sig"] + [f"n{i}" for i in range(5)]
        U = tf.univariate(Fd, cols, "win", fp)
        self.assertLess(U.set_index("feature").q_bh["sig"], fp.fdr_q)
        C = tf.confirm(U, Fc, "win", fp)
        self.assertTrue(C.set_index("feature").confirmed["sig"])

    def test_model_signal_and_noise(self):
        fp = tf.FParams(n_boot=300, n_perm=60, min_feature_n=50)
        cols = ["sig"] + [f"n{i}" for i in range(5)]
        good = tf.model_test(self._frame(800, 1.0, 3), self._frame(200, 1.0, 4), cols, "win", fp)
        self.assertTrue(good["passed"])
        bad = tf.model_test(self._frame(800, 0.0, 5), self._frame(200, 0.0, 6), cols, "win", fp)
        self.assertFalse(bad["passed"])
        self.assertEqual(len(tf.tercile_economics(self._frame(200, 1.0, 4), good["scores_confirm"])), 3)


if __name__ == "__main__":
    unittest.main()
