"""Synthetic tests for vol_ols_5m v0.1.1; no research DuckDB required.

Run from repo root: python -m unittest discover -s tests -p "test_vol_ols_5m.py" -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_engineering.modeling import vol_ols_5m as m


def toy():
    t = pd.date_range("2026-09-22 00:00", periods=6, freq="5min", tz="UTC")
    return pd.DataFrame({
        "underlying": ["BTC"] * 6,
        "state_time": t,
        "session_id": ["a", "a", "a", "b", "b", "b"],
        "expiry_date": ["2026-09-22"] * 3 + ["2026-09-23"] * 3,
        "high": [101., 102., 103., 104., 105., 106.],
        "low": [99., 98., 97., 96., 95., 94.],
        "perp_integrity_ok": [True] * 6,
        "utc_minute_of_day": t.hour * 60 + t.minute,
        "session_progress_fraction": np.linspace(0.1, 0.9, 6),
        "settlement_locked_fraction": np.zeros(6),
        "is_weekend": [False] * 6,
        "minutes_to_expiry": [30., 25., 20., 15., 10., 5.],
        "return_5m": [0.001, -0.002, 0.003, -0.004, 0.005, -0.006],
        "trade_count": [10, 0, 5, 7, 1, 3],
        "intrabar_range_bps": [20., 0., 60., 80., 100., 120.],
        "book_twa_spread_bps": [1., 1., 1., 1., 1., 1.],
    })


def synthetic_sessions(n_sessions=16, bars=48, seed=0):
    """n_sessions consecutive 4h 'sessions' with a volatility signal in x."""
    rng = np.random.default_rng(seed)
    frames = []
    start = pd.Timestamp("2026-09-22 08:00", tz="UTC")
    for s in range(n_sessions):
        t = start + pd.Timedelta(days=s) + pd.to_timedelta(np.arange(bars) * 5, "min")
        x = rng.normal(size=bars)
        sig = np.exp(-7 + 0.5 * x)              # bar-t vol driven by its x
        mid = 100.0
        frames.append(pd.DataFrame({
            "underlying": "BTC", "state_time": t, "session_id": f"s{s}",
            "expiry_date": (start + pd.Timedelta(days=s + 1)).strftime("%Y-%m-%d"),
            # high/low of bar t are generated from sig[t]
            "high": mid * np.exp(sig * m.PARKINSON_C / 2),
            "low": mid * np.exp(-sig * m.PARKINSON_C / 2),
            "perp_integrity_ok": True,
            "utc_minute_of_day": t.hour * 60 + t.minute,
            "session_progress_fraction": np.linspace(0, 1, bars),
            "settlement_locked_fraction": 0.0,
            "is_weekend": False,
            "minutes_to_expiry": np.arange(bars)[::-1] * 5.0 + 5,
            "return_5m": sig * rng.normal(size=bars),
            # predictor for NEXT bar's vol, known at t
            "trade_flow_imbalance": np.r_[x[1:], 0.0],
            "trade_count": 50, "intrabar_range_bps": 10.0,
            "book_twa_spread_bps": 1.0,
        }))
    return pd.concat(frames, ignore_index=True)


class VolOLS5MTests(unittest.TestCase):
    def test_target_next_bar_only_and_session_boundary(self):
        g = toy()
        meta, X, _ = m.make_panel(g)
        expected = np.log(102 / 98) / np.sqrt(4 * np.log(2))
        self.assertAlmostEqual(meta.loc[0, "sigma_next_5m"], expected)
        self.assertAlmostEqual(meta.loc[0, "index_sigma_next_5m"], 0.002 / np.sqrt(2 / np.pi))
        for col in ("sigma_next_5m", "index_sigma_next_5m"):
            self.assertTrue(pd.isna(meta.loc[2, col]))   # session change
            self.assertTrue(pd.isna(meta.loc[5, col]))   # end of data
        self.assertEqual(X.loc[0, "return_5m"], 0.001)
        self.assertNotIn("intrabar_range_bps", X.columns)
        g.loc[1, "high"] = 120
        meta2, X2, _ = m.make_panel(g)
        self.assertNotEqual(meta2.loc[0, "sigma_next_5m"], meta.loc[0, "sigma_next_5m"])
        self.assertEqual(X.loc[0, "return_5m"], X2.loc[0, "return_5m"])

    def test_gap_and_target_integrity(self):
        g = toy()
        g.loc[1, "state_time"] += pd.Timedelta(minutes=1)
        meta, _, _ = m.make_panel(g)
        self.assertTrue(pd.isna(meta.loc[0, "sigma_next_5m"]))
        self.assertTrue(pd.isna(meta.loc[0, "index_sigma_next_5m"]))
        g = toy()
        g.loc[1, "perp_integrity_ok"] = False
        meta, _, _ = m.make_panel(g)
        self.assertTrue(pd.isna(meta.loc[0, "sigma_next_5m"]))
        self.assertFalse(pd.isna(meta.loc[0, "index_sigma_next_5m"]))  # index unaffected

    def test_train_only_epsilon(self):
        train = np.array([0.0, 0.01, 0.02, 0.03] * 10)
        eps = m.fit_log_floor(train)
        self.assertAlmostEqual(eps, 0.0002)
        self.assertEqual(eps, m.fit_log_floor(np.r_[train, [100000.0]]))

    def test_oos_r2_uses_train_mean(self):
        truth = np.array([1.0, 2.0, 3.0])
        pred = truth.mean() * np.ones(3)          # perfect "split-mean" forecast
        s = m.score(truth, pred, train_mean=0.0, label="test")
        self.assertGreater(s["r2_vs_train_mean"], 0.8)
        s = m.score(truth, np.zeros(3), train_mean=0.0, label="test")
        self.assertAlmostEqual(s["r2_vs_train_mean"], 0.0)

    def test_ols_and_collinear_pruning(self):
        rng = np.random.default_rng(42)
        n = 300
        x1, x2 = rng.normal(size=n), rng.normal(size=n)
        X = pd.DataFrame({"a": x1, "b": x2, "a_clone": x1.copy()})
        y = 2. + 3 * x1 - 2 * x2
        train = np.arange(n) < 200
        names, drops = m.remove_aliases(X, train, list(X.columns))
        self.assertEqual((len(names), len(drops)), (2, 1))
        fit = m.fit_ols(X, y, train, names)
        r = m.metrics(fit, X, y, ~train, "test", float(y[train].mean()))
        self.assertLess(r["log_rmse"], 1e-10)

    def test_frozen_common_rows_in_backward_search(self):
        rng = np.random.default_rng(43)
        n = 400
        X = pd.DataFrame({"signal": rng.normal(size=n), "junk": rng.normal(size=n)})
        y = X["signal"].to_numpy() + rng.normal(size=n) * 0.01
        tr = np.arange(n) < 280
        kept, history = m.backward_validation(X, y, tr, ~tr, ["signal", "junk"], max_drops=1)
        self.assertIn("signal", kept)

    def test_session_split_partitions(self):
        g = synthetic_sessions(16)
        meta, _, _ = m.make_panel(g)
        valid = meta["sigma_next_5m"].notna().to_numpy()
        tr, se, te, sessions = m.session_split(meta, valid, "2026-12-31")
        self.assertEqual(len(sessions), 16)
        lab = m.session_labels(meta)
        sets = [set(lab[x]) for x in (tr, se, te)]
        self.assertEqual([len(s) for s in sets], [9, 3, 4])
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertLess(max(sets[0]), min(sets[1]))
        self.assertLess(max(sets[1]), min(sets[2]))
        # No target crosses a session, hence no target crosses a partition.
        last = meta.groupby("session_id").tail(1).index
        self.assertTrue(meta.loc[last, "sigma_next_5m"].isna().all())

    def test_development_end_excludes_later_sessions(self):
        g = synthetic_sessions(16)
        meta, _, _ = m.make_panel(g)
        valid = meta["sigma_next_5m"].notna().to_numpy()
        tr, se, te, sessions = m.session_split(meta, valid, "2026-10-05")
        self.assertTrue(all(s <= "2026-10-05" for s in sessions))
        lab = m.session_labels(meta)
        self.assertTrue((lab[tr | se | te] <= "2026-10-05").all())

    def test_prepare_asset_both_targets_same_rows_and_signal(self):
        g = synthetic_sessions(16)
        dp = m.prepare_asset(g, "parkinson", development_end="2026-12-31")
        di = m.prepare_asset(g, "index_abs", development_end="2026-12-31")
        for split in ("train", "selection", "test"):
            np.testing.assert_array_equal(dp[split], di[split])
        self.assertIn("tfi", dp["features"])
        tr = dp["train"] & dp["common"]
        te = dp["test"] & dp["common"]
        fit = m.fit_ols(dp["X"], dp["y"], tr, dp["features"])
        r = m.metrics(fit, dp["X"], dp["y"], te, "test", dp["train_mean"])
        self.assertGreater(r["r2_vs_train_mean"], 0.9)   # planted signal recovered
        cc = m.clock_coverage(dp)
        self.assertEqual(set(cc["split"]), {"train", "selection", "test"})
        self.assertTrue((cc["n_scored"] <= cc["n_target_valid"]).all())

    def test_target_audit(self):
        meta, _, _ = m.make_panel(toy())
        rows = meta["sigma_next_5m"].notna().to_numpy()
        a = m.target_audit(meta, rows)
        self.assertEqual(a["n"], 4)
        self.assertAlmostEqual(a["share_trades_le_1"], 0.5)   # next counts 0,5,1,3
        self.assertAlmostEqual(a["share_range_le_spread"], 0.25)  # next ranges 0,60,100,120


if __name__ == "__main__":
    unittest.main()
