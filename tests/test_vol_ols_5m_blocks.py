"""Synthetic tests for vol_ols_5m_blocks (Volatility Regression 03).

Run from repo root: python -m unittest discover -s tests -p "test_vol_ols_5m_blocks.py" -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from data_engineering.modeling import vol_ols_5m_blocks as vb
from data_engineering.modeling.full_information import FEATURE_SPECS


def synthetic(n_sessions=14, bars=96, seed=0):
    """Sessions with a planted next-bar volatility signal in trade_flow_imbalance."""
    rng = np.random.default_rng(seed)
    frames = []
    start = pd.Timestamp("2026-09-22 08:00", tz="UTC")
    x_next = None
    for s in range(n_sessions):
        t = start + pd.Timedelta(days=s) + pd.to_timedelta(np.arange(bars) * 5, "min")
        x = rng.normal(size=bars)
        sig = np.exp(-7 + 0.6 * x + 0.1 * rng.normal(size=bars))
        frames.append(pd.DataFrame({
            "underlying": "BTC", "state_time": t, "session_id": f"s{s}",
            "expiry_date": (start + pd.Timedelta(days=s + 1)).strftime("%Y-%m-%d"),
            "high": 100 * np.exp(sig * vb.base.PARKINSON_C / 2),
            "low": 100 * np.exp(-sig * vb.base.PARKINSON_C / 2),
            "perp_integrity_ok": True,
            "utc_minute_of_day": t.hour * 60 + t.minute,
            "session_progress_fraction": np.linspace(0, 1, bars),
            "settlement_locked_fraction": 0.0, "is_weekend": False,
            "minutes_to_expiry": np.arange(bars)[::-1] * 5.0 + 5,
            "return_5m": sig * rng.normal(size=bars) + 1e-6,
            "trade_flow_imbalance": np.r_[x[1:], 0.0],      # known at t, drives t+1
            "volume_usd": np.exp(rng.normal(10, 1, bars)),  # junk
            "trade_count": 100, "intrabar_range_bps": 10 * sig / sig.mean(),
            "book_twa_spread_bps": 1.0,
        }))
    return pd.concat(frames, ignore_index=True)


class BlocksTests(unittest.TestCase):
    def test_blocks_exclude_shape_fit_wing_and_instability(self):
        allowed = {n for b in vb.BLOCK_ORDER for n in vb.BLOCKS[b]}
        for s in FEATURE_SPECS:
            if s.scope in ("shape", "fit", "option", "instability"):
                self.assertNotIn(s.name, allowed, s.name)
        self.assertNotIn("atm_iv_d60", allowed)
        self.assertNotIn("log_rv_4h", allowed)
        self.assertIn("log_rv_4h_tol", allowed)

    def test_tolerant_rv4h(self):
        t = pd.Series(pd.date_range("2026-10-01", periods=60, freq="5min", tz="UTC"))
        r = np.full(60, 0.001)
        meta = pd.DataFrame({"state_time": t})
        X = vb.add_tolerant_rv4h(pd.DataFrame({"return_5m": r}), meta)
        full = X["log_rv_4h_tol"].iloc[47]
        self.assertTrue(np.isnan(X["log_rv_4h_tol"].iloc[42]))   # < 44 returns yet
        r2 = r.copy(); r2[50] = np.nan                            # 1 missing: kept, same level
        X2 = vb.add_tolerant_rv4h(pd.DataFrame({"return_5m": r2}), meta)
        self.assertAlmostEqual(X2["log_rv_4h_tol"].iloc[55], full)
        r3 = r.copy(); r3[50:55] = np.nan                         # 5 missing: dropped
        X3 = vb.add_tolerant_rv4h(pd.DataFrame({"return_5m": r3}), meta)
        self.assertTrue(np.isnan(X3["log_rv_4h_tol"].iloc[56]))

    def test_session_plan_forward_chaining(self):
        meta, X = vb.panel(synthetic())
        plan = vb.session_plan(meta, vb.paired_valid(meta))
        self.assertEqual(len(plan["test"]), 4)
        for tr, va in plan["folds"]:
            self.assertLess(max(tr), va)
        self.assertLess(max(plan["dev"]), min(plan["test"]))

    def test_budget_gate(self):
        meta, X = vb.panel(synthetic(n_sessions=10, bars=12))
        valid = vb.paired_valid(meta)
        y, _, _ = vb.target_y(meta, "parkinson", valid)
        r = vb.fit_score(X, y, valid, valid, vb.nested_features("B1_flow_basis"))
        self.assertFalse(r["ok"])
        self.assertIn("budget", r["reason"])

    def test_evaluate_blocks_recovers_flow_signal_and_backward_keeps_it(self):
        meta, X = vb.panel(synthetic())
        valid = vb.paired_valid(meta)
        plan = vb.session_plan(meta, valid)
        dev = np.isin(plan["labels"], plan["dev"]) & valid
        avail, adm = vb.admit_features(X, dev)
        self.assertNotIn("ret_15m", avail)                      # column absent
        self.assertTrue(adm.set_index("feature").loc["tfi", "admitted"])
        res, final = vb.evaluate_blocks(meta, X, valid, plan, "parkinson", avail)
        test = res[(res.stage == "test") & (res.population == "own") & res.fitted]
        r = test.set_index("model")["r2_vs_train_mean"]
        self.assertGreater(r["B1_flow_basis"], r["B0_returns_vol_clock"] + 0.3)
        # No L2/option data: B2/B3 reduce to B1's features, so they fit identically.
        self.assertAlmostEqual(r["B2_l2_book"], r["B1_flow_basis"])
        names = final[("own", "B1_flow_basis")]["names"]
        keep, hist = vb.backward_by_folds(meta, X, valid, plan, "parkinson", names, max_drops=5)
        self.assertIn("tfi", keep)

    def test_assign_vintage_and_unique_loss(self):
        t = pd.Series(pd.date_range("2026-10-04 16:10", periods=4, freq="5min", tz="UTC"))
        b = pd.DataFrame({"underlying": ["BTC", "BTC"],
                          "vintage": ["eventtime_fix_unstamped", "futures_5m_l2_v0.3.0"],
                          "first": ["2026-10-03 13:00+00:00", "2026-10-04 16:20+00:00"]})
        v = vb.assign_vintage(t, b, "BTC")
        self.assertEqual(list(v), ["eventtime_fix_unstamped"] * 2 + ["futures_5m_l2_v0.3.0"] * 2)
        X = pd.DataFrame({"a": [1, np.nan, 1, 1], "b": [1, 1, 1, np.nan]})
        u = vb.unique_loss(X, np.ones(4, bool), ["a", "b"]).set_index("feature")
        self.assertEqual(int(u.loc["a", "unique_loss_rows"]), 1)


if __name__ == "__main__":
    unittest.main()
