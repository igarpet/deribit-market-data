import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_engineering.modeling import liquidity_absorption_w01 as w  # noqa: E402


class TestW01(unittest.TestCase):
    def test_split_labels(self):
        s = pd.Series(["2025-06-30 23:59:00+00:00", "2025-07-01 00:00:00+00:00", "2026-08-01 00:00:00+00:00", "2026-10-02 00:00:00+00:00"])
        self.assertEqual(w.split_label(s).tolist(), ["MINING", "CONFIRM", "INSAMPLE", "CONFIRM"])

    def test_partial_spearman_removes_shared_driver(self):
        rng = np.random.default_rng(1)
        z = rng.normal(size=2000)
        x = z + 0.3 * rng.normal(size=2000)          # x only correlates with y through z
        y = z + 0.3 * rng.normal(size=2000)
        self.assertGreater(pd.Series(x).corr(pd.Series(y), method="spearman"), 0.8)
        self.assertLess(abs(w.partial_spearman(x, y, z)), 0.06)
        y2 = y + 2 * x
        self.assertGreater(w.partial_spearman(x, y2, z), 0.3)

    def test_profile_features_geometry(self):
        # volume concentrated at 100 (POC); thin between 100.3 and 100.8; entry 100, long target 101
        px = np.r_[np.full(500, 100.0), np.full(20, 100.5), np.full(300, 101.2)]
        usd = np.ones(len(px))
        f = w.profile_features(px, usd, 100.0, 101.0, 50.0, "long")
        self.assertLess(abs(f["pf_poc_R"]), 0.11)                     # POC within one 5 bp bin of entry (R = 50 bp)
        self.assertLess(f["pf_min_frac_path"], 0.2)                    # LVN on the way to the target
        self.assertEqual(f["pf_target_outside_range"], 0.0)
        g = w.profile_features(px, usd, 100.0, 103.0, 50.0, "long")
        self.assertEqual(g["pf_target_outside_range"], 1.0)

    def test_landmark_remaining_and_open_filter(self):
        rel = np.arange(-w.WINDOW, w.WINDOW + 1)
        F = pd.DataFrame({"a02_candidate_row": [1, 2], "entry_ms": [0, 0], "exit_ms": [30 * 60_000, 10 * 60_000],
                          "final_R": [1.0, -1.0], "net_R": [0.8, -1.2], "cost_R": [0.2, 0.2], "win": [1, 0],
                          "reason": ["TARGET", "STOP"], "split": "MINING", "day": pd.Timestamp("2025-01-01", tz="UTC"),
                          "symbol": "BTCUSDT"})
        P = pd.concat([pd.DataFrame({"a02_candidate_row": a, "rel": rel, "px_R": np.where(rel >= 0, 0.01 * rel, 0.0),
                                     "fav_R": 0.0, "adv_R": 0.0, "flow_n": 1.0, "vol_n": 1.0, "large_n": 0.0,
                                     "liq_with_n": 0.0, "liq_against_n": 0.0, "premium_bp": 0.0, "fut_basis": 0.0,
                                     "atm_iv": 0.0, "skew": 0.0, "opt_flow_n": 0.0, "opt_amt_n": 0.0}) for a in (1, 2)])
        X = w.landmark_features(F, P, 15)
        self.assertEqual(X.a02_candidate_row.tolist(), [1])              # trade 2 closed at minute 10
        self.assertAlmostEqual(float(X.L15_prog_R.iloc[0]), 0.14)        # close of minute 14
        self.assertAlmostEqual(float(X.L15_remaining_R.iloc[0]), 0.86)
        self.assertAlmostEqual(float(X.L15_cvd.iloc[0]), 1.0)

    def test_rule_eval_exits_only_triggered_open_trades(self):
        F = pd.DataFrame({"a02_candidate_row": range(60), "net_R": -0.2, "split": "MINING", "symbol": "BTCUSDT",
                          "day": pd.date_range("2025-01-01", periods=60, freq="D", tz="UTC")})
        X = pd.DataFrame({"a02_candidate_row": range(30), "L15_prog_R": 0.5, "cost_R": 0.1,
                          "L15_cvd": np.r_[np.zeros(10), np.ones(20)]})
        r = w.rule_eval(F, X, 15, "L15_cvd", 0.5, "t", w.Spec(boot_b=50)).iloc[0]
        self.assertAlmostEqual(r.triggered_share, 10 / 60)
        self.assertAlmostEqual(r.rule_net_R, (10 * 0.4 + 50 * -0.2) / 60)

    def test_synthetic_end_to_end_detects_planted_flow(self):
        with tempfile.TemporaryDirectory() as t:
            T = Path(t)
            w.synthetic_ext(T / "EXT", T / "hist", n=160)
            F, P = w.build(T / "EXT", T / "hist", T / "local", log=None)
            self.assertEqual(len(F), 160)
            r = w.run_analysis(F, P, T / "out", w.Spec(boot_b=100))
            self.assertGreater(r["H1_primary"]["confirm_partial_rho"], 0.2)
            for f in ("W01_feature_screen.csv", "W01_management_rules.csv", "W01_burst_summary.csv",
                      "W01_curves_exit.csv", "W01_decision.json"):
                self.assertTrue((T / "out" / f).exists())

    def test_minute_rows_open_filter_and_outcome(self):
        rel = np.arange(-w.WINDOW, w.WINDOW + 1)
        F = pd.DataFrame({"a02_candidate_row": [1], "entry_ms": [0], "exit_ms": [10 * 60_000], "exit_rel_min": [10.0],
                          "final_R": [1.0], "cost_R": [0.1], "R_bp": [50.0], "split": "MINING", "reason": "TARGET",
                          "day": pd.Timestamp("2025-01-01", tz="UTC"), "symbol": "BTCUSDT"})
        P = pd.DataFrame({"a02_candidate_row": 1, "rel": rel, "px_R": 0.1 * rel, "flow_n": rel.astype(float),
                          "vol_n": 1.0, "large_n": 0.0})
        D = w.minute_rows(F, P)
        self.assertEqual(D.rel.tolist(), list(range(10)))              # minutes 0..9 complete before exit at 10
        r3 = D[D.rel == 3].iloc[0]
        self.assertAlmostEqual(r3.rem_R, 1.0 - 0.3)
        self.assertAlmostEqual(r3.p1, 0.1)
        self.assertAlmostEqual(r3.fwd5_bp, 0.5 * 50)
        self.assertAlmostEqual(r3.addon_net_R, 0.6)

    def test_partial_multi_matches_single(self):
        rng = np.random.default_rng(3)
        x, y, z = rng.normal(size=(3, 500))
        self.assertAlmostEqual(w.partial_spearman_multi(x, y, [z]), w.partial_spearman(x, y, z), places=10)

    def test_minute_study_synthetic(self):
        with tempfile.TemporaryDirectory() as t:
            T = Path(t)
            w.synthetic_ext(T / "EXT", T / "hist", n=160)
            F, P = w.build(T / "EXT", T / "hist", T / "local", log=None)
            r = w.minute_study(F, P, T / "out", w.MinuteSpec(boot_b=50), log=None)
            self.assertGreater(r["M1"][1]["partial_rho_d1_rem"], 0.0)     # planted winner flow -> positive
            for f in ("W01M_partial_corr.csv", "W01M_events.csv", "W01M_deciles.csv", "W01M_decision.json"):
                self.assertTrue((T / "out" / f).exists())


if __name__ == "__main__":
    unittest.main()
