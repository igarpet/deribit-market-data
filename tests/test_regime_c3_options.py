"""Synthetic tests for Regime Structure 02 (regime_c3_options).

Run from repo root: python -m unittest discover -s tests -p "test_regime_c3_options.py" -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_engineering.modeling import regime_c3_options as rc
from data_engineering.modeling import regime_structure as rs
from data_engineering.modeling import vrp_long_straddle as ls

P = rc.Params(n_boot=300)


class TestPartA(unittest.TestCase):
    def setUp(self):
        self.df = rs.synthetic_klines(n_days=160, seed=4, regime=True)
        self.sf, self.logc, self.t0 = rc.signal_frame(self.df, P)

    def test_signal_frame_alignment(self):
        sf = self.sf
        self.assertTrue(((sf["t_end"] - sf["t"]) == pd.Timedelta(minutes=15)).all())
        self.assertTrue(sf["signal"].sum() > 100)
        # log price at window end equals cumulative 1m returns
        m = rs.prepare_minutes(self.df)
        k = 50
        self.assertAlmostEqual(sf["logp"].iloc[k], m.r1[:(k + 1) * 15].sum(), places=10)

    def test_settlement_horizon_and_eligibility(self):
        rows = rc.part_a_rows(self.sf, self.logc, self.t0, P)
        ok = rows["zsettle"].notna()
        self.assertTrue(ok.any())
        settle = pd.to_datetime(rows.loc[ok, "session"], utc=True) + pd.Timedelta(hours=8)
        tte = (settle - rows.loc[ok, "t_end"]).dt.total_seconds() / 60
        self.assertTrue(((tte >= 120) & (tte <= 1200)).all())

    def test_matched_ratio_planted_and_null(self):
        rows = rc.part_a_rows(self.sf, self.logc, self.t0, P)
        rng = np.random.default_rng(0)
        r0 = rc.matched_ratio(rows, "z60", P, rng)
        self.assertTrue(np.isfinite(r0["ratio"]))
        planted = rows.copy()
        planted.loc[planted["signal"], "z60"] *= 1.5
        r1 = rc.matched_ratio(planted, "z60", P, rng)
        self.assertGreater(r1["ratio"], r0["ratio"] * 1.4)
        self.assertGreater(r1["lo"], 1.0)


class TestPartB(unittest.TestCase):
    def setUp(self):
        chain, settle = rc.synthetic_chain(n_sessions=25, seed=1)
        self.legs = rc.prepare_legs_grid(chain, 60)
        st = rc.entry_straddles(self.legs)
        empty = pd.DataFrame(columns=["index_name", "session", "delivery_price"])
        self.x0 = ls.straddle_pnl(ls.attach_settlement(st, empty, settle))

    def test_entries_respect_tte(self):
        self.assertTrue(((self.x0["minutes_to_expiry"] >= 120) & (self.x0["minutes_to_expiry"] <= 1200)).all())

    def test_tagging_first_snapshot_after_signal(self):
        t = self.x0["state_time"].sort_values().unique()
        sig = pd.DataFrame({"asset": "BTC", "t_end": [pd.Timestamp(t[3]) - pd.Timedelta(minutes=5)]})
        x = rc.tag_signals(self.x0, sig, rc.Params(entry_tolerance_min=10))
        self.assertEqual(set(x.loc[x["signal_entry"], "state_time"]), {pd.Timestamp(t[3])})
        far = pd.DataFrame({"asset": "BTC", "t_end": [pd.Timestamp(t[3]) - pd.Timedelta(minutes=30)]})
        self.assertFalse(rc.tag_signals(self.x0, far, rc.Params(entry_tolerance_min=10))["signal_entry"].any())

    def test_variants_and_stats(self):
        x = rc.add_variants(self.x0, self.legs, rc.Params(exit_after_min=60))
        self.assertTrue(x["R_call_net"].notna().all())
        self.assertTrue(x["R_exit60_net"].notna().any())
        # exit at the bid one hour later of a fairly priced straddle loses roughly the spread + fees
        self.assertLess(x["R_exit60_net"].median(), 0)
        rng = np.random.default_rng(0)
        x["signal_entry"] = rng.random(len(x)) < 0.2
        out = rc.part_b(x, P)
        self.assertIn(out["primary"]["verdict"], ("LONG_EDGE_CANDIDATE", "NO_LONG_EDGE", "INCONCLUSIVE"))
        d = out["table"]
        self.assertTrue({"signal", "all", "delta_signal_minus_all"} <= set(d["set"]))


if __name__ == "__main__":
    unittest.main()
