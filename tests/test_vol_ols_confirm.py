"""Synthetic tests for the pre-registered confirmation (vol_ols_confirm).

Run from repo root: python -m unittest discover -s tests -p "test_vol_ols_confirm.py" -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from data_engineering.modeling import vol_ols_confirm as vc
from test_vol_ols_5m_blocks import synthetic


def with_extras(n_sessions=30, atm_signal=True, seed=0):
    """Synthetic sessions spanning the train/confirm boundary, with ATM and mid-path."""
    d = synthetic(n_sessions=n_sessions, bars=160, seed=seed)
    rng = np.random.default_rng(seed + 100)
    sig = np.log(d["high"] / d["low"]) / vc.base.PARKINSON_C
    nxt = sig.groupby(d["session_id"]).shift(-1).fillna(sig)
    d["option_iv_atm_forward"] = np.exp(np.log(nxt) * (1.0 if atm_signal else 0.0)
                                         + rng.normal(0, 1, len(d)))
    d["option_atm_quote_ok"] = True
    d["in_settlement_window"] = False
    for c in ("atm_iv_change_5m", "atm_iv_change_15m", "atm_iv_change_30m"):
        d[c] = rng.normal(size=len(d))
    d["mid_path_realized_variance"] = (sig * 0.9) ** 2
    d["mid_path_increment_count"] = 29
    d.loc[d.index % 50 == 0, "mid_path_increment_count"] = 10   # sparse bars
    return d


class ConfirmTests(unittest.TestCase):
    def test_constants_match_registration(self):
        r = vc.registration()
        self.assertEqual(r["train_last_expiry"], "2026-10-08")
        self.assertEqual(r["confirm_first_expiry"], "2026-10-09")
        self.assertEqual(r["primary_assets"], ["BTC", "ETH", "SOL"])
        self.assertEqual(r["readout_sessions"], 20)
        self.assertNotIn("log_book_updates", r["models"]["B0_ATM"])
        self.assertEqual(r["models"]["B0_ATM"][-4:], vc.ATM_FEATURES)

    def test_mid_sigma_gate_and_rescale(self):
        g = pd.DataFrame({"mid_path_realized_variance": [0.0004, 0.0004, 0.0004],
                          "mid_path_increment_count": [29, 26, 25]})
        s = vc.mid_sigma(g)
        self.assertAlmostEqual(s[0], 0.02)
        self.assertAlmostEqual(s[1], np.sqrt(0.0004 * 29 / 26))
        self.assertTrue(np.isnan(s[2]))

    def test_freeze_uses_training_expiries_only(self):
        d = with_extras()
        meta, X = vc.panel(d)
        spec = vc.freeze_asset(meta, X, "parkinson")
        self.assertTrue(all(s <= vc.TRAIN_LAST_EXPIRY for s in spec["train_sessions"]))
        # Changing confirmation-period data must not change the frozen fit.
        d2 = d.copy()
        late = vc.base.session_labels(meta) >= vc.CONFIRM_FIRST_EXPIRY
        d2.loc[late, "trade_flow_imbalance"] = 99.0
        d2.loc[late, "option_iv_atm_forward"] = 5.0
        spec2 = vc.freeze_asset(*vc.panel(d2), "parkinson")
        self.assertEqual(spec["models"]["B0_ATM"]["beta"], spec2["models"]["B0_ATM"]["beta"])

    def test_score_and_verdict_detects_atm_signal(self):
        d = with_extras(n_sessions=45, atm_signal=True)
        meta, X = vc.panel(d)
        spec = vc.freeze_asset(meta, X, "parkinson")
        s = vc.score_asset(meta, X, spec)
        self.assertTrue((s["session"] >= vc.CONFIRM_FIRST_EXPIRY).all())
        v = vc.asset_verdict(s)
        self.assertGreaterEqual(v["counted_sessions"], vc.READOUT_SESSIONS)
        self.assertEqual(v["status"], "pass")

    def test_verdict_fails_without_atm_signal_and_interim_is_nonbinding(self):
        d = with_extras(n_sessions=45, atm_signal=False, seed=3)
        meta, X = vc.panel(d)
        s = vc.score_asset(meta, X, vc.freeze_asset(meta, X, "parkinson"))
        self.assertEqual(vc.asset_verdict(s)["status"], "fail")
        short = s.iloc[:5].copy()
        short.attrs = s.attrs
        self.assertTrue(vc.asset_verdict(short)["status"].startswith("interim"))
        self.assertEqual(vc.overall_verdict({"BTC": {"status": "pass"}, "ETH": {"status": "pass"},
                                             "SOL": {"status": "fail"}}), "confirmed")
        self.assertEqual(vc.overall_verdict({"BTC": {"status": "pass"},
                                             "ETH": {"status": "interim (3/20)"}}),
                         "interim (non-binding)")

    def test_mid_rv_target_freezes(self):
        meta, X = vc.panel(with_extras())
        spec = vc.freeze_asset(meta, X, "mid_rv")
        self.assertTrue(spec["models"]["B0"]["fitted"])


if __name__ == "__main__":
    unittest.main()
