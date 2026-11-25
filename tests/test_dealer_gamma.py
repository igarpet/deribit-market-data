"""Dealer Gamma (DG 01-05) unit tests on synthetic data. Run: python -m unittest tests.test_dealer_gamma -v"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_engineering.modeling import dealer_gamma as dg  # noqa: E402


def _trades(rows, t0):
    tr = pd.DataFrame(rows)
    tr["ts"] = t0 + tr["ts"]
    return pd.concat([tr, dg.parse_names(tr.name)], axis=1)


class GammaAndBook(unittest.TestCase):
    def test_gamma_matches_finite_difference_of_delta(self):
        S, K, sig, T, h = 60000.0, 62000.0, 0.6, 3 / 365.25, 1.0
        fd = (dg.bs_call_delta(S + h, K, sig, T) - dg.bs_call_delta(S - h, K, sig, T)) / (2 * h)
        self.assertAlmostEqual(float(dg.bs_gamma(S, K, sig, T)) / float(fd), 1.0, places=5)

    def test_customer_buys_make_dealer_short_gamma(self):
        t0 = dg.ms("2024-03-01")
        tr = _trades({"ts": [1000, 2000], "name": ["BTC-8MAR24-60000-C"] * 2, "amt": [5.0, 2.0], "sgn": [1.0, -1.0],
                      "iv": [55.0, 60.0], "idx": [60000.0] * 2, "blk": [False, True]}, t0)
        b = dg.Book(dg.ms(dg.CLEAN_FROM))
        b.apply(b.index(tr), tr)
        self.assertEqual((b.a["net"][0], b.a["net_nb"][0], b.a["gross"][0]), (3.0, 5.0, 7.0))
        self.assertAlmostEqual(b.a["iv"][0], 0.60)                    # last print's iv
        live, g = b.gamma_usd(b.candidates(t0), t0 + 3000, 60000.0)
        r = b.gex(live, g, t0 + 3000)
        self.assertLess(r["gex_recon"], 0); self.assertLess(r["gex_short_all"], 0)
        self.assertGreater(r["gex_call_long_put_short"], 0)
        self.assertAlmostEqual(r["gex_recon_clean"], r["gex_recon"])  # first seen after CLEAN_FROM

    def test_expired_instrument_drops_out(self):
        t0 = dg.ms("2024-03-08")
        tr = _trades({"ts": [1000], "name": ["BTC-8MAR24-60000-C"], "amt": [1.0], "sgn": [1.0], "iv": [50.0],
                      "idx": [60000.0], "blk": [False]}, t0)
        b = dg.Book(0); b.apply(b.index(tr), tr)
        live, _ = b.gamma_usd(b.candidates(t0), t0 + 8 * 3_600_000, 60000.0)
        self.assertEqual(len(live), 0)

    def test_expiry_types(self):
        self.assertEqual(dg.expiry_type("2024-03-29"), "quarterly")
        self.assertEqual(dg.expiry_type("2024-04-26"), "monthly")
        self.assertEqual(dg.expiry_type("2024-04-19"), "weekly")
        self.assertEqual(dg.expiry_type("2024-04-18"), "daily")


class Synthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.days = dg.day_list("2024-06-25", "2024-07-20")
        cls.root = dg.synthetic_root(Path(cls.tmp.name), cls.days, seed=1)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_gate_passes_on_consistent_data_and_fails_when_prints_are_lost(self):
        H = self.root / "history"
        for c in dg.CURRENCIES:
            s = dg.gate_summary(dg.gate_compare(H, dg.load_collector(self.root, c), c))
            self.assertEqual(s["status"], "PASS"); self.assertAlmostEqual(s["volume_ratio_pooled"], 1.0)
        p = dg.trades_path(H, "BTC", "option", self.days[-2])
        full = pd.read_parquet(p)
        full.iloc[::2].to_parquet(p, index=False)                      # drop half the prints of one day
        try:
            s = dg.gate_summary(dg.gate_compare(H, dg.load_collector(self.root, "BTC"), "BTC"))
            self.assertEqual(s["status"], "FAIL"); self.assertLess(s["volume_ratio_pooled"], 0.95)
        finally:
            full.to_parquet(p, index=False)

    def test_build_and_all_tests_run(self):
        grid, pins, settle, cov = dg.build(self.root / "history", "BTC", self.days[0], self.days[-1])
        self.assertEqual(len(grid), dg.BARS * len(self.days)); self.assertTrue(cov.option_file.all())
        pan = dg.daily_panel(grid, dg.load_dvol(self.root, "BTC"))
        self.assertTrue({"log_rv", "ac1", "x_recon", "log_dvol"} <= set(pan.columns))
        r = dg.flow_fit(dg.flow_panel(grid, pan), "discovery")
        self.assertGreater(r["n"], 500); self.assertTrue(np.isfinite(r["t"]))
        ev = dg.pin_events(pins, settle)
        self.assertTrue(len(ev) > 0 and set(ev.rule) <= set(dg.PIN_HYP))
        row = ev.iloc[0]
        self.assertAlmostEqual(row.placebo, 2 * row.S_ref - row.strike)
        self.assertAlmostEqual(row.delta_bp,
                               1e4 * (abs(row.settle - row.strike) - abs(row.settle - row.placebo)) / row.S_ref)


class HedgeFlowRecovery(unittest.TestCase):
    def test_planted_hedging_is_recovered(self):
        """Flow = 0.3 x predicted hedge + noise -> b_hedge ~ 0.3 and significant."""
        rng = np.random.default_rng(0)
        n = 288 * 60
        ts = pd.date_range("2024-07-01 00:05", periods=n, freq="5min", tz="UTC")
        r = rng.normal(0, 0.002, n); G = rng.normal(0, 1, n) * 1e7; adv5 = np.full(n, 1e7)
        h = -100 * np.roll(G, 1) * r / adv5
        fp = pd.DataFrame({"ts": ts, "sess": ts.floor("D"), "period": "discovery", "r": r, "adv5": adv5,
                           "f_fwd3": 0.3 * h + rng.normal(0, 0.05, n), "h_recon": h, "gl_recon": np.roll(G, 1) / adv5})
        out = dg.flow_fit(fp, "discovery")
        self.assertAlmostEqual(out["b_hedge"], 0.3, delta=0.03); self.assertLess(out["p_one_sided"], 0.01)


if __name__ == "__main__":
    unittest.main()
