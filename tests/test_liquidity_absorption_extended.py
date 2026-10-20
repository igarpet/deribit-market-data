import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_engineering.modeling import liquidity_absorption_extended as ex  # noqa: E402


class TestExtended(unittest.TestCase):
    def test_day_inputs_schema_and_fill(self):
        t0 = int(pd.Timestamp("2025-01-02", tz="UTC").value // 10**6)
        pr = pd.DataFrame({"time_ms": [t0 + 30_000, t0 + 90_000, t0 + 95_000, t0 + 5 * 60_000 + 1],
                           "seq": [1, 2, 3, 4], "price": [100.0, 101.0, 99.0, 100.5], "usd": [1000.0, 2020.0, 990.0, 100.5],
                           "buy": [True, False, True, False], "index": [99.9, 100.9, 98.9, 100.4]})
        agg, bars, ix = ex.day_inputs(pr, "2025-01-02", prev_close=98.0)
        np.testing.assert_allclose(agg.qty, [10.0, 20.0, 10.0, 1.0])
        np.testing.assert_array_equal(agg.is_buyer_maker, [False, True, False, True])
        self.assertEqual(len(bars), 1440)
        b1 = bars.iloc[1]
        self.assertEqual((b1.open, b1.high, b1.low, b1.close), (101.0, 101.0, 99.0, 99.0))
        self.assertAlmostEqual(b1.taker_buy_quote_volume, 990.0)
        self.assertEqual(bars.iloc[3].close, 99.0)                  # empty minute carries the previous close
        self.assertEqual(bars.iloc[3].quote_volume, 0.0)
        self.assertEqual(ix.state_time.iloc[0], pd.Timestamp("2025-01-02 00:05", tz="UTC"))
        self.assertAlmostEqual(ix.close.iloc[0], 98.9)              # last index price in (00:00, 00:05]
        self.assertAlmostEqual(ix.close.iloc[1], 100.4)

    def test_period_labels_and_bootstrap(self):
        s = pd.Series(["2024-05-01 00:00+00:00", "2025-06-01 00:00+00:00", "2026-08-01 00:00+00:00", "2026-10-05 00:00+00:00"])
        self.assertEqual(list(ex.period_label(s)), ["H1", "H2", "INSAMPLE_JUL_SEP_2026", "H2"])
        x = pd.Series(np.r_[np.full(50, 5.0), np.full(50, 7.0)])
        d = pd.Series(np.repeat(np.arange(20), 5))
        m, lo = ex.cluster_lower(x, d)
        self.assertAlmostEqual(m, 6.0)
        self.assertGreater(lo, 5.0)

    def test_deribit_repricing(self):
        df = pd.DataFrame({"entry_fill_assumed": [100.0], "exit_fill_assumed": [101.0], "entry_print": [100.0],
                           "entry_fee_bp": [6.0], "exit_fee_bp": [6.0], "net_return_bp_assumed_costs": [50.0]})
        self.assertAlmostEqual(ex.deribit_net(df).iloc[0], 50.0 + (6 * 100 + 6 * 101) / 100 - (5 * 100 + 5 * 101) / 100)

    def test_dedupe_same_signal_two_streams(self):
        with tempfile.TemporaryDirectory() as d:
            b = Path(d) / "02"
            b.mkdir()
            rows = pd.DataFrame({"symbol": ["BTCUSDT"] * 3, "side": ["long"] * 3,
                                 "signal_time_utc": ["2025-01-01 10:00:00+00:00"] * 2 + ["2025-01-01 11:00:00+00:00"],
                                 "level_price": [100.0, 100.0, 100.0], "level_type": ["prior_day", "pivot_5m", "pivot_5m"]})
            rows.to_csv(b / "02_price_only_candidates.csv", index=False)
            rows.assign(vwap_utc_session=[1.0, 2.0, 3.0]).to_csv(b / "02_causal_flow_features.csv", index=False)
            self.assertEqual(ex.dedupe_a02(Path(d)), 1)
            p = pd.read_csv(b / "02_price_only_candidates.csv")
            e = pd.read_csv(b / "02_causal_flow_features.csv")
            self.assertEqual(list(p.level_type), ["prior_day", "pivot_5m"])
            self.assertEqual(list(e.vwap_utc_session), [1.0, 3.0])          # alignment kept
            self.assertEqual(ex.dedupe_a02(Path(d)), 0)                     # idempotent

    def test_full_chain_synthetic(self):
        repo = ROOT
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            hist, root, shadow = d / "history", d / "root", d / "shadow"
            days = [str(x.date()) for x in pd.date_range("2026-06-01", periods=12, freq="D")]
            ex.synthetic_history(hist, days, seed=3)
            cov = ex.prepare_venue_root(hist, root, ex.Spec(start="2026-06-01"), log=None)
            self.assertEqual(len(cov), 24)
            ex.stage_a02(root, shadow, ex.Spec(start="2026-06-04", end="2026-06-12"))
            ex.stage_a02b(root, shadow)
            a3 = ex.stage_a03(root, shadow)
            b = ex.stage_a03b(root, shadow)
            self.assertTrue(b.loc[b.reason.isin(ex.CLOSED), "symmetric_1to1_before_costs"].all())
            ex.stage_a03c(root, shadow)
            f = ex.stage_a03d(root, shadow, repo)
            self.assertGreater(len(f), 0)
            z = ex.stage_a04(root, shadow)
            self.assertEqual(len(z), len(b))
            T, dec = ex.evaluate(shadow)
            self.assertIn(dec["verdict"], ("PASS", "FAIL_CLOSE_LA"))
            self.assertTrue((T.variant == "A03b_forward_1to1_taker").any())
            # resumable: a second prepare does not rewrite day files
            m0 = (root / "aggTrades" / "BTCUSDT" / f"BTCUSDT-aggTrades-{days[0]}.parquet").stat().st_mtime
            ex.prepare_venue_root(hist, root, ex.Spec(start="2026-06-01"), log=None)
            self.assertEqual(m0, (root / "aggTrades" / "BTCUSDT" / f"BTCUSDT-aggTrades-{days[0]}.parquet").stat().st_mtime)


if __name__ == "__main__":
    unittest.main()
