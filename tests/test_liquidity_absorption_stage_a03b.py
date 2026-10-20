"""A03b regression tests: mirror existing target, never retarget at a new R."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_engineering.modeling.liquidity_absorption_stage_a03 import (
    EventSimulation, Config, millis,
)
from src.data_engineering.modeling.liquidity_absorption_stage_a03b import (
    MirroredStopSimulation, check_against_original, equity_paths, plot_equity, build_coverage,
)

T0 = millis("2026-07-02T00:02:00Z")


def bars(prior_high=120., prior_low=80., n=10):
    return pd.DataFrame([{"close_ms": T0 + i * 60_000, "high": 101., "low": 99.,
                          "prior_high": prior_high, "prior_low": prior_low}
                         for i in range(-1, n+1)]).set_index("close_ms", drop=False)


def event(side="long", **kwargs):
    v = {"a02_candidate_row": 7, "symbol": "BTCUSDT", "side": side,
         "signal_time_utc": "2026-07-02T00:02:00Z", "level_price": 100.,
         "zone_halfwidth": 0.2, "level_type": "prior_day", "vwap_utc_session": np.nan,
         "first_touch_trade_kline_quote_error": 0., "second_touch_trade_kline_quote_error": 0.,
         "zone_quote_touch_ratio": 2., "opposing_zone_share_change": .3,
         "zone_share_touch_change": .3}
    v.update(kwargs)
    return v


class TestA03b(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(hold_minutes=10)

    def test_long_mirror_keeps_A03_target(self):
        standard = EventSimulation(event(), bars(), self.cfg)
        mirrored = MirroredStopSimulation(event(), bars(), self.cfg)
        standard.feed(T0, 100.4)
        mirrored.feed(T0, 100.4)
        self.assertEqual(standard.target, mirrored.target)
        self.assertEqual(standard.target_source, mirrored.target_source)
        self.assertEqual(standard.entry_ms, mirrored.entry_ms)
        self.assertEqual(mirrored.target, 120.)
        self.assertAlmostEqual(mirrored.initial_stop, 80.8)
        self.assertAlmostEqual(mirrored.risk, 19.6)
        self.assertAlmostEqual(mirrored.a03_original_stop, 99.8)
        self.assertAlmostEqual(mirrored.fixed_record()["target_rr_initial"], 1.0)

    def test_short_mirror_keeps_existing_target(self):
        s = MirroredStopSimulation(event("short"), bars(), self.cfg)
        s.feed(T0, 99.6)
        self.assertEqual(s.target, 80.)
        self.assertAlmostEqual(s.initial_stop, 119.2)
        self.assertAlmostEqual(s.risk, 19.6)
        self.assertAlmostEqual(s.fixed_record()["target_rr_initial"], 1.)

    def test_keep_same_vwap_target_selection_based_on_original_risk(self):
        standard = EventSimulation(event(vwap_utc_session=101.7), bars(), self.cfg)
        mirrored = MirroredStopSimulation(event(vwap_utc_session=101.7), bars(), self.cfg)
        standard.feed(T0, 100.4)
        mirrored.feed(T0, 100.4)
        self.assertEqual(standard.target_source, "SESSION_VWAP")
        self.assertEqual(mirrored.target, standard.target)
        self.assertAlmostEqual(mirrored.initial_stop, 99.1)

    def test_target_hit_has_exactly_plus_one_gross_r(self):
        s = MirroredStopSimulation(event(), bars(), self.cfg)
        s.feed(T0, 100.4)
        s.feed(T0 + 1000, 120.)
        r = s.fixed_record()
        self.assertEqual(r["reason"], "TARGET")
        self.assertAlmostEqual(r["gross_r"], 1.)
        self.assertGreater(r["net_r_assumed_costs"], 0.)
        self.assertTrue(r["symmetric_1to1_before_costs"])

    def test_exact_stop_hit_is_minus_one_gross_r(self):
        s = MirroredStopSimulation(event(), bars(), self.cfg)
        s.feed(T0, 100.4)
        s.feed(T0 + 1000, 80.8)
        r = s.fixed_record()
        self.assertEqual(r["reason"], "STOP")
        self.assertAlmostEqual(r["gross_r"], -1.)
        self.assertLess(r["net_r_assumed_costs"], -1.)

    def test_gap_through_stop_causes_more_than_one_r_loss(self):
        s = MirroredStopSimulation(event(), bars(), self.cfg)
        s.feed(T0, 100.4)
        s.feed(T0 + 1000, 70.)
        self.assertLess(s.fixed_record()["gross_r"], -1.)

    def test_plus_one_r_be_trigger_cannot_change_1to1_outcome(self):
        s = MirroredStopSimulation(event(), bars(), self.cfg)
        s.feed(T0, 100.4)
        s.feed(T0 + 1000, 119.9)
        self.assertFalse(s.states["BREAKEVEN"].be_enabled)
        s.feed(T0 + 2000, 120.0)
        r = s.fixed_record()
        self.assertEqual(r["reason"], "TARGET")
        self.assertEqual(r["trail_updates"], 0)

    def test_timeout_may_have_nonunit_gross_return(self):
        s = MirroredStopSimulation(event(), bars(), Config(hold_minutes=2))
        s.feed(T0, 100.4)
        s.feed(T0 + 2*60_000, 102.)
        r = s.fixed_record()
        self.assertEqual(r["reason"], "TIMEOUT")
        self.assertNotAlmostEqual(r["gross_r"], 1.)

    def test_missing_trade_data_is_censored_not_simulated(self):
        s = MirroredStopSimulation(event(), bars(), self.cfg)
        s.feed(T0, 100.4)
        r = s.fixed_record()
        self.assertEqual(r["reason"], "CENSORED_TAPE_END")
        self.assertTrue(np.isnan(r["net_r_assumed_costs"]))

    def test_no_target_retains_original_skip(self):
        standard = EventSimulation(event(vwap_utc_session=np.nan), bars(prior_high=99.), self.cfg)
        mirror = MirroredStopSimulation(event(vwap_utc_session=np.nan), bars(prior_high=99.), self.cfg)
        standard.feed(T0, 100.4)
        mirror.feed(T0, 100.4)
        self.assertEqual(standard.entry_skip_reason, "NO_STRUCTURAL_TARGET")
        self.assertEqual(standard.entry_skip_reason, mirror.entry_skip_reason)
        self.assertFalse(mirror.fixed_record()["symmetric_1to1_before_costs"])

    def test_comparison_checks_entries_targets_and_skips(self):
        old = EventSimulation(event(), bars(), self.cfg)
        mirror = MirroredStopSimulation(event(), bars(), self.cfg)
        old.feed(T0, 100.4); mirror.feed(T0, 100.4)
        old.feed(T0+1000,120.); mirror.feed(T0+1000,120.)
        orig = pd.DataFrame(old.records())
        updated = pd.DataFrame([mirror.fixed_record()])
        joined = check_against_original(orig, updated)
        self.assertEqual(len(joined), 1)
        self.assertTrue(joined.iloc[0].paired_closed)
        self.assertAlmostEqual(joined.iloc[0].target_old, joined.iloc[0].target_1to1)
        tampered = orig.copy(); tampered.loc[tampered.index[0], "target"] = 121.
        with self.assertRaises(AssertionError):
            check_against_original(tampered, updated)
        late = orig.copy(); late.loc[:, "entry_time_ms"] += 1
        with self.assertRaises(AssertionError):
            check_against_original(late, updated)

    def test_equity_paths_paired_only_and_exit_time_order(self):
        old = EventSimulation(event(), bars(), self.cfg)
        mirror = MirroredStopSimulation(event(), bars(), self.cfg)
        old.feed(T0, 100.4); mirror.feed(T0, 100.4)
        old.feed(T0+1000, 120.); mirror.feed(T0+1000, 120.)
        joined=check_against_original(pd.DataFrame(old.records()), pd.DataFrame([mirror.fixed_record()]))
        paths=equity_paths(joined)
        self.assertEqual(len(paths), 4)  # BTC and combined; two policies
        self.assertTrue((paths.ordinal_closed==1).all())
        self.assertTrue((paths.model=='A03_ORIGINAL_FIXED').any())
        self.assertTrue((paths.model=='A03B_MIRRORED_1TO1').any())
        with tempfile.TemporaryDirectory() as d:
            plot_equity(paths, Path(d))
            for filename in ("03b_equity_net.png", "03b_equity_gross.png", "03b_1to1_r_curve.png"):
                self.assertGreater((Path(d)/filename).stat().st_size, 1000)
        coverage=build_coverage(pd.DataFrame([mirror.fixed_record()]),joined)
        self.assertEqual(coverage[0]["paired_closed_original_a03_fixed"],1)

    def test_pre_entry_future_prints_do_not_change_target_or_stop(self):
        a = MirroredStopSimulation(event(), bars(), self.cfg)
        b = MirroredStopSimulation(event(), bars(), self.cfg)
        a.feed(T0, 100.4);b.feed(T0,100.4)
        a.feed(T0+1000, 90.)
        b.feed(T0+1000, 110.)
        self.assertEqual(a.target,b.target)
        self.assertEqual(a.initial_stop,b.initial_stop)


if __name__ == '__main__':
    unittest.main()
