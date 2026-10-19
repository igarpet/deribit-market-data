"""Pure synthetic regressions for LA-A03 (no local market files required)."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import sys
import types
import numpy as np
import pandas as pd

from src.data_engineering.modeling.liquidity_absorption_stage_a03 import (
    Config, EventSimulation, millis, round_stop_adverse, select_target, scan_event_tapes, MODES,
)

T0 = millis("2026-07-02T00:02:00Z")


def example_bars(side="long", mins=12):
    vals = []
    for i in range(-1, mins + 1):
        t = T0 + i * 60_000
        vals.append({"close_ms": t, "high": 100.7, "low": 99.9,
                     "prior_high": 120., "prior_low": 80.})
    return pd.DataFrame(vals).set_index("close_ms", drop=False)


def event(side="long", **kwargs):
    data = {"symbol": "BTCUSDT", "side": side, "signal_time_utc": "2026-07-02T00:02:00Z",
            "level_price": 100., "zone_halfwidth": .2, "a02_candidate_row": 0,
            "level_type": "pivot_5m", "vwap_utc_session": np.nan,
            "first_touch_trade_kline_quote_error": 0., "second_touch_trade_kline_quote_error": 0.,
            "zone_quote_touch_ratio": 1.5, "opposing_zone_share_change": .2, "zone_share_touch_change": .1}
    data.update(kwargs)
    return data


class TestA03(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(hold_minutes=12)

    def test_all_policies_begin_at_same_first_postclose_print(self):
        s = EventSimulation(event(), example_bars(), self.cfg)
        s.feed(T0-1, 100.)  # no pre-close entry
        self.assertIsNone(s.entry_ms)
        s.feed(T0, 100.4)
        self.assertEqual(s.entry_ms, T0)
        self.assertTrue(all(z.entered for z in s.states.values()))
        self.assertEqual(s.target_source, "OPPOSING_PRIOR_DAY")
        self.assertAlmostEqual(s.risk, .6, places=6)

    def test_break_even_stop_can_hit_while_fixed_still_open(self):
        s = EventSimulation(event(), example_bars(), self.cfg)
        s.feed(T0, 100.4)
        s.feed(T0+5_000, 101.1)
        self.assertFalse(s.states["FIXED"].be_enabled)
        self.assertTrue(s.states["BREAKEVEN"].be_enabled)
        self.assertAlmostEqual(s.states["BREAKEVEN"].stop,100.4)
        s.feed(T0+15_000, 100.4)
        self.assertTrue(s.states["BREAKEVEN"].closed)
        self.assertEqual(s.states["BREAKEVEN"].reason, "STOP")
        self.assertFalse(s.states["FIXED"].closed)

    def test_stop_market_never_fills_better_than_stop_on_gap(self):
        s = EventSimulation(event(), example_bars(), self.cfg)
        s.feed(T0, 100.4)
        s.feed(T0+4_000, 99.2)
        x = s.states["FIXED"]
        self.assertEqual(x.reason, "STOP")
        self.assertEqual(x.exit_print,99.2)
        self.assertLess(x.exit_fill, 99.2)

    def test_target_fill_is_capped_at_target_not_favorable_print(self):
        s = EventSimulation(event(), example_bars(), self.cfg)
        s.feed(T0,100.4)
        s.feed(T0+2_000,125.)
        x=s.states["FIXED"]
        self.assertEqual(x.reason, "TARGET")
        self.assertLess(x.exit_fill, 120.)
        self.assertEqual(x.exit_print,125.)
        self.assertEqual(x.exit_reference_px,120.)
        row=next(x for x in s.records() if x["mode"]=="FIXED")
        self.assertAlmostEqual(row["gross_r"],row["target_rr_initial"])

    def test_vwap_target_requires_displacement_from_entry(self):
        a = select_target("long",100.,1.,120.,80.,103.,self.cfg)
        b = select_target("long",100.,1.,120.,80.,101.,self.cfg)
        self.assertEqual(a,(103.,"SESSION_VWAP"))
        self.assertEqual(b,(120.,"OPPOSING_PRIOR_DAY"))
        self.assertEqual(select_target("short",100.,1.,120.,80.,97.,self.cfg), (97.,"SESSION_VWAP"))

    def test_no_structural_target_is_skipped_not_future_fitted(self):
        e=event(vwap_utc_session=np.nan)
        b=example_bars()
        b.loc[T0,"prior_high"]=99.  # opposing high already below entry
        s=EventSimulation(e,b,self.cfg)
        s.feed(T0,100.4)
        self.assertEqual(s.entry_skip_reason,"NO_STRUCTURAL_TARGET")
        self.assertTrue(s.all_closed())

    def test_long_trail_waits_for_confirmed_high_then_higher_low(self):
        b=example_bars()
        for j, high, low in ((1,101.8,100.5),(2,102.4,101.3),(3,102.,101.),(4,102.2,101.2)):
            b.loc[T0+j*60_000,["high","low"]]=[high,low]
        s=EventSimulation(event(), b, self.cfg)
        s.feed(T0,100.4)
        s.feed(T0+10_000,101.3)  # activate BE
        for j in (1,2,3):
            s.feed(T0+j*60_000,101.4)
            self.assertEqual(s.states["TRAIL_1M"].trail_updates,0)
        s.feed(T0+4*60_000,101.4)
        self.assertEqual(s.states["TRAIL_1M"].trail_updates,1)
        self.assertAlmostEqual(s.states["TRAIL_1M"].stop,100.8)
        s.feed(T0+4*60_000+1,100.75)
        self.assertEqual(s.states["TRAIL_1M"].reason,"STOP")
        self.assertFalse(s.states["BREAKEVEN"].closed)

    def test_short_trail_waits_for_confirmed_lower_low_then_lower_high(self):
        b=example_bars()
        b.loc[T0,["low","high"]]=[100.1,101.]
        for j, high, low in ((1,100.3,99.3),(2,99.1,98.7),(3,99.3,99.),(4,99.2,98.9)):
            b.loc[T0+j*60_000,["high","low"]]=[high,low]
        s=EventSimulation(event("short"), b,self.cfg)
        s.feed(T0,99.6)  # initial stop 100.2 (risk 0.6)
        s.feed(T0+10_000,98.9)  # BE
        for j in (1,2,3):
            s.feed(T0+j*60_000,99.3)
        self.assertEqual(s.states["TRAIL_1M"].trail_updates,0)
        s.feed(T0+4*60_000,99.2)
        self.assertEqual(s.states["TRAIL_1M"].trail_updates,1)
        self.assertAlmostEqual(s.states["TRAIL_1M"].stop,99.5)
        s.feed(T0+4*60_000+1,99.55)
        self.assertEqual(s.states["TRAIL_1M"].reason,"STOP")
        self.assertFalse(s.states["BREAKEVEN"].closed)

    def test_future_candles_cannot_change_pre_confirmation_stop(self):
        b=example_bars()
        x=EventSimulation(event(),b,self.cfg)
        x.feed(T0,100.4); x.feed(T0+10_000,101.3)
        x.feed(T0+MINUTE(),101.4)
        frozen=x.states["TRAIL_1M"].stop
        for j in (2,3,4):
            b.loc[T0+j*60_000,"low"]=999.
            b.loc[T0+j*60_000,"high"]=1000.
        y=EventSimulation(event(),b,self.cfg)
        y.feed(T0,100.4);y.feed(T0+10_000,101.3)
        y.feed(T0+MINUTE(),101.4)
        self.assertEqual(frozen,y.states["TRAIL_1M"].stop)

    def test_outward_tick_rounding_never_tightens_stop(self):
        self.assertAlmostEqual(round_stop_adverse(99.87,"long",0.1),99.8)
        self.assertAlmostEqual(round_stop_adverse(100.21,"short",0.1),100.3)

    def test_reconciliation_flags_are_reported_not_removed_by_outcome(self):
        s=EventSimulation(event(first_touch_trade_kline_quote_error=.16),example_bars(),self.cfg)
        s.feed(T0,100.4);s.feed(T0+1000,99.)
        result=s.records()
        self.assertEqual(len(result),3)
        self.assertTrue(all(x["flow_integrity_flag"]=="RECONCILIATION_OUTLIER" for x in result))

    def test_no_fabricated_close_when_tape_disappears(self):
        s=EventSimulation(event(),example_bars(),self.cfg)
        s.feed(T0,100.4)
        r=s.records()
        self.assertEqual(len(r),3)
        self.assertTrue(all(x["reason"]=="CENSORED_TAPE_END" for x in r))
        self.assertTrue(all(np.isnan(x["gross_r"]) for x in r))

    def test_entry_deadline_no_prompt_print(self):
        s=EventSimulation(event(),example_bars(),self.cfg)
        s.feed(T0+61_000,100.4)
        self.assertEqual(s.entry_skip_reason,"NO_PROMPT_ENTRY_PRINT")
        self.assertTrue(s.all_closed())

    def test_multiple_events_share_one_daily_file_stream(self):
        a=EventSimulation(event(a02_candidate_row=1),example_bars(),self.cfg)
        b=EventSimulation(event(a02_candidate_row=2),example_bars(),self.cfg)
        trade=pd.DataFrame({"time_ms": [T0, T0+10_000, T0+20_000],
                            "price": [100.4, 101.1, 99.1]})
        class Batch:
            def to_pandas(self): return trade.copy()
        class ParquetFile:
            def __init__(self, path): self.schema=types.SimpleNamespace(names=["time_ms","price"])
            def iter_batches(self, **kwargs): yield Batch()
        mod=types.ModuleType("pyarrow")
        mod.__path__=[]
        pq=types.ModuleType("pyarrow.parquet")
        pq.ParquetFile=ParquetFile
        mod.parquet=pq
        with tempfile.TemporaryDirectory() as t:
            f=Path(t)/"aggTrades"/"BTCUSDT"/"BTCUSDT-aggTrades-2026-07-02.parquet"
            f.parent.mkdir(parents=True)
            f.touch()
            with patch.dict(sys.modules,{"pyarrow":mod,"pyarrow.parquet":pq}):
                audit=scan_event_tapes(Path(t),[a,b])
        self.assertEqual(audit["read_daily_parquet_files"],1)
        self.assertEqual(a.states["FIXED"].reason,"STOP")
        self.assertEqual(b.states["FIXED"].reason,"STOP")

    def test_execution_costs_can_make_gross_profit_net_loss(self):
        s=EventSimulation(event(vwap_utc_session=101.7),example_bars(),self.cfg)
        s.feed(T0,100.4)
        s.feed(T0+5_000,101.7)
        row=next(x for x in s.records() if x["mode"]=="FIXED")
        self.assertEqual(row["reason"],"TARGET")
        self.assertGreater(row["gross_r"],row["net_r_assumed_costs"])


def MINUTE():
    return 60_000

if __name__ == "__main__":
    unittest.main()
