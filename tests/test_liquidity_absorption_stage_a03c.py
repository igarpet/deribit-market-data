"""Synthetic regression tests: A03c reversed tape replay & unverified passive fills."""
import sys
import tempfile
import unittest
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_engineering.modeling.liquidity_absorption_stage_a03c import (
    Config, TapeStudy, MODES, CLOSED, scan_tapes, equity, _summaries,
    _fee_sensitivity, utc_ms,
)

T = utc_ms("2026-07-02T00:10:00+00:00")


def event(side="long", target=101, e=100.15, ident=0, level=100, close=None):
    if side == "short":
        target = 99 if target == 101 else target
        e = 99.85 if e == 100.15 else e
    return {"a02_candidate_row": ident, "symbol": "BTCUSDT", "side": side,
            "signal_time_utc": "2026-07-02 00:10:00+00:00", "level_price": level,
            "bar_close": (100.15 if side=="long" else 99.85) if close is None else close,
            "target": target, "initial_stop": 2*e-target,
            "entry_print": e, "entry_time_ms": T+100,
            "flow_integrity_flag": "OK", "absorption_like_exploratory":False}


class TestA03c(unittest.TestCase):
    def test_mode_count(self):
        self.assertEqual(len(MODES),3)

    def test_reverse_long_to_short_swaps_barriers_and_same_entry(self):
        a=TapeStudy(event(), "REVERSE_TAKER_1TO1", Config())
        self.assertEqual(a.side,"short")
        self.assertAlmostEqual(a.target,99.3)
        self.assertAlmostEqual(a.stop,101)
        a.feed(T+100,100.15,False)
        self.assertEqual(a.entry_px,100.15)
        self.assertEqual(a.entry_time,T+100)
        a.feed(T+200,99.3,True)
        r=a.record()
        self.assertEqual(r["reason"],"TARGET")
        self.assertAlmostEqual(r["gross_r"],1, places=7)
        self.assertLess(r["net_r_assumed_costs"],1)

    def test_reverse_short_to_long_swaps_barriers(self):
        a=TapeStudy(event(side="short"), "REVERSE_TAKER_1TO1", Config())
        self.assertEqual(a.side,"long")
        a.feed(T+100,99.85,True)
        a.feed(T+200,100.7,False)
        self.assertEqual(a.reason,"TARGET")
        self.assertAlmostEqual(a.record()["gross_r"],1,places=7)

    def test_original_source_entry_cannot_drift(self):
        a=TapeStudy(event(), "REVERSE_TAKER_1TO1", Config())
        with self.assertRaises(AssertionError):
            a.feed(T+200,100.15,False)

    def test_long_maker_needs_seller_aggressor_at_limit_and_latency(self):
        a=TapeStudy(event(), "MAKER_TOUCH_3RD", Config())
        a.feed(T+500,99.9,True)
        self.assertIsNone(a.entry_time)
        a.feed(T+2000,99.9,False)  # price crosses but BUY aggressor; not a resting bid fill
        self.assertIsNone(a.entry_time)
        a.feed(T+3000,100.0,True)
        self.assertEqual(a.entry_time,T+3000)
        self.assertEqual(a.entry_px,100)
        self.assertEqual(a.stop,99)
        a.feed(T+5000,101.1,False)
        self.assertEqual(a.reason,"TARGET")
        self.assertAlmostEqual(a.record()["gross_r"],1)
        self.assertEqual(a.record()["entry_fee_bp"],2)
        self.assertEqual(a.record()["exit_fee_bp"],6)

    def test_through_vs_touch_requires_minimum_tick(self):
        a=TapeStudy(event(), "MAKER_THROUGH_3RD", Config())
        a.feed(T+1200,100.0,True)
        self.assertIsNone(a.entry_time)
        a.feed(T+1300,99.95,True)
        self.assertIsNone(a.entry_time)
        a.feed(T+1600,99.9,True)
        self.assertIsNotNone(a.entry_time)
        self.assertTrue(a.side_eligible_print)

    def test_short_maker_needs_buyer_aggressor(self):
        a=TapeStudy(event(side="short"), "MAKER_TOUCH_3RD", Config())
        a.feed(T+2000,100.0,True)
        self.assertIsNone(a.entry_time)
        a.feed(T+3000,100.0,False)
        self.assertEqual(a.entry_px,100)
        a.feed(T+3100,101.0,True)
        self.assertEqual(a.reason,"STOP")
        self.assertAlmostEqual(a.record()["gross_r"],-1)

    def test_limit_nonmarketable_proxy_fail(self):
        a=TapeStudy(event(close=99.99),"MAKER_TOUCH_3RD",Config())
        self.assertEqual(a.reason,"NOT_POST_ONLY_AT_SIGNAL_CLOSE_PROXY")

    def test_no_order_fills_after_expiry(self):
        a=TapeStudy(event(),"MAKER_TOUCH_3RD",Config(limit_order_ttl_min=2))
        a.feed(T+120_000,100,True)
        self.assertEqual(a.reason,"NO_THIRD_TOUCH")
        self.assertIsNone(a.entry_time)
        self.assertTrue(np.isnan(a.record()["net_return_bp_assumed_costs"]))

    def test_touch_but_wrong_aggressor(self):
        a=TapeStudy(event(),"MAKER_TOUCH_3RD",Config(limit_order_ttl_min=2))
        a.feed(T+3000,100,False)
        a.feed(T+120_000,101,True)
        self.assertEqual(a.reason,"NO_OPPOSITE_AGGRESSOR_PRINT")

    def test_through_not_reached_reason(self):
        a=TapeStudy(event(),"MAKER_THROUGH_3RD",Config(limit_order_ttl_min=2))
        a.feed(T+3000,100,True)
        a.feed(T+120_000,101,True)
        self.assertEqual(a.reason,"NO_TRADE_THROUGH")

    def test_timeout_closes_at_first_print_after_deadline(self):
        a=TapeStudy(event(),"MAKER_TOUCH_3RD",Config(hold_minutes=1))
        a.feed(T+2000,100,True)
        a.feed(T+62_000,100.2,True)
        self.assertEqual(a.reason,"TIMEOUT")
        self.assertGreater(a.record()["net_return_bp_assumed_costs"],0)

    def test_gap_beyond_stop_is_worse_than_minus_one_r(self):
        a=TapeStudy(event(),"MAKER_TOUCH_3RD",Config())
        a.feed(T+2000,100,True)
        a.feed(T+3000,98,True)
        self.assertEqual(a.reason,"STOP")
        self.assertLess(a.record()["gross_r"],-1)

    def test_unresolved_censor_is_not_zero_trade(self):
        a=TapeStudy(event(),"MAKER_TOUCH_3RD",Config())
        a.feed(T+2000,100,True)
        a.finish({})
        self.assertEqual(a.reason,"CENSORED_TAPE_END")
        self.assertTrue(np.isnan(a.record()["net_r_assumed_costs"]))
        unfilled=TapeStudy(event(ident=1),"MAKER_TOUCH_3RD",Config())
        unfilled.feed(T+2000,101,True)
        unfilled.finish({})
        self.assertEqual(unfilled.reason,"CENSORED_ORDER_WINDOW")

    def test_missing_day_between_entry_and_exit_invalidates_later_target(self):
        a=TapeStudy(event(),"MAKER_TOUCH_3RD",Config())
        a.feed(T+2000,100,True)
        a.feed(T+5000,101,False)
        self.assertEqual(a.reason,"TARGET")
        a.missing_days=["2026-07-02"]
        a.finish({})
        self.assertEqual(a.reason,"CENSORED_TAPE_GAP")
        self.assertTrue(np.isnan(a.record()["net_r_assumed_costs"]))

    def test_missing_day_after_completed_exit_does_not_change_result(self):
        a=TapeStudy(event(),"MAKER_TOUCH_3RD",Config())
        a.feed(T+2000,100,True)
        a.feed(T+5000,101,False)
        a.missing_days=["2026-07-03"]
        a.finish({})
        self.assertEqual(a.reason,"TARGET")

    def test_shows_zero_for_no_trade_but_not_censored_in_partial_ev(self):
        a=TapeStudy(event(),"MAKER_TOUCH_3RD",Config())
        a.feed(T+2000,100,True)
        a.feed(T+3000,101,False)
        b=TapeStudy(event(ident=1),"MAKER_TOUCH_3RD",Config(limit_order_ttl_min=1))
        b.feed(T+MINUTE,101,True)
        frame=pd.DataFrame([a.record(),b.record()]); s=_summaries(frame,Config())
        self.assertEqual(s[0]["signals"],2)
        self.assertEqual(s[0]["closed"],1)
        self.assertAlmostEqual(s[0]["mean_net_bp_per_signal_zero_if_no_trade_PARTIAL"],frame.loc[0,"net_return_bp_assumed_costs"]/2)

    def test_independent_shared_daily_tape_scan_and_aggressor(self):
        events=[TapeStudy(event(ident=0),"MAKER_TOUCH_3RD",Config(limit_order_ttl_min=1)),
                TapeStudy(event(ident=1),"MAKER_THROUGH_3RD",Config(limit_order_ttl_min=1))]
        trade=pd.DataFrame({"time_ms":[T+1200,T+1300,T+3000,T+7000,T+62000],
                            "price":[100.0,99.9,100.,101.,101.],
                            "is_buyer_maker":[True,True,True,False,False]})
        class Batch:
            def to_pandas(self):return trade
        class File:
            schema=types.SimpleNamespace(names=["time_ms","price","is_buyer_maker"])
            def __init__(self,path):pass
            def iter_batches(self,**kwargs):yield Batch()
        arrow=types.ModuleType("pyarrow");arrow.__path__=[]
        pq=types.ModuleType("pyarrow.parquet");pq.ParquetFile=File
        arrow.parquet=pq
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"aggTrades"/"BTCUSDT"/"BTCUSDT-aggTrades-2026-07-02.parquet";p.parent.mkdir(parents=True);p.touch()
            with patch.dict(sys.modules,{"pyarrow":arrow,"pyarrow.parquet":pq}):
                meta=scan_tapes(Path(tmp),events)
        self.assertEqual(meta["trade_daily_files_read"],1)
        self.assertEqual(events[0].reason,"TARGET")
        self.assertEqual(events[1].reason,"TARGET")

    def test_end_to_end_run_creates_all_outputs_without_original_file_mutation(self):
        from src.data_engineering.modeling.liquidity_absorption_stage_a03c import run
        synthetic = [event(ident=i) for i in range(452)]
        for s in synthetic:
            s.update(reason="TARGET", target_rr_initial=1.0)
        e=pd.DataFrame(synthetic)
        t=pd.DataFrame({"time_ms":[T+100,T+2000,T+3000,T+10_000],
                        "price":[100.15,100.,99.9,101.],
                        "is_buyer_maker":[False,True,True,False]})
        class Batch:
            def to_pandas(self):return t
        class PFile:
            schema=types.SimpleNamespace(names=["time_ms","price","is_buyer_maker"])
            def __init__(self,p):pass
            def iter_batches(self,**kw):yield Batch()
        fake=types.ModuleType("pyarrow.parquet");fake.ParquetFile=PFile
        parent=types.ModuleType("pyarrow");parent.__path__=[];parent.parquet=fake
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); b=root/"notebooks"/"Liquidity_Absorption"/"03b";b.mkdir(parents=True)
            (b/"03b_decision.json").write_text('{"version":"LA-A03b-0.1.0","source_events":452}')
            e.to_csv(b/"03b_1to1_events.csv",index=False)
            before=(b/"03b_1to1_events.csv").read_bytes()
            fp=root/"aggTrades"/"BTCUSDT"/"BTCUSDT-aggTrades-2026-07-02.parquet";fp.parent.mkdir(parents=True);fp.touch()
            with patch.dict(sys.modules,{"pyarrow":parent,"pyarrow.parquet":fake}):
                result=run(root,root,root/"03c")
            self.assertEqual(result["policy_rows"],452*3)
            self.assertEqual(result["tape"]["trade_daily_files_read"],1)
            self.assertEqual(before,(b/"03b_1to1_events.csv").read_bytes())
            for name in ("03c_events.csv","03c_coverage.csv","03c_equity_paths.csv",
                         "03c_fee_sensitivity.csv","03c_contract.json","03c_decision.json",
                         "03c_net_equity.png","03c_gross_equity.png"):
                self.assertTrue((root/"03c"/name).exists(), name)

    def test_equity_and_fee_sensitivity_have_no_required_full_tape(self):
        a=TapeStudy(event(),"MAKER_TOUCH_3RD",Config())
        a.feed(T+2000,100,True)
        a.feed(T+3000,101,False)
        df=pd.DataFrame([a.record()]); paths=equity(df)
        self.assertEqual(len(paths),2)  # BTC and combined
        sen=_fee_sensitivity(df,Config())
        self.assertTrue(len(sen)>0)
        self.assertGreater(float(sen.iloc[2]["mean_net_bp_per_closed"]),float(sen.iloc[0]["mean_net_bp_per_closed"]))

MINUTE=60_000
if __name__=="__main__":unittest.main()
