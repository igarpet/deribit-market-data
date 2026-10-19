"""LA-A02b synthetic causality, mapping and missing-data tests. No market data required."""
from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data_engineering.modeling import liquidity_absorption_stage_a02b as a02b


FIRST = pd.Timestamp("2026-07-01 00:05:00Z")
SECOND = pd.Timestamp("2026-07-01 00:09:00Z")


def stamps(*offsets):
    return np.array([int((FIRST + pd.Timedelta(seconds=d)).timestamp() * 1000) for d in offsets], dtype="int64")


def make_acc(side="long"):
    return a02b.Accumulator(row=3, symbol="BTCUSDT", side=side,
        first_ms=a02b._ms(FIRST), second_ms=a02b._ms(SECOND),
        level=100.0, halfwidth=.5, config=a02b.Config())


def candles(first=FIRST, second=SECOND):
    times = pd.date_range(first, second, freq="min", tz="UTC")
    out = pd.DataFrame({"open":100., "high":100.8, "low":99.8,
                        "close":100.3, "quote_volume":1000., "taker_buy_quote_volume":400.},index=(times.astype("int64") // 1_000_000))
    out.index.name="close_ms"
    return out


def sample_prints():
    return pd.DataFrame({
        # first touch 00:04:10..20, second touch 00:08:15..30;
        # 00:09:00 is future and must be excluded.
        "time_ms":stamps(-300,-290,-50,-40,100,195,200,240),
        "price":[101.,101.,100.,100.,100.8,100.,100.,100.],
        "qty":[5.,1.,1.,2.,100.,3.,1.,1000.],
        "is_buyer_maker":[False,True,False,True,True,True,False,False],
    })


class TestA02b(unittest.TestCase):
    def test_touch_flow_exact_prices_and_excludes_post_signal(self):
        a=make_acc()
        t=sample_prints()
        a.accept_batch(t.time_ms.to_numpy(),t.price.to_numpy(),t.qty.to_numpy(),t.is_buyer_maker.to_numpy())
        out=a02b.finalize_features(a,candles())
        self.assertEqual(out["touch_status"],"PRESENT")
        self.assertAlmostEqual(out["first_zone_quote"],300.)
        self.assertAlmostEqual(out["first_zone_buy_quote"],100.)
        self.assertAlmostEqual(out["first_zone_sell_quote"],200.)
        self.assertAlmostEqual(out["second_zone_quote"],400.)
        self.assertAlmostEqual(out["second_zone_buy_quote"],100.)
        self.assertAlmostEqual(out["second_zone_sell_quote"],300.)
        self.assertAlmostEqual(out["zone_quote_touch_ratio"],4/3)
        self.assertAlmostEqual(out["first_opposing_zone_share"],2/3)
        self.assertAlmostEqual(out["second_opposing_zone_share"],3/4)
        # 00:09:00 trade volume 100000 and 00:06:40 out-of-zone trade ignored.
        self.assertAlmostEqual(out["zone_quote_between_touches"],700.)
        self.assertAlmostEqual(out["pre_first_profile_quote"],606.)
        self.assertAlmostEqual(out["pre_first_profile_zone_share"],0.)
        self.assertEqual(out["zone_active_print_seconds_between_touches"],4)

    def test_short_side_opposing_buy_is_mirrored(self):
        a=make_acc("short")
        t=sample_prints()
        a.accept_batch(t.time_ms.to_numpy(),t.price.to_numpy(),t.qty.to_numpy(),t.is_buyer_maker.to_numpy())
        out=a02b.finalize_features(a,candles())
        self.assertAlmostEqual(out["first_opposing_zone_share"],1/3)
        self.assertAlmostEqual(out["second_opposing_zone_share"],1/4)

    def test_missing_first_touch_volume_remains_undefined_not_infinite(self):
        a=make_acc()
        t=sample_prints().iloc[[5,6]]
        a.accept_batch(t.time_ms.to_numpy(),t.price.to_numpy(),t.qty.to_numpy(),t.is_buyer_maker.to_numpy())
        out=a02b.finalize_features(a,candles())
        self.assertEqual(out["first_zone_quote"],0.)
        self.assertTrue(np.isnan(out["zone_quote_touch_ratio"]))
        self.assertTrue(np.isnan(out["first_opposing_zone_share"]))
        self.assertEqual(out["second_zone_quote"],400.)

    def test_missing_day_masks_partial_touch_measurements(self):
        a=make_acc()
        a.missing_touch_days=["2026-07-01"]
        out=a02b.finalize_features(a,candles())
        self.assertEqual(out["touch_status"],"MISSING_TRADE_DAY")
        self.assertTrue(np.isnan(out["first_zone_quote"]))
        self.assertTrue(np.isnan(out["zone_quote_touch_ratio"]))
        self.assertTrue(np.isnan(out["zone_quote_early"]))
        self.assertTrue(np.isnan(out["zone_quote_per_intersecting_1m"]))

    def test_missing_pretouch_archive_not_claimed_zero_profile(self):
        a=make_acc()
        a.missing_pre_days=["2026-06-30"]
        out=a02b.finalize_features(a,candles())
        self.assertEqual(out["pre_first_profile_status"],"MISSING_TRADE_DAY")
        self.assertTrue(np.isnan(out["pre_first_profile_quote"]))
        self.assertEqual(out["touch_status"],"PRESENT")

    def test_not_using_future_trades_or_future_candles(self):
        a=make_acc()
        b=make_acc()
        source=sample_prints()
        a.accept_batch(source.time_ms.to_numpy(),source.price.to_numpy(),source.qty.to_numpy(),source.is_buyer_maker.to_numpy())
        changed=pd.concat([source,pd.DataFrame({"time_ms":stamps(250,450),"price":[100.,100.],
                             "qty":[999999.,999999.],"is_buyer_maker":[False,True]})],ignore_index=True)
        b.accept_batch(changed.time_ms.to_numpy(),changed.price.to_numpy(),changed.qty.to_numpy(),changed.is_buyer_maker.to_numpy())
        x,y=a02b.finalize_features(a,candles()),a02b.finalize_features(b,candles())
        for k in ("first_zone_quote","second_zone_quote","pre_first_poc_price","zone_active_print_seconds_between_touches"):
            self.assertEqual(x[k],y[k])

    def test_price_and_time_zone_boundaries(self):
        a=make_acc()
        tm=np.array([a.first_start,a.first_start+1,a.first_start+2,a.second_ms],dtype="int64")
        prices=np.array([99.5,100.5,100.50001,100.],dtype="float64")
        a.accept_batch(tm,prices,np.ones(4),np.array([True,False,True,False]))
        x=a02b.finalize_features(a,candles())
        self.assertAlmostEqual(x["first_zone_quote"],200.)
        self.assertAlmostEqual(x["second_zone_quote"],0.)

    def test_early_late_normalize_by_visiting_candles(self):
        a=make_acc()
        a.zone_quote[:]=[200.,0.,0.,0.,600.]
        c=candles()
        c.loc[c.index[1:4],"low"]=100.65  # no zone intersection between touches
        c.loc[c.index[1:4],"high"]=101.
        out=a02b.finalize_features(a,c)
        self.assertEqual(out["intersecting_1m_candles"],2)
        self.assertEqual(out["early_intersecting_1m"],1)
        self.assertEqual(out["late_intersecting_1m"],1)
        self.assertAlmostEqual(out["late_early_zone_rate_ratio"],3.)

    def test_read_candidate_provenance_and_reject_signal_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"c.csv"
            frame=pd.DataFrame({"symbol":["BTCUSDT"],"side":["long"],"level_price":[100.],
                 "zone_halfwidth":[.5],"first_touch_close":[FIRST.isoformat()],
                 "second_touch_close":[SECOND.isoformat()],"signal_time_utc":[SECOND.isoformat()]})
            frame.to_csv(p,index=False)
            out=a02b.read_candidates(p)
            self.assertEqual(out.a02_candidate_row.iloc[0],0)
            frame.loc[0,"signal_time_utc"]=FIRST.isoformat()
            frame.to_csv(p,index=False)
            with self.assertRaises(ValueError):a02b.read_candidates(p)

    def test_cross_midnight_day_classification(self):
        ms=a02b._ms(pd.Timestamp("2026-07-01T00:00:00Z"))
        ds=a02b._utc_days(ms-2*MS,ms+2*MS)
        self.assertEqual(ds,{"2026-06-30","2026-07-01"})

    def test_streaming_joins_source_rows_without_new_event_selection(self):
        a=make_acc()
        in_events=pd.DataFrame({"a02_candidate_row":[3],"symbol":[a.symbol],"side":[a.side],
         "first_touch_close":[FIRST],"second_touch_close":[SECOND],
         "level_price":[a.level],"zone_halfwidth":[a.halfwidth]})
        t=sample_prints()
        class Batch:
            def to_pandas(self): return t
        class ParquetFile:
            schema=types.SimpleNamespace(names=list(t.columns))
            def __init__(self,path): pass
            def iter_batches(self, **kwargs): yield Batch()
        lib=types.ModuleType("pyarrow")
        lib.__path__=[]
        m=types.ModuleType("pyarrow.parquet")
        m.ParquetFile=ParquetFile
        lib.parquet=m
        with tempfile.TemporaryDirectory() as tmp:
            fp=Path(tmp)/"aggTrades"/"BTCUSDT"/"BTCUSDT-aggTrades-2026-07-01.parquet"
            fp.parent.mkdir(parents=True)
            fp.touch()
            with patch.object(a02b,"_load_minute_context",return_value=candles()), patch.dict(sys.modules,{"pyarrow":lib,"pyarrow.parquet":m}):
                result=a02b.enrich_a02_candidates(in_events,Path(tmp))
        self.assertEqual(len(result),1)
        self.assertEqual(result.a02_candidate_row.iloc[0],3)
        self.assertAlmostEqual(result.second_zone_quote.iloc[0],400.)
        # The pre-first profile begins on 2026-06-30; preserve the partial/missing classification.
        self.assertEqual(result.pre_first_profile_status.iloc[0],"MISSING_TRADE_DAY")


MS=MS_PER_MINUTE=60_000

if __name__=="__main__":unittest.main()
