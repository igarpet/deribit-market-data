"""Synthetic, causal regression tests for LA-A04. No historical/strategy optimization."""
from __future__ import annotations
import sys
import unittest
from unittest.mock import patch
import types
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'data_engineering' / 'modeling'))
from liquidity_absorption_stage_a04 import (one_landmark, build_landmarks, validate_primary,
    summarize_landmarks, cumulative_competing_exits, load_minute_bars, Config, MIN_MS)


def fixtures():
    signal = pd.Timestamp('2026-07-01 12:00:00Z')
    t = signal.value // 1_000_000
    bars=[]
    for i in range(-35, 71):
        c=t+i*MIN_MS
        px=100.0+(i if i>0 else 0)*.05
        bars.append(dict(close_ms=c,open=px,high=px+.1,low=px-.1,close=px,
                         quote_volume=1000.,taker_buy_quote_volume=600.))
    frame=pd.DataFrame(bars).set_index('close_ms',drop=False)
    event={'a02_candidate_row':1,'symbol':'BTCUSDT','side':'long','signal_ms':t,
           'entry_time_ms':t+100,'exit_time_ms':t+45*MIN_MS,'reason':'TARGET',
           'entry_print':100.,'level_price':99.8,'initial_stop':99.,'target':102.,
           'zone_halfwidth':.2,'vol_sigma_bp_60':100.,'target_to_sigma60':2.,
           'time_to_exit_min':45.,'target_source':'SESSION_VWAP','level_type':'pivot_5m'}
    return frame,event


class TestLandmarks(unittest.TestCase):
    def test_archive_datetime_ms_unit_does_not_collapse_minute_index(self):
        # Regression for a pandas datetime64[ms, UTC] parser output.
        # Mock the one-minute parquet reader and upstream parser; keep the
        # actual A04 load_minute_bars code under test.
        starts = pd.Series([1787740800000, 1787740860000, 1787740920000])
        candles = pd.DataFrame({"open_time": starts,
            "open": [100., 101., 102.], "high": [101., 102., 103.],
            "low": [99., 100., 101.], "close": [100., 101., 102.],
            "quote_volume": [1000., 1000., 1000.],
            "taker_buy_quote_volume": [500., 500., 500.]})
        ms_series = pd.to_datetime(starts, unit="ms", utc=True).dt.as_unit("ms")
        module = types.ModuleType("src.data_engineering.modeling.liquidity_absorption_stage_a")
        module.parse_kline_open_time = lambda _: ms_series
        with patch("pandas.read_parquet", return_value=candles), patch.dict(
            sys.modules, {module.__name__: module}):
            result = load_minute_bars(Path("synthetic.parquet"))
        expected = (starts + MIN_MS).tolist()
        self.assertEqual(result.index.to_list(), expected)
        self.assertEqual(result.close_ms.to_list(), expected)
        self.assertEqual(result.index.is_unique, True)
        self.assertEqual(np.diff(result.index.to_numpy()).tolist(), [MIN_MS, MIN_MS])

    def test_archive_datetime_microseconds_also_converts_to_epoch_ms(self):
        starts = pd.Series([1787740800000, 1787740860000])
        candles = pd.DataFrame({"open_time": starts,
            "open": [100., 101.], "high": [101., 102.],
            "low": [99., 100.], "close": [100., 101.],
            "quote_volume": [1000., 1000.],
            "taker_buy_quote_volume": [500., 500.]})
        us_series = pd.to_datetime(starts, unit="ms", utc=True).dt.as_unit("us")
        module = types.ModuleType("src.data_engineering.modeling.liquidity_absorption_stage_a")
        module.parse_kline_open_time = lambda _: us_series
        with patch("pandas.read_parquet", return_value=candles), patch.dict(
            sys.modules, {module.__name__: module}):
            result = load_minute_bars(Path("synthetic.parquet"))
        self.assertEqual(result.index.to_list(), (starts + MIN_MS).tolist())

    def test_first_five_min_price_and_flow(self):
        a,b=fixtures();o=one_landmark(b,a,5)
        self.assertEqual(o['observation_status'],'PRESENT')
        self.assertTrue(o['at_risk'])
        self.assertEqual(o['bars_complete'],5)
        self.assertAlmostEqual(o['signed_return_bp'],25.)
        self.assertAlmostEqual(o['flow_alignment'],.2)
        self.assertAlmostEqual(o['opposing_aggressor_share'],.4)
        self.assertAlmostEqual(o['progress_to_target'],.125)
        self.assertAlmostEqual(o['quote_activity_vs_past30'],1.)

    def test_no_future_candle_leakage(self):
        a,b=fixtures();x=one_landmark(b,a,5)
        a.loc[a.close_ms>=(b['signal_ms']+6*MIN_MS),'high']=100000.
        a.loc[a.close_ms>=(b['signal_ms']+6*MIN_MS),'low']=.00001
        a.loc[a.close_ms>=(b['signal_ms']+6*MIN_MS),'taker_buy_quote_volume']=0.
        y=one_landmark(b,a,5)
        for k in ('max_favorable_bp','max_adverse_bp','flow_alignment','signed_return_bp'):
            self.assertAlmostEqual(x[k],y[k])

    def test_stop_before_landmark_removes_from_risk_set(self):
        a,b=fixtures();b['reason']='STOP';b['exit_time_ms']=b['signal_ms']+3*MIN_MS
        z=one_landmark(b,a,5)
        self.assertEqual(z['observation_status'],'EXITED_BEFORE_LANDMARK')
        self.assertFalse(z['at_risk'])

    def test_exact_exit_at_checkpoint_is_not_alive(self):
        a,b=fixtures();b['exit_time_ms']=b['signal_ms']+5*MIN_MS
        self.assertEqual(one_landmark(b,a,5)['observation_status'],'EXITED_BEFORE_LANDMARK')

    def test_incomplete_one_minute_window_is_not_imputed(self):
        a,b=fixtures();a=a.drop(index=b['signal_ms']+2*MIN_MS)
        self.assertEqual(one_landmark(b,a,5)['observation_status'],'INCOMPLETE_MINUTE_WINDOW')

    def test_fresh_landmark_window_does_not_include_signal_bar(self):
        a,b=fixtures(); x=one_landmark(b,a,1)
        self.assertEqual(x['bars_complete'],1)
        self.assertAlmostEqual(x['signed_return_bp'],5.)
        self.assertAlmostEqual(x['max_favorable_bp'],15.)

    def test_reject_bad_active_geometry(self):
        a,b=fixtures();b['target']=99.
        with self.assertRaises(ValueError):one_landmark(b,a,5)

    def test_missing_exit_is_not_implicitly_positive(self):
        a,b=fixtures();b['reason']='CENSORED_TAPE_END';b['exit_time_ms']=np.nan
        self.assertEqual(one_landmark(b,a,5)['observation_status'],'NOT_COMPLETED_COHORT')

    def test_short_trade_aggressive_flow_alignment_and_adverse(self):
        a,b=fixtures();b.update(side='short',entry_print=100.,initial_stop=101.,target=98.,level_price=100.2)
        z=one_landmark(b,a,5)
        self.assertAlmostEqual(z['flow_alignment'],-.2)
        self.assertLess(z['signed_return_bp'],0)
        self.assertLess(z['max_adverse_r'],0)

    def test_build_cohort_has_all_landmarks_and_statuses(self):
        a,b=fixtures();c=b.copy();c['a02_candidate_row']=2;c['reason']='STOP';c['exit_time_ms']=c['signal_ms']+2*MIN_MS
        z=build_landmarks(pd.DataFrame([b,c]),{'BTCUSDT':a}, Config(landmarks_min=(1,5)))
        self.assertEqual(len(z),4)
        self.assertEqual((z.observation_status=='PRESENT').sum(),3)

    def test_summary_is_labeled_by_eventual_outcome_and_horizon(self):
        a,b=fixtures();c=b.copy();c['a02_candidate_row']=2;c['reason']='STOP';c['exit_time_ms']=c['signal_ms']+30*MIN_MS
        z=build_landmarks(pd.DataFrame([b,c]),{'BTCUSDT':a},Config(landmarks_min=(5,15)))
        out=summarize_landmarks(z)
        self.assertEqual(len(out),4)
        self.assertTrue((out.n_at_risk==1).all())

    def test_cumulative_outcomes_do_not_count_future_exit(self):
        a,b=fixtures();b['signal_time_utc']='2026-07-01T12:00:00Z'
        c=b.copy();c['a02_candidate_row']=2;c['reason']='STOP';c['time_to_exit_min']=2.
        z=cumulative_competing_exits(pd.DataFrame([b,c]), horizons=(5,60))
        five=z[(z.symbol=='ALL')&(z.horizon_min==5)].iloc[0]
        self.assertEqual(five.n_stop,1)
        self.assertEqual(five.n_target,0)
        sixty=z[(z.symbol=='ALL')&(z.horizon_min==60)].iloc[0]
        self.assertEqual(sixty.n_stop,1)
        self.assertEqual(sixty.n_target,1)

    def test_primary_enforces_452_unique_candidate_ids(self):
        a,b=fixtures();base={k:v for k,v in b.items() if k!='signal_ms'}
        base.update(signal_time_utc='2026-07-01T12:00:00Z', entry_kind='TAKER_FIRST_POST_SIGNAL',
                    model_mode='VD18_VOL_FIXED', hold_minutes=240)
        df=pd.DataFrame([{**base,'a02_candidate_row':i} for i in range(452)])
        r=validate_primary(df)
        self.assertEqual(len(r),452)
        self.assertAlmostEqual(r.target_to_sigma60.iloc[0],2.)
        with self.assertRaises(ValueError):validate_primary(df.iloc[1:])
        df.loc[1,'a02_candidate_row']=0
        with self.assertRaises(ValueError):validate_primary(df)

    def test_pre_entry_baseline_does_not_read_future_quote_volume(self):
        a,b=fixtures();baseline=one_landmark(b,a,5)['quote_activity_vs_past30']
        a.loc[a.close_ms>b['signal_ms']+5*MIN_MS,'quote_volume']=1e9
        self.assertAlmostEqual(one_landmark(b,a,5)['quote_activity_vs_past30'],baseline)


if __name__=='__main__':unittest.main()
