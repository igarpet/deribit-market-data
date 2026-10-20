"""A05a: strictly timestamped 1m event-study tests; synthetic only."""
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

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"/"data_engineering"/"modeling"))
from liquidity_absorption_stage_a05a import (
    VERSION, MIN_MS, Config, check_config, ts_millis, load_event_cohort,
    load_context_minutes, event_minute_panel, build_panel, summaries,phase_panels,run)

S=pd.Timestamp("2026-07-10T12:00:00Z")
SM=int(S.timestamp()*1000)


def event(outcome="TARGET",sym="BTCUSDT",id=0):
    return dict(a02_candidate_row=id,symbol=sym,side="long",eventual_outcome=outcome,
      feasibility_band="TARGET_LE_1SIGMA60",signal_ms=SM,
      bar_close=100.,level_price=99.9,zone_halfwidth=.05,
      entry_time_ms=SM+200,exit_time_ms=SM+10*MIN_MS+300,
      first_touch_ms=SM-5*MIN_MS,level_source_ms=SM-45*MIN_MS,
      vol_sigma_bp_60=25.,target_to_sigma60=.7,level_type="prior_day")


def minutes(gap=None):
    offsets=np.arange(-310,245)
    if gap is not None: offsets=offsets[offsets!=gap]
    times=SM+offsets*MIN_MS
    px=np.where(offsets<0,100+offsets*.01,100+offsets*.03).astype(float)
    px[offsets==0]=100.
    d=pd.DataFrame({"close_ms":times,"open":px-.015,"high":px+.025,"low":px-.035,
                    "close":px,"quote_volume":np.full(len(times),100.),
                    "taker_buy_quote_volume":np.full(len(times),65.)})
    group=(d.close_ms.diff().ne(MIN_MS)).cumsum()
    d['log_return_1m']=np.log(d.close/d.close.shift()).where(d.close_ms.diff().eq(MIN_MS))
    for n in [15,60]:
        d[f'rv_{n}m_bp']=d.log_return_1m.pow(2).groupby(group).transform(lambda z:np.sqrt(z.rolling(n,min_periods=n).sum())*10000)
    d['quote_prior60_mean']=d.quote_volume.groupby(group).transform(lambda s:s.shift(1).rolling(60,min_periods=60).mean())
    d['quote_vs_prior60']=d.quote_volume/d.quote_prior60_mean
    d['net_buy_quote']=2*d.taker_buy_quote_volume-d.quote_volume
    net=d.net_buy_quote.groupby(group).transform(lambda s:s.rolling(15,min_periods=15).sum())
    vol=d.quote_volume.groupby(group).transform(lambda s:s.rolling(15,min_periods=15).sum())
    d['flow_15m_buy_signed']=net/vol
    d['volume_15m_vs_prior60']=d.quote_volume.groupby(group).transform(lambda s:s.rolling(15,min_periods=15).mean())/d.quote_prior60_mean
    return d.set_index('close_ms',drop=False)


class TestA05a(unittest.TestCase):
    def test_contract_and_versions(self):
        self.assertEqual(VERSION,"LA-A05a-0.1.0")
        check_config(Config())
        with self.assertRaises(ValueError):check_config(Config(after_min=180))
        with self.assertRaises(ValueError):check_config(Config(target_sigma_ratio_split=1.5))

    def test_all_481_minutes_and_axis(self):
        r=event_minute_panel(event(),minutes())
        self.assertEqual(len(r),481)
        self.assertEqual([r.relative_min.min(),r.relative_min.max()],[-240,240])
        self.assertEqual(int(r.loc[r.relative_min.eq(0),'utc_close_ms'].iloc[0]),SM)
        self.assertTrue(bool(r[r.relative_min.eq(0)].is_signal_bar.iloc[0]))

    def test_no_future_1m_data_in_pre_signal(self):
        a=minutes();e=event();x=event_minute_panel(e,a)
        a.loc[a.close_ms>SM,['close','high','low','taker_buy_quote_volume']]=[1e8,1e8,1e8,0.]
        y=event_minute_panel(e,a)
        for field in ['signed_close_from_signal_bp','rv_15m_bp','flow_alignment_15m']:
            x0=x.loc[x.relative_min.le(0),field].to_numpy(float)
            y0=y.loc[y.relative_min.le(0),field].to_numpy(float)
            np.testing.assert_allclose(x0,y0,equal_nan=True)

    def test_anchor_price_and_symmetry(self):
        x=event_minute_panel(event(),minutes())
        self.assertAlmostEqual(float(x.loc[x.relative_min.eq(0),'signed_close_from_signal_bp'].iloc[0]),0.)
        self.assertAlmostEqual(float(x.loc[x.relative_min.eq(-10),'signed_close_from_signal_bp'].iloc[0]),-10.)
        self.assertAlmostEqual(float(x.loc[x.relative_min.eq(10),'signed_close_from_signal_bp'].iloc[0]),30.)

    def test_first_second_and_level_source_markers(self):
        x=event_minute_panel(event(),minutes())
        self.assertEqual(x.loc[x.first_touch_marker,'relative_min'].tolist(),[-5])
        self.assertEqual(x.loc[x.level_source_marker,'relative_min'].tolist(),[-45])
        self.assertEqual(x.loc[x.second_touch_marker,'relative_min'].tolist(),[0])
        self.assertFalse(x[x.relative_min.eq(-46)].level_source_timestamp_le_bar.iloc[0])

    def test_post_exit_market_path_does_not_count_as_active_trade(self):
        z=event_minute_panel(event(),minutes())
        t=z[z.relative_min.eq(60)].iloc[0]
        self.assertTrue(t.has_minute_bar)
        self.assertTrue(np.isfinite(t.signed_close_from_signal_bp))
        self.assertFalse(t.trade_active_at_t)
        self.assertTrue(t.trade_exited_by_t)

    def test_active_marker_is_not_at_signal_and_exit_not_counted_active(self):
        z=event_minute_panel(event(),minutes())
        self.assertFalse(z[z.relative_min.eq(0)].trade_entered_by_t.iloc[0])
        self.assertTrue(z[z.relative_min.eq(1)].trade_active_at_t.iloc[0])
        self.assertFalse(z[z.relative_min.eq(11)].trade_active_at_t.iloc[0])

    def test_missing_market_minute_no_forward_fill(self):
        a=minutes(gap=-100)
        z=event_minute_panel(event(),a)
        row=z[z.relative_min.eq(-100)].iloc[0]
        self.assertFalse(row.has_minute_bar)
        self.assertTrue(np.isnan(row.signed_close_from_signal_bp))
        self.assertTrue(np.isnan(row.flow_alignment_1m))

    def test_flow_direction_short_sign_reverses(self):
        a=minutes();long=event_minute_panel(event(),a)
        short=event();short['side']='short';s=event_minute_panel(short,a)
        self.assertAlmostEqual(long[long.relative_min.eq(15)].flow_alignment_15m.iloc[0],.3)
        self.assertAlmostEqual(s[s.relative_min.eq(15)].flow_alignment_15m.iloc[0],-.3)

    def test_roll_volume_before_minute_not_future(self):
        a=minutes();z=event_minute_panel(event(),a)
        x=z[z.relative_min.eq(-120)].iloc[0]
        self.assertAlmostEqual(x.quote_vs_prior60,1.)
        a.loc[a.close_ms>SM-120*MIN_MS,'quote_volume']=1e12
        y=event_minute_panel(event(),a)
        self.assertAlmostEqual(y[y.relative_min.eq(-120)].quote_vs_prior60.iloc[0],1.)

    def test_invalid_signal_close_raises(self):
        a=event();a['bar_close']=102.
        with self.assertRaisesRegex(ValueError,'Frozen A02b signal close'):
            event_minute_panel(a,minutes())

    def test_full_panel_propagates_all_signals(self):
        a=event();b=event(outcome='STOP',id=1)
        res=build_panel(pd.DataFrame([a,b]),{'BTCUSDT':minutes()})
        self.assertEqual(len(res),962)
        self.assertEqual(res.a02_candidate_row.nunique(),2)
        self.assertEqual(res.eventual_outcome.value_counts()['STOP'],481)

    def test_summaries_separate_outcomes_and_feasibility(self):
        a=event();b=event(id=1,outcome='STOP');c=event(id=2,outcome='TIMEOUT')
        c['feasibility_band']='TARGET_GT_1SIGMA60'
        pp=build_panel(pd.DataFrame([a,b,c]),{'BTCUSDT':minutes()})
        summ,marks,qc=summaries(pp,pd.DataFrame([a,b,c]))
        allx=summ[(summ.symbol=='ALL')&(summ.feasibility_band=='ALL')&(summ.relative_min==0)]
        self.assertEqual(len(allx),3)
        self.assertEqual(len(marks[marks.relative_min==0]),len(summ[summ.relative_min==0]))
        self.assertEqual(int(qc.complete_pre_240.sum()),3)
        self.assertEqual(int(qc.complete_post_240.sum()),3)

    def test_phases_are_fixed_and_complete(self):
        sample=event_minute_panel(event(),minutes())
        phases,summary=phase_panels(sample)
        self.assertEqual(len(phases),6)
        self.assertEqual(phases.expected_bars.sum(),481)
        self.assertTrue(phases.complete_window.all())
        self.assertEqual(int(phases.loc[phases.phase.eq('formation_-15_0'),'first_touch_in_phase'].iloc[0]),1)
        self.assertEqual(int(phases.loc[phases.phase.eq('immediate_+1_+15'),'trade_active_at_phase_end'].iloc[0]),0)
        self.assertEqual(len(summary),12)  # observed feasibility and ALL band

    def test_phase_missing_bar_does_not_silently_use_partial_window(self):
        sample=event_minute_panel(event(),minutes(gap=-100))
        phases,_=phase_panels(sample)
        z=phases.loc[phases.phase.eq('context_-240_-61')].iloc[0]
        self.assertFalse(z.complete_window)
        self.assertEqual(z.observed_bars,z.expected_bars-1)
        self.assertTrue(np.isnan(z.signed_price_change_bp))

    def test_ms_and_us_precision(self):
        values=pd.Series(['2026-07-10T12:00:00Z','2026-07-10T12:01:00Z'])
        for unit in ('ms','us','ns'):
            series=pd.to_datetime(values,utc=True).dt.as_unit(unit)
            self.assertEqual(np.diff(ts_millis(series).to_numpy()).tolist(),[MIN_MS])

    def test_load_minute_bar_timestamp_from_ms_precision(self):
        a=minutes().iloc[:80].reset_index(drop=True)
        a['open_time']=pd.to_datetime(a.close_ms-MIN_MS,unit='ms',utc=True).dt.as_unit('ms')
        module=types.ModuleType('src.data_engineering.modeling.liquidity_absorption_stage_a')
        module.parse_kline_open_time=lambda _:a.open_time
        with patch('pandas.read_parquet',return_value=a),patch.dict(sys.modules,{module.__name__:module}):
            z=load_context_minutes(Path('synthetic.parquet'))
        self.assertTrue(z.index.is_unique)
        self.assertEqual(np.unique(np.diff(z.index)).tolist(),[MIN_MS])
        self.assertTrue(np.isfinite(z.iloc[-1].rv_15m_bp))

    def test_gap_invalidates_rolling_and_restores_after_complete_window(self):
        a=minutes(gap=0).reset_index(drop=True)
        a['open_time']=pd.to_datetime(a.close_ms-MIN_MS,unit='ms',utc=True)
        module=types.ModuleType('src.data_engineering.modeling.liquidity_absorption_stage_a')
        module.parse_kline_open_time=lambda _:a.open_time
        with patch('pandas.read_parquet',return_value=a),patch.dict(sys.modules,{module.__name__:module}):
            z=load_context_minutes(Path('synthetic.parquet'))
        self.assertTrue(np.isnan(z.loc[SM+MIN_MS,'rv_15m_bp']))
        self.assertTrue(np.isfinite(z.loc[SM+62*MIN_MS,'rv_60m_bp']))

    def test_join_identity_and_touch_validation(self):
        with tempfile.TemporaryDirectory() as td:
            repo=Path(td);b=repo/'notebooks'/'Liquidity_Absorption'
            (b/'04').mkdir(parents=True);(b/'02b').mkdir(parents=True)
            (b/'04'/'04_decision.json').write_text(json.dumps({'version':'LA-A04-0.1.1','original_candidates':452}))
            rows=[];features=[]
            for i in range(452):
                rows.append(dict(a02_candidate_row=i,symbol='BTCUSDT',side='long',reason='TARGET',
                    signal_time_utc=str(S),entry_time_ms=SM+200,exit_time_ms=SM+20*MIN_MS,
                    level_price=99.9,zone_halfwidth=.05,target=101.,initial_stop=99.,entry_print=100.,
                    vol_sigma_bp_60=25.,target_to_sigma60=.7,entry_kind='TAKER_FIRST_POST_SIGNAL',
                    model_mode='VD18_VOL_FIXED',hold_minutes=240))
                features.append(dict(a02_candidate_row=i,symbol='BTCUSDT',side='long',signal_time_utc=str(S),
                    level_price=99.9,bar_close=100.,first_touch_close=str(S-pd.Timedelta(minutes=5)),
                    second_touch_close=str(S),level_source_time=str(S-pd.Timedelta(minutes=60)),level_type='pivot_5m'))
            pd.DataFrame(rows).to_csv(b/'04'/'04_primary_event_outcomes.csv',index=False)
            pd.DataFrame(features).to_csv(b/'02b'/'02b_touch_features.csv',index=False)
            x=load_event_cohort(repo)
            self.assertEqual(len(x),452)
            self.assertEqual(x.first_touch_ms.iloc[0],SM-5*MIN_MS)
            features[20]['level_price']=97.
            pd.DataFrame(features).to_csv(b/'02b'/'02b_touch_features.csv',index=False)
            with self.assertRaisesRegex(ValueError,'level-price drift'):load_event_cohort(repo)

    def test_roll_15m_aggregate_flow_matches_formula(self):
        z=event_minute_panel(event(),minutes())
        for i in [-180,-60,0,30,240]:
            t=z[z.relative_min.eq(i)].iloc[0]
            self.assertAlmostEqual(t.flow_alignment_15m,.3,places=10)

if __name__=='__main__':unittest.main()
