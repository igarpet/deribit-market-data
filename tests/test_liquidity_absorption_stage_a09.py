"""LA09 frozen-level lifecycle, strict availability and no future leakage tests."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_engineering.modeling import liquidity_absorption_stage_a09 as la


def synthetic_bar_data(start='2026-07-01 09:00:00Z', n=460):
    ts=pd.Timestamp(start)
    times=np.array([(ts.value//1_000_000)+i*la.MIN_MS for i in range(n)],dtype='int64')
    c=100+(np.arange(n)*.012)
    out=pd.DataFrame({'close_ms':times,'open':c-.005,'high':c+.2,'low':c-.2,'close':c,
                      'quote_volume':1000.,'taker_buy_quote_volume':550.,
                      'logret_bp':.012,'quote_vs_prior60':1.0})
    return out.set_index('close_ms',drop=False)


def fake_event(bars,side='long', level_type='pivot_5m'):
    base=int(bars.close_ms.iloc[0]);avail=base+90*la.MIN_MS;first=avail+15*la.MIN_MS;sig=first+5*la.MIN_MS
    source=avail-10*la.MIN_MS if level_type=='pivot_5m' else avail-1440*la.MIN_MS
    return {'a02_candidate_row':0,'symbol':'BTCUSDT','side':side,'level_type':level_type,
            'level_price':101.,'zone_halfwidth':.5,'level_source_time_ms':source,
            'available_ms':avail,'first_touch_close_ms':first,'second_touch_close_ms':sig,
            'signal_time_utc_ms':sig,'reason':'TARGET','target':103.,'entry_time_ms':sig+1500,
            'exit_time_ms':sig+11*la.MIN_MS,'a03c_maker_touch_reason':'NO_THIRD_TOUCH'}


class TestCausalTimeline(unittest.TestCase):
    def setUp(self):
        self.b=synthetic_bar_data();self.e=fake_event(self.b)

    def test_times_assigned_at_completed_close(self):
        z=la.event_stages(self.e,self.b)
        self.assertEqual(len(z),8)
        self.assertEqual(z[0]['expected_minute_bars'],60)
        self.assertEqual(z[1]['expected_minute_bars'],10)
        self.assertEqual(z[2]['expected_minute_bars'],15)
        self.assertEqual(z[3]['expected_minute_bars'],5)
        self.assertEqual(z[4]['expected_minute_bars'],5)
        self.assertEqual(z[5]['expected_minute_bars'],10)
        self.assertEqual(z[6]['expected_minute_bars'],45)
        self.assertEqual(z[7]['expected_minute_bars'],180)

    def test_available_excludes_pivot_source_and_pre_available_zone(self):
        stages=la.event_stages(self.e,self.b)
        self.assertFalse(stages[0]['level_known_during_stage'])
        self.assertTrue(np.isnan(stages[0]['zone_intersect_fraction']))
        self.assertFalse(stages[1]['level_known_during_stage'])
        self.assertTrue(np.isnan(stages[1]['zone_intersect_fraction']))
        self.assertTrue(stages[2]['level_known_during_stage'])
        a=la.event_alignment(self.e,self.b)
        pre=a[(a.milestone=='LEVEL_AVAILABLE')&(a.relative_min==-1)].iloc[0]
        at=a[(a.milestone=='LEVEL_AVAILABLE')&(a.relative_min==0)].iloc[0]
        self.assertFalse(pre.level_known)
        self.assertTrue(np.isnan(pre.signed_close_from_level_bp))
        self.assertTrue(at.level_known)
        self.assertTrue(np.isfinite(at.signed_close_from_level_bp))

    def test_future_bars_do_not_change_pre_signal_stages(self):
        orig=la.event_stages(self.e,self.b)
        b2=self.b.copy()
        b2.loc[b2.close_ms>self.e['second_touch_close_ms'],'close']=9000.
        got=la.event_stages(self.e,b2)
        for i in range(4):
            self.assertAlmostEqual(orig[i]['signed_price_change_bp'],got[i]['signed_price_change_bp'])
            self.assertAlmostEqual(orig[i]['aligned_aggressive_quote_fraction'],got[i]['aligned_aggressive_quote_fraction'])

    def test_future_bars_do_not_change_before_or_at_milestone(self):
        orig=la.event_alignment(self.e,self.b)
        changed=self.b.copy();key=self.e['first_touch_close_ms']
        changed.loc[changed.close_ms>key,'close']=1234.
        post=la.event_alignment(self.e,changed)
        mask=(orig.milestone=='FIRST_TOUCH')&(orig.relative_min<=0)
        self.assertTrue(np.allclose(orig.loc[mask,'signed_price_from_anchor_bp'],
                                    post.loc[mask,'signed_price_from_anchor_bp'],equal_nan=True))

    def test_post_exit_is_only_market_context(self):
        m=la.event_alignment(self.e,self.b)
        a=m[(m.milestone=='SECOND_TOUCH')&(m.relative_min==5)].iloc[0]
        b=m[(m.milestone=='SECOND_TOUCH')&(m.relative_min==12)].iloc[0]
        self.assertFalse(a.post_exit_market_only)
        self.assertTrue(b.post_exit_market_only)

    def test_missing_minute_does_not_forward_fill(self):
        b=self.b.drop(index=self.e['first_touch_close_ms']-la.MIN_MS)
        z=la.event_stages(self.e,b)
        self.assertFalse(z[2]['complete'])
        self.assertTrue(np.isnan(z[2]['signed_price_change_bp']))
        a=la.event_alignment(self.e,b)
        p=a[(a.milestone=='FIRST_TOUCH')&(a.relative_min==-1)].iloc[0]
        self.assertFalse(p.observed)

    def test_1min_aggressive_flow_alignment(self):
        z=la.event_stages(self.e,self.b)
        self.assertAlmostEqual(z[3]['aligned_aggressive_quote_fraction'],.1)
        e=self.e|{'side':'short'}
        self.assertAlmostEqual(la.event_stages(e,self.b)[3]['aligned_aggressive_quote_fraction'],-.1)

    def test_exact_boundary_window_excludes_start(self):
        x=la.slice_minutes(self.b,self.e['available_ms'],self.e['available_ms']+2*la.MIN_MS)
        self.assertEqual(len(x),2)
        self.assertEqual(x.index[0],self.e['available_ms']+la.MIN_MS)

    def test_milestone_alignment_cardinality(self):
        a=la.event_alignment(self.e,self.b)
        self.assertEqual(len(a),4*(15+30+1))
        self.assertEqual(a.milestone.nunique(),4)
        for _,z in a.groupby('milestone'):
            at=z[z.relative_min.eq(0)]
            self.assertEqual(len(at),1)
            self.assertAlmostEqual(float(at.signed_price_from_anchor_bp.iloc[0]),0)

    def test_event_phase_summary_has_separate_level_families(self):
        z=pd.DataFrame(la.event_stages(self.e,self.b))
        a=la.event_alignment(self.e,self.b)
        phases,paths=la.summary_frames(z,a)
        self.assertIn('ALL_LEVELS',set(phases.level_type))
        self.assertIn('pivot_5m',set(phases.level_type))
        self.assertEqual(len(paths.query("level_type=='ALL_LEVELS' and milestone=='SECOND_TOUCH'")),46)

    def test_prior_day_availability_rule(self):
        e=fake_event(self.b,level_type='prior_day')
        self.assertEqual(e['available_ms']-e['level_source_time_ms'],1440*la.MIN_MS)

    def test_zero_interval_phase_is_explicitly_missing(self):
        z=la.phase_segment(self.e,self.b,'TEST',self.e['available_ms'],self.e['available_ms'],True)
        self.assertFalse(z['complete'])
        self.assertEqual(z['expected_minute_bars'],0)

    def test_no_future_in_preavailable_stage(self):
        a=la.event_alignment(self.e,self.b)
        pre=a[(a.milestone=='LEVEL_AVAILABLE')&(a.relative_min<0)]
        self.assertFalse(pre.level_known.any())
        self.assertTrue(pre.zone_intersect.isna().all())

    def test_03c_status_never_treated_as_primary_outcome(self):
        x=la.event_stages(self.e,self.b)[3]
        self.assertEqual(x['eventual_a03b_outcome'],'TARGET')
        self.assertEqual(x['a03c_maker_touch_reason'],'NO_THIRD_TOUCH')


class TestCohort(unittest.TestCase):
    def fixture_repo(self,root:Path,bad=False):
        base=root/'notebooks'/'Liquidity_Absorption'
        for x in ('02','02b','03b','03c'):(base/x).mkdir(parents=True,exist_ok=True)
        (base/'02'/'02_feature_contract.json').write_text(json.dumps({'version':'LA-A02-0.1.0','config':{'pivot_right_5m':2}}))
        start=pd.Timestamp('2026-07-01 12:00:00Z')
        rows=[];fix=[];mkr=[]
        for i in range(452):
            signal=start+pd.Timedelta(minutes=i*5)
            ready=signal-pd.Timedelta(minutes=20)
            source=ready-pd.Timedelta(minutes=10)
            first=signal-pd.Timedelta(minutes=5)
            r={'a02_candidate_row':i,'symbol':'BTCUSDT' if i%2==0 else 'ETHUSDT',
               'side':'long' if i%2==0 else 'short','level_type':'pivot_5m',
               'level_source_time':str(source),'level_price':100.,'zone_halfwidth':.2,
               'first_touch_close':str(first),'second_touch_close':str(signal),'signal_time_utc':str(signal)}
            rows.append(r)
            fix.append({k:r[k] for k in ('a02_candidate_row','symbol','side','signal_time_utc','level_price')} |
                       {'mode':'FIXED','reason':'TARGET','target':101.,'initial_stop':99.,
                        'entry_time_ms':int(signal.value//1_000_000)+100,
                        'exit_time_ms':int(signal.value//1_000_000)+15*la.MIN_MS})
            mkr.append({'a02_candidate_row':i,'source_side':r['side'],'signal_time_utc':r['signal_time_utc'],
                        'mode':'MAKER_TOUCH_3RD','reason':'NO_ENTRY',
                        'observed_zone_touch':False,'entry_time_ms':np.nan})
        if bad:rows[0]['first_touch_close']=str(start-pd.Timedelta(minutes=28))
        pd.DataFrame(rows).to_csv(base/'02b'/'02b_touch_features.csv',index=False)
        pd.DataFrame(fix).to_csv(base/'03b'/'03b_1to1_events.csv',index=False)
        pd.DataFrame(mkr).to_csv(base/'03c'/'03c_events.csv',index=False)

    def test_merge_452_1to1_and_delay_from_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture_repo(root)
            a=la.load_cohort(root)
            self.assertEqual(len(a),452)
            self.assertEqual(a.minutes_source_to_available.median(),10)
            self.assertEqual(a.minutes_available_to_first.median(),15)
            self.assertEqual(a.minutes_first_to_second.median(),5)
            self.assertTrue(a.reason.eq('TARGET').all())
            self.assertTrue(a.a03c_maker_touch_reason.eq('NO_ENTRY').all())

    def test_never_use_level_before_becoming_available(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture_repo(root,bad=True)
            with self.assertRaisesRegex(ValueError,'Temporal causality violation'):
                la.load_cohort(root)

    def test_same_close_as_first_availability_is_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture_repo(root)
            p=root/'notebooks'/'Liquidity_Absorption'/'02b'/'02b_touch_features.csv'
            b=pd.read_csv(p)
            b.loc[0,'first_touch_close']=str(pd.Timestamp(b.loc[0,'level_source_time'])+pd.Timedelta(minutes=10))
            b.to_csv(p,index=False)
            q=la.load_cohort(root)
            self.assertTrue(bool(q.loc[0,'first_touch_confirms_at_availability']))
            self.assertFalse(bool(q.loc[0,'first_touch_strictly_after_availability']))
            bars=synthetic_bar_data()
            e=fake_event(bars)
            e['first_touch_close_ms']=e['available_ms']
            stages=la.event_stages(e,bars)
            self.assertFalse(stages[2]['complete'])
            self.assertEqual(stages[2]['expected_minute_bars'],0)

    def test_a03c_maker_identity_drift_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture_repo(root)
            p=root/'notebooks'/'Liquidity_Absorption'/'03c'/'03c_events.csv'
            m=pd.read_csv(p)
            m.loc[3,'source_side']='long' if m.loc[3,'source_side']=='short' else 'short'
            m.to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'maker source identity drift'):
                la.load_cohort(root)

    def test_reject_identity_drift(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture_repo(root)
            path=root/'notebooks'/'Liquidity_Absorption'/'03b'/'03b_1to1_events.csv'
            b=pd.read_csv(path);b.loc[1,'symbol']='BTCUSDT';b.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'symbol differs'):
                la.load_cohort(root)

    def test_end_to_end_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture_repo(root)
            a=la.load_cohort(root);bars=synthetic_bar_data('2026-07-01 00:00:00Z',n=3500)
            with patch.object(la,'load_cohort',return_value=a),patch.object(la,'load_minutes',return_value=bars):
                decision=la.run(root,root/'unused',root/'out')
            self.assertEqual(decision['original_A02_signals'],452)
            self.assertEqual(decision['stage_rows'],452*8)
            self.assertEqual(decision['minute_landmark_rows'],452*4*46)
            self.assertEqual(decision['stage_missing_or_empty'],0)
            for f in ('09_lifecycle_stages.csv','09_landmark_summary.csv','09_landmark_minute_panel.csv.gz',
                      '09_decision.json','09_milestone_price_paths.png','09_lifecycle_phase_changes.png'):
                self.assertTrue((root/'out'/f).is_file(),f)
            self.assertEqual(len(pd.read_csv(root/'out'/'09_lifecycle_stages.csv')),452*8)

    def test_preregistered_timestamp_availability(self):
        t=pd.Series(['2026-07-01T00:00:00Z','2026-07-01T00:01:00Z'])
        a=la.to_ms(t)
        self.assertEqual(a.iloc[1]-a.iloc[0],60_000)

    def test_jupyter_every_code_cell_compiles(self):
        path=Path(__file__).resolve().parents[1]/'notebooks'/'Liquidity_Absorption'/'09_level_lifecycle_anatomy.ipynb'
        nb=json.loads(path.read_text(encoding='utf-8'))
        for i,c in enumerate(nb['cells']):
            if c['cell_type']=='code':compile(''.join(c['source']),f'cell_{i}','exec')

if __name__=='__main__':unittest.main()
