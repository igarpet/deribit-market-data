"""Synthetic cross-venue millisecond test; no downloading, Parquet dependency or outcomes-based strategy fitting."""
import sys,unittest,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'/'data_engineering'/'modeling'))
import liquidity_absorption_stage_a05b as m

CENTER=1782864060000

def event(anchor='SIGNAL',eid=1):
    return {'a02_candidate_row':eid,'anchor':anchor,'anchor_ms':CENTER,
            'symbol':'BTCUSDT','deribit_instrument':'BTC-PERPETUAL','side':'long',
            'eventual_outcome':'TARGET','target_to_sigma60':.8}

def ticks(times=None,amount=None,directions=None,seq=None):
    times=times if times is not None else [CENTER-10,CENTER-1,CENTER,CENTER,CENTER+1,CENTER+11]
    n=len(times)
    return pd.DataFrame({'timestamp':pd.to_datetime(times,unit='ms',utc=True).as_unit('ms'),
      'instrument_name':['BTC-PERPETUAL']*n,'trade_id':[f'b{i}' for i in range(n)],
      'trade_seq':seq if seq is not None else list(range(1,n+1)),
      'price':list(range(100,100+n)),'amount':amount or [100]*n,
      'direction':directions or ['sell','sell','buy','buy','buy','sell'][:n]})

class TestMicro(unittest.TestCase):
    def test_ms_timestamp_conversions(self):
        z=pd.Series(pd.to_datetime([CENTER,CENTER+1],unit='ms',utc=True))
        for unit in ('ms','us','ns'):
            self.assertEqual(m.ms(z.dt.as_unit(unit)).tolist(),[CENTER,CENTER+1])

    def test_filter_only_perpetual_and_time_window(self):
        a=ticks();b=a.copy();b['instrument_name']='BTC-25DEC26'
        s=pd.concat([a,b],ignore_index=True)
        out=m.filter_trade_batch(s,'BTC-PERPETUAL',[event()])
        self.assertEqual(len(out[(1,'SIGNAL')]),len(a))

    def test_no_forward_future_trade_reference(self):
        a=ticks([CENTER+1,CENTER+2],directions=['buy','sell'])
        got=m.filter_trade_batch(a,'BTC-PERPETUAL',[event()])[(1,'SIGNAL')]
        b,p,q=m.summarize_one(event(),got)
        self.assertEqual(q[0]['status'],'NO_RECENT_PRE_ANCHOR_PRINT')
        self.assertTrue(b.price_signed_bp_vs_anchor.isna().all())

    def test_anchor_preceding_not_future(self):
        a=ticks();got=m.filter_trade_batch(a,'BTC-PERPETUAL',[event()])[(1,'SIGNAL')]
        b,p,q=m.summarize_one(event(),got)
        self.assertEqual(q[0]['anchor_price_reference'],103.)
        self.assertEqual(q[0]['anchor_ref_staleness_ms'],0)
        self.assertEqual(q[0]['max_trades_one_ms'],2)
        self.assertEqual(q[0]['status'],'PRESENT')

    def test_only_nonempty_ms_buckets_are_emitted(self):
        a=ticks();raw=m.filter_trade_batch(a,'BTC-PERPETUAL',[event()])[(1,'SIGNAL')]
        b,_,_=m.summarize_one(event(),raw)
        z=b[b.bin_ms==1]
        self.assertEqual(len(z),5)
        self.assertTrue((z.trade_count>=1).all())
        self.assertEqual(z.trade_count.sum(),6)
        self.assertEqual(z.loc[z.bucket_relative_ms==0,'trade_count'].iloc[0],2)

    def test_duplicate_trade_id_is_fatal_not_silently_dropped(self):
        e=event()
        tr=ticks()
        tr.loc[1,'trade_id']=tr.loc[0,'trade_id']
        filtered=m.filter_trade_batch(tr,'BTC-PERPETUAL',[e])[(1,'SIGNAL')]
        with self.assertRaisesRegex(ValueError, 'Duplicate or null Deribit trade_id'):
            m.summarize_one(e,filtered)

    def test_same_millisecond_distinct_ids_remain_valid(self):
        e=event()
        tr=ticks()
        filtered=m.filter_trade_batch(tr,'BTC-PERPETUAL',[e])[(1,'SIGNAL')]
        buckets,_,_=m.summarize_one(e,filtered)
        self.assertEqual(int(buckets[buckets.bin_ms==1].trade_count.sum()),len(tr))

    def test_direction_sign_short(self):
        a=ticks();raw=m.filter_trade_batch(a,'BTC-PERPETUAL',[event()])[(1,'SIGNAL')]
        b,p,q=m.summarize_one(event()|{'side':'short'},raw)
        z=b[(b.bin_ms==1)&(b.bucket_relative_ms==1)].iloc[0]
        self.assertLess(z.price_signed_bp_vs_anchor,0)

    def test_no_synthetic_millisecond_ordering_assumption(self):
        a=ticks([CENTER,CENTER],directions=['sell','buy'],seq=[12,11]);a['price']=[101.,100.]
        raw=m.filter_trade_batch(a,'BTC-PERPETUAL',[event()])[(1,'SIGNAL')]
        b,p,q=m.summarize_one(event(),raw)
        z=b[(b.bin_ms==1)&(b.bucket_relative_ms==0)].iloc[0]
        self.assertEqual(z.open_price,100.)
        self.assertEqual(z.close_price,101.)
        self.assertEqual(z.trade_count,2)

    def test_day_boundary_requests_both_days(self):
        e=event();e['anchor_ms']=int(pd.Timestamp('2026-07-02 00:00:00Z').timestamp()*1000)
        out=m.needed_days(pd.DataFrame([e]))
        self.assertEqual(set(out),{('BTC','2026-07-01'),('BTC','2026-07-02')})

    def test_missing_day_no_lookahead_imputation(self):
        a=event();days=[]
        windows,stat=m.capture_windows(Path('/nonexistent'),pd.DataFrame([a]),batch_source=lambda _:None)
        self.assertEqual(stat.status.iloc[0],'MISSING_DAY')
        self.assertTrue(windows[0][1].empty)

    def test_same_id_multiple_anchors_kept_separate(self):
        es=pd.DataFrame([event('FIRST_TOUCH'),event('SIGNAL')])
        f=lambda p:[ticks()]
        out,stat=m.capture_windows(Path('ignored'),es,batch_source=f)
        self.assertEqual(len(out),2)
        self.assertEqual({k[1] for k,_ in out},{'SIGNAL','FIRST_TOUCH'})

    def test_future_ticks_dont_change_earlier_phase(self):
        a=ticks();x=m.filter_trade_batch(a,'BTC-PERPETUAL',[event()])[(1,'SIGNAL')]
        _,p,_=m.summarize_one(event(),x)
        extra=ticks([CENTER+50_000],directions=['buy']);extra['price']=1e10;extra['trade_id']='later_unique_trade'
        y=m.filter_trade_batch(pd.concat([a,extra]),'BTC-PERPETUAL',[event()])[(1,'SIGNAL')]
        _,q,_=m.summarize_one(event(),y)
        pre=lambda z:z[(z.phase_s_start==-1)&(z.phase_s_end==0)].iloc[0]
        self.assertEqual(pre(p).signed_price_change_bp,pre(q).signed_price_change_bp)

    def test_sparse_buckets_all_resolutions(self):
        a=ticks();raw=m.filter_trade_batch(a,'BTC-PERPETUAL',[event()])[(1,'SIGNAL')]
        b,_,_=m.summarize_one(event(),raw)
        self.assertEqual(set(b.bin_ms),{1,10,100,1000})

    def test_missing_nearby_prints_staleness_warning(self):
        a=ticks([CENTER-45000,CENTER+1],directions=['buy','buy'])
        raw=m.filter_trade_batch(a,'BTC-PERPETUAL',[event()])[(1,'SIGNAL')]
        _,_,q=m.summarize_one(event(),raw)
        self.assertEqual(q[0]['status'],'NO_RECENT_PRE_ANCHOR_PRINT')

    def test_same_day_utc_day_capture(self):
        es=pd.DataFrame([event('SIGNAL')]);captured,stat=m.capture_windows(Path('.'),es,batch_source=lambda _: [ticks()])
        self.assertEqual(len(captured),1)
        self.assertEqual(stat.status.tolist(),['SCANNED'])
        self.assertEqual(len(captured[0][1]),6)


# Additional pipeline coverage: real output contracts with fake in-memory archive batches.
class TestPipeline(unittest.TestCase):
    def test_end_to_end_fake_archive_writes_sparse_file_and_qc(self):
        from unittest.mock import patch
        es=[]
        for i,label in enumerate(('TARGET','STOP','TIMEOUT')):
            for anchor in ('FIRST_TOUCH','SIGNAL'):
                z=event(anchor,i)
                z['eventual_outcome']=label
                es.append(z)
        anchors=pd.DataFrame(es)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'05b'
            with patch.object(m,'load_anchors',return_value=anchors):
                result=m.build_results(Path(td),Path(td),out,batch_source=lambda path:[ticks()])
            self.assertEqual(result['signal_events'],3)
            self.assertEqual(result['anchors'],6)
            self.assertGreater(result['captured_trade_buckets'],0)
            self.assertEqual(result['anchor_reference_present'],6)
            self.assertTrue((out/'05b_sparse_trade_buckets.csv.gz').exists())
            b=pd.read_csv(out/'05b_sparse_trade_buckets.csv.gz')
            self.assertEqual(set(b.bin_ms),{1,10,100,1000})
            self.assertTrue((b.trade_count>0).all())
            self.assertTrue((out/'05b_signal_sparse_1ms_examples.png').exists())

    def test_window_context_does_not_reuse_invalid_maker_or_venue_level(self):
        a=event(); raw=m.filter_trade_batch(ticks(),'BTC-PERPETUAL',[a])[(1,'SIGNAL')]
        b,phase,summary=m.summarize_one(a,raw)
        self.assertNotIn('level_price',b.columns)
        self.assertNotIn('binance_entry_price',b.columns)
        self.assertFalse(phase.signed_price_change_bp.notna().all())


class TestRegressionContracts(unittest.TestCase):
    def test_notebook_every_code_cell_compiles(self):
        import json
        f=ROOT/'notebooks'/'Liquidity_Absorption'/'05b_deribit_millisecond_microstructure.ipynb'
        nb=json.loads(f.read_text(encoding='utf-8'))
        for index,cell in enumerate(nb['cells']):
            if cell['cell_type']=='code':
                with self.subTest(cell=index):
                    compile(''.join(cell['source']),f'{f.name}:cell-{index}','exec')

    def test_no_trades_summary_can_be_read_as_csv(self):
        from unittest.mock import patch
        anchor=pd.DataFrame([event()])
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'05b'
            with patch.object(m,'load_anchors',return_value=anchor):
                outcome=m.build_results(Path(td),Path(td),target,batch_source=lambda path:[])
            self.assertEqual(outcome['anchor_rows'],1)
            group=pd.read_csv(target/'05b_phase_summary.csv')
            raw=pd.read_csv(target/'05b_event_phases.csv')
            self.assertEqual(len(group),0)
            self.assertIn('feasibility_band',group.columns)
            self.assertEqual(len(raw),0)
            self.assertIn('complete_calendar_days',raw.columns)

    def test_stale_anchor_is_not_included_in_reaction_group(self):
        from unittest.mock import patch
        e=event()
        # Prints only after t=0 -> no causal pre-anchor mark/valid price base.
        forward=ticks([CENTER+10,CENTER+20],directions=['buy','sell'])
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'05b'
            with patch.object(m,'load_anchors',return_value=pd.DataFrame([e])):
                m.build_results(Path(td),Path(td),target,batch_source=lambda path:[forward])
            quality=pd.read_csv(target/'05b_anchor_coverage.csv')
            grouped=pd.read_csv(target/'05b_phase_summary.csv')
            self.assertEqual(quality.status.iloc[0],'NO_RECENT_PRE_ANCHOR_PRINT')
            self.assertEqual(len(grouped),0)

    def test_phase_summary_contains_near_far_and_all_groups(self):
        from unittest.mock import patch
        anchors=[]
        for i,ratio in enumerate((.5,2.0)):
            z=event('SIGNAL',i)
            z['target_to_sigma60']=ratio
            anchors.append(z)
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'05b'
            with patch.object(m,'load_anchors',return_value=pd.DataFrame(anchors)):
                m.build_results(Path(td),Path(td),target,batch_source=lambda path:[ticks()])
            g=pd.read_csv(target/'05b_phase_summary.csv')
            self.assertEqual(set(g.feasibility_band),{
                'TARGET_LE_1SIGMA60','TARGET_GT_1SIGMA60','ALL'})
            self.assertTrue((g.n_events==1).any())
            self.assertEqual(g[g.feasibility_band=='ALL'].n_events.max(),2)

    def test_missing_adjacent_day_excludes_event_from_summary(self):
        from unittest.mock import patch
        a=event('SIGNAL')
        a['anchor_ms']=int(pd.Timestamp('2026-07-02T00:00:00Z').timestamp()*1000)
        f=lambda path: [ticks([a['anchor_ms']-500,a['anchor_ms']+500])] if path.stem=='2026-07-02' else None
        with tempfile.TemporaryDirectory() as td:
            dst=Path(td)/'05b'
            with patch.object(m,'load_anchors',return_value=pd.DataFrame([a])):
                m.build_results(Path(td),Path(td),dst,batch_source=f)
            # In a synthetic source, a missing day is observed in day status;
            # a file-existence precheck must not mistakenly bless completeness.
            status=pd.read_csv(dst/'05b_day_scan_status.csv')
            self.assertIn('MISSING_DAY',set(status.status))
            coverage=pd.read_csv(dst/'05b_anchor_coverage.csv')
            self.assertEqual(coverage.status.iloc[0],'INCOMPLETE_CALENDAR_DAYS')
            summary=pd.read_csv(dst/'05b_phase_summary.csv')
            self.assertTrue(summary.empty)

if __name__=='__main__':
    unittest.main()
