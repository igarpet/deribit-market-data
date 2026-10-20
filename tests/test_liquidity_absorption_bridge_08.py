"""LA-08 contract tests: synthetic replay only; no historical edge claim."""
from __future__ import annotations
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data_engineering.modeling import liquidity_absorption_bridge_08 as a


def fixtures(n=16):
    dates=pd.date_range('2024-02-01',periods=n,freq='120D',tz='UTC')
    symbols=['BTCUSDT','ETHUSDT']*(n//2)+(['BTCUSDT'] if n%2 else [])
    sides=['long','short']*(n//2)+(['long'] if n%2 else [])
    b=[];s=[];v=[]
    reasons=['TARGET','STOP','TIMEOUT','NO_STRUCTURAL_TARGET']
    for i in range(n):
        sym,side=symbols[i],sides[i]
        entry=60000.0 if sym=='BTCUSDT' else 3000.0
        sign=1 if side=='long' else -1
        target=entry+sign*100.0
        stop=entry-sign*100.0
        r=reasons[i%4]
        common=dict(a02_candidate_row=i,symbol=sym,side=side,signal_time_utc=dates[i].isoformat())
        b.append({**common,'reason':r,'entry_print':entry if r in a.CLOSED else np.nan,
            'target':target,'initial_stop':stop,'target_rr_initial':1.,
            'entry_time_ms':i*100000+1 if r in a.CLOSED else np.nan,
            'exit_time_ms':i*100000+2000 if r in a.CLOSED else np.nan,
            'net_return_bp_assumed_costs':[-9.,-12.,-3.,np.nan][i%4],
            'gross_return_bp':[2.,-3.,4.,np.nan][i%4],
            'entry_fill_assumed':entry,'exit_fill_assumed':entry,
            'entry_fee_bp':6.,'exit_fee_bp':6.})
        s.append({**common,'cvd_ok':i%2==0,'cvd_flow': (-1)**i*100.,
                'v2_target_source':a.TARGET_OK if i%3!=1 else 'NO_PROFILE_TARGET',
                'v2_target':target+sign*10})
        v.append({**common,'v2_variant':'TRAIL_1M','reason':r if i%5!=0 else 'NO_PROFILE_TARGET',
                'entry_print':entry if i%5 else np.nan,'target':target+sign*10,
                'initial_stop':stop,'gross_return_bp':[-1.,-2.,3.,np.nan][i%4],
                'net_return_bp_assumed_costs':[-10.,-12.,-4.,np.nan][i%4],
                'entry_fill_assumed':entry,'exit_fill_assumed':entry,
                'entry_fee_bp':6.,'exit_fee_bp':6.})
    return pd.DataFrame(b),pd.DataFrame(s),pd.DataFrame(v)


class TestBridgeContract(unittest.TestCase):
    def test_frozen_geometry_and_identity(self):
        x,y,z=fixtures();a1,b1=a.ensure_invariants(x,y,z)
        self.assertEqual(len(a1),16);self.assertEqual(len(b1),16)
        self.assertAlmostEqual(a1.loc[0,'target'],x.loc[0,'target'])
        self.assertAlmostEqual(a1.loc[0,'initial_stop'],x.loc[0,'initial_stop'])

    def test_reject_duplicate_signal(self):
        b,s,v=fixtures();s.loc[1,'a02_candidate_row']=0
        with self.assertRaisesRegex(ValueError,'duplicated'):a.ensure_invariants(b,s,v)

    def test_reject_unmatched_signal_universe(self):
        b,s,v=fixtures();s=s.iloc[:-1]
        with self.assertRaisesRegex(ValueError,'universe mismatch'):a.ensure_invariants(b,s,v)

    def test_reject_wrong_timestamps(self):
        b,s,v=fixtures();s.loc[0,'signal_time_utc']='2024-02-02T00:00:00Z'
        with self.assertRaisesRegex(ValueError,'signal-time drift'):a.ensure_invariants(b,s,v)

    def test_reject_side_mismatch(self):
        b,s,v=fixtures();s.loc[0,'side']='short'
        with self.assertRaisesRegex(ValueError,'Signal identity drift'):a.ensure_invariants(b,s,v)

    def test_reject_widened_stop(self):
        b,s,v=fixtures();b.loc[0,'initial_stop']=b.loc[0,'entry_print']-300
        with self.assertRaisesRegex(ValueError,'1:1 invariant'):a.ensure_invariants(b,s,v)

    def test_reject_malformed_cvd_label(self):
        b,s,v=fixtures();s['cvd_ok']=s['cvd_ok'].astype(object);s.loc[0,'cvd_ok']='?' 
        with self.assertRaisesRegex(ValueError,'CVD eligibility'):a.ensure_invariants(b,s,v)

    def test_v2_only_no_cut_trail(self):
        b,s,v=fixtures();extra=v.iloc[:2].copy();extra.v2_variant='CUT15_TRAIL_1M'
        a1,b1=a.ensure_invariants(b,s,pd.concat([v,extra]))
        self.assertEqual(len(b1),16)

    def test_reject_comparator_duplicate(self):
        b,s,v=fixtures();v=pd.concat([v,v.iloc[[0]]],ignore_index=True)
        with self.assertRaisesRegex(ValueError,'duplicated'):a.ensure_invariants(b,s,v)

    def test_reject_comparator_id_drift(self):
        b,s,v=fixtures();v.loc[0,'side']='short'
        with self.assertRaisesRegex(ValueError,'comparator differs'):a.ensure_invariants(b,s,v)

    def test_overlay_preserves_frozen_outcomes_and_filters_by_signal(self):
        b,s,v=fixtures();p,c=a.ensure_invariants(b,s,v)
        with patch.object(a,'cost_reprice',side_effect=lambda df:df.net_return_bp_assumed_costs.astype(float)+2):
            out=a.prepare_rows(p,c)
        self.assertEqual(len(out),80)
        self.assertEqual(len(out.policy.unique()),5)
        self.assertEqual(out[out.policy=='A03B_ALL'].policy_eligible.sum(),16)
        self.assertEqual(out[out.policy=='A03B_CVD_ONLY'].policy_eligible.sum(),8)
        both=out[out.policy=='A03B_CVD_AND_PROFILE']
        self.assertEqual(int(both.policy_eligible.sum()),len(s[s.cvd_ok & s.v2_target_source.eq(a.TARGET_OK)]))
        self.assertEqual(out[(out.policy=='A03B_ALL')&(out.a02_candidate_row==0)].policy_outcome.iloc[0],'TARGET')
        self.assertTrue((out[~out.policy_eligible].policy_outcome=='FILTERED_OUT').all())

    def test_v2_is_different_payoff_replay(self):
        b,s,v=fixtures();p,c=a.ensure_invariants(b,s,v)
        with patch.object(a,'cost_reprice',side_effect=lambda df:df.net_return_bp_assumed_costs.astype(float)+2):
            x=a.prepare_rows(p,c)
        q=x[(x.policy=='CLAUDE_V2_CVD_PROFILE_VOL_TRAIL')&(x.a02_candidate_row==2)]
        r=x[(x.policy=='A03B_ALL')&(x.a02_candidate_row==2)]
        self.assertTrue(q.policy_eligible.iloc[0]);self.assertAlmostEqual(q.policy_realized_net_bp.iloc[0],-2)
        self.assertAlmostEqual(r.policy_realized_net_bp.iloc[0],-1)

    def test_filtered_and_unresolved_have_zero_realized_contribution_not_closed(self):
        b,s,v=fixtures();p,c=a.ensure_invariants(b,s,v)
        with patch.object(a,'cost_reprice',side_effect=lambda df:df.net_return_bp_assumed_costs.astype(float)+2):
            x=a.prepare_rows(p,c)
        self.assertTrue((x.loc[~x.policy_closed,'policy_realized_net_bp']==0).all())
        self.assertTrue((x.loc[x.policy_closed,'policy_realized_net_bp'].notna()).all())
        self.assertFalse(x.loc[x.a02_candidate_row==3,'policy_closed'].any())

    def test_period_boundaries(self):
        d=pd.Series(['2025-03-31T23:59:59Z','2025-04-01T00:00:00Z',
                     '2026-07-01T00:00:00Z','2026-09-30T23:59:59Z','2026-10-01T00:00:00Z'])
        self.assertEqual(a.classify_period(d).tolist(),['H1','H2','INSAMPLE_JUL_SEP_2026',
                    'INSAMPLE_JUL_SEP_2026','H2'])

    def test_bootstrap_day_cluster_deterministic(self):
        d=pd.Series(pd.date_range('2024-01-01',periods=12,freq='D',tz='UTC'))
        x=pd.Series(np.arange(12,dtype=float))
        l=a.day_bootstrap(x,d,a.Spec(bootstrap_iterations=50))
        self.assertEqual(l,a.day_bootstrap(x,d,a.Spec(bootstrap_iterations=50)))
        self.assertAlmostEqual(l[0],5.5)
        self.assertLessEqual(l[1],l[0])

    def test_paired_delta_all_reference_zero(self):
        b,s,v=fixtures(48);p,c=a.ensure_invariants(b,s,v)
        with patch.object(a,'cost_reprice',side_effect=lambda df:df.net_return_bp_assumed_costs.astype(float)+2):
            x=a.prepare_rows(p,c)
        self.assertEqual(len(x),48*5)
        o=a.paired_selection_effect(x,a.Spec(bootstrap_iterations=60))
        self.assertGreater(len(o),0)
        self.assertTrue((o.n_positive+o.n_negative+o.n_unchanged==o.n_resolved_paired_signals).all())

    def test_paired_comparator_censorship_is_excluded_not_assumed_zero(self):
        b,s,v=fixtures(48)
        # Baseline closed, V2 unresolved: pairing must not count V2 as 0 P&L.
        v.loc[0, 'reason']='CENSORED_TAPE_END'
        p,c=a.ensure_invariants(b,s,v)
        with patch.object(a,'cost_reprice',side_effect=lambda df:df.net_return_bp_assumed_costs.astype(float)+2):
            x=a.prepare_rows(p,c)
        out=a.paired_selection_effect(x,a.Spec(bootstrap_iterations=30))
        row=out[(out.policy=='CLAUDE_V2_CVD_PROFILE_VOL_TRAIL')&
                (out.symbol=='ALL')&(out.period=='HOLDOUT_INSPECTED')].iloc[0]
        self.assertGreaterEqual(row.n_excluded_censored,1)
        self.assertEqual(row.n_resolved_paired_signals+row.n_excluded_censored,
            int((a.classify_period(b.signal_time_utc)!='INSAMPLE_JUL_SEP_2026').sum()))

    def test_summary_accounting_and_censor_policy(self):
        b,s,v=fixtures(48);p,c=a.ensure_invariants(b,s,v)
        with patch.object(a,'cost_reprice',side_effect=lambda df:df.net_return_bp_assumed_costs.astype(float)+2):
            x=a.prepare_rows(p,c)
        sm=a.summarize(x,a.Spec(bootstrap_iterations=30))
        self.assertFalse(sm.empty)
        self.assertTrue((sm.n_closed<=sm.n_eligible).all())
        self.assertTrue((sm.n_target+sm.n_stop+sm.n_timeout==sm.n_closed).all())

    def test_run_end_to_end_on_fake_files(self):
        b,s,v=fixtures(48)
        with tempfile.TemporaryDirectory() as td:
            repo=Path(td);root=repo/'notebooks'/'Liquidity_Absorption'
            for key,df in [('base',b),('signals',s),('v2',v)]:
                f=root/a.INPUTS[key];f.parent.mkdir(parents=True,exist_ok=True);df.to_csv(f,index=False)
            with patch.object(a,'cost_reprice',side_effect=lambda df:df.net_return_bp_assumed_costs.astype(float)+2):
                d=a.run(repo,repo/'out',a.Spec(bootstrap_iterations=20))
            self.assertEqual(d['original_signals'],48)
            for name in ['08_policy_candidate_panel.csv','08_policy_comparison.csv',
                         '08_paired_selection_effects.csv','08_decision.json','08_comparison_net_bp.png']:
                self.assertTrue((repo/'out'/name).exists(),name)
            self.assertEqual(len(pd.read_csv(repo/'out'/'08_policy_candidate_panel.csv')),240)

    def test_notebook_cells_compile(self):
        f=ROOT/'notebooks'/'Liquidity_Absorption'/'08_original_strategy_claude_method_bridge.ipynb'
        n=json.loads(f.read_text(encoding='utf-8'))
        code=[x for x in n['cells'] if x['cell_type']=='code']
        self.assertGreaterEqual(len(code),4)
        for i,c in enumerate(code):
            source=''.join(c['source'])
            compile(source,f'notebook_cell_{i}','exec')
            self.assertEqual(c.get('outputs',[]),[])

if __name__=='__main__':unittest.main()
