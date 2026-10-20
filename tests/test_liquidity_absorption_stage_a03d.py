"""LA-A03d read-only frozen VD18 inference, stop sizing and causal origin tests."""
import json
import sys
import types
from dataclasses import dataclass
from unittest.mock import patch
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"/"data_engineering"/"modeling"))
from liquidity_absorption_stage_a03d import (RiskConfig, _check, build_spot_history,
    forecast_origins, load_vd18_fits, paired_old_comparison, predict_saved_har,
    reference_position_size, risk_stop, spot_history_directory, build_results, HORIZONS)

UTC="2026-07-01T00:00:00Z"


def fit(asset_names=("BTC", "ETH")):
    return {"design":{"features":["har_logrv_30","har_logrv_60","har_logrv_240","har_logrv_1440"],
                      "center":[-12,-11,-10,-9],"std":[1,2,3,4],"asset_names":list(asset_names)},
            "coefficients":[-12,1,.2,.3,.4,.1,-.2,.5],
            "net_return_scales":{"BTC":2.0,"ETH":3.0},"lookbacks":[30,60,240,1440]}


def write_spot(folder,asset="BTC",periods=315,omit=None,extra_after=False):
    t=pd.date_range(UTC,periods=periods,freq="5min")
    if omit is not None:t=t.delete(omit)
    p=100*np.exp(np.arange(len(t))*0.0001)
    frame=pd.DataFrame({"state_time":t,"close":p})
    if extra_after:
        frame.loc[len(frame)]={"state_time":t[-1]+pd.Timedelta(minutes=5),"close":9999999.}
    frame.to_csv(folder/f"{asset}.csv.gz",index=False,compression="gzip")
    return t


class TestRiskBridge(unittest.TestCase):
    def test_registered_horizons(self):
        self.assertEqual(HORIZONS,(15,60,240))
        _check(RiskConfig())
        with self.assertRaises(ValueError):_check(RiskConfig(sigma_stop_multiplier=0))
        with self.assertRaises(ValueError):_check(RiskConfig(hold_windows=(23,)))

    def test_saved_har_design_including_asset_dummy(self):
        f=fit()
        frame=pd.DataFrame({"har_logrv_30":[-11.],"har_logrv_60":[-11.],
                            "har_logrv_240":[-10.],"har_logrv_1440":[-9.],"slot":[72]})
        x=np.array([1,1,0,0,0,np.sin(2*np.pi*72/288),np.cos(2*np.pi*72/288),0])
        expected=np.exp(np.dot(x,f['coefficients']))*2
        self.assertAlmostEqual(predict_saved_har(f,frame,'BTC')[0],expected,places=12)
        y=x.copy();y[-1]=1
        self.assertAlmostEqual(predict_saved_har(f,frame,'ETH')[0],np.exp(np.dot(y,f['coefficients']))*3,places=12)

    def test_missing_model_asset_or_bad_feature_contract_fails(self):
        with self.assertRaises(ValueError):predict_saved_har(fit(),pd.DataFrame(),"SOL")
        f=fit();f['design']['features']=['bad']
        with self.assertRaises(ValueError):predict_saved_har(f,pd.DataFrame(),"BTC")

    def test_strict_completed_return_windows(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);write_spot(folder)
            z=build_spot_history(folder/'BTC.csv.gz')
            self.assertTrue(np.isnan(z.har_logrv_1440.iloc[0]))
            self.assertTrue(np.isfinite(z.har_logrv_1440.iloc[288]))
            self.assertEqual(z.slot.iloc[-1],((z.state_time.iloc[-1].hour*60+z.state_time.iloc[-1].minute)//5))

    def test_gap_invalidates_rolling_5m_history(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);write_spot(folder,omit=30)
            z=build_spot_history(folder/'BTC.csv.gz')
            self.assertTrue(np.isnan(z.return_5m.iloc[30]))
            self.assertTrue(np.isnan(z.har_logrv_1440.iloc[290]))

    def test_no_future_spot_changes_forecast_at_past_origin(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);write_spot(folder,extra_after=False)
            ts=pd.Timestamp(UTC)+pd.Timedelta(minutes=5*300+2)
            ev=pd.DataFrame([{"a02_candidate_row":5,"symbol":"BTCUSDT","signal_time_utc":str(ts)}])
            f={k:fit() for k in HORIZONS}
            x=forecast_origins(ev,folder,f)
            write_spot(folder,extra_after=True)
            y=forecast_origins(ev,folder,f)
            self.assertEqual(x.vd18_origin_utc.iloc[0],str(ts.floor('5min')))
            self.assertAlmostEqual(x.variance_60.iloc[0],y.variance_60.iloc[0])
            self.assertEqual(x.forecast_status.iloc[0],"PRESENT")

    def test_missing_last_spot_bar_does_not_forward_fill(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);t=write_spot(folder,omit=300)
            ts=t[300]+pd.Timedelta(minutes=2)
            ev=pd.DataFrame([{"a02_candidate_row":5,"symbol":"BTCUSDT","signal_time_utc":str(ts)}])
            x=forecast_origins(ev,folder,{k:fit() for k in HORIZONS})
            self.assertEqual(x.forecast_status.iloc[0],"NO_COMPLETE_CAUSAL_5M_HISTORY")

    def test_wick_and_zone_stops_long(self):
        cfg=RiskConfig()
        s=risk_stop("long",100.,99.8,.05,99.25,25.,.01,cfg)
        self.assertLess(s,99.25)
        self.assertAlmostEqual(s,99.24)

    def test_wick_and_zone_stops_short(self):
        s=risk_stop("short",100.,100.2,.05,100.75,25.,.01,RiskConfig())
        self.assertGreater(s,100.75)
        self.assertAlmostEqual(s,100.76)

    def test_vol_stop_binds_when_higher_risk(self):
        s=risk_stop("long",100,99.99,.001,99.98,300,.01,RiskConfig())
        self.assertAlmostEqual(s,97.)

    def test_stop_ignores_forecast_direction(self):
        c=RiskConfig()
        self.assertLess(risk_stop('long',100,99.99,.01,99.97,100,.01,c),100)
        self.assertGreater(risk_stop('short',100,100.01,.01,100.03,100,.01,c),100)

    def test_degenerate_stop_or_nan_rejected(self):
        with self.assertRaises(ValueError):risk_stop('long',1.,.01,.01,.01,100,.1,RiskConfig())
        with self.assertRaises(ValueError):risk_stop('long',1.,1.,.01,.99,float('nan'),.01,RiskConfig())

    def test_position_notional_inversely_scales_with_risk(self):
        c=RiskConfig(max_notional_usdt=100000)
        a=reference_position_size(100,99,c,False)
        b=reference_position_size(100,95,c,False)
        self.assertGreater(a['reference_notional_usdt'],b['reference_notional_usdt'])
        self.assertLessEqual(a['reference_notional_usdt']*(a['risk_distance_bp']+a['cost_allowance_bp'])/10000,25.000001)

    def test_maker_notional_higher_for_same_stop_due_cost_allowance(self):
        c=RiskConfig(max_notional_usdt=100000)
        self.assertGreater(reference_position_size(100,99,c,True)['reference_notional_usdt'],
                           reference_position_size(100,99,c,False)['reference_notional_usdt'])

    def test_saved_model_contract_disallows_wrong_version(self):
        with tempfile.TemporaryDirectory() as td:
            m=Path(td)/'notebooks'/'Volatility_Dynamics'/'18'/'18_manifest.json';m.parent.mkdir(parents=True)
            m.write_text(json.dumps({'parameters':{'version':'FAKE'}}))
            with self.assertRaises(ValueError):load_vd18_fits(Path(td))

    def test_paired_comparison_on_matching_closed_ids(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);d=r/'notebooks'/'Liquidity_Absorption'/'03';d.mkdir(parents=True)
            x=pd.DataFrame([{'a02_candidate_row':0,'symbol':'BTCUSDT','mode':'FIXED','reason':'TARGET',
                             'net_return_bp_assumed_costs':1.,'gross_return_bp':4.}]);x.to_csv(d/'03_execution_events.csv',index=False)
            new=pd.DataFrame([{'a02_candidate_row':0,'symbol':'BTCUSDT','model_mode':'VD18_VOL_FIXED',
                               'entry_kind':'TAKER_FIRST_POST_SIGNAL','hold_minutes':240,'reason':'TARGET',
                               'net_return_bp_assumed_costs':2.,'gross_return_bp':4.}]);p,s=paired_old_comparison(new,r)
            self.assertEqual(int(s.iloc[0].paired_closed),1)
            self.assertAlmostEqual(s.iloc[0].mean_vol_minus_original_bp,1.)

    def test_synthetic_full_pipeline_with_frozen_old_targets(self):
        # Validate source joins and result cardinality with simulated A03/A03c
        # interfaces, without historical files or regression refits.
        @dataclass
        class BaseConfig:
            hold_minutes: int = 240
        class State:
            def __init__(self,mode):
                self.mode=mode; self.stop=np.nan; self.initial_risk_price=np.nan
        class FakeTaker:
            def __init__(self,event,bars,cfg):
                self.event=event;self.cfg=cfg;self.symbol=event['symbol'];self.side=event['side']
                self.d=1 if self.side=='long' else -1
                self.level=float(event['level_price']);self.width=float(event['zone_halfwidth'])
                self.signal_low=float(event['bar_low']);self.signal_high=float(event['bar_high'])
                self.tick=.1;self.entry_ms=None;self.entry_print=np.nan;self.target=np.nan
                self.risk=np.nan;self.initial_stop=self.level-self.d*self.width
                self.states={m:State(m) for m in ('FIXED','BREAKEVEN','TRAIL_1M')}
            def _start(self,tm,px):
                self.entry_ms=tm;self.entry_print=px;self.target=101.;self.risk=abs(px-self.initial_stop)
            def records(self):
                out=[]
                for mode,st in self.states.items():
                    gross=(self.target-self.entry_print)*self.d
                    out.append({'a02_candidate_row':self.event['a02_candidate_row'],
                      'symbol':self.symbol,'mode':mode,'reason':'TARGET',
                      'entry_time_ms':self.entry_ms,'entry_print':self.entry_print,
                      'exit_time_ms':self.entry_ms+60_000,'initial_stop':self.initial_stop,
                      'initial_risk_bps':self.risk/self.entry_print*10000,
                      'target':self.target,'gross_return_bp':gross/self.entry_print*10000,
                      'net_return_bp_assumed_costs':(gross/self.entry_print*10000-14),
                      'net_r_assumed_costs':(gross-0.14)/self.risk})
                return out
        class FakeMaker:
            def __init__(self,source,mode,cfg):
                self.source=source;self.mode=mode;self.cfg=cfg
                self.symbol=source['symbol'];self.side=source['side'];self.d=1
                self.level=source['level_price'];self.target=source['target']
                self.stop=99.;self.reason='';self.entry_time=None;self.entry_px=np.nan
                self.__post_init__()
            def __post_init__(self):pass
            def _entry(self,tm,entry_price,print_price):pass
            def record(self):
                return {'a02_candidate_row':self.source['a02_candidate_row'],'symbol':self.symbol,
                 'mode':self.mode,'reason':'TARGET','entry_time_ms':self.entry_time,
                 'entry_price':self.entry_px,'exit_time_ms':self.entry_time+60_000,
                 'stop':self.stop,'target':self.target,'gross_return_bp':100.,
                 'net_return_bp_assumed_costs':91.,'net_r_assumed_costs':.91,
                 'initial_risk_bp':abs(self.entry_px-self.stop)/self.entry_px*10000}
        base03=types.ModuleType('src.data_engineering.modeling.liquidity_absorption_stage_a03')
        base03.Config=BaseConfig;base03.EventSimulation=FakeTaker
        base03.prepare_bars=lambda root,sym:pd.DataFrame()
        base03.tick_for=lambda symbol,cfg:.1
        def taker_scan(root, sims):
            for sim in sims:
                sim._start(2000,100.)
            return {'read_daily_parquet_files':1}
        base03.scan_event_tapes=taker_scan
        base03.load_candidates=lambda repo:None
        base03c=types.ModuleType('src.data_engineering.modeling.liquidity_absorption_stage_a03c')
        base03c.Config=BaseConfig;base03c.TapeStudy=FakeMaker
        def maker_scan(root,sims):
            for sim in sims:sim._entry(2000,sim.level,sim.level)
            return {'read_daily_parquet_files':1}
        base03c.scan_tapes=maker_scan
        base03c.validate_inputs=lambda repo:None
        sig='2026-07-02 00:10:00+00:00'
        src=pd.DataFrame([{'a02_candidate_row':i,'symbol':'BTCUSDT','side':'long',
                           'level_price':100.,'zone_halfwidth':.05,'bar_low':99.9,'bar_high':100.4,
                           'signal_time_utc':sig,'bar_close':100.1,'target':101.,
                           'entry_print':100.,'entry_time_ms':2000} for i in range(452)])
        cfg=RiskConfig()
        forecast=src[['a02_candidate_row','symbol','signal_time_utc']].copy()
        forecast['vd18_origin_utc']='2026-07-02 00:10:00+00:00'
        forecast['forecast_status']='PRESENT'
        for h in HORIZONS:
            forecast[f'variance_{h}']=1e-4
            forecast[f'sigma_bp_{h}']=100.
        import liquidity_absorption_stage_a03d as module
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with patch.dict(sys.modules,{base03.__name__:base03,base03c.__name__:base03c}), \
                 patch.object(module,'_read_event_inputs',return_value=(src,src)), \
                 patch.object(module,'load_vd18_fits',return_value=({},{'model':'VD18'})), \
                 patch.object(module,'forecast_origins',return_value=forecast), \
                 patch.object(module,'plot_diagnostics',return_value=None):
                result=build_results(root,root,root,root/'03d',cfg)
            self.assertEqual(result['n_events'],452)
            self.assertEqual(result['n_forecasts'],452)
            self.assertEqual(result['n_taker_sims'],904)
            self.assertEqual(result['n_maker_sims'],1808)
            self.assertEqual(result['n_rows'],4520)
            self.assertTrue((root/'03d'/'03d_summary.csv').exists())
            out=pd.read_csv(root/'03d'/'03d_risk_execution.csv')
            self.assertTrue((out[out.entry_kind=='TAKER_FIRST_POST_SIGNAL'].target==101).all())
            self.assertTrue((out.initial_risk_bp>0).all())

if __name__=='__main__':unittest.main()
