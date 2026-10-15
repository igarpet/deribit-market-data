"""CS04-A causal feature, target, benchmark and scan regression tests."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import pytest
from data_engineering.modeling import candle_streak_feature_matrix as fm


def simulated_inputs(days=3,seed=2606,start='2025-12-01T00:00:00Z'):
    rng=np.random.default_rng(seed)
    n=days*288
    minute_time=pd.date_range(start,periods=n*5,freq='1min',tz='UTC')
    flow=rng.uniform(.05,.95,n*5)
    vol=rng.uniform(100,900,n*5)
    rets=rng.normal(0,.00025,n*5)
    pr=200*np.exp(np.cumsum(rets))
    startprice=pr/np.exp(rets)
    minute=pd.DataFrame({'open_time':minute_time,'open':startprice,'close':pr,
           'high':np.maximum(startprice,pr)*1.0001,
           'low':np.minimum(startprice,pr)*.9999,
           'quote_volume':vol,'taker_buy_quote_volume':vol*flow,
           'trades':rng.integers(2,100,n*5)})
    f=fm.minute_to_five(minute)
    st=f.state_time
    sr=np.log(f.f_close.to_numpy()/f.f_close.shift().to_numpy())
    sr[0]=np.nan
    sr[3::97]=0.
    spot=pd.DataFrame({'state_time':st,'return_5m':sr})
    close=pd.DataFrame({'state_time':st,'close':f.f_close.to_numpy()})
    return minute,spot,close,f


def test_count_and_unique_feature_groups():
    c=fm.combinations_frozen()
    assert len(fm.FAMILY_NAMES)==10
    assert len(c)==175
    assert len(set(c))==175
    assert len([x for x in c if len(x)==1])==10
    assert len([x for x in c if len(x)==2])==45
    assert len([x for x in c if len(x)==3])==120
    features=[c for cols in fm.FAMILY_COLUMNS.values() for c in cols]
    assert len(features)==len(set(features))


def test_aggregation_uses_last_minute_but_not_next_bucket():
    minute,spot,close,f=simulated_inputs()
    before=f.iloc[0].copy()
    minute.loc[5,'taker_buy_quote_volume']=0
    after=fm.minute_to_five(minute)
    for col in ['flow_5m','flow_late','minute_rv','candle_body']:
        assert before[col]==pytest.approx(after.iloc[0][col])
    assert f.iloc[1].flow_5m!=after.iloc[1].flow_5m
    assert after.iloc[0].state_time==pd.Timestamp('2025-12-01 00:05:00Z')


def test_missing_minute_invalidates_bucket_and_no_fill():
    minute,*_=simulated_inputs()
    a=fm.minute_to_five(minute.drop(index=3))
    assert not bool(a.iloc[0].flow_valid)
    assert pd.isna(a.iloc[0].flow_late)
    assert bool(a.iloc[1].flow_valid)


def test_bad_ohlc_and_duplicate_minute_fail():
    m,*_=simulated_inputs()
    with pytest.raises(ValueError,match='bad minute timestamps'):
        fm.minute_to_five(pd.concat([m,m.iloc[[1]]]))
    m.loc[0,'low']=m.loc[0,'high']*1.01
    with pytest.raises(ValueError,match='OHLC inconsistent'):
        fm.minute_to_five(m)


def test_features_do_not_use_future_prices_or_flow():
    minute,spot,close,f=simulated_inputs(days=4)
    peer=f.copy()
    x=fm.attach_features(spot,f,close,peer,fast_bars=2,slow_bars=3)
    cols=[col for family in fm.FAMILY_NAMES for col in fm.FAMILY_COLUMNS[family]]+list(fm.BASE)
    # A future price candle cannot modify earlier causal features.
    q=spot.copy();q.loc[30,'return_5m']*=15
    p=fm.attach_features(q,f,close,peer,fast_bars=2,slow_bars=3)
    pd.testing.assert_frame_equal(x.loc[:29,cols],p.loc[:29,cols])
    assert x.loc[30,'mom_r5']!=p.loc[30,'mom_r5']
    assert np.isfinite(x.loc[20,cols].to_numpy(float)).all()


def test_future_labels_are_net_log_return_and_censored_day_boundary():
    _,spot,close,f=simulated_inputs(days=3)
    x=fm.attach_targets(fm.attach_features(spot,f,close,f,fast_bars=2,slow_bars=3))
    t=10
    for h in fm.HORIZONS:
        expected=spot.return_5m.iloc[t+1:t+1+h//5].sum()
        assert x.loc[t,f'ret_{h}m']==pytest.approx(expected)
        assert x.loc[t,f'abs_{h}m']==pytest.approx(abs(expected))
        assert x.loc[t,f'sq_{h}m']==pytest.approx(expected**2)
        assert x.loc[t,f'upside_{h}m']==pytest.approx(max(expected,0))
        assert x.loc[t,f'downside_{h}m']==pytest.approx(max(-expected,0))
        assert x.loc[t,f'path_rv_{h}m']==pytest.approx((spot.return_5m.iloc[t+1:t+1+h//5]**2).sum())
        assert x.loc[t,f'path_abs_sum_{h}m']==pytest.approx(spot.return_5m.iloc[t+1:t+1+h//5].abs().sum())
    day0=spot.state_time.iloc[1].floor('D')
    last=x.loc[x.state_time.dt.floor('D').eq(day0)].index.max()
    assert np.isnan(x.loc[last,'ret_5m'])
    assert np.isnan(x.loc[last-1,'ret_30m'])
    assert x.loc[last-6,'all_targets_ready']


def test_gap_invalidates_all_forward_horizons():
    _,sp,close,f=simulated_inputs(days=3)
    x=fm.attach_features(sp,f,close,f,fast_bars=2,slow_bars=3)
    remove=12
    x=x.drop(index=remove).reset_index(drop=True)
    q=fm.attach_targets(x)
    assert np.isnan(q.loc[remove-1,'ret_5m'])
    assert np.isnan(q.loc[remove-3,'ret_30m'])


def test_exact_peer_join_not_forward_filled():
    _,spot,close,f=simulated_inputs(days=4)
    p=f.drop(index=20).reset_index(drop=True)
    z=fm.attach_features(spot,f,close,p,fast_bars=2,slow_bars=3)
    assert not z.loc[20,'feature_ready']
    assert np.isnan(z.loc[20,'peer_ret5'])
    assert z.loc[21,'feature_ready']


def test_fold_discovery_omits_legacy_last_20_percent():
    cfg=fm.Specification()
    dates=pd.date_range('2024-10-01',periods=730,freq='D',tz='UTC')
    folds=fm.folds_from_days(dates,cfg)
    assert len(folds)==3
    legacy_start=dates[int(len(dates)*.8)]
    for fold in folds:
        assert max(fold['validation_days'])<legacy_start
        assert max(fold['train_days'])+pd.Timedelta(days=cfg.embargo_days)<min(fold['validation_days'])


def test_fit_score_probability_and_magnitude():
    rng=np.random.default_rng(7);n=2000
    X=rng.normal(size=(n,6));y=np.where(rng.random(n)<.02,0,np.where(X[:,0]>0,1,-1))
    mag={f'abs_{h}m':np.abs(rng.normal(.001+.0002*X[:,1],.0002,n)) for h in fm.HORIZONS}
    frame=pd.DataFrame(mag)
    score=fm.fit_score(X[:1400],X[1400:],y[:1400],y[1400:],frame.iloc[:1400],frame.iloc[1400:],fm.Specification())
    assert score['n_train']==1400
    assert np.isfinite(score['binary_logloss'])
    assert all(np.isfinite(score[f'abs_{h}m_mse_bps2']) for h in fm.HORIZONS)
    assert all(score[f'abs_{h}m_forecast_mean_bps']>=0 for h in fm.HORIZONS)


def test_ranking_uses_matching_asset_fold_benchmark():
    rows=[]
    for a in ('BTC','ETH'):
        for fold in (1,2):
            for combo,delta in [('base',0),('momentum',-.02),('signed_flow',+.01)]:
                row={'asset':a,'fold':fold,'combination':combo,'family_count':int(combo!='base'),
                     'n_features':6,'n_train':1000,'n_validation':300}
                for key in ['threeclass_logloss','binary_logloss']+[f'abs_{h}m_mse_bps2' for h in fm.HORIZONS]:
                    row[key]=1+(10 if a=='ETH' else 0)+(1 if fold==2 else 0)+delta
                rows.append(row)
    r=fm.rank_combinations(pd.DataFrame(rows))
    assert r.iloc[0].combination=='momentum'
    assert r.iloc[0].delta_threeclass_logloss==pytest.approx(-.02)
    assert r.iloc[0].n_asset_folds==4


def test_scan_one_combination_with_checkpoint(tmp_path):
    minute,spot,close,f=simulated_inputs(days=64)
    pnl=fm.attach_targets(fm.attach_features(spot,f,close,f,fast_bars=2,slow_bars=3))
    pnl['asset']='BTC';pnl['date']=pnl.state_time.dt.floor('D')
    cfg=fm.Specification()
    file=tmp_path/'checkpoint.csv'
    scores,folds,dates=fm.scan_asset(pnl,cfg,max_combos=1,checkpoint=file)
    assert len(scores)==6
    assert scores.combination.nunique()==2
    assert file.exists()
    second,_,_=fm.scan_asset(pnl,cfg,max_combos=1,checkpoint=file)
    pd.testing.assert_frame_equal(scores,second)
    assert len(dates)>=60


def test_parquet_roundtrip_if_engine_available(tmp_path):
    pytest.importorskip('pyarrow')
    minute,*_=simulated_inputs()
    path=tmp_path/'BTCUSDT_um_1m.parquet'
    minute.to_parquet(path,index=False)
    back=pd.read_parquet(path)
    assert len(fm.minute_to_five(back))==3*288
