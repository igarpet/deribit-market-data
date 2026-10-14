"""Synthetic regression tests for strictly causal CS03 Binance trade-flow joins."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import pytest
from data_engineering.modeling import candle_streak_trade_flow as tf


def minute_data(nbars=10, start='2026-01-01T00:00:00Z'):
    t=pd.date_range(start,periods=nbars*5,freq='1min',tz='UTC')
    vol=np.full(len(t),100.0)
    buy=np.where(np.arange(len(t))<15,80.,20.)
    return pd.DataFrame({'open_time':t,'quote_volume':vol,'taker_buy_quote_volume':buy,'trades':np.ones(len(t))*5})


def spot_data(nbars=10,start='2026-01-01T00:00:00Z',signs=None):
    t=pd.date_range(start,periods=nbars,freq='5min',tz='UTC')+tf.STEP
    if signs is None:signs=[1,1,1,-1,-1,0,1,-1,0,0][:nbars]
    return pd.DataFrame({'state_time':t,'return_5m':np.array(signs,dtype=float)*.001})


def panel(signs=None,nbars=10):
    f=tf.one_minute_futures_flow(minute_data(nbars))
    s=spot_data(nbars,signs=signs)
    return tf.build_origin_panel(s,f,fast_bars=2,slow_bars=3,flow_bars=2)


def test_minute_buckets_and_asof():
    z=tf.one_minute_futures_flow(minute_data())
    assert len(z)==10
    assert str(z.state_time.iloc[0])=='2026-01-01 00:05:00+00:00'
    assert np.allclose(z.flow_5m[:3],.6)
    assert np.allclose(z.flow_5m[3:],-.6)
    assert (z.n_minutes==5).all()


def test_missing_minute_invalidates_bucket():
    m=minute_data();m=m.drop(3)
    z=tf.one_minute_futures_flow(m)
    assert not z.flow_valid.iloc[0]
    assert np.isnan(z.flow_5m.iloc[0])
    assert z.flow_valid.iloc[1]


def test_duplicate_minute_rejected():
    m=minute_data()
    with pytest.raises(ValueError,match='duplicated'):
        tf.one_minute_futures_flow(pd.concat([m,m.iloc[[0]]]))


def test_bad_buy_volume_rejected():
    m=minute_data();m.loc[0,'taker_buy_quote_volume']=200
    with pytest.raises(ValueError,match='invalid'):
        tf.one_minute_futures_flow(m)


def test_missing_input_column_rejected():
    with pytest.raises(ValueError,match='Missing'):
        tf.one_minute_futures_flow(minute_data().drop(columns='trades'))


def test_streaks_flat_start_stop_and_direction():
    q=panel()
    assert q.sign.tolist()==[1,1,1,-1,-1,0,1,-1,0,0]
    assert q.length.tolist()==[1,2,3,1,2,0,1,1,0,0]
    assert q.outcome.tolist()==['continue','continue','reverse_new_streak','continue',
                                'stop_flat','start_up_after_flat','reverse_new_streak',
                                'stop_flat','flat_stays','censored']


def test_end_of_bar_flow_predicts_only_next():
    q=panel()
    assert q.flow_5m.iloc[2]==.6
    assert q.next_sign.iloc[2]==-1
    assert q.flow_5m.iloc[3]==-.6
    assert q.next_sign.iloc[3]==-1
    assert q.flow_30m.iloc[2]==pytest.approx(.6)
    assert q.flow_30m.iloc[3]==pytest.approx(0.0)
    # Mutating a future minute changes only its own and subsequent features.
    m=minute_data();m.loc[m.open_time.eq(pd.Timestamp('2026-01-01T00:16:00Z')),'taker_buy_quote_volume']=100
    new=tf.build_origin_panel(spot_data(),tf.one_minute_futures_flow(m),fast_bars=2,slow_bars=3,flow_bars=2)
    assert np.allclose(q.flow_5m.iloc[:3],new.flow_5m.iloc[:3])
    assert q.flow_5m.iloc[3]!=new.flow_5m.iloc[3]


def test_flat_still_starts_after_zero():
    q=panel()
    assert q.loc[5,'length']==0
    assert q.loc[5,'outcome']=='start_up_after_flat'
    assert q.loc[8,'outcome']=='flat_stays'


def test_missing_flow_not_filled():
    f=tf.one_minute_futures_flow(minute_data().drop(index=10))
    q=tf.build_origin_panel(spot_data(),f,fast_bars=2,slow_bars=3,flow_bars=2)
    assert not q.features_ready.iloc[2]
    assert not q.features_ready.iloc[3] # rolling two-bar feature must rebuild
    assert q.features_ready.iloc[4]


def test_history_gap_resets_age():
    s=spot_data();s=s.drop(index=1).reset_index(drop=True)
    q=tf.build_origin_panel(s,tf.one_minute_futures_flow(minute_data()),fast_bars=2,slow_bars=3,flow_bars=2)
    assert q.length.iloc[1]==1
    assert not q.next_valid.iloc[0]


def test_full_days_require_both_sources():
    t=pd.date_range('2026-01-01',periods=288*2,freq='5min',tz='UTC')
    q=pd.DataFrame({'state_time':t,'date':t.floor('D'),'features_ready':True})
    assert len(tf.full_days(q))==2
    q.loc[q.index[-1],'features_ready']=False
    assert len(tf.full_days(q))==1


def test_split_excludes_cross_split_labels():
    t=pd.date_range('2026-01-01',periods=90*288,freq='5min',tz='UTC')
    panel=pd.DataFrame({'state_time':t,'date':t.floor('D'),'features_ready':True,
                        'next_valid':True})
    x,c=tf.split_panel(panel,pd.DatetimeIndex(sorted(panel.date.unique())))
    assert c['train_days']==54 and c['valid_days']==18 and c['test_days']==18
    # Exactly two cross-period boundary forecasts excluded.
    assert (~x.scorable).sum()==3


def test_design_age_is_not_capped():
    d=panel();d.loc[5,'length']=1000000;d.loc[5,'sign']=1
    x=tf.raw_design(d)
    assert x.loc[5,'age_signed']==pytest.approx(np.log1p(1000000))


def sample_fit_dataset(n=2100):
    rng=np.random.default_rng(25)
    sign=rng.choice([-1,0,1],n,p=[.46,.08,.46])
    age=rng.integers(1,11,n);age[sign==0]=0
    flow=rng.uniform(-1,1,n)
    retlog=rng.normal(-13,1,n)
    target=rng.choice([-1,0,1],n,p=[.47,.06,.47])
    t=pd.date_range('2026-01-01',periods=n,freq='5min',tz='UTC')
    return pd.DataFrame({'state_time':t,'date':t.floor('D'),'sign':sign,'length':age,
         'flow_5m':flow,'flow_30m':flow*.8,'log_rv24h':retlog,
         'log_fastslow':rng.normal(0,.5,n),'log_quote_volume':rng.normal(12,1,n),
         'log_trades':rng.normal(8,1,n),'next_sign':target,'scorable':True,
         'outcome':np.where(sign==target,'continue','reverse_new_streak')})


def test_models_fit_normalization_and_frozen_predictions():
    g=sample_fit_dataset()
    fit=tf.fit_models(g)
    p=tf.predict_models(g,fit)
    assert tuple(p)==tf.MODELS
    for v in p.values():
        assert v.shape==(len(g),3)
        assert np.isfinite(v).all()
        assert (v>0).all()
        assert np.allclose(v.sum(axis=1),1)
    q=g.copy();q.loc[0,'next_sign']=-q.loc[0,'next_sign']
    assert np.array_equal(p['sign_streak_vol_flow'],tf.predict_models(q,fit)['sign_streak_vol_flow'])


def test_flow_signal_can_be_detected_synthetically():
    g=sample_fit_dataset(n=3500)
    g['next_sign']=np.where(g.flow_5m>.1,1,np.where(g.flow_5m<-.1,-1,0))
    train=g.iloc[:2500].copy();test=g.iloc[2500:].copy()
    fit=tf.fit_models(train)
    pred=tf.predict_models(test,fit)
    scores,_=tf.score_models(test,pred)
    s=scores.set_index('model')
    assert s.loc['sign_streak_vol_flow','threeclass_logloss']+0.1 < s.loc['sign_streak_vol','threeclass_logloss']


def test_transition_table_sum():
    g=sample_fit_dataset();g['period']='test';g.loc[g.sign==0,'outcome']='flat_stays'
    q=tf.observed_transitions(g,[-.3,.3])
    assert q.n.sum()==len(g)
    assert q.n.min()>=1


def test_bootstrap_and_shuffled_control():
    g=sample_fit_dataset(n=55*288)
    fit=tf.fit_models(g.iloc[:5000])
    test=g.iloc[5000:].copy()
    scores,los=tf.score_models(test,tf.predict_models(test,fit))
    boot=tf.paired_day_bootstrap(los,B=50)
    assert len(scores)==6 and len(boot)==18
    assert (boot.days>=20).all()
    null=tf.shuffle_test_flow(test,fit,B=3)
    assert np.isfinite(null['observed_delta'])
    assert null['reps']==3


def test_existing_parquet_schema_is_readable(tmp_path):
    pytest.importorskip('pyarrow')
    p=tmp_path/'BTCUSDT_um_1m.parquet'
    minute_data().to_parquet(p,index=False)
    z=tf.read_futures_parquet(p)
    assert len(z)==10 and z.flow_valid.all()
    assert z.state_time.iloc[0]==pd.Timestamp('2026-01-01T00:05:00Z')
    z2=tf.read_futures_parquet(p,start='2026-01-01T00:05:00Z',end='2026-01-01T00:10:00Z')
    assert len(z2)==1 and z2.state_time.iloc[0]==pd.Timestamp('2026-01-01T00:10:00Z')


def test_no_future_return_leak_through_prepared_predictors():
    s=spot_data()
    f=tf.one_minute_futures_flow(minute_data())
    before=tf.build_origin_panel(s,f,fast_bars=2,slow_bars=3,flow_bars=2)
    s.loc[9,'return_5m']=12345.0
    after=tf.build_origin_panel(s,f,fast_bars=2,slow_bars=3,flow_bars=2)
    cols=['sign','length','flow_5m','flow_30m','log_rv24h','log_fastslow','log_quote_volume','log_trades']
    pd.testing.assert_frame_equal(before.loc[:8,cols],after.loc[:8,cols])
    assert before.loc[8,'next_sign']!=after.loc[8,'next_sign']


def test_rare_zero_outcomes_and_correlated_trade_flow_converge():
    """Realistic small zero-return class plus correlated flow/volume features.

    Regression for the repository run whose L-BFGS fit exhausted 300 steps.
    No change to fixed C=.25 or the feature families is permitted.
    """
    g=sample_fit_dataset(n=12000)
    rng=np.random.default_rng(2614)
    g['next_sign']=np.where(rng.random(len(g))<.002,0,
                            np.where(rng.random(len(g))<.5,-1,1))
    g['flow_30m']=.98*g.flow_5m+.005*rng.normal(size=len(g))
    g['log_trades']=.97*g.log_quote_volume+.02*rng.normal(size=len(g))
    fit=tf.fit_models(g,C=.25)
    assert len(fit['models'])==5
    for terms,scaler,model in fit['models'].values():
        assert model.solver=='newton-cholesky'
        assert model.C==.25
        assert model.n_iter_[0]<model.max_iter
        pp=model.predict_proba(scaler.transform(tf.raw_design(g.iloc[:100])[list(terms)].to_numpy(float)))
        assert pp.shape==(100,3) and np.isfinite(pp).all()
        assert np.allclose(pp.sum(axis=1),1)


def test_solver_matches_converged_lbfgs_fixed_objective():
    """The correction changes numerical optimization, not the L2 loss surface."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    g=sample_fit_dataset(n=6000)
    y=g.next_sign.to_numpy(int)
    terms=tf.BASE_FEATURES['sign_streak_vol_flow']
    X=StandardScaler().fit_transform(tf.raw_design(g)[list(terms)].to_numpy(float))
    baseline=LogisticRegression(C=.25,max_iter=2500,tol=1e-8,solver='lbfgs').fit(X,y)
    fixed=LogisticRegression(C=.25,max_iter=120,tol=1e-7,solver='newton-cholesky').fit(X,y)
    assert baseline.n_iter_[0]<baseline.max_iter
    assert fixed.n_iter_[0]<fixed.max_iter
    np.testing.assert_allclose(baseline.predict_proba(X[:300]),
                               fixed.predict_proba(X[:300]),atol=1e-3,rtol=0)
