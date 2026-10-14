"""CS03b numerical, causal and attribution regression tests. Synthetic only."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import pytest
from data_engineering.modeling import candle_streak_flow_attribution as a
from data_engineering.modeling import candle_streak_trade_flow as tf


def sample(n=2500,seed=25):
    rng=np.random.default_rng(seed)
    sign=rng.choice([-1,0,1],n,p=[.44,.12,.44])
    ages=rng.integers(1,12,n);ages[sign==0]=0
    f=rng.uniform(-.9,.9,n)
    y=rng.choice([-1,0,1],n,p=[.47,.06,.47])
    t=pd.date_range('2025-02-01',periods=n,freq='5min',tz='UTC')
    d=pd.DataFrame({'state_time':t,'date':t.floor('D'),'period':'train','sign':sign,'length':ages,
        'flow_5m':f,'flow_30m':.92*f+.025*rng.normal(size=n),
        'log_rv24h':rng.normal(-13,.9,n),'log_fastslow':rng.normal(0,.8,n),
        'log_quote_volume':rng.normal(15,.6,n),'log_trades':rng.normal(10,.5,n),
        'next_sign':y,'scorable':True})
    d['outcome']=np.where(d.sign.eq(0),np.where(d.next_sign.eq(0),'flat_stays',
                            np.where(d.next_sign.eq(1),'start_up_after_flat','start_down_after_flat')),
                   np.where(d.sign.eq(d.next_sign),'continue',
                            np.where(d.next_sign.eq(0),'stop_flat','reverse_new_streak')))
    return d


def test_feature_families_immutable_and_nested():
    assert len(a.FEATURES)==4
    assert set(a.SIGNED).isdisjoint(a.ACTIVITY)
    assert a.FEATURES['price_vol']==tf.BASE_FEATURES['sign_streak_vol']
    assert a.FEATURES['price_vol_both']==tf.BASE_FEATURES['sign_streak_vol_flow']
    assert len(a.FEATURES['price_vol_both'])==12


def test_identical_to_cs03_full_fit():
    g=sample(n=2600)
    old=tf.fit_models(g,C=.25)
    fit=a.fit_ablation(g,C=.25)
    p=a.predict_ablation(g.iloc[:350],fit)
    p0=tf.predict_models(g.iloc[:350],old)
    np.testing.assert_allclose(p['price_vol'],p0['sign_streak_vol'],rtol=0,atol=2e-9)
    np.testing.assert_allclose(p['price_vol_both'],p0['sign_streak_vol_flow'],rtol=0,atol=2e-9)
    assert all(v.shape==(350,3) and np.allclose(v.sum(axis=1),1) for v in p.values())


def test_future_labels_do_not_affect_frozen_probabilities():
    g=sample(n=2600)
    fit=a.fit_ablation(g.iloc[:1800]);later=g.iloc[1800:].copy()
    p=a.predict_ablation(later,fit)
    later.loc[later.index[:100],'next_sign']=-later.loc[later.index[:100],'next_sign']
    q=a.predict_ablation(later,fit)
    for name in p:np.testing.assert_array_equal(p[name],q[name])


def test_sparse_zero_class_and_collinear_flow_converge():
    g=sample(n=5000)
    g['next_sign']=np.where(np.arange(len(g))%113==0,0,
                           np.where(np.arange(len(g))%2,1,-1))
    g['log_trades']=g.log_quote_volume*.98+.02
    g['flow_30m']=g.flow_5m*.997
    fit=a.fit_ablation(g)
    assert all(m.n_iter_[0]<m.max_iter for _,_,m in fit['models'].values())


def test_missing_class_raises():
    g=sample(n=350);g.loc[g.next_sign==0,'next_sign']=1
    with pytest.raises(ValueError,match='all DOWN/ZERO/UP'):
        a.fit_ablation(g)


def test_binary_score_conditional_and_zero_always_in_threeclass():
    g=sample(n=500)
    p={m:np.tile([.3,.2,.5],(len(g),1)) for m in a.MODEL_NAMES}
    l=a.origin_losses(g,p)
    c=a.scores_from_losses(l)
    for m in a.MODEL_NAMES:
        assert int(c.query('model==@m and metric=="threeclass_logloss"').n.iloc[0])==len(g)
        assert int(c.query('model==@m and metric=="binary_logloss"').n.iloc[0])==int(g.next_sign.ne(0).sum())
        assert int(c.query('model==@m and metric=="flat_vs_move_logloss"').n.iloc[0])==int(g.sign.ne(0).sum())
        assert int(c.query('model==@m and metric=="start_up_logloss"').n.iloc[0])==int((g.sign.eq(0)&g.next_sign.ne(0)).sum())
    gg=c.query('model=="price_vol" and metric=="threeclass_logloss"').score.iloc[0]
    target=np.searchsorted(a.CLASSES,g.next_sign.to_numpy(int))
    np.testing.assert_allclose(gg,-np.log(np.array([.3,.2,.5])[target]).mean())


def test_reversal_probabilities_correct_for_both_current_signs():
    g=sample(n=300)
    p={"price_vol":np.tile([.2,.1,.7],(len(g),1))}
    l=a.origin_losses(g,p)
    r=l[l.metric=='reversal_logloss'].loss.to_numpy()
    cur=g[(g.sign!=0)&(g.next_sign!=0)]
    pred=np.where(cur.sign.to_numpy()==1,.2/.9,.7/.9)
    yy=(cur.next_sign.to_numpy()!=cur.sign.to_numpy()).astype(int)
    expected=-yy*np.log(pred)-(1-yy)*np.log1p(-pred)
    np.testing.assert_allclose(r,expected)


def test_probability_matrices_are_not_retroactively_refitted():
    g=sample(n=2300)
    fit=a.fit_ablation(g)
    future=g.iloc[:100].copy()
    pred=a.predict_ablation(future,fit)
    future['next_sign']=0;future['outcome']='stop_flat'
    np.testing.assert_array_equal(a.predict_ablation(future,fit)['price_vol_both'],pred['price_vol_both'])


def test_bootstrap_negative_delta_paired_and_sparse():
    rng=np.random.default_rng(211)
    dates=pd.date_range('2025-01-01',periods=45,freq='D',tz='UTC')
    losses=[]
    for date in dates:
        for m in a.MODEL_NAMES:
            for _ in range(15):
                losses.append({'date':date,'metric':'threeclass_logloss','model':m,
                    'loss':.7+(0 if m=='price_vol' else -.025)+.02*rng.normal()})
    df=pd.DataFrame(losses)
    bs=a.day_bootstrap(df,B=150)
    row=bs[(bs.model=='price_vol_signed')&(bs.reference=='price_vol')].iloc[0]
    assert row.status=='OK' and row.ci975<0
    assert len(bs.query('metric=="threeclass_logloss"'))==len(a.COMPARISONS)
    s=a.day_bootstrap(df[df.date.isin(dates[:2])],B=20)
    assert (s.status=='SPARSE').all()


def test_missing_period_or_asset_not_success():
    b=pd.DataFrame([{'asset':'BTC','period':'test','metric':'binary_logloss','model':'price_vol_signed',
                    'reference':'price_vol','status':'OK','delta_equal_day':-.5,'ci975':-.2}])
    d=a.decision(b)
    assert d['verdict']=='INSUFFICIENT_PRIMARY_EVALUATION'
    assert not d['signed_binary_gate']


def test_transition_probabilities_have_correct_orientation():
    g=sample(n=360)
    prob={m:np.tile([.2,.15,.65],(len(g),1)) for m in a.MODEL_NAMES}
    t=a.transition_calibration(g,prob,[-.33,.33])
    pos=t[t.sign==1]
    neg=t[t.sign==-1]
    assert np.allclose(pos.p_continue,.65)
    assert np.allclose(neg.p_continue,.2)
    assert (t.n>0).all()


def test_negative_control_returns_both_components():
    g=sample(n=600)
    fit=a.fit_ablation(g.iloc[:350]);test=g.iloc[350:]
    null=a.feature_shuffle_null(test,fit,reps=3)
    assert set(null.component)=={'signed','activity'}
    assert (null.n_shuffle==3).all()
    assert np.isfinite(null.null_median).all()


def test_positive_synthetic_signed_pressure_recovered():
    g=sample(n=3500)
    f=g.flow_5m.to_numpy();g['next_sign']=np.where(f>.08,1,np.where(f<-.08,-1,0))
    fit=a.fit_ablation(g.iloc[:2500]);z=g.iloc[2500:]
    preds=a.predict_ablation(z,fit)
    loss=a.scores_from_losses(a.origin_losses(z,preds))
    x=loss.query('metric=="threeclass_logloss"').set_index('model').score
    assert x['price_vol_signed']+.15<x['price_vol']


def test_generated_event_panel_is_strictly_future_target():
    m=pd.DataFrame({'open_time':pd.date_range('2026-01-01',periods=50,freq='1min',tz='UTC'),
                    'quote_volume':100.,'taker_buy_quote_volume':60.,'trades':4})
    s=pd.DataFrame({'state_time':pd.date_range('2026-01-01T00:05:00Z',periods=10,freq='5min'),
                    'return_5m':[.002,.001,-.002,-.003,0,.002,.001,-.002,.002,.001]})
    f=tf.one_minute_futures_flow(m)
    a0=tf.build_origin_panel(s,f,fast_bars=2,slow_bars=3,flow_bars=2)
    sp=s.copy();sp.loc[9,'return_5m']*= -3
    a1=tf.build_origin_panel(sp,f,fast_bars=2,slow_bars=3,flow_bars=2)
    cols=['flow_5m','flow_30m','log_rv24h','log_fastslow','sign','length']
    pd.testing.assert_frame_equal(a0.loc[:8,cols],a1.loc[:8,cols])
    assert a0.next_sign.iloc[8]!=a1.next_sign.iloc[8]
