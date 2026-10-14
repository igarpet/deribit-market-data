"""Synthetic causality, calendar, full-matrix and forecast regressions for CS02."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import pytest
from data_engineering.modeling import candle_streak_volatility as m


def raw_days(days=82,asset='BTC',seed=21,start='2025-01-01'):
    rng=np.random.default_rng(seed)
    t=pd.date_range(start,periods=days*288,freq='5min',tz='UTC')
    sig=np.where(rng.random(len(t))<.5,1.,-1.)
    # volatility changing by day, occasional zero returns
    scales=(.0007+.001*np.sin(np.arange(len(t))/288/11)**2)
    r=sig*scales*rng.uniform(.25,1.75,len(t))
    r[100::711]=0.
    return pd.DataFrame({'underlying':asset,'state_time':t,'return_5m':r})


def prepared(days=82):
    raw=raw_days(days)
    e,_=m.build_vol_events(raw)
    full=m.complete_utc_days(raw)['BTC']
    q,meta=m.split_eligible(e,full)
    bins=m.fit_vol_bins(q[q.period=='train'])
    q=m.apply_vol_bins(q,bins)
    return raw,q,meta,bins


def test_volatility_uses_only_past_including_current_close():
    raw=raw_days(8)
    a=m.add_past_volatility(raw)
    altered=raw.copy();altered.loc[450,'return_5m']*=200
    b=m.add_past_volatility(altered)
    assert np.array_equal(a.log_rv24h.iloc[:450].to_numpy(),b.log_rv24h.iloc[:450].to_numpy(),equal_nan=True)
    assert a.loc[449,'log_fastslow']==b.loc[449,'log_fastslow']
    assert a.loc[450,'log_fastslow']!=b.loc[450,'log_fastslow']


def test_gap_restarts_volatility_without_false_stitch():
    raw=raw_days(4);raw=raw.drop(index=[320]).reset_index(drop=True)
    q=m.add_past_volatility(raw)
    assert not q.vol_ready.iloc[320:607].any()
    assert q.vol_ready.iloc[607]


def test_flat_return_is_zero_next_outcome_not_deleted():
    raw=raw_days(2)
    e,_=m.build_vol_events(raw,fast_bars=2,slow_bars=3)
    zero=raw.loc[100,'state_time']
    prior=e.loc[e.state_time==zero-pd.Timedelta(minutes=5)]
    assert len(prior)==1 and prior.next_valid.iloc[0]
    assert prior.next_sign.iloc[0]==0


def test_partial_utc_days_excluded_even_if_precedes_cutoff():
    raw=raw_days(90)
    raw=raw.iloc[7:-281].reset_index(drop=True)
    full=m.complete_utc_days(raw)['BTC']
    assert len(full)==88
    e,_=m.build_vol_events(raw)
    q,meta=m.split_eligible(e,full)
    assert q.date.min()==full.min()
    assert q.date.max()==full.max()
    assert not q.date.isin([pd.Timestamp('2025-01-01',tz='UTC')]).any()
    assert meta['full_utc_days']==88


def test_all_three_bins_training_only_and_frozen():
    _,q,meta,b=prepared()
    assert len(b['level_cut'])==len(b['trend_cut'])==2
    c=q.copy();c.loc[c.period=='test','log_rv24h']+=200
    other=m.apply_vol_bins(c,b)
    assert (other.loc[other.period=='test','level_bin']==2).all()
    assert np.array_equal(other.loc[other.period=='train','level_bin'],q.loc[q.period=='train','level_bin'])


def test_split_boundaries_cannot_cross_future_label():
    _,q,meta,b=prepared()
    assert meta['full_utc_days']==82
    for period in ['train','valid']:
        latest=q.loc[q.period==period].sort_values('state_time').iloc[-1]
        assert not latest.scorable
    assert q.loc[q.period=='test','scorable'].sum()>1000


def test_prior_and_matrix_normalized_all_k_unbounded():
    _,q,_,_=prepared()
    fit=m.fit_matrix(q[q.period=='train'])
    probe=q.loc[q.period=='test'].head(4).copy()
    probe.loc[probe.index[-1],'length']=100000
    p=m.predict_matrix(probe,fit)
    assert list(p)==list(m.MODELS)
    for x in p.values():
        assert np.allclose(x.sum(axis=1),1) and np.all(x>0) and np.all(x<1)
    assert np.isfinite(m.probability_matrix(q[q.period=='train'],fit)[['p_down','p_zero','p_up']].to_numpy()).all()


def test_label_mutations_do_not_change_predictive_probabilities():
    _,q,_,_=prepared()
    tr=q[q.period=='train'];te=q[(q.period=='test')&q.scorable]
    fit=m.fit_matrix(tr)
    a=m.predict_matrix(te.head(10),fit)
    changed=te.head(10).copy()
    changed['next_sign']*=-1;changed['continuation']=~changed.continuation
    b=m.predict_matrix(changed,fit)
    for key in a:assert np.array_equal(a[key],b[key])


def test_training_labels_alter_model_but_test_labels_cannot():
    _,q,_,_=prepared()
    tr=q[q.period=='train'];te=q.loc[(q.period=='test')&q.scorable].head(40)
    fit=m.fit_matrix(tr)
    first=m.predict_matrix(te,fit)
    altered=tr.copy()
    altered.loc[altered.scorable,'next_sign']=1
    fitted=m.fit_matrix(altered)
    second=m.predict_matrix(te,fitted)
    assert not np.allclose(first['iid_3class'],second['iid_3class'])


def test_matrix_reduces_to_parents_with_unseen_cells():
    _,q,_,_=prepared()
    f=m.fit_matrix(q.loc[q.period=='train'])
    probe=q.head(1).copy();probe['length']=9999
    p=m.predict_matrix(probe,f)
    assert np.allclose(p['streak_3class'],p['sign_3class'])
    assert np.allclose(p['streak_vol_matrix_3class'],.5*(p['streak_3class']+p['sign_vol_3class']))


def test_actual_score_threeclass_and_bootstrap_finite():
    _,q,_,_=prepared()
    tr=q[q.period=='train'];te=q[(q.period=='test')&q.scorable].copy()
    fit=m.fit_matrix(tr)
    scores,per=m.score_threeclass(te,m.predict_matrix(te,fit))
    boots=m.day_bootstrap(per,B=30)
    assert len(scores)==6 and len(boots)==14
    assert (scores.n>0).all()
    assert np.isfinite(scores[['logloss','brier3']].to_numpy()).all()
    assert np.isfinite(boots[['ci025','ci975','mean_day_difference']].to_numpy()).all()
    assert boots.min_events_day.min()>0


def test_descriptive_distribution_and_monthly_provenance():
    _,q,_,_=prepared()
    tr=q[q.period=='train'];fit=m.fit_matrix(tr)
    keys=[tuple(x) for x in q[['sign','length','level_bin','trend_bin']].drop_duplicates().itertuples(index=False,name=None)]
    trained=m.probability_matrix(tr,fit,include_keys=keys)
    descr=m.descriptive_matrix(q)
    assert set(keys).issubset(set(map(tuple,trained[['sign','length','level_bin','trend_bin']].to_numpy())))
    assert (descr[['n_up','n_down','n_zero']].sum(axis=1)==descr.n).all()
    month=m.monthly_stability(q,min_n=10)
    assert len(month)>0 and (month.n>=10).all()


def test_multiclass_zero_brier_and_nll():
    a=pd.DataFrame({'state_time':pd.date_range('2025-01-01',periods=3,freq='5min',tz='UTC'),
                    'next_sign':[-1,0,1]})
    pr={model:np.full((3,3),1/3) for model in m.MODELS}
    scores,_=m.score_threeclass(a,pr)
    assert np.allclose(scores.logloss,np.log(3))
    assert np.allclose(scores.brier3,2/3)


def test_matrix_not_importing_legacy_fitted_har_parameters():
    assert m.registration()['status']=='HISTORICAL_EXPLORATORY'
    assert m.registration()['slow_bars']==288


def test_volatility_alignment_null_reproducible_and_finite():
    _,q,_,_=prepared()
    fit=m.fit_matrix(q[q.period=='train'])
    te=q[(q.period=='test') & q.scorable]
    a=m.volatility_alignment_null(te,fit,reps=8,seed=27)
    b=m.volatility_alignment_null(te,fit,reps=8,seed=27)
    assert a==b and a['permutations']==8
    assert np.isfinite(list(a.values())).all()
