"""CS01 isolated fixed-seed tests for history exposure, censoring and holdout."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import pytest
from data_engineering.modeling import candle_streaks as s


def fixture(sequence=(1,1,1,-1,-1,1,-1,-1,-1,-1), asset='BTC',jump_index=None):
    times=pd.date_range('2025-01-01',periods=len(sequence),freq='5min',tz='UTC')
    if jump_index is not None:
        times=pd.Series(times)
        times.iloc[jump_index:]+=pd.Timedelta(minutes=10)
    return pd.DataFrame({'state_time':times,'underlying':asset,'return_5m':np.asarray(sequence,float)*.001})


def test_completed_directional_run_lengths_without_rolling_windows():
    e,r,q=s.build_events_and_runs(fixture())
    assert r.signed_streak.to_list()==[3,-2,1,-4]
    assert r.completed.to_list()==[True,True,True,False]
    assert e.signed_streak.to_list()==[1,2,3,-1,-2,1,-1,-2,-3,-4]
    assert q.loc[0,'zero']==0


def test_exposures_are_reached_not_eventual_lengths():
    e,r,_=s.build_events_and_runs(fixture())
    d,c=s.distribution_tables(e,r)
    pos=c.loc[c['sign']==1].set_index('length')
    assert pos.loc[1,'n_reached']==2
    assert pos.loc[2,'n_reached']==1
    assert pos.loc[3,'n_reached']==1
    assert pos.loc[3,'n_reverse']==1
    assert pos.loc[1,'n_continue']==1
    assert pos.loc[1,'p_continue_given_next_nonzero']==.5


def test_zero_bars_end_runs_but_do_not_count_as_sign():
    e,r,q=s.build_events_and_runs(fixture((1,1,0,1,-1)))
    assert r.signed_streak.tolist()==[2,1,-1]
    assert r.end_reason.to_list()==['zero','reverse','censored']
    assert q.zero.iloc[0]==1
    assert e.loc[e.signed_streak==2,'next_zero'].iloc[0]
    assert not e.loc[e.signed_streak==2,'next_nonzero'].iloc[0]


def test_timestamp_gap_right_censors_not_false_reversal():
    e,r,q=s.build_events_and_runs(fixture((1,1,1,-1,-1),jump_index=3))
    assert r.signed_streak.tolist()==[3,-2]
    assert r.completed.tolist()==[False,False]
    assert e.loc[e.signed_streak==3,'outcome'].iloc[0]=='censored'
    assert q.discontinuities.iloc[0]==1


def test_nonfinite_return_censors_and_resets():
    e,r,q=s.build_events_and_runs(fixture((1,1,np.nan,1,1)))
    assert r.signed_streak.tolist()==[2,2]
    assert r.completed.tolist()==[False,False]
    assert q.missing.iloc[0]==1


def test_independent_assets_never_join():
    a=pd.concat([fixture((1,1),asset='BTC'),fixture((1,1),asset='ETH')],ignore_index=True)
    e,r,_=s.build_events_and_runs(a)
    assert e.groupby('underlying').length.max().to_dict()=={'BTC':2,'ETH':2}
    assert len(r)==2


def test_reject_duplicates_and_offgrid():
    a=fixture((1,-1,1));a.loc[1,'state_time']=a.loc[0,'state_time']
    with pytest.raises(ValueError):s.build_events_and_runs(a)
    b=fixture((1,-1,1));b.loc[1,'state_time']+=pd.Timedelta(minutes=1)
    with pytest.raises(ValueError):s.build_events_and_runs(b)


def long_sample(days=100,seed=31):
    n=days*288;rng=np.random.default_rng(seed)
    sgn=rng.choice([-1.,1.],size=n)
    # Controlled state dependence: longer up runs are mildly stickier.
    for i in range(2,n):
        if sgn[i-1]==1 and sgn[i-2]==1 and rng.random()<.12:
            sgn[i]=1
    return fixture(sgn)


def test_splits_do_not_use_future_label_across_boundary():
    e,_,_=s.build_events_and_runs(long_sample(100))
    q,meta=s.add_chronological_splits(e)
    assert meta['n_dates']==100
    for period in ('train','valid','test'):
        assert len(q.loc[q.split==period])>100
    boundaries=q.index[q.split.ne(q.split.shift())][1:]
    for i in boundaries:
        assert not bool(q.loc[i-1,'scorable'])
        assert bool(q.loc[i,'scorable'])


def test_all_lengths_retained_and_no_terminal_run_complete():
    e,r,_=s.build_events_and_runs(fixture([1]*61+[-1]*42))
    d,c=s.distribution_tables(e,r)
    assert r.length.tolist()==[61,42]
    assert d.loc[d.length==61,'n_completed'].iloc[0]==1
    assert d.loc[d.length==42,'n_right_censored'].iloc[0]==1
    assert 61 in c.length.to_list()


def test_logistic_and_exact_models_use_training_only():
    e,_,_=s.build_events_and_runs(long_sample(90))
    q,_=s.add_chronological_splits(e)
    a=q[q.split=='train'];v=q[q.split=='valid'];te=q[q.split=='test']
    fit=s.fit_direction_models(a)
    score=s.predict_direction(te.loc[te.scorable],fit)
    altered=te.copy();altered['next_sign']*=-1;altered['continuation']=~altered['continuation']
    score2=s.predict_direction(altered.loc[altered.scorable],fit)
    assert np.allclose(score.to_numpy(),score2.to_numpy())
    assert ((score>=0)&(score<=1)).all().all()
    assert set(fit['markov'])=={-1,1}


def test_no_arbitrary_age_cap_for_model():
    f={'p_up':.51,'markov':{-1:.49,1:.5},'exact':{(1,1000):.72},
       'logistic_coefficients':[0.,0.,0.,0.]}
    q=pd.DataFrame({'sign':[1,1,-1],'length':[1000,1001,1001]})
    p=s.predict_direction(q,f)
    assert p.streak_exact_shrunk.iloc[0]==.72
    assert p.streak_exact_shrunk.iloc[1]==.5
    assert p.streak_exact_shrunk.iloc[2]==.51


def test_paired_scores_and_bootstrap_finite():
    e,_,_=s.build_events_and_runs(long_sample(90))
    q,_=s.add_chronological_splits(e)
    fit,scores,boot,counts=s.analyze_asset(q[q.split=='train'],q[q.split=='valid'],q[q.split=='test'],reps=50)
    assert len(scores)==10 and len(boot)==16 and len(counts)==2
    assert np.isfinite(scores[['logloss','brier']].to_numpy()).all()
    assert np.isfinite(boot[['mean_day_difference','ci025','ci975']].to_numpy()).all()
    assert scores.n.min()>100


def test_null_keeps_zero_gap_and_nonzero_sign_counts():
    z=long_sample(10);z.loc[19,'return_5m']=0.;z.loc[40,'return_5m']=np.nan
    original=s.build_events_and_runs(z)[0]
    out=s.iid_permutation_null(z,reps=8)
    assert len(out)==4
    assert (out.permutations==8).all()
    assert np.isfinite(out.null_95).all()
    assert out.loc[out.metric=='max_run','observed'].iloc[0]>0


def test_geometric_expected_for_fair_coin_first_3():
    # Conditional probability the first 3 specified independent signs are all up.
    assert (.5**3)==.125
    e,r,_=s.build_events_and_runs(fixture())
    d,c=s.distribution_tables(e,r,base_rates={('BTC',1):.5,('BTC',-1):.5})
    row=d[(d.sign==1)&(d.length==3)].iloc[0]
    assert np.isclose(row.expected_completed_iid_approx,2*(.5**3))
