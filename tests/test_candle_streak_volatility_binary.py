"""CS02b: immediately-next-bar UP/DOWN causal and numerical regression tests."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data_engineering.modeling import candle_streak_volatility as cv
from data_engineering.modeling import candle_streak_volatility_binary as b


def fixture(n=720,seed=45):
    rng=np.random.default_rng(seed)
    current=rng.choice([-1.,0.,1.],size=n,p=[.47,.06,.47])
    q=pd.DataFrame({'underlying':'BTC','state_time':pd.date_range('2025-01-01',periods=n,freq='5min',tz='UTC'),
                    'return_5m':current*.001})
    ev,_=cv.build_vol_events(q,fast_bars=3,slow_bars=8)
    ev=ev.loc[ev.vol_ready].reset_index(drop=True)
    # emulate splitter for small synthetic study, no split-crossing.
    ev['period']='train';ev['scorable']=ev.next_valid.astype(bool)
    cut={'level_cut':[-14.0,-13.5],'trend_cut':[-.1,.1]}
    ev=cv.apply_vol_bins(ev,cut)
    return b.filter_binary_next(ev)


def test_next_zero_is_removed_not_future_skipped():
    e=fixture()
    assert e.excluded_next_zero.sum()>5
    assert (e.loc[e.excluded_next_zero,'next_sign']==0).all()
    assert not e.loc[e.excluded_next_zero,'scorable'].any()
    assert (e.loc[e.scorable,'next_sign']!=0).all()
    assert (e.loc[e.scorable,'next_valid']).all()


def test_zero_current_breaks_streak_without_stitching():
    q=pd.DataFrame({'underlying':'BTC','state_time':pd.date_range('2025-01-01',periods=15,freq='5min',tz='UTC'),
                    'return_5m':[1.,1.,1.,0.,1.,1.,0.,0.,-1.,-1.,1.,0.,1.,1.,1.]})
    ev,_=cv.build_vol_events(q,fast_bars=2,slow_bars=3)
    got=ev.loc[ev.vol_ready & ev.state_time.isin([q.state_time[i] for i in [4,5,8,9,12,13]])]
    assert got.length.tolist()==[1,2,1,2,1,2]
    assert ev.loc[ev.state_time==q.state_time[2],'next_sign'].iloc[0]==0


def test_zero_current_rows_do_not_enter_forecast_population():
    e=fixture()
    assert e.sign.isin([-1,1]).all() and e.length.ge(1).all()


def test_retains_all_streak_lengths_no_arbitrary_cap():
    n=32
    frame=pd.DataFrame({'underlying':'BTC','state_time':pd.date_range('2025-01-01',periods=n,freq='5min',tz='UTC'),
                        'return_5m':np.r_[np.ones(25),-np.ones(7)]})
    ev,_=cv.build_vol_events(frame,fast_bars=2,slow_bars=4)
    assert ev.length.max()==25
    ev=ev.loc[ev.vol_ready].copy();ev['period']='train';ev['scorable']=ev.next_valid
    ev['level_bin']=0;ev['trend_bin']=0
    f=b.filter_binary_next(ev)
    assert f.length.max()==25


def test_filter_missing_columns():
    with pytest.raises(ValueError):b.filter_binary_next(pd.DataFrame({'sign':[1]}))


def test_binary_fit_exact_and_unseen_probability_normalize():
    e=fixture(1400)
    fit=b.fit_binary_matrix(e)
    x=e.loc[e.scorable].copy()
    p=b.predict_binary(x,fit)
    assert set(p)==set(b.MODELS)
    assert all(np.isfinite(v).all() and ((v>0)&(v<1)).all() for v in p.values())
    unseen=x.iloc[:2].copy();unseen.loc[:,'length']=10_000
    q=b.predict_binary(unseen,fit)
    assert np.allclose(q['streak_vol_matrix_binary'],q['additive_binary'])
    assert not any(k[1]==10_000 for k in fit['joint'])


def test_current_target_cannot_change_predictions():
    e=fixture(1100)
    fit=b.fit_binary_matrix(e.iloc[:600])
    evaluation=e.iloc[600:].copy()
    before=b.predict_binary(evaluation,fit)
    changed=evaluation.copy();changed['next_sign']=-changed['next_sign']
    after=b.predict_binary(changed,fit)
    assert all(np.array_equal(before[k],after[k]) for k in before)


def test_test_label_changes_cannot_refit_model():
    e=fixture(1000)
    tr=e.iloc[:650].copy();te=e.iloc[650:].copy()
    fit=b.fit_binary_matrix(tr)
    p=b.predict_binary(te,fit)
    te['next_sign']=-te.next_sign
    assert all(np.array_equal(v,b.predict_binary(te,fit)[k]) for k,v in p.items())


def test_train_filter_and_cs02_binary_direct_consistency():
    e=fixture(1300)
    raw=e.copy();raw['scorable']=raw.scorable_threeclass
    f3=cv.fit_matrix(raw)
    f2=b.fit_binary_matrix(e)
    te=e.loc[e.scorable]
    normalized=b.conditionalize_cs02(te,f3)
    assert len(normalized)==len(cv.MODELS)
    assert all(x.shape==(len(te),) and np.isfinite(x).all() for x in normalized.values())
    arr=cv.predict_matrix(te,f3)['streak_vol_matrix_3class']
    assert np.allclose(normalized['cs02_normalized_streak_vol_matrix_3class'],arr[:,2]/(arr[:,0]+arr[:,2]))
    assert np.isfinite(b.predict_binary(te,f2)['sign_vol_binary']).all()


def test_binary_score_rejects_zero_and_invalid_probabilities():
    e=fixture(400).loc[lambda g:g.scorable]
    p=np.full(len(e),.5)
    b.score_binary(e,{'p':p})
    with pytest.raises(ValueError):b.score_binary(e.assign(next_sign=0),{'p':p})
    with pytest.raises(ValueError):b.score_binary(e,{'p':np.zeros(len(e))})


def test_binary_random50_baseline_always_ln2_and_quarter():
    e=fixture(1200).loc[lambda g:g.scorable]
    fit=b.fit_binary_matrix(e)
    preds=b.predict_binary(e,fit)
    s,_=b.score_binary(e,preds)
    q=s.set_index('model').loc['random_50']
    assert np.isclose(q.binary_logloss,np.log(2)) and np.isclose(q.binary_brier,.25)


def test_bootstrap_mean_day_and_weighted_ci():
    e=fixture(1500).loc[lambda x:x.scorable]
    fit=b.fit_binary_matrix(e)
    raw=e.copy();raw['scorable']=raw.scorable_threeclass
    fit3=cv.fit_matrix(raw)
    forecasts={**b.predict_binary(e,fit),**b.conditionalize_cs02(e,fit3)}
    _,per=b.score_binary(e,forecasts)
    bb=b.day_bootstrap(per,B=50,seed=2)
    assert len(bb)==2*len(b.PAIRS)
    assert np.isfinite(bb[['delta_mean_day','delta_event_weighted','ci025','ci975']].to_numpy()).all()
    assert all(bb.ci025<=bb.ci975)


def test_matrix_carries_train_counts_and_joint_symmetry():
    e=fixture(1000);fit=b.fit_binary_matrix(e)
    m=b.fitted_probability_matrix(fit)
    assert m.n_train.sum()==fit['n_train']
    assert np.allclose(m.p_up_nonzero+m.p_down_nonzero,1)
    assert np.allclose(m.p_continue_nonzero+m.p_reverse_nonzero,1)
    assert set(m.sign)=={-1,1}


def test_shuffling_vol_does_not_use_test_outcomes_in_predictors():
    e=fixture(1200);fit=b.fit_binary_matrix(e.iloc[:700]);te=e.iloc[700:]
    a=b.shuffle_volatility_null(te,fit,B=8)
    assert a['B']==8 and np.isfinite(a['actual_signvol_minus_sign'])
    assert 0<=a['fraction_null_le_actual']<=1


def test_reject_bad_categories_or_zero_training_labels():
    e=fixture(1000)
    broken=e.copy();broken.loc[broken.scorable,'next_sign']=0
    with pytest.raises(ValueError):b.fit_binary_matrix(broken)
    broken=e.copy();broken['level_bin']=3
    with pytest.raises(ValueError):b.fit_binary_matrix(broken)
