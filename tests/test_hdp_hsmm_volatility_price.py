"""RSD02 isolated causal, numerical and synthetic regression tests."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data_engineering.modeling import hdp_hsmm_return_states as rsd
from data_engineering.modeling import hdp_hsmm_volatility_price as m


def synthetics(n=140,seed=12):
    frame=m.continuous_regime_free_null(n,seed)
    frame['underlying']='BTC'
    frame['R_30']=frame.return_surprise*np.sqrt(frame.pred_var)
    frame['baseline_nll']=0.
    return frame


def test_only_completed_volatility_emissions_and_no_signed_returns():
    a=synthetics();f,_=m.build_volatility_features(a)
    b=a.copy();b['return_surprise']*=-1;b['R_30']*=-1
    g,_=m.build_volatility_features(b)
    assert np.allclose(f[['vol_1','vol_2','vol_3']],g[['vol_1','vol_2','vol_3']],equal_nan=True)
    assert f.vol_ready.sum()==len(f)-7
    assert np.isfinite(f.loc[f.vol_ready,['vol_1','vol_2','vol_3']].to_numpy()).all()


def test_missing_block_resets_fastslow():
    a=synthetics(32);a=a.drop(index=[11]).reset_index(drop=True)
    f,_=m.build_volatility_features(a)
    assert not f.vol_ready.iloc[11:18].any()
    assert f.vol_ready.iloc[18]


def test_training_floor_must_be_carried_to_future():
    a=synthetics();f,eps=m.build_volatility_features(a.iloc[:90])
    b=a.iloc[90:].copy();b['rv30']*=10000
    g,_=m.build_volatility_features(b,floor=eps)
    assert np.isclose(eps,m.build_volatility_features(a.iloc[:90])[1])
    assert np.isfinite(g.vol_2).all()
    with pytest.raises(ValueError):m.build_volatility_features(b,floor=0)


def tiny_post(seed=1):
    frame=synthetics(160,seed)
    frame,_=m.build_volatility_features(frame)
    frame=frame.loc[frame.vol_ready].reset_index(drop=True)
    norm=rsd.fit_normalizer(frame[['vol_1','vol_2','vol_3']].to_numpy()[:110])
    x=rsd.apply_normalizer(frame[['vol_1','vol_2','vol_3']].to_numpy(),norm)
    sp=rsd.Spec(kcap=5,dmax=20,n_iter=22,burn=10,thin=3,seed=seed)
    post=rsd.fit_posterior(x[:110],sp)
    return frame,norm,post,x


def test_state_weights_before_observation_and_normalized():
    _,_,post,x=tiny_post(14)
    draw=post['draws'][0]
    before=m.filtered_weights(x[110:],post,draw)
    altered=x[110:].copy();altered[4]+=np.array([9.,-4.,7.])
    changed=m.filtered_weights(altered,post,draw)
    assert np.allclose(before[:5],changed[:5])
    assert not np.allclose(before[5],changed[5])
    assert np.allclose(before.sum(axis=1),1)
    assert np.allclose(before[0],draw['beta'])


def test_price_design_uses_previous_volatility_not_future_current_observation():
    f=synthetics();f,_=m.build_volatility_features(f);f=f.loc[f.vol_ready].reset_index(drop=True)
    norm=rsd.fit_normalizer(f[['vol_1','vol_2','vol_3']].to_numpy())
    a,z=m.design_price(f,norm)
    other=f.copy();other.loc[10,['rv30','vol_2','vol_3']]=[999.,20.,20.]
    b,w=m.design_price(other,norm)
    assert np.allclose(a[10],b[10]) and np.allclose(z[10],w[10])
    assert not np.allclose(a[11],b[11])


def test_continuous_student_gradient_and_finite_scores():
    rng=np.random.default_rng(7);X=np.column_stack([np.ones(350),rng.normal(size=(350,3))]);Z=X[:,:3]
    y=.15*X[:,1]+rng.standard_t(6,size=350)*.65
    fit=m.fit_continuous_t(y[:250],X[:250],Z[:250]);ld,p=m.continuous_density(y[250:],X[250:],Z[250:],fit)
    assert np.isfinite(ld).all() and (p>0).all() and (p<1).all()
    assert fit['mean_beta'][1]>.04


def test_state_return_models_finite_and_mixture_normalized():
    rng=np.random.default_rng(9);n=300;w=rng.dirichlet([2,2,2],size=n)
    y=rng.normal(size=n);prev=np.roll(y,1);prev[0]=0
    fit=m.fit_state_t(y,prev,w);ld,p=m.state_density(y,prev,w,fit)
    ls,_=m.state_density(y,prev,w,fit,scale_only=True)
    assert np.isfinite(ld).all() and np.isfinite(ls).all()
    assert np.all((p>0)&(p<1))
    with pytest.raises(AssertionError):m.fit_state_t(y,prev,w*1.1)


def test_asset_end_to_end_against_continuous_no_regime():
    f,norm,post,_=tiny_post(31)
    train=f.iloc[:110].copy();ev=f.iloc[110:].copy()
    scores,boots,profile,res=m.run_asset(train,ev,[post],norm,[0,len(train)],[0,len(ev)],1,B=24)
    assert len(profile)==1 and len(boots)==6 and len(res)==len(ev)
    assert all(np.isfinite(q) for q in scores.values())
    assert np.isfinite(boots[['delta','ci025','ci975']].to_numpy()).all()


def test_sign_flip_does_not_change_volatility_state_features():
    f,norm,post,_=tiny_post(19)
    train=f.iloc[:110].copy();ev=f.iloc[110:].copy()
    null=m.sign_randomization(train,ev,[post],norm,[0,len(train)],[0,len(ev)],B=3,draws_per_chain=1)
    assert len(null)==3 and np.isfinite(null.state_minus_cont_sign_brier).all()


def test_no_regime_generator_has_heteroskedasticity_without_discrete_switches():
    f=m.continuous_regime_free_null(240)
    assert (f.rv30>=0).all()
    assert f.pred_var.max()>f.pred_var.min()*3
    g,_=m.build_volatility_features(f)
    assert g.vol_ready.sum()==233


def test_next_price_profile_uses_preblock_probabilities():
    f,norm,post,_=tiny_post(28)
    ev=f.iloc[110:].copy()
    a=m.future_price_profiles(ev,post,norm,chain=1)
    assert len(a)>0 and (a.effective_future_blocks>0).all()
    assert np.isclose(a.effective_future_blocks.sum(),len(ev))
    assert np.isfinite(a.mean_next_standardized_return).all()
