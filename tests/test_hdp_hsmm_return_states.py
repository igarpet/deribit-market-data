"""Synthetic statistical contract for Bayesian nonparametric return-state discovery."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data_engineering.modeling import hdp_hsmm_return_states as m


def tiny_spec(seed=11):
    return m.Spec(kcap=7,dmax=30,n_iter=24,burn=12,thin=3,seed=seed)


def test_nonoverlap_six_returns_causal_completion():
    r=np.array([[.01,.02,-.01,.03,.01,-.01],[-.01,-.01,.0,.02,.0,.01]])
    v=np.array([.0006,.0003]);x=m.block_features(r,v)
    assert x.shape==(2,3) and np.isfinite(x).all()
    assert x[0,1]>0 and x[1,1]>0
    assert x[0,2]<0  # within-block reversals
    with pytest.raises(ValueError):m.block_features(np.zeros((2,5)),v)
    with pytest.raises(ValueError):m.block_features(r,np.array([0.,1.]))


def test_normalizer_is_train_only():
    rng=np.random.default_rng(5);train=rng.normal(size=(100,3));test=rng.normal(20,2,(15,3))
    normal=m.fit_normalizer(train)
    assert m.fit_normalizer(train)==normal
    t=m.apply_normalizer(test,normal)
    assert t.shape==(15,3) and np.isfinite(t).all()


def test_duration_is_non_geometric_and_normalized():
    x=m._log_nb_durations(np.array([.25,.45]),40,2)
    assert np.allclose(np.exp(x).sum(axis=1),1)
    assert np.argmax(x[0])>0


def test_ffbs_no_diagonal_transitions():
    rng=np.random.default_rng(17);T=80;K=4
    ll=rng.normal(size=(T,K));b=np.array([.4,.3,.2,.1])
    pi=np.full((K,K),1/(K-1));np.fill_diagonal(pi,1e-300)
    logdur=m._log_nb_durations(np.array([.4,.3,.5,.2]),25,2)
    a,tr,prefix=m._ffbs_forward(ll,np.log(pi),np.log(b),logdur)
    assert np.isfinite(a[-1]).all()
    z=m._ffbs_back(a,tr,prefix,np.log(pi),np.log(b),logdur,rng.random((T,3)))
    assert z.shape==(T,) and z.min()>=0 and z.max()<K
    _,states,lens=m._transition_counts(z,K)
    assert np.all(states[1:]!=states[:-1]) and max(lens)<=25


def test_posterior_runs_and_produces_finite_predictions():
    rng=np.random.default_rng(10)
    x=np.vstack([rng.normal(-1,.5,(42,3)),rng.normal(1,.5,(48,3)),
                 rng.normal(-1,.5,(36,3))])
    fit=m.fit_posterior(x,tiny_spec())
    summary=m.posterior_summary(fit)
    assert len(fit['draws'])==4
    assert 1<=summary['K_occupied_median']<=7
    assert np.isfinite(fit['mh_acceptance'])
    for dr in fit['draws']:
        assert dr['labels'].shape==(len(x),)
        assert (dr['pi'].diagonal()==0).all()
        assert np.allclose(dr['pi'].sum(axis=1),1)
        assert np.isfinite(dr['mu']).all()
    pred=m.predict_logscore(x[-35:],fit)
    assert pred.shape==(35,) and np.isfinite(pred).all()


def test_regime_free_heavy_t_control_and_null_density():
    x=m.simulate_price_blocks(150,variance=np.linspace(.5,2,150))
    fit=m.fit_null_t(x[:100])
    score=m.null_logscore(x[100:],fit)
    assert score.shape==(50,) and np.isfinite(score).all()
    assert x.shape==(150,3)


def test_no_regime_does_not_mandate_two_states():
    # We don't impose a K=2 or assert false-perfect recovery under a weak-limit prior.
    # Instead the model must report posterior uncertainty and any ceiling pressure.
    x=m.simulate_price_blocks(105,seed=41)
    fit=m.fit_posterior(m.apply_normalizer(x,m.fit_normalizer(x)),tiny_spec(12))
    s=m.posterior_summary(fit)
    assert isinstance(s['K_histogram'],dict)
    assert s['K_occupied_05_95'][0]<=s['K_occupied_median']<=s['K_occupied_05_95'][1]
    assert s['warning'] in ('NONE','RERUN_HIGHER_KCAP')


def test_return_only_prediction_causally_excludes_current_block_characteristics():
    rng=np.random.default_rng(104)
    x=rng.normal(size=(160,3));fit=m.fit_posterior(x[:110],tiny_spec(18))
    pred=m.predict_return_logscore(x[110:],fit)
    changed=x[110:].copy();changed[-1,1:]+=500
    pred_changed=m.predict_return_logscore(changed,fit)
    assert np.isfinite(pred).all()
    assert np.allclose(pred[-1],pred_changed[-1])  # no within-block feature leakage
    null=m.fit_null_t(x[:110]);q=m.null_return_logscore(x[110:],null)
    assert np.isfinite(q).all()


def test_multiple_independent_contiguous_segments_are_supported():
    rng=np.random.default_rng(11);x=rng.normal(size=(120,3))
    fit=m.fit_posterior(x,tiny_spec(),segments=[0,60,120])
    assert np.isfinite(m.predict_logscore(x,fit,segments=[0,60,120])).all()
    assert np.isfinite(m.predict_return_logscore(x,fit,segments=[0,60,120])).all()
    with pytest.raises(ValueError):m.fit_posterior(x,tiny_spec(),segments=[0,10,120])


def test_age_cap_does_not_force_extra_states_after_48_hours():
    T=220;K=2;D=12
    ll=np.zeros((T,K));ll[:,1]=-12
    pi=np.array([[0.,1.],[1.,0.]])
    beta=np.array([.999999999,.000000001])
    haz=np.full((K,D),.1);haz[0]=1e-9
    z,lik=m._age_ffbs(ll,pi,beta,haz,np.random.default_rng(2).random((T,3)))
    assert np.isfinite(lik)
    assert len(np.unique(z))==1  # ONE STATE must survive beyond computational age cap


def test_duration_hazard_survives_capped_age():
    p=np.array([.1,.3])
    haz=m._duration_hazards(p,24,2)
    assert haz.shape==(2,24)
    assert np.all((haz>0)&(haz<1))
    assert haz[0,-1]<1  # cap is NOT a compulsory switch


def test_three_persistent_price_states_recovered_without_fixed_k():
    rng=np.random.default_rng(81)
    archetypes=np.array([[-3.,-1.,0.],[0.,1.,-1.],[3.,-1.,1.]])
    samples=[]
    for _ in range(4):
        for state in range(3):
            samples.append(archetypes[state]+rng.standard_t(6,size=(35,3))*.30)
    x=np.vstack(samples)
    prior=m.Spec(kcap=9,dmax=40,n_iter=100,burn=50,thin=5,seed=1901)
    result=m.fit_posterior(x,prior)
    summary=m.posterior_summary(result)
    assert summary['K_occupied_median']==3  # three states, no K=3 requested
    assert summary['at_Kcap_probability']==0
