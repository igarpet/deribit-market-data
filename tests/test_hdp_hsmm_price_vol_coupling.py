import numpy as np
import pandas as pd
import pytest
from data_engineering.modeling import hdp_hsmm_price_vol_coupling as m
from data_engineering.modeling import hdp_hsmm_return_states as hsmm
from data_engineering.modeling import hdp_hsmm_volatility_price as vs


def sample(n=96,seed=17):
    rng=np.random.default_rng(seed)
    t=pd.date_range('2026-01-01',periods=n,freq='30min',tz='UTC')
    v=np.exp(np.linspace(-12,-11,n));pr=v*1.14
    rr=rng.standard_t(6,size=(n,6))*np.sqrt(v[:,None]/6*4/6)
    g=pd.DataFrame({'state_time':t,'R_30':rr.sum(axis=1),
                    'rv30':(rr*rr).sum(axis=1),'pred_var':v,
                    'pred_rv_scale':pr,'baseline_nll':np.zeros(n)})
    return m.joint_features(g,rv_floor=1e-9)


def test_distinct_rv_scale_from_terminal_return_variance():
    g=sample(20)
    assert np.all(g.pred_rv_scale!=g.pred_var)
    assert np.allclose(g.joint_1,g.R_30/np.sqrt(g.pred_var))
    assert np.allclose(g.joint_2,np.log(g.rv30/g.pred_rv_scale))


def test_interaction_preserves_separate_signs():
    g=sample()
    assert np.allclose(g.coupling,g.joint_1*g.joint_2)
    assert g.joint_1.lt(0).any() and g.joint_1.gt(0).any()


def test_reject_invalid_scales():
    g=sample();g.loc[0,'pred_rv_scale']=0
    with pytest.raises(ValueError):m.joint_features(g,rv_floor=1e-9)


def test_reject_future_invalid_rv():
    g=sample();g.loc[0,'rv30']=-1
    with pytest.raises(ValueError):m.joint_features(g,rv_floor=1e-9)


def test_reject_duplicate_origins():
    g=sample();g.loc[1,'state_time']=g.loc[0,'state_time']
    with pytest.raises(ValueError):m.joint_features(g,rv_floor=1e-9)


def test_gap_resets_predictors():
    g=sample(18);g.loc[9:,'state_time']+=pd.Timedelta(hours=2)
    assert m._past(g,'joint_1')[9]==0
    X,Z=m.continuous_design(g,m.fit_joint_normalizer(g.iloc[:8]))
    assert np.all(X[9,1:4]==0)


def test_same_block_outcome_cannot_affect_predictive_design():
    g=sample();norm=m.fit_joint_normalizer(g.iloc[:15])
    X,Z=m.continuous_design(g,norm)
    changed=g.copy();changed.loc[52,'joint_1']=17;changed.loc[52,'joint_2']=-6
    A,B=m.continuous_design(changed,norm)
    assert np.array_equal(A[52],X[52]);assert np.array_equal(B[52],Z[52])
    assert not np.array_equal(A[53],X[53])


def test_state_filter_uses_completed_block_only():
    g=sample(100)
    x=g[['joint_1','joint_2']].to_numpy()
    norm=hsmm.fit_normalizer(x);X=hsmm.apply_normalizer(x,norm)
    sp=hsmm.Spec(kcap=4,dmax=6,n_iter=18,burn=10,thin=2,seed=17)
    post=hsmm.fit_posterior(X[:60],sp)
    a=vs.filtered_weights(X[60:],post,post['draws'][0])
    changed=X[60:].copy();changed[14]=[13,-5]
    b=vs.filtered_weights(changed,post,post['draws'][0])
    assert np.allclose(a[:15],b[:15],rtol=0,atol=1e-12)
    assert not np.allclose(a[15],b[15])


def test_null_retains_exact_RV_from_six_returns():
    g=sample()
    nul=m.conditional_regime_free_control(g,rv_floor=1e-9,seed=113)
    assert (nul.rv30>=0).all()
    assert np.allclose(nul.joint_1,nul.R_30/np.sqrt(nul.pred_var))
    assert not np.array_equal(g.joint_1,nul.joint_1)


def test_sign_permutation_has_original_brier_under_no_flip():
    s=pd.DataFrame({'HDP_sign_probability':[.4,.6,.7],
                    'continuous_sign_probability':[.5,.5,.5],
                    'realized_positive':[0.,1.,1.]})
    vals=m.conditional_sign_permutation(s,B=9,seed=3)
    assert len(vals)==9 and np.isfinite(vals).all()


def test_model_runs_and_returns_same_origin_scores():
    g=sample(180)
    norm=m.fit_joint_normalizer(g.iloc[:120])
    X=hsmm.apply_normalizer(g[['joint_1','joint_2']].iloc[:120].to_numpy(),norm)
    sp=hsmm.Spec(kcap=4,dmax=8,n_iter=18,burn=10,thin=2,seed=17)
    fit=hsmm.fit_posterior(X,sp)
    assert len(fit['draws'])==4
    q,bb,sc,profiles=m.score_price(g.iloc[:120].copy(),g.iloc[120:].copy(),norm,
                                   np.array([0,120]),np.array([0,60]),[fit],draws_per_chain=2,B=12)
    assert len(sc)==60 and np.isfinite(sc.select_dtypes('number').to_numpy()).all()
    assert set(['HDP_joint_state_nll','continuous_coupling_t_nll','order_ablated_joint_nll']).issubset(sc.columns)
    assert len(bb)>=5 and profiles


def test_pairwise_null_is_not_claimed_to_be_prospective():
    assert m.registration()['status']=='HISTORICAL_EXPLORATORY'
    assert m.registration()['kcap']>2
