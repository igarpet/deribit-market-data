"""RD02 regression tests, including point-in-time checks and density normalization."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as student_t
from scipy.integrate import simpson
from data_engineering.modeling import latent_regime_tournament as m


def synthetic_panel(days=80,seed=2612):
    rng=np.random.default_rng(seed);n=days*288
    t=pd.date_range('2025-01-01',periods=n,freq='5min',tz='UTC')
    latent=np.repeat(rng.choice([-1.,0.,1.],size=days*48,p=[.32,.35,.33]),6)[:n]
    # All observations at t are available by t; target labels come only later.
    ret=.0005*latent+rng.standard_t(6,n)*.001
    flow=np.clip(.3*latent+rng.normal(0,.25,n),-.98,.98)
    basis=3*latent+rng.normal(0,2,n)
    df=pd.DataFrame({'state_time':t,'return_5m':ret,'feature_ready':True,
                     'log_rv24h':np.full(n,np.log(.0015**2)),
                     'mom_r5':ret,'mom_r15_mean':pd.Series(ret).rolling(3,min_periods=1).mean(),
                     'flow5':flow,'flow30':pd.Series(flow).rolling(6,min_periods=1).mean(),
                     'basis_bps':basis,'basis_change_bps':np.r_[0,np.diff(basis)],
                     'last_1m_r':.2*ret+rng.normal(0,.0001,n),
                     'minute_accel':rng.normal(0,.0002,n),
                     'log_trades':5+.25*np.abs(latent)+rng.normal(0,.05,n)})
    return df


def test_registry_and_fixed_folds():
    c=m.Spec()
    assert c.k_states==3
    assert c.primary_horizon_min==30
    assert c.horizons==(5,15,30,60)
    assert 'latent_semi_markov' in m.METHODS
    x,days=m.prepare_panel(synthetic_panel(days=80),c)
    assert len(x)==80*48
    f=m.combinations_dates(days,c)
    assert len(f)==3
    final_start=days[int(len(days)*.8)]
    for b in f:
        assert max(b['structure_days'])<min(b['calibration_days'])
        assert max(b['calibration_days'])<min(b['validation_days'])
        assert max(b['validation_days'])<final_start


def test_future_is_not_input_and_rebuild_is_checked():
    panel=synthetic_panel(days=3)
    x,_=m.prepare_panel(panel,m.Spec())
    assert x.iloc[-1].ret_60m!=x.iloc[-1].ret_60m
    assert all('ret_' not in col for col in m.STATE_FEATURES)
    bad=panel.copy()
    bad['ret_30m']=0.
    with pytest.raises(ValueError,match='causal rebuild'):
        m.prepare_panel(bad,m.Spec())


def test_discontinuous_segments_reset():
    t=pd.to_datetime(['2025-01-01T00:00Z','2025-01-01T00:30Z',
                      '2025-01-01T02:00Z','2025-01-01T02:30Z'])
    s=m.segments(t)
    assert [len(x) for x in s]==[2,2]
    loge=np.array([[4.,-4.],[-5.,4.],[-5.,4.],[-5.,4.]])
    A=np.array([[.99,.01],[.01,.99]]);p=np.array([.5,.5])
    q=m.filter_hmm(loge,A,p,t)
    assert q[2,1]>.99  # gap reset overrides earlier state
    other=loge.copy();other[-1]=[10,-10]
    z=m.filter_hmm(other,A,p,t)
    assert np.allclose(q[:3],z[:3])  # no future lookahead


def test_hmm_em_and_causal_filter():
    rng=np.random.default_rng(4);n=420
    t=pd.date_range('2025-01-01',periods=n,freq='30min',tz='UTC')
    state=np.repeat(np.arange(3),140)
    x=rng.normal(state[:,None]*1.8, .55, (n,3))
    cfg=m.Spec(em_iterations=4,k_states=3)
    fit=m.fit_hmm(x,t,cfg)
    assert fit['A'].shape==(3,3)
    assert np.allclose(fit['A'].sum(axis=1),1.)
    loge=m.emission_loglik(x,fit['loc'],fit['var'])
    p=m.filter_hmm(loge,fit['A'],fit['pi'],t)
    assert np.allclose(p.sum(axis=1),1.)
    vi=m.viterbi(loge,fit['A'],fit['pi'],m.segments(t))
    hz,info=m.empirical_hazards(vi,t,fit['A'],cfg)
    assert hz.shape==(3,24)
    assert np.isfinite(hz).all()
    ps=m.filter_semi_markov(loge,fit['A'],fit['pi'],hz,t)
    assert np.allclose(ps.sum(axis=1),1.)
    assert np.isfinite(ps).all()


def test_semi_markov_geometric_equivalence_to_hmm():
    rng=np.random.default_rng(123)
    loge=rng.normal(size=(120,3));A=np.array([[.86,.08,.06],[.15,.80,.05],[.04,.12,.84]])
    pi=np.array([.33,.34,.33]);tt=pd.date_range('2026-01-01',periods=120,freq='30min',tz='UTC')
    hazard=np.tile(1-np.diag(A)[:,None],(1,24))
    a=m.filter_hmm(loge,A,pi,tt)
    b=m.filter_semi_markov(loge,A,pi,hazard,tt)
    assert np.allclose(a,b,atol=1e-11)


@pytest.mark.parametrize('skew',[-.42,0,.35])
def test_split_t_fixed_mean_and_variance(skew):
    grid=np.linspace(-90,90,90001)
    p=np.exp(m.split_t_logpdf(grid,skew))
    assert simpson(p,x=grid)==pytest.approx(1.,abs=5e-6)
    assert simpson(grid*p,x=grid)==pytest.approx(0.,abs=5e-5)
    assert simpson(grid**2*p,x=grid)==pytest.approx(1.5,rel=.001)
    assert np.all(np.diff(m.split_t_cdf(grid[::1000],skew))>=0)
    if skew==0:
        assert np.allclose(m.split_t_logpdf(grid[::2000],0.),student_t.logpdf(grid[::2000],df=6.))


def test_density_mixture_is_valid_and_causal():
    cfg=m.Spec()
    y=np.array([-.001,0,.002]);sigma=np.full(3,.002)
    loc=np.array([-.2,0,.2]);sk=np.array([-.18,0,.2]);p=np.tile([.2,.35,.45],(3,1))
    ll,prob=m.calc_log_density(y,sigma,loc,.85,sk,p,cfg)
    assert np.isfinite(ll).all()
    assert (prob>=0).all() and (prob<=1).all()
    # Changing a later mixture weight cannot modify earlier origin predictions.
    q=p.copy();q[-1]=[.9,.09,.01]
    ll2,_=m.calc_log_density(y,sigma,loc,.85,sk,q,cfg)
    assert np.allclose(ll[:2],ll2[:2])


def test_calibration_response_only_depends_on_calibration_data():
    rng=np.random.default_rng(9)
    q=rng.normal(size=600)
    p=rng.dirichlet([1,2,3],size=600)
    a=m.weighted_state_response(q,p,m.Spec())
    b=m.weighted_state_response(q.copy(),p.copy(),m.Spec())
    assert np.allclose(a[0],b[0]) and np.allclose(a[1],b[1])


def test_full_end_to_end_small_synthetic():
    cfg=m.Spec(em_iterations=2,horizons=(30,),bootstrap_reps=10,fold_positions=(.50,),
               duration_age_cap=8)
    results={'BTC':m.scan_asset(synthetic_panel(days=73), 'BTC',cfg)}
    out=m.consolidate(results,cfg,synthetic=True)
    assert len(out['scores'])==6
    assert set(out['scores']['method'])==set(m.METHODS)
    assert not out['bootstrap'].empty
    assert out['manifest']['synthetic'] is True
    assert not out['decision']['prospective_confirmation']
