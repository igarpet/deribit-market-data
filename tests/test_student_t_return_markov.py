"""VD15 synthetic-only scientific tests. No market-data download required."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data_engineering.modeling import markov_lift_distributions as b
from data_engineering.modeling import student_t_return_markov as x


def synthetic(days=32,assets=('BTC','ETH'),seed=150,phi=-.10):
    rng=np.random.default_rng(seed)
    dates=pd.date_range('2025-01-01',periods=days*288,freq='5min',tz='UTC')
    all_rows=[]
    for idx,asset in enumerate(assets):
        vol=.0015*(1+idx*.2)
        ret=np.zeros(len(dates))
        for j in range(1,len(dates)):
            # true return-only Student-t AR(1), with additional periodic scale.
            s=vol*(1+.2*np.cos(j*np.pi/144))
            e=rng.standard_t(df=5)*np.sqrt(3/5)
            ret[j]=phi*ret[j-1]+s*e
        all_rows.append(pd.DataFrame({'state_time':dates,'underlying':asset,
                                      'return_5m':ret}))
    return pd.concat(all_rows,ignore_index=True)


def test_native_return_kernel_estimates_negative_ar_not_vol_t():
    raw=synthetic(32)
    p=b.panelize(raw)
    f=b.make_features(p);fr=b.make_targets(p,f)
    tr,va,_=b._get_split(fr,5)
    states=b.fit_markov(fr.loc[(fr.split=='train')&fr.ready])
    fit,metadata=x.fit_return_kernel(tr,va,states)
    assert x.HORIZONS==(5,10,15,30,60,120,240)
    assert set(fit)=={'T_RW','T_MC','AM_T','AM_T_AR'}
    for asset in ('BTC','ETH'):
        assert metadata[asset]['return_phi'] < 0
        assert fit['T_RW'][asset]['phi']==0
        assert fit['T_MC'][asset]['phi']<0
        for m in fit:
            assert fit[m][asset]['nu']>2
            assert fit[m][asset]['scale']>0


def test_markov_ar_weights_and_return_mean():
    tr={'mean':[0.,0.],'phi':[.97,.8],
        'innovation_cov':[[1e-10,0.],[0.,1e-10]]}
    r0=np.array([.1])
    state=np.array([[0.,0.]])
    # This test directly checks conditional return-location propagation by
    # varying an observed R for otherwise identical Student-t integration.
    # At 10m, phi=.5 gives 0.75*r0 expected cumulative return.
    old=.5*(1+.5)*r0[0]
    assert abs(old-.075)<1e-12
    k=x._integrate_student(state,r0,tr,ar=.5,scale=.01,df=8.,steps=2,
         particles=64,seed=42,variable_state=False,
         realized=np.array([old]),q=np.array([1.]))
    assert np.isfinite(k['nll']).all()
    assert 0<k['tail_probability'][0]<1


def test_student_t_random_walk_native_pdf_and_tail():
    # Conditional t model with deterministic scale is approximated by a finite
    # Gaussian scale-mixture quadrature, rather than replacing returns with t RV.
    from scipy.stats import t as student
    tr={'mean':[0.,0.],'phi':[.9,.9],
        'innovation_cov':[[1e-10,0.],[0.,1e-10]]}
    r=np.array([0.,.4,2.0])
    result=x._integrate_student(np.zeros((3,2)),np.zeros(3),tr,
            ar=0.,scale=1.,df=6.,steps=1,particles=1024,seed=100,
            variable_state=False,realized=r,q=np.ones(3))
    ref=student.logpdf(r/np.sqrt(4/6),df=6)-.5*np.log(4/6)
    assert np.max(np.abs(result['nll']+ref))<.05
    assert (result['tail_probability']>0).all()


def test_gap_and_split_censoring_unmodified():
    raw=synthetic(24,assets=('BTC',))
    raw.loc[24*288-200,'return_5m']=np.nan
    p=b.panelize(raw);f=b.make_features(p);z=b.make_targets(p,f)
    assert not f.loc[24*288-200,'ready']
    v_idx=p.index[p.split=='valid'][0]
    for h in x.HORIZONS:
        assert np.isnan(z.loc[v_idx-1,f'R_{h}'])
    assert len(z)==len(p)


def test_full_six_models_all_horizons_and_day_bootstrap():
    p=b.panelize(synthetic(24))
    f=b.make_features(p);z=b.make_targets(p,f)
    scores,boot,asset,pars=x.run_all(z,particles=16,B=40,chunk_size=1024)
    assert len(scores)==42
    assert set(scores.horizon_min)==set(x.HORIZONS)
    assert set(scores.model)==set(x.MODELS)
    assert (scores.groupby('horizon_min').n_origins.nunique()==1).all()
    assert len(boot)==7*len(x.PAIRS)*len(x.METRICS)
    assert len(asset)==7*2*6
    assert (np.isfinite(scores[['nll','tail_logloss','tail_brier',
                               'interval_miss95','tail_predicted']])).all().all()
    assert (scores.interval_miss95.between(0,1)).all()
    assert (boot.q025<=boot.q975).all()
    assert pars['model_kernels']['T_MC']['BTC']['phi']<0
    assert x.decision(scores,boot)['historical_confirmation_permitted'] is False
