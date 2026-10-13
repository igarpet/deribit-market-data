"""Synthetic scientific checks for VD14; no external market data."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data_engineering.modeling import markov_lift_distributions as v


def synthetic(days=32,assets=('BTC','ETH'),seed=14):
    rng=np.random.default_rng(seed)
    ts=pd.date_range('2025-01-01',periods=days*288,freq='5min',tz='UTC')
    frames=[]
    for j,a in enumerate(assets):
        vol=np.empty(len(ts));vol[0]=.001*(j+1)
        noise=rng.standard_normal(len(ts))
        r=np.zeros(len(ts))
        for t in range(1,len(ts)):
            vol[t]=np.clip(.985*vol[t-1]+.015*(.0015*(j+1))+.00004*rng.standard_normal(),.0004,.008)
            r[t]=vol[t]*noise[t]
        frames.append(pd.DataFrame({'state_time':ts,'underlying':a,'return_5m':r}))
    return pd.concat(frames,ignore_index=True)


def test_all_seven_horizons_and_split_censoring():
    p=v.panelize(synthetic(days=24,assets=('BTC',)))
    f=v.make_features(p);z=v.make_targets(p,f)
    assert v.HORIZONS==(5,10,15,30,60,120,240)
    j=p.index[p.split=='valid'][0]
    for h in v.HORIZONS:
        assert np.isnan(z.loc[j-1,f'R_{h}'])
        assert np.isfinite(z.loc[j,f'R_{h}'])
        assert np.isnan(z.loc[j-1,f'RV_{h}'])
    assert f.loc[j,'ready']


def test_gap_handling_and_no_future_feature_leakage():
    raw=synthetic(24,('BTC',)); gap=20*288+11
    raw.loc[gap,'return_5m']=np.nan
    p=v.panelize(raw);f=v.make_features(p);z=v.make_targets(p,f)
    assert np.isnan(f.loc[gap,'now'])
    assert not f.loc[gap,'ready']
    assert np.isfinite(f.loc[gap+1,'fast'])
    assert np.isnan(z.loc[gap-1,'R_5'])
    assert np.isnan(z.loc[gap-12,'R_120'])
    altered=raw.copy();altered.loc[gap+2,'return_5m']+=0.04
    g=v.make_features(v.panelize(altered))
    for col in ['now','fast','slow','gap','slope15']:
        a=f.loc[gap+1,col];b=g.loc[gap+1,col]
        assert (np.isnan(a) and np.isnan(b)) or np.isclose(a,b)


def test_gaussian_mixture_and_student_are_valid_densities():
    from scipy.integrate import trapezoid
    x=np.linspace(-20,20,30001)
    pdf=np.exp(v.mix_logpdf(x,1.0,0.09,5))
    assert np.isclose(trapezoid(pdf,x),1.,atol=3e-6)
    assert np.isclose(trapezoid(pdf*x*x,x),1.,atol=1e-5)
    assert np.isclose(v.mix_abs_prob(1.,1.,.09,5),2*trapezoid(pdf[x>=1],x[x>=1]),atol=3e-4)
    for df in v.DF_GRID:
        assert np.isfinite(v.t_logpdf(np.array([0.,3.]),np.array([1.,1.]),df)).all()


def test_markov_expected_variance_and_hazard_are_causal():
    fit={'mean':[0.,0.],'phi':[.98,.7],'innovation_cov':[[.0001,0.],[0.,.001]]}
    now=np.array([[.0,.5],[.0,-.5]])
    a=v._markov_rv_path(now,fit,1)
    b=v._markov_rv_path(now,fit,12)
    assert a[0]>a[1] and (b>a).all()
    phase=pd.DataFrame({'slow':np.linspace(-1,1,800),'gap':np.linspace(-.2,.2,800)})
    rg=np.random.default_rng(4)
    y=rg.standard_normal(800)
    var=np.ones(800)
    vfit=v.fit_conditional_mix(y,var,phase,5.,.01,cap=800)
    p=v.conditional_mix_prob(vfit,phase)
    assert np.all((p>0)&(p<.5))


def test_complete_end_to_end_every_model_and_paired_bootstrap():
    p=v.panelize(synthetic(days=32))
    f=v.make_features(p)
    z=v.make_targets(p,f)
    scores,boot,meta,preds=v.run_all(z,horizons=(5,10,15,30,60,120,240),B=50,store_predictions=True)
    assert len(scores)==35
    assert set(scores.model)==set(v.MODELS)
    assert set(scores.horizon_min)==set(v.HORIZONS)
    assert (scores.n_origins>100).all()
    assert (scores.n_days>=6).all()
    assert np.isfinite(scores[['nll','rv_log_mse','tail_logloss','tail_brier','interval_miss95']].to_numpy(float)).all()
    assert (scores.interval_miss95.between(0,1)).all()
    assert (scores.groupby('horizon_min').n_origins.nunique()==1).all()
    assert meta['AM_fitted_5m_hazard']['ratio_5m'] in v.SHOCK_GRID
    assert len(boot)==len(v.HORIZONS)*len(v.PAIRS)*len(v.METRICS)
    for h in (5,60,240):
        bt=v.paired_day_bootstrap(preds,'AM_markov_lift','M4_conditional_mix',h,'nll',B=100,seed=2)
        assert np.isfinite([bt['delta'],bt['q025'],bt['q975']]).all()
        assert bt['q025']<=bt['q975']
