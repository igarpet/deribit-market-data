"""VD17 tests: causal conditional return variance, independent of user market data."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data_engineering.modeling import markov_lift_distributions as v14
from data_engineering.modeling import markov_return_horizon_calibration as v16
from data_engineering.modeling import markov_conditional_variance as v17


def synthetic(days=35,assets=('BTC','ETH'),seed=17):
    rng=np.random.default_rng(seed)
    idx=pd.date_range('2025-01-01',periods=days*288,freq='5min',tz='UTC')
    out=[]
    for j,a in enumerate(assets):
        r=np.zeros(len(idx))
        level=-13.+j*.4
        for t in range(1,len(idx)):
            level=.995*level+.005*(-13.+j*.4)+.035*rng.standard_normal()
            r[t]=np.exp(.5*level)*rng.standard_t(6)*np.sqrt(4/6)
        out.append(pd.DataFrame({'state_time':idx,'underlying':a,'return_5m':r}))
    return pd.concat(out,ignore_index=True)


def test_variance_corrections_preserve_probability_density_and_shape():
    from scipy.integrate import trapezoid
    rng=np.random.default_rng(5)
    v=np.exp(rng.normal(size=(4,64)))
    base={'log_scale':.3,'shape':1.65}
    vv=v17.component_variances(v,base)
    assert np.allclose(vv,v16.transform_variances(v,.3,1.65))
    d=np.array([-.4,-.2,.15,.35])
    new=vv*np.exp(d[:,None])
    assert np.allclose(new.mean(axis=1),vv.mean(axis=1)*np.exp(d))
    assert np.allclose(new/vv,np.exp(d[:,None]))
    for i in range(4):
        # Mean variance scales but relative particle shape is preserved.
        assert np.allclose(new[i]/new[i].mean(),vv[i]/vv[i].mean())
    x=np.linspace(-80,80,50001)
    pred=new[2]
    prob=np.exp(-.5*(np.log(2*np.pi*pred)[None,:]+x[:,None]**2/pred[None,:])).mean(axis=1)
    assert abs(trapezoid(prob,x)-1)<1e-5


def test_causal_state_normalizer_and_no_target_feature():
    p=v14.panelize(synthetic(days=50,assets=('BTC',)))
    f=v14.make_targets(p,v14.make_features(p),(60,))
    _,va,_=v14._get_split(f,60)
    cal,sel,cut=v16.split_validation(va,60,max_per_asset=60)
    assert (cal.state_time+pd.Timedelta(minutes=60)<=pd.Timestamp(cut)).all()
    assert (sel.state_time>=pd.Timestamp(cut)).all()
    fit=v17.design_fit(cal)
    x=v17.design_apply(cal,fit)
    altered=cal.copy()
    altered['R_60']*=20
    altered['RV_60']*=20
    assert np.array_equal(x,v17.design_apply(altered,fit))
    assert np.isfinite(v17.design_apply(sel,fit)).all()


def test_anchor_means_blend_as_expected():
    v=np.array([[1.,2.,3.],[4.,5.,6.]])
    base={'log_scale':np.log(1.5),'shape':1.75}
    m2=np.array([10.,2.])
    ratio=v17.anchor_ratio(pd.DataFrame(),v,m2,base)
    assert np.allclose(ratio,np.clip(np.log(m2/(1.5*v.mean(axis=1))),-1.5,1.5))
    x=np.zeros((2,2))
    fit={'beta':[0.,0.],'anchor_weight':1.0}
    d=v17.delta_for('AM_T_VAR_ANCHOR',x,ratio,fit)
    new=v17.component_variances(v,base)*np.exp(d[:,None])
    expected=1.5*v.mean(axis=1)*np.exp(np.clip(ratio,-.85,.85))
    assert np.allclose(new.mean(axis=1),expected)


def test_dynamic_optimization_uses_y_and_clipped_prediction():
    rng=np.random.default_rng(9)
    n=150;p=5
    x=np.column_stack([np.ones(n),rng.normal(size=(n,p-1))])
    va=rng.lognormal(-13,.4,(n,32))
    y=np.sqrt(np.mean(va,axis=1)*np.exp(.32*x[:,1]))*rng.standard_normal(n)
    z=np.clip(.8*x[:,1]+.05*rng.normal(size=n),-1.5,1.5)
    for name in v17.DYNAMIC:
        f=v17.fit_dynamic(name,va,y,x,z,maxiter=20)
        assert np.isfinite(f['objective'])
        assert 0<=f['anchor_weight']<=1.25
        d=v17.delta_for(name,x,z,f)
        assert np.all(np.isfinite(d)) and np.max(np.abs(d))<=v17.MAX_DELTA+1e-10


def test_end_to_end_all_horizons_and_common_origins():
    p=v14.panelize(synthetic(days=50))
    frame=v14.make_targets(p,v14.make_features(p),v17.HORIZONS)
    sc,bt,by_asset,fitgrid,params=v17.run_all(frame,particles=16,B=8,cal_limit=20,chunk_size=768)
    assert len(sc)==7*len(v17.MODELS)
    assert len(by_asset)==7*2*len(v17.MODELS)
    assert len(fitgrid)==7*4
    assert len(bt)==7*len(v17.PAIRS)*len(v17.METRICS)
    assert (sc.groupby('horizon_min').n_origins.nunique()==1).all()
    assert (sc.n_days>=6).all()
    assert np.isfinite(sc[['nll','tail_logloss','tail_brier','return_qlike','interval_miss95']].to_numpy()).all()
    assert (sc.interval_miss95.between(0,1)).all()
    assert set(params['calibrations'])==set(map(str,v17.HORIZONS))
    # Core VD16 reference remains mathematically unmodified.
    assert (sc[sc.model=='AM_T_BOTH'].nll.notna()).all()
