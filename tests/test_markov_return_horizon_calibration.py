"""VD16 scientific tests, requiring no private market data."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import logsumexp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data_engineering.modeling import markov_lift_distributions as v14
from data_engineering.modeling import student_t_return_markov as v15
from data_engineering.modeling import markov_return_horizon_calibration as v16


def synthetic(days=32,assets=('BTC','ETH'),seed=23):
    rng=np.random.default_rng(seed)
    ts=pd.date_range('2025-01-01',periods=days*288,freq='5min',tz='UTC')
    out=[]
    for j,asset in enumerate(assets):
        rv=np.zeros(len(ts));lv=-13.5
        for i in range(1,len(ts)):
            lv=.992*lv+.008*(-13.4+j*.22)+.02*rng.normal()
            rv[i]=np.exp(lv/2)*rng.standard_t(6)*np.sqrt((6-2)/6)
        out.append(pd.DataFrame({'underlying':asset,'state_time':ts,'return_5m':rv}))
    return pd.concat(out,ignore_index=True)


def test_identity_shape_and_scale_and_variance_preservation():
    rng=np.random.default_rng(1)
    v=np.exp(rng.normal(size=(18,64)))
    assert np.allclose(v16.transform_variances(v,0.,1.),v,rtol=1e-12)
    out=v16.transform_variances(v,np.log(1.5),1.7)
    assert np.allclose(out.mean(axis=1),v.mean(axis=1)*1.5,rtol=1e-12)
    # a=0 is Gaussian-like in the limit; a>1 has greater particle dispersion
    narrow=v16.transform_variances(v,0.,.5)
    wide=v16.transform_variances(v,0.,1.6)
    assert np.mean(wide.var(axis=1))>np.mean(narrow.var(axis=1))


def test_density_normalization_for_calibration():
    from scipy.integrate import trapezoid
    v=np.array([[.3,.6,1.4,3.6]])
    for shape in (.5,1.,1.6):
        x=np.linspace(-20,20,16001)
        vv=v16.transform_variances(v,.1,shape)
        lp=-.5*(np.log(2*np.pi*vv)+x[:,None]**2/vv)
        pdf=np.exp(logsumexp(lp,axis=1)-np.log(4))
        assert np.isclose(trapezoid(pdf,x),1,atol=1e-6)
        assert np.isclose(trapezoid(pdf*x*x,x),vv.mean(),atol=1e-5)


def test_native_amt_stays_identical_to_vd15():
    state=np.array([[.02,.1],[-.05,.2]])
    trans={'mean':[0.,0.], 'phi':[.99,.9], 'innovation_cov':[[.003,.002],[.002,.02]]}
    kernel={'phi':0.,'scale':.000003,'nu':6.}
    v=list(v16.variance_particles(state,trans,kernel,3,64,143,block_size=2))[0][1]
    y=np.array([.002,-.007]);q=np.array([.01,.01]);prev=np.array([.001,-.002])
    ref=v15._integrate_student(state,prev,trans,ar=0,scale=kernel['scale'],df=6.,steps=3,
        particles=64,seed=143,realized=y,q=q)
    fresh=v16.predict_loss(v,y,q)
    for col in ('nll','tail_logloss','tail_brier','tail_predicted','interval_miss95'):
        new_name='tail_probability' if col=='tail_predicted' else col
        assert np.allclose(fresh[col],ref[new_name],rtol=1e-11,atol=1e-12)


def test_validation_embargo_and_subsampling_are_causal():
    p=v14.panelize(synthetic(days=50,assets=('BTC',)))
    f=v14.make_targets(p,v14.make_features(p),horizons=(240,))
    _,va,_=v14._get_split(f,240)
    a,b,cut=v16.split_validation(va,240,max_per_asset=30)
    assert (a.state_time+pd.Timedelta(minutes=240)<=pd.Timestamp(cut)).all()
    assert (b.state_time>=pd.Timestamp(cut)).all()
    assert a.state_time.max()<b.state_time.min()


def test_constrained_optimization_keeps_families_distinct():
    rng=np.random.default_rng(16)
    v=np.exp(rng.normal(size=(180,32)))
    y=rng.normal(0,.8,180)
    for family in v16.FAMILIES:
        fit=v16.fit_calibration(v,y,family,max_iter=12)
        assert np.isfinite(fit['fit_nll'])
        assert -.70001<=fit['log_scale']<=.70001
        assert .44999<=fit['shape']<=2.00001
        if family=='AM_T_SCALE':assert fit['shape']==1
        if family=='AM_T_SHAPE':assert fit['log_scale']==0
        if family=='AM_T':assert fit['log_scale']==0 and fit['shape']==1


def test_end_to_end_seven_horizons_common_origins():
    p=v14.panelize(synthetic(days=50))
    f=v14.make_targets(p,v14.make_features(p),horizons=v16.HORIZONS)
    scores,bs,per_asset,cal_grid,parameters=v16.run_all(
        f,particles=16,B=12,cal_limit=20,chunk_size=512)
    assert len(scores)==7*len(v16.MODEL_NAMES)
    assert len(per_asset)==7*2*len(v16.MODEL_NAMES)
    assert len(cal_grid)==7*len(v16.FAMILIES)
    assert len(bs)==7*len(v16.PAIRS)*len(v16.METRICS)
    assert (scores.groupby('horizon_min').origins.nunique()==1).all()
    assert (scores.test_days>=5).all()
    assert np.isfinite(scores[['nll','tail_logloss','tail_brier','interval_miss95']].to_numpy()).all()
    assert parameters['version']==v16.VERSION
    assert set(parameters['calibrations'])==set(map(str,v16.HORIZONS))
    assert (scores.interval_miss95.between(0,1)).all()
