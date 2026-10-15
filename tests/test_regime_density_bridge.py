"""RD03 causal integration and proper mixture-density regression tests."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data_engineering.modeling import regime_density_bridge as m


def test_zero_shift_recovers_normal_pdf_exactly():
    y=np.array([0.0,0.002,-0.003]);v=np.array([[.01,.04],[.03,.03],[.01,.05]])
    s=m.normal_mixture_scores(y,v,np.zeros(len(y)))
    pdf=(np.exp(-y[:,None]**2/(2*v))/np.sqrt(2*np.pi*v)).mean(axis=1)
    np.testing.assert_allclose(s['nll'],-np.log(pdf),rtol=1e-12)
    np.testing.assert_allclose(s['up_probability'],.5,rtol=1e-12)
    np.testing.assert_allclose(s['variance'],v.mean(axis=1),rtol=0)


def test_mean_shift_preserves_variance_and_probability_direction():
    rng=np.random.default_rng(4)
    v=rng.uniform(1e-7,1e-5,(101,16));y=rng.normal(0,.001,101)
    x=m.normal_mixture_scores(y,v,np.zeros(101))
    plus=m.normal_mixture_scores(y,v,np.ones(101)*.0004)
    minus=m.normal_mixture_scores(y,v,np.ones(101)*-.0004)
    np.testing.assert_array_equal(plus['variance'],x['variance'])
    assert np.all(plus['up_probability']>.5)
    assert np.all(minus['up_probability']<.5)


def test_single_particle_matches_gaussian_identity():
    y=np.array([0.002,-.001]);v=np.array([[.004**2],[.005**2]])
    mu=np.array([.001,.002]);s=m.normal_mixture_scores(y,v,mu)
    expectation=.5*(np.log(2*np.pi*v[:,0])+(y-mu)**2/v[:,0])
    np.testing.assert_allclose(s['nll'],expectation,rtol=1e-13)
    np.testing.assert_allclose(s['up_probability'],ndtr(mu/np.sqrt(v[:,0])))


def test_invalid_density_inputs_rejected():
    with pytest.raises(ValueError):m.normal_mixture_scores(np.zeros(2),np.ones((1,4)),np.zeros(2))
    with pytest.raises(ValueError):m.normal_mixture_scores(np.zeros(2),-np.ones((2,4)),np.zeros(2))
    with pytest.raises(ValueError):m.normal_mixture_scores(np.zeros(2),np.ones((2,4)),np.array([0,np.nan]))


def test_archived_manifest_fails_closed():
    cfg=m.Specification()
    with pytest.raises(ValueError):m.confirm_parameters({'module':'wrong'},cfg)
    with pytest.raises(ValueError):m.confirm_parameters({
        'module':'Volatility_Dynamics_18','registration':{'version':'VD18-0.1.0'},
        'sample_last':'2026-10-01T00:00:00Z','parameters':{}},cfg)
    good={'module':'Volatility_Dynamics_18','registration':{'version':cfg.baseline_version},
          'sample_last':'2026-09-18T23:55:00Z',
          'parameters':{'fitted_har':{},'vd17_state_fits':{},'base_cal':{},
           'native_return_kernels':{'AM_T':{}},'volatility_transition':{}}}
    with pytest.raises(ValueError,match='missing horizon'):m.confirm_parameters(good,cfg)
    for h in cfg.horizons:
        good['parameters']['fitted_har'][str(h)]={'overlay':{'selected_har':'HAR_2'},'har_fits':{'HAR_2':{}}}
        good['parameters']['base_cal'][str(h)]={}
    assert m.confirm_parameters(good,cfg)['native_return_kernels']=={'AM_T':{}}


def test_causal_period_boundaries():
    cfg=m.Specification()
    frame=pd.DataFrame({'state_time':pd.to_datetime([
       '2026-04-30T23:30Z','2026-05-01T00:00Z',
       '2026-08-31T23:30Z','2026-09-01T00:00Z',
       '2026-09-18T23:30Z','2026-09-19T00:00Z',
       '2026-10-13T23:30Z','2026-10-14T00:00Z'])})
    actual=m.periods(frame,cfg)
    assert actual.rd_period.tolist()==['structure','response','response','diagnostic',
                                     'diagnostic','evaluation','evaluation']


def test_shrunk_residual_mean_and_no_future_labels():
    e=np.array([1.,2.,-4.,2.])
    w=np.array([[1,0],[1,0],[0,1],[0,1]],float)
    np.testing.assert_allclose(m.shrunk_residual_means(e,w,2.),[.75,-.5])
    with pytest.raises(ValueError):m.shrunk_residual_means(e,w*2,2.)


def test_train_only_standardization_and_magnitude_normalization():
    cfg=m.Specification()
    t=pd.date_range('2026-05-01',periods=200,freq='30min',tz='UTC')
    cal=pd.DataFrame({k:np.random.default_rng(i).normal(size=len(t)) for i,k in enumerate(m.RETURN_FEATURES)})
    cal['state_time']=t
    cal['ret_target']=np.random.default_rng(400).normal(0,.001,len(t))
    v=np.full((200,8),4e-6)
    fitted=m.response_location_fit(cal,v,cfg)
    pred=m.response_location_predict(cal,fitted,cfg)
    assert np.all(np.abs(pred)<=cfg.max_shift_sigma)
    shifted=cal.iloc[:3].copy()
    shifted.loc[:,m.RETURN_FEATURES[0]]=1e12
    assert np.all(np.isfinite(m.response_location_predict(shifted,fitted,cfg)))


def test_paired_bootstrap_day_resampling_and_sparse_behavior():
    dates=pd.date_range('2026-09-19',periods=18,freq='D')
    rows=[]
    for day in dates:
        for model,loss in [('AM_T_HAR_BLEND',.8),('sign_only_mean',.797),('direct_mean',.79),
                           ('observable_tree_mean',.795),('latent_static_mean',.81),
                           ('latent_hmm_mean',.805),('latent_semi_markov_mean',.803)]:
            rows.append({'date':day,'method':model,'horizon_min':30,'nll':loss,'up_brier':.25})
    b=m.paired_day_bootstrap(pd.DataFrame(rows),m.Specification(bootstrap_draws=100))
    row=b[(b.horizon_min==30)&(b.method=='direct_mean')&(b.metric=='nll')].iloc[0]
    assert row.days==18 and row.ci975<0 and row.status=='EXPLORATORY'
    assert (b[b.horizon_min!=30].status=='INSUFFICIENT_DAYS').all()


def test_decision_never_declares_confirmation():
    cfg=m.Specification()
    output=m.decision(pd.DataFrame(),pd.DataFrame({'horizon_min':[30], 'metric':['nll'],
        'method':['direct_mean'],'ci975':[-.2], 'asset':['BTC'], 'status':['EXPLORATORY'], 'reference':['AM_T_HAR_BLEND']}),cfg)
    assert output['prospective_confirmation'] is False
    assert output['AM_T_HAR_Blend_modified'] is False
