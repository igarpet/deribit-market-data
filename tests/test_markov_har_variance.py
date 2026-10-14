"""VD18 causal HAR-variance checks; completely synthetic, no local market data."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data_engineering.modeling import markov_lift_distributions as v14
from data_engineering.modeling import markov_har_variance as v18
def synthetic(days=50,assets=('BTC','ETH'),seed=17):
    rng=np.random.default_rng(seed)
    idx=pd.date_range('2025-01-01',periods=days*288,freq='5min',tz='UTC')
    out=[]
    for j,a in enumerate(assets):
        ret=np.zeros(len(idx));level=-13+j*.4
        for t in range(1,len(idx)):
            level=.995*level+.005*(-13+j*.4)+.035*rng.standard_normal()
            ret[t]=np.exp(.5*level)*rng.standard_t(6)*np.sqrt(4/6)
        out.append(pd.DataFrame({'state_time':idx,'underlying':a,'return_5m':ret}))
    return pd.concat(out,ignore_index=True)


def make_sample(days=50,assets=('BTC','ETH')):
    p=v14.panelize(synthetic(days,assets))
    f=v14.make_features(p)
    return v18.add_har_lags(v14.make_targets(p,f,horizons=v18.HORIZONS))


def test_har_lags_use_only_current_and_past_bars():
    f=make_sample(50,('BTC',))
    t=600
    original=f.loc[t,[f'har_logrv_{h}' for h in v18.ALL_FEATURES]].to_numpy(float)
    changed=f.copy();changed.loc[t+1,'return_5m']+=.5
    recalc=v18.add_har_lags(changed)
    np.testing.assert_array_equal(original,recalc.loc[t,[f'har_logrv_{h}' for h in v18.ALL_FEATURES]].to_numpy(float))
    assert not np.isclose(recalc.loc[t+1,'har_logrv_30'],f.loc[t+1,'har_logrv_30'])


def test_missing_current_return_blocks_all_affected_lags():
    f=make_sample(50,('BTC',))
    i=1200
    modified=f.copy(); modified.loc[i,'return_5m']=np.nan
    changed=v18.add_har_lags(modified)
    for lookback in v18.ALL_FEATURES:
        n=lookback//5
        assert changed.loc[i:i+n-1,f'har_logrv_{lookback}'].isna().all()
        assert np.isfinite(changed.loc[i+n,f'har_logrv_{lookback}'])


def test_har_prediction_does_not_use_test_targets():
    f=make_sample(50,('BTC',))
    tr,va,te=v18.complete(f,60)
    fit=v18.fit_har(tr,60,v18.FEATURE_SETS['HAR_2'])
    before=v18.predict_har(te,fit)
    altered=te.copy();altered['R_60']*=20;altered['RV_60']*=20
    assert np.array_equal(before,v18.predict_har(altered,fit))
    assert (before>0).all() and np.isfinite(before).all()
    assert fit['train_origins']==len(tr)
    assert fit['train_zero_RV_counts']['BTC']>=0


def test_return_variance_matching_is_train_only():
    f=make_sample(50,('BTC',))
    tr,va,te=v18.complete(f,60)
    fit=v18.fit_har(tr,60,v18.FEATURE_SETS['HAR_2'])
    p=v18.predict_har(te,fit)
    modified=tr.copy();modified['R_60']*=1.7
    other=v18.fit_har(modified,60,v18.FEATURE_SETS['HAR_2'])
    assert other['net_return_scales']['BTC']>fit['net_return_scales']['BTC']
    # Modifying TEST observations does not refit/alter a frozen HAR forecast.
    altered=te.copy();altered['R_60']*=100
    assert np.array_equal(p,v18.predict_har(altered,fit))


def test_har_anchor_ratio_and_proper_density():
    from scipy.integrate import trapezoid
    from data_engineering.modeling import markov_conditional_variance as v17
    v=np.array([[.5,1,2,4],[1,2,3,5]],float)
    base={'log_scale':np.log(1.4),'shape':1.5}
    pred=np.array([9.,1.2])
    ratio=v18._predict_var_ratio(pred,v,base)
    assert np.isfinite(ratio).all()
    warped=v17.component_variances(v,base)*np.exp(np.clip(ratio,-.85,.85))[:,None]
    assert (warped>0).all()
    z=np.linspace(-30,30,10001)
    pdf=np.exp(-.5*(np.log(2*np.pi*warped[0])[None,:]+z[:,None]**2/warped[0][None,:])).mean(axis=1)
    assert abs(trapezoid(pdf,z)-1)<1e-5


def test_end_to_end_all_seven_horizons_and_fair_origins():
    f=make_sample(50,('BTC','ETH'))
    scores,bootstrap,byasset,grid,parameters=v18.run_all(f,particles=16,B=8,cal_limit=12,block=1024)
    assert len(scores)==7*len(v18.MODELS)
    assert len(byasset)==7*2*len(v18.MODELS)
    assert len(grid)==7*2
    assert len(bootstrap)==7*len(v18.PAIRS)*len(v18.METRICS)
    assert (scores.groupby('horizon_min').n_origins.nunique()==1).all()
    assert (scores.n_days>=8).all()
    assert np.isfinite(scores[['nll','tail_logloss','tail_brier','return_qlike','pred_variance']].to_numpy(float)).all()
    assert (scores.pred_variance>0).all()
    assert (scores.interval_miss95.between(0,1)).all()
    assert set(parameters['fitted_har'])==set(map(str,v18.HORIZONS))
    assert grid.groupby('horizon_min').chosen.sum().eq(1).all()
    assert v18.decision(scores,bootstrap,parameters)['status']=='HISTORICAL_EXPLORATORY'
