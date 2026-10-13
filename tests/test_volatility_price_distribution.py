"""Tests for VD13; never read user market data or external services."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data_engineering.modeling import volatility_price_distribution as vp


def synthetic(days=35, assets=('BTC','ETH'), seed=13, reverse=False):
    rng=np.random.default_rng(seed)
    dates=pd.date_range('2025-01-01',periods=days*288,freq='5min',tz='UTC')
    rows=[]
    for j,asset in enumerate(assets):
        eps=rng.standard_normal(len(dates))
        r=np.zeros(len(dates))
        for t in range(1,len(dates)):
            clock=0.001*(1+.25*np.cos(t*2*np.pi/288))
            r[t]=clock*eps[t]-(.33*r[t-1] if reverse else 0)
        frame=pd.DataFrame({'state_time':dates,'underlying':asset,'return_5m':r})
        rows.append(frame)
    return pd.concat(rows,ignore_index=True)


def test_missing_bar_is_not_interpolated_and_causal():
    raw=synthetic(35,assets=('BTC',))
    ix=35*288-800
    raw.loc[ix,'return_5m']=np.nan
    p=vp.prepare_panel(raw)
    f,clock=vp.make_features(p)
    assert len(clock)==24
    assert np.isnan(f.loc[ix,'now'])
    assert not f.loc[ix,'history_ok']
    assert np.isfinite(f.loc[ix+1,'fast'])
    assert np.isfinite(f.loc[ix+1,'slow'])
    # Modify next bar: features at t must remain identical.
    changed=raw.copy()
    changed.loc[ix+2,'return_5m']+=.2
    g,_=vp.make_features(vp.prepare_panel(changed))
    for name in vp.PHASE+vp.INTERACTION:
        av=f.loc[ix+1,name]; bv=g.loc[ix+1,name]
        assert (np.isnan(av) and np.isnan(bv)) or np.isclose(av,bv)
    fw,_=vp.add_forward_targets(p,f,(60,))
    assert np.isnan(fw.loc[ix-3,'R_60'])  # future outage censored
    assert np.isnan(fw.loc[ix-3,'tail_60'])


def test_future_never_crosses_splits():
    p=vp.prepare_panel(synthetic(35,assets=('BTC',)))
    f,_=vp.make_features(p)
    out,_=vp.add_forward_targets(p,f,(15,60))
    j=p.index[(p.split=='valid')&(p.underlying=='BTC')][0]
    assert np.isnan(out.loc[j-1,'R_15'])
    assert np.isnan(out.loc[j-5,'RV_60'])
    assert np.isfinite(out.loc[j,'R_60'])


def test_two_assets_independent_and_thresholds_training_only():
    p=vp.prepare_panel(synthetic())
    f,_=vp.make_features(p)
    targets,q=vp.add_forward_targets(p,f,(15,60,240))
    assert set(q['60'])=={'BTC','ETH'}
    altered=p.copy()
    altered.loc[altered.split=='test','return_5m']*=6.0
    tf,_=vp.make_features(altered)
    _,q2=vp.add_forward_targets(altered,tf,(15,60,240))
    assert q==q2
    assert targets.loc[targets.split=='train','tail_60'].mean()>0


def test_end_to_end_common_origin_scores_and_day_paired_ci():
    p=vp.prepare_panel(synthetic(days=35,reverse=True))
    f,_=vp.make_features(p)
    frame,threshold=vp.add_forward_targets(p,f)
    scores,preds=vp.fit_horizon(frame,60)
    assert set(scores.model)=={'K_random_walk','E_two_ema','P_vol_phase',
       'M_price_reversion','PM_conditional'}
    assert len(scores)==5 and scores.n_origins.nunique()==1
    assert all(np.isfinite(scores.nll))
    assert (scores.tail_logloss>0).all()
    b=vp.paired_day_bootstrap(preds,('P_vol_phase','E_two_ema'),
                              'tail_logloss',60,B=31)
    assert b['test_days']==scores.n_days.min()
    assert b['q025']<=b['q975']
    dec=vp.decide(scores,pd.DataFrame([b]),frame)
    assert dec['status'] in ('INSUFFICIENT_DATA','HISTORICAL_EXPLORATORY')
    assert dec['historical_confirmation_permitted'] is False


def test_perfectly_flat_price_is_well_defined():
    z=np.r_[np.nan,np.zeros(30)]
    v=vp._ewma_observed(z,1-np.exp(-np.log(2)*5/30))
    assert np.isnan(v[0]) and np.isclose(v[1:],0).all()
    with pytest.raises(ValueError):
        vp.prepare_panel(pd.DataFrame({'underlying':['BTC']*2,
        'state_time':['2025-01-01T00:00:00Z']*2,
        'return_5m':[0.,1.]}))
