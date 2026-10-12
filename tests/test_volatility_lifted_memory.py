"""Run from repository root: python -m pytest -q tests/test_volatility_lifted_memory.py"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
from data_engineering.modeling import volatility_lifted_memory as lm


def synthetic(days=15, seed=11):
    rng = np.random.default_rng(seed)
    t = pd.date_range('2026-09-01', periods=days*288, freq='5min', tz='UTC')
    frames = []
    for asset in ('BTC', 'ETH'):
        sig = np.exp(.7*np.sin(np.arange(len(t))/49))
        r = rng.standard_t(5, size=len(t))*sig*.001
        r[1000:1008] *= 6
        frames.append(pd.DataFrame({'state_time':t,'underlying':asset,'return_5m':r}))
    return pd.concat(frames,ignore_index=True)


def test_causal_ema_gap_and_history():
    p = pd.DataFrame({'underlying':['AAA']*5,'y':[1.,2.,3.,np.nan,10.]})
    e = lm.ema_prior(p,np.array([5.]))
    assert np.isnan(e[0,0])
    assert e[1,0] == 1.0  # DOES NOT include current y=2
    assert abs(e[2,0]-1.5) < 1e-10
    assert np.isnan(e[3,0])
    assert np.isnan(e[4,0])


def test_quadrature_positive_and_shape():
    x,w = lm.quadrature(.10,6)
    assert np.all(x>0) and np.all(w>0)
    assert np.all(np.diff(x)<0)
    err, *rest = lm.kernel_error(.10,x,w)
    assert np.isfinite(err) and err < 1.0
    p = pd.DataFrame({'underlying':['AAA']*700, 'y':np.sin(np.arange(700)/30)})
    f1,_ = lm.lifted_features(p,.05,6)
    f2,_ = lm.lifted_features(p,.40,6)
    assert np.nanmax(abs(f1.kernel_memory-f2.kernel_memory)) > 1e-5


def test_train_clock_only():
    p = lm.dense_panel(synthetic())
    a, before = lm.training_clock(p)
    q=p.copy()
    q.loc[q.split=='test','return_5m'] *= 100
    q['v']=q.return_5m**2
    b, after=lm.training_clock(q)
    pd.testing.assert_frame_equal(before,after)
    assert a.loc[a.split=='train','y'].notna().any()


def test_missing_grid_and_label_censor():
    raw=synthetic()
    drop=(raw.underlying=='BTC')&(raw.state_time==pd.Timestamp('2026-09-05 08:05Z'))
    p=lm.dense_panel(raw.loc[~drop])
    row=p[(p.underlying=='BTC')&(p.state_time==pd.Timestamp('2026-09-05 08:05Z'))]
    assert len(row)==1 and np.isnan(row.v.iloc[0])
    labels,thresholds=lm.q99_labels(p)
    assert len(thresholds)==2
    target=p[(p.underlying=='BTC')&(p.state_time==pd.Timestamp('2026-09-05 08:00Z'))].index[0]
    assert np.isnan(labels.loc[target,'y_30m']) # missing in future
    tr_end=p[(p.underlying=='BTC')&(p.split=='train')].index[-1]
    assert np.isnan(labels.loc[tr_end,'y_30m']) # forbidden train->valid labels
    pp=p.copy()
    pp.loc[pp.split=='test','v'] *= 10000
    _, threshold_after=lm.q99_labels(pp)
    pd.testing.assert_frame_equal(thresholds, threshold_after)


def test_end_to_end_same_row_scores():
    p,_=lm.training_clock(lm.dense_panel(synthetic()))
    labels, _=lm.q99_labels(p)
    base=p.join(labels[['onset','eligible_now','y_60m']])
    lag=lm.fixed_lags(p)
    base=base.join(lag)
    common=lm.regularity_mask(p)
    base['eligible_now'] &= common
    exp=lm.simple_exponential_features(p)
    base=base.join(exp[['ema_30m','ema_240m']])
    lift, meta=lm.lifted_features(p,.10,4)
    base=base.join(lift[['kernel_memory','fast_minus_slow']])
    specs={'B':['now','lag_5m','lag_10m','lag_30m','lag_60m','lag_120m'],
           'C':['now','kernel_memory','fast_minus_slow']}
    res={}
    for name,names in specs.items():
        metrics, preds=lm.fit_predict(base,names,1e-3,60,eval_split='test')
        assert np.isfinite(metrics['logloss'])
        res[name]=preds
    joined=lm.align_comparison(res['C'],res['B'])
    boot=lm.bootstrap_day_diff(joined, B=50)
    assert np.isfinite(boot['delta_logloss'])
    path=lm.event_factor_path(p,labels,lift.fast_minus_slow)
    assert {'relative_min','mean_factor'}.issubset(path.columns)


def test_equal_multiscale_has_same_rates_as_rough():
    p = pd.DataFrame({'underlying':['AAA']*500,'y':np.cos(np.arange(500)/19)})
    N=6
    x,_=lm.quadrature(.1,N)
    em=lm.ema_prior(p,np.log(2)/x)
    null=lm.equal_exponential_features(p,N,precomputed=em)
    rough,_=lm.lifted_features(p,.1,N,precomputed=em)
    assert np.allclose(null.fast_minus_slow.iloc[2:],rough.fast_minus_slow.iloc[2:])
    assert np.allclose(null.equal_memory.iloc[2:],em[2:].mean(axis=1))
    assert not np.allclose(null.equal_memory.iloc[2:],rough.kernel_memory.iloc[2:])
