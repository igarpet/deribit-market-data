"""RD01-c synthetic math, point-in-time and market-independent regression tests."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import simpson
from scipy.stats import t as student_t
from data_engineering.modeling import observable_regime_discovery as rd
from data_engineering.modeling import observable_regime_discovery_c as rc
from test_observable_regime_discovery import synthetic_panel


def test_split_t_reverts_exactly_to_student_t():
    q=np.linspace(-8,8,101)
    assert np.allclose(rc.split_t_logpdf(q,0),student_t.logpdf(q,df=6),rtol=1e-12,atol=1e-12)
    assert np.allclose(rc.split_t_cdf(q,0),student_t.cdf(q,df=6),rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('skew',[-.5,-.25,0.,.3,.55])
def test_split_t_is_normalized_zero_mean_and_fixed_variance(skew):
    q=np.linspace(-100,100,100001)
    p=np.exp(rc.split_t_logpdf(q,skew))
    assert simpson(p,x=q)==pytest.approx(1.,abs=1e-6)
    assert simpson(q*p,x=q)==pytest.approx(0.,abs=1e-5)
    assert simpson(q*q*p,x=q)==pytest.approx(1.5,rel=5e-4)
    assert np.all(np.diff(rc.split_t_cdf(q[::100],skew))>=0)


def test_split_t_shapes_vectorize_and_probability_bounds():
    q=np.array([-.8,-.1,0.,.6])
    k=np.array([-.5,0,.2,.5])
    assert rc.split_t_logpdf(q,k).shape==(4,)
    probs=rc.split_t_cdf(q,k)
    assert np.isfinite(probs).all() and np.all((probs>=0)&(probs<=1))
    with pytest.raises(ValueError):rc.split_t_components(np.array([np.nan]))


def test_regime_skew_is_training_only_and_sparse_defaults_to_global():
    panel=rd.forward_labels(synthetic_panel(days=8))
    cal=panel.iloc[:6*288].loc[lambda g:np.isfinite(g.ret_30m)].copy()
    labels=np.full(len(cal),'A',dtype=object)
    labels[:100]='B'
    cfg=rc.Spec(minimum_state_events=200,minimum_state_days=4)
    y=cal.ret_30m.to_numpy(float)/rd.volatility_proxy(cal,30)
    residual=y/np.std(y)
    param=rc.fit_state_skews(cal,labels,residual,.18,30,cfg).set_index('state')
    assert param.loc['B','skew']==pytest.approx(.18)
    assert param.loc['B','status']=='SPARSE_FALLBACK'
    assert (param['skew'].abs()<=cfg.max_skew).all()


def test_discovery_folds_leave_legacy_test_unseen():
    x=synthetic_panel(days=94)
    days=rd.full_feature_days(x)
    folds=rd.research_folds(days,rd.Spec())
    stop=int(len(days)*rd.Spec().discovery_fraction)
    for fold in folds:
        assert max(fold['validation_days'])<days[stop]
        assert max(fold['structure_days'])<min(fold['calibration_days'])
        assert max(fold['calibration_days'])<min(fold['validation_days'])


def test_fixed_location_and_scale_probabilities_differ_when_skew_changes():
    mu=np.zeros(3)
    q=np.array([.2,-.4,2.])
    k=np.array([.45,-.4,0.])
    prob=rc.split_t_cdf(q,k)
    assert np.isfinite(prob).all()
    assert abs(float(prob[0]-student_t.cdf(q[0],6)))>1e-5
    for val in [-.5,0,.5]:
        a,b,m,s=rc.split_t_components(val)
        assert s>0


def test_bootstrap_requires_matching_methods_and_days():
    days=pd.date_range('2025-01-01',periods=25,freq='1D',tz='UTC')
    data=[]
    for day in days:
        for method,delta in [('direct_symmetric',0),('interaction_symmetric',.003),
                             ('direct_global_skew',-.003),('direct_continuous_skew',-.005),
                             ('direct_regime_skew',-.01)]:
            data.append(dict(date=day,method=method,horizon_min=30,asset='BTC',fold=1,
                             nll=delta,up_tail_brier=.1+delta,down_tail_brier=.1+delta))
    out=rc.paired_day_bootstrap(pd.DataFrame(data),rc.Spec(bootstrap_repetitions=30))
    sub=out.query("metric=='nll' and model=='direct_regime_skew' and reference=='direct_continuous_skew'")
    assert len(sub)==1
    assert sub.iloc[0]['delta']==pytest.approx(-.005)
    assert sub.iloc[0]['ci975']<0
