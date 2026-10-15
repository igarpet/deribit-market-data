"""Synthetic RD01 tests: no historical-data result is implied."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import pytest
from data_engineering.modeling import observable_regime_discovery as rd


def synthetic_panel(days=94,seed=2607):
    rng=np.random.default_rng(seed)
    n=days*288
    t=pd.date_range('2025-01-01',periods=n,freq='5min',tz='UTC')
    f=rng.normal(0,.32,n)
    b=rng.normal(0,6,n).cumsum()*.012+rng.normal(0,.6,n)
    ret=rng.normal(0,.001,n)
    # Positive information from signed flow that is known at bar t, for r[t+1].
    for j in range(1,n):
        ret[j]+=.00022*np.sign(f[j-1])-.08*ret[j-1]
    r=pd.Series(ret)
    mom3=r.rolling(3,min_periods=1).mean().to_numpy()
    z=pd.DataFrame({
        'state_time':t,'return_5m':ret,'feature_ready':True,
        'log_rv24h':np.log((r.pow(2).rolling(288,min_periods=1).mean()+1e-7).to_numpy()),
        'mom_r5':ret,'mom_r15_mean':mom3,
        'flow5':f,'flow30':pd.Series(f).rolling(6,min_periods=1).mean().to_numpy(),
        'basis_bps':b,'basis_change_bps':np.r_[0,np.diff(b)],
        'last_1m_r':ret*.21+rng.normal(0,.0001,n),
        'minute_accel':ret*.07+rng.normal(0,.0001,n),
        'log_trades':np.log1p(rng.integers(10,200,n))})
    return z


def test_175_combinations_compatibility():
    assert len(rd.BASE_COLS)==3
    assert len(rd.TREE_COLS)==9
    assert len(rd.REGIMES)==27
    assert rd.HORIZONS==(5,15,30,60)


def test_forward_targets_no_current_return_leak():
    x=synthetic_panel(days=4)
    x=rd.forward_labels(x)
    for j in (17,31,110):
        assert x.loc[j,'ret_5m']==pytest.approx(x.loc[j+1,'return_5m'])
        assert x.loc[j,'ret_15m']==pytest.approx(x.loc[j+1:j+3,'return_5m'].sum())
        assert x.loc[j,'ret_60m']==pytest.approx(x.loc[j+1:j+12,'return_5m'].sum())
    assert np.isnan(x.loc[287,'ret_5m'])
    assert not np.isnan(x.loc[275,'ret_60m'])
    assert np.isnan(x.loc[276,'ret_60m'])


def test_forward_gap_invalidates_all_horizons():
    x=synthetic_panel(days=4)
    x=x.drop(index=[22]).reset_index(drop=True)
    a=rd.forward_labels(x)
    assert np.isnan(a.loc[21,'ret_5m'])
    assert np.isnan(a.loc[18,'ret_30m'])
    assert np.isnan(a.loc[11,'ret_60m'])
    assert len(rd.full_feature_days(a))==3


def test_label_audit_rejects_modified_existing_target():
    x=rd.forward_labels(synthetic_panel(days=3))
    x.loc[5,'ret_30m']+=.01
    with pytest.raises(ValueError,match='disagrees'):
        rd.forward_labels(x)


def test_state_assignments_fit_only_on_structure():
    x=synthetic_panel(days=35)
    rule=rd.structure_rules(x,rd.Spec())
    y=rd.assign_rules(x,rule)
    assert len(y)==len(x)
    assert set(y)<=set(rd.REGIMES)
    x2=x.copy();x2['return_5m']+=100
    np.testing.assert_array_equal(y,rd.assign_rules(x2,rule))


def test_tree_size_and_fixed_training():
    x=rd.forward_labels(synthetic_panel(days=35))
    tree,desc=rd.fit_tree(x,rd.Spec())
    assert 2<=tree.get_n_leaves()<=8
    assert isinstance(desc,str)
    a=rd.tree_labels(tree,x.head(40))
    assert len(a)==40
    assert all(v.startswith('T') for v in a)


def test_scoring_valid_probability_and_sample():
    x=rd.forward_labels(synthetic_panel(days=20))
    x=x[np.isfinite(x.ret_5m)].copy()
    cfg=rd.Spec()
    cut=rd.structure_rules(x.iloc[:2000],cfg)
    cal=x.iloc[:2500].copy();val=x.iloc[2500:].copy()
    state_c=rd.assign_rules(cal,cut);state_v=rd.assign_rules(val,cut)
    states={'rule_regime':state_c,'tree_regime':np.full(len(cal),'T001')}
    valstates={'rule_regime':state_v,'tree_regime':np.full(len(val),'T001')}
    loc,aux=rd.fit_predict_locations(cal,val,states,valstates,cfg,5)
    score,daily=rd.score_returns(val,loc,aux,cfg,5)
    assert len(score)==4
    assert score.nll.notna().all()
    assert score.pred_up_mean.between(0,1).all()
    assert (daily.n>0).all()


def test_episode_duration_and_right_censor():
    t=pd.date_range('2025-01-01',periods=7,freq='5min',tz='UTC')
    df=pd.DataFrame({'state_time':t})
    a=np.array(['A','A','B','B','B','A','A'])
    trans,episode,hazard=rd.transitions_episodes(df,a,'test')
    assert episode.n_bars.tolist()==[2,3,2]
    assert episode.right_censored.tolist()==[False,False,True]
    assert episode.left_censored.tolist()==[True,False,False]
    assert trans['count'].sum()==6
    assert episode.duration_minutes.tolist()==[10,15,10]


def test_reject_future_prediction_no_60m_at_boundary():
    x=rd.forward_labels(synthetic_panel(days=3))
    assert x.loc[275,'ret_60m']==pytest.approx(x.loc[276:287,'return_5m'].sum())
    assert pd.isna(x.loc[276,'ret_60m'])


def test_folds_do_not_overlap():
    x=synthetic_panel(days=94)
    folds=rd.research_folds(rd.full_feature_days(x),rd.Spec())
    assert len(folds)==3
    for f in folds:
        s=set(f['structure_days']);c=set(f['calibration_days']);v=set(f['validation_days'])
        assert not (s&c or c&v or s&v)
        assert max(c)<min(v)


def test_end_to_end_full_synthetic():
    cfg=rd.Spec(bootstrap_repetitions=24)
    z=rd.scan_asset(synthetic_panel(days=94),cfg,asset='BTC')
    assert len(z['scores'])==3*4*4
    assert set(z['scores'].method)==set(rd.METHODS)
    assert set(z['scores'].horizon_min)==set(rd.HORIZONS)
    assert z['episodes'].duration_minutes.min()>=5
    assert len(z['transitions'])>0
    assert len(z['regimes'])>0
    out=rd.consolidate({'BTC':z},cfg)
    assert not out['paired_bootstrap'].empty
    assert len(out['comparison_summary'])==16
    assert out['manifest']['legacy_test_used_for_selection'] is False
    assert out['manifest']['AM_T_HAR_Blend_modified'] is False
