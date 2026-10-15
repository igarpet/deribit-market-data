"""RD01-b causal residual-regime controls; synthetic smoke data only."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import pytest
from data_engineering.modeling import observable_regime_discovery as rd
from data_engineering.modeling import observable_regime_discovery_b as b
from test_observable_regime_discovery import synthetic_panel


def test_registration_and_frozen_methods():
    assert b.METHODS==('baseline','direct_ridge','interaction_ridge','residual_tree','residual_tree_sticky')
    assert b.PRIMARY_HORIZON==30
    assert b.Spec().confirm_bars==3
    assert b.registration()['status']=='HISTORICAL_DISCOVERY_NOT_CONFIRMATION'


def test_confirm_no_lookahead():
    x=synthetic_panel(days=2).iloc[:25].copy()
    raw=np.array(['A','B','B','B','A','B','B','B']+['A']*17,dtype=object)
    labels=b.causal_confirmed_labels(x,raw)
    assert labels[:8].tolist()==['A','A','A','B','B','B','B','B']
    for i in range(len(x)):
        a=b.causal_confirmed_labels(x.iloc[:i+1],raw[:i+1])
        assert a[-1]==labels[i]


def test_confirm_gap_and_day_reset():
    x=synthetic_panel(days=2)
    raw=np.full(len(x),'A',dtype=object);raw[286:291]='B'
    labs=b.causal_confirmed_labels(x,raw)
    assert labs[287]=='A' and labs[288]=='B'
    cut=x.drop(index=[12]).reset_index(drop=True)
    raw=np.full(len(cut),'A',dtype=object);raw[12]='B'
    assert b.causal_confirmed_labels(cut,raw)[12]=='B'


def test_confirm_invalid_inputs():
    x=synthetic_panel(days=1)
    with pytest.raises(ValueError,match='state length'):b.causal_confirmed_labels(x,['A'])
    with pytest.raises(ValueError,match='confirmation'):b.causal_confirmed_labels(x,np.full(len(x),'A'),0)
    with pytest.raises(ValueError,match='chronologically'):b.causal_confirmed_labels(x.iloc[::-1],np.full(len(x),'A'))


def test_no_future_feature_scaling_leak():
    x=synthetic_panel(days=12);a=x.iloc[:1000];v=x.iloc[1000:1030].copy()
    z1,y1=b.feature_design(a,v,interactions=True)
    changed=v.copy();changed.loc[changed.index[-1],'basis_bps']+=100000
    z2,y2=b.feature_design(a,changed,interactions=True)
    np.testing.assert_allclose(z1,z2)
    np.testing.assert_allclose(y1[:-1],y2[:-1])
    assert y1.shape[1]==len(rd.TREE_COLS)+len(b.INTERACTIONS)


def test_missing_and_zero_horizon_labels():
    x=rd.forward_labels(synthetic_panel(days=3));assert np.isnan(x.ret_60m.iloc[276])
    assert rd.full_feature_days(x).size==3


def test_tree_uses_horizon_specific_complete_targets():
    x=rd.forward_labels(synthetic_panel(days=75))
    c=b.Spec(tree_min_leaf_absolute=25)
    m,meta=b.fit_residual_tree(x,30,c)
    assert 1<m.get_n_leaves()<=6
    assert meta['tree_fit_origins']>200
    assert meta['early_days']+meta['late_days']==75
    labels=b.label_tree(m,x.iloc[:5]);assert all(s.startswith('T') for s in labels)


def test_regime_correction_reproduces_ridge_when_shrinkage_huge():
    x=rd.forward_labels(synthetic_panel(days=40))
    x=x.loc[np.isfinite(x.ret_5m)].copy()
    cal=x.iloc[:4900];val=x.iloc[4900:6800]
    ca=np.full(len(cal),'A',dtype=object);va=np.full(len(val),'B',dtype=object)
    c=b.Spec(regime_shrinkage=1e30)
    loc,_,_=b.fit_locations(cal,val,{},c,5,ca,va,ca,va)
    np.testing.assert_allclose(loc['direct_ridge'],loc['residual_tree'])
    np.testing.assert_allclose(loc['direct_ridge'],loc['residual_tree_sticky'])


def test_regime_transitions_censor_midnight():
    x=synthetic_panel(days=2)
    lab=np.full(len(x),'A',dtype=object)
    lab[:10]='B';lab[287]='B';lab[288]='A'
    p,e,tr=b.summarize_durations(x,lab,b.causal_confirmed_labels(x,lab))
    assert p.n_adjacent_pairs.min()==2*287
    assert e.n_episodes.sum()>=4
    # midnight A/B change is NOT an observed state transition
    assert tr.groupby('method')['count'].sum().eq(2*287).all()


def test_end_to_end_full_synthetic():
    z=b.scan_asset(synthetic_panel(days=94),b.Spec(tree_min_leaf_absolute=25,bootstrap_repetitions=25,minimum_bootstrap_days=5),'BTC')
    assert len(z['scores'])==3*4*5
    assert set(z['scores'].method)==set(b.METHODS)
    assert (z['persistence'].n_adjacent_pairs>0).all()
    assert z['definitions'].n_leaves.between(1,6).all()
    assert (z['scores'].pred_up_mean.between(0,1)).all()
    out=b.consolidate({'BTC':z},b.Spec(bootstrap_repetitions=25,minimum_bootstrap_days=5))
    assert len(out['comparison_summary'])==20
    assert not out['bootstrap'].empty
    assert not out['decision']['AM_T_HAR_Blend_modified']
