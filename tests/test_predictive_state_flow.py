"""Synthetic PSH02 safety/causality checks; no historical data needed."""
import numpy as np
import pandas as pd
import pytest

from data_engineering.modeling import predictive_state_flow as sf


def make_perp_inputs(n=18):
    dates = pd.date_range("2026-01-01 00:05", periods=n, freq="5min", tz="UTC")
    r = np.linspace(-.002, .002, n)
    returns = pd.DataFrame({"state_time":dates,"return_5m":r})
    flow = pd.DataFrame({"state_time":dates, "flow_5m":np.linspace(-.6,.6,n),
                         "flow_valid":True})
    return returns,flow


def test_complete_futures_labels_and_past_flow(monkeypatch, tmp_path):
    ret,flow = make_perp_inputs()
    monkeypatch.setattr(sf.csp, "read_perp", lambda *_a,**_k:(ret.copy(),flow.copy()))
    path = tmp_path/"dummy.parquet"
    path.write_bytes(b"stub")
    f = sf.read_perp_panel(path)
    assert np.isclose(f.iloc[0].perp_fwd_5m, ret.iloc[1].return_5m)
    assert np.isclose(f.iloc[0].perp_fwd_30m, ret.iloc[1:7].return_5m.sum())
    # Change future returns but never the past source flow.
    ret.loc[6:,"return_5m"] += .02
    changed = sf.read_perp_panel(path)
    assert np.isclose(f.loc[0,"flow_5m"],changed.loc[0,"flow_5m"])
    assert not np.isclose(f.loc[0,"perp_fwd_30m"],changed.loc[0,"perp_fwd_30m"])
    ret.loc[4,"return_5m"] = np.nan
    g = sf.read_perp_panel(path)
    assert np.isnan(g.loc[0,"perp_fwd_30m"])
    assert np.isfinite(g.loc[0,"perp_fwd_5m"])


def test_no_splice_over_missing_future_bucket(monkeypatch, tmp_path):
    ret,flow=make_perp_inputs()
    ret=ret.drop(index=4).reset_index(drop=True)
    monkeypatch.setattr(sf.csp,"read_perp",lambda *_a,**_k:(ret,flow))
    p=tmp_path/"x.parquet";p.write_bytes(b"x")
    x=sf.read_perp_panel(p)
    assert np.isnan(x.loc[0,"perp_fwd_30m"])


def test_flow_interaction_and_train_only_scaling():
    rng=np.random.default_rng(9)
    n=120
    tr=pd.DataFrame({c:rng.normal(size=n) for c in sf.CONT})
    te=tr.iloc[:25].copy()
    post=rng.dirichlet(np.ones(4),size=n)
    xp, xt=sf.design_for_arm(tr,te,post,post[:25],"hmm_flow_interaction")
    assert xp.shape==(n,len(sf.INTER)+3+3)
    assert xt.shape==(25,len(sf.INTER)+3+3)
    assert np.isfinite(xp).all()
    # Huge future-only change should not modify TRAIN design.
    later=te.copy()
    later.loc[:,"flow_5m"] += 100
    xa,_=sf.design_for_arm(tr,later,post,post[:25],"continuous_interaction")
    xb,_=sf.design_for_arm(tr,te,post,post[:25],"continuous_interaction")
    assert np.allclose(xa,xb)


def test_paired_bootstrap_sign_and_familywise_threshold():
    cfg=sf.Spec(bootstrap_reps=100,seed=1)
    days=np.repeat(pd.date_range("2026-03-01",periods=70,tz="UTC"),5)
    d=np.full(len(days),-.01)
    good=sf.day_bootstrap(d,days,cfg)
    assert good["n_days"]==70 and good["favorable_bonferroni"]
    bad=sf.day_bootstrap(-d,days,cfg)
    assert not bad["favorable_bonferroni"]


def test_manifest_primary_market_and_outcome():
    spec=sf.registration()
    assert spec["primary_horizon_minutes"]==5
    assert spec["k_states"]==4
    assert spec["trade_status"]=="NO_EXECUTION_APPROVAL"
    assert spec["options_economics"]=="NOT_TESTED"
    assert len(sf.COMPS)==5
