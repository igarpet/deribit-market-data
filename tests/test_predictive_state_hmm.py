"""PSH01 synthetic regression checks. Run from repo root with pytest."""
from pathlib import Path
import numpy as np
import pandas as pd
from data_engineering.modeling import predictive_state_hmm as psh


def test_perp_30m_timestamp_alignment_and_gaps(tmp_path, monkeypatch):
    times = pd.date_range('2026-01-01 00:00', periods=180, freq='min', tz='UTC')
    q = pd.DataFrame({'open_time': times, 'close': 100*np.exp(.00005*np.arange(len(times)))})
    path = Path(tmp_path)/'perp.parquet'
    monkeypatch.setattr(psh.pd, 'read_parquet', lambda *_args, **_kw: q.copy())
    rows = psh.perp_30m_returns(path)
    assert np.isclose(rows.iloc[0].perp_fwd_30m, .00005*30)
    # Future prices cannot alter past-state observations, but may change forward returns.
    q.loc[100:,'close'] *= 2
    modified = psh.perp_30m_returns(path)
    assert np.isclose(rows.iloc[0].perp_fwd_30m, modified.iloc[0].perp_fwd_30m)
    # Missing any minute invalidates its end bucket and adjacent forward outcomes.
    q=q.drop(index=45)
    gap = psh.perp_30m_returns(path)
    assert pd.isna(gap.iloc[0].perp_fwd_30m)


def test_bootstrap_paired_sign_and_monotonic():
    cfg=psh.Spec(bootstrap_days=100,seed=1)
    actual=np.tile(np.asarray([1,0,0,1]), 100)
    days=np.repeat(pd.date_range('2026-01-01',periods=100,tz='UTC'),4)
    b=psh.bootstrap_delta(actual,np.where(actual==1,.7,.3),np.repeat(.5,len(actual)),days,cfg)
    assert b['delta_logloss'] < 0 and b['ci975'] < 0 and b['test_days']==100


def test_registration_no_option_economics_claim():
    conf=psh.registration()
    assert conf['option_PnL'] == 'NOT_TESTED_WITHOUT_POINT_IN_TIME_QUOTES'
    assert set(conf['allowed_predictor_evidence']) == set(psh.EMISSIONS)
    assert conf['perp_roundtrip_cost_bp'] > 0