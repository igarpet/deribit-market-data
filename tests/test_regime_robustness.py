"""Synthetic-only RD04 historical audit invariants; no fitted market claims."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from data_engineering.modeling import regime_robustness as rd

@pytest.fixture(scope='module')
def frame():
    rows=[]; rng=np.random.default_rng(23)
    for asset in ('BTC','ETH','SOL','XRP'):
      for period,start in (('diagnostic','2026-09-01'),('evaluation','2026-09-19')):
        for h in rd.Specification().horizons:
          for d in pd.date_range(start,periods=16,freq='D',tz='UTC'):
            for j in range(4):
              v=1e-6*(1+0.1*j+0.008*int((d-pd.Timestamp(start,tz='UTC')).days))
              y=float(rng.normal(0,np.sqrt(v)))
              for m in rd.METHODS:
                bias=0 if m=='AM_T_HAR_BLEND' else (-0.0002 if m=='direct_mean' else .00015)
                rows.append({'asset':asset,'state_time':d+pd.Timedelta(minutes=30*j),
                 'period':period,'horizon_min':h,'method':m,
                 'nll':-4.7+bias+0.001*float(rng.normal()),
                 'up_brier':.25+bias/2,'variance':v,'realized_return':y,
                 'up_probability':.5})
    return pd.DataFrame(rows)


def test_validation_and_coverage(frame):
    q=rd.validate_origin_scores(frame,rd.Specification())
    assert q.date.notna().all()
    assert q.groupby(['asset','horizon_min','period','state_time']).size().eq(7).all()


def test_duplicates_rejected(frame):
    with pytest.raises(ValueError,match='duplicate'):
        rd.validate_origin_scores(pd.concat([frame,frame.iloc[:1]]),rd.Specification())


def test_missing_method_rejected(frame):
    with pytest.raises(ValueError,match='different origin grids'):
        rd.validate_origin_scores(frame.iloc[1:],rd.Specification())


def test_mismatched_variance_rejected(frame):
    q=frame.copy();q.loc[q.index[0],'variance'] *= 2
    with pytest.raises(ValueError,match='variance'):
        rd.validate_origin_scores(q,rd.Specification())


def test_mismatched_return_rejected(frame):
    q=frame.copy();q.loc[q.index[0],'realized_return'] += .1
    with pytest.raises(ValueError,match='return labels'):
        rd.validate_origin_scores(q,rd.Specification())


def test_rd03_archive_match(frame):
    q=rd.validate_origin_scores(frame,rd.Specification())
    ev=q[q.period.eq('evaluation')]
    ref=ev.groupby(['asset','horizon_min','method'],as_index=False).agg(
        nll=('nll','mean'),up_brier=('up_brier','mean'),n_origins=('nll','size'))
    assert rd.verify_rd03_archive(q,ref).abs_nll_difference.max()<1e-12
    corrupt=ref.copy();corrupt.loc[0,'nll']+=.01
    with pytest.raises(ValueError,match='not reproduced'):
        rd.verify_rd03_archive(q,corrupt)


def test_pair_and_time_stability(frame):
    cfg=rd.Specification(bootstrap_reps=30)
    q=rd.paired_daily(rd.validate_origin_scores(frame,cfg))
    assert q.groupby(['asset','horizon_min','date','period']).size().eq(13).all()
    t=rd.time_stability(q,cfg)
    assert t.slice.nunique()==3
    assert t.query('slice=="full"').n_days.min()==16


def test_vol_stress_and_primary(frame):
    cfg=rd.Specification(bootstrap_reps=30)
    q=rd.validate_origin_scores(frame,cfg)
    vol,cuts=rd.volatility_stress(q,cfg)
    assert set(vol.vol_bucket)=={0,1,2}
    assert len(cuts)==4*4
    p=rd.primary_family(rd.paired_daily(q),cfg)
    assert len(p)==6
    assert p.familywise_alpha_per_comparison.eq(.05/6).all()


def test_decision_fails_closed(frame):
    cfg=rd.Specification(bootstrap_reps=30)
    d=rd.paired_daily(rd.validate_origin_scores(frame,cfg))
    t=rd.time_stability(d,cfg)
    p=rd.primary_family(d,cfg)
    assert not rd.decision(t,p,cfg,archive_verified=True)['prospective_confirmation']
    with pytest.raises(ValueError,match='mandatory'):
        rd.decision(t,p,cfg,archive_verified=False)
    assert rd.decision(t,p,cfg,archive_verified=False,synthetic=True)['verdict'].startswith('SYNTHETIC')


def test_sparse_interval():
    cfg=rd.Specification(bootstrap_reps=20)
    _,lo,hi,tag=rd.interval([-.2,-.3],cfg)
    assert np.isnan(lo) and np.isnan(hi) and tag=='INSUFFICIENT_DAYS'
