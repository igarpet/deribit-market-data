"""CS02b: strictly next-five-minute UP vs DOWN conditional on a NONZERO return.

All CS02 data/volatility preparation is reused without touching the original model.
Flat candles break streaks, and next-zero labels are EXCLUDED from binary fitting
and scoring (not skipped forward to a later nonzero candle). Volatility and sign
are observed by the *current* close; next-bar sign is never a feature.

Conditional P(up | next return != 0) is not unconditional P(up): the event
'next != 0' is unknown at forecast time. Recombine with P(next=0) in any
future live application, and validate joint three-class / density forecasts.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Mapping
import numpy as np
import pandas as pd
from data_engineering.modeling import candle_streak_volatility as cv

VERSION = 'CS02b-0.1.0'
MODELS = (
    'random_50', 'iid_binary', 'sign_binary', 'streak_binary',
    'sign_vol_binary', 'additive_binary', 'streak_vol_matrix_binary',
)
CS02_MODELS = tuple(cv.MODELS)
DIAGNOSTIC_MODELS = tuple('cs02_normalized_'+m for m in CS02_MODELS)
PAIRS = (
    ('sign_vol_binary','sign_binary'),
    ('sign_vol_binary','iid_binary'),
    ('streak_vol_matrix_binary','streak_binary'),
    ('streak_vol_matrix_binary','sign_vol_binary'),
    ('streak_vol_matrix_binary','additive_binary'),
    ('streak_binary','sign_binary'),
    ('cs02_normalized_sign_vol_3class','sign_vol_binary'),
    ('cs02_normalized_streak_vol_matrix_3class','streak_vol_matrix_binary'),
)

@dataclass(frozen=True)
class Specification:
    version: str = VERSION
    assets: tuple = ('BTC', 'ETH', 'SOL', 'XRP')
    cutoff_utc: str = '2026-10-14T00:00:00Z'
    train_fraction: float = .6
    valid_fraction: float = .2
    fast_bars: int = 12
    slow_bars: int = 288
    shrink_sign: float = 20.
    shrink_streak: float = 80.
    shrink_sign_vol: float = 100.
    shrink_joint: float = 125.
    boot: int = 1000
    permutation_reps: int = 100
    seed: int = 2603
    status: str = 'HISTORICAL_EXPLORATORY'

def registration():
    return asdict(Specification())

def filter_binary_next(frame: pd.DataFrame) -> pd.DataFrame:
    """Mask only *immediate* zero next returns; preserve the observed time grid.

    `frame` comes from CS02 split_eligible and apply_vol_bins. Its streak age
    was computed with zeros breaking streaks. Filtering rows cannot change age.
    Return rows in original order with a new boolean `scorable`; keep original
    zero observations as metadata for auditing selection and coverage.
    """
    required={'sign','length','next_sign','scorable','next_valid','next_nonzero','level_bin','trend_bin','state_time'}
    if not required.issubset(frame):raise ValueError(f'missing columns: {required-set(frame)}')
    if not frame.sign.isin([-1,1]).all() or (frame.length<1).any():raise ValueError('invalid current streak')
    if not frame.next_sign.isin([-1,0,1]).all():raise ValueError('invalid next sign')
    q=frame.copy()
    q['scorable_threeclass']=q.scorable.astype(bool)
    q['excluded_next_zero']=q.scorable_threeclass & q.next_valid & q.next_sign.eq(0)
    q['scorable']=q.scorable_threeclass & q.next_nonzero & q.next_sign.ne(0)
    if (q.loc[q.scorable,'next_sign']==0).any():raise AssertionError('zero target in binary eval')
    return q

def _counts(frame:pd.DataFrame, cols:list[str]) -> dict[tuple,np.ndarray]:
    out={}
    for key,group in frame.groupby(cols, sort=False):
        if not isinstance(key,tuple):key=(key,)
        lab=group.next_sign.to_numpy(int)
        out[tuple(int(v) for v in key)]=np.array([(lab==-1).sum(),(lab==1).sum()],dtype=float)
    return out

def fit_binary_matrix(train:pd.DataFrame,*,shrink_sign=20.,shrink_streak=80.,
                      shrink_sign_vol=100.,shrink_joint=125.):
    """Refit fixed CS02 hierarchy on TRAIN ONLY, using immediate nonzero targets.

    Fitted entries store [P(down | nonzero), P(up | nonzero)]. No unbounded
    lookup truncation: every observed k is supported, unseen tuples back off.
    """
    if min(shrink_sign,shrink_streak,shrink_sign_vol,shrink_joint)<=0:raise ValueError('nonpositive pseudocount')
    tr=train.loc[train.scorable].copy()
    if len(tr)<100:raise ValueError('too few binary events')
    if not tr.next_sign.isin([-1,1]).all():raise ValueError('zero target in training')
    if not tr.sign.isin([-1,1]).all() or (tr.length<1).any():raise ValueError('invalid signed streak')
    if not tr.level_bin.isin([0,1,2]).all() or not tr.trend_bin.isin([0,1,2]).all():raise ValueError('invalid volatility category')
    g=_counts(tr.assign(key=0),['key'])[(0,)]
    p=(g+.5)/(g.sum()+1.)  # Jeffreys binary prior
    sign_c=_counts(tr,['sign'])
    streak_c=_counts(tr,['sign','length'])
    vol_c=_counts(tr,['sign','level_bin','trend_bin'])
    joint_c=_counts(tr,['sign','length','level_bin','trend_bin'])
    def blend(c,prior,pseudo):return (c+pseudo*prior)/(c.sum()+pseudo)
    bysign={key:blend(c,p,shrink_sign) for key,c in sign_c.items()}
    bylength={key:blend(c,bysign[(key[0],)],shrink_streak) for key,c in streak_c.items()}
    byvol={key:blend(c,bysign[(key[0],)],shrink_sign_vol) for key,c in vol_c.items()}
    joint={}
    for key,c in joint_c.items():
        sign,k,level,phase=key
        prior=.5*bylength[(sign,k)]+.5*byvol[(sign,level,phase)]
        joint[key]=blend(c,prior,shrink_joint)
    return {'global':p,'sign':bysign,'streak':bylength,'sign_vol':byvol,
            'joint':joint,'counts':{'sign':sign_c,'streak':streak_c,'sign_vol':vol_c,'joint':joint_c},
            'n_train':len(tr),'shrinkage':[shrink_sign,shrink_streak,shrink_sign_vol,shrink_joint]}

def predict_binary(frame:pd.DataFrame, fit:dict) -> dict[str,np.ndarray]:
    """P(UP|nonzero) for exactly the same valid origins across every model."""
    n=len(frame)
    forecasts={m:np.empty(n,dtype=float) for m in MODELS}
    for i,(sign,k,level,phase) in enumerate(frame[['sign','length','level_bin','trend_bin']].itertuples(index=False,name=None)):
        sign,k,level,phase=int(sign),int(k),int(level),int(phase)
        p0=fit['global'];ps=fit['sign'].get((sign,),p0)
        pk=fit['streak'].get((sign,k),ps)
        pv=fit['sign_vol'].get((sign,level,phase),ps)
        pm=fit['joint'].get((sign,k,level,phase),.5*(pk+pv))
        for name,p in zip(MODELS,(.5,p0[-1],ps[-1],pk[-1],pv[-1],(.5*(pk+pv))[-1],pm[-1])):
            forecasts[name][i]=p
    return forecasts

def conditionalize_cs02(frame:pd.DataFrame, cs02_fit:dict) -> dict[str,np.ndarray]:
    """Keep CS02's original TRAIN-fitted 3-class models; condition on nonzero.

    This is a diagnostic comparison with CS02, evaluated on the binary subset.
    P(UP | nonzero) = P3(UP)/(P3(UP)+P3(DOWN)). No TEST fitting.
    """
    pred=cv.predict_matrix(frame,cs02_fit)
    out={}
    for name,matrix in pred.items():
        total=matrix[:,0]+matrix[:,2]
        if (total<=0).any() or not np.isfinite(total).all():raise ValueError('nonpositive directional mass')
        out['cs02_normalized_'+name]=matrix[:,2]/total
    return out

def score_binary(events:pd.DataFrame, predictions:Mapping[str,np.ndarray]):
    """Proper binary log loss/Brier on immediate-next-nonzero events only."""
    y=events.next_sign.to_numpy(int)
    if not np.isin(y,[-1,1]).all():raise ValueError('zero outcome must not enter scores')
    observed_up=(y==1).astype(float)
    dates=pd.to_datetime(events.state_time,utc=True).dt.floor('D')
    summary=[];losses=[]
    for name,p0 in predictions.items():
        p=np.asarray(p0,dtype=float)
        if p.shape!=(len(y),) or not np.isfinite(p).all() or (p<=0).any() or (p>=1).any():
            raise ValueError('probabilities must be strictly between 0 and 1')
        ll=-observed_up*np.log(p)-(1-observed_up)*np.log1p(-p)
        b=(p-observed_up)**2
        summary.append({'model':name,'n':len(y),'binary_logloss':ll.mean(),
                        'binary_brier':b.mean(),'mean_p_up':p.mean(),'realized_up':observed_up.mean()})
        losses.append(pd.DataFrame({'date':dates.to_numpy(),'model':name,'binary_logloss':ll,'binary_brier':b}))
    return pd.DataFrame(summary),pd.concat(losses,ignore_index=True)

def day_bootstrap(per_origin:pd.DataFrame,pairs=PAIRS,*,B=1000,seed=2603):
    """Paired complete-day 95% intervals; both equal-day and event-weighted delta.

    Eligibility excludes incomplete UTC days and last-bar split transitions;
    each individual day's observations carry their actual weights for event
    aggregates; paired bootstrap resamples complete days without row shuffling.
    """
    if B<10:raise ValueError('bootstrap reps below 10')
    gp=per_origin.groupby(['date','model'],sort=True)[['binary_logloss','binary_brier']].agg(['mean','count'])
    days=np.sort(per_origin.date.unique())
    rng=np.random.default_rng(seed)
    draw=rng.integers(0,len(days),size=(B,len(days)))
    out=[]
    for a,b in pairs:
        for metric in ['binary_logloss','binary_brier']:
            s1=gp[(metric,'mean')].unstack('model').reindex(days)
            if a not in s1.columns or b not in s1.columns:raise ValueError(f'missing model {a} / {b}')
            diff=(s1[a]-s1[b]).to_numpy(float)
            if not np.isfinite(diff).all():raise ValueError('missing daily scores')
            n=gp[(metric,'count')].unstack('model').reindex(days)[a].to_numpy(int)
            # Equal complete-day weights; separately report event-weighted delta.
            ci=np.quantile(diff[draw].mean(axis=1),[.025,.975])
            out.append({'model':a,'reference':b,'metric':metric,
                        'delta_mean_day':diff.mean(),
                        'delta_event_weighted':np.average(diff,weights=n),
                        'ci025':ci[0],'ci975':ci[1],
                        'n_days':len(days),'min_events_day':n.min(),'max_events_day':n.max()})
    return pd.DataFrame(out)

def fitted_probability_matrix(fit:dict, include_keys=None):
    """TRAIN-frozen matrix, all observed signed ages; counts are true TRAIN samples."""
    keys=set(fit['counts']['joint'])
    if include_keys is not None:keys.update(tuple(int(v) for v in key) for key in include_keys)
    keytable=pd.DataFrame(sorted(keys),columns=['sign','length','level_bin','trend_bin'])
    predicted=predict_binary(keytable,fit)['streak_vol_matrix_binary']
    keytable['n_train']=[int(fit['counts']['joint'].get(key,np.zeros(2)).sum()) for key in sorted(keys)]
    keytable['p_up_nonzero']=predicted
    keytable['p_down_nonzero']=1-predicted
    keytable['p_continue_nonzero']=np.where(keytable.sign==1,predicted,1-predicted)
    keytable['p_reverse_nonzero']=1-keytable.p_continue_nonzero
    keytable['signed_streak']=keytable.sign*keytable.length
    keytable['vol_level']=keytable.level_bin.map(dict(enumerate(cv.LEVELS)))
    keytable['vol_phase']=keytable.trend_bin.map(dict(enumerate(cv.TRENDS)))
    return keytable.sort_values(['signed_streak','level_bin','trend_bin']).reset_index(drop=True)

def descriptive_probability_matrix(all_events:pd.DataFrame):
    """Retrospective all-dates matrix, NOT for out-of-sample trading."""
    g=all_events.loc[all_events.scorable].copy()
    q=g.groupby(['sign','length','level_bin','trend_bin'],as_index=False).agg(
        n=('next_sign','size'),n_up=('next_sign',lambda x:int((x==1).sum())))
    q['p_up_nonzero']=q.n_up/q.n
    q['p_down_nonzero']=1-q.p_up_nonzero
    q['signed_streak']=q.sign*q.length
    return q.sort_values(['signed_streak','level_bin','trend_bin']).reset_index(drop=True)

def monthly_stability(events:pd.DataFrame,min_n=40):
    """Descriptive conditional direction by month, coarsened ages for display."""
    e=events.loc[events.scorable].copy()
    e['month']=pd.to_datetime(e.state_time,utc=True).dt.strftime('%Y-%m')
    e['age_group']=pd.cut(e.length,bins=[0,1,2,3,5,8,np.inf],labels=['1','2','3','4–5','6–8','9+'])
    q=e.groupby(['month','sign','age_group','level_bin','trend_bin'],observed=True).agg(
        n=('next_sign','size'), n_up=('next_sign',lambda x:int((x==1).sum()))).reset_index()
    q=q.loc[q.n>=min_n].copy()
    q['p_up_nonzero']=q.n_up/q.n
    q['p_continue_nonzero']=np.where(q.sign==1,q.p_up_nonzero,1-q.p_up_nonzero)
    return q

def shuffle_volatility_null(test:pd.DataFrame,fit:dict,*,B=100,seed=2603):
    """Permutation diagnostic: frozen TEST labels and fit; shuffle bin pairs.

    Permute within current-sign/coarse-age groups; this is NOT a simulated
    price/volatility market world, and not a trading strategy p-value.
    """
    te=test.loc[test.scorable].copy().reset_index(drop=True)
    if len(te)<100:raise ValueError('few eligible binary events')
    original=predict_binary(te,fit)
    y=te.next_sign.to_numpy(int)
    yi=(y==1).astype(int);idx=np.arange(len(y))
    def loss(p):return float(np.mean(-yi*np.log(p)-(1-yi)*np.log1p(-p)))
    base=loss(original['sign_binary'])
    actual=loss(original['sign_vol_binary'])-base
    signage=te[['sign','length']].to_numpy(int)
    pairs,inv=np.unique(signage,axis=0,return_inverse=True)
    every=pd.DataFrame({'sign':np.repeat(pairs[:,0],9),'length':np.repeat(pairs[:,1],9),
                        'level_bin':np.tile(np.repeat(np.arange(3),3),len(pairs)),
                        'trend_bin':np.tile(np.arange(3),len(pairs)*3)})
    table=predict_binary(every,fit)['sign_vol_binary'].reshape(len(pairs),9)
    rng=np.random.default_rng(seed)
    groups=te.groupby(['sign',pd.cut(te.length,[0,1,2,3,5,8,np.inf],labels=False)],observed=True).indices
    initial=te[['level_bin','trend_bin']].to_numpy(int)
    values=[]
    for _ in range(B):
        shuffled=initial.copy()
        for ix in groups.values():shuffled[ix]=initial[rng.permutation(ix)]
        p=table[inv,shuffled[:,0]*3+shuffled[:,1]]
        values.append(loss(p)-base)
    sims=np.array(values)
    return {'actual_signvol_minus_sign':actual,'null05':np.quantile(sims,.05),
            'null_median':np.median(sims),'null95':np.quantile(sims,.95),
            'fraction_null_le_actual':np.mean(sims<=actual),'B':B}
