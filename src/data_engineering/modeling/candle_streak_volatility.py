"""CS02: causal volatility-conditioned probability matrix for five-minute candle streaks.

No arbitrary run-length cap. A volatility observation at five-minute close t uses
only returns <=t; all three next-bar outcomes (-1,0,+1) are scored at t+1.
Absolute 24h variance level and 1h-versus-24h expansion quantile bins are
fitted on TRAIN ONLY. Does not import a fitted AM-T-HAR-Blend model.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import numpy as np
import pandas as pd
from data_engineering.modeling import candle_streaks as cs

VERSION='CS02-0.1.0'
STEP=pd.Timedelta(minutes=5)
CLASSES=(-1,0,1)
LEVELS=('low','mid','high')
TRENDS=('compression','neutral','expansion')
MODELS=('iid_3class','sign_3class','streak_3class','sign_vol_3class','additive_3class','streak_vol_matrix_3class')

@dataclass(frozen=True)
class Specification:
    version:str=VERSION
    assets:tuple=('BTC','ETH','SOL','XRP')
    cutoff_utc:str='2026-10-14T00:00:00Z'
    train_fraction:float=.6
    valid_fraction:float=.2
    fast_bars:int=12
    slow_bars:int=288
    n_bins:int=3
    shrink_sign:float=20.
    shrink_streak:float=80.
    shrink_sign_vol:float=100.
    shrink_joint:float=125.
    boot:int=1000
    seed:int=2602
    status:str='HISTORICAL_EXPLORATORY'

def registration():return asdict(Specification())


def add_past_volatility(raw:pd.DataFrame,fast_bars=12,slow_bars=288):
    """Strict complete lag windows; all predictor data known at end of bar t."""
    if not 0<fast_bars<slow_bars:raise ValueError('bad lag durations')
    raw=cs._checked(raw);parts=[]
    for asset,g in raw.groupby('underlying',sort=True):
        g=g.reset_index(drop=True).copy()
        t=pd.to_datetime(g.state_time,utc=True)
        r=g.return_5m.to_numpy(float)
        finite=np.isfinite(r)
        # Discontinuities invalidate windows, even if intermediate rows absent.
        gap=np.r_[True,(t.diff().iloc[1:]!=STEP).to_numpy()]
        segment=np.cumsum(gap)
        sq=pd.Series(r*r)
        fast=sq.groupby(segment).transform(lambda x:x.rolling(fast_bars,min_periods=fast_bars).mean())
        slow=sq.groupby(segment).transform(lambda x:x.rolling(slow_bars,min_periods=slow_bars).mean())
        fast=fast.to_numpy(float);slow=slow.to_numpy(float)
        ok=np.isfinite(fast)&np.isfinite(slow)&(fast>0)&(slow>0)
        g['log_rv24h']=np.where(ok,np.log(np.maximum(slow,1e-30)),np.nan)
        g['log_fastslow']=np.where(ok,np.log(np.maximum(fast,1e-30)/np.maximum(slow,1e-30)),np.nan)
        g['vol_ready']=ok
        parts.append(g)
    return pd.concat(parts,ignore_index=True)


def complete_utc_days(raw:pd.DataFrame):
    """Only complete 00:00..23:55 UTC days with 288 finite five-minute returns."""
    raw=cs._checked(raw)
    all_=raw.assign(date=raw.state_time.dt.floor('D'))
    days={}
    for asset,g in all_.groupby('underlying',sort=True):
        q=g.groupby('date').agg(count=('return_5m','size'),valid=('return_5m',lambda x:int(np.isfinite(x.to_numpy(float)).sum())),
                                start=('state_time','min'),finish=('state_time','max'))
        start=q.index.to_series().to_numpy()
        good=(q['count'].to_numpy()==288)&(q.valid.to_numpy()==288)&(q.start.to_numpy()==start)&(q.finish.to_numpy()==(q.index+pd.Timedelta(hours=23,minutes=55)).to_numpy())
        days[asset]=pd.DatetimeIndex(q.index[good])
    return days


def build_vol_events(raw:pd.DataFrame,*,fast_bars=12,slow_bars=288):
    """Join causal rolling-vol features at current completed bar, not at next bar."""
    vol=add_past_volatility(raw,fast_bars,slow_bars)
    events,_,quality=cs.build_events_and_runs(raw)
    join=vol[['underlying','state_time','log_rv24h','log_fastslow','vol_ready']]
    ev=events.merge(join,on=['underlying','state_time'],how='left',validate='one_to_one')
    assert len(ev)==len(events)
    return ev,quality


def split_eligible(ev:pd.DataFrame,full_dates:pd.DatetimeIndex,train_fraction=.6,valid_fraction=.2):
    """Prevent incomplete-day scoring and split-crossing labels."""
    dates=pd.DatetimeIndex(sorted(full_dates))
    if len(dates)<60:raise ValueError('too few complete UTC days')
    cut1=dates[int(len(dates)*train_fraction)];cut2=dates[int(len(dates)*(train_fraction+valid_fraction))]
    if not dates.min()<cut1<cut2<dates.max():raise ValueError('split failure')
    x=ev.loc[ev.date.isin(dates)&ev.vol_ready].copy()
    x['period']=np.where(x.date<cut1,'train',np.where(x.date<cut2,'valid','test'))
    nextdate=(x.state_time+STEP).dt.floor('D')
    nextperiod=np.where(nextdate<cut1,'train',np.where(nextdate<cut2,'valid','test'))
    x['scorable']=x.next_valid & nextdate.isin(dates) & (nextperiod==x.period)
    return x.reset_index(drop=True),{'fit_before':str(cut1),'test_from':str(cut2),'full_utc_days':len(dates),
                'excluded_incomplete_or_invalid_days':int(ev.date.nunique()-len(dates))}


def fit_vol_bins(train:pd.DataFrame):
    q=train.loc[train.scorable,['log_rv24h','log_fastslow']].to_numpy(float)
    if len(q)<100 or not np.isfinite(q).all():raise ValueError('invalid training features')
    boundaries=np.quantile(q,[1/3,2/3],axis=0).T
    if np.any(np.diff(boundaries,axis=1)<=0):raise ValueError('indistinguishable quantile bins')
    return {'level_cut':boundaries[0].tolist(),'trend_cut':boundaries[1].tolist(),'fit_n':len(q)}


def apply_vol_bins(frame:pd.DataFrame,bins:dict):
    q=frame.copy()
    if not q[['log_rv24h','log_fastslow']].notna().all().all():raise ValueError('missing volatility')
    q['level_bin']=np.searchsorted(bins['level_cut'],q.log_rv24h.to_numpy(float),side='right').astype(int)
    q['trend_bin']=np.searchsorted(bins['trend_cut'],q.log_fastslow.to_numpy(float),side='right').astype(int)
    return q


def _counts(g,keys):
    counts={}
    for key,gr in g.groupby(keys,sort=False):
        if not isinstance(key,tuple):key=(key,)
        label=gr.next_sign.to_numpy(int)
        counts[tuple(int(i) for i in key)]=np.array([(label==s).sum() for s in CLASSES],float)
    return counts


def fit_matrix(train:pd.DataFrame,*,shrink_sign=20.,shrink_streak=80.,shrink_sign_vol=100.,shrink_joint=125.):
    """Hierarchical Dirichlet-smoothed 3-class probabilities, training only.

    Joint cell prior = 0.5*streak + 0.5*sign-vol model. This is an explicitly
    registered interpolation, not retrospectively selected model weights.
    """
    tr=train.loc[train.scorable].copy()
    if len(tr)<100:raise ValueError('not enough transitions')
    if (tr.level_bin.gt(2)|tr.trend_bin.gt(2)).any():raise ValueError('invalid bin')
    global_count=_counts(tr.assign(key=0),['key'])[(0,)]
    p=(global_count+.5)/(global_count.sum()+1.5)
    sign_count=_counts(tr,['sign'])
    streak_count=_counts(tr,['sign','length'])
    sv_count=_counts(tr,['sign','level_bin','trend_bin'])
    joint_count=_counts(tr,['sign','length','level_bin','trend_bin'])
    def parent(c,ref,pseudo):return (c+pseudo*ref)/(c.sum()+pseudo)
    ps={key:parent(v,p,shrink_sign) for key,v in sign_count.items()}
    psk={key:parent(v,ps[(key[0],)],shrink_streak) for key,v in streak_count.items()}
    psv={key:parent(v,ps[(key[0],)],shrink_sign_vol) for key,v in sv_count.items()}
    pjoint={}
    for key,v in joint_count.items():
        sign,length,lvl,trend=key
        q=.5*psk[(sign,length)]+.5*psv[(sign,lvl,trend)]
        pjoint[key]=parent(v,q,shrink_joint)
    return {'global':p,'sign':ps,'streak':psk,'sign_vol':psv,'joint':pjoint,
            'counts':{'sign':sign_count,'streak':streak_count,'sign_vol':sv_count,'joint':joint_count},
            'prior':{'sign':shrink_sign,'streak':shrink_streak,'sign_vol':shrink_sign_vol,'joint':shrink_joint},
            'n_train':len(tr)}


def predict_matrix(frame:pd.DataFrame,fit:dict):
    """Predictions use current sign, reached length and known volatility only."""
    n=len(frame);pred={name:np.empty((n,3),float) for name in MODELS}
    ps=fit['sign'];pk=fit['streak'];pv=fit['sign_vol'];pj=fit['joint']
    for i,(s,k,l,v) in enumerate(frame[['sign','length','level_bin','trend_bin']].itertuples(index=False,name=None)):
        s,k,l,v=int(s),int(k),int(l),int(v)
        pa=ps.get((s,),fit['global'])
        pb=pk.get((s,k),pa)
        pc=pv.get((s,l,v),pa)
        pd_=pj.get((s,k,l,v),.5*(pb+pc))
        for model,p in zip(MODELS,(fit['global'],pa,pb,pc,.5*(pb+pc),pd_)):
            pred[model][i]=p
    return pred


def score_threeclass(events:pd.DataFrame,preds:dict):
    """Proper unconditional up/down/zero NLL and multiclass Brier. No target conditioning."""
    y=events.next_sign.to_numpy(int)
    if not set(np.unique(y)).issubset(set(CLASSES)):raise ValueError('unknown target')
    idx=np.searchsorted(np.asarray(CLASSES),y)
    result=[];pair=[]
    for model,p in preds.items():
        if p.shape!=(len(events),3) or not np.allclose(p.sum(axis=1),1):raise ValueError('bad prediction matrix')
        prob=np.clip(p,1e-12,1-1e-12)
        ll=-np.log(prob[np.arange(len(y)),idx])
        onehot=np.eye(3)[idx]
        brier=np.sum((p-onehot)**2,axis=1)
        result.append({'model':model,'n':len(y),'logloss':float(ll.mean()),'brier3':float(brier.mean()),
                       'mean_p_up':float(p[:,2].mean()),'mean_p_down':float(p[:,0].mean()),'mean_p_zero':float(p[:,1].mean()),
                       'actual_up_rate':float(np.mean(y==1)),'actual_zero_rate':float(np.mean(y==0))})
        pair.append(pd.DataFrame({'date':pd.to_datetime(events.state_time,utc=True).dt.floor('D'),
                                  'model':model,'logloss':ll,'brier3':brier}))
    return pd.DataFrame(result),pd.concat(pair,ignore_index=True)


def day_bootstrap(per_origin,*,B=1000,seed=2602):
    """Paired day bootstrap; discard incomplete days upstream, weight each day equally.

    Also report bar-weighted difference; differences in per-day observation counts
    are exposed. Full UTC days can have different counts of scorable sign events.
    """
    q=per_origin.groupby(['date','model']).agg(logloss=('logloss','mean'),brier3=('brier3','mean'),count=('logloss','size')).unstack('model')
    rng=np.random.default_rng(seed)
    out=[]
    pairs=[('streak_vol_matrix_3class',m) for m in ('streak_3class','sign_vol_3class','additive_3class','sign_3class','iid_3class')]
    pairs += [('sign_vol_3class','sign_3class'),('streak_3class','sign_3class')]
    for a,b in pairs:
        for metric in ('logloss','brier3'):
            vals=(q[(metric,a)]-q[(metric,b)]).to_numpy(float)
            if not np.isfinite(vals).all():raise ValueError('misaligned bootstrap dates')
            simulations=np.mean(vals[rng.integers(0,len(vals),size=(B,len(vals)))],axis=1)
            counts=q[('count',a)].to_numpy(float)
            out.append({'model':a,'reference':b,'metric':metric,'mean_day_difference':float(vals.mean()),
                        'mean_bar_weighted_difference':float(np.average(vals,weights=counts)),
                        'ci025':float(np.quantile(simulations,.025)),'ci975':float(np.quantile(simulations,.975)),
                        'n_days':len(vals),'min_events_day':int(counts.min()),'max_events_day':int(counts.max())})
    return pd.DataFrame(out)


def probability_matrix(train:pd.DataFrame,fit:dict,*,include_keys=None):
    """All observed signed lengths, including sparse and never fitted OOS keys.

    Matrix entries are frozen TRAIN-model estimates, not post-hoc in-sample
    frequencies. Raw train observations and shrinkage strength are explicit.
    """
    keys=set(fit['counts']['joint'])
    if include_keys is not None:keys|=set(include_keys)
    keys=sorted(keys)
    q=pd.DataFrame(keys,columns=['sign','length','level_bin','trend_bin'])
    probs=predict_matrix(q,fit)['streak_vol_matrix_3class']
    q['n_train']= [int(fit['counts']['joint'].get(key,np.zeros(3)).sum()) for key in keys]
    for j,name in enumerate(('p_down','p_zero','p_up')):q[name]=probs[:,j]
    q['p_continue']=np.where(q.sign==1,q.p_up,q.p_down)
    q['p_reverse']=np.where(q.sign==1,q.p_down,q.p_up)
    q['p_continue_given_nonzero']=q.p_continue/(q.p_continue+q.p_reverse)
    q['signed_streak']=q.sign*q.length
    q['vol_level']=q.level_bin.map(dict(enumerate(LEVELS)))
    q['vol_phase']=q.trend_bin.map(dict(enumerate(TRENDS)))
    return q.sort_values(['signed_streak','level_bin','trend_bin']).reset_index(drop=True)


def descriptive_matrix(all_events:pd.DataFrame):
    """Counts and empirical unconditional probabilities for all observed dates.

    DO NOT use this retrospective table as a trading or OOS probability model.
    """
    g=all_events.loc[all_events.next_valid].copy()
    keys=['sign','length','level_bin','trend_bin']
    q=g.groupby(keys,as_index=False).agg(n=('next_sign','size'),n_up=('next_sign',lambda x:int((x==1).sum())),
                                n_down=('next_sign',lambda x:int((x==-1).sum())),
                                n_zero=('next_sign',lambda x:int((x==0).sum())))
    for c in ('up','down','zero'):q['p_'+c]=q['n_'+c]/q.n
    q['signed_streak']=q.sign*q.length
    return q.sort_values(['signed_streak','level_bin','trend_bin']).reset_index(drop=True)


def monthly_stability(events:pd.DataFrame,*,min_n=50):
    """Descriptive continuation by month and tier; separate from OOS scores."""
    g=events.loc[events.next_valid].copy()
    g['month']=pd.to_datetime(g.state_time,utc=True).dt.strftime('%Y-%m')
    # Month-level grouping uses coarse age buckets ONLY for readable stability output;
    # no age truncation in train matrix, predictor, or full-matrix output.
    g['age_group']=pd.cut(g.length,bins=[0,1,2,3,5,8,np.inf],labels=['1','2','3','4-5','6-8','9+'])
    return g.groupby(['underlying','month','sign','age_group','level_bin','trend_bin'],observed=True).agg(
             n=('next_sign','size'),n_continue=('continuation','sum'),n_zero=('next_zero','sum')).reset_index().assign(
               p_continue=lambda x:x.n_continue/x.n).query('n >= @min_n')


def volatility_alignment_null(test:pd.DataFrame,fit:dict,*,reps=100,seed=2602):
    """Secondary diagnostic; shuffle joint volatility labels within sign/age groups.

    Predictors and training frozen; high-speed factorized probability lookup.
    This is not a generative volatility or trading significance test.
    """
    te=test.loc[test.scorable].copy().reset_index(drop=True)
    if len(te)<100:raise ValueError('not enough test events')
    observed=predict_matrix(te,fit)
    y=te.next_sign.to_numpy(int)
    trueidx=np.searchsorted(CLASSES,y)
    rows=np.arange(len(y))
    reference=-np.log(np.maximum(observed['streak_3class'][rows,trueidx],1e-12)).mean()
    actual=-np.log(np.maximum(observed['streak_vol_matrix_3class'][rows,trueidx],1e-12)).mean()-reference
    # All observed sign/k pairs, but never truncate k. Precompute every possible
    # volatility class for each observed pair to accelerate null replications.
    agepairs=np.column_stack([te.sign.to_numpy(int),te.length.to_numpy(int)])
    distinct,inv=np.unique(agepairs,axis=0,return_inverse=True)
    possibilities=pd.DataFrame({
        'sign':np.repeat(distinct[:,0],9),
        'length':np.repeat(distinct[:,1],9),
        'level_bin':np.tile(np.repeat(np.arange(3),3),len(distinct)),
        'trend_bin':np.tile(np.arange(3),len(distinct)*3)})
    lookup=predict_matrix(possibilities,fit)['streak_vol_matrix_3class'].reshape(len(distinct),9,3)
    rng=np.random.default_rng(seed)
    groups=te.groupby(['sign',pd.cut(te.length,bins=[0,1,2,3,5,8,np.inf],labels=False)],observed=True).indices
    initial=te[['level_bin','trend_bin']].to_numpy(int)
    null=[]
    for _ in range(reps):
        vol=initial.copy()
        for ix in groups.values():vol[ix]=initial[rng.permutation(ix)]
        indices=vol[:,0]*3+vol[:,1]
        pred=lookup[inv,indices,trueidx]
        matrix=-np.log(np.maximum(pred,1e-12)).mean()
        null.append(float(matrix-reference))
    return {'actual_matrix_minus_streak_logloss':float(actual),
            'null_05':float(np.quantile(null,.05)),'null_median':float(np.median(null)),
            'null_95':float(np.quantile(null,.95)),
            'null_fraction_le_actual':float(np.mean(np.asarray(null)<=actual)),
            'permutations':reps}
