"""RD04: frozen RD03 regime-return density robustness/falsification audit.

RD04 does not fit or select models. An RD03 replay supplies causal frozen
AM-T-HAR-Blend mixture forecasts and responses. All post-selection historical
results are exploratory, even under multiplicity control.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Mapping
import numpy as np
import pandas as pd

VERSION = 'RD04-0.1.0'
PRIMARY_ASSETS = ('BTC', 'ETH')
REGIME_METHODS = ('observable_tree_mean', 'latent_hmm_mean', 'latent_semi_markov_mean')
METHODS = ('AM_T_HAR_BLEND','sign_only_mean','direct_mean','observable_tree_mean',
           'latent_static_mean','latent_hmm_mean','latent_semi_markov_mean')
REFERENCES = {
  'sign_only_mean': ('AM_T_HAR_BLEND',),
  'direct_mean': ('AM_T_HAR_BLEND','sign_only_mean'),
  'observable_tree_mean': ('AM_T_HAR_BLEND','direct_mean'),
  'latent_static_mean': ('AM_T_HAR_BLEND','direct_mean'),
  'latent_hmm_mean': ('AM_T_HAR_BLEND','direct_mean','latent_static_mean'),
  'latent_semi_markov_mean': ('AM_T_HAR_BLEND','direct_mean','latent_hmm_mean'),
}

@dataclass(frozen=True)
class Specification:
    version: str = VERSION
    primary_horizon: int = 30
    horizons: tuple[int,...] = (5,15,30,60)
    primary_assets: tuple[str,...] = PRIMARY_ASSETS
    periods: tuple[str,...] = ('diagnostic','evaluation')
    bootstrap_reps: int = 1000
    seed: int = 2614
    min_days: int = 12
    min_days_half: int = 6
    min_days_vol_bucket: int = 5
    min_days_weekly: int = 8
    max_multiplicity_tests: int = 6
    status: str = 'REPEATEDLY_INSPECTED_HISTORICAL_ROBUSTNESS'


def registration() -> dict:
    return asdict(Specification())


def validate_origin_scores(x:pd.DataFrame,cfg:Specification) -> pd.DataFrame:
    req={'asset','state_time','period','horizon_min','method','nll','up_brier',
         'variance','realized_return','up_probability'}
    if missing:=req-set(x):raise ValueError(f'missing origin columns: {missing}')
    z=x.copy()
    z['state_time']=pd.to_datetime(z.state_time,utc=True)
    if z.state_time.isna().any():raise ValueError('invalid UTC times')
    if not z.period.isin(cfg.periods).all():raise ValueError('non-held-out period')
    if not z.horizon_min.isin(cfg.horizons).all():raise ValueError('unregistered horizon')
    if not z.method.isin(METHODS).all():raise ValueError('unregistered method')
    for c in ('nll','up_brier','variance','realized_return','up_probability'):
        z[c]=pd.to_numeric(z[c],errors='coerce')
        if not np.isfinite(z[c].to_numpy(float)).all():raise ValueError(f'bad {c}')
    if (z.variance<=0).any() or ((z.up_probability<0)|(z.up_probability>1)).any():
        raise ValueError('invalid forecast variance/probability')
    key=['asset','period','horizon_min','state_time','method']
    if z.duplicated(key).any():raise ValueError('duplicate scored origin')
    z['date']=z.state_time.dt.floor('D')
    # Each model must score the *same* observations, labels and variance particles.
    origin=['asset','period','horizon_min','state_time']
    q=z.groupby(origin,sort=False).agg(k=('method','size'),
             unique=('method','nunique'),var_min=('variance','min'),
             var_max=('variance','max'),rmin=('realized_return','min'),
             rmax=('realized_return','max'))
    if ((q.k!=len(METHODS))|(q.unique!=len(METHODS))).any():
        raise ValueError('different origin grids across methods')
    if not np.allclose(q.var_min,q.var_max,rtol=1e-11,atol=1e-14):
        raise ValueError('variance particles not matched across methods')
    if not np.allclose(q.rmin,q.rmax,rtol=0,atol=1e-14):
        raise ValueError('return labels not matched across methods')
    if not z.groupby(['asset','period','horizon_min']).method.nunique().eq(len(METHODS)).all():
        raise ValueError('a method was omitted')
    return z


def verify_rd03_archive(scores:pd.DataFrame,old:pd.DataFrame,*,tol:float=5e-9)->pd.DataFrame:
    """Refuse robustness conclusions if evaluation densities differ from frozen RD03."""
    ev=scores.loc[scores.period.eq('evaluation')]
    fresh=ev.groupby(['asset','horizon_min','method'],as_index=False).agg(
               nll=('nll','mean'),up_brier=('up_brier','mean'),
               n_origins=('nll','size'))
    target=old[['asset','horizon_min','method','nll','up_brier','n_origins']].copy()
    if len(fresh)!=len(target):raise ValueError('RD03 archive grid differs')
    q=fresh.merge(target,on=['asset','horizon_min','method'],how='outer',validate='one_to_one',
                 suffixes=('_replay','_archive'),indicator=True)
    if not q._merge.eq('both').all():raise ValueError('missing RD03 archive key')
    for c in ('nll','up_brier'):
        if not np.allclose(q[c+'_replay'],q[c+'_archive'],atol=tol,rtol=0):
            raise ValueError(f'RD03 {c} not reproduced')
    if not np.array_equal(q.n_origins_replay.to_numpy(int),q.n_origins_archive.to_numpy(int)):
        raise ValueError('RD03 origin counts differ')
    q['abs_nll_difference']=np.abs(q.nll_replay-q.nll_archive)
    q['abs_brier_difference']=np.abs(q.up_brier_replay-q.up_brier_archive)
    return q.drop(columns=['_merge'])


def paired_daily(x:pd.DataFrame)->pd.DataFrame:
    keys=['asset','period','horizon_min','date','method']
    agg=x.groupby(keys,as_index=False).agg(nll=('nll','mean'),
                                           up_brier=('up_brier','mean'),n=('nll','size'))
    ref=agg.set_index(['asset','period','horizon_min','date','method'])
    rows=[]
    for method,baselines in REFERENCES.items():
        for baseline in baselines:
            left=agg.loc[agg.method.eq(method)]
            right=agg.loc[agg.method.eq(baseline)]
            m=left.merge(right,on=['asset','period','horizon_min','date'],
                         validate='one_to_one',suffixes=('_model','_ref'))
            if (m.n_model!=m.n_ref).any():raise ValueError('unmatched daily origins')
            for a in m.itertuples(index=False):
                rows.append({'asset':a.asset,'period':a.period,'horizon_min':a.horizon_min,
                    'date':a.date,'method':method,'reference':baseline,
                    'n_origins':a.n_model,'delta_nll':a.nll_model-a.nll_ref,
                    'delta_up_brier':a.up_brier_model-a.up_brier_ref})
    return pd.DataFrame(rows)


def interval(z:np.ndarray, cfg:Specification,*,minimum:int|None=None,seed:int|None=None):
    """Paired equal-day bootstrap; no claim of independence or selected-model p-values."""
    a=np.asarray(z,float)
    if not np.isfinite(a).all():raise ValueError('nonfinite daily differences')
    n=len(a)
    min_n=cfg.min_days if minimum is None else minimum
    if n<min_n:return float(a.mean()) if n else np.nan,np.nan,np.nan,'INSUFFICIENT_DAYS'
    rng=np.random.default_rng(cfg.seed if seed is None else seed)
    b=rng.integers(0,n,size=(cfg.bootstrap_reps,n))
    samples=a[b].mean(axis=1)
    low,high=np.quantile(samples,[.025,.975])
    return float(a.mean()),float(low),float(high),'EXPLORATORY'


def time_stability(daily:pd.DataFrame,cfg:Specification)->pd.DataFrame:
    rows=[]
    keys=['asset','horizon_min','method','reference']
    for key,all_group in daily.groupby(keys,sort=True):
        asset,h,model,reference=key
        for period in cfg.periods:
            g=all_group.loc[all_group.period.eq(period)].sort_values('date')
            if g.empty:continue
            groups={'full':g}
            first=len(g)//2
            if first:
                groups['early_half']=g.iloc[:first]
                groups['late_half']=g.iloc[first:]
            for part,q in groups.items():
                for metric in ('nll','up_brier'):
                    val=q['delta_'+metric].to_numpy(float)
                    mn=cfg.min_days if part=='full' else cfg.min_days_half
                    avg,lo,hi,status=interval(val,cfg,minimum=mn,
                            seed=cfg.seed+len(rows))
                    rows.append({'asset':asset,'horizon_min':h,'method':model,
                         'reference':reference,'period':period,'slice':part,
                         'metric':metric,'n_days':len(q),'n_origins':int(q.n_origins.sum()),
                         'delta_equal_day':avg,'ci025':lo,'ci975':hi,
                         'favorable_days':int((val<0).sum()),'status':status})
    return pd.DataFrame(rows)


def volatility_stress(x:pd.DataFrame,cfg:Specification)->pd.DataFrame:
    """Use pre-evaluation (diagnostic) *forecast* sigma cutoffs, never labels."""
    orig=x.loc[x.method.eq('AM_T_HAR_BLEND'),
             ['asset','period','horizon_min','state_time','variance']].copy()
    data=[]; cutrows=[]
    for (asset,h),q in orig.groupby(['asset','horizon_min']):
        source=q.loc[q.period.eq('diagnostic'),'variance'].to_numpy(float)
        if len(source)<30:raise ValueError('no diagnostic variance sample')
        low,high=np.quantile(source,[1/3,2/3]);
        if not high>low:raise ValueError('degenerate sigma terciles')
        cutrows.append({'asset':asset,'horizon_min':h,'sigma2_q33':low,'sigma2_q67':high})
        z=q[['asset','period','horizon_min','state_time']].copy()
        z['bucket']=np.searchsorted([low,high],q.variance.to_numpy(float),'right')
        data.append(z)
    slices=pd.concat(data,ignore_index=True)
    merged=x.merge(slices,on=['asset','period','horizon_min','state_time'],
                   validate='many_to_one')
    daily=paired_daily(merged.assign(period=merged.period+'__'+merged.bucket.astype(str)))
    daily['vol_bucket']=daily.period.str.rsplit('__',n=1).str[-1].astype(int)
    daily['period']=daily.period.str.rsplit('__',n=1).str[0]
    out=[]
    for keys,g in daily.groupby(['asset','period','horizon_min','method','reference','vol_bucket']):
        asset,period,h,method,reference,bucket=keys
        for metric in ('nll','up_brier'):
            avg,lo,hi,stat=interval(g['delta_'+metric].to_numpy(float),cfg,
                minimum=cfg.min_days_vol_bucket,seed=cfg.seed+len(out))
            out.append({'asset':asset,'period':period,'horizon_min':h,'method':method,
                 'reference':reference,'vol_bucket':int(bucket),'metric':metric,
                 'n_days':len(g),'n_origins':int(g.n_origins.sum()),
                 'delta_equal_day':avg,'ci025':lo,'ci975':hi,'status':stat})
    return pd.DataFrame(out),pd.DataFrame(cutrows)


def primary_family(daily:pd.DataFrame,cfg:Specification)->pd.DataFrame:
    """Six Bonferroni one-sided 95%-familywise bootstrap screening bounds.

    This adjusts just the frozen within-RD04 comparisons. It cannot repair
    selecting models in RD01–03 or repeatedly inspecting the same history.
    """
    q=daily[(daily.asset.isin(cfg.primary_assets))&
       daily.horizon_min.eq(cfg.primary_horizon)&daily.period.eq('diagnostic')&
       daily.method.isin(REGIME_METHODS)&daily.reference.eq('direct_mean')]
    expected={(a,m) for a in cfg.primary_assets for m in REGIME_METHODS}
    if set(zip(q.asset,q.method))!=expected:raise ValueError('primary comparisons incomplete')
    alpha=.05/len(expected)
    rows=[]
    for asset,method in sorted(expected):
        g=q[(q.asset==asset)&q.method.eq(method)].sort_values('date')
        if len(g)<cfg.min_days:raise ValueError('primary diagnostic too sparse')
        z=g.delta_nll.to_numpy(float)
        rng=np.random.default_rng(cfg.seed+len(rows))
        draws=z[rng.integers(0,len(z),size=(cfg.bootstrap_reps,len(z)))].mean(axis=1)
        upper=float(np.quantile(draws,1-alpha))
        rows.append({'asset':asset,'method':method,'reference':'direct_mean',
          'period':'diagnostic','horizon_min':cfg.primary_horizon,'n_days':len(z),
          'delta_equal_day':float(z.mean()),'bonferroni_upper_one_sided':upper,
          'familywise_alpha_per_comparison':alpha,'familywise_screen_pass':bool(upper<0)})
    return pd.DataFrame(rows)


def decision(time:pd.DataFrame,primary:pd.DataFrame,cfg:Specification,
             *,archive_verified:bool,synthetic:bool=False)->dict:
    if synthetic:return {'verdict':'SYNTHETIC_NO_MARKET_INFERENCE','status':'SYNTHETIC',
                         'AM_T_HAR_Blend_modified':False}
    if not archive_verified:raise ValueError('RD03 archive reproduction is mandatory')
    key=time[(time.horizon_min==cfg.primary_horizon)&time.metric.eq('nll')&
       time.slice.eq('full')&time.asset.isin(cfg.primary_assets)]
    flags={}
    for method in REGIME_METHODS:
        cond=[]
        for asset in cfg.primary_assets:
            for period in cfg.periods:
                for ref in ('AM_T_HAR_BLEND','direct_mean'):
                    q=key[(key.asset==asset)&(key.period==period)&
                          (key.method==method)&(key.reference==ref)]
                    if len(q)!=1:raise ValueError('missing RD04 evaluation cell')
                    x=q.iloc[0]
                    cond.append(bool(x.status=='EXPLORATORY' and x.ci975<0))
        flags[method]={'all_primary_windows_and_baselines_nominally_favorable':all(cond),
                       'n_nominal_favorable':sum(cond),'n_required':len(cond)}
    # Historical period and data reuse mean no prospective regime-confirmation,
    # regardless of nominal interval signs. A conservative go/no-go research gate:
    pursue=any(v['all_primary_windows_and_baselines_nominally_favorable'] for v in flags.values())
    return {'status':cfg.status,'verdict':('EXPLORATORY_ROBUSTNESS_CANDIDATE' if pursue
                       else 'NO_STABLE_INCREMENTAL_REGIME_EVIDENCE'),
            'methods':flags,'primary_family_tests':len(primary),
            'multiplicity_adjusted_confirmation':False,
            'prospective_confirmation':False,'AM_T_HAR_Blend_modified':False,
            'note':'Daily bootstrap and nominal intervals are descriptive after many previous searches. '
                   'RD04 does not confer clean confirmation or authorize trading.'}
