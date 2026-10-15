"""RD01-c: test market-state conditional RETURN ASYMMETRY, not another mean-rule fit.

The original RD01-b residual-location tree failed its descriptive primary gate.
RD01-c asks whether current, observable market conditions alter the asymmetric
shape of the NEXT signed return after a continuous conditional mean is removed.

All models use the same TRAIN-only Student-t(6) scale and RD01 trailing-RV proxy.
A variance-preserving, mean-centered two-piece Student-t has skew parameter k;
k=0 recovers the symmetric Student-t exactly. No regime-specific scale/mean
adjustment is allowed. No HMM, HSMM, AM-T-HAR-Blend modification, or claim of
independent prospective confirmation is made.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Mapping
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln
from scipy.stats import t as student_t
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor, export_text
from data_engineering.modeling import observable_regime_discovery as rd
from data_engineering.modeling import observable_regime_discovery_b as rb

VERSION='RD01c-0.1.0'
HORIZONS=(5,15,30,60)
METHODS=('direct_symmetric','interaction_symmetric','direct_global_skew',
         'direct_continuous_skew','direct_regime_skew')
CONTINUOUS_COLS=('mom_r5','flow5','basis_bps')
PRIMARY=30

@dataclass(frozen=True)
class Spec:
    version:str=VERSION
    assets:tuple[str,...]=('BTC','ETH','SOL','XRP')
    horizons:tuple[int,...]=HORIZONS
    primary_horizon_min:int=PRIMARY
    student_df:float=6.
    ridge_alpha:float=100.
    structure_early_fraction:float=.65
    tree_max_depth:int=3
    tree_max_leaf_nodes:int=5
    tree_min_leaf_fraction:float=.08
    tree_min_leaf_absolute:int=50
    max_skew:float=.55
    skew_shrinkage:float=350.
    continuous_l2:float=.04
    minimum_state_events:int=200
    minimum_state_days:int=20
    bootstrap_repetitions:int=300
    minimum_bootstrap_days:int=20
    random_state:int=2610
    status:str='HISTORICAL_DISCOVERY_NOT_CONFIRMATION'

def registration():return asdict(Spec())


def split_t_components(k,df=6.):
    """Centre and rescale split Student-t to preserve its zero mean and variance.

    Original split density is 2/(a+b) t(z/a) on z<0 and 2/(a+b) t(z/b)
    on z>=0, where a=e^-k and b=e^k. q=(z-mean_z)/factor has exactly
    mean 0, variance df/(df-2). At k=0 q has the usual Student-t(df) law.
    """
    if df<=2:raise ValueError('Student-t variance must be finite')
    k=np.asarray(k,float)
    if not np.isfinite(k).all():raise ValueError('nonfinite skew')
    a=np.exp(-k);b=np.exp(k)
    mean_abs=2*np.sqrt(df)*np.exp(gammaln((df+1)/2)-gammaln(df/2))/((df-1)*np.sqrt(np.pi))
    mean=mean_abs*(b-a)
    second=(df/(df-2))*(a*a*a+b*b*b)/(a+b)
    var=second-mean*mean
    if not np.isfinite(var).all() or np.any(var<=0):raise ValueError('invalid split-t variance')
    factor=np.sqrt(var/(df/(df-2)))
    return a,b,mean,factor


def split_t_logpdf(q,k,df=6.):
    q=np.asarray(q,float); k=np.asarray(k,float)
    a,b,m,f=split_t_components(k,df)
    z=q*f+m
    side=np.where(z<0,a,b)
    return np.log(2)-np.log(a+b)+student_t.logpdf(z/side,df=df)+np.log(f)


def split_t_cdf(q,k,df=6.):
    q=np.asarray(q,float);k=np.asarray(k,float)
    a,b,m,f=split_t_components(k,df);z=q*f+m
    p=np.where(z<0,2*a/(a+b)*student_t.cdf(z/a,df=df),
               a/(a+b)+2*b/(a+b)*(student_t.cdf(z/b,df=df)-.5))
    return np.clip(p,0,1)


def _std_outcome(frame:pd.DataFrame,h:int):
    return frame[f'ret_{h}m'].to_numpy(float)/rd.volatility_proxy(frame,h)


def _common_location_fit(cal:pd.DataFrame,val:pd.DataFrame,h:int,cfg:Spec):
    """No downstream skew model may change continuous location or common scale."""
    y=_std_outcome(cal,h)
    if not np.isfinite(y).all():raise ValueError('calibration target nonfinite')
    lo,hi=np.quantile(y,[.005,.995]); clipped=np.clip(y,lo,hi)
    a,b=rb.feature_design(cal,val,interactions=False)
    direct=Ridge(alpha=cfg.ridge_alpha).fit(a,clipped)
    ai,bi=rb.feature_design(cal,val,interactions=True)
    interaction=Ridge(alpha=cfg.ridge_alpha).fit(ai,clipped)
    # Inherit RD01-b's common scale convention; same for every method.
    mu=float(clipped.mean()); residual=y-mu
    cap=np.quantile(residual*residual,.995)
    s=np.sqrt(max(np.minimum(residual*residual,cap).mean(),1e-8))*np.sqrt((cfg.student_df-2)/cfg.student_df)
    return direct.predict(a),direct.predict(b),interaction.predict(bi),float(s)


def _skew_sample(frame:pd.DataFrame,h:int):
    # Nonoverlapping return targets are used for skew calibration to reduce
    # dependency from repeated, heavily overlapping future-return labels.
    return rb._anchor_mask(frame,h)


def fit_global_skew(q:np.ndarray,cfg:Spec, *, anchor:np.ndarray|None=None):
    z=np.asarray(q,float)
    if anchor is not None:z=z[anchor]
    if len(z)<100 or not np.isfinite(z).all():raise ValueError('insufficient skew calibration')
    opt=minimize_scalar(lambda k:-float(split_t_logpdf(z,k,cfg.student_df).mean()),
                        bounds=(-cfg.max_skew,cfg.max_skew),method='bounded',options={'xatol':.0005})
    if not opt.success:raise RuntimeError('global split-t optimization failed')
    return float(opt.x)


def fit_skew_tree(structure:pd.DataFrame,h:int,cfg:Spec):
    """The tree is fit to out-of-sample early-Ridge residual skew on late STRUCTURE."""
    days=pd.DatetimeIndex(sorted(structure.date.unique()))
    cut=max(12,min(len(days)-12,int(len(days)*cfg.structure_early_fraction)))
    pre=structure[structure.date.isin(days[:cut])].copy()
    post=structure[structure.date.isin(days[cut:])].copy()
    pre=pre[np.isfinite(pre[f'ret_{h}m'])]
    post=post[np.isfinite(post[f'ret_{h}m'])&_skew_sample(post,h)]
    minimum=max(cfg.tree_min_leaf_absolute,int(np.ceil(len(post)*cfg.tree_min_leaf_fraction)))
    if len(pre)<500 or len(post)<minimum*3:raise ValueError('insufficient independent structure samples')
    prior=rb.ridge_predict(pre,post,h,rb.Spec(ridge_alpha=cfg.ridge_alpha))
    resid=_std_outcome(post,h)-prior
    # H3(x)=x^3-3x isolates third-moment shape rather than a pure mean split;
    # bounded observations stop rare shocks defining all splits.
    scale=max(np.std(np.clip(resid,*np.quantile(resid,[.005,.995]))),1e-6)
    z=np.clip(resid/scale,-3,3)
    target=z**3-3*z
    tree=DecisionTreeRegressor(max_depth=cfg.tree_max_depth,
        max_leaf_nodes=cfg.tree_max_leaf_nodes,min_samples_leaf=minimum,random_state=cfg.random_state)
    tree.fit(post[list(rd.TREE_COLS)],target)
    info={'horizon_min':h,'early_days':cut,'late_days':len(days)-cut,
          'fit_anchors':len(post),'leaves':int(tree.get_n_leaves()),
          'definition':export_text(tree,feature_names=list(rd.TREE_COLS),decimals=6)}
    return tree,info


def fit_continuous_skew(cal:pd.DataFrame,residual_q:np.ndarray,k_global:float,h:int,cfg:Spec):
    """Non-regime conditional skew comparator: smooth 3-covariate location-shape.

    Every scale and threshold is CALIBRATION-fitted, not validation fitted.
    X is clipped by calibration 0.5/99.5% quantiles for optimizer stability.
    """
    sel=_skew_sample(cal,h)
    frame=cal.loc[sel]
    q=np.asarray(residual_q,float)[sel]
    raw=frame[list(CONTINUOUS_COLS)].to_numpy(float)
    med=raw.mean(axis=0);sd=np.maximum(raw.std(axis=0),1e-10)
    limlo=np.quantile(raw,.005,axis=0);limhi=np.quantile(raw,.995,axis=0)
    x=(np.clip(raw,limlo,limhi)-med)/sd
    def loss(v):
        k=np.clip(v[0]+x@v[1:],-cfg.max_skew,cfg.max_skew)
        return -float(split_t_logpdf(q,k,cfg.student_df).mean())+cfg.continuous_l2*float(np.dot(v[1:],v[1:]))
    opt=minimize(loss,np.r_[k_global,np.zeros(x.shape[1])],method='L-BFGS-B',
                 bounds=[(-cfg.max_skew,cfg.max_skew)]+[(-.4,.4)]*x.shape[1],
                 options={'maxiter':55,'ftol':1e-9})
    if not opt.success:raise RuntimeError(f'continuous skew optimization failed: {opt.message}')
    return {'theta':opt.x,'mean':med,'std':sd,'low':limlo,'high':limhi,'fit_n':len(q)}


def predict_continuous_skew(frame:pd.DataFrame,fit:Mapping,cfg:Spec):
    x=frame[list(CONTINUOUS_COLS)].to_numpy(float)
    x=(np.clip(x,fit['low'],fit['high'])-fit['mean'])/fit['std']
    theta=fit['theta']
    return np.clip(theta[0]+x@theta[1:],-cfg.max_skew,cfg.max_skew)


def fit_state_skews(cal:pd.DataFrame,states:np.ndarray,q:np.ndarray,k_global:float,h:int,cfg:Spec):
    """State fits use nonoverlapping CAL origins and shrink toward global skew."""
    mask=_skew_sample(cal,h)
    labels=np.asarray(states,dtype=object)
    out=[]
    for state in sorted(set(labels)):
        sel=mask&(labels==state)
        z=np.asarray(q,float)[sel]
        days=int(cal.loc[sel,'date'].nunique())
        if len(z)<max(100,cfg.minimum_state_events) or days<cfg.minimum_state_days:
            k=float(k_global);status='SPARSE_FALLBACK'
        else:
            raw=fit_global_skew(z,cfg)
            shrink=len(z)/(len(z)+cfg.skew_shrinkage)
            k=float(k_global+shrink*(raw-k_global));status='CALIBRATED'
        out.append({'state':str(state),'skew':k,'n_cal_anchors':int(len(z)),
                    'n_cal_days':days,'status':status})
    return pd.DataFrame(out)


def score_density(val:pd.DataFrame,h:int,locations:Mapping[str,np.ndarray],
                  skews:Mapping[str,np.ndarray],scale:float,cfg:Spec,qtail:float):
    y=_std_outcome(val,h)
    vol=rd.volatility_proxy(val,h)
    rows=[];days=[]
    for name,mu in locations.items():
        k=np.asarray(skews[name],float)
        q=(y-np.asarray(mu,float))/scale
        ll=-split_t_logpdf(q,k,cfg.student_df)+np.log(scale)+np.log(vol)
        # Signed-return exceedance thresholds measured relative to the common
        # conditional LOCATION and SCALE (not outcome-fitted thresholds).
        hi=1-split_t_cdf(qtail,k,cfg.student_df)
        lo=split_t_cdf(-qtail,k,cfg.student_df)
        up=(q>qtail).astype(float); down=(q<-qtail).astype(float)
        brier_up=(up-hi)**2;brier_down=(down-lo)**2
        pred_up=1-split_t_cdf(-np.asarray(mu,float)/scale,k,cfg.student_df)
        obs_up=(y>0).astype(float)
        rows.append({'method':name,'horizon_min':h,'n':len(y),'nll':float(ll.mean()),
                     'up_tail_brier':float(brier_up.mean()),'down_tail_brier':float(brier_down.mean()),
                     'pred_up_mean':float(pred_up.mean()),'observed_up':float(obs_up.mean()),
                     'mean_skew':float(np.mean(k))})
        x=pd.DataFrame({'date':val.date,'nll':ll,'up_tail_brier':brier_up,
                        'down_tail_brier':brier_down})
        z=x.groupby('date').agg(nll=('nll','mean'),up_tail_brier=('up_tail_brier','mean'),
                                 down_tail_brier=('down_tail_brier','mean')).reset_index()
        z['method']=name;z['horizon_min']=h;days.append(z)
    return pd.DataFrame(rows),pd.concat(days,ignore_index=True)


def state_descriptives(val:pd.DataFrame,states:np.ndarray,locations:np.ndarray,scale:float,h:int,cfg:Spec):
    standardized=(_std_outcome(val,h)-locations)/scale
    z=pd.DataFrame({'state':states,'date':val.date.to_numpy(),
                    'standardized_residual':standardized,'return':val[f'ret_{h}m'].to_numpy(float)})
    out=[]
    for key,g in z.groupby('state'):
        n=len(g);days=g.date.nunique();r=g.standardized_residual.to_numpy(float)
        out.append({'state':str(key),'n':int(n),'n_days':int(days),'horizon_min':h,
                    'observed_up_rate':float((g['return']>0).mean()),
                    'residual_mean':float(r.mean()),
                    'residual_left_q05':float(np.quantile(r,.05)),
                    'residual_right_q95':float(np.quantile(r,.95)),
                    'residual_sign_rate':float((r>0).mean()),
                    'supported':bool(n>=cfg.minimum_state_events and days>=cfg.minimum_state_days)})
    return pd.DataFrame(out)



def temporal_diagnostics(frame:pd.DataFrame,states:np.ndarray,*,h:int=30):
    """Aggregate only; date boundaries censor episodes and do not form transitions.

    No sticky smoothing: dwell is entirely from observed raw CART labels. An IID
    per-day occupancy reference is descriptive, not a persistence p-value.
    """
    tags=np.asarray(states,dtype=object)
    if len(tags)!=len(frame):raise ValueError('state length mismatch')
    e_list=[];tr_list=[];hazard_list=[];persist=[]
    frame=frame.reset_index(drop=True)
    days=pd.to_datetime(frame.state_time,utc=True).dt.floor('D').to_numpy()
    for d in np.unique(days):
        ix=np.flatnonzero(days==d)
        day=frame.iloc[ix]
        labels=tags[ix]
        tr,ep,ha=rd.transitions_episodes(day,labels,'rd01c_raw_skew_tree')
        if not ep.empty:e_list.append(ep)
        if not tr.empty:tr_list.append(tr)
        if not ha.empty:hazard_list.append(ha)
        if len(labels)>1:
            for state in np.unique(labels):
                cnt=int(np.sum(labels[:-1]==state))
                if cnt:
                    persist.append({'state':str(state),'n_transitions':cnt,
                         'stay_count':int(np.sum((labels[:-1]==state)&(labels[1:]==state))),
                         'iid_expected_stay_count':cnt*float(np.mean(labels==state))})
    epall=pd.concat(e_list,ignore_index=True) if e_list else pd.DataFrame()
    trall=pd.concat(tr_list,ignore_index=True) if tr_list else pd.DataFrame()
    hzall=pd.concat(hazard_list,ignore_index=True) if hazard_list else pd.DataFrame()
    pp=pd.DataFrame(persist)
    if epall.empty:return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    episodes=epall.groupby('regime',as_index=False).agg(
        n_episodes=('n_bars','size'),mean_minutes=('duration_minutes','mean'),
        median_minutes=('duration_minutes','median'),right_censored=('right_censored','sum'))
    if trall.empty:transitions=pd.DataFrame()
    else:
        transitions=trall.groupby(['from_state','to_state'],as_index=False).agg(count=('count','sum'))
        transitions['n_from']=transitions.groupby('from_state')['count'].transform('sum')
        transitions['probability']=transitions['count']/transitions['n_from']
    if hzall.empty:hazards=pd.DataFrame()
    else:
        hazards=hzall.groupby(['regime','age_bars'],as_index=False).agg(at_risk=('at_risk','sum'),exits=('exits','sum'))
        hazards['exit_hazard']=hazards.exits/hazards.at_risk
    if pp.empty:persistence=pd.DataFrame()
    else:
        persistence=pp.groupby('state',as_index=False).agg(
            n_transitions=('n_transitions','sum'),stay_count=('stay_count','sum'),
            iid_expected_stay_count=('iid_expected_stay_count','sum'))
        persistence['observed_stay']=persistence.stay_count/persistence.n_transitions
        persistence['iid_expected_stay']=persistence.iid_expected_stay_count/persistence.n_transitions
    return episodes,transitions,hazards,persistence

def scan_asset(panel:pd.DataFrame,cfg:Spec,asset:str):
    """Fixed RD01 folds, no legacy final-20% selection and no forward leakage."""
    x=rd.forward_labels(panel)
    dates=rd.full_feature_days(x)
    x=x.loc[x.date.isin(dates)&x.feature_ready.fillna(False)].reset_index(drop=True)
    folds=rd.research_folds(dates,rd.Spec())
    all_scores=[];all_days=[];all_states=[];all_definitions=[];all_cal=[];all_episodes=[];all_transitions=[];all_hazards=[];all_persistence=[]
    for fold in folds:
        structure=x.loc[x.date.isin(fold['structure_days'])].copy()
        cal_all=x.loc[x.date.isin(fold['calibration_days'])].copy()
        val_all=x.loc[x.date.isin(fold['validation_days'])].copy()
        for h in cfg.horizons:
            tree,info=fit_skew_tree(structure,h,cfg)
            states_cal_all=rd.tree_labels(tree,cal_all)
            states_val_all=rd.tree_labels(tree,val_all)
            cm=np.isfinite(cal_all[f'ret_{h}m'].to_numpy(float))
            vm=np.isfinite(val_all[f'ret_{h}m'].to_numpy(float))
            cal=cal_all.loc[cm].copy();val=val_all.loc[vm].copy()
            states_cal=states_cal_all[cm];states_val=states_val_all[vm]
            mu_cal,mu_val,mu_inter,scale=_common_location_fit(cal,val,h,cfg)
            residual=( _std_outcome(cal,h)-mu_cal)/scale
            anchor=_skew_sample(cal,h)
            kg=fit_global_skew(residual,cfg,anchor=anchor)
            cont=fit_continuous_skew(cal,residual,kg,h,cfg)
            ks=predict_continuous_skew(val,cont,cfg)
            state_params=fit_state_skews(cal,states_cal,residual,kg,h,cfg)
            lookup=dict(zip(state_params['state'],state_params['skew']))
            kr=np.array([lookup.get(str(t),kg) for t in states_val],float)
            methods={
                'direct_symmetric':(mu_val,np.zeros(len(val))),
                'interaction_symmetric':(mu_inter,np.zeros(len(val))),
                'direct_global_skew':(mu_val,np.full(len(val),kg)),
                'direct_continuous_skew':(mu_val,ks),
                'direct_regime_skew':(mu_val,kr),
            }
            tailq=float(np.quantile(np.abs(residual[anchor]),.95))
            sc,dy=score_density(val,h,{k:v[0] for k,v in methods.items()},
                                {k:v[1] for k,v in methods.items()},scale,cfg,tailq)
            all_scores.append(sc.assign(asset=asset,fold=fold['fold']))
            all_days.append(dy.assign(asset=asset,fold=fold['fold']))
            all_states.append(state_descriptives(val,states_val,mu_val,scale,h,cfg).assign(asset=asset,fold=fold['fold']))
            all_cal.append(state_params.assign(asset=asset,fold=fold['fold'],horizon_min=h,
                                            global_skew=kg,continuous_coeff_norm=float(np.linalg.norm(cont['theta'][1:])),
                                            scale_proxy=scale))
            all_definitions.append({'asset':asset,'fold':fold['fold'],**info})
            if h==cfg.primary_horizon_min:
                ep,tr,hz,ps=temporal_diagnostics(val_all,states_val_all,h=h)
                for table,collector in ((ep,all_episodes),(tr,all_transitions),
                                        (hz,all_hazards),(ps,all_persistence)):
                    if not table.empty:collector.append(table.assign(asset=asset,fold=fold['fold'],horizon_min=h))
    return {'scores':pd.concat(all_scores,ignore_index=True),
            'daily':pd.concat(all_days,ignore_index=True),
            'states':pd.concat(all_states,ignore_index=True),
            'calibration':pd.concat(all_cal,ignore_index=True),
            'definitions':pd.DataFrame(all_definitions),
            'episodes':pd.concat(all_episodes,ignore_index=True) if all_episodes else pd.DataFrame(),
            'transitions':pd.concat(all_transitions,ignore_index=True) if all_transitions else pd.DataFrame(),
            'hazards':pd.concat(all_hazards,ignore_index=True) if all_hazards else pd.DataFrame(),
            'persistence':pd.concat(all_persistence,ignore_index=True) if all_persistence else pd.DataFrame(),
            'folds':folds,'full_days':len(dates)}


def paired_day_bootstrap(daily:pd.DataFrame,cfg:Spec):
    out=[]
    comparators=[('direct_regime_skew','direct_continuous_skew'),
                 ('direct_regime_skew','direct_global_skew'),
                 ('direct_regime_skew','direct_symmetric'),
                 ('direct_regime_skew','interaction_symmetric'),
                 ('direct_continuous_skew','direct_global_skew'),
                 ('direct_global_skew','direct_symmetric')]
    for (asset,fold,h),g in daily.groupby(['asset','fold','horizon_min']):
        q=g.pivot(index='date',columns='method',values=['nll','up_tail_brier','down_tail_brier'])
        for metric in ['nll','up_tail_brier','down_tail_brier']:
            for method,reference in comparators:
                d=(q[(metric,method)]-q[(metric,reference)]).dropna().to_numpy(float)
                if len(d)<cfg.minimum_bootstrap_days:continue
                rng=np.random.default_rng(cfg.random_state+int(fold)+int(h)+sum(map(ord,method)))
                ix=rng.integers(0,len(d),size=(cfg.bootstrap_repetitions,len(d)))
                samples=d[ix].mean(axis=1)
                lo,hi=np.quantile(samples,[.025,.975])
                out.append({'asset':asset,'fold':fold,'horizon_min':h,'metric':metric,
                            'model':method,'reference':reference,'delta':float(d.mean()),
                            'ci025':float(lo),'ci975':float(hi),'n_days':len(d),
                            'status':'HISTORICAL_UNADJUSTED'})
    return pd.DataFrame(out)


def consolidate(results:Mapping[str,dict],cfg:Spec,synthetic=False):
    keys=['scores','daily','states','calibration','definitions','episodes','transitions','hazards','persistence']
    out={k:pd.concat([v[k] for v in results.values()],ignore_index=True) for k in keys}
    out['bootstrap']=paired_day_bootstrap(out['daily'],cfg)
    summary=out['scores'].groupby(['horizon_min','method'],as_index=False).agg(
        mean_nll=('nll','mean'),mean_up_tail_brier=('up_tail_brier','mean'),
        mean_down_tail_brier=('down_tail_brier','mean'),n_asset_folds=('n','size'))
    z=summary[summary.method.eq('direct_symmetric')].set_index('horizon_min')
    summary['delta_nll_vs_symmetric']=summary.apply(lambda r:float(r.mean_nll-z.loc[r.horizon_min,'mean_nll']),axis=1)
    out['summary']=summary
    sub=out['scores'][out['scores'].horizon_min.eq(cfg.primary_horizon_min)]
    pivot=sub.pivot(index=['asset','fold'],columns='method',values='nll')
    comparisons=[]
    for ref in ('direct_continuous_skew','interaction_symmetric','direct_global_skew'):
        d=pivot['direct_regime_skew']-pivot[ref]
        primary_ok=all(asset in d.index.get_level_values('asset') and
                       float(d.xs(asset,level='asset').mean())<0 and
                       int((d.xs(asset,level='asset')<0).sum())>=2
                       for asset in ('BTC','ETH'))
        comparisons.append({'reference':ref,'mean_delta_nll':float(d.mean()),
                            'favorable_asset_folds':int((d<0).sum()),'n_asset_folds':len(d),
                            'BTC_ETH_each_2_of_3':bool(primary_ok),
                            'exploratory_favorable':bool(len(d)==12 and d.mean()<0 and
                                                          (d<0).sum()>=9 and primary_ok)})
    support={asset:int((g.supported).sum()) for asset,g in out['states'].query('horizon_min==30').groupby('asset')}
    out['decision']={'status':cfg.status,'primary_horizon_min':cfg.primary_horizon_min,
                    'primary_comparisons':comparisons,
                    'screening_candidate':bool(not synthetic and all(c['exploratory_favorable'] for c in comparisons)),
                    'state_support_counts':support,'independent_confirmation':False,
                    'AM_T_HAR_Blend_modified':False,
                    'note':'Historical folds already studied; testing different distributional shape after negative RD01-b is exploratory. Student-t scale is trailing-RV proxy, NOT AM-T-HAR-Blend.'}
    out['manifest']={'module':'RD01-c','registration':registration(),
                     'assets_evaluated':list(results),
                     'folds':[{'asset':a,'fold':f['fold'],'structure_end':f['structure_end'],
                               'calibration_start':f['calibration_start'],'validation_start':f['validation_start'],
                               'validation_end':f['validation_end']} for a,r in results.items() for f in r['folds']],
                     'local_input':'CS04-A per-origin parquet only',
                     'cutoff_utc':'2026-10-14T00:00:00Z',
                     'normalizer':'RD01_TRAILING_24H_RV_PROXY_NOT_AM_T_HAR_BLEND',
                     'historical_discovery_only':True,'legacy_test_used_for_selection':False,
                     'synthetic':bool(synthetic),'AM_T_HAR_Blend_modified':False}
    return out
