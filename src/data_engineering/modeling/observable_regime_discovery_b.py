"""RD01-b: causal residual regimes beyond continuous signed-return forecasts.

Historical exploratory research. Residual CART learns 5/15/30/60m signed-return
errors from an EARLY structure subperiod, independent of late structure tree
responses, and states receive shrinkage coefficients from a later CAL period.
Three-confirmation regime assignment is online and resets on discontinuities.
It creates persistence mechanically and is NOT evidence for structural regimes.
Volatility uses RD01's trailing RV proxy, NOT AM-T-HAR-Blend ex-ante variance.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text
from data_engineering.modeling import observable_regime_discovery as rd

VERSION='RD01b-0.1.0'
HORIZONS=(5,15,30,60)
METHODS=('baseline','direct_ridge','interaction_ridge','residual_tree','residual_tree_sticky')
INTERACTIONS=('mom_x_flow','flow_x_basis','mom_x_basis','lastminute_x_flow')
PRIMARY_HORIZON=30

@dataclass(frozen=True)
class Spec:
    version: str=VERSION
    assets: tuple[str,...]=('BTC','ETH','SOL','XRP')
    horizons: tuple[int,...]=HORIZONS
    primary_horizon_min: int=PRIMARY_HORIZON
    structure_early_fraction:float=.65
    tree_max_leaf_nodes:int=6
    tree_max_depth:int=3
    tree_min_leaf_fraction:float=.07
    tree_min_leaf_absolute:int=55
    confirm_bars:int=3
    ridge_alpha:float=100.
    regime_shrinkage:float=500.
    student_df:float=6.
    bootstrap_repetitions:int=300
    minimum_bootstrap_days:int=20
    minimum_state_events:int=200
    minimum_state_days:int=20
    random_state:int=2608
    status:str='HISTORICAL_DISCOVERY_NOT_CONFIRMATION'

def registration():return asdict(Spec())

def _finite(a):
    a=np.asarray(a,float)
    if not np.isfinite(a).all():raise ValueError('nonfinite input')
    return a

def feature_design(train:pd.DataFrame,test:pd.DataFrame,*,interactions:bool=False):
    """TRAIN-only scaling; named polynomial terms from TRAIN-standardized data."""
    terms=list(rd.TREE_COLS)
    scal=StandardScaler();a=scal.fit_transform(_finite(train[terms]));b=scal.transform(_finite(test[terms]))
    if interactions:
        idx={k:i for i,k in enumerate(terms)}
        pairs=[('mom_r5','flow5'),('flow5','basis_bps'),('mom_r5','basis_bps'),('last_1m_r','flow5')]
        xa=np.column_stack([a[:,idx[u]]*a[:,idx[v]] for u,v in pairs])
        xb=np.column_stack([b[:,idx[u]]*b[:,idx[v]] for u,v in pairs])
        # Interaction scaling is separately fit using TRAIN only.
        iscal=StandardScaler();xa=iscal.fit_transform(xa);xb=iscal.transform(xb)
        a=np.column_stack((a,xa));b=np.column_stack((b,xb))
    return a,b

def ridge_predict(train:pd.DataFrame,test:pd.DataFrame,h:int,cfg:Spec,*,interactions=False):
    y=_finite(train[f'ret_{h}m'].to_numpy(float)/rd.volatility_proxy(train,h))
    xtr,xte=feature_design(train,test,interactions=interactions)
    lo,hi=np.quantile(y,[.005,.995]);y=np.clip(y,lo,hi)
    mod=Ridge(alpha=cfg.ridge_alpha).fit(xtr,y)
    return np.asarray(mod.predict(xte),float)

def _anchor_mask(frame:pd.DataFrame,h:int)->np.ndarray:
    """Nonoverlapping target windows, UTC anchors, no look-ahead."""
    t=pd.to_datetime(frame.state_time,utc=True)
    return ((t.dt.hour*60+t.dt.minute)%h==0).to_numpy(bool)

def fit_residual_tree(structure:pd.DataFrame,h:int,cfg:Spec):
    """Causal structure split: early ridge; later residual-tree, no label reuse."""
    days=pd.DatetimeIndex(sorted(structure.date.unique()))
    cut=max(12,min(len(days)-12,int(len(days)*cfg.structure_early_fraction)))
    if cut<12:raise ValueError('insufficient early structure days')
    pre=structure.loc[structure.date.isin(days[:cut])].copy()
    post=structure.loc[structure.date.isin(days[cut:])].copy()
    pre=pre.loc[np.isfinite(pre[f'ret_{h}m'])]
    post=post.loc[np.isfinite(post[f'ret_{h}m'])&_anchor_mask(post,h)]
    if len(post)<cfg.tree_min_leaf_absolute*3:raise ValueError('too few residual-tree fit observations')
    prediction=ridge_predict(pre,post,h,cfg)
    actual=post[f'ret_{h}m'].to_numpy(float)/rd.volatility_proxy(post,h)
    residual=actual-prediction
    caps=np.quantile(residual,[.005,.995]);target=np.clip(residual,*caps)
    minimum=max(cfg.tree_min_leaf_absolute,int(np.ceil(len(post)*cfg.tree_min_leaf_fraction)))
    m=DecisionTreeRegressor(max_depth=cfg.tree_max_depth,max_leaf_nodes=cfg.tree_max_leaf_nodes,
                            min_samples_leaf=minimum,random_state=cfg.random_state)
    m.fit(post[list(rd.TREE_COLS)],target)
    return m,dict(early_days=cut,late_days=len(days)-cut,tree_fit_origins=len(post),
                   n_leaves=m.get_n_leaves(),definition=export_text(m,feature_names=list(rd.TREE_COLS),decimals=6))

def label_tree(tree:DecisionTreeRegressor,x:pd.DataFrame)->np.ndarray:
    return np.array([f'T{int(i):03d}' for i in tree.apply(x[list(rd.TREE_COLS)])],dtype=object)

def causal_confirmed_labels(frame:pd.DataFrame,raw:np.ndarray,confirm_bars:int=3)->np.ndarray:
    """Online run-state debounce. At t, inputs are raw labels through t only.

    First observation after each timestamp gap or UTC day boundary initializes
    to its current raw state; a change requires k consecutive current raw states.
    This induces dwell mechanically; no retrospective/Viterbi smoothing is used.
    """
    if confirm_bars<1:raise ValueError('confirmation must be >=1')
    x=rd._datetime_frame(frame)
    if len(raw)!=len(x):raise ValueError('state length mismatch')
    if not pd.to_datetime(frame.state_time,utc=True).reset_index(drop=True).equals(x.state_time):
        raise ValueError('state inputs must be chronologically sorted')
    arr=np.asarray(raw,dtype=object);out=np.empty(len(arr),dtype=object)
    if not len(arr):return out
    current=None;pending=None;count=0
    for i,tag in enumerate(arr):
        reset=(i==0 or x.state_time.iloc[i]-x.state_time.iloc[i-1]!=pd.Timedelta(minutes=5)
               or x.date.iloc[i]!=x.date.iloc[i-1])
        if reset:
            current=tag;pending=None;count=0
        elif tag==current:
            pending=None;count=0
        elif confirm_bars==1:
            current=tag;pending=None;count=0
        else:
            if tag==pending:count+=1
            else:pending=tag;count=1
            if count>=confirm_bars:current=tag;pending=None;count=0
        out[i]=current
    return out

def _regime_means(cal:pd.DataFrame,states:np.ndarray,residual:np.ndarray,cfg:Spec):
    df=pd.DataFrame({'state':states,'residual':_finite(residual)})
    cap=np.quantile(df.residual,[.005,.995]);df['residual']=df.residual.clip(*cap)
    g=df.groupby('state').residual.agg(['size','mean']);return {
        str(s):float(row['mean']*row['size']/(row['size']+cfg.regime_shrinkage))
        for s,row in g.iterrows()}

def fit_locations(cal:pd.DataFrame,val:pd.DataFrame,trees:Mapping[int,DecisionTreeRegressor],cfg:Spec,h:int,
                  cal_raw_all:np.ndarray,val_raw_all:np.ndarray,
                  cal_sticky_all:np.ndarray,val_sticky_all:np.ndarray):
    """Identical fixed-variance Student-t comparison; correction on Ridge residuals."""
    y=_finite(cal[f'ret_{h}m'].to_numpy(float)/rd.volatility_proxy(cal,h))
    lo,hi=np.quantile(y,[.005,.995]);wins=np.clip(y,lo,hi)
    global_mu=float(wins.mean())
    cap=np.quantile((y-global_mu)**2,.995)
    sigma2=max(np.minimum((y-global_mu)**2,cap).mean(),1e-8)
    tscale=np.sqrt(sigma2)*np.sqrt((cfg.student_df-2)/cfg.student_df)
    xa,xb=feature_design(cal,val,interactions=False)
    reg=Ridge(alpha=cfg.ridge_alpha).fit(xa,wins)
    pred_cal=reg.predict(xa);pred_val=reg.predict(xb)
    xia,xib=feature_design(cal,val,interactions=True)
    ir=Ridge(alpha=cfg.ridge_alpha).fit(xia,wins)
    interactions=ir.predict(xib)
    residual=y-pred_cal
    raw_lookup=_regime_means(cal,cal_raw_all,residual,cfg)
    sticky_lookup=_regime_means(cal,cal_sticky_all,residual,cfg)
    raw_add=np.array([raw_lookup.get(s,0.) for s in val_raw_all],float)
    sticky_add=np.array([sticky_lookup.get(s,0.) for s in val_sticky_all],float)
    predictions={'baseline':np.full(len(val),global_mu),
                 'direct_ridge':pred_val,'interaction_ridge':interactions,
                 'residual_tree':pred_val+raw_add,'residual_tree_sticky':pred_val+sticky_add}
    return predictions,tscale,{'n_cal':len(cal),'residual_correction_raw':raw_lookup,
                'residual_correction_sticky':sticky_lookup,'sigma_std':float(tscale)}

def score_horizon(valid:pd.DataFrame,predictions:Mapping[str,np.ndarray],tscale:float,cfg:Spec,h:int):
    y=valid[f'ret_{h}m'].to_numpy(float)/rd.volatility_proxy(valid,h)
    s=rd.volatility_proxy(valid,h)
    up=(valid[f'ret_{h}m'].to_numpy(float)>0)
    nz=(valid[f'ret_{h}m'].to_numpy(float)!=0)
    rows=[];daily=[]
    for method,p in predictions.items():
        p=_finite(p)
        if len(p)!=len(y):raise ValueError('prediction sample mismatch')
        loss=-student_t.logpdf((y-p)/tscale,df=cfg.student_df)+np.log(tscale)+np.log(s)
        prob=student_t.sf(-p/tscale,df=cfg.student_df)
        bl=np.full(len(y),np.nan)
        bl[nz]=-(up[nz]*np.log(np.clip(prob[nz],1e-10,1-1e-10))+
                 (~up[nz])*np.log1p(-np.clip(prob[nz],1e-10,1-1e-10)))
        rows.append({'method':method,'horizon_min':h,'n':len(y),'n_nonzero':int(nz.sum()),
                     'nll':float(np.mean(loss)),'binary_logloss':float(np.nanmean(bl)),
                     'pred_up_mean':float(np.mean(prob)),'actual_up_rate':float(np.mean(up[nz])),
                     'mean_location_std':float(np.mean(p))})
        d=pd.DataFrame({'date':valid.date.to_numpy(),'nll':loss,'binloss':bl})
        g=d.groupby('date').agg(nll=('nll','mean'),binloss=('binloss','mean'),n=('nll','size')).reset_index()
        g['method']=method;g['horizon_min']=h;daily.append(g)
    return pd.DataFrame(rows),pd.concat(daily,ignore_index=True)

def characterize_states(valid:pd.DataFrame,raw:np.ndarray,sticky:np.ndarray,predictions:Mapping[str,np.ndarray],
                         tscale:float,h:int,cfg:Spec):
    """Held-out state conditional return statistics and incremental Ridge residuals."""
    y=valid[f'ret_{h}m'].to_numpy(float)
    sy=y/rd.volatility_proxy(valid,h)
    continuous=np.asarray(predictions['direct_ridge'],float)
    reg={k:predictions[k] for k in ('residual_tree','residual_tree_sticky')}
    out=[]
    for method,labels in (('residual_tree',raw),('residual_tree_sticky',sticky)):
        f=pd.DataFrame({'date':valid.date.to_numpy(),'state':labels,'return':y,
                        'standardized':sy,'ridge_residual':sy-continuous,
                        'correction':np.asarray(reg[method])-continuous})
        for state,g in f.groupby('state'):
            out.append({'method':method,'state':str(state),'horizon_min':h,
                        'n':len(g),'n_days':g.date.nunique(),'mean_return_bps':float(g['return'].mean()*1e4),
                        'positive_rate':float((g['return']>0).mean()),'negative_rate':float((g['return']<0).mean()),
                        'mean_std_return':float(g.standardized.mean()),
                        'mean_residual_vs_ridge':float(g.ridge_residual.mean()),
                        'predicted_residual_correction':float(g.correction.mean()),
                        'valid_cohort':bool(len(g)>=cfg.minimum_state_events and g.date.nunique()>=cfg.minimum_state_days)})
    return pd.DataFrame(out)

def summarize_durations(valid:pd.DataFrame,raw:np.ndarray,sticky:np.ndarray):
    """Compute within-day episodes/transitions; daily resets are censoring, NOT exits."""
    valid=rd._datetime_frame(valid)
    persistence=[];episode_parts=[];trans_parts=[]
    for method,labs in (('residual_tree',raw),('residual_tree_sticky',sticky)):
        s_pairs=0;s_stays=0;null_sum=0.;eps=[];tr=[]
        for _,indices in valid.groupby('date',sort=True).indices.items():
            frame=valid.iloc[indices].copy()
            state=np.asarray(labs,dtype=object)[indices]
            if len(frame)<2:continue
            stats=rd.persistence_diagnostics(frame,state,method)
            pairs=int(stats['n_adjacent_pairs'])
            s_pairs+=pairs;s_stays+=pairs*stats['observed_stay']
            null_sum+=pairs*stats['within_day_iid_expected_stay']
            trans,ep,_=rd.transitions_episodes(frame,state,method)
            if not ep.empty:eps.append(ep)
            if not trans.empty:tr.append(trans)
        if s_pairs==0:raise ValueError('no adjacent state observations')
        p=s_stays/s_pairs;reference=null_sum/s_pairs
        persistence.append({'method':method,'n_adjacent_pairs':s_pairs,'observed_stay':p,
                            'within_day_iid_expected_stay':reference,
                            'excess_stay_vs_iid':p-reference,
                            'warning':'Three-bar confirmation induces persistence; within-day iid is NOT a temporal null'})
        ep=pd.concat(eps,ignore_index=True)
        ep['method']=method
        summary=rd.aggregate_episode_summary(ep.assign(asset='local',fold=0))
        episode_parts.append(summary.drop(columns=['asset','fold']))
        if tr:
            combined=pd.concat(tr,ignore_index=True).groupby(['method','from_state','to_state'],as_index=False)['count'].sum()
            combined['n_from']=combined.groupby(['method','from_state'])['count'].transform('sum')
            combined['probability']=combined['count']/combined['n_from']
            trans_parts.append(combined)
    return pd.DataFrame(persistence),pd.concat(episode_parts,ignore_index=True),pd.concat(trans_parts,ignore_index=True)

def scan_asset(panel:pd.DataFrame,cfg:Spec,asset:str):
    """Same RD01 chronological folds, no reuse of CS03b legacy last-20% TEST."""
    original=rd.Spec()
    x=rd.forward_labels(panel)
    dates=rd.full_feature_days(x)
    x=x.loc[x.date.isin(dates)&x.feature_ready.fillna(False)].reset_index(drop=True)
    folds=rd.research_folds(dates,original)
    scores=[];daily=[];characteristics=[];persistence=[];episodes=[];transitions=[];definitions=[]
    for fold in folds:
        structure=x.loc[x.date.isin(fold['structure_days'])].copy()
        cal=x.loc[x.date.isin(fold['calibration_days'])].copy()
        valid=x.loc[x.date.isin(fold['validation_days'])].copy()
        for h in cfg.horizons:
            tree,info=fit_residual_tree(structure,h,cfg)
            definitions.append({'asset':asset,'fold':fold['fold'],'horizon_min':h,**info})
            # Label ALL cal/validation 5m observations before h-specific target removal.
            rc=label_tree(tree,cal);rv=label_tree(tree,valid)
            sc=causal_confirmed_labels(cal,rc,cfg.confirm_bars)
            sv=causal_confirmed_labels(valid,rv,cfg.confirm_bars)
            cm=np.isfinite(cal[f'ret_{h}m'].to_numpy(float));vm=np.isfinite(valid[f'ret_{h}m'].to_numpy(float))
            c=cal.loc[cm];v=valid.loc[vm]
            p,tscale,meta=fit_locations(c,v,{h:tree},cfg,h,rc[cm],rv[vm],sc[cm],sv[vm])
            a,d=score_horizon(v,p,tscale,cfg,h)
            scores.append(a.assign(asset=asset,fold=fold['fold']))
            daily.append(d.assign(asset=asset,fold=fold['fold']))
            characteristics.append(characterize_states(v,rv[vm],sv[vm],p,tscale,h,cfg).assign(asset=asset,fold=fold['fold']))
            if h==PRIMARY_HORIZON:
                ps,ep,trans=summarize_durations(valid,rv,sv)
                persistence.append(ps.assign(asset=asset,fold=fold['fold']))
                episodes.append(ep.assign(asset=asset,fold=fold['fold']))
                transitions.append(trans.assign(asset=asset,fold=fold['fold']))
    return {'scores':pd.concat(scores,ignore_index=True), 'daily':pd.concat(daily,ignore_index=True),
            'characteristics':pd.concat(characteristics,ignore_index=True),
            'persistence':pd.concat(persistence,ignore_index=True),
            'episode_summary':pd.concat(episodes,ignore_index=True),
            'transitions':pd.concat(transitions,ignore_index=True),
            'definitions':pd.DataFrame(definitions),'folds':folds,'n_days':len(dates)}

def paired_bootstrap(daily:pd.DataFrame,cfg:Spec)->pd.DataFrame:
    out=[]
    for (asset,fold,h),z in daily.groupby(['asset','fold','horizon_min']):
        q=z.pivot(index='date',columns='method',values=['nll','binloss'])
        for metric in ('nll','binloss'):
            for model in METHODS[1:]:
                for ref in (('baseline','direct_ridge','interaction_ridge') if model.startswith('residual_tree') else ('baseline',)):
                    if ref==model:continue
                    delta=(q[(metric,model)]-q[(metric,ref)]).dropna().to_numpy(float)
                    if len(delta)<cfg.minimum_bootstrap_days:continue
                    rng=np.random.default_rng(cfg.random_state+int(fold)+h+int(sum(ord(c) for c in model)))
                    ix=rng.integers(0,len(delta),size=(cfg.bootstrap_repetitions,len(delta)))
                    samples=delta[ix].mean(axis=1);lo,hi=np.quantile(samples,[.025,.975])
                    out.append({'asset':asset,'fold':fold,'horizon_min':h,'metric':metric,
                                'method':model,'reference':ref,'n_days':len(delta),
                                'delta':float(delta.mean()),'ci025':float(lo),'ci975':float(hi),
                                'status':'EXPLORATORY_UNADJUSTED'})
    return pd.DataFrame(out)

def consolidate(results:Mapping[str,dict],cfg:Spec)->dict:
    keys=('scores','daily','characteristics','persistence','episode_summary','transitions','definitions')
    out={k:pd.concat([r[k] for r in results.values()],ignore_index=True) for k in keys}
    out['bootstrap']=paired_bootstrap(out['daily'],cfg)
    summary=out['scores'].groupby(['method','horizon_min'],as_index=False).agg(
        n_asset_folds=('asset','size'),mean_nll=('nll','mean'),mean_binary_logloss=('binary_logloss','mean'))
    ref=summary[summary.method.eq('direct_ridge')].set_index('horizon_min')
    summary['delta_nll_vs_direct']=summary.apply(lambda z:float(z.mean_nll-ref.loc[z.horizon_min,'mean_nll']),axis=1)
    summary['delta_binary_vs_direct']=summary.apply(lambda z:float(z.mean_binary_logloss-ref.loc[z.horizon_min,'mean_binary_logloss']),axis=1)
    out['comparison_summary']=summary
    sub=out['scores'][out['scores'].horizon_min.eq(cfg.primary_horizon_min)]
    wide=sub.pivot(index=['asset','fold'],columns='method',values='nll')
    candidates=['residual_tree','residual_tree_sticky']
    verdict=[]
    for model in candidates:
        diff=wide[model]-wide['interaction_ridge']
        # Purely historical descriptive gate (selection-adjusted evidence impossible here).
        verdict.append({'model':model,'primary_horizon_min':cfg.primary_horizon_min,
                        'average_delta_nll_vs_interaction':float(diff.mean()),
                        'favorable_asset_folds':int((diff<0).sum()),
                        'n_asset_folds':int(len(diff)),
                        'historical_candidate':bool(diff.mean()<0 and (diff<0).sum()>=9),
                        'independent_confirmation':False})
    out['decision']={'status':cfg.status,'primary_horizon_min':cfg.primary_horizon_min,
                     'descriptive_candidates':verdict,
                     'no_prospective_claim':True,'AM_T_HAR_Blend_modified':False,
                     'note':'Persistence from three-bar confirmation is mechanically induced; cannot prove structural regimes.'}
    out['manifest']={'module':'RD01-b','registration':registration(),'assets_evaluated':list(results),
                     'folds':[{'asset':asset,'fold':f['fold'],
                               'structure_end':f['structure_end'],
                               'calibration_start':f['calibration_start'],
                               'validation_start':f['validation_start'],
                               'validation_end':f['validation_end']} for asset,r in results.items() for f in r['folds']],
                     'n_full_days':{asset:r['n_days'] for asset,r in results.items()},
                     'source':'LOCAL CS04-A per-origin Parquet, causal RD01 labels',
                     'normalization':'TRAILING_24H_VARIANCE_PROXY_NOT_AM_T_HAR_BLEND',
                     'legacy_test_used_for_selection':False,'independent_confirmation':False,
                     'AM_T_HAR_Blend_modified':False,
                     'disclaimer':'All historical development comparisons exploratory; repeated development inspection.'}
    return out
