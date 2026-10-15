"""RD01: causal observable regime discovery from the local CS04-A feature store.

Historical discovery only. Not an HMM and NOT a test of AM-T-HAR-Blend: its
point-in-time density/variance forecasts are not in the CS04-A saved features.

Observable states:
* fixed 3x3x3 quantile-rule grid (momentum, aggressive flow, futures basis)
* shallow CART partition fit to volatility-standardized NEXT 30m return.

Structure fits on an EARLY training block; location/scale calibration is done
only on a later disjoint training block; chronological validation remains unseen.
Regime-conditioned location is scored with a COMMON train-only Student-t scale.
No regime-specific scale fitting may produce a spurious 'directional' gain.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Mapping
import json
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

VERSION = 'RD01-0.1.0'
HORIZONS = (5,15,30,60)
BASE_COLS = ('mom_r5','flow5','basis_bps')
TREE_COLS = ('mom_r5','mom_r15_mean','flow5','flow30','basis_bps',
             'basis_change_bps','last_1m_r','minute_accel','log_trades')
METHODS = ('baseline','direct_ridge','rule_regime','tree_regime')
REGIMES = tuple(f'M{m}/F{f}/B{b}' for m in ('L','N','H')
                for f in ('L','N','H') for b in ('L','N','H'))

@dataclass(frozen=True)
class Spec:
    version: str=VERSION
    assets: tuple[str,...]=('BTC','ETH','SOL','XRP')
    horizons: tuple[int,...]=HORIZONS
    discovery_fraction: float=.80
    fold_positions: tuple[float,...]=(.50,.64,.78)
    validation_days_fraction: float=.08
    embargo_days: int=1
    structure_fraction: float=.70
    rule_quantiles: tuple[float,float]=(1/3,2/3)
    tree_max_leaf_nodes: int=8
    tree_max_depth: int=3
    tree_min_leaf_fraction: float=.04
    tree_min_leaf_absolute: int=120
    tree_anchor_minutes: tuple[int,int]=(0,30)
    ridge_alpha: float=100.0
    regime_shrinkage: float=350.
    student_df: float=6.
    minimum_diagnostic_events: int=200
    bootstrap_repetitions: int=300
    random_state: int=2607
    status: str='HISTORICAL_DISCOVERY_NOT_CONFIRMATION'

def registration():return asdict(Spec())

def _datetime_frame(frame:pd.DataFrame)->pd.DataFrame:
    x=frame.copy();x['state_time']=pd.to_datetime(x.state_time,utc=True)
    if x.state_time.isna().any() or x.state_time.duplicated().any():
        raise ValueError('Invalid or duplicate 5m timestamps')
    x=x.sort_values('state_time').reset_index(drop=True)
    if (x.state_time.dt.minute.mod(5).ne(0)|x.state_time.dt.second.ne(0)).any():
        raise ValueError('State timestamps must be 5-minute grid')
    x['date']=x.state_time.dt.floor('D')
    return x

def forward_labels(panel:pd.DataFrame,horizons=HORIZONS)->pd.DataFrame:
    """Rebuild 5/15/30/60m net signed return using strictly future 5m bars.

    Missing intermediate bars and any target crossing UTC midnight => NaN.
    Existing CS04-A labels are independently verified, never blindly trusted.
    """
    x=_datetime_frame(panel)
    rr=x.return_5m.to_numpy(float);tt=x.state_time.to_numpy();n=len(x)
    for h in horizons:
        if h<=0 or h%5:raise ValueError('Horizon must be positive multiple of 5m')
        k=h//5; sums=np.zeros(n,float);good=np.ones(n,bool)
        for j in range(1,k+1):
            y=np.full(n,np.nan);y[:n-j]=rr[j:]
            adjacent=np.zeros(n,bool);adjacent[:n-j]=tt[j:]-tt[:n-j]==np.timedelta64(5*j,'m')
            good &= adjacent&np.isfinite(y)
            sums+=np.nan_to_num(y,nan=0.)
        same=(x.state_time+pd.Timedelta(minutes=h)).dt.floor('D').eq(x.date).to_numpy()
        name=f'ret_{h}m';result=np.where(good&same,sums,np.nan)
        if name in x:
            previous=x[name].to_numpy(float)
            # Ignore preexisting unknown values only where target is undefined;
            # finite precomputed labels must agree exactly with causal rebuilding.
            both=np.isfinite(previous)&np.isfinite(result)
            if both.any() and not np.allclose(previous[both],result[both],rtol=1e-9,atol=1e-12):
                raise ValueError(f'Existing {name} disagrees with causal next-bar target')
        x[name]=result
    return x

def full_feature_days(panel:pd.DataFrame)->pd.DatetimeIndex:
    x=_datetime_frame(panel)
    req=['feature_ready',*BASE_COLS,*TREE_COLS,'log_rv24h','return_5m']
    missing=set(req)-set(x.columns)
    if missing:raise ValueError(f'Missing CS04-A feature columns: {sorted(missing)}')
    ready=x.feature_ready.fillna(False).to_numpy(bool)&np.isfinite(x[list(set(req)-{'feature_ready'})].to_numpy(float)).all(axis=1)
    t=x.state_time
    g=pd.DataFrame({'date':x.date,'time':t,'ready':ready}).groupby('date').agg(
        n=('time','size'),ready=('ready','sum'),first=('time','min'),last=('time','max'))
    mask=g.n.eq(288)&g.ready.eq(288)&g['first'].eq(g.index)&g['last'].eq(g.index+pd.Timedelta(hours=23,minutes=55))
    return pd.DatetimeIndex(g.index[mask])

def research_folds(days:pd.DatetimeIndex,cfg:Spec)->list[dict]:
    d=pd.DatetimeIndex(sorted(days));disc=d[:int(len(d)*cfg.discovery_fraction)]
    if len(disc)<55:raise ValueError('Need >=55 complete discovery days')
    width=max(5,int(len(disc)*cfg.validation_days_fraction));out=[]
    for i,p in enumerate(cfg.fold_positions,1):
        start=int(len(disc)*p);end=min(len(disc),start+width)
        train_end=start-cfg.embargo_days
        if train_end<25 or end-start<5:raise ValueError('Insufficient fold days')
        cut=max(12,int(train_end*cfg.structure_fraction))
        if cut>=train_end-8:raise ValueError('Calibration block too short')
        out.append(dict(fold=i,structure_days=disc[:cut],calibration_days=disc[cut:train_end],
                        validation_days=disc[start:end],
                        structure_end=str(disc[cut-1]),calibration_start=str(disc[cut]),
                        calibration_end=str(disc[train_end-1]),validation_start=str(disc[start]),
                        validation_end=str(disc[end-1]),embargo_days=cfg.embargo_days))
    return out

def volatility_proxy(x:pd.DataFrame,h:int)->np.ndarray:
    """Ex-ante trailing 24h squared 5m return * number of future 5m bars.

    Reconstruct rv_slow from CS04-A's saved log_rv24h (raw rv_slow is not saved).

    A RESEARCH NORMALIZER, not the actual AM-T-HAR-Blend forecast.
    """
    return np.sqrt(np.maximum(np.exp(np.clip(x.log_rv24h.to_numpy(float),-70,20)),1e-12)*(h//5))

def structure_rules(structure:pd.DataFrame,cfg:Spec)->dict:
    out={}
    for col in BASE_COLS:
        cut=np.quantile(structure[col].to_numpy(float),cfg.rule_quantiles)
        if not np.isfinite(cut).all() or cut[0]>=cut[1]:raise ValueError(f'Invalid terciles: {col}')
        out[col]=[float(v) for v in cut]
    return out

def assign_rules(x:pd.DataFrame,cuts:Mapping)->np.ndarray:
    codes=[]
    for col in BASE_COLS:
        codes.append(np.searchsorted(cuts[col],x[col].to_numpy(float),side='right'))
    a,b,c=codes
    return np.asarray([REGIMES[int(9*i+3*j+k)] for i,j,k in zip(a,b,c)],dtype=object)

def fit_tree(structure:pd.DataFrame,cfg:Spec)->tuple[DecisionTreeRegressor,str]:
    z=structure.loc[structure.state_time.dt.minute.isin(cfg.tree_anchor_minutes)].copy()
    z=z[np.isfinite(z.ret_30m.to_numpy(float))]
    if len(z)<cfg.tree_min_leaf_absolute*3:raise ValueError('Too few tree-structure 30m anchors')
    y=z.ret_30m.to_numpy(float)/volatility_proxy(z,30)
    # TRAIN-only clipping stops rare tails dominating least-squares tree splitting.
    lo,hi=np.quantile(y,[.005,.995]);y=np.clip(y,lo,hi)
    minimum=max(cfg.tree_min_leaf_absolute,int(np.ceil(len(z)*cfg.tree_min_leaf_fraction)))
    tree=DecisionTreeRegressor(max_depth=cfg.tree_max_depth,
        max_leaf_nodes=cfg.tree_max_leaf_nodes,min_samples_leaf=minimum,
        random_state=cfg.random_state)
    tree.fit(z[list(TREE_COLS)],y)
    return tree,export_text(tree,feature_names=list(TREE_COLS),decimals=5)

def tree_labels(tree:DecisionTreeRegressor,x:pd.DataFrame)->np.ndarray:
    return np.asarray([f'T{leaf:03d}' for leaf in tree.apply(x[list(TREE_COLS)])],dtype=object)

def _winsor(y:np.ndarray,lohi:tuple)->np.ndarray:
    return np.clip(y,lohi[0],lohi[1])

def fit_predict_locations(cal:pd.DataFrame,valid:pd.DataFrame,
                          cal_states:Mapping[str,np.ndarray],valid_states:Mapping[str,np.ndarray],
                          cfg:Spec,h:int)->tuple[dict,dict]:
    """Compare one TRAIN-calibrated location for four methods at shared fixed scale."""
    cy=cal[f'ret_{h}m'].to_numpy(float)/volatility_proxy(cal,h)
    vy=valid[f'ret_{h}m'].to_numpy(float)/volatility_proxy(valid,h)
    lo,hi=np.quantile(cy,[.005,.995]);yc=_winsor(cy,(lo,hi))
    global_mu=float(yc.mean())
    # One common robust standardized residual scale: do NOT tune it by regime.
    residual=(cy-global_mu)**2
    cap=np.quantile(residual,.995)
    common_sigma=float(np.sqrt(np.maximum(np.minimum(residual,cap).mean(),1e-8)))
    locations={'baseline':np.full(len(valid),global_mu)}
    params={'mean':global_mu,'common_sigma':common_sigma,'clipping':[float(lo),float(hi)],
            'horizon_min':h,'n_calibration':len(cal),'states':{}}
    scaler=StandardScaler()
    zz=scaler.fit_transform(cal[list(TREE_COLS)].to_numpy(float))
    ridge=Ridge(alpha=cfg.ridge_alpha)
    ridge.fit(zz,yc)
    locations['direct_ridge']=ridge.predict(scaler.transform(valid[list(TREE_COLS)].to_numpy(float)))
    params['ridge_coef']=dict(zip(TREE_COLS,map(float,ridge.coef_)))
    for method in ('rule_regime','tree_regime'):
        q=pd.DataFrame({'state':cal_states[method],'target':yc})
        means=q.groupby('state').target.agg(['mean','size'])
        shrink=(means['size']/(means['size']+cfg.regime_shrinkage)).to_numpy(float)
        post=global_mu+shrink*(means['mean'].to_numpy(float)-global_mu)
        lookup=dict(zip(means.index,post))
        locations[method]=np.asarray([lookup.get(s,global_mu) for s in valid_states[method]],float)
        params['states'][method]={str(k):{'n':int(means.loc[k,'size']),'mu_standardized':float(v)}
                                   for k,v in lookup.items()}
    # Student t scale adjusted for exact variance=common_sigma^2.
    tscale=common_sigma*np.sqrt((cfg.student_df-2)/cfg.student_df)
    return locations,{'params':params,'y_validation':vy,'sigma':tscale}

def score_returns(valid:pd.DataFrame,locations:Mapping[str,np.ndarray],aux:dict,cfg:Spec,h:int)->tuple[pd.DataFrame,pd.DataFrame]:
    y=aux['y_validation'];scale=aux['sigma'];ex=volatility_proxy(valid,h)
    actual_up=valid[f'ret_{h}m'].to_numpy(float)>0
    mask=valid[f'ret_{h}m'].to_numpy(float)!=0
    rows=[];day_parts=[]
    for method,mu in locations.items():
        density=student_t.logpdf((y-mu)/scale,df=cfg.student_df)-np.log(scale)-np.log(ex)
        nll=-density
        pu=np.clip(student_t.sf((-mu)/scale,df=cfg.student_df),1e-8,1-1e-8)
        # sf(-mu/scale) = P(standardized innovation > -mu/scale)
        binary_ll=np.full(len(valid),np.nan);brier=np.full(len(valid),np.nan)
        binary_ll[mask]=-(actual_up[mask]*np.log(pu[mask])+
                          (~actual_up[mask])*np.log1p(-pu[mask]))
        brier[mask]=(actual_up[mask]-pu[mask])**2
        rows.append(dict(method=method,horizon_min=h,n=len(valid),n_nonzero=int(mask.sum()),
            nll=float(nll.mean()),binary_logloss=float(np.nanmean(binary_ll)),
            binary_brier=float(np.nanmean(brier)),pred_up_mean=float(pu.mean()),
            actual_up_rate=float(actual_up[mask].mean()),
            loc_std_mean=float(np.mean(mu)),loc_std_sd=float(np.std(mu)),
            mean_abs_return_bps=float(np.mean(np.abs(valid[f'ret_{h}m']))*1e4)))
        tmp=pd.DataFrame({'date':valid.date,'method':method,'horizon_min':h,
                          'nll':nll,'bin_loss':binary_ll,'n_nonzero':mask.astype(int)})
        daily=tmp.groupby(['date','method','horizon_min'],as_index=False).agg(
            mean_nll=('nll','mean'),mean_binary_logloss=('bin_loss','mean'),
            n=('nll','size'),n_nonzero=('n_nonzero','sum'))
        day_parts.append(daily)
    return pd.DataFrame(rows),pd.concat(day_parts,ignore_index=True)

def state_forecast_calibration(valid:pd.DataFrame,labels:Mapping[str,np.ndarray],
                               locations:Mapping[str,np.ndarray],aux:dict,
                               cfg:Spec,h:int)->pd.DataFrame:
    """Summarize frozen ex-ante regime forecasts by identified state.

    Actual outcomes appear only in held-out descriptive columns, never in a
    fitted predictor or in state assignment.
    """
    y=aux['y_validation'];scale=aux['sigma'];ex=volatility_proxy(valid,h)
    observed=valid[f'ret_{h}m'].to_numpy(float)
    baseline_nll=-student_t.logpdf((y-locations['baseline'])/scale,df=cfg.student_df)+np.log(scale)+np.log(ex)
    out=[]
    for method,states in labels.items():
        mu=locations[method]
        probability=student_t.sf(-mu/scale,df=cfg.student_df)
        loss=-student_t.logpdf((y-mu)/scale,df=cfg.student_df)+np.log(scale)+np.log(ex)
        frame=pd.DataFrame({'state':states,'date':valid.date.to_numpy(),
            'pred_up':probability,'mu':mu,'nll_delta':loss-baseline_nll,
            'up':(observed>0).astype(float),'nonzero':(observed!=0).astype(int)})
        for state,g in frame.groupby('state'):
            nz=g.loc[g.nonzero.eq(1)]
            out.append(dict(method=method,regime=str(state),horizon_min=h,
                n=len(g),n_nonzero=len(nz),n_days=int(g.date.nunique()),
                pred_up=float(g.pred_up.mean()),observed_up_nonzero=float(nz.up.mean()) if len(nz) else np.nan,
                mean_mu_standardized=float(g.mu.mean()),
                nll_delta_vs_baseline=float(g.nll_delta.mean()),
                diagnostic_sufficient=bool(len(g)>=cfg.minimum_diagnostic_events and g.date.nunique()>=20)))
    return pd.DataFrame(out)


def aggregate_episode_summary(episodes:pd.DataFrame)->pd.DataFrame:
    """Compact output. Do not push raw state-episode timestamps or identities."""
    out=[]
    for (asset,fold,method,regime),g in episodes.groupby(['asset','fold','method','regime']):
        uncensored=g.loc[~(g.left_censored|g.right_censored)]
        out.append(dict(asset=asset,fold=int(fold),method=method,regime=regime,
            n_episodes=len(g),n_uncensored=len(uncensored),
            n_right_censored=int(g.right_censored.sum()),
            mean_duration_minutes=float(uncensored.duration_minutes.mean()) if len(uncensored) else np.nan,
            median_duration_minutes=float(uncensored.duration_minutes.median()) if len(uncensored) else np.nan,
            p90_duration_minutes=float(uncensored.duration_minutes.quantile(.9)) if len(uncensored) else np.nan,
            max_duration_minutes=float(g.duration_minutes.max())))
    return pd.DataFrame(out)


def summarize_states(valid:pd.DataFrame,labels:Mapping[str,np.ndarray],cfg:Spec)->pd.DataFrame:
    """Descriptive conditional signed-return distributions, not fitted predictions."""
    out=[]
    for method,states in labels.items():
        base=valid[['date','state_time']].copy();base['regime']=states
        for h in HORIZONS:
            usable=np.isfinite(valid[f'ret_{h}m'].to_numpy(float))
            g=base.loc[usable].copy();g['ret']=valid.loc[usable,f'ret_{h}m'].to_numpy(float)
            g['standardized']=g.ret.to_numpy(float)/volatility_proxy(valid.loc[usable],h)
            for regime,block in g.groupby('regime'):
                yy=block.ret.to_numpy(float);ss=block.standardized.to_numpy(float)
                out.append(dict(method=method,regime=str(regime),horizon_min=h,
                    n=len(block),n_days=int(block.date.nunique()),
                    mean_return_bps=float(yy.mean()*1e4),median_return_bps=float(np.median(yy)*1e4),
                    std_return_bps=float(np.std(yy)*1e4),
                    standardized_mean=float(ss.mean()),
                    positive_rate=float(np.mean(yy>0)),negative_rate=float(np.mean(yy<0)),
                    zero_rate=float(np.mean(yy==0)),
                    q05_bps=float(np.quantile(yy,.05)*1e4),
                    q95_bps=float(np.quantile(yy,.95)*1e4),
                    diagnostic_sufficient=bool(len(block)>=cfg.minimum_diagnostic_events and block.date.nunique()>=20)))
    return pd.DataFrame(out)

def transitions_episodes(frame:pd.DataFrame,states:np.ndarray,method:str)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Only chronological consecutive 5m origins; segment boundaries are censored."""
    x=_datetime_frame(frame)
    if len(states)!=len(x):raise ValueError('label count mismatch')
    state=np.asarray(states,dtype=object);t=x.state_time
    forward=t.shift(-1).sub(t).eq(pd.Timedelta(minutes=5)).fillna(False).to_numpy()
    pairs=pd.DataFrame({'from_state':state[:-1][forward[:-1]],
                        'to_state':state[1:][forward[:-1]]})
    trans=[]
    if not pairs.empty:
        for a,chunk in pairs.groupby('from_state'):
            counts=chunk.to_state.value_counts();total=int(counts.sum())
            for b,cnt in counts.items():
                trans.append(dict(method=method,from_state=str(a),to_state=str(b),
                                  count=int(cnt),probability=float(cnt/total),n_from=total))
    discontinuity=~np.r_[False,forward[:-1]]
    starts=discontinuity|np.r_[True,state[1:]!=state[:-1]]
    # At the first state in a contiguous run, mark right censor when next bar is unavailable.
    episode=np.cumsum(starts)
    temp=pd.DataFrame({'episode':episode,'state':state,'time':t})
    stats=temp.groupby('episode',sort=False).agg(
        regime=('state','first'),n_bars=('time','size'),first=('time','first'),last=('time','last'))
    starts_idx=np.flatnonzero(starts)
    ends_idx=np.r_[starts_idx[1:]-1,len(x)-1]
    stats['left_censored']=discontinuity[starts_idx]
    stats['right_censored']=~forward[ends_idx]
    ep=stats.reset_index(drop=True)
    ep['method']=method;ep['duration_minutes']=ep.n_bars*5
    ep['date']=ep['first'].dt.floor('D')
    hazards=[]
    complete=ep.loc[~ep.left_censored]
    for regime,g in complete.groupby('regime'):
        ages=sorted(set([1,2,3,4,5,6,9,12,18,24,36,48]))
        for a in ages:
            risk=int((g.n_bars>=a).sum())
            if risk==0:continue
            failures=int(((g.n_bars==a)&~g.right_censored).sum())
            hazards.append(dict(method=method,regime=str(regime),age_bars=a,
                                at_risk=risk,exits=failures,exit_hazard=failures/risk))
    return pd.DataFrame(trans),ep,pd.DataFrame(hazards)

def persistence_diagnostics(frame:pd.DataFrame,states:np.ndarray,method:str)->dict:
    """Observed five-minute persistence against within-day iid-occupancy reference.

    This occupancy reference is NOT a null preserving return autocorrelation or
    the serial dependence of the original features. It cannot confirm regimes.
    """
    x=_datetime_frame(frame);a=np.asarray(states,dtype=object)
    adjacent=(x.state_time.shift(-1)-x.state_time).eq(pd.Timedelta(minutes=5)).fillna(False).to_numpy()
    n=int(adjacent[:-1].sum());stay=int(((a[:-1]==a[1:])&adjacent[:-1]).sum())
    if n==0:raise ValueError('No sequential state pairs')
    reference=0.;denom=0
    for _,g in pd.DataFrame({'date':x.date,'state':a}).groupby('date'):
        ps=g.state.value_counts(normalize=True).to_numpy(float)
        valid=max(0,len(g)-1)
        reference+=valid*float(np.sum(ps**2));denom+=valid
    return {'method':method,'n_adjacent_pairs':n,'observed_stay':stay/n,
        'within_day_iid_expected_stay':reference/denom if denom else np.nan,
        'excess_stay_vs_iid':stay/n-reference/denom if denom else np.nan,
        'reference_warning':'Descriptive within-day occupancy only, not a temporal null'}


def daily_paired_deltas(daily:pd.DataFrame,cfg:Spec)->pd.DataFrame:
    """Equal-day paired loss differences, exploratory CIs (not selection adjusted)."""
    out=[]
    for (asset,fold,h),g in daily.groupby(['asset','fold','horizon_min']):
        for metric in ('mean_nll','mean_binary_logloss'):
            pivot=g.pivot(index='date',columns='method',values=metric)
            for m in METHODS[1:]:
                if 'baseline' not in pivot or m not in pivot:continue
                d=(pivot[m]-pivot.baseline).dropna().to_numpy(float)
                if len(d)<5:continue
                rng=np.random.default_rng(cfg.random_state+int(h)+fold)
                ix=rng.integers(0,len(d),size=(cfg.bootstrap_repetitions,len(d)))
                lo,hi=np.quantile(d[ix].mean(axis=1),[.025,.975])
                out.append(dict(asset=asset,fold=fold,horizon_min=h,metric=metric,
                    method=m,reference='baseline',n_days=len(d),delta=float(d.mean()),
                    ci025=float(lo),ci975=float(hi),status='EXPLORATORY_UNADJUSTED'))
    return pd.DataFrame(out)

def scan_asset(panel:pd.DataFrame,cfg:Spec,asset:str):
    x=forward_labels(panel)
    good=full_feature_days(x)
    x=x.loc[x.date.isin(good)&x.feature_ready.astype(bool)].copy().reset_index(drop=True)
    if not np.isfinite(x[list(set(BASE_COLS+TREE_COLS))+['log_rv24h']].to_numpy(float)).all():
        raise ValueError('Invalid state predictors')
    folds=research_folds(good,cfg)
    scores=[];byday=[];regimes=[];calibrations=[];trans=[];episodes=[];haz=[];rules=[];tree_txt=[];persistence=[]
    for fold in folds:
        f=fold['fold'];s=x.loc[x.date.isin(fold['structure_days'])].copy()
        c=x.loc[x.date.isin(fold['calibration_days'])].copy()
        v=x.loc[x.date.isin(fold['validation_days'])].copy()
        # Fit structure on earlier TRAIN only, including 30m outcome labels for CART.
        cuts=structure_rules(s,cfg)
        tree,definition=fit_tree(s,cfg)
        rules.append(dict(asset=asset,fold=f,method='rule_regime',structure=json.dumps(cuts)))
        tree_txt.append(dict(asset=asset,fold=f,method='tree_regime',structure=definition,
                             n_leaves=int(tree.get_n_leaves()),n_structure=int(len(s))))
        labels_c={'rule_regime':assign_rules(c,cuts),'tree_regime':tree_labels(tree,c)}
        labels_v={'rule_regime':assign_rules(v,cuts),'tree_regime':tree_labels(tree,v)}
        ss=s.copy(); ss['regime_rule']=assign_rules(ss,cuts)
        for h in HORIZONS:
            mask_c=np.isfinite(c[f'ret_{h}m'].to_numpy(float))
            mask_v=np.isfinite(v[f'ret_{h}m'].to_numpy(float))
            cal=c.loc[mask_c];val=v.loc[mask_v]
            cs={m:labs[mask_c] for m,labs in labels_c.items()}
            vs={m:labs[mask_v] for m,labs in labels_v.items()}
            if len(cal)<100 or len(val)<50:raise ValueError('Too few horizon labels')
            locations,aux=fit_predict_locations(cal,val,cs,vs,cfg,h)
            scores_h,day_h=score_returns(val,locations,aux,cfg,h)
            scores.append(scores_h.assign(asset=asset,fold=f))
            byday.append(day_h.assign(asset=asset,fold=f))
            calibrations.append(state_forecast_calibration(val,vs,locations,aux,cfg,h).assign(asset=asset,fold=f))
        reg=summarize_states(v,labels_v,cfg);regimes.append(reg.assign(asset=asset,fold=f))
        for method,lab in labels_v.items():
            tm,ep,ha=transitions_episodes(v,lab,method)
            persistence.append(dict(asset=asset,fold=f,**persistence_diagnostics(v,lab,method)))
            if not tm.empty:trans.append(tm.assign(asset=asset,fold=f))
            episodes.append(ep.assign(asset=asset,fold=f))
            if not ha.empty:haz.append(ha.assign(asset=asset,fold=f))
    result=dict(scores=pd.concat(scores,ignore_index=True),
        daily_losses=pd.concat(byday,ignore_index=True),
        regimes=pd.concat(regimes,ignore_index=True),
        state_forecasts=pd.concat(calibrations,ignore_index=True),
        transitions=pd.concat(trans,ignore_index=True),
        episodes=pd.concat(episodes,ignore_index=True),
        hazards=pd.concat(haz,ignore_index=True),
        rule_definitions=pd.DataFrame(rules),tree_definitions=pd.DataFrame(tree_txt),
        persistence=pd.DataFrame(persistence),
        folds=[dict(asset=asset,fold=v['fold'],structure_end=v['structure_end'],
                    calibration_start=v['calibration_start'],calibration_end=v['calibration_end'],
                    validation_start=v['validation_start'],validation_end=v['validation_end'],
                    n_structure_days=len(v['structure_days']),n_calibration_days=len(v['calibration_days']),
                    n_validation_days=len(v['validation_days'])) for v in folds],
        n_days=len(good),n_rows=len(x))
    return result

def consolidate(results:Mapping[str,dict],cfg:Spec)->dict:
    names=('scores','daily_losses','regimes','state_forecasts','transitions','episodes','hazards','rule_definitions','tree_definitions','persistence')
    out={k:pd.concat([v[k] for v in results.values()],ignore_index=True) for k in names}
    out['paired_bootstrap']=daily_paired_deltas(out['daily_losses'],cfg)
    out['episode_summary']=aggregate_episode_summary(out['episodes'])
    # Aggregate exploratory comparison by equal asset-fold weighting; no winner claims.
    group=out['scores'].groupby(['method','horizon_min'],as_index=False).agg(
        n_asset_folds=('asset','size'),mean_nll=('nll','mean'),
        mean_binary_logloss=('binary_logloss','mean'),mean_binary_brier=('binary_brier','mean'))
    base=group[group.method.eq('baseline')].set_index('horizon_min')
    group['delta_nll']=group.apply(lambda r:r.mean_nll-base.loc[r.horizon_min,'mean_nll'],axis=1)
    group['delta_binary_logloss']=group.apply(
        lambda r:r.mean_binary_logloss-base.loc[r.horizon_min,'mean_binary_logloss'],axis=1)
    out['comparison_summary']=group
    out['manifest']={'module':'RD01','registration':registration(),
        'synthetic':False,'status':cfg.status,'assets_evaluated':list(results),
        'folds':[f for v in results.values() for f in v['folds']],
        'n_full_days':{a:v['n_days'] for a,v in results.items()},
        'feature_source':'LOCAL CS04-A per-origin historical Parquet',
        'normalization':'TRAILING_24H_VARIANCE_PROXY_NOT_AM_T_HAR_BLEND',
        'density':'Student-t(df=6) fixed common per-asset/fold/horizon scale, regime-dependent TRAIN-calibrated location only',
        'legacy_test_used_for_selection':False,'independent_confirmation':False,
        'AM_T_HAR_Blend_modified':False,
        'caution':'Previously inspected project history; do not infer structural regimes merely from CART leaves or sign separation.'}
    return out
