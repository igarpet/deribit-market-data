"""RD03: causally fitted return-location overlays on *archived* VD18 density.

No change to VD18 AM-T-HAR-Blend model, volatility components, Student-t
shape, horizon-dependent HAR selection, or state transition laws. The sole
allowed overlay moves the origin-specific predictive density location.

Historical Sep19-Oct10 2026 is repeatedly inspected PROJECT data: exploratory.
Return-response families are trained only before Sep1 2026. Archived VD18
parameters are calibrated only on its pre-Sep19 training/validation windows.

CAVEAT: VD18 native volatility paths are horizon-specific HAR overlays, not a
jointly path-consistent Markov distribution. RD03 preserves that exact model.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Mapping
import numpy as np
import pandas as pd
from scipy.special import logsumexp, ndtr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor, export_text

VERSION='RD03-0.1.0'
HORIZONS=(5,15,30,60)
METHODS=('AM_T_HAR_BLEND','sign_only_mean','direct_mean','observable_tree_mean',
         'latent_static_mean','latent_hmm_mean','latent_semi_markov_mean')
STATE_FEATURES=('mom_r5','flow5','basis_bps','basis_change_bps','last_1m_r','log_trades')
RETURN_FEATURES=('mom_r5','mom_r15_mean','flow5','flow30','basis_bps',
                 'basis_change_bps','last_1m_r','minute_accel','log_trades')

@dataclass(frozen=True)
class Specification:
    version:str=VERSION
    assets:tuple[str,...]=('BTC','ETH','SOL','XRP')
    horizons:tuple[int,...]=HORIZONS
    primary_horizon:int=30
    anchor_minutes:int=30
    baseline_version:str='VD18-0.1.0'
    archived_training_cutoff:str='2026-09-19T00:00:00Z'
    structure_start:str='2026-03-01T00:00:00Z'
    structure_end:str='2026-05-01T00:00:00Z'
    response_end:str='2026-09-01T00:00:00Z'
    diagnostic_end:str='2026-09-19T00:00:00Z'
    evaluation_end:str='2026-10-14T00:00:00Z'
    ridge_alpha:float=100.
    response_shrinkage:float=120.
    max_shift_sigma:float=.5
    tree_depth:int=3
    tree_max_leaves:int=5
    tree_min_leaf:int=100
    latent_states:int=3
    latent_em_iterations:int=10
    particle_count:int=256
    bootstrap_draws:int=500
    seed:int=2613
    status:str='HISTORICAL_EXPLORATORY_REPEATEDLY_INSPECTED'

def registration():return asdict(Specification())

def confirm_parameters(manifest:Mapping,cfg:Specification)->dict:
    """Fail closed on mismatched or missing VD18 frozen historical fit."""
    if manifest.get('module')!='Volatility_Dynamics_18':raise ValueError('Not VD18 archive')
    if manifest.get('registration',{}).get('version')!=cfg.baseline_version:
        raise ValueError('VD18 version mismatch')
    if pd.Timestamp(manifest['sample_last'])>=pd.Timestamp(cfg.archived_training_cutoff):
        raise ValueError('Archived VD18 might have fitted on RD03 evaluation')
    params=manifest.get('parameters',{})
    required={'fitted_har','vd17_state_fits','base_cal','native_return_kernels','volatility_transition'}
    if not required.issubset(params):raise ValueError(f'Missing VD18 parameters: {required-set(params)}')
    if 'AM_T' not in params['native_return_kernels']:raise ValueError('Missing AM_T kernel')
    for h in cfg.horizons:
        k=str(h)
        if k not in params['fitted_har'] or k not in params['base_cal']:
            raise ValueError(f'VD18 missing horizon {h}')
        over=params['fitted_har'][k]['overlay']
        if over['selected_har'] not in params['fitted_har'][k]['har_fits']:
            raise ValueError('Archived HAR selection inconsistent')
    return params

def archived_split_frame(history:Path,cfg:Specification):
    """Reconstruct VD18 TRAIN-normalized features without shifting split dates.

    Calling VD14.panelize on the extended sample directly changes train-day
    assignments and introduces different hour-of-day normalizers. Instead
    reconstruct PRE-CUTOFF VD18 split boundaries and fix them in expanded panel.
    """
    from data_engineering.modeling import markov_lift_distributions as v14
    from data_engineering.modeling import markov_har_variance as v18
    # All archived assets must be loaded: they determine the original date split.
    archive_assets=('AVAX','BTC','ETH','SOL','TRX','XRP')
    raw,excluded=v14.load_returns(history,archive_assets,end=pd.Timestamp(cfg.evaluation_end))
    needed=set(cfg.assets)
    if not needed.issubset(set(raw.underlying.unique())):raise ValueError(f'Missing primary spot history: {excluded}')
    old=raw.loc[raw.state_time<pd.Timestamp(cfg.archived_training_cutoff)]
    reference=v14.panelize(old)
    tr_dates=reference.loc[reference.split=='train','date']
    va_dates=reference.loc[reference.split=='valid','date']
    if tr_dates.empty or va_dates.empty:raise ValueError('Missing archived TRAIN/VALID')
    valid_start=va_dates.min()
    test_start=reference.loc[reference.split=='test','date'].min()
    expanded=v14.panelize(raw)
    expanded['split']=np.where(expanded.date<valid_start,'train',
                               np.where(expanded.date<test_start,'valid','test'))
    pre=expanded.loc[expanded.state_time<pd.Timestamp(cfg.archived_training_cutoff)]
    # Refuse a changed split, a common subtle future-normalization leak.
    old_keys=reference[['underlying','state_time','split']]
    new_keys=pre[['underlying','state_time','split']]
    aligned=old_keys.merge(new_keys,on=['underlying','state_time'],suffixes=('_old','_new'),validate='one_to_one')
    if len(aligned)!=len(old_keys) or not aligned.split_old.eq(aligned.split_new).all():
        raise AssertionError('Cannot exactly reconstruct archived VD18 split')
    features=v14.make_features(expanded)
    target=v14.make_targets(expanded,features,horizons=cfg.horizons)
    return v18.add_har_lags(target),{'archived_valid_from':str(valid_start),
              'archived_test_from':str(test_start),'raw_excluded':excluded}

def merge_market_features(vd18_panel:pd.DataFrame,store:pd.DataFrame,asset:str)->pd.DataFrame:
    """Strict origin match. Forward labels always come from VD14 not CS04."""
    needed=['state_time','feature_ready',*STATE_FEATURES,*RETURN_FEATURES]
    cols=list(dict.fromkeys(needed))
    if not set(cols).issubset(store):raise ValueError(f'CS04 store lacks: {set(cols)-set(store)}')
    z=store[cols].copy();z.state_time=pd.to_datetime(z.state_time,utc=True)
    if z.state_time.duplicated().any():raise ValueError('Duplicate local CS04 timestamp')
    x=vd18_panel.loc[vd18_panel.underlying==asset].copy()
    x=x.merge(z,on='state_time',how='inner',validate='one_to_one')
    t=x.state_time
    mask=((t.dt.minute%30==0)&(t.dt.second==0)&x.ready&x.feature_ready)
    for h in HORIZONS:
        mask &= np.isfinite(x[f'R_{h}'].to_numpy(float))
        mask &= np.isfinite(x[f'RV_{h}'].to_numpy(float))
    mask &= np.isfinite(x[list(dict.fromkeys([*STATE_FEATURES,*RETURN_FEATURES]))].to_numpy(float)).all(axis=1)
    x=x.loc[mask].sort_values('state_time').reset_index(drop=True)
    if len(x)<500:raise ValueError(f'{asset}: insufficient causal join')
    return x

def periods(x:pd.DataFrame,cfg:Specification)->pd.DataFrame:
    out=x.copy();t=pd.to_datetime(out.state_time,utc=True)
    c=lambda s:pd.Timestamp(s)
    out=out[(t>=c(cfg.structure_start))&(t<c(cfg.evaluation_end))].copy()
    t=out.state_time
    out['rd_period']=np.select([t<c(cfg.structure_end),t<c(cfg.response_end),
                  t<c(cfg.diagnostic_end)],['structure','response','diagnostic'],default='evaluation')
    if not {'structure','response','diagnostic','evaluation'}.issubset(set(out.rd_period)):
        raise ValueError('Required chronological phases missing')
    return out.reset_index(drop=True)

def vd18_particle_variances(frame:pd.DataFrame,asset:str,h:int,params:dict,*,particles=256,block=256):
    """Exact VD18 archived HAR-BLEND density variances, *not* trailing-RV proxy.

    Inputs must be a strictly ordered subset of VD18 complete forecast origins.
    No future return target is consulted in this function.
    """
    from data_engineering.modeling import markov_return_horizon_calibration as v16
    from data_engineering.modeling import markov_conditional_variance as v17
    from data_engineering.modeling import markov_har_variance as v18
    from data_engineering.modeling import markov_lift_distributions as v14
    if h not in HORIZONS:raise ValueError('Unregistered horizon')
    if particles!=256 and particles!=16:raise ValueError('256 archival or 16 synthetic particles only')
    missing=frame.loc[~frame.ready|~np.isfinite(frame[[f'har_logrv_{j}' for j in v18.ALL_FEATURES]].to_numpy(float)).all(axis=1)]
    if len(missing):raise ValueError('VD18 incomplete feature rows')
    f=params['fitted_har'][str(h)]
    base=params['base_cal'][str(h)]['families']['AM_T_BOTH']
    over=f['overlay']; fit=f['har_fits'][over['selected_har']]
    har_pred=v18.predict_har(frame,fit)
    design=v17.design_apply(frame,over['design'])
    result=[]
    for sl,p in v16.variance_particles(frame[['slow','gap']].to_numpy(float),
            params['volatility_transition'][asset],params['native_return_kernels']['AM_T'][asset],
            h//5,particles,v17._seed(asset,h),block):
        base_components=v17.component_variances(p,base)
        z=v18._predict_var_ratio(har_pred[sl],p,base)
        delta=v17.delta_for('AM_T_VAR_ANCHOR',design[sl],z,over['blend_fit'])
        result.append(np.maximum(base_components*np.exp(delta[:,None]),1e-14))
    return np.concatenate(result,axis=0)

def normal_mixture_scores(y:np.ndarray,variance:np.ndarray,mean:np.ndarray):
    """Proper density, sign Brier: fixed VD18 variance particles, shifted mean.

    Same μ shift applied to every particle; conditional predictive variance is
    exactly mean(VD18 particle variances) regardless of the shift.
    Latent state probabilities affect μ *only* (not a mixture of latent locations).
    """
    y=np.asarray(y,float);v=np.asarray(variance,float);mu=np.asarray(mean,float)
    if y.ndim!=1 or mu.shape!=y.shape or v.ndim!=2 or v.shape[0]!=len(y):
        raise ValueError('Invalid density shapes')
    if not np.isfinite(v).all() or (v<=0).any() or not np.isfinite(y).all() or not np.isfinite(mu).all():
        raise ValueError('Nonfinite density inputs')
    z=(y-mu)[:,None]
    nll=-logsumexp(-.5*(np.log(2*np.pi*v)+z*z/v),axis=1)+np.log(v.shape[1])
    up=ndtr(mu[:,None]/np.sqrt(v)).mean(axis=1)
    brier=(up-(y>0).astype(float))**2
    return {'nll':nll,'up_probability':up,'up_brier':brier,
            'variance':v.mean(axis=1),'shift':mu}

def response_location_fit(cal:pd.DataFrame,variance:np.ndarray,cfg:Specification,features=RETURN_FEATURES):
    """CAL-only standardized linear location model. Same for all alternatives."""
    if len(cal)!=len(variance):raise ValueError('CAL variance alignment')
    y=cal.ret_target.to_numpy(float)/np.sqrt(variance.mean(axis=1))
    sc=StandardScaler().fit(cal[list(features)].to_numpy(float))
    a=np.clip(sc.transform(cal[list(features)].to_numpy(float)),-8.,8.)
    model=Ridge(alpha=cfg.ridge_alpha).fit(a,y)
    return sc,model,tuple(features)

def response_location_predict(frame:pd.DataFrame,fit,cfg:Specification):
    sc,model,features=fit
    x=np.clip(sc.transform(frame[list(features)].to_numpy(float)),-8.,8.)
    return np.clip(model.predict(x),-cfg.max_shift_sigma,cfg.max_shift_sigma)

def shrunk_residual_means(residual:np.ndarray,weights:np.ndarray,lam:float):
    """State effect conditional on common direct mean; weighted CAL-only fit."""
    residual=np.asarray(residual,float)
    w=np.asarray(weights,float)
    if w.ndim!=2 or w.shape[0]!=len(residual):raise ValueError('bad weights')
    if not np.isfinite(w).all() or (w<0).any() or not np.allclose(w.sum(axis=1),1):
        raise ValueError('weights must be posterior probabilities')
    mass=w.sum(axis=0)
    est=np.sum(w*residual[:,None],axis=0)/(mass+lam)
    return est

def posterior_causal_history(frame:pd.DataFrame,fit:dict,scale:StandardScaler,cfg:Specification):
    """Causal current-state FILTER, not Viterbi/smoothing, for prediction.

    Filters contiguous 30-minute observations in chronological order. At a
    missing timestamp they reset rather than use a stale hidden state.
    """
    from data_engineering.modeling import latent_regime_tournament as rd02
    q=np.clip(scale.transform(frame[list(STATE_FEATURES)].to_numpy(float)),-6.,6.)
    loge=rd02.emission_loglik(q,fit['loc'],fit['var'],6.)
    t=frame.state_time
    hmm=rd02.filter_hmm(loge,fit['A'],fit['pi'],t,step_minutes=cfg.anchor_minutes)
    static=rd02.static_states(loge,fit['pi'])
    path=rd02.viterbi(rd02.emission_loglik(q[:sum(frame.rd_period.eq('structure'))],fit['loc'],fit['var'],6.),
                       fit['A'],fit['pi'],rd02.segments(t.iloc[:sum(frame.rd_period.eq('structure'))],cfg.anchor_minutes))
    from dataclasses import replace
    cfg2=replace(rd02.Spec(),anchor_minutes=cfg.anchor_minutes,seed=cfg.seed,
                 k_states=cfg.latent_states,em_iterations=cfg.latent_em_iterations)
    hazard,_=rd02.empirical_hazards(path,t.iloc[:len(path)],fit['A'],cfg2)
    semi=rd02.filter_semi_markov(loge,fit['A'],fit['pi'],hazard,t,step_minutes=cfg.anchor_minutes)
    return {'latent_static_mean':static,'latent_hmm_mean':hmm,'latent_semi_markov_mean':semi},hazard

def prepare_methods(frame:pd.DataFrame,variances:Mapping[int,np.ndarray],cfg:Specification):
    """All regime structures and return responses trained strictly pre-eval."""
    from data_engineering.modeling import latent_regime_tournament as rd02
    from dataclasses import replace
    if not frame.state_time.is_monotonic_increasing:raise ValueError('nonmonotone')
    st=frame.rd_period.eq('structure').to_numpy();cal=frame.rd_period.eq('response').to_numpy()
    ev=frame.rd_period.eq('evaluation').to_numpy()
    if not st.any() or not cal.any() or not ev.any():raise ValueError('no fit/evaluation phases')
    if frame.loc[st,'state_time'].max()>=frame.loc[ev,'state_time'].min():raise AssertionError('future information in structure')
    if frame.loc[cal,'state_time'].max()>=frame.loc[ev,'state_time'].min():raise AssertionError('future information in calibration')
    sc=StandardScaler().fit(frame.loc[st,list(STATE_FEATURES)].to_numpy(float))
    z=np.clip(sc.transform(frame.loc[st,list(STATE_FEATURES)].to_numpy(float)),-6,6)
    lcfg=replace(rd02.Spec(),seed=cfg.seed,k_states=cfg.latent_states,
                 em_iterations=cfg.latent_em_iterations,anchor_minutes=cfg.anchor_minutes)
    hmmfit=rd02.fit_hmm(z,frame.loc[st,'state_time'],lcfg)
    post,hazard=posterior_causal_history(frame,hmmfit,sc,cfg)
    # Supervised tree structure relies on earlier signed returns only, no CAL/VALID.
    tstruct=frame.loc[st].copy(); tcal=frame.loc[cal].copy()
    tree=DecisionTreeRegressor(max_depth=cfg.tree_depth,max_leaf_nodes=cfg.tree_max_leaves,
        min_samples_leaf=cfg.tree_min_leaf,random_state=cfg.seed)
    zstruct=tstruct['R_30'].to_numpy(float)/np.sqrt(variances[30][st].mean(axis=1))
    tree.fit(tstruct[list(RETURN_FEATURES)],np.clip(zstruct,-5.,5.))
    leaves=tree.apply(frame[list(RETURN_FEATURES)])
    frame=frame.copy();frame['s_sign']=np.sign(frame.mom_r5.to_numpy(float))
    tcal=frame.loc[cal].copy()
    models={}; diagnostics={}
    for h,v in variances.items():
        sigma=np.sqrt(v.mean(axis=1))
        zreturn=frame[f'R_{h}'].to_numpy(float)/sigma
        calframe=tcal.copy();calframe['ret_target']=tcal[f'R_{h}'].to_numpy(float)
        direct=response_location_fit(calframe,v[cal],cfg)
        sign_only=response_location_fit(calframe,v[cal],cfg,features=('s_sign',))
        q=response_location_predict(frame,direct,cfg)
        e=zreturn[cal]-q[cal]
        # Leaf IDs can be sparse node indexes; use sorted leaf vocabulary.
        vocabulary=np.unique(leaves[st]);treeweights=(leaves[:,None]==vocabulary[None,:]).astype(float)
        tree_effect=shrunk_residual_means(e,treeweights[cal],cfg.response_shrinkage)
        preds={'AM_T_HAR_BLEND':np.zeros(len(frame)),
               'sign_only_mean':response_location_predict(frame,sign_only,cfg),'direct_mean':q,
               'observable_tree_mean':np.clip(q+treeweights@tree_effect,-cfg.max_shift_sigma,cfg.max_shift_sigma)}
        for name,weights in post.items():
            states=shrunk_residual_means(e,weights[cal],cfg.response_shrinkage)
            preds[name]=np.clip(q+weights@states,-cfg.max_shift_sigma,cfg.max_shift_sigma)
        models[h]=preds
        diagnostics[h]={'direct_coefficients':direct[1].coef_.tolist(),
             'tree_leaf_support':treeweights[cal].sum(axis=0).astype(int).tolist(),
             'tree_effect':tree_effect.tolist(),
             'state_response':{k:shrunk_residual_means(e,w[cal],cfg.response_shrinkage).tolist()
                               for k,w in post.items()}}
    return models,{'tree':export_text(tree,feature_names=list(RETURN_FEATURES)),
                   'hmm_transitions':hmmfit['A'].tolist(),'hmm_state_centers':hmmfit['loc'].tolist(),
                   'hmm_structure_loglik':hmmfit['loglike'],
                   'hazards':hazard.tolist(),'horizon_response':diagnostics,
                   'n_structure':int(st.sum()),'n_response':int(cal.sum()),
                   'n_evaluation':int(ev.sum())}

def score_asset(frame:pd.DataFrame,variances:Mapping[int,np.ndarray],cfg:Specification):
    fitted,fitinfo=prepare_methods(frame,variances,cfg)
    ev=frame.rd_period.eq('evaluation').to_numpy()
    rows=[];losses=[]
    for h,models in fitted.items():
        y=frame[f'R_{h}'].to_numpy(float)[ev]
        v=variances[h][ev]
        for method,z in models.items():
            assert method in METHODS
            sigma=np.sqrt(v.mean(axis=1))
            mean=z[ev]*sigma
            loss=normal_mixture_scores(y,v,mean)
            rows.append({'horizon_min':h,'method':method,'n_origins':len(y),
                         'n_days':frame.loc[ev,'date'].nunique(),
                         'nll':float(loss['nll'].mean()),
                         'up_brier':float(loss['up_brier'].mean()),
                         'predicted_up':float(loss['up_probability'].mean()),
                         'observed_up':float((y>0).mean()),
                         'mean_location_bps':float(mean.mean()*1e4),
                         'mean_sigma_bps':float(sigma.mean()*1e4)})
            losses.append(pd.DataFrame({'date':frame.loc[ev,'date'].to_numpy(),
                 'horizon_min':h,'method':method,'nll':loss['nll'],
                 'up_brier':loss['up_brier']}))
    return pd.DataFrame(rows),pd.concat(losses,ignore_index=True),fitinfo

def paired_day_bootstrap(losses:pd.DataFrame,cfg:Specification):
    """Whole-day paired baseline, incremental-regime and temporal-ablation tests."""
    rng=np.random.default_rng(cfg.seed)
    pairs=[(name,'AM_T_HAR_BLEND') for name in METHODS[1:]]
    pairs += [(name,'direct_mean') for name in METHODS if name.startswith('latent_') or name=='observable_tree_mean']
    pairs += [('latent_hmm_mean','latent_static_mean'),
              ('latent_semi_markov_mean','latent_hmm_mean')]
    rows=[]
    for h in cfg.horizons:
        q=losses[losses.horizon_min.eq(h)]
        for method,ref in pairs:
            for metric in ('nll','up_brier'):
                baseline=q[q.method.eq(ref)].groupby('date')[metric].mean()
                candidate=q[q.method.eq(method)].groupby('date')[metric].mean()
                diff=(candidate-baseline).dropna()
                days=len(diff)
                if days<12:
                    low=high=np.nan;status='INSUFFICIENT_DAYS'
                else:
                    z=diff.to_numpy(float)
                    draws=z[rng.integers(0,days,size=(cfg.bootstrap_draws,days))].mean(axis=1)
                    low,high=np.quantile(draws,[.025,.975]);status='EXPLORATORY'
                rows.append({'horizon_min':h,'method':method,'reference':ref,'metric':metric,
                     'delta_equal_day':float(diff.mean()) if days else np.nan,
                     'ci025':float(low),'ci975':float(high),'days':days,'status':status})
    return pd.DataFrame(rows)

def decision(scores:pd.DataFrame,boots:pd.DataFrame,cfg:Specification,synthetic=False):
    if synthetic:return {'status':'SYNTHETIC','verdict':'NO_MARKET_INFERENCE','prospective_confirmation':False}
    primary=boots[(boots.horizon_min==cfg.primary_horizon)&boots.metric.eq('nll')]
    primary=primary[primary.status.eq('EXPLORATORY')] if 'status' in primary else primary.iloc[:0]
    # These are ONLY descriptive screens; historical period already studied.
    results=[]
    for asset in ('BTC','ETH'):
        q=primary[primary.asset.eq(asset)]
        for method in METHODS[3:]:
            checks=[]
            for ref in ('AM_T_HAR_BLEND','direct_mean'):
                v=q[(q.method.eq(method))&(q.reference.eq(ref))]
                checks.append(bool(len(v)==1 and v.iloc[0].ci975<0))
            results.append({'asset':asset,'method':method,
                            'beats_original_nominally':checks[0],
                            'beats_direct_nominally':checks[1]})
    return {'status':cfg.status,'verdict':'EXPLORATORY_DENSITY_SCREEN',
            'primary_horizon_min':cfg.primary_horizon,
            'historical_primary_checks':results,
            'confirmation':False,'prospective_confirmation':False,
            'AM_T_HAR_Blend_modified':False,
            'note':'Sep-Oct 2026 has been inspected in other project branches. '
                   'The mean-only overlays preserve the VD18 predictive variance. '
                   'HMM and duration overlay are RD02-style refits, not a joint HSMM.'}
