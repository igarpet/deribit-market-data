"""VD17 — Conditional variance corrections to VD16's Student-t RETURN Markov model.

Exploratory science, *not* a new structural Markov transition, not trading.
Native 5m Student-t return kernel, VD14 Markov factors and VD16 calibrated
particle-variance shape remain frozen. Compare conditional scale overlays:
  BASE       : VD16 AM_T_BOTH, unchanged
  STATE      : pooled, regularized time-varying log scale from causal Markov proxies
  ANCHOR     : geometric blend of BASE expected variance and independently
               TRAIN-fitted horizon-specific M2 variance forecast
  COMBO      : STATE + ANCHOR in a single penalized conditional forecast
  AM_T       : original uncalibrated VD15 AM_T
  M2         : VD14 direct horizon-specific Student-t baseline

BASE is fit on early VALID, conditional scales are fit on the SAME early VALID;
late VALID is purged, never fit on and used to compare models. TEST labels are
already inspected in prior branches: all outcomes remain retrospective exploratory.
No per-origin outputs saved; all QMC forecast arrays are block streamed.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import gc
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp, ndtr
from data_engineering.modeling import markov_lift_distributions as v14
from data_engineering.modeling import student_t_return_markov as v15
from data_engineering.modeling import markov_return_horizon_calibration as v16

VERSION='VD17-0.1.0'
HORIZONS=v15.HORIZONS
PRIMARY=60
PARTICLES=256
BOOT=1000
CAL_LIMIT=512
SELECT_LIMIT=512
MODELS=('AM_T','AM_T_BOTH','AM_T_VAR_STATE','AM_T_VAR_ANCHOR','AM_T_VAR_COMBO','M2_student_t')
DYNAMIC=('AM_T_VAR_STATE','AM_T_VAR_ANCHOR','AM_T_VAR_COMBO')
PAIRS=(('AM_T_BOTH','AM_T'),('AM_T_VAR_STATE','AM_T_BOTH'),
       ('AM_T_VAR_ANCHOR','AM_T_BOTH'),('AM_T_VAR_COMBO','AM_T_BOTH'),
       ('AM_T_VAR_COMBO','AM_T_VAR_ANCHOR'),('AM_T_VAR_COMBO','AM_T_VAR_STATE'),
       ('AM_T_VAR_COMBO','M2_student_t'),('AM_T_VAR_ANCHOR','M2_student_t'))
METRICS=('nll','tail_logloss','tail_brier','return_qlike')
FEATURES=('slow','gap','now','slope15')
MAX_DELTA=.85
MAX_ANCHOR_RATIO=1.5
STATE_RIDGE=.025
ANCHOR_RIDGE=.003

@dataclass(frozen=True)
class Registration:
    version:str=VERSION
    horizons:tuple=HORIZONS
    primary_horizon:int=PRIMARY
    source:str='VD14 state and TRAIN two-EMA variance, VD15 5m Student-t return kernel, VD16 shape/scale overlay'
    test_status:str='HISTORICAL_EXPLORATORY (test previously viewed)'
    early_valid_fraction:float=.60
    per_asset_early_valid_origins:int=CAL_LIMIT
    per_asset_late_valid_origins:int=SELECT_LIMIT
    particles:int=PARTICLES
    bootstrap:int=BOOT
    state_predictors:tuple=FEATURES
    state_log_variance_clip:float=MAX_DELTA
    m2_log_variance_ratio_clip:float=MAX_ANCHOR_RATIO
    state_l2_penalty:float=STATE_RIDGE
    anchor_l2_penalty:float=ANCHOR_RIDGE
    primary_metric:str='return_nll'
    additional_variance_metric:str='return_qlike (predictive net-return variance proper score)'
    native_kernel_unchanged:bool=True
    note:str='No future price observations in features. Validation sample overlap with native t df selection; no untouched inference.'

def registration():return asdict(Registration())

def _seed(asset,h):
    return 15000+int(np.sum(np.frombuffer(asset.encode(),dtype=np.uint8)))+(h//5)*100

def design_fit(g:pd.DataFrame):
    """Fit design normalizers using only early VALID; no future labels used."""
    names=sorted(g.underlying.unique())
    x=g[list(FEATURES)].to_numpy(float)
    center=x.mean(axis=0);std=np.maximum(x.std(axis=0),1e-6)
    return {'names':names,'center':center.tolist(),'std':std.tolist()}

def design_apply(g:pd.DataFrame,fit:dict):
    names=fit['names']
    z=(g[list(FEATURES)].to_numpy(float)-np.asarray(fit['center']))/np.asarray(fit['std'])
    z=np.clip(z,-5,5)
    ang=2*np.pi*g.slot.to_numpy(float)/288
    extras=[(g.underlying.to_numpy()==n).astype(float) for n in names[1:]]
    return np.column_stack([np.ones(len(g)),z,np.sin(ang),np.cos(ang),*extras])

def anchor_ratio(g:pd.DataFrame,v:np.ndarray,m2var:np.ndarray,base:dict):
    """A causal predictor: compare 2 ex-ante variance forecasts, never labels."""
    base_mean=np.mean(v,axis=1)*np.exp(base['log_scale'])
    r=np.log(np.maximum(m2var,1e-14)/np.maximum(base_mean,1e-14))
    return np.clip(r,-MAX_ANCHOR_RATIO,MAX_ANCHOR_RATIO)

def component_variances(v:np.ndarray,base:dict):
    return v16.transform_variances(v,base['log_scale'],base['shape'])

def nll_from_components(components:np.ndarray,y:np.ndarray,delta:np.ndarray):
    """Integrate normalized Gaussian conditional components after variance scale correction."""
    v=components*np.exp(np.clip(np.asarray(delta,float)[:,None],-MAX_DELTA,MAX_DELTA))
    z=np.asarray(y,float)[:,None]
    return -logsumexp(-.5*(np.log(2*np.pi*v)+z*z/v),axis=1)+np.log(v.shape[1])

def delta_for(model:str,X:np.ndarray,z:np.ndarray,fit:dict):
    if model=='AM_T_VAR_STATE':
        out=X@np.asarray(fit['beta'])
    elif model=='AM_T_VAR_ANCHOR':
        out=fit['anchor_weight']*z
    elif model=='AM_T_VAR_COMBO':
        out=X@np.asarray(fit['beta'])+fit['anchor_weight']*z
    else:raise ValueError('unknown dynamic family')
    return np.clip(out,-MAX_DELTA,MAX_DELTA)

def fit_dynamic(model:str,components:np.ndarray,y:np.ndarray,X:np.ndarray,z:np.ndarray,
                *,maxiter:int=75):
    """NLL fitting on early VALID ONLY, frozen variance-conditional predictors."""
    if model not in DYNAMIC:raise ValueError(model)
    p=X.shape[1]
    if model=='AM_T_VAR_ANCHOR':
        def objective(w):
            d=np.clip(float(w)*z,-MAX_DELTA,MAX_DELTA)
            return nll_from_components(components,y,d).mean()+ANCHOR_RIDGE*float(w)**2
        opt=minimize_scalar(objective,bounds=(0,1.25),method='bounded',options={'xatol':1e-4,'maxiter':60})
        return {'beta':[0.]*p,'anchor_weight':float(opt.x),'objective':float(opt.fun),'converged':bool(opt.success)}
    def objective(param):
        if model=='AM_T_VAR_STATE':beta=param;w=0.
        else:beta=param[:-1];w=float(param[-1])
        d=np.clip(X@beta+w*z,-MAX_DELTA,MAX_DELTA)
        return float(nll_from_components(components,y,d).mean()+STATE_RIDGE*np.sum(beta**2)+ANCHOR_RIDGE*w*w)
    initial=np.zeros(p+(model=='AM_T_VAR_COMBO'))
    bounds=[(-.6,.6)]*p+([(0,1.25)] if model=='AM_T_VAR_COMBO' else [])
    opt=minimize(objective,initial,bounds=bounds,method='L-BFGS-B',options={'maxiter':maxiter,'ftol':1e-8})
    return {'beta':opt.x[:p].tolist(),'anchor_weight':float(opt.x[-1]) if model=='AM_T_VAR_COMBO' else 0.,
            'objective':float(opt.fun),'converged':bool(opt.success),'iterations':int(opt.nit)}

def loss_components(v:np.ndarray,y:np.ndarray,q:np.ndarray):
    """Returns ex-ante density, tail, interval and proper variance scores."""
    v=np.maximum(v,1e-14)
    y=np.asarray(y,float);q=np.asarray(q,float)
    assert v.shape[0]==len(y)==len(q)
    lp=-.5*(np.log(2*np.pi*v)+y[:,None]**2/v)
    nll=-logsumexp(lp,axis=1)+np.log(v.shape[1])
    sd=np.sqrt(v)
    prob=np.clip((2*ndtr(-q[:,None]/sd)).mean(axis=1),1e-10,1-1e-10)
    observed=(np.abs(y)>=q).astype(float)
    tail_logloss=-(observed*np.log(prob)+(1-observed)*np.log1p(-prob))
    meanv=v.mean(axis=1)
    return {'nll':nll,'tail_logloss':tail_logloss,'tail_brier':(observed-prob)**2,
            'tail_predicted':prob,'tail_observed':observed,
            'interval_miss95':((2*ndtr(-np.abs(y)[:,None]/sd)).mean(axis=1)<.05).astype(float),
            'return_qlike':np.log(meanv)+y*y/meanv,
            'pred_variance':meanv}

def _m2_variance(tr:pd.DataFrame,g:pd.DataFrame,h:int):
    return v14._rv_scale(tr,g,h)[0]

def sampled_candidates(frame:pd.DataFrame,h:int,kernels:dict,transitions:dict,*,particles=PARTICLES,cal_limit=CAL_LIMIT):
    tr,va,_=v14._get_split(frame,h)
    cal,select,boundary=v16.split_validation(va,h,max_per_asset=cal_limit)
    # evaluate TEST only after fit; CAL and SELECT deterministic and outcome-agnostic
    subsets=[]
    for tag,g in [('cal',cal),('select',select)]:
        v=np.zeros((len(g),particles))
        for asset,a in g.groupby('underlying',sort=True):
            sel=np.flatnonzero((g.underlying==asset).to_numpy())
            v[sel]=v16.integrate_subset(a,asset,h,kernels['AM_T'][asset],transitions[asset],particles)
        m2=_m2_variance(tr,g,h)
        subsets.append((g,v,m2))
    return subsets,boundary

def fit_variance_models(frame:pd.DataFrame,kernels:dict,transitions:dict,
                        base_cal:dict,*,horizons=HORIZONS,particles=PARTICLES,cal_limit=CAL_LIMIT):
    """Separate earliest 60% VALID fit, latest 40% VALID choice. Strict embargo."""
    if particles<8 or particles&(particles-1) or cal_limit<8:
        raise ValueError('Particle budget must be a power of 2 >=8 and calibration sample >=8')
    out={};rows=[]
    for h in horizons:
        (cal,sel),boundary=sampled_candidates(frame,h,kernels,transitions,particles=particles,cal_limit=cal_limit)
        a,va,m2a=cal;b,vb,m2b=sel
        base=base_cal[str(h)]['families']['AM_T_BOTH']
        design=design_fit(a)
        X=design_apply(a,design);Xs=design_apply(b,design)
        za=anchor_ratio(a,va,m2a,base);zb=anchor_ratio(b,vb,m2b,base)
        vc=component_variances(va,base);vs=component_variances(vb,base)
        y=a[f'R_{h}'].to_numpy(float);ys=b[f'R_{h}'].to_numpy(float)
        base_sel_nll=float(nll_from_components(vs,ys,np.zeros(len(ys))).mean())
        fits={};data={'horizon_min':h,'model':'AM_T_BOTH','cal_nll':float(nll_from_components(vc,y,np.zeros(len(y))).mean()),
                       'select_nll':base_sel_nll,'anchor_weight':0.,'max_abs_beta':0.,'cal_origins':len(a),'select_origins':len(b),'boundary':boundary}
        rows.append(data)
        for model in DYNAMIC:
            f=fit_dynamic(model,vc,y,X,za)
            dcal=delta_for(model,X,za,f);dsel=delta_for(model,Xs,zb,f)
            scores={'cal_nll':float(nll_from_components(vc,y,dcal).mean()),
                    'select_nll':float(nll_from_components(vs,ys,dsel).mean())}
            fits[model]=f
            rows.append({'horizon_min':h,'model':model,**scores,'anchor_weight':f['anchor_weight'],
                         'max_abs_beta':float(np.max(np.abs(f['beta']))),
                         'cal_origins':len(a),'select_origins':len(b),'boundary':boundary})
        names=['AM_T_BOTH',*DYNAMIC]
        score_map={r['model']:r['select_nll'] for r in rows if r['horizon_min']==h}
        chosen=min(names,key=lambda name:score_map[name])
        out[str(h)]={'baseline':base,'design':design,'dynamic':fits,'selected_late_valid':chosen,
                     'boundary':boundary,'cal_origins':len(a),'select_origins':len(b)}
    return out,pd.DataFrame(rows)

def _baseline_m2_scores(frame:pd.DataFrame,h:int,transitions:dict,hazard:dict):
    tr,va,te,q,ref,_=v15._reference_models(frame,h,transitions,hazard)
    v_m2=_m2_variance(tr,te,h)
    return tr,te,q,ref,v_m2

def evaluate_horizon(frame:pd.DataFrame,h:int,kernels:dict,transitions:dict,hazard:dict,
                     fit:dict,*,particles=PARTICLES,block=1024):
    tr,te,q,ref,m2v=_baseline_m2_scores(frame,h,transitions,hazard)
    n=len(te);y=te[f'R_{h}'].to_numpy(float)
    results={model:{k:np.zeros(n) for k in ('nll','tail_logloss','tail_brier','tail_predicted',
             'tail_observed','interval_miss95','return_qlike','pred_variance')}
             for model in MODELS}
    for key in ('nll','tail_logloss','tail_brier','interval_miss95'):
        results['M2_student_t'][key]=ref['M2_student_t'][key].astype(float)
    results['M2_student_t']['tail_predicted']=ref['M2_student_t']['tail_probability'].astype(float)
    results['M2_student_t']['tail_observed']=(np.abs(y)>=q).astype(float)
    results['M2_student_t']['pred_variance']=m2v
    results['M2_student_t']['return_qlike']=np.log(m2v)+y*y/m2v
    pa=[]
    for asset,g in te.groupby('underlying',sort=True):
        mask=(te.underlying==asset).to_numpy();ix=np.flatnonzero(mask)
        state=g[['slow','gap']].to_numpy(float)
        m2=m2v[mask]
        xa=design_apply(g,fit['design'])
        gy=y[mask];gq=q[mask]
        base=fit['baseline']
        for sl,v in v16.variance_particles(state,transitions[asset],kernels['AM_T'][asset],h//5,
                                           particles,_seed(asset,h),block):
            original=loss_components(v,gy[sl],gq[sl])
            fixed=component_variances(v,base)
            ratio=anchor_ratio(g.iloc[sl],v,m2[sl],base)
            changed={'AM_T':original,'AM_T_BOTH':loss_components(fixed,gy[sl],gq[sl])}
            for model in DYNAMIC:
                correction=delta_for(model,xa[sl],ratio,fit['dynamic'][model])
                vv=fixed*np.exp(correction[:,None])
                changed[model]=loss_components(vv,gy[sl],gq[sl])
            for model,values in changed.items():
                for key,val in values.items():results[model][key][ix[sl]]=val
        for model in MODELS:
            pa.append({'horizon_min':h,'asset':asset,'model':model,'n_origins':len(g),
                       **{metric:float(results[model][metric][mask].mean()) for metric in
                          ('nll','tail_logloss','tail_brier','tail_predicted','tail_observed',
                           'interval_miss95','return_qlike','pred_variance')}})
    scores=pd.DataFrame([{'horizon_min':h,'model':model,'n_origins':n,'n_days':int(te.date.nunique()),
              **{metric:float(results[model][metric].mean()) for metric in
                ('nll','tail_logloss','tail_brier','tail_predicted','tail_observed',
                 'interval_miss95','return_qlike','pred_variance')}}
                  for model in MODELS])
    return scores,pd.DataFrame(pa),te[['date','underlying']],results

def run_all(frame:pd.DataFrame,*,horizons=HORIZONS,particles=PARTICLES,B=BOOT,
            cal_limit=CAL_LIMIT,chunk_size=1024):
    if particles!=PARTICLES and particles!=16:raise ValueError('Only 256 particles for full run; 16 for tests')
    transitions=v14.fit_markov(frame.loc[(frame.split=='train')&frame.ready].copy())
    hazard=v14.fit_onebar_hazard(frame,transitions)
    tr5,va5,_=v14._get_split(frame,5)
    kernels,_=v15.fit_return_kernel(tr5,va5,transitions)
    base_cal,_=v16.prepare_calibrations(frame,kernels,transitions,horizons=horizons,
        particles=particles,max_per_asset=cal_limit)
    # sampled_candidates relies on the same particle size + limits as the base
    all_fits,fit_grid=fit_variance_models(frame,kernels,transitions,base_cal,
        horizons=horizons,particles=particles,cal_limit=cal_limit)
    scores=[];asset_scores=[];boots=[]
    for h in horizons:
        sc,pa,te,p=evaluate_horizon(frame,h,kernels,transitions,hazard,all_fits[str(h)],
            particles=particles,block=chunk_size)
        scores.append(sc);asset_scores.append(pa)
        for i,(a,b) in enumerate(PAIRS):
            for j,key in enumerate(METRICS):
                delta,lo,hi=v15._bootstrap(p[a][key],p[b][key],te.date.to_numpy(),B,
                       seed=1701+h*101+10*i+j)
                boots.append({'horizon_min':h,'comparison':f'{a} - {b}','metric':key,
                              'delta':delta,'q025':lo,'q975':hi,'test_days':len(te.date.unique()),
                              'origins':len(te)})
        del p,te
        gc.collect()
    return (pd.concat(scores,ignore_index=True),pd.DataFrame(boots),
            pd.concat(asset_scores,ignore_index=True),fit_grid,
            {'calibrations':all_fits,'native_kernels':kernels,'markov_transitions':transitions,
             'base_calibration':base_cal,'version':VERSION})

def decide(scores:pd.DataFrame,boot:pd.DataFrame,fits:dict):
    pri=boot.loc[(boot.horizon_min==PRIMARY)&(boot.metric=='nll')]
    def pass_gate(a,b):
        m=pri[pri.comparison==f'{a} - {b}']
        return bool(len(m)==1 and m.delta.iloc[0]<0 and m.q975.iloc[0]<0)
    return {'status':'HISTORICAL_EXPLORATORY','primary_horizon_min':PRIMARY,
         'test_days':int(scores.loc[scores.horizon_min==PRIMARY,'n_days'].min()),
         'primary_flags':{'STATE_beats_BASE':pass_gate('AM_T_VAR_STATE','AM_T_BOTH'),
                          'ANCHOR_beats_BASE':pass_gate('AM_T_VAR_ANCHOR','AM_T_BOTH'),
                          'COMBO_beats_BASE':pass_gate('AM_T_VAR_COMBO','AM_T_BOTH'),
                          'COMBO_beats_M2':pass_gate('AM_T_VAR_COMBO','M2_student_t')},
         'late_valid_selected':{h:f['selected_late_valid'] for h,f in fits['calibrations'].items()},
         'historical_confirmation_permitted':False,'trading_approved':False}

def particle_sensitivity_validation(frame:pd.DataFrame,kernels:dict,transitions:dict,
                                    calibrations:dict,*,horizons=(5,60,240),
                                    particles=PARTICLES,max_per_asset=96):
    """Numerical stability on late VALID only: nested 256/512 QMC, no TEST tuning."""
    rows=[]
    for h in horizons:
        tr,va,_=v14._get_split(frame,h)
        _,select,_=v16.split_validation(va,h,max_per_asset=max_per_asset)
        fit=calibrations[str(h)]
        base=fit['baseline']
        m2=_m2_variance(tr,select,h)
        for asset,g in select.groupby('underlying',sort=True):
            mask=(select.underlying==asset).to_numpy()
            m2_asset=m2[mask]
            X=design_apply(g,fit['design'])
            y=g[f'R_{h}'].to_numpy(float)
            score={}
            for k in (particles,2*particles):
                v=v16.integrate_subset(g,asset,h,kernels['AM_T'][asset],transitions[asset],k)
                fixed=component_variances(v,base)
                z=anchor_ratio(g,v,m2_asset,base)
                for model in ('AM_T_BOTH',*DYNAMIC):
                    d=np.zeros(len(g)) if model=='AM_T_BOTH' else delta_for(model,X,z,fit['dynamic'][model])
                    score[(model,k)]=float(nll_from_components(fixed,y,d).mean())
            for model in ('AM_T_BOTH',*DYNAMIC):
                a=score[(model,particles)];b=score[(model,2*particles)]
                rows.append({'horizon_min':h,'asset':asset,'model':model,'origins':len(g),
                             'particles_low':particles,'particles_high':2*particles,
                             'nll_low':a,'nll_high':b,'high_minus_low':b-a})
    return pd.DataFrame(rows)
