"""VD18: HAR versus Markov-state forecasts of Student-t RETURN variance.

Research-only historical exploration. Never reads later targets into features.
HAR is trained on TRAIN forward realized variance (causal target), its net-return
variance mapping is calibrated on TRAIN, and blend weights on EARLY VALID.
The native 5m Student-t return kernel and VD16 forecast-mixture tail shape are
unchanged. HAR anchor overlays are NOT a path-consistent Markov process.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import gc
import numpy as np
import pandas as pd
from data_engineering.modeling import markov_lift_distributions as v14
from data_engineering.modeling import student_t_return_markov as v15
from data_engineering.modeling import markov_return_horizon_calibration as v16
from data_engineering.modeling import markov_conditional_variance as v17

VERSION='VD18-0.1.0'
HORIZONS=v17.HORIZONS
PRIMARY=60
PARTICLES=256
BOOT=1000
CAL_LIMIT=512
FEATURE_SETS={'HAR_2':(30,240),'HAR_4':(30,60,240,1440)}
ALL_FEATURES=(30,60,240,1440)
MODELS=('AM_T_BOTH','AM_T_VAR_STATE','AM_T_VAR_ANCHOR',
        'AM_T_HAR_DIRECT','AM_T_HAR_BLEND','AM_T_HAR_STATE','M2_student_t')
METRICS=v17.METRICS
PAIRS=(('AM_T_VAR_STATE','AM_T_BOTH'),('AM_T_VAR_ANCHOR','AM_T_BOTH'),
       ('AM_T_HAR_DIRECT','AM_T_BOTH'),('AM_T_HAR_BLEND','AM_T_BOTH'),
       ('AM_T_HAR_STATE','AM_T_BOTH'),('AM_T_HAR_BLEND','AM_T_VAR_STATE'),
       ('AM_T_HAR_BLEND','AM_T_VAR_ANCHOR'),('AM_T_HAR_STATE','AM_T_HAR_BLEND'),
       ('AM_T_HAR_BLEND','M2_student_t'),('AM_T_HAR_STATE','M2_student_t'))

@dataclass(frozen=True)
class Registration:
    version:str=VERSION
    horizons:tuple=HORIZONS
    primary_horizon_min:int=PRIMARY
    har_lookbacks_min:tuple=ALL_FEATURES
    har_candidates:tuple=('HAR_2','HAR_4')
    particle_budget:int=PARTICLES
    calibration_max_origins_per_asset:int=CAL_LIMIT
    bootstrap_draws:int=BOOT
    train_fraction:float=.60
    valid_fraction:float=.20
    baseline_historical_status:str='HISTORICAL_EXPLORATORY'
    scope:str='Binance 5m spot returns; no additional market data or tick storage'
    note:str='HAR-anchored densities are horizon overlays, not coherent five-minute return transitions.'

def registration(): return asdict(Registration())

def add_har_lags(frame:pd.DataFrame)->pd.DataFrame:
    """Rolling lagged RV uses current/previous 5m returns only, never (t,t+h].

    Strict complete windows: any missing 5m return forces NaN in affected HAR
    features. Panels are already dense-reindexed to the uniform five-minute grid.
    """
    out=frame.copy()
    out=out.sort_values(['underlying','state_time']).copy()
    for lag in ALL_FEATURES:
        n=lag//5
        rv=(out.return_5m.astype(float)**2).groupby(out.underlying,sort=False).transform(
              lambda r:r.rolling(n,min_periods=n).sum())
        out[f'har_logrv_{lag}']=np.log(np.maximum(rv.to_numpy(float),1e-16))
    return out.sort_index()

def complete(frame:pd.DataFrame,h:int):
    """Identical origins for HAR and reference models in each horizon."""
    cols=[f'har_logrv_{k}' for k in ALL_FEATURES]
    z=frame.loc[frame.ready & np.isfinite(frame[cols].to_numpy(float)).all(axis=1)]
    return v14._get_split(z,h)

def _har_design(train:pd.DataFrame,other:pd.DataFrame,lookbacks:tuple):
    cols=[f'har_logrv_{k}' for k in lookbacks]
    asset_names=sorted(train.underlying.unique())
    c=train[cols].to_numpy(float).mean(axis=0)
    std=np.maximum(train[cols].to_numpy(float).std(axis=0),1e-7)
    def matrix(g):
        v=(g[cols].to_numpy(float)-c)/std
        ang=2*np.pi*g.slot.to_numpy(float)/288
        extras=[(g.underlying.to_numpy()==x).astype(float) for x in asset_names[1:]]
        return np.column_stack([np.ones(len(g)),v,np.sin(ang),np.cos(ang),*extras])
    return matrix(train),matrix(other),{'features':cols,'asset_names':asset_names,'center':c.tolist(),'std':std.tolist()}

def fit_har(train:pd.DataFrame,h:int,lookbacks:tuple):
    """Per-horizon HAR log future RV ridge, then TRAIN-only net-return scale.

    RV target differs from net-return variance. Smearing is calibrated directly
    against winsorized R_h^2 on TRAIN and never uses valid/test targets.
    """
    x,_,design=_har_design(train,train,lookbacks)
    # Per-asset TRAIN epsilon stabilizes log RV near zero; records zero incidence.
    floors={};zeros={}
    for a,g in train.groupby('underlying'):
        z=g[f'RV_{h}'].to_numpy(float)
        nonzero=z[z>0]
        floors[a]=float(max(np.median(nonzero)*.001 if len(nonzero) else 1e-12,1e-16))
        zeros[a]=int(np.sum(z==0))
    floor=train.underlying.map(floors).to_numpy(float)
    y=np.log(np.maximum(train[f'RV_{h}'].to_numpy(float),floor))
    beta=v14._ridge(x,y,alpha=.02)
    forecast=np.exp(np.clip(x@beta,-40,5))
    scales={}
    for asset,g in train.groupby('underlying'):
        ix=(train.underlying==asset).to_numpy()
        r=g[f'R_{h}'].to_numpy(float)
        wins=np.minimum(r*r,np.quantile(r*r,.995)).mean()
        scales[asset]=float(np.clip(wins/max(forecast[ix].mean(),1e-16),.01,100.))
    return {'lookbacks':list(lookbacks),'coefficients':beta.tolist(),
            'design':design,'net_return_scales':scales,'train_logrv_floors':floors,
            'train_zero_RV_counts':zeros,'train_origins':len(train)}

def predict_har(g:pd.DataFrame,fit:dict):
    d=fit['design']; cols=d['features']
    v=(g[cols].to_numpy(float)-np.asarray(d['center']))/np.asarray(d['std'])
    angle=2*np.pi*g.slot.to_numpy(float)/288
    extra=[(g.underlying.to_numpy()==x).astype(float) for x in d['asset_names'][1:]]
    X=np.column_stack([np.ones(len(g)),v,np.sin(angle),np.cos(angle),*extra])
    estimate=np.exp(np.clip(X@np.asarray(fit['coefficients']),-40,5))
    return np.maximum(estimate*g.underlying.map(fit['net_return_scales']).to_numpy(float),1e-12)

def _raw_var_ratio(harvar:np.ndarray,particles_var:np.ndarray,base:dict):
    """Exact log-HAR / log-baseline variance ratio, with NO clipping."""
    mean=particles_var.mean(axis=1)*np.exp(base['log_scale'])
    return np.log(np.maximum(harvar,1e-14)/np.maximum(mean,1e-14))

def _predict_var_ratio(harvar:np.ndarray,particles_var:np.ndarray,base:dict):
    """VD17-matched clipped ratio used ONLY by blend/joint calibrations."""
    return np.clip(_raw_var_ratio(harvar,particles_var,base),
                   -v17.MAX_ANCHOR_RATIO,v17.MAX_ANCHOR_RATIO)

def _calibration_sample(frame,h,kernels,transitions,har_fits,particles,limit):
    tr,va,te=complete(frame,h)
    cal,sel,boundary=v16.split_validation(va,h,max_per_asset=limit)
    pieces=[]
    for g in (cal,sel):
        v=np.zeros((len(g),particles))
        for asset,a in g.groupby('underlying',sort=True):
            where=np.flatnonzero((g.underlying==asset).to_numpy())
            v[where]=v16.integrate_subset(a,asset,h,kernels['AM_T'][asset],
                                          transitions[asset],particles)
        forecasts={key:predict_har(g,fit) for key,fit in har_fits.items()}
        pieces.append((g,v,forecasts))
    return pieces,boundary

def fit_har_overlays(frame,h,kernels,transitions,base,har_fits,*,particles=PARTICLES,limit=CAL_LIMIT):
    """HAR2/4 engine chosen on late VALID; all weights fitted on early VALID."""
    (cal,sel),boundary=_calibration_sample(frame,h,kernels,transitions,har_fits,particles,limit)
    g,orig,har=cal; sg,sorig,shar=sel
    transformed=v17.component_variances(orig,base)
    transformed_s=v17.component_variances(sorig,base)
    design=v17.design_fit(g);X=v17.design_apply(g,design);Xs=v17.design_apply(sg,design)
    y=g[f'R_{h}'].to_numpy(float);ys=sg[f'R_{h}'].to_numpy(float)
    trials=[]
    for family in FEATURE_SETS:
        za=_predict_var_ratio(har[family],orig,base)
        zs=_predict_var_ratio(shar[family],sorig,base)
        fit=v17.fit_dynamic('AM_T_VAR_ANCHOR',transformed,y,X,za)
        zdirect=_raw_var_ratio(shar[family],sorig,base)
        # Pure HAR variance replacement; predictive tail shape stays VD16's.
        pred_direct=float(v17.loss_components(transformed_s*np.exp(zdirect[:,None]),ys,
                        np.ones(len(ys)))['nll'].mean())
        pred_blend=float(v17.nll_from_components(transformed_s,ys,
                   v17.delta_for('AM_T_VAR_ANCHOR',Xs,zs,fit)).mean())
        trials.append({'family':family,'fit':fit,'cal_nll':float(v17.nll_from_components(transformed,y,
                    v17.delta_for('AM_T_VAR_ANCHOR',X,za,fit)).mean()),
                    'select_nll':pred_blend,'direct_select_nll':pred_direct})
    selected=min(trials,key=lambda z:z['select_nll'])
    family=selected['family']; z=_predict_var_ratio(har[family],orig,base)
    joint=v17.fit_dynamic('AM_T_VAR_COMBO',transformed,y,X,z)
    zs=_predict_var_ratio(shar[family],sorig,base)
    joint_sel=float(v17.nll_from_components(transformed_s,ys,
                 v17.delta_for('AM_T_VAR_COMBO',Xs,zs,joint)).mean())
    return {'selected_har':family,'har_trials':trials,'blend_fit':selected['fit'],
            'joint_fit':joint,'joint_select_nll':joint_sel,'design':design,
            'calibration_boundary':str(boundary),'cal_origins':len(g),'select_origins':len(sg)}

def evaluate_horizon(frame,h,kernels,transitions,hazard,basefit,statefit,har_fits,har_overlays,
                     *,particles=PARTICLES,block=1024):
    tr,va,te=complete(frame,h)
    # Re-use the original M2 density refitted on precisely the HAR-complete origins.
    _,_,_,q,ref,_=v15._reference_models(frame.loc[frame.index.isin(te.index)|
                  frame.index.isin(tr.index)|frame.index.isin(va.index)],h,transitions,hazard)
    # Ref helper returns separate train/valid/test; re-censoring on HAR features
    # avoids origin mismatch across models.
    assert len(q)==len(te)
    y=te[f'R_{h}'].to_numpy(float)
    m2v=v17._m2_variance(tr,te,h)
    predictions={name:{metric:np.zeros(len(te),float) for metric in
                 ('nll','tail_logloss','tail_brier','tail_predicted','tail_observed',
                  'interval_miss95','return_qlike','pred_variance')} for name in MODELS}
    for key in ('nll','tail_logloss','tail_brier','interval_miss95'):
        predictions['M2_student_t'][key]=ref['M2_student_t'][key].astype(float)
    predictions['M2_student_t']['tail_predicted']=ref['M2_student_t']['tail_probability'].astype(float)
    predictions['M2_student_t']['tail_observed']=(np.abs(y)>=q).astype(float)
    predictions['M2_student_t']['return_qlike']=np.log(m2v)+y*y/m2v
    predictions['M2_student_t']['pred_variance']=m2v
    pa=[]
    for asset,g in te.groupby('underlying',sort=True):
        mask=(te.underlying==asset).to_numpy(); ix=np.flatnonzero(mask)
        ph=g[['slow','gap']].to_numpy(float)
        x=v17.design_apply(g,statefit['design'])
        hx=v17.design_apply(g,har_overlays['design'])
        har_pred=predict_har(g,har_fits[har_overlays['selected_har']])
        q_asset=q[mask]; y_asset=y[mask]
        m2_asset=m2v[mask]
        for sl,pvar in v16.variance_particles(ph,transitions[asset],kernels['AM_T'][asset],
                                  h//5,particles,v17._seed(asset,h),block):
            original=v17.component_variances(pvar,basefit)
            z_m2=v17.anchor_ratio(g.iloc[sl],pvar,m2_asset[sl],basefit)
            z_har=_predict_var_ratio(har_pred[sl],pvar,basefit)
            delta={
                'AM_T_BOTH':np.zeros(len(z_har)),
                'AM_T_VAR_STATE':v17.delta_for('AM_T_VAR_STATE',x[sl],z_m2,statefit['dynamic']['AM_T_VAR_STATE']),
                'AM_T_VAR_ANCHOR':v17.delta_for('AM_T_VAR_ANCHOR',x[sl],z_m2,statefit['dynamic']['AM_T_VAR_ANCHOR']),
                'AM_T_HAR_DIRECT':_raw_var_ratio(har_pred[sl],pvar,basefit),
                'AM_T_HAR_BLEND':v17.delta_for('AM_T_VAR_ANCHOR',hx[sl],z_har,har_overlays['blend_fit']),
                'AM_T_HAR_STATE':v17.delta_for('AM_T_VAR_COMBO',hx[sl],z_har,har_overlays['joint_fit']),
            }
            for name,d in delta.items():
                val=v17.loss_components(original*np.exp(d[:,None]),y_asset[sl],q_asset[sl])
                for metric,value in val.items():predictions[name][metric][ix[sl]]=value
        for name in MODELS:
            pa.append({'horizon_min':h,'asset':asset,'model':name,'n_origins':len(g),
                       **{k:float(predictions[name][k][mask].mean()) for k in
                          ('nll','tail_logloss','tail_brier','tail_predicted','tail_observed',
                           'interval_miss95','return_qlike','pred_variance')}})
    scores=pd.DataFrame([{'horizon_min':h,'model':name,'n_origins':len(te),'n_days':te.date.nunique(),
                **{k:float(predictions[name][k].mean()) for k in
                   ('nll','tail_logloss','tail_brier','tail_predicted','tail_observed',
                    'interval_miss95','return_qlike','pred_variance')}} for name in MODELS])
    return scores,pd.DataFrame(pa),te[['date','underlying']],predictions

def run_all(frame,*,horizons=HORIZONS,particles=PARTICLES,B=BOOT,cal_limit=CAL_LIMIT,block=1024):
    if particles!=PARTICLES and particles!=16:raise ValueError('Only 256 for real data, 16 for synthetic tests')
    state=v14.fit_markov(frame.loc[(frame.split=='train')&frame.ready].copy())
    hazard=v14.fit_onebar_hazard(frame,state)
    tr5,va5,_=v14._get_split(frame,5)
    kernels,_=v15.fit_return_kernel(tr5,va5,state)
    base_cal,_=v16.prepare_calibrations(frame,kernels,state,horizons=horizons,
                                         particles=particles,max_per_asset=cal_limit)
    state_fits,_=v17.fit_variance_models(frame,kernels,state,base_cal,horizons=horizons,
                                         particles=particles,cal_limit=cal_limit)
    scores=[];asset_scores=[];boots=[];fits={};har_grid=[]
    for h in horizons:
        tr,_,_=complete(frame,h)
        candidates={k:fit_har(tr,h,FEATURE_SETS[k]) for k in FEATURE_SETS}
        baseline=base_cal[str(h)]['families']['AM_T_BOTH']
        overlay=fit_har_overlays(frame,h,kernels,state,baseline,candidates,
                                 particles=particles,limit=cal_limit)
        fits[str(h)]={'har_fits':candidates,'overlay':overlay}
        for c in overlay['har_trials']:
            har_grid.append({'horizon_min':h,'family':c['family'],'weight':c['fit']['anchor_weight'],
                             'cal_nll':c['cal_nll'],'late_valid_nll':c['select_nll'],
                             'direct_late_valid_nll':c['direct_select_nll'],
                             'chosen':c['family']==overlay['selected_har']})
        s,pa,te,p=evaluate_horizon(frame,h,kernels,state,hazard,baseline,state_fits[str(h)],candidates,overlay,
                                   particles=particles,block=block)
        scores.append(s);asset_scores.append(pa)
        for i,(a,b) in enumerate(PAIRS):
            for j,metric in enumerate(METRICS):
                d,lo,hi=v15._bootstrap(p[a][metric],p[b][metric],te.date.to_numpy(),B,
                                      seed=1801+h*107+10*i+j)
                boots.append({'horizon_min':h,'comparison':f'{a} - {b}','metric':metric,
                              'delta':d,'q025':lo,'q975':hi,'test_days':int(s.n_days.min()),
                              'origins':int(s.n_origins.min())})
        del s,pa,p,te,tr
        gc.collect()
    return (pd.concat(scores,ignore_index=True),pd.DataFrame(boots),
            pd.concat(asset_scores,ignore_index=True),pd.DataFrame(har_grid),
            {'version':VERSION,'fitted_har':fits,'vd17_state_fits':state_fits,
             'base_cal':base_cal,'native_return_kernels':kernels,'volatility_transition':state})


def numerical_sensitivity_validation(frame:pd.DataFrame,parameters:dict,
                                     *,horizons=(5,60,240),particles_low=256,
                                     max_per_asset=96):
    """Late-VALID-only paired QMC precision audit; frozen model parameters.

    Returns high-minus-low NLL at 2N vs N integration nodes for each HAR
    variant. Not used to tune any hyperparameter or choose HAR family.
    """
    if particles_low<8 or (particles_low & (particles_low-1)):
        raise ValueError('Sobol particle budget must be power of two')
    rows=[]
    kernels=parameters['native_return_kernels']
    transitions=parameters['volatility_transition']
    for h in horizons:
        _,va,_=complete(frame,h)
        _,g,_=v16.split_validation(va,h,max_per_asset=max_per_asset)
        fitted=parameters['fitted_har'][str(h)]
        overlay=fitted['overlay']
        har_fit=fitted['har_fits'][overlay['selected_har']]
        base=parameters['base_cal'][str(h)]['families']['AM_T_BOTH']
        for asset,chunk in g.groupby('underlying',sort=True):
            harv=predict_har(chunk,har_fit)
            X=v17.design_apply(chunk,overlay['design'])
            actual=chunk[f'R_{h}'].to_numpy(float)
            results={}
            for n in (particles_low,2*particles_low):
                particles=v16.integrate_subset(chunk,asset,h,kernels['AM_T'][asset],
                                               transitions[asset],n)
                fixed=v17.component_variances(particles,base)
                zblend=_predict_var_ratio(harv,particles,base)
                deltas={
                    'AM_T_HAR_DIRECT':_raw_var_ratio(harv,particles,base),
                    'AM_T_HAR_BLEND':v17.delta_for('AM_T_VAR_ANCHOR',X,zblend,overlay['blend_fit']),
                    'AM_T_HAR_STATE':v17.delta_for('AM_T_VAR_COMBO',X,zblend,overlay['joint_fit']),
                }
                results[n]={key:float(v17.nll_from_components(fixed,actual,d).mean())
                            for key,d in deltas.items()}
            for model in ('AM_T_HAR_DIRECT','AM_T_HAR_BLEND','AM_T_HAR_STATE'):
                rows.append({'horizon_min':h,'asset':asset,'model':model,'origins':len(chunk),
                             'particles_low':particles_low,'particles_high':2*particles_low,
                             'late_valid_nll_low':results[particles_low][model],
                             'late_valid_nll_high':results[2*particles_low][model],
                             'high_minus_low':results[2*particles_low][model]-results[particles_low][model]})
    return pd.DataFrame(rows)

def decision(scores,bs,fits):
    one=bs[(bs.horizon_min==PRIMARY)&(bs.metric=='nll')]
    def wins(a,b):
        q=one[one.comparison==f'{a} - {b}']
        return bool(len(q)==1 and q.delta.iloc[0]<0 and q.q975.iloc[0]<0)
    return {'status':'HISTORICAL_EXPLORATORY','primary_horizon_min':PRIMARY,
            'primary_test_days':int(scores[scores.horizon_min==PRIMARY].n_days.min()),
            'har_blend_beats_state':wins('AM_T_HAR_BLEND','AM_T_VAR_STATE'),
            'har_blend_beats_m2':wins('AM_T_HAR_BLEND','M2_student_t'),
            'har_state_beats_m2':wins('AM_T_HAR_STATE','M2_student_t'),
            'selected_har_by_horizon':{str(k):v['overlay']['selected_har'] for k,v in fits['fitted_har'].items()},
            'prospective_confirmation':False,'options_trade_approved':False}
