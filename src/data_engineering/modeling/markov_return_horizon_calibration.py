"""VD16: Exploratory horizon-specific density overlays for VD15's native AM-T.

The five-minute Student-t *return* transition kernel, volatility Markov factors
and QMC distribution integrator are inherited unchanged from VD15. At each
forecast horizon, a common scale factor s and mixture-shape factor a modify the
forecast *distribution*, not the 5m price transition. The horizon-specific
adjustments are NOT jointly path-consistent/Markovian by construction.

Train native transition on historical TRAIN, calibrate s/a using an early,
purged VALID portion, assess calibration selection on later VALID, and score
on historical TEST. Because this TEST history has previously been inspected,
ALL VD16 results remain exploratory and non-confirmatory.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import gc
import numpy as np
import pandas as pd
from scipy.special import logsumexp, ndtr
from scipy.optimize import minimize
from data_engineering.modeling import markov_lift_distributions as vd14
from data_engineering.modeling import student_t_return_markov as vd15

VERSION = 'VD16-0.1.0'
HORIZONS = vd15.HORIZONS
PRIMARY = 60
PARTICLES = 256
BOOT = 1000
CAL_ORIGINS_PER_ASSET = 512
SELECT_ORIGINS_PER_ASSET = 512
MODEL_NAMES = ('AM_T','AM_T_SCALE','AM_T_SHAPE','AM_T_BOTH','AM_G','M2_student_t')
FAMILIES = ('AM_T','AM_T_SCALE','AM_T_SHAPE','AM_T_BOTH')
PAIRS = (
    ('AM_T_SCALE','AM_T'), ('AM_T_SHAPE','AM_T'),
    ('AM_T_BOTH','AM_T'), ('AM_T_BOTH','M2_student_t'),
    ('AM_T_BOTH','AM_T_SCALE'), ('AM_T_BOTH','AM_T_SHAPE'),
    ('AM_T','M2_student_t'),
)
METRICS = ('nll','tail_logloss','tail_brier')

@dataclass(frozen=True)
class Registration:
    version: str=VERSION
    horizons: tuple=HORIZONS
    primary_horizon_min: int=PRIMARY
    source: str='VD15 native 5m Student-t RETURN transition and two-factor VAR state'
    particle_count: int=PARTICLES
    max_calibration_origins_per_asset: int=CAL_ORIGINS_PER_ASSET
    max_selection_origins_per_asset: int=SELECT_ORIGINS_PER_ASSET
    validation_calibration_fraction: float=.60
    min_horizon_embargo_min: int=5
    log_scale_bounds: tuple=(-.70,.70)
    shape_bounds: tuple=(.45,2.00)
    small_shrinkage_penalty: float=.002
    bootstrap_draws: int=BOOT
    status: str='HISTORICAL_EXPLORATORY'
    confirmation: str='Prior branches examined historical test period; requires fresh post-2026-10-14 data'

def registration(): return asdict(Registration())


def transform_variances(v:np.ndarray, log_scale:float=0.,shape:float=1.)->np.ndarray:
    """Preserve each forecast's particle-mean variance under the shape warp.

    v'(origin,i) = exp(log_scale)*mean(v) * v_i^shape / mean(v^shape).
    Shape 1, scale 1 recovers VD15 exactly (up to floating point).
    Shape 0 produces a single Gaussian variance; shape >1 creates broader
    variance mixing and heavier predictive tails, without changing E[var].
    """
    v=np.maximum(np.asarray(v,float),1e-14)
    if v.ndim!=2: raise ValueError('v must be (origins, particles)')
    if shape<=0: raise ValueError('shape must be strictly positive')
    lv=np.log(v)
    z=shape*lv
    normalized=np.exp(z-logsumexp(z,axis=1,keepdims=True)+np.log(v.shape[1]))
    return np.maximum(np.exp(log_scale)*v.mean(axis=1,keepdims=True)*normalized,1e-14)


def predict_loss(v:np.ndarray,y:np.ndarray,q:np.ndarray,*,log_scale:float=0.,shape:float=1.):
    """Proper normalized mixture-of-normals density and two-sided tail scores."""
    vp=transform_variances(v,log_scale,shape)
    yy=np.asarray(y,float)
    threshold=np.asarray(q,float)
    if vp.shape[0]!=len(yy) or len(threshold)!=len(yy):
        raise ValueError('Outcome/threshold length mismatch')
    logpdf=-.5*(np.log(2*np.pi*vp)+yy[:,None]**2/vp)
    nll=-logsumexp(logpdf,axis=1)+np.log(vp.shape[1])
    sd=np.sqrt(vp)
    tail=np.clip((2*ndtr(-threshold[:,None]/sd)).mean(axis=1),1e-10,1-1e-10)
    event=(np.abs(yy)>=threshold).astype(float)
    tll=-(event*np.log(tail)+(1-event)*np.log1p(-tail))
    outside=(2*ndtr(-np.abs(yy)[:,None]/sd)).mean(axis=1)<.05
    return {'nll':nll,'tail_logloss':tll,'tail_brier':(event-tail)**2,
            'tail_predicted':tail,'tail_observed':event,'interval_miss95':outside.astype(float)}


def variance_particles(state:np.ndarray, transition:dict, kernel:dict,
                       steps:int, particles:int,seed:int,block_size:int=2048):
    """Exact same QMC conditional Gaussian component variances as VD15 AM_T.

    Compute one block at a time to keep full historical testing memory-bounded.
    Output is yielded (slice, variance array); do not retain whole test panel.
    """
    state=np.asarray(state,float)
    if not state.ndim==2 or state.shape[1]!=2: raise ValueError('state must be Nx2')
    df=float(kernel['nu'])
    if abs(float(kernel['phi']))>1e-12:
        raise ValueError('VD16 calibrates native zero-AR AM-T; return AR not authorized')
    deviations,inv_chi=vd15._sobol_paths(steps,particles,df,transition,seed)
    mean=np.asarray(transition['mean'],float)
    phi=np.asarray(transition['phi'],float)
    powers=phi[None,:]**np.arange(1,steps+1)[:,None]
    for start in range(0,len(state),block_size):
        end=min(len(state),start+block_size)
        x=state[start:end]
        v=np.zeros((len(x),particles),float)
        for j in range(steps):
            m=mean[None,:]+(x-mean[None,:])*powers[j]
            intensity=np.exp(np.clip(m.sum(axis=1),-20,15))
            extra=np.exp(np.clip(deviations[j].sum(axis=1),-15,15))
            v+=float(kernel['scale'])*intensity[:,None]*(extra*inv_chi[j])[None,:]
        yield slice(start,end),np.maximum(v,1e-14)


def integrate_subset(g:pd.DataFrame,asset:str,horizon:int,kernel:dict,
                     transition:dict,particles:int,block_size:int=2048):
    """Small VALID subset of component variances for fitting calibration."""
    seed=15000+int(np.sum(np.frombuffer(asset.encode(),dtype=np.uint8)))+(horizon//5)*100
    vs=[v for _,v in variance_particles(g[['slow','gap']].to_numpy(float),transition,kernel,
                  horizon//5,particles,seed,block_size)]
    return np.concatenate(vs,axis=0)


def sample_uniform_days(g:pd.DataFrame,limit:int):
    """Deterministic temporal coverage; no label-dependent sample selection."""
    if len(g)<=limit:return g
    pos=np.linspace(0,len(g)-1,limit,dtype=int)
    return g.iloc[pos].copy()


def split_validation(valid:pd.DataFrame,horizon:int,fraction=.6,
                     max_per_asset=CAL_ORIGINS_PER_ASSET):
    """Early validation calibration and later validation model-choice window.

    Exclude origin cal observations whose *targets* end after select boundary.
    """
    days=np.sort(valid.date.unique())
    if len(days)<8:raise ValueError('Need >=8 validation days for chronological split')
    cut=days[max(1,min(len(days)-1,int(len(days)*fraction)))]
    max_end=pd.Timestamp(cut)-pd.Timedelta(minutes=horizon)
    cal=valid.loc[valid.state_time<=max_end]
    choose=valid.loc[valid.state_time>=cut]
    if cal.empty or choose.empty:raise ValueError('Purged validation split is empty')
    cal=pd.concat([sample_uniform_days(x.sort_values('state_time'),max_per_asset)
                   for _,x in cal.groupby('underlying')],ignore_index=True)
    choose=pd.concat([sample_uniform_days(x.sort_values('state_time'),max_per_asset)
                   for _,x in choose.groupby('underlying')],ignore_index=True)
    return cal,choose,str(cut)


def fit_calibration(v:np.ndarray,y:np.ndarray, family:str,
                    *,penalty:float=.002,max_iter:int=55):
    if family not in FAMILIES:raise ValueError('Invalid calibration family')
    if family=='AM_T': return {'log_scale':0.,'shape':1.,'fit_nll':float(predict_nll(v,y,0.,1.)),'success':True}
    def decode(x):
        if family=='AM_T_SCALE':return float(x[0]),1.
        if family=='AM_T_SHAPE':return 0.,float(x[0])
        return float(x[0]),float(x[1])
    bounds={'AM_T_SCALE':[(-.70,.70)],'AM_T_SHAPE':[(.45,2.)],
            'AM_T_BOTH':[(-.70,.70),(.45,2.)]}[family]
    x0=[0.] if family=='AM_T_SCALE' else [1.] if family=='AM_T_SHAPE' else [0.,1.]
    def fn(x):
        scale,shape=decode(x)
        regularizer=penalty*((scale/.35)**2+((shape-1)/.5)**2)
        return predict_nll(v,y,scale,shape)+regularizer
    opt=minimize(fn,x0,bounds=bounds,method='L-BFGS-B',options={'maxiter':max_iter,'ftol':1e-8})
    s,a=decode(opt.x)
    return {'log_scale':s,'shape':a,'fit_nll':float(predict_nll(v,y,s,a)),
            'penalized_fit_objective':float(opt.fun),'success':bool(opt.success),
            'iterations':int(opt.nit)}


def predict_nll(v,y,log_scale,shape):
    vp=transform_variances(v,log_scale,shape)
    z=np.asarray(y,float)[:,None]
    loss=-logsumexp(-.5*(np.log(2*np.pi*vp)+z*z/vp),axis=1)+np.log(vp.shape[1])
    return float(loss.mean())


def prepare_calibrations(frame:pd.DataFrame,kernels:dict,markov:dict,*,
                         horizons=HORIZONS,particles=PARTICLES,
                         max_per_asset=CAL_ORIGINS_PER_ASSET):
    """Fit global, horizon-specific calibrators on early VALID only.

    All six assets equally sampled; cannot choose parameters using TEST.
    Second VALID slice reports independently held-back choice scores.
    """
    fits={};sweep=[]
    for h in horizons:
        _,va,_=vd14._get_split(frame,h)
        cal,select,boundary=split_validation(va,h,max_per_asset=max_per_asset)
        arrays=[]
        for sub,tag in [(cal,'cal'),(select,'select')]:
            all_v=[];all_y=[]
            for asset,g in sub.groupby('underlying',sort=True):
                v=integrate_subset(g,asset,h,kernels['AM_T'][asset],markov[asset],particles)
                all_v.append(v);all_y.append(g[f'R_{h}'].to_numpy(float))
            arrays.append((np.concatenate(all_v),np.concatenate(all_y)))
        (vcal,ycal),(vsel,ysel)=arrays
        candidates={}
        for family in FAMILIES:
            fit=fit_calibration(vcal,ycal,family)
            fit['select_nll']=predict_nll(vsel,ysel,fit['log_scale'],fit['shape'])
            candidates[family]=fit
            sweep.append({'horizon_min':h,'family':family,'log_scale':fit['log_scale'],
                          'variance_multiplier':np.exp(fit['log_scale']),
                          'shape':fit['shape'],'fit_nll':fit['fit_nll'],
                          'select_nll':fit['select_nll'],'optimization_success':fit['success'],
                          'cal_origins':len(ycal),'select_origins':len(ysel),'select_boundary':boundary})
        selected=min(FAMILIES,key=lambda n:candidates[n]['select_nll'])
        fits[str(h)]={'families':candidates,'validation_selected_family':selected,
                      'validation_selection_boundary':boundary,
                      'cal_origins':len(ycal),'select_origins':len(ysel)}
    return fits,pd.DataFrame(sweep)


def evaluate_horizon(frame:pd.DataFrame,horizon:int,kernels:dict,
                     markov:dict,hazard:dict,calibration:dict,*,
                     particles=PARTICLES,block_size=1536):
    tr,va,te,q,ref,_=vd15._reference_models(frame,horizon,markov,hazard)
    actual=te[f'R_{horizon}'].to_numpy(float)
    results={n:{k:np.zeros(len(te)) for k in ('nll','tail_logloss','tail_brier',
                'tail_predicted','tail_observed','interval_miss95')} for n in MODEL_NAMES}
    for old,new in [('M2_student_t','M2_student_t'),('AM_G','AM_G')]:
        for k in ('nll','tail_logloss','tail_brier','interval_miss95'):
            results[new][k]=ref[old][k].astype(float)
        results[new]['tail_predicted']=ref[old]['tail_probability'].astype(float)
        results[new]['tail_observed']=(np.abs(actual)>=q).astype(float)
    per_asset=[]
    for asset,g in te.groupby('underlying',sort=True):
        mask=(te.underlying==asset).to_numpy()
        asset_y=g[f'R_{horizon}'].to_numpy(float)
        asset_q=q[mask]
        kernel=kernels['AM_T'][asset]
        trans=markov[asset]
        seed=15000+int(np.sum(np.frombuffer(asset.encode(),dtype=np.uint8)))+(horizon//5)*100
        st=g[['slow','gap']].to_numpy(float)
        for sl,v in variance_particles(st,trans,kernel,horizon//5,particles,seed,block_size):
            for name in FAMILIES:
                c=calibration['families'][name]
                out=predict_loss(v,asset_y[sl],asset_q[sl],log_scale=c['log_scale'],shape=c['shape'])
                for metric in out:results[name][metric][np.flatnonzero(mask)[sl]]=out[metric]
        for name in MODEL_NAMES:
            per_asset.append({'horizon_min':horizon,'asset':asset,'model':name,
                'origins':len(g),**{metric:float(results[name][metric][mask].mean()) for metric in
                ('nll','tail_logloss','tail_brier','tail_predicted','tail_observed','interval_miss95')}})
    scores=pd.DataFrame([{'horizon_min':horizon,'model':name,'origins':len(te),
        'test_days':te.date.nunique(),**{metric:float(np.mean(results[name][metric])) for metric in
               ('nll','tail_logloss','tail_brier','tail_predicted','tail_observed','interval_miss95')}}
              for name in MODEL_NAMES])
    return scores,pd.DataFrame(per_asset),te[['date','underlying']],results


def run_all(frame:pd.DataFrame,*,horizons=HORIZONS,particles=PARTICLES,B=BOOT,
            cal_limit=CAL_ORIGINS_PER_ASSET,chunk_size=1536):
    markov=vd14.fit_markov(frame.loc[(frame.split=='train')&frame.ready].copy())
    hazard=vd14.fit_onebar_hazard(frame,markov)
    tr5,va5,_=vd14._get_split(frame,5)
    kernels,_=vd15.fit_return_kernel(tr5,va5,markov)
    calibrations,calgrid=prepare_calibrations(frame,kernels,markov,horizons=horizons,
                                            particles=particles,max_per_asset=cal_limit)
    scores=[];assets=[];boot=[]
    for h in horizons:
        sc,pa,te,p=evaluate_horizon(frame,h,kernels,markov,hazard,
            calibrations[str(h)],particles=particles,block_size=chunk_size)
        scores.append(sc);assets.append(pa)
        for i,(a,b) in enumerate(PAIRS):
            for j,metric in enumerate(METRICS):
                delta,lo,hi=vd15._bootstrap(p[a][metric],p[b][metric],te.date.to_numpy(),B,
                   seed=1601+h*100+10*i+j)
                boot.append({'horizon_min':h,'comparison':f'{a} - {b}','metric':metric,
                             'delta':delta,'q025':lo,'q975':hi,
                             'test_days':te.date.nunique(),'origins':len(te)})
        del p,te
        gc.collect()
    return pd.concat(scores,ignore_index=True),pd.DataFrame(boot),pd.concat(assets,ignore_index=True),calgrid,{'calibrations':calibrations,'native_return_kernels':kernels,'state_transitions':markov,
         'particle_count':particles,'version':VERSION}


def decision(scores:pd.DataFrame, boot:pd.DataFrame, fits:dict):
    pri=boot[(boot.horizon_min==PRIMARY)&(boot.metric=='nll')]
    def beats(a,b):
        x=pri.loc[pri.comparison==f'{a} - {b}']
        return bool(len(x)==1 and x.delta.iloc[0]<0 and x.q975.iloc[0]<0)
    return {'status':'HISTORICAL_EXPLORATORY','primary_horizon_min':PRIMARY,
       'n_test_days':int(scores.loc[scores.horizon_min==PRIMARY,'test_days'].min()),
       'flags':{'scale_beats_AM_T':beats('AM_T_SCALE','AM_T'),
                'shape_beats_AM_T':beats('AM_T_SHAPE','AM_T'),
                'both_beats_AM_T':beats('AM_T_BOTH','AM_T'),
                'both_beats_M2':beats('AM_T_BOTH','M2_student_t')},
       'selected_using_validation_only':{h:data['validation_selected_family'] for h,data in fits['calibrations'].items()},
       'prospective_confirmation':False,'trading_approved':False,
       'warning':'TEST segment previously viewed in VD13-15. No inferential novelty/confirmation.'}


def particle_sensitivity_validation(frame:pd.DataFrame,kernels:dict,markov:dict,
                                  calibrations:dict,*,horizons=(5,60,240),
                                  particles=PARTICLES,max_per_asset=96):
    """On late VALID only: nested Sobol N and 2N, never historical TEST.

    Compares calibrated-versus-uncalibrated NLL gain for numerical stability.
    No model adjustments based on this diagnostic are permitted on TEST.
    """
    rows=[]
    for h in horizons:
        _,va,_=vd14._get_split(frame,h)
        _,select,_=split_validation(va,h,max_per_asset=max_per_asset)
        both=calibrations[str(h)]['families']['AM_T_BOTH']
        for asset,g in select.groupby('underlying',sort=True):
            kernel=kernels['AM_T'][asset]
            state=g[['slow','gap']].to_numpy(float)
            y=g[f'R_{h}'].to_numpy(float)
            seed=15000+int(np.sum(np.frombuffer(asset.encode(),dtype=np.uint8)))+(h//5)*100
            vp_low=integrate_subset(g,asset,h,kernel,markov[asset],particles)
            vp_hi=integrate_subset(g,asset,h,kernel,markov[asset],2*particles)
            for name,settings in [('AM_T',{'log_scale':0.,'shape':1.}),
                                  ('AM_T_BOTH',both)]:
                low=predict_nll(vp_low,y,settings['log_scale'],settings['shape'])
                high=predict_nll(vp_hi,y,settings['log_scale'],settings['shape'])
                rows.append({'horizon_min':h,'asset':asset,'family':name,'origins':len(g),
                       'particles_low':particles,'particles_high':particles*2,
                       'validation_nll_low':low,'validation_nll_high':high,
                       'high_minus_low':high-low})
    return pd.DataFrame(rows)
