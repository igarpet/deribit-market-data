"""VD15: Student-t Markov *return* transitions (five-minute price dynamics).

Pre-registered research extension within Volatility Dynamics. Explicitly reuses
VD14's causal Binance spot panel, 30m/240m de-clocked features and two archived
benchmark specifications. No cross-branch models, options data or DB writes.

r[t+1] | r[t],L[t+1],F[t+1] ~ standardized Student-t_nu(
    location=phi*r[t], variance=c_asset*exp(L[t+1]+F[t+1])).
State (L,F) follows VD14's train-estimated, diagonal Gaussian VAR(1).

Prediction at k bars is the implied convolution of *native 5m Student-t kernels*;
the Student-t is NOT fitted anew to terminal k-bar returns. The calculation uses
Student-t's exact inverse-chi-square Gaussian scale-mixture representation, and
Rao-Blackwellized randomized-Sobol integration over future state innovations and
innovation scales. A finite particle count is numerical approximation, not a
structural change to the native Student-t Markov law.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
from pathlib import Path
import gc
import numpy as np
import pandas as pd
from scipy.special import logsumexp,ndtr
from scipy.stats import qmc,chi2

from data_engineering.modeling import markov_lift_distributions as vd14

VERSION='VD15-0.1.0'
HORIZONS=(5,10,15,30,60,120,240)
MODELS=('T_RW','T_MC','AM_G','AM_T','AM_T_AR','M2_student_t')
PRIMARY=60
DF_GRID=(3.,4.,6.,10.,20.,40.)
PARTICLES_DEFAULT=256
B_DEFAULT=1000
MIN_TEST_DAYS=30
PAIRS=(('T_MC','T_RW'),('AM_T','AM_G'),('AM_T','M2_student_t'),
       ('AM_T_AR','AM_T'),('AM_T_AR','M2_student_t'),('AM_G','M2_student_t'))
METRICS=('nll','tail_logloss','tail_brier')

@dataclass(frozen=True)
class Spec:
    version:str=VERSION
    horizons:tuple[int,...]=HORIZONS
    primary_horizon_min:int=PRIMARY
    df_grid:tuple[float,...]=DF_GRID
    sobol_particles:int=PARTICLES_DEFAULT
    n_bootstrap:int=B_DEFAULT
    split:str='VD14 60/20/20 chronological UTC days; same valid origins per horizon'
    historical_end_exclusive:str=str(vd14.SAMPLE_END)
    historical_is_exploratory:bool=True
    source_module:str='markov_lift_distributions.py (same Volatility Dynamics branch)'
    model_selection:str='AR and scale TRAIN; native 5m nu VALID; no TEST tuning'
    nll_primary:str='AM_T vs both AM_G and M2; density and tail calibration jointly'


def registration():
    return asdict(Spec())


def winsorized_second_moment(y:np.ndarray, quantile=.995)->float:
    yy=np.asarray(y,float)**2
    cap=np.quantile(yy,quantile)
    return float(max(np.mean(np.minimum(yy,cap)),1e-12))


def fit_return_kernel(train:pd.DataFrame,valid:pd.DataFrame,transitions:dict):
    """Fit *native 5m* location, residual scale and nu, then freeze at all horizons.

    Independent train/validation decisions per of the four new variants. Markov
    scale uses next-step state forecast (without observing a future state).
    """
    params={name:{} for name in ('T_RW','T_MC','AM_T','AM_T_AR')}
    diagnostics={}
    for asset in sorted(train.underlying.unique()):
        tr=train.loc[train.underlying==asset]
        va=valid.loc[valid.underlying==asset]
        rt=tr['R_5'].to_numpy(float)
        rcur=tr.return_5m.to_numpy(float)
        phi_raw=float(np.dot(rcur,rt)/(np.dot(rcur,rcur)+.01*len(tr)*np.var(rcur)))
        phi=float(np.clip(phi_raw,-.5,.5))
        state_tr=tr[['slow','gap']].to_numpy(float)
        state_va=va[['slow','gap']].to_numpy(float)
        etr=vd14._markov_rv_path(state_tr,transitions[asset],1)
        eva=vd14._markov_rv_path(state_va,transitions[asset],1)
        models={
           'T_RW':(0., np.ones(len(tr)),np.ones(len(va))),
           'T_MC':(phi, np.ones(len(tr)),np.ones(len(va))),
           'AM_T':(0.,etr,eva),
           'AM_T_AR':(phi,etr,eva),
        }
        rv=va['R_5'].to_numpy(float)
        for model,(ar,scale_tr,scale_va) in models.items():
            residual=rt-ar*rcur
            # scale_tr is a forecast for the next bar's ex-ante variance proxy.
            c=winsorized_second_moment(residual)/float(scale_tr.mean())
            vval=np.maximum(c*scale_va,1e-12)
            loc=ar*va.return_5m.to_numpy(float)
            ll={nu:float(vd14.t_logpdf(rv-loc,vval,nu).mean()) for nu in DF_GRID}
            df=max(DF_GRID,key=lambda d:ll[d])
            params[model][asset]={'phi':ar,'scale':float(c),'nu':float(df),
                'validation_nll':float(-ll[df])}
        diagnostics[asset]={'return_phi':phi,'return_phi_unclipped':phi_raw,
                            'n_train':len(tr),'n_valid':len(va)}
    return params,diagnostics


def _sobol_paths(steps:int,particles:int,df:float,transitions:dict,seed:int):
    """Common integrator nodes; paths vary by particle, not across target origins.

    QMC columns: 2 correlated Gaussian factor innovations + 1 chi-square scale
    per time step. The same integration nodes across models reduce numerical
    comparison noise. Scrambling is deterministic and independent of test data.
    """
    if particles<8 or particles&(particles-1):
        raise ValueError('particles must be a power of two >= 8')
    from scipy.special import ndtri
    x=qmc.Sobol(d=3*steps,scramble=True,seed=seed).random_base2(int(np.log2(particles)))
    x=np.clip(x,1e-10,1-1e-10).reshape(particles,steps,3)
    z=ndtri(x[:,:,:2]);Q=np.asarray(transitions['innovation_cov'],float)
    L=np.linalg.cholesky(Q)
    shocks=z@L.T
    gamma=chi2.ppf(x[:,:,2],df=df)
    factors=np.clip((df-2)/gamma,1e-4,1e5)
    # innovations in the two conditional-volatility states as offsets from their
    # deterministic conditional means, starting with zero deviation at origin.
    deviations=np.zeros((steps,particles,2),float)
    phi=np.asarray(transitions['phi'],float)
    state=np.zeros((particles,2),float)
    for j in range(steps):
        state=phi[None,:]*state+shocks[:,j,:]
        deviations[j]=state
    return deviations,factors.T


def _integrate_student(
    state:np.ndarray,previous_return:np.ndarray,
    transition:dict, *, ar:float,scale:float,df:float,steps:int,
    particles:int,seed:int,
    variable_state:bool=True,
    q:np.ndarray|None=None,realized:np.ndarray|None=None,
    block_size:int=2048,
):
    """Conditional multi-step Student-t price-return predictive density.

    Student t is at *each 5m return transition*. Conditional on inverse-chi²
    precision scales and future (L,F) states, cumulative return is Gaussian.
    For AR(1): R_h mean=sum_{j=1}^k ar^j * r_t; each future innovation j
    contributes weight sum_{ell=j}^k ar^(ell-j). Integrating conditional normal
    densities over fixed randomized Sobol nodes is Rao-Blackwellization.

    Identical return-innovation law at every horizon, not separately fitted
    Student-t terminal laws. Outputs arrays in origin order.
    """
    if steps<1:raise ValueError('steps must be >=1')
    state=np.asarray(state,float)
    r0=np.asarray(previous_return,float)
    n=len(r0)
    if state.shape!=(n,2):raise ValueError('state must be (origins,2)')
    if realized is not None and (q is None or len(realized)!=n or len(q)!=n):
        raise ValueError('realized, q must both have length origins')
    dev,gamma=_sobol_paths(steps,particles,df,transition,seed)
    mean=np.asarray(transition['mean'],float)
    phi=np.asarray(transition['phi'],float)
    powers=phi[None,:]**np.arange(1,steps+1)[:,None]
    # Resulting gain of innovation at j on the sum of future returns.
    j=np.arange(steps,0,-1)
    if abs(1-ar)<1e-8:
        weights=j.astype(float)
    else:
        weights=(1-np.power(ar,j))/(1-ar)
    drift=r0*sum(ar**i for i in range(1,steps+1))
    output={'nll':np.empty(n),'tail_logloss':np.empty(n),
            'tail_brier':np.empty(n),'interval_miss95':np.empty(n),
            'tail_probability':np.empty(n)} if realized is not None else {}
    for start in range(0,n,block_size):
        end=min(n,start+block_size)
        x=state[start:end]
        # particle conditional covariance is integrated, not collapsed to mean.
        v=np.zeros((len(x),particles),float)
        for k in range(steps):
            if variable_state:
                mu=mean[None,:]+(x-mean[None,:])*powers[k]
                log_intensity=np.clip(mu.sum(axis=1),-20,15)
                vol=scale*np.exp(log_intensity)
                extra=np.exp(np.clip(dev[k].sum(axis=1),-15,15))
                v+=(weights[k]**2)*vol[:,None]*(extra*gamma[k])[None,:]
            else:
                v+=(weights[k]**2)*scale*gamma[k][None,:]
        v=np.maximum(v,1e-14)
        if realized is not None:
            obs=np.asarray(realized[start:end],float)
            th=np.asarray(q[start:end],float)
            m=drift[start:end]
            z=(obs-m)[:,None]
            logpdf=-.5*(np.log(2*np.pi*v)+z*z/v)
            loss=-logsumexp(logpdf,axis=1)+np.log(particles)
            # Tail event is |total return|>training q, around zero not drift.
            sd=np.sqrt(v)
            prob=np.clip(np.mean(ndtr((m[:,None]-th[:,None])/sd)+
                                 ndtr((-m[:,None]-th[:,None])/sd),axis=1),1e-9,1-1e-9)
            event=np.abs(obs)>=th
            tail_loss=-(event*np.log(prob)+(~event)*np.log1p(-prob))
            brier=(event-prob)**2
            miss=np.mean(2*ndtr(-np.abs(z)/sd),axis=1)<.05
            for key,data in [('nll',loss),('tail_logloss',tail_loss),('tail_brier',brier),
                             ('interval_miss95',miss),('tail_probability',prob)]:
                output[key][start:end]=data
    return output


def _reference_models(frame:pd.DataFrame,horizon:int,markov:dict,hazard:dict):
    """Recompute VD14 *original definitions* on current identical origin mask.

    Benchmarks are not approximated from aggregated published summaries. Their
    per-origin losses must be recomputed for valid paired comparisons.
    """
    tr,va,te=vd14._get_split(frame,horizon)
    vtr,_=vd14._rv_scale(tr,tr,horizon)
    vva,_=vd14._rv_scale(tr,va,horizon)
    vte,_=vd14._rv_scale(tr,te,horizon)
    rv=va[f'R_{horizon}'].to_numpy(float)
    actual=te[f'R_{horizon}'].to_numpy(float)
    df=min(vd14.DF_GRID,key=lambda d:-vd14.t_logpdf(rv,vva,d).mean())
    _,_,am_v,_,am_logrv=vd14.markov_variances(tr,va,te,markov,horizon)
    amp=vd14._markov_hazard_prob(te,markov,hazard['models'],horizon//5)
    ratio=np.sqrt(1+(hazard['ratio_5m']**2-1)/(horizon//5))
    # The 2 baseline predictive distributions exactly match VD14's code.
    q_asset={asset:float(np.quantile(np.abs(g[f'R_{horizon}']),vd14.TAIL_Q))
             for asset,g in tr.groupby('underlying')}
    q=te.underlying.map(q_asset).to_numpy(float)
    target=(np.abs(actual)>=q)
    scores={}
    b=[('M2_student_t',vd14.t_logpdf(actual,vte,df),
        vd14.t_abs_prob(q,vte,df),vd14.t_abs_prob(np.abs(actual),vte,df)),
       ('AM_G',vd14.mix_logpdf(actual,am_v,amp,ratio),
        vd14.mix_abs_prob(q,am_v,amp,ratio),vd14.mix_abs_prob(np.abs(actual),am_v,amp,ratio))]
    for name,logp,p_tail,pit in b:
        p_tail=np.clip(p_tail,1e-9,1-1e-9)
        scores[name]={'nll':-logp,'tail_logloss':-(target*np.log(p_tail)+(~target)*np.log1p(-p_tail)),
              'tail_brier':(target-p_tail)**2,'interval_miss95':(pit<.05),
              'tail_probability':p_tail}
    return tr,va,te,q,scores,{'reference_student_df':float(df),'q95_asset':q_asset}


def _score_horizon(frame:pd.DataFrame,horizon:int,kernels:dict,transitions:dict,
                   hazard:dict,particles:int=PARTICLES_DEFAULT,chunk_size:int=2048):
    tr,va,te,threshold,ref,meta=_reference_models(frame,horizon,transitions,hazard)
    k=horizon//5
    observations=te[f'R_{horizon}'].to_numpy(float)
    predictions=[]
    per_asset={}
    # Shared origins and real outcomes for the entire six-model comparison.
    all_data={name:{key:np.zeros(len(te),float) for key in
              ('nll','tail_logloss','tail_brier','interval_miss95','tail_probability')}
              for name in MODELS}
    for name in ref:
        for key,ar in ref[name].items():all_data[name][key]=ar.astype(float)
    for asset,g in te.groupby('underlying',sort=True):
        mask=(te.underlying==asset).to_numpy()
        base=g[['slow','gap']].to_numpy(float)
        previous=g.return_5m.to_numpy(float)
        sample_y=g[f'R_{horizon}'].to_numpy(float)
        sample_q=threshold[mask]
        per_asset[asset]={}
        sample_event=np.abs(sample_y)>=sample_q
        for name in ('AM_G','M2_student_t'):
            ref_loss=all_data[name]
            per_asset[asset][name]={
                'nll':float(np.mean(ref_loss['nll'][mask])),
                'tail_logloss':float(np.mean(ref_loss['tail_logloss'][mask])),
                'interval_miss95':float(np.mean(ref_loss['interval_miss95'][mask])),
                'tail_predicted':float(np.mean(ref_loss['tail_probability'][mask])),
                'n_origins':int(len(g))}
        for model in ('T_RW','T_MC','AM_T','AM_T_AR'):
            kernel=kernels[model][asset]
            is_augmented=model in ('AM_T','AM_T_AR')
            x=_integrate_student(base,previous,transitions[asset],ar=kernel['phi'],
                   scale=kernel['scale'],df=kernel['nu'],steps=k,particles=particles,
                   seed=15000+int(np.sum(np.frombuffer(asset.encode(),dtype=np.uint8)))+k*100,
                   variable_state=is_augmented,realized=sample_y,q=sample_q,
                   block_size=chunk_size)
            for key,val in x.items():all_data[model][key][mask]=val
            per_asset[asset][model]={'nll':float(np.mean(x['nll'])),
              'tail_logloss':float(np.mean(x['tail_logloss'])),
              'interval_miss95':float(np.mean(x['interval_miss95'])),
              'tail_predicted':float(np.mean(x['tail_probability'])),
              'n_origins':int(len(g))}
    days=int(te.date.nunique());n=len(te)
    outcome=(np.abs(observations)>=threshold).astype(float)
    for name in MODELS:
        d=all_data[name]
        predictions.append({'horizon_min':horizon,'model':name,'n_origins':n,'n_days':days,
            'nll':float(d['nll'].mean()),'tail_logloss':float(d['tail_logloss'].mean()),
            'tail_brier':float(d['tail_brier'].mean()),
            'tail_observed':float(outcome.mean()),
            'tail_predicted':float(d['tail_probability'].mean()),
            'interval_miss95':float(d['interval_miss95'].mean())})
    meta['per_asset']=per_asset
    meta['train_origins']=len(tr)
    meta['valid_origins']=len(va)
    meta['test_origins']=len(te)
    return pd.DataFrame(predictions),te[['date','underlying']].copy(),all_data,meta


def _bootstrap(a:np.ndarray,b:np.ndarray,day_index:np.ndarray,B:int,seed:int):
    if len(a)!=len(b) or len(a)!=len(day_index):raise ValueError('mismatched origins')
    dif=a-b;days,index=np.unique(day_index,return_inverse=True)
    sums=np.bincount(index,weights=dif,minlength=len(days))
    counts=np.bincount(index,minlength=len(days))
    rng=np.random.default_rng(seed)
    draws=rng.integers(len(days),size=(B,len(days)))
    sampled=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    return float(dif.mean()),float(np.quantile(sampled,.025)),float(np.quantile(sampled,.975))


def run_all(frame:pd.DataFrame,*,horizons=HORIZONS,particles=PARTICLES_DEFAULT,
            B=B_DEFAULT,chunk_size=2048,include_asset_scores=True):
    """Full run with frozen 5m Student-t kernels, seven horizons, paired day bootstrap.

    No per-origin predictions exported. Scores/uncertainty/per-asset diagnostics only.
    """
    f5=vd14._get_split(frame,5)
    train,valid=f5[:2]
    markov=vd14.fit_markov(frame.loc[(frame.split=='train')&frame.ready])
    hazard=vd14.fit_onebar_hazard(frame,markov)
    kernels,fitdiag=fit_return_kernel(train,valid,markov)
    out=[];boot=[];by_asset=[];detail={}
    for horizon in horizons:
        s,te,p,m=_score_horizon(frame,horizon,kernels,markov,hazard,particles,chunk_size)
        out.append(s);detail[str(horizon)]={k:v for k,v in m.items() if k!='per_asset'}
        for asset,models in m['per_asset'].items():
            for model,score in models.items():
                by_asset.append({'horizon_min':horizon,'asset':asset,'model':model,**score})
        for i,(a,b) in enumerate(PAIRS):
            for j,key in enumerate(METRICS):
                diff,lo,hi=_bootstrap(p[a][key],p[b][key],te.date.to_numpy(),B,
                           seed=1501+horizon*100+10*i+j)
                boot.append({'horizon_min':horizon,'comparison':f'{a} - {b}',
                     'metric':key,'delta':diff,'q025':lo,'q975':hi,
                     'test_days':int(s.n_days.min()),'origins':int(s.n_origins.min())})
        del p,te,s
        gc.collect()
    scores=pd.concat(out,ignore_index=True)
    bs=pd.DataFrame(boot)
    per_asset=pd.DataFrame(by_asset)
    parameters={'model_kernels':kernels,'return_fit_diagnostics':fitdiag,
                'volatility_markov_states':markov,'reference_am_hazard':hazard,
                'horizon_meta':detail,'particles':particles}
    return scores,bs,per_asset,parameters


def numerical_sensitivity(frame:pd.DataFrame,kernels:dict,transitions:dict,*,
                          horizon=PRIMARY,particles=PARTICLES_DEFAULT,
                          max_origins_per_asset=512):
    """Fixed VALID-only QMC precision check (not a hyperparameter decision).

    Compare N and 2N nested scrambled-Sobol integration nodes on a sparse,
    chronologically spaced subset from VALID, with no inspection of TEST.
    """
    tr,va,_=vd14._get_split(frame,horizon)
    threshold={a:float(np.quantile(np.abs(g[f'R_{horizon}']),vd14.TAIL_Q))
               for a,g in tr.groupby('underlying')}
    res=[]
    for asset,g in va.groupby('underlying',sort=True):
        if len(g)>max_origins_per_asset:
            idx=np.linspace(0,len(g)-1,max_origins_per_asset,dtype=int)
            g=g.iloc[idx]
        base=g[['slow','gap']].to_numpy(float)
        previous=g.return_5m.to_numpy(float)
        actual=g[f'R_{horizon}'].to_numpy(float)
        q=np.full(len(g),threshold[asset])
        seed=15000+int(np.sum(np.frombuffer(asset.encode(),dtype=np.uint8)))+(horizon//5)*100
        for model in ('AM_T','AM_T_AR'):
            par=kernels[model][asset]
            common=dict(ar=par['phi'],scale=par['scale'],df=par['nu'],
              steps=horizon//5,seed=seed,variable_state=True,
              q=q,realized=actual)
            low=_integrate_student(base,previous,transitions[asset],
                    particles=particles,**common)
            high=_integrate_student(base,previous,transitions[asset],
                    particles=2*particles,**common)
            res.append({'asset':asset,'model':model,'horizon_min':horizon,
                 'origins':len(g),'particles_low':particles,'particles_high':2*particles,
                 'nll_delta_high_minus_low':float(np.mean(high['nll']-low['nll'])),
                 'nll_mean_absolute_difference':float(np.mean(np.abs(high['nll']-low['nll']))),
                 'tail_mean_absolute_difference':float(np.mean(np.abs(high['tail_probability']-low['tail_probability'])))})
    return pd.DataFrame(res)


def decision(scores:pd.DataFrame,bs:pd.DataFrame):
    primary=bs[(bs.horizon_min==PRIMARY)&(bs.metric=='nll')]
    def improves(a,b):
        r=primary[primary.comparison==f'{a} - {b}']
        return bool(len(r)==1 and r.iloc[0].delta<0 and r.iloc[0].q975<0)
    n=int(scores[scores.horizon_min==PRIMARY].n_days.min())
    return {'status':'INSUFFICIENT_DATA' if n<MIN_TEST_DAYS else 'HISTORICAL_EXPLORATORY',
      'n_test_days':n,'primary_horizon_min':PRIMARY,
      'flags':{'AM_T_beats_AM_G':improves('AM_T','AM_G'),
               'AM_T_beats_M2':improves('AM_T','M2_student_t'),
               'AM_T_AR_beats_AM_T':improves('AM_T_AR','AM_T'),
               'AM_T_AR_beats_M2':improves('AM_T_AR','M2_student_t'),
               'T_MC_beats_T_RW':improves('T_MC','T_RW')},
      'strong_candidate':improves('AM_T','AM_G') and improves('AM_T','M2_student_t'),
      'historical_confirmation_permitted':False,
      'note':'Tail calibration, particle sensitivity, per-asset stability and prospective sample required.'}
