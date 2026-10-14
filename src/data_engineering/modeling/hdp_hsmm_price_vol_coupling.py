"""RSD03: joint price/volatility coupling states, Bayesian HSMM + predictive controls.

Research-only. 30m COMPLETED nonoverlapping blocks, current block available only
for the next block's filter state. HAR log-RV estimate and net-return variance
are distinct forecasts. Truncated HDP state count is *not* an economic regime count.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import t
from data_engineering.modeling import hdp_hsmm_return_states as hsmm
from data_engineering.modeling import hdp_hsmm_volatility_price as vs

VERSION='RSD03-0.1.0'
JOINT=('price_z','vol_surprise_u')

@dataclass(frozen=True)
class Registration:
    version:str=VERSION
    primary_assets:tuple=('BTC','ETH')
    secondary_assets:tuple=('SOL','XRP')
    horizon_min:int=30
    kcap:int=16
    dmax:int=96
    seeds:tuple=(1801,2909)
    n_iter:int=160
    burn:int=80
    thin:int=4
    posterior_draws_per_chain:int=4
    boot:int=1000
    sign_permutations:int=100
    null_seeds:tuple=(311,719)
    status:str='HISTORICAL_EXPLORATORY'

def registration(): return asdict(Registration())


def har_log_rv_scale(g:pd.DataFrame,fit:dict):
    """TRAIN-fitted HAR *log realized variance* scale, NOT net-return variance.

    Module18 predict_har returns TRAIN-calibrated net-return variance.  Undo
    its known per-asset net-return scale to recover exp(E[log RV | features]).
    This is a log-RV scale/conditional median proxy, not an unbiased E[RV].
    """
    from data_engineering.modeling import markov_har_variance as v18
    scale=g.underlying.map(fit['net_return_scales']).to_numpy(float)
    if not np.isfinite(scale).all() or np.any(scale<=0):raise ValueError('invalid HAR net-return scales')
    v=np.asarray(v18.predict_har(g,fit),float)/scale
    if not np.all(np.isfinite(v)) or np.any(v<=0):raise ValueError('invalid HAR log-RV prediction')
    return v


def joint_features(blocks:pd.DataFrame, *, rv_floor:float,clip_u:float=9.):
    """Forecasts known at origin; realized R and RV known only at block END."""
    needed={'state_time','pred_var','pred_rv_scale','R_30','rv30'}
    if not needed.issubset(blocks):raise ValueError(f'missing {needed-set(blocks)}')
    g=blocks.sort_values('state_time').copy().reset_index(drop=True)
    if g.state_time.duplicated().any():raise ValueError('duplicate block starts')
    for field in ('pred_var','pred_rv_scale'):
        if not np.isfinite(g[field]).all() or (g[field]<=0).any():raise ValueError(f'invalid {field}')
    if not np.isfinite(g[['R_30','rv30']].to_numpy()).all() or (g.rv30<0).any():raise ValueError('invalid return/realized RV')
    if not np.isfinite(rv_floor) or rv_floor<=0:raise ValueError('invalid train floor')
    g['joint_1']=g.R_30.to_numpy(float)/np.sqrt(g.pred_var.to_numpy(float))
    g['joint_2']=np.clip(np.log(np.maximum(g.rv30.to_numpy(float),rv_floor)/
                                        g.pred_rv_scale.to_numpy(float)),-clip_u,clip_u)
    g['coupling']=g.joint_1*g.joint_2
    return g


def _past(g:pd.DataFrame,field:str):
    a=g[field].to_numpy(float)
    prev=np.r_[0.,a[:-1]]
    fresh=(pd.to_datetime(g.state_time,utc=True).diff()==pd.Timedelta(minutes=30)).to_numpy()
    prev[~fresh]=0
    return prev


def fit_joint_normalizer(train:pd.DataFrame):
    """Train-only centering of joint emissions and ex-ante log variance."""
    n=hsmm.fit_normalizer(train[['joint_1','joint_2']].to_numpy(float))
    logv=np.log(train.pred_var.to_numpy(float))
    n['logv_center']=float(np.median(logv))
    n['logv_scale']=float(max((np.quantile(logv,.75)-np.quantile(logv,.25))/1.349,1e-8))
    return n


def continuous_design(g:pd.DataFrame,normalizer:dict):
    """No same-block future feature on the RHS. Includes continuous interaction."""
    z=_past(g,'joint_1')
    u=_past(g,'joint_2')
    level=(np.log(g.pred_var.to_numpy(float))-normalizer['logv_center'])/normalizer['logv_scale']
    z=np.clip(z,-8,8);u=np.clip(u,-8,8);level=np.clip(level,-6,6)
    X=np.column_stack([np.ones(len(g)),z,u,np.clip(z*u,-12,12),level])
    Z=np.column_stack([np.ones(len(g)),u,level,np.clip(np.abs(z),0,8)])
    return X,Z


def _density_average(logs):
    a=np.asarray(logs,float)
    return logsumexp(a,axis=0)-np.log(len(a))


def joint_log_scores(train:pd.DataFrame,test:pd.DataFrame,normalizer,train_bounds,test_bounds,chains,draws_per_chain=4):
    """Compare temporal HSMM joint-feature score to order-ablated same emissions.

    An order-ablated static mixture is NOT independently refitted; it isolates
    temporal prediction with identical emission parameters, not a full iid-DPM.
    """
    xtr=hsmm.apply_normalizer(train[list(('joint_1','joint_2'))].to_numpy(float),normalizer)
    xte=hsmm.apply_normalizer(test[list(('joint_1','joint_2'))].to_numpy(float),normalizer)
    tf=hsmm.fit_null_t(xtr,df=6.)
    single=hsmm.null_logscore(xte,tf)
    time_ll=[];static_ll=[]
    for post in chains:
        draws=post['draws'];ix=np.unique(np.linspace(0,len(draws)-1,min(draws_per_chain,len(draws)),dtype=int))
        for i in ix:
            d=draws[int(i)]
            pseudo={'spec':post['spec'],'draws':[d]}
            time_ll.append(hsmm.predict_logscore(xte,pseudo,limit_draws=1,segments=test_bounds))
            emit=hsmm._student_log_emissions(xte,d['mu'],d['cov'],post['spec']['df'])
            # Explicitly remove history: static predictive weights per posterior draw.
            weight=np.maximum(d['occupancy'].astype(float),1e-3)
            weight/=weight.sum()
            static_ll.append(logsumexp(emit+np.log(weight)[None,:],axis=1))
    return {'HSMM_joint_logscore':_density_average(time_ll),
            'static_emission_mixture_logscore':_density_average(static_ll),
            'single_joint_t_logscore':single}


def conditional_return_predictions(train:pd.DataFrame,test:pd.DataFrame,normalizer,
                                   train_bounds,test_bounds,chains,draws_per_chain=4):
    """Causal 30m signed-return density using filtered *prior* joint-state mass."""
    xtr=hsmm.apply_normalizer(train[['joint_1','joint_2']].to_numpy(float),normalizer)
    xte=hsmm.apply_normalizer(test[['joint_1','joint_2']].to_numpy(float),normalizer)
    ytr=train.joint_1.to_numpy(float);yte=test.joint_1.to_numpy(float)
    pytr=_past(train,'joint_1');pyte=_past(test,'joint_1')
    ll=[];llscale=[];sign=[];static=[];static_sign=[];profiles=[]
    for chain,post in enumerate(chains,1):
        draws=post['draws'];ix=np.unique(np.linspace(0,len(draws)-1,min(draws_per_chain,len(draws)),dtype=int))
        for drawid in ix:
            d=draws[int(drawid)]
            wtr=vs.filtered_weights(xtr,post,d,train_bounds)
            wte=vs.filtered_weights(xte,post,d,test_bounds)
            fit=vs.fit_state_t(ytr,pytr,wtr,df=post['spec']['df'],shrink=100.)
            pred,pr=vs.state_density(yte,pyte,wte,fit)
            sl,_=vs.state_density(yte,pyte,wte,fit,scale_only=True)
            # Temporal ablation: identical fitted state-conditioned emissions,
            # but no filtered transition information on the test observations.
            static_w=np.broadcast_to(np.maximum(fit['mass'],1e-6)/max(fit['mass'].sum(),1e-9),wte.shape)
            pstatic,psign=vs.state_density(yte,pyte,static_w,fit)
            ll.append(pred);llscale.append(sl);sign.append(pr)
            static.append(pstatic);static_sign.append(psign)
            profiles.append({'chain':chain,'draw':int(drawid),
                             'n_states_mass_gt_100':int(np.sum(fit['mass']>=100)),
                             'state_means':fit['beta'][:,0].tolist(),
                             'state_ar':fit['beta'][:,1].tolist(),
                             'state_scales':fit['scale'].tolist()})
    return {'HSMM_return_log':_density_average(ll),'scale_only_return_log':_density_average(llscale),
            'static_state_mixture_return_log':_density_average(static),
            'HSMM_sign':np.asarray(sign).mean(axis=0),
            'static_state_mixture_sign':np.asarray(static_sign).mean(axis=0),
            'profiles':profiles}


def score_price(train,test,normalizer,train_bounds,test_bounds,chains,draws_per_chain=4,B=1000):
    """All return scores are comparable RAW-RETURN NLL on same 30m origins."""
    predictions=conditional_return_predictions(train,test,normalizer,train_bounds,test_bounds,chains,draws_per_chain)
    joint=joint_log_scores(train,test,normalizer,train_bounds,test_bounds,chains,draws_per_chain)
    Xtr,Ztr=continuous_design(train,normalizer);Xte,Zte=continuous_design(test,normalizer)
    ytr=train.joint_1.to_numpy(float);yte=test.joint_1.to_numpy(float)
    cont=vs.fit_continuous_t(ytr,Xtr,Ztr,df=6.)
    c_log,c_sign=vs.continuous_density(yte,Xte,Zte,cont)
    # More parsimonious recent return Student-t; no joint-volatility interaction.
    fixed=vs.fit_continuous_t(ytr,Xtr[:,:2],np.ones((len(train),1)),df=6.)
    s_log,s_sign=vs.continuous_density(yte,Xte[:,:2],np.ones((len(test),1)),fixed)
    jac=np.log(np.sqrt(test.pred_var.to_numpy(float)))
    d=pd.DataFrame({'state_time':pd.to_datetime(test.state_time,utc=True),
                    'date':pd.to_datetime(test.state_time,utc=True).dt.floor('D'),
                    'AM_T_HAR_Blend_nll':test.baseline_nll.to_numpy(float),
                    'HDP_joint_state_nll':-predictions['HSMM_return_log']+jac,
                    'HDP_scale_only_nll':-predictions['scale_only_return_log']+jac,
                    'order_ablated_state_nll':-predictions['static_state_mixture_return_log']+jac,
                    'continuous_coupling_t_nll':-c_log+jac,
                    'recent_t_nll':-s_log+jac,
                    'HDP_sign_brier':(predictions['HSMM_sign']-(yte>0))**2,
                    'continuous_sign_brier':(c_sign-(yte>0))**2,
                    'HDP_sign_probability':predictions['HSMM_sign'],
                    'continuous_sign_probability':c_sign,
                    'realized_positive':(yte>0).astype(float),
                    'HDP_joint_observation_nll':-joint['HSMM_joint_logscore'],
                    'order_ablated_joint_nll':-joint['static_emission_mixture_logscore'],
                    'single_joint_t_nll':-joint['single_joint_t_logscore']})
    pairs=[('HDP_joint_state_nll','continuous_coupling_t_nll'),
           ('HDP_joint_state_nll','AM_T_HAR_Blend_nll'),
           ('HDP_joint_state_nll','recent_t_nll'),
           ('HDP_joint_state_nll','HDP_scale_only_nll'),
           ('HDP_joint_state_nll','order_ablated_state_nll'),
           ('HDP_joint_observation_nll','order_ablated_joint_nll'),
           ('HDP_joint_observation_nll','single_joint_t_nll'),
           ('HDP_sign_brier','continuous_sign_brier')]
    bs=vs.day_bootstrap(d,pairs,B=B)
    summary={k:float(d[k].mean()) for k in d.columns if k.endswith(('nll','brier'))}
    return summary,bs,d,predictions['profiles']


def volatility_coupling_profiles(test:pd.DataFrame,normalizer,post,segments,*,chain=1,min_mass=30):
    """Future block statistics weighted by state mass inferred before this block."""
    draw=post['draws'][len(post['draws'])//2]
    x=hsmm.apply_normalizer(test[['joint_1','joint_2']].to_numpy(float),normalizer)
    w=vs.filtered_weights(x,post,draw,segments)
    z=test.joint_1.to_numpy(float);u=test.joint_2.to_numpy(float)
    prev=_past(test,'joint_1')
    results=[]
    for k in range(w.shape[1]):
        weight=w[:,k];mass=float(weight.sum())
        if mass<min_mass:continue
        corr_n=np.dot(weight,(z-np.dot(weight,z)/mass)*(u-np.dot(weight,u)/mass))/mass
        zvar=np.dot(weight,(z-np.dot(weight,z)/mass)**2)/mass
        uvar=np.dot(weight,(u-np.dot(weight,u)/mass)**2)/mass
        cc=float(corr_n/max(np.sqrt(zvar*uvar),1e-12))
        valid=(prev!=0)&(z!=0)
        results.append({'chain':chain,'state_id_in_draw':k,'effective_future_blocks':mass,
                        'next_mean_z':float(np.dot(weight,z)/mass),
                        'next_mean_u':float(np.dot(weight,u)/mass),
                        'next_positive_probability':float(np.dot(weight,(z>0))/mass),
                        'next_sign_continuation':float(np.dot(weight,valid*(np.sign(z)==np.sign(prev)))/max(np.dot(weight,valid),1e-9)),
                        'next_z_u_correlation':cc,
                        'next_coupling_mean':float(np.dot(weight,z*u)/mass),
                        'next_abs_z':float(np.dot(weight,np.abs(z))/mass)})
    return pd.DataFrame(results)


def posterior_coupling_state_profiles(post,*,chain=1):
    d=post['draws'][len(post['draws'])//2]
    labels=np.asarray(d['labels']);results=[]
    for k,mass in enumerate(d['occupancy']):
        if mass<=0:continue
        cv=d['cov'][k];scale=np.sqrt(np.diag(cv))
        corr=float(cv[0,1]/max(scale.prod(),1e-12))
        lens=d['episode_lengths'][d['episode_states']==k]
        results.append({'chain':chain,'state_id_in_draw':k,'train_occupancy':int(mass),
                        'emission_z_mean_normalized':float(d['mu'][k,0]),
                        'emission_u_mean_normalized':float(d['mu'][k,1]),
                        'emission_z_u_correlation':corr,
                        'n_episodes':int(len(lens)),
                        'median_episode_30m_blocks':float(np.median(lens)) if len(lens) else np.nan})
    return pd.DataFrame(results)


def conditional_regime_free_control(train:pd.DataFrame,rv_floor:float,seed:int):
    """Regime-free Student-t conditional on OBSERVED ex-ante forecast scales.

    This preserves the nonlinear return-to-RV identity but does not recursively
    refit HAR.  K from this null is descriptive only (not a calibrated p-value).
    """
    rng=np.random.default_rng(seed);n=len(train)
    # Scale 5m innovations so their six-bar integrated variance is pred_rv_scale.
    sd=np.sqrt(train.pred_rv_scale.to_numpy(float)[:,None]/6)
    samples=rng.standard_t(6,size=(n,6))*np.sqrt(4/6)*sd
    generated=train[['state_time','pred_var','pred_rv_scale']].copy()
    generated['R_30']=samples.sum(axis=1)
    generated['rv30']=np.sum(samples*samples,axis=1)
    return joint_features(generated,rv_floor=rv_floor)


def conditional_sign_permutation(price_scores:pd.DataFrame,B=100,seed=601):
    """Randomize ONLY held-out return signs, holding causal prediction fixed.

    Diagnostic test of directional outcome association, NOT a refitted world or
    a full randomization test of the joint HSMM's existence.
    """
    rng=np.random.default_rng(seed)
    prob=price_scores.HDP_sign_probability.to_numpy(float)
    cont=price_scores.continuous_sign_probability.to_numpy(float)
    sign=price_scores.realized_positive.to_numpy(float)
    center=2*sign-1
    vals=[]
    for _ in range(B):
        flipped=(center*rng.choice(np.array([-1.,1.]),size=len(sign))>0).astype(float)
        vals.append(float(np.mean((prob-flipped)**2-(cont-flipped)**2)))
    return np.asarray(vals)
