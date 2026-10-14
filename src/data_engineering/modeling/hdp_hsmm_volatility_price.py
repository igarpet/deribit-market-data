"""RSD02: discover volatility states; causally test price dynamics CONDITIONAL on them.

Research-only. HDP-HSMM inference reuses the finite weak-limit sampler from RSD01.
The state emissions contain no signed return, sign persistence or price-path
features; volatility summaries become observable only AFTER completed 30m blocks.
All return predictions condition on filtered *pre-block* state probabilities.
The output is an exploratory diagnostic, not evidence of a trading signal.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import t
from numba import njit
from data_engineering.modeling import hdp_hsmm_return_states as rsd

VERSION='RSD02-0.1.0'
VOL_FEATURES=('log_predictive_variance','log_RV_surprise','log_RV_fast_slow_gap')

@dataclass(frozen=True)
class Registration:
    version:str=VERSION
    horizon_min:int=30
    primary_assets:tuple=('BTC','ETH')
    confirm_assets:tuple=('SOL','XRP')
    kcap:int=12
    age_cap_blocks:int=96
    hsmm_chains:tuple=(1801,2909)
    n_iter:int=140
    burn:int=80
    thin:int=4
    posterior_draws_per_chain:int=3
    volatility_memory_blocks:int=8
    df:float=6.0
    n_bootstrap:int=1000
    n_sign_null:int=40
    status:str='HISTORICAL_EXPLORATORY'
    note:str='Sign-free volatility emission; filtered pre-block probabilities; split after original VD18 train/valid.'

def registration():return asdict(Registration())


def build_volatility_features(blocks:pd.DataFrame, *, floor:float|None=None, history_blocks:int=8):
    """Create observed-volatility emission at END of the block.

    Input data must include ex-ante forecast pred_var at block ORIGIN and
    completed forward rv30= sum six next 5m squared returns.
    RV floor must be determined from training blocks and passed unchanged to
    every later period. No 30m block overlaps another and no missing gap is crossed.
    """
    if not {'state_time','pred_var','rv30'}.issubset(blocks.columns):
        raise ValueError('missing volatility columns')
    g=blocks.sort_values('state_time').copy().reset_index(drop=True)
    if g.state_time.duplicated().any():raise ValueError('duplicate 30m origin')
    v=g.pred_var.to_numpy(float);rv=g.rv30.to_numpy(float)
    if not (np.isfinite(v).all() and (v>0).all() and np.isfinite(rv).all() and (rv>=0).all()):
        raise ValueError('nonfinite or invalid variance')
    if floor is None:
        positive=rv[rv>0]
        floor=float(max(np.median(positive)*.001 if len(positive) else 1e-12,1e-16))
    if not np.isfinite(floor) or floor<=0:raise ValueError('bad floor')
    # The rolling average sees only completed blocks and is restarted on gaps.
    fresh=(pd.to_datetime(g.state_time,utc=True).diff()!=pd.Timedelta(minutes=30)).to_numpy()
    group=np.cumsum(fresh)
    logrv=np.log(np.maximum(rv,floor))
    fastslow=np.full(len(g),np.nan)
    for _,ix in pd.Series(np.arange(len(g))).groupby(group):
        idx=ix.to_numpy(dtype=int)
        if len(idx)<history_blocks:continue
        denom=pd.Series(rv[idx]).rolling(history_blocks,min_periods=history_blocks).mean().to_numpy()
        fastslow[idx]=np.log(np.maximum(rv[idx],floor)/np.maximum(denom,floor))
    g['vol_1']=np.log(v)
    g['vol_2']=logrv-np.log(v)
    g['vol_3']=fastslow
    g['vol_ready']=np.isfinite(fastslow)
    return g,float(floor)


def attach_realized_variance(frame:pd.DataFrame, blocks:pd.DataFrame):
    """Reconstruct strictly future six five-minute returns for each forecast origin.

    The realized RV observation is only emitted after the six returns complete;
    the ex-ante forecast was computed BEFORE those returns. No price-file mutation.
    """
    ret=frame.set_index(['underlying','state_time'])['return_5m']
    b=blocks.copy()
    arrays=[]
    for j in range(1,7):
        ix=pd.MultiIndex.from_arrays([b.underlying.to_numpy(),pd.to_datetime(b.state_time,utc=True)+pd.Timedelta(minutes=5*j)])
        arrays.append(ret.reindex(ix).to_numpy(float))
    r=np.column_stack(arrays)
    if not np.isfinite(r).all():raise ValueError('incomplete future five-minute block')
    if not np.allclose(r.sum(axis=1),b.R_30.to_numpy(float),atol=1e-9,rtol=1e-8):
        raise AssertionError('30m target not equal to six native returns')
    b['rv30']=(r*r).sum(axis=1)
    b['return_surprise']=b.R_30.to_numpy(float)/np.sqrt(b.pred_var.to_numpy(float))
    return b


@njit(cache=True)
def _causal_age_filter(log_emit,pi,beta,haz):
    """Returns state probabilities BEFORE each block's emission is observed.

    At t=0, probabilities are the prior beta; after observing block t, the
    filter updates and propagates episode durations before forecasting t+1.
    """
    T,K=log_emit.shape;D=haz.shape[1]
    age=np.zeros((K,D));age[:,0]=beta
    out=np.zeros((T,K))
    for tt in range(T):
        if tt:
            new=np.zeros((K,D));depart=np.zeros(K)
            for k in range(K):
                for a in range(D):
                    mass=age[k,a];depart[k]+=mass*haz[k,a]
                    new[k,min(a+1,D-1)]+=mass*(1-haz[k,a])
            for k in range(K):
                for j in range(K):
                    if j!=k:new[j,0]+=depart[k]*pi[k,j]
            age=new
        total=np.sum(age)
        if total<=0:raise ValueError('invalid state filter')
        age/=total
        for k in range(K):out[tt,k]=np.sum(age[k])
        # Normalized update using log-emission shift protects underflow.
        top=-1e300
        for k in range(K):
            if log_emit[tt,k]>top:top=log_emit[tt,k]
        for k in range(K):age[k,:]*=np.exp(log_emit[tt,k]-top)
        z=np.sum(age)
        if z<=0:raise ValueError('zero emission')
        age/=z
    return out


def filtered_weights(x,posterior,draw,segments=None):
    spec=posterior['spec'];x=np.asarray(x,float)
    boundaries=np.asarray([0,len(x)] if segments is None else segments,dtype=int)
    if boundaries[0]!=0 or boundaries[-1]!=len(x) or np.any(np.diff(boundaries)<=0):
        raise ValueError('invalid contiguous segment boundaries')
    ll=rsd._student_log_emissions(x,draw['mu'],draw['cov'],spec['df'])
    haz=rsd._duration_hazards(draw['duration_p'],spec['dmax'],spec['duration_shape'])
    return np.vstack([_causal_age_filter(ll[lo:hi],draw['pi'],draw['beta'],haz)
                      for lo,hi in zip(boundaries[:-1],boundaries[1:])])


def _previous_known(blocks):
    """At origin t, previous completed-block return and volatility are known."""
    time=pd.to_datetime(blocks.state_time,utc=True)
    consecutive=time.diff()==pd.Timedelta(minutes=30)
    y=blocks.return_surprise.to_numpy(float)
    prev=np.r_[0.,y[:-1]]
    prev[~consecutive.to_numpy()]=0
    prev=np.clip(prev,-8.,8.)
    return prev


def design_price(blocks,vol_normalizer):
    """Continuous-volatility controls measured BEFORE return being scored.

    Uses previous completed volatility emission, never current future RV.
    Current block's ex-ante log predicted variance is allowed.
    """
    g=blocks
    time=pd.to_datetime(g.state_time,utc=True)
    consecutive=time.diff()==pd.Timedelta(minutes=30)
    current_logv=(np.log(g.pred_var.to_numpy(float)) - vol_normalizer['center'][0])/vol_normalizer['scale'][0]
    volprev=rsd.apply_normalizer(g[['vol_1','vol_2','vol_3']].to_numpy(float),vol_normalizer)
    pv=np.vstack([np.zeros((1,3)),volprev[:-1]])
    pv[~consecutive.to_numpy()]=0
    py=_previous_known(g)
    x=np.column_stack([np.ones(len(g)),py,np.clip(current_logv,-6,6),
                       np.clip(pv[:,1],-6,6),np.clip(pv[:,2],-6,6),
                       py*np.clip(pv[:,1],-3,3),py*np.clip(pv[:,2],-3,3)])
    z=np.column_stack([np.ones(len(g)),np.clip(current_logv,-6,6),
                       np.clip(pv[:,1],-6,6),np.clip(pv[:,2],-6,6)])
    return x,z


def fit_continuous_t(y,X,Z,df=6.0,penalty=.015):
    """Train-only, ridge-regularized heteroskedastic Student-t NLL with exact gradient."""
    y=np.asarray(y,float);X=np.asarray(X,float);Z=np.asarray(Z,float)
    if not (len(y)==len(X)==len(Z)):raise ValueError('length mismatch')
    n,m=X.shape;d=Z.shape[1]
    theta0=np.zeros(m+d);theta0[m]=np.log(max(np.std(y),.2)*np.sqrt((df-2)/df))
    def objective(theta):
        b=theta[:m];gamma=theta[m:];loc=X@b
        raw=Z@gamma;ls=np.clip(raw,-3.,2.5);sc=np.exp(ls)
        e=y-loc;d2=df*sc**2+e**2
        nll=-np.mean(t.logpdf(y,df,loc=loc,scale=sc))
        gb=-(X.T@((df+1)*e/d2))/n
        gl=-1+(df+1)*e**2/d2
        gg=-(Z.T@(gl*((raw>-3)&(raw<2.5))))/n
        reg=penalty*(np.sum(b[1:]**2)+np.sum(gamma[1:]**2))
        gb[1:]+=2*penalty*b[1:];gg[1:]+=2*penalty*gamma[1:]
        return nll+reg,np.r_[gb,gg]
    res=minimize(objective,theta0,jac=True,method='L-BFGS-B',options={'maxiter':180})
    if not res.success and not np.isfinite(res.fun):raise RuntimeError('continuous t fit failed')
    return {'mean_beta':res.x[:m], 'logscale_beta':res.x[m:], 'df':df,'converged':res.success}


def continuous_density(y,X,Z,fit):
    loc=X@fit['mean_beta'];sc=np.exp(np.clip(Z@fit['logscale_beta'],-3,2.5));df=fit['df']
    ld=t.logpdf(y,df,loc=loc,scale=sc)
    prob=t.sf(0,df,loc=loc,scale=sc)
    return ld,prob


def fit_state_t(y,prev_y,weights,df=6.,shrink=100.):
    """State-specific mean, return-AR slope, residual scale; train-only shrinkage.

    Fits a weighted regression with pseudo-count shrink toward the pooled fit.
    The weights are CAUSAL predictive state probabilities, not smoothing labels.
    State scale is deliberately frozen to its training residual estimate.
    """
    y=np.asarray(y,float);prev_y=np.asarray(prev_y,float)
    w=np.asarray(weights,float);N,K=w.shape
    if len(y)!=N or len(prev_y)!=N:raise ValueError('bad dimensions')
    if not np.allclose(w.sum(axis=1),1,atol=1e-7):raise AssertionError('probabilities not normalized')
    X=np.column_stack([np.ones(N),prev_y])
    pooled=np.linalg.solve(X.T@X+np.diag([1e-3,8.]),X.T@y)
    basevar=float(np.mean((y-X@pooled)**2));basevar=max(basevar,.02)
    betas=np.zeros((K,2));vars=np.zeros(K);masses=w.sum(axis=0)
    for k in range(K):
        wk=w[:,k]
        # Shrink both coefficients; stronger penalty for return momentum than drift.
        regular=np.diag([shrink,3*shrink])
        betas[k]=np.linalg.solve((X.T*wk)@X+regular,(X.T*wk)@y+regular@pooled)
        res=(y-X@betas[k])**2
        vars[k]=((wk@res)+shrink*basevar)/(masses[k]+shrink)
    sc=np.sqrt(np.maximum(vars,.02)*(df-2)/df)
    return {'beta':betas,'scale':sc,'mass':masses,'df':df,'pooled':pooled,'base_var':basevar}


def state_density(y,prev_y,weights,fit,*,scale_only=False):
    y=np.asarray(y,float);prev_y=np.asarray(prev_y,float);w=np.asarray(weights,float)
    loc=fit['beta'][:,0][None,:]+prev_y[:,None]*fit['beta'][:,1][None,:]
    if scale_only:loc=np.broadcast_to(fit['pooled'][0]+prev_y[:,None]*fit['pooled'][1],loc.shape)
    sc=fit['scale'][None,:];df=fit['df']
    logmix=logsumexp(np.log(np.maximum(w,1e-300))+t.logpdf(y[:,None],df,loc=loc,scale=sc),axis=1)
    prob=(w*t.sf(0,df,loc=loc,scale=sc)).sum(axis=1)
    return logmix,prob


def mixture_predictions(train,eval_data,fit_bounds,eval_bounds,normalizer,chains,draws_per_chain=3):
    """Fit train-only state→return mapping, predict evaluated blocks causally.

    Per-posterior-draw models are combined as DENSITIES, never merged by state
    label; this also avoids illegal alignment across independent chains.
    """
    xtr=rsd.apply_normalizer(train[['vol_1','vol_2','vol_3']].to_numpy(float),normalizer)
    xev=rsd.apply_normalizer(eval_data[['vol_1','vol_2','vol_3']].to_numpy(float),normalizer)
    ytr=train.return_surprise.to_numpy(float);yev=eval_data.return_surprise.to_numpy(float)
    pytr=_previous_known(train);pyev=_previous_known(eval_data)
    emissions=[];scaleonly=[];sign=[];profile=[]
    for ci,post in enumerate(chains):
        ds=post['draws'];ix=np.unique(np.linspace(0,len(ds)-1,min(draws_per_chain,len(ds)),dtype=int))
        for di in ix:
            draw=ds[int(di)]
            weights_train=filtered_weights(xtr,post,draw,fit_bounds)
            weights_eval=filtered_weights(xev,post,draw,eval_bounds)
            fit=fit_state_t(ytr,pytr,weights_train,df=post['spec']['df'])
            ll,pr=state_density(yev,pyev,weights_eval,fit)
            ll_scale,pr_scale=state_density(yev,pyev,weights_eval,fit,scale_only=True)
            emissions.append(ll);scaleonly.append(ll_scale);sign.append(pr)
            profile.append({'chain':ci+1,'draw':int(di),'n_strong_states':int(np.sum(fit['mass']>=100)),
                            'state_means':fit['beta'][:,0].tolist(),
                            'state_AR':fit['beta'][:,1].tolist(),
                            'state_weight':(fit['mass']/len(ytr)).tolist(),
                            'state_scale':fit['scale'].tolist()})
    return {'state_logdensity':logsumexp(emissions,axis=0)-np.log(len(emissions)),
            'scaleonly_logdensity':logsumexp(scaleonly,axis=0)-np.log(len(scaleonly)),
            'state_sign_probability':np.mean(np.asarray(sign),axis=0),
            'profiles':profile}


def matched_metrics(eval_data,mixture,continuous_log,continuous_sign,single_log,single_sign):
    v=eval_data.pred_var.to_numpy(float);y=eval_data.return_surprise.to_numpy(float)
    rawjac=.5*np.log(v)
    truth=(y>0).astype(float)
    eps=1e-10
    continuous_sign=np.clip(continuous_sign,eps,1-eps)
    single_sign=np.clip(single_sign,eps,1-eps)
    state_sign=np.clip(mixture['state_sign_probability'],eps,1-eps)
    result=eval_data[['state_time']].copy()
    result['date']=pd.to_datetime(result.state_time,utc=True).dt.floor('D')
    result['AM_T_HAR_Blend_nll']=eval_data.baseline_nll.to_numpy(float)
    for nm,lp in [('HDP_VOL_STATE',mixture['state_logdensity']),
                  ('VOL_STATE_SCALE_ONLY',mixture['scaleonly_logdensity']),
                  ('CONTINUOUS_T',continuous_log),('RECENT_SINGLE_T',single_log)]:
        result[nm+'_nll']=-lp+rawjac
    for nm,p in [('HDP_VOL_STATE',state_sign),('CONTINUOUS_T',continuous_sign),('RECENT_SINGLE_T',single_sign)]:
        result[nm+'_sign_brier']=(p-truth)**2
        result[nm+'_sign_logloss']=-(truth*np.log(p)+(1-truth)*np.log1p(-p))
    result['HALF_SIGN_NULL_sign_brier']=(.5-truth)**2
    result['direction']=truth
    result['return_surprise']=y
    return result


def day_bootstrap(df, pairs, *, B=1000,seed=2026):
    rng=np.random.default_rng(seed);days=df.date.drop_duplicates().sort_values().to_numpy()
    rows=[]
    for a,b in pairs:
        diff=(df[a]-df[b]).groupby(df.date).mean().reindex(days).to_numpy(float)
        pool=rng.integers(0,len(days),size=(B,len(days)))
        draws=diff[pool].mean(axis=1)
        rows.append({'a':a,'b':b,'delta':float(diff.mean()),'ci025':float(np.quantile(draws,.025)),
                     'ci975':float(np.quantile(draws,.975)),'days':len(days)})
    return pd.DataFrame(rows)


def run_asset(train,eval_data,chains,normalizer,fit_bounds,eval_bounds,draws_per_chain=3,B=1000):
    xtr,ztr=design_price(train,normalizer);xev,zev=design_price(eval_data,normalizer)
    ytr=train.return_surprise.to_numpy(float);yev=eval_data.return_surprise.to_numpy(float)
    continuous=fit_continuous_t(ytr,xtr,ztr)
    contll,contprob=continuous_density(yev,xev,zev,continuous)
    # A single Student-t + recent-return autoregression is a further non-switching control.
    nullX=xtr[:,:2];nullEval=xev[:,:2]
    tnull=fit_continuous_t(ytr,nullX,np.ones((len(train),1)),penalty=.015)
    nullll,nullprob=continuous_density(yev,nullEval,np.ones((len(eval_data),1)),tnull)
    mix=mixture_predictions(train,eval_data,fit_bounds,eval_bounds,normalizer,chains,draws_per_chain)
    result=matched_metrics(eval_data,mix,contll,contprob,nullll,nullprob)
    comps=[('HDP_VOL_STATE_nll','AM_T_HAR_Blend_nll'),
           ('HDP_VOL_STATE_nll','CONTINUOUS_T_nll'),
           ('HDP_VOL_STATE_nll','RECENT_SINGLE_T_nll'),
           ('HDP_VOL_STATE_nll','VOL_STATE_SCALE_ONLY_nll'),
           ('HDP_VOL_STATE_sign_brier','CONTINUOUS_T_sign_brier'),
           ('HDP_VOL_STATE_sign_logloss','CONTINUOUS_T_sign_logloss')]
    bs=day_bootstrap(result,comps,B=B)
    score={k:float(result[k].mean()) for k in result.columns if k.endswith(('nll','sign_brier','sign_logloss'))}
    return score,bs,mix['profiles'],result


def sign_randomization(train,eval_data,chains,normalizer,fit_bounds,eval_bounds,*,B=40,seed=2601,draws_per_chain=3):
    """Paired block-sign flips; preserves observed volatility emissions EXACTLY.

    This null only addresses directional dependence, not whether *volatility
    states* are discrete. Fixed HSMM states, re-fit return mappings and a
    continuous Student-t comparator on sign-flipped TRAIN, score sign-flipped
    EVALUATION. No sign or direction is used to infer volatility states.
    """
    rng=np.random.default_rng(seed)
    xtr=rsd.apply_normalizer(train[['vol_1','vol_2','vol_3']].to_numpy(float),normalizer)
    xev=rsd.apply_normalizer(eval_data[['vol_1','vol_2','vol_3']].to_numpy(float),normalizer)
    cached=[]
    for post in chains:
        ds=post['draws'];ix=np.unique(np.linspace(0,len(ds)-1,min(draws_per_chain,len(ds)),dtype=int))
        for di in ix:
            draw=ds[int(di)]
            cached.append((filtered_weights(xtr,post,draw,fit_bounds),
                           filtered_weights(xev,post,draw,eval_bounds)))
    rows=[]
    for b in range(B):
        # One random sign per WHOLE 30m block leaves sum(r_5m²) unchanged.
        tr=train.copy();ev=eval_data.copy()
        tr['return_surprise']=tr.return_surprise.to_numpy()*rng.choice([-1.,1.],size=len(tr))
        ev['return_surprise']=ev.return_surprise.to_numpy()*rng.choice([-1.,1.],size=len(ev))
        ytr=tr.return_surprise.to_numpy(float);yev=ev.return_surprise.to_numpy(float)
        p_tr=_previous_known(tr);p_ev=_previous_known(ev)
        xc,zc=design_price(tr,normalizer);xe,ze=design_price(ev,normalizer)
        cont=fit_continuous_t(ytr,xc,zc)
        _,p_cont=continuous_density(yev,xe,ze,cont)
        preds=[]
        for wt,we in cached:
            statefit=fit_state_t(ytr,p_tr,wt)
            _,q=state_density(yev,p_ev,we,statefit)
            preds.append(q)
        p_state=np.mean(preds,axis=0)
        truth=(yev>0).astype(float)
        brier_state=np.mean((p_state-truth)**2)
        brier_cont=np.mean((p_cont-truth)**2)
        rows.append({'seed_iteration':b,'state_minus_cont_sign_brier':float(brier_state-brier_cont)})
    return pd.DataFrame(rows)


def continuous_regime_free_null(n:int,seed=8001,df=6.,phi=.94):
    """Continuous-vol, heavy-tailed *illustrative* null; no latent regime.

    Simulates a log-variance AR(1) with Student-t five-minute innovations.
    This isn't a full recursively refitted VD18 process; conclusions remain
    conditional on the limitations described in the preregistration.
    """
    rng=np.random.default_rng(seed);u=np.zeros(n)
    for k in range(1,n):u[k]=phi*u[k-1]+.22*rng.normal()
    pred=np.exp(-11+u)
    returns=rng.standard_t(df,size=(n,6))*np.sqrt((df-2)/df)*np.sqrt(pred[:,None]/6)
    times=pd.date_range('2026-01-01',periods=n,freq='30min',tz='UTC')
    return pd.DataFrame({'state_time':times,'pred_var':pred,'rv30':(returns**2).sum(axis=1),
                         'return_surprise':returns.sum(axis=1)/np.sqrt(pred)})


def fixed_variance_regime_free_null(observed_blocks:pd.DataFrame, *,seed=5101,df=6.):
    """Retain actual causal predicted-variance path, simulate iid t increments.

    NO state process is simulated. This does not recurse variance through HAR;
    it is a conditional null, not a perfect generative VD18 counterfactual.
    """
    b=observed_blocks.copy()
    pred=b.pred_var.to_numpy(float)
    if not (np.isfinite(pred).all() and (pred>0).all()):raise ValueError('bad variance')
    rng=np.random.default_rng(seed)
    inc=rng.standard_t(df,size=(len(b),6))*np.sqrt((df-2)/df)*np.sqrt(pred[:,None]/6)
    b['rv30']=(inc**2).sum(axis=1)
    b['return_surprise']=inc.sum(axis=1)/np.sqrt(pred)
    return b


def future_price_profiles(eval_data,posterior,normalizer,segments=None,*,chain=1):
    """Descriptive *future* return outcomes grouped by strictly PRE-block state weights.

    Internal state IDs are never aligned across draws or chains. Evaluated
    outcomes are not used to refit the HSMM/response model. The reported
    state-conditional outcomes are retrospective exploratory descriptions.
    """
    draw=posterior['draws'][len(posterior['draws'])//2]
    x=rsd.apply_normalizer(eval_data[['vol_1','vol_2','vol_3']].to_numpy(float),normalizer)
    w=filtered_weights(x,posterior,draw,segments)
    y=eval_data.return_surprise.to_numpy(float)
    prev=_previous_known(eval_data)
    next_pos=(y>0).astype(float)
    cont=(np.sign(y)==np.sign(prev)) & (prev!=0) & (y!=0)
    valid=(prev!=0)&(y!=0)
    eff=np.tanh(2*eval_data['feature_2'].to_numpy(float)) if 'feature_2' in eval_data else np.full(len(y),np.nan)
    dep=np.tanh(eval_data['feature_3'].to_numpy(float)) if 'feature_3' in eval_data else np.full(len(y),np.nan)
    rows=[]
    for k in range(w.shape[1]):
        wt=w[:,k];mass=wt.sum()
        if mass<=1e-5:continue
        denom=np.dot(wt,valid)
        rows.append({'chain':chain,'state_internal_id':int(k),'effective_future_blocks':float(mass),
                     'mean_next_standardized_return':float(np.dot(wt,y)/mass),
                     'mean_abs_next_standardized_return':float(np.dot(wt,np.abs(y))/mass),
                     'next_return_variance':float(np.dot(wt,(y-np.dot(wt,y)/mass)**2)/mass),
                     'next_positive_probability':float(np.dot(wt,next_pos)/mass),
                     'next_sign_continuation':float(np.dot(wt,cont)/denom) if denom>5 else np.nan,
                     'next_abs_path_efficiency':float(np.dot(wt,np.abs(eff))/mass) if np.isfinite(eff).all() else np.nan,
                     'next_within_block_dependence':float(np.dot(wt,dep)/mass) if np.isfinite(dep).all() else np.nan})
    return pd.DataFrame(rows)
