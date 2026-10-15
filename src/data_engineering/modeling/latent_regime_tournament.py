"""RD02: causal HMM vs explicit-duration semi-Markov overlay vs observable states.

The three-state Student-t-emission HMM is fitted to past/current market features
only (STRUCTURE). A separate CALIBRATION period fits return response parameters.
The duration model is a TWO-STAGE, Viterbi-episode hazard overlay on the fitted
HMM emissions and switch destinations; it is NOT a jointly Bayesian-fitted HSMM.
For reference every method uses the same trailing-RV variance proxy, NOT
point-in-time AM-T-HAR-Blend variance, which is left completely untouched.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy.special import logsumexp, gammaln
from scipy.stats import t as student_t
from scipy.optimize import minimize_scalar, minimize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor, export_text

VERSION='RD02-0.1.0'
HORIZONS=(5,15,30,60)
STATE_FEATURES=('mom_r5','flow5','basis_bps','basis_change_bps','last_1m_r','log_trades')
RETURN_FEATURES=('mom_r5','mom_r15_mean','flow5','flow30','basis_bps',
                 'basis_change_bps','last_1m_r','minute_accel','log_trades')
METHODS=('direct_symmetric','direct_continuous_skew','observable_tree_skew',
         'latent_static','latent_hmm','latent_semi_markov')

@dataclass(frozen=True)
class Spec:
    version:str=VERSION
    assets:tuple[str,...]=('BTC','ETH','SOL','XRP')
    horizons:tuple[int,...]=HORIZONS
    primary_horizon_min:int=30
    anchor_minutes:int=30
    k_states:int=3
    emission_df:float=6.
    student_return_df:float=6.
    em_iterations:int=10
    covariance_floor:float=.05
    transition_pseudocount:float=8.
    duration_age_cap:int=24  # 24 x 30m = 12h, tail hazard held constant
    duration_prior_strength:float=80.
    response_shrinkage:float=200.
    skew_shrinkage:float=250.
    max_skew:float=.45
    ridge_alpha:float=100.
    discovery_fraction:float=.80
    fold_positions:tuple[float,...]=(.50,.64,.78)
    validation_days_fraction:float=.08
    embargo_days:int=1
    structure_fraction:float=.70
    tree_max_depth:int=3
    tree_leaf_fraction:float=.08
    bootstrap_reps:int=300
    seed:int=2612
    status:str='HISTORICAL_DISCOVERY_NOT_CONFIRMATION'

def registration():return asdict(Spec())

def combinations_dates(days,cfg:Spec):
    days=pd.DatetimeIndex(sorted(pd.to_datetime(days,utc=True)))
    disc=days[:int(len(days)*cfg.discovery_fraction)]
    if len(disc)<55:raise ValueError('Fewer than 55 discovery days')
    width=max(5,int(len(disc)*cfg.validation_days_fraction))
    out=[]
    for index,f in enumerate(cfg.fold_positions,1):
        s=int(len(disc)*f); stop=s-cfg.embargo_days;cut=max(12,int(stop*cfg.structure_fraction))
        if cut>=stop-8 or stop<25:raise ValueError('Training/calibration folds too short')
        out.append({'fold':index,'structure_days':disc[:cut],
                    'calibration_days':disc[cut:stop],
                    'validation_days':disc[s:min(s+width,len(disc))],
                    'structure_end':str(disc[cut-1]),'calibration_start':str(disc[cut]),
                    'validation_start':str(disc[s]),
                    'validation_end':str(disc[min(s+width,len(disc))-1])})
    return out

def prepare_panel(panel:pd.DataFrame,cfg:Spec):
    need={'state_time','return_5m','feature_ready','log_rv24h',*STATE_FEATURES,*RETURN_FEATURES}
    if not need.issubset(panel):raise ValueError(f'Missing feature store fields: {sorted(need-set(panel))}')
    x=panel.copy();x['state_time']=pd.to_datetime(x.state_time,utc=True)
    x=x.sort_values('state_time').reset_index(drop=True)
    if x.state_time.duplicated().any() or (x.state_time.dt.minute%5).any():
        raise ValueError('Duplicate or off-grid state timestamp')
    x['date']=x.state_time.dt.floor('D')
    rr=x.return_5m.to_numpy(float);tt=x.state_time.to_numpy();n=len(x)
    for h in cfg.horizons:
        s=np.zeros(n,float);valid=np.ones(n,bool)
        for j in range(1,h//5+1):
            future=np.full(n,np.nan);future[:n-j]=rr[j:]
            adj=np.zeros(n,bool);adj[:n-j]=(tt[j:]-tt[:n-j])==np.timedelta64(j*5,'m')
            valid&=adj&np.isfinite(future)
            s+=np.nan_to_num(future,nan=0.)
        same=(x.state_time+pd.Timedelta(minutes=h)).dt.floor('D').eq(x.date).to_numpy()
        result=np.where(valid&same,s,np.nan)
        col=f'ret_{h}m'
        if col in x:
            prior=x[col].to_numpy(float)
            mask=np.isfinite(prior)&np.isfinite(result)
            if mask.any() and not np.allclose(prior[mask],result[mask],rtol=1e-9,atol=1e-12):
                raise ValueError(f'{col} violates causal rebuild')
        x[col]=result
    needed=list(set(STATE_FEATURES+RETURN_FEATURES+('log_rv24h','return_5m')))
    x['ready']=x.feature_ready.fillna(False).astype(bool)&np.isfinite(x[needed].to_numpy(float)).all(axis=1)
    g=x.groupby('date').agg(n=('state_time','size'),k=('ready','sum'),first=('state_time','min'),last=('state_time','max'))
    good=g.index[(g.n==288)&(g.k==288)&g['first'].eq(g.index)&g['last'].eq(g.index+pd.Timedelta(hours=23,minutes=55))]
    x=x[x.date.isin(good)].copy()
    x=x[(x.state_time.dt.hour*60+x.state_time.dt.minute)%cfg.anchor_minutes==0].reset_index(drop=True)
    if x.empty:raise ValueError('No aligned anchor origins')
    return x,pd.DatetimeIndex(good)

def normalized_emissions(structure,others,cfg:Spec):
    sc=StandardScaler();raw=structure[list(STATE_FEATURES)].to_numpy(float)
    sc.fit(raw)
    def apply(v):return np.clip(sc.transform(v[list(STATE_FEATURES)].to_numpy(float)),-6.,6.)
    return [apply(structure)]+[apply(v) for v in others],sc

def segments(times,step_minutes=30):
    t=pd.to_datetime(times,utc=True)
    if len(t)==0:return []
    split=np.flatnonzero((t[1:].to_numpy()-t[:-1].to_numpy())!=np.timedelta64(step_minutes,'m'))+1
    return [s for s in np.split(np.arange(len(t)),split) if len(s)]

def emission_loglik(X,loc,var,nu=6.):
    X=np.asarray(X,float)
    if X.ndim!=2:raise ValueError('X must be 2d')
    if not np.isfinite(X).all() or (np.asarray(var)<=0).any():raise ValueError('Invalid emission inputs')
    z=(X[:,None,:]-loc[None,:,:])/np.sqrt(var[None,:,:])
    return (student_t.logpdf(z,df=nu)-.5*np.log(var)[None,:,:]).sum(axis=2)

def forward_backward(loge,A,pi,bounds):
    T,K=loge.shape
    if T==0:raise ValueError('Empty sequence')
    logA=np.log(A);logpi=np.log(pi)
    alpha=np.zeros((T,K));beta=np.zeros((T,K));xi=np.zeros((K,K));ll=0.
    for idx in bounds:
        s=int(idx[0]);e=int(idx[-1])+1
        alpha[s]=logpi+loge[s]
        for j in range(s+1,e):alpha[j]=loge[j]+logsumexp(alpha[j-1,:,None]+logA,axis=0)
        ln=logsumexp(alpha[e-1]);ll+=ln
        for j in range(e-2,s-1,-1):beta[j]=logsumexp(logA+loge[j+1,None,:]+beta[j+1,None,:],axis=1)
        for j in range(s,e-1):xi+=np.exp(alpha[j,:,None]+logA+loge[j+1,None,:]+beta[j+1,None,:]-ln)
    loggamma=alpha+beta
    loggamma-=logsumexp(loggamma,axis=1,keepdims=True)
    return np.exp(loggamma),xi,float(ll)

def fit_hmm(X,times,cfg:Spec):
    X=np.asarray(X,float);n,d=X.shape;K=cfg.k_states
    if len(X)<max(300,20*K):raise ValueError('Insufficient structure samples')
    km=KMeans(n_clusters=K,n_init=4,random_state=cfg.seed).fit(X)
    loc=km.cluster_centers_.copy();var=np.maximum(np.var(X,axis=0)[None,:]*np.ones((K,1)),cfg.covariance_floor)
    A=np.full((K,K),.07/max(K-1,1));np.fill_diagonal(A,.93)
    pi=np.full(K,1/K)
    bounds=segments(times,cfg.anchor_minutes)
    history=[]
    for _ in range(cfg.em_iterations):
        loge=emission_loglik(X,loc,var,cfg.emission_df)
        gamma,xi,ll=forward_backward(loge,A,pi,bounds)
        history.append(ll)
        # Robust Student-t ECM emission update; gamma is the HMM state weight.
        z2=(X[:,None,:]-loc[None,:,:])**2/var[None,:,:]
        tw=(cfg.emission_df+1)/(cfg.emission_df+z2)
        w=gamma[:,:,None]*tw
        den=np.maximum(w.sum(axis=0),1e-6)
        newloc=(w*X[:,None,:]).sum(axis=0)/den
        # Parameterize scale of t residuals, not raw Gaussian variance.
        newvar=(gamma[:,:,None]*tw*(X[:,None,:]-newloc[None,:,:])**2).sum(axis=0)/np.maximum(gamma.sum(axis=0)[:,None],1e-6)
        loc=newloc;var=np.maximum(newvar,cfg.covariance_floor)
        sm=xi+cfg.transition_pseudocount*A # prior centered on previous A, fitted on STRUCT only
        A=sm/np.maximum(sm.sum(axis=1,keepdims=True),1e-12)
        pi=np.maximum(np.sum([gamma[int(b[0])] for b in bounds],axis=0),1e-6)
        pi/=pi.sum()
    return {'loc':loc,'var':var,'A':A,'pi':pi,'loglike':history,'K':K,'n_structure':n}

def viterbi(loge,A,pi,bounds):
    n,K=loge.shape;paths=np.zeros(n,dtype=int);logA=np.log(A)
    for ii in bounds:
        s=int(ii[0]);e=int(ii[-1])+1
        v=np.zeros((e-s,K));back=np.zeros((e-s,K),dtype=int)
        v[0]=np.log(pi)+loge[s]
        for j in range(1,e-s):
            z=v[j-1,:,None]+logA
            back[j]=np.argmax(z,axis=0);v[j]=loge[s+j]+z[back[j],np.arange(K)]
        paths[e-1]=np.argmax(v[-1])
        for j in range(e-s-2,-1,-1):paths[s+j]=back[j+1,paths[s+j+1]]
    return paths

def empirical_hazards(path,times,A,cfg:Spec):
    """Viterbi duration histogram with right-censored final episodes.

    This is an explicit age-dependent *two-stage overlay*, not a jointly fitted
    HSMM likelihood. Episodes at the ends of discontinuous segments are censored.
    """
    cap=cfg.duration_age_cap;K=cfg.k_states
    events=np.zeros((K,cap));exposure=np.zeros((K,cap));n_episodes=np.zeros(K,int)
    for ix in segments(times,cfg.anchor_minutes):
        states=path[ix];starts=np.r_[0,np.flatnonzero(states[1:]!=states[:-1])+1]
        ends=np.r_[starts[1:],len(states)]
        for start,end in zip(starts,ends):
            k=int(states[start]);dur=int(end-start);n_episodes[k]+=1
            for age in range(1,dur+1):exposure[k,min(age,cap)-1]+=1
            if end<len(states):events[k,min(dur,cap)-1]+=1
    baseline=np.clip(1-np.diag(A),.005,.8)
    hazard=(events+cfg.duration_prior_strength*baseline[:,None])/(exposure+cfg.duration_prior_strength)
    hazard=np.clip(hazard,.002,.95)
    return hazard,{'exits':events,'exposure':exposure,'episodes':n_episodes}

def filter_hmm(loge,A,pi,times,step_minutes=30):
    n,K=loge.shape;out=np.zeros((n,K))
    for ix in segments(times,step_minutes):
        prior=np.asarray(pi,float).copy()
        for j in ix:
            q=np.exp(loge[j]-loge[j].max())*prior
            q/=q.sum();out[j]=q;prior=q@A
    return out

def filter_semi_markov(loge,A,pi,hazard,times,step_minutes=30):
    """Logically exact filtering of the stated finite age-expanded overlay."""
    n,K=loge.shape;D=hazard.shape[1]
    out=np.zeros((n,K));off=A.copy();np.fill_diagonal(off,0)
    off/=off.sum(axis=1,keepdims=True)
    for ix in segments(times,step_minutes):
        prior=np.zeros((K,D));prior[:,0]=pi
        for j in ix:
            posterior=prior*np.exp(loge[j]-loge[j].max())[:,None]
            posterior/=posterior.sum();out[j]=posterior.sum(axis=1)
            leave=(posterior*hazard).sum(axis=1)
            follow=np.zeros_like(posterior)
            follow[:,0]=leave@off
            follow[:,1:]+=posterior[:,:-1]*(1-hazard[:,:-1])
            follow[:,-1]+=posterior[:,-1]*(1-hazard[:,-1])
            prior=follow/follow.sum()
    return out

def static_states(loge,pi):
    z=loge+np.log(pi)[None,:]
    return np.exp(z-logsumexp(z,axis=1,keepdims=True))

def split_t_logpdf(q,k,df=6.):
    """Mean-centered, variance-preserving two-piece Student-t; k=0 is t(df)."""
    q=np.asarray(q,float);k=np.asarray(k,float)
    a=np.exp(-k);b=np.exp(k)
    mean_abs=2*np.sqrt(df)*np.exp(gammaln((df+1)/2)-gammaln(df/2))/((df-1)*np.sqrt(np.pi))
    center=mean_abs*(b-a)
    second=df/(df-2)*(a*a*a+b*b*b)/(a+b)
    scale=np.sqrt((second-center*center)/(df/(df-2)))
    z=q*scale+center
    side=np.where(z<0,a,b)
    return np.log(2)-np.log(a+b)+student_t.logpdf(z/side,df=df)+np.log(scale)

def split_t_cdf(q,k,df=6.):
    q=np.asarray(q,float);k=np.asarray(k,float)
    a=np.exp(-k);b=np.exp(k)
    mean_abs=2*np.sqrt(df)*np.exp(gammaln((df+1)/2)-gammaln(df/2))/((df-1)*np.sqrt(np.pi))
    center=mean_abs*(b-a)
    second=df/(df-2)*(a*a*a+b*b*b)/(a+b)
    scale=np.sqrt((second-center*center)/(df/(df-2)))
    z=q*scale+center
    p=np.where(z<0,2*a/(a+b)*student_t.cdf(z/a,df=df),
               a/(a+b)+2*b/(a+b)*(student_t.cdf(z/b,df=df)-.5))
    return np.clip(p,0,1)

def weighted_state_response(q,posterior,cfg:Spec):
    """CAL-only shrunk residual locations and skew parameters per latent state."""
    q=np.asarray(q,float);p=np.asarray(posterior,float)
    if q.ndim!=1 or p.shape[0]!=len(q):raise ValueError('Response alignment')
    n=p.sum(axis=0);mu=(p*q[:,None]).sum(axis=0)/(n+cfg.response_shrinkage)
    global_k=float(minimize_scalar(lambda k:-split_t_logpdf(q, k,cfg.student_return_df).mean(),
                                   bounds=(-cfg.max_skew,cfg.max_skew),method='bounded').x)
    kval=[]
    for z,w,m in zip(range(p.shape[1]),p.T,mu):
        def loss(k):
            return (-float(np.dot(w,split_t_logpdf(q-m,k,cfg.student_return_df)))/max(w.sum(),1e-6)
                    +cfg.skew_shrinkage/max(w.sum(),1.)*(k-global_k)**2)
        kval.append(float(minimize_scalar(loss,bounds=(-cfg.max_skew,cfg.max_skew),method='bounded').x))
    return mu,np.asarray(kval,float),global_k,n

def calc_log_density(y,sigma,mu,scale,skew,post,cfg:Spec):
    """Predictive MIXTURE of conditional Student-t densities, never averaged params."""
    y=np.asarray(y,float);sigma=np.asarray(sigma,float);post=np.asarray(post,float)
    if not (post.ndim==2 and len(y)==len(sigma)==len(post)):raise ValueError('Unmatched origin arrays')
    q=(y[:,None]/sigma[:,None]-mu[None,:])/scale
    component=split_t_logpdf(q,skew[None,:],cfg.student_return_df)-np.log(scale)-np.log(sigma[:,None])
    nll=-logsumexp(component+np.log(np.maximum(post,1e-300)),axis=1)
    positive=1-split_t_cdf((-mu[None,:]/scale),skew[None,:],cfg.student_return_df)
    prob=(post*positive).sum(axis=1)
    return nll,prob

def direct_feature_predict(cal,val,y,cfg:Spec):
    a=cal[list(RETURN_FEATURES)].to_numpy(float);b=val[list(RETURN_FEATURES)].to_numpy(float)
    sc=StandardScaler().fit(a);z=sc.transform(a);v=sc.transform(b)
    lo,hi=np.quantile(y,[.005,.995]);fit=Ridge(alpha=cfg.ridge_alpha).fit(z,np.clip(y,lo,hi))
    return fit.predict(z),fit.predict(v),z,v

def continuous_skew_fit(cal_q,cal_X,cfg:Spec):
    """Proper-NLL penalized continuous skew comparator, fitted only on CAL."""
    x=np.asarray(cal_X,float);q=np.asarray(cal_q,float)
    if len(q)!=len(x):raise ValueError('Continuous skew fit alignment')
    design=np.column_stack((np.ones(len(q)),x))
    def fn(theta):
        k=cfg.max_skew*np.tanh(design@theta)
        return float(-np.mean(split_t_logpdf(q,k,cfg.student_return_df))+.04*np.dot(theta[1:],theta[1:]))
    result=minimize(fn,np.zeros(design.shape[1]),method='L-BFGS-B',
                    options={'maxiter':70,'ftol':1e-10})
    if not np.isfinite(result.fun):raise RuntimeError('Continuous skew fit nonfinite')
    return result.x

def continuous_skew_predict(theta,X,cfg:Spec):
    design=np.column_stack((np.ones(len(X)),np.asarray(X,float)))
    return cfg.max_skew*np.tanh(design@theta)

def observable_skew_tree(structure,cfg:Spec):
    """Tree from STRUCTURE only: half-sample residual-asymmetry proxy."""
    dates=pd.DatetimeIndex(sorted(structure.date.unique()));cut=max(8,int(.65*len(dates)))
    pre=structure[structure.date.isin(dates[:cut])]
    post=structure[structure.date.isin(dates[cut:])]
    h=cfg.primary_horizon_min
    pre=pre[np.isfinite(pre[f'ret_{h}m'])];post=post[np.isfinite(post[f'ret_{h}m'])]
    if len(post)<120:raise ValueError('Structure tree lacks anchors')
    ypre=pre[f'ret_{h}m'].to_numpy(float)/vol_proxy(pre,h)
    qpre,qpost,_,_=direct_feature_predict(pre,post,ypre,cfg)
    residual=post[f'ret_{h}m'].to_numpy(float)/vol_proxy(post,h)-qpost
    scale=max(np.std(np.clip(residual,*np.quantile(residual,[.005,.995]))),1e-6)
    z=np.clip(residual/scale,-3,3)
    target=z**3-3*z
    minimum=max(30,int(len(post)*cfg.tree_leaf_fraction))
    tree=DecisionTreeRegressor(max_depth=cfg.tree_max_depth,max_leaf_nodes=5,
                               min_samples_leaf=minimum,random_state=cfg.seed).fit(post[list(RETURN_FEATURES)],target)
    return tree

def fit_observable_response(tree,cal,q,scale,cfg:Spec):
    ids=tree.apply(cal[list(RETURN_FEATURES)])
    global_k=float(minimize_scalar(lambda k:-split_t_logpdf(q/scale,k,cfg.student_return_df).mean(),
                     bounds=(-cfg.max_skew,cfg.max_skew),method='bounded').x)
    labels=sorted(set(map(int,ids)));params={'global_skew':global_k}
    for k in labels:
        mask=ids==k
        # Relative to already fitted direct location: only residual skew, not scale/mean.
        qq=q[mask]/scale
        n=int(mask.sum())
        if n<80:continue
        kval=float(minimize_scalar(lambda z:-split_t_logpdf(qq,z,cfg.student_return_df).mean(),
                 bounds=(-cfg.max_skew,cfg.max_skew),method='bounded').x)
        params[int(k)]=global_k+(kval-global_k)*n/(n+cfg.skew_shrinkage)
    return params

def mean_daily_bootstrap(daily,cfg:Spec):
    # Bootstrap whole UTC days; every reference uses exact matched date grid.
    rng=np.random.default_rng(cfg.seed);out=[]
    for (asset,fold,h),group in daily.groupby(['asset','fold','horizon_min']):
        table=group.pivot(index='date',columns='method',values='nll')
        if not set(METHODS).issubset(table):continue
        if table.isna().any().any():raise ValueError('Unmatched daily observations')
        for target,ref in [('latent_hmm','direct_continuous_skew'),('latent_semi_markov','direct_continuous_skew'),
                           ('latent_semi_markov','latent_hmm'),('latent_hmm','latent_static'),
                           ('latent_hmm','observable_tree_skew'),('latent_semi_markov','observable_tree_skew')]:
            d=(table[target]-table[ref]).to_numpy(float);mean=float(d.mean())
            ix=rng.integers(0,len(d),size=(cfg.bootstrap_reps,len(d)))
            ci=np.quantile(d[ix].mean(axis=1),[.025,.975]) if len(d)>=20 else (np.nan,np.nan)
            out.append({'asset':asset,'fold':int(fold),'horizon_min':int(h),'model':target,'reference':ref,
                        'delta_nll':mean,'ci025':float(ci[0]),'ci975':float(ci[1]),
                        'n_days':len(d),'historical_only':True})
    return pd.DataFrame(out)

def scan_asset(panel:pd.DataFrame,asset:str,cfg:Spec):
    x,days=prepare_panel(panel,cfg);folds=combinations_dates(days,cfg)
    scores=[];daily=[];latent_diag=[];episodes=[];hazards=[];defs=[];state_resp=[]
    transition_rows=[];validation_states=[]
    for fold in folds:
        st=x[x.date.isin(fold['structure_days'])].copy()
        cal=x[x.date.isin(fold['calibration_days'])].copy()
        val=x[x.date.isin(fold['validation_days'])].copy()
        matrices,_=normalized_emissions(st,[cal,val],cfg);xs,xc,xv=matrices
        fit=fit_hmm(xs,st.state_time,cfg)
        est=emission_loglik(xs,fit['loc'],fit['var'],cfg.emission_df)
        path=viterbi(est,fit['A'],fit['pi'],segments(st.state_time,cfg.anchor_minutes))
        hazard,hd=empirical_hazards(path,st.state_time,fit['A'],cfg)
        for k in range(cfg.k_states):
            for a in range(cfg.duration_age_cap):
                hazards.append({'asset':asset,'fold':fold['fold'],'state':k,'age_blocks':a+1,
                                'exit_hazard':hazard[k,a], 'exposures':hd['exposure'][k,a],
                                'exits':hd['exits'][k,a]})
            for j in range(cfg.k_states):
                transition_rows.append({'asset':asset,'fold':fold['fold'],'from_state':k,'to_state':j,
                                        'hmm_transition_probability':float(fit['A'][k,j])})
            latent_diag.append({'asset':asset,'fold':fold['fold'],'state':k,
                                'prior':fit['pi'][k],'stay_hmm':fit['A'][k,k],
                                'viterbi_episodes':hd['episodes'][k],
                                'emission_mean_standardized':float(np.mean(fit['loc'][k])),
                                'structure_obs':len(st),'loglike_last':fit['loglike'][-1]})
        ec=emission_loglik(xc,fit['loc'],fit['var'],cfg.emission_df)
        ev=emission_loglik(xv,fit['loc'],fit['var'],cfg.emission_df)
        cp={'latent_static':static_states(ec,fit['pi']),
            'latent_hmm':filter_hmm(ec,fit['A'],fit['pi'],cal.state_time,cfg.anchor_minutes),
            'latent_semi_markov':filter_semi_markov(ec,fit['A'],fit['pi'],hazard,cal.state_time,cfg.anchor_minutes)}
        vp={'latent_static':static_states(ev,fit['pi']),
            'latent_hmm':filter_hmm(ev,fit['A'],fit['pi'],val.state_time,cfg.anchor_minutes),
            'latent_semi_markov':filter_semi_markov(ev,fit['A'],fit['pi'],hazard,val.state_time,cfg.anchor_minutes)}
        # Causal duration/transition descriptors are based on FILTERED argmax states,
        # never backward-smoothed labels on validation.
        lab=vp['latent_hmm'].argmax(axis=1)
        for inds in segments(val.state_time,cfg.anchor_minutes):
            piece=lab[inds]
            b=np.r_[0,np.flatnonzero(piece[1:]!=piece[:-1])+1]
            for i,j in zip(b,np.r_[b[1:],len(piece)]):
                episodes.append({'asset':asset,'fold':fold['fold'],'state':int(piece[i]),
                                 'duration_blocks':int(j-i),'censored_at_segment_end':bool(j==len(piece))})
        mask30=np.isfinite(val[f'ret_{cfg.primary_horizon_min}m'].to_numpy(float))
        for k in range(cfg.k_states):
            mk=(lab==k)&mask30
            if mk.any():
                raw=val.loc[mk,f'ret_{cfg.primary_horizon_min}m'].to_numpy(float)
                validation_states.append({'asset':asset,'fold':fold['fold'],'state':k,
                       'n_forecast_origins':int(mk.sum()),'n_days':int(val.loc[mk,'date'].nunique()),
                       'occupancy':float((lab==k).mean()),'next30_up_frequency':float((raw>0).mean()),
                       'next30_mean_bps':float(np.mean(raw)*1e4),
                       'next30_median_bps':float(np.median(raw)*1e4)})
        tree=observable_skew_tree(st,cfg)
        defs.append({'asset':asset,'fold':fold['fold'],'tree_definition':export_text(tree,feature_names=list(RETURN_FEATURES)),
                     'structure_end':fold['structure_end'],'calibration_start':fold['calibration_start'],
                     'validation_start':fold['validation_start'],'validation_end':fold['validation_end']})
        for h in cfg.horizons:
            mc=np.isfinite(cal[f'ret_{h}m'].to_numpy(float));mv=np.isfinite(val[f'ret_{h}m'].to_numpy(float))
            ca=cal.loc[mc];va=val.loc[mv]
            if len(ca)<150 or len(va)<100:raise ValueError('Insufficient return calibration')
            yc=ca[f'ret_{h}m'].to_numpy(float)/vol_proxy(ca,h)
            yv=va[f'ret_{h}m'].to_numpy(float)/vol_proxy(va,h)
            mu_c,mu_v,z_c,z_v=direct_feature_predict(ca,va,yc,cfg)
            resid=yc-mu_c
            cap=np.quantile(resid*resid,.995)
            tscale=float(np.sqrt(max(np.minimum(resid*resid,cap).mean(),1e-8)*(cfg.student_return_df-2)/cfg.student_return_df))
            residual_q=resid/tscale
            cont=continuous_skew_fit(residual_q,z_c,cfg)
            cont_k=continuous_skew_predict(cont,z_v,cfg)
            # Matched continuous control, with no changes to volatility scale.
            methods={}
            for method,kval in [('direct_symmetric',np.zeros(len(va))),('direct_continuous_skew',cont_k)]:
                q=(yv-mu_v)/tscale
                loss=-split_t_logpdf(q,kval,cfg.student_return_df)+np.log(tscale)+np.log(vol_proxy(va,h))
                pup=1-split_t_cdf(-mu_v/tscale,kval,cfg.student_return_df)
                methods[method]=(loss,pup)
            obslookup=fit_observable_response(tree,ca,resid,tscale,cfg)
            obsids=tree.apply(va[list(RETURN_FEATURES)])
            obs_k=np.array([obslookup.get(int(z),obslookup['global_skew']) for z in obsids])
            q=(yv-mu_v)/tscale
            methods['observable_tree_skew']=(-split_t_logpdf(q,obs_k,cfg.student_return_df)+np.log(tscale)+np.log(vol_proxy(va,h)),
                                             1-split_t_cdf(-mu_v/tscale,obs_k,cfg.student_return_df))
            for method in ['latent_static','latent_hmm','latent_semi_markov']:
                loc,kval,kg,weights=weighted_state_response(residual_q,cp[method][mc],cfg)
                # calc_log_density uses zero direct location: shift observed return
                # before mixture evaluation while preserving the original scale.
                loss,pup=calc_log_density(va[f'ret_{h}m'].to_numpy(float)-vol_proxy(va,h)*mu_v,
                            vol_proxy(va,h),loc*tscale,tscale,kval,vp[method][mv],cfg)
                # Returned sign P needs threshold relative to baseline direct mu.
                pz=1-split_t_cdf((-mu_v[:,None]-loc[None,:]*tscale)/tscale,kval[None,:],cfg.student_return_df)
                pup=(vp[method][mv]*pz).sum(axis=1)
                methods[method]=(loss,pup)
                if h==cfg.primary_horizon_min:
                    for k in range(cfg.k_states):
                        state_resp.append({'asset':asset,'fold':fold['fold'],'method':method,'state':k,
                                           'standardized_return_shift':loc[k]*tscale,
                                           'student_scale_units_shift':loc[k],'skew':kval[k],
                                           'cal_effective_origins':weights[k]})
            dates=va.date.to_numpy()
            for method,(loss,pup) in methods.items():
                if not np.isfinite(loss).all() or not np.isfinite(pup).all():raise ValueError('Nonfinite predictions')
                brier=(pup-(va[f'ret_{h}m'].to_numpy(float)>0).astype(float))**2
                scores.append({'asset':asset,'fold':fold['fold'],'method':method,'horizon_min':h,'n':len(va),
                               'nll':float(np.mean(loss)),'sign_brier':float(brier.mean()),
                               'mean_pred_up':float(pup.mean()),'mean_observed_up':float((yv>0).mean())})
                vals=pd.DataFrame({'date':dates,'nll':loss,'sign_brier':brier})
                per=vals.groupby('date',as_index=False).mean(numeric_only=True)
                for row in per.itertuples(index=False):
                    daily.append({'asset':asset,'fold':fold['fold'],'horizon_min':h,'method':method,
                                  'date':row.date,'nll':row.nll,'sign_brier':row.sign_brier})
    return {'scores':pd.DataFrame(scores),'daily':pd.DataFrame(daily),
            'states':pd.DataFrame(latent_diag),'hazards':pd.DataFrame(hazards),
            'episodes':pd.DataFrame(episodes),'definitions':pd.DataFrame(defs),
            'responses':pd.DataFrame(state_resp),'transitions':pd.DataFrame(transition_rows),
            'validation_states':pd.DataFrame(validation_states),'folds':folds,'complete_days':len(days)}

def vol_proxy(frame,h):
    return np.sqrt(np.maximum(np.exp(np.clip(frame.log_rv24h.to_numpy(float),-70,20)),1e-12)*(h//5))

def consolidate(results:dict[str,dict],cfg:Spec,synthetic=False):
    keys=['scores','daily','states','hazards','episodes','definitions','responses','transitions','validation_states']
    out={key:pd.concat([r[key] for r in results.values()],ignore_index=True) for key in keys}
    out['bootstrap']=mean_daily_bootstrap(out['daily'],cfg)
    tab=out['scores'].pivot_table(index=['asset','fold','horizon_min'],columns='method',values='nll')
    for method in METHODS:
        if method not in tab:raise ValueError(f'Missing fitted method {method}')
    deltas={method:(tab[method]-tab['direct_continuous_skew']) for method in METHODS if method!='direct_continuous_skew'}
    rows=[]
    for method,v in deltas.items():
        for h in cfg.horizons:
            w=v.xs(h,level='horizon_min')
            rows.append({'method':method,'horizon_min':h,'delta_nll_vs_continuous_skew':float(w.mean()),
                         'favorable_asset_folds':int((w<0).sum()),'n_asset_folds':len(w)})
    out['comparison']=pd.DataFrame(rows)
    primary=out['comparison'].query('horizon_min==@cfg.primary_horizon_min')
    passed={z.method:bool(z.favorable_asset_folds>=10 and z.delta_nll_vs_continuous_skew<0)
            for z in primary.itertuples(index=False)}
    out['decision']={'status':cfg.status,'primary_horizon_min':cfg.primary_horizon_min,
                     'heuristic_favorable':passed,'latent_hmm_competitive':passed.get('latent_hmm',False),
                     'latent_duration_competitive':passed.get('latent_semi_markov',False),
                     'historical_exploratory':True,'prospective_confirmation':False,
                     'synthetic':bool(synthetic),'AM_T_HAR_Blend_modified':False,
                     'warning':'RD02 semi-Markov is a two-stage duration overlay; not a jointly fitted HSMM. '
                               'Normalization is trailing RV, NOT AM-T-HAR-Blend. Historical development folds already inspected.'}
    out['manifest']={'module':'RD02','registration':registration(),
                     'assets':list(results),'folds':{k:[{w:z for w,z in f.items() if not w.endswith('_days')} for f in r['folds']] for k,r in results.items()},
                     'complete_days':{k:r['complete_days'] for k,r in results.items()},
                     'local_inputs':'CS04-A feature store only','historical_legacy_test_used':False,
                     'normalizer':'TRAILING_24H_RV_PROXY_NOT_AM_T_HAR_BLEND',
                     'latent_model':'Student-t-emission 3-state HMM EM; Viterbi-trained semi-Markov age-hazard overlay',
                     'synthetic':bool(synthetic),'AM_T_HAR_Blend_modified':False}
    return out
