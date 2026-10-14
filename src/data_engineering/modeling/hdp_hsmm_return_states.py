"""Return State Discovery 01: finite weak-limit hierarchical-DP explicit-duration HSMM.

Research-only. State observations are NON-OVERLAPPING completed 30m return blocks,
standardized by a causal variance forecast supplied by VD18. No K-grid selection.
Finite truncation Kcap is a numerical ceiling, not a selected state count.
Age Dmax is a capped-age representation; its last hazard continues thereafter.
Thus a single state can persist arbitrarily long in a contiguous observation run.

Global stick surrogate beta ~ Dirichlet(gamma/Kcap), off-diagonal rows
pi_i,-i ~ Dirichlet(alpha * beta_-i/(1-beta_i)); no self-transitions,
which is essential to identifying explicit negative-binomial durations.
The beta MH update integrates transition rows analytically.  Age-expanded
FFBS samples states/episodes; Student-t emissions use Gamma scale augmentation and
Normal-Inverse-Wishart Gibbs updates. This is a finite weak-limit HDP-HSMM,
not an exact infinite-state posterior or a 'sticky' self-transition HSMM.
State persistence is encouraged by the explicit duration distribution, avoiding
redundant stickiness in pi_ii. Final episodes are right-censored, with survival contributions in the duration
update; first episodes are treated as starting at the observation boundary.
Age-cap and boundary sensitivity remain prerequisites for duration interpretation.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import invwishart, nbinom
from numba import njit

VERSION='RSD01-0.1.0'
FEATURES=('return_surprise','signed_path_efficiency','within_block_dependence')

@dataclass(frozen=True)
class Spec:
    kcap:int=12
    dmax:int=96  # age representation capped at 48 h; states may persist longer
    alpha:float=3.0
    gamma:float=0.7
    duration_shape:int=2  # shifted NB count of successes; not geometric
    duration_a:float=2.0
    duration_b:float=8.0
    df:float=6.0
    n_iter:int=140
    burn:int=80
    thin:int=4
    seed:int=1801
    beta_proposal_concentration:float=250.0
    def validate(self):
        if self.kcap<2 or self.dmax<2 or self.duration_shape<2 or self.df<=2:
            raise ValueError('invalid HSMM dimensions/df')
        if min(self.alpha,self.gamma,self.duration_a,self.duration_b)<=0:
            raise ValueError('prior concentration must be positive')
        if self.burn>=self.n_iter or self.thin<1:
            raise ValueError('no posterior draws possible')
        return self


def block_features(returns6:np.ndarray,ex_ante_var:np.ndarray)->np.ndarray:
    """Features calculated only WHEN the 30m block ends.

    Input rows are six consecutive observed returns AFTER the forecast origin.
    No overlap is allowed between successive blocks within a chain.
    """
    r=np.asarray(returns6,float);v=np.asarray(ex_ante_var,float)
    if r.ndim!=2 or r.shape[1]!=6 or v.shape!=(len(r),):raise ValueError('six 5m returns required')
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(v)) and np.all(v>0)):
        raise ValueError('invalid return blocks/forecast variance')
    total=r.sum(axis=1)
    surprise=np.clip(total/np.sqrt(v),-15,15)
    # Signed path efficiency captures directional travel, not just magnitude.
    eff=total/np.maximum(np.abs(r).sum(axis=1),1e-12)
    # Return-product statistic: positive for same-direction runs, negative for alternations.
    dep=np.sum(r[:,:-1]*r[:,1:],axis=1)/np.maximum(np.sum(r*r,axis=1),1e-24)
    return np.column_stack([surprise, np.arctanh(np.clip(eff,-.999,.999))/2.0,
                            np.arctanh(np.clip(dep,-.95,.95))])


def fit_normalizer(x:np.ndarray)->dict:
    x=np.asarray(x,float)
    med=np.median(x,axis=0)
    mad=np.median(np.abs(x-med),axis=0)*1.4826
    sd=np.where(mad>.05,mad,np.maximum(x.std(axis=0),.25))
    return {'center':med.tolist(),'scale':sd.tolist()}


def apply_normalizer(x:np.ndarray,norm:dict)->np.ndarray:
    out=(np.asarray(x,float)-np.asarray(norm['center']))/np.asarray(norm['scale'])
    return out


def _nb_unnormalized(p:np.ndarray,D:int,r:int)->np.ndarray:
    """Shifted NB: d=1+failures before r successes, truncated at D.
    D-normalization explicitly included; if fitted durations mass near D,
    the report flags truncation and no duration interpretation is allowed.
    """
    ds=np.arange(1,D+1,dtype=float)
    coef=gammaln(ds+r-1)-gammaln(ds)-gammaln(r)
    val=coef[None,:]+r*np.log(p[:,None])+(ds[None,:]-1)*np.log1p(-p[:,None])
    return val


def _log_nb_durations(p:np.ndarray,D:int,r:int)->np.ndarray:
    val=_nb_unnormalized(p,D,r)
    return val-logsumexp(val,axis=1)[:,None]

@njit(cache=True)
def _lse2(a,b):
    if a==-np.inf:return b
    if b==-np.inf:return a
    m=max(a,b)
    return m+np.log(np.exp(a-m)+np.exp(b-m))

@njit(cache=True)
def _ffbs_forward(ll:np.ndarray,logpi:np.ndarray,logbeta:np.ndarray,logdur:np.ndarray):
    T,K=ll.shape;D=logdur.shape[1]
    prefix=np.zeros((T+1,K))
    for t in range(T):
        prefix[t+1]=prefix[t]+ll[t]
    alpha=np.full((T,K),-np.inf)
    trans=np.full((T,K),-np.inf)
    for t in range(T):
        for k in range(K):
            acc=-np.inf
            for d in range(1,min(D,t+1)+1):
                s=t-d+1
                prev=logbeta[k] if s==0 else trans[s-1,k]
                a=prev+logdur[k,d-1]+prefix[t+1,k]-prefix[s,k]
                acc=_lse2(acc,a)
            alpha[t,k]=acc
        for j in range(K):
            acc=-np.inf
            for k in range(K):
                if k!=j:acc=_lse2(acc,alpha[t,k]+logpi[k,j])
            trans[t,j]=acc
    return alpha,trans,prefix

@njit(cache=True)
def _pick(logw:np.ndarray,u:float):
    m=np.max(logw)
    if not np.isfinite(m):return -1
    w=np.exp(logw-m);total=np.sum(w)
    target=u*total;val=0.
    for i in range(len(w)):
        val+=w[i]
        if val>=target:return i
    return len(w)-1

@njit(cache=True)
def _ffbs_back(alpha,trans,prefix,logpi,logbeta,logdur,uniform):
    T,K=alpha.shape;D=logdur.shape[1]
    z=np.zeros(T,np.int64)
    t=T-1;state=_pick(alpha[t],uniform[t,0])
    while t>=0:
        ld=np.full(min(D,t+1),-np.inf)
        for d in range(1,len(ld)+1):
            s=t-d+1
            prev=logbeta[state] if s==0 else trans[s-1,state]
            ld[d-1]=prev+logdur[state,d-1]+prefix[t+1,state]-prefix[s,state]
        idx=_pick(ld,uniform[t,1]);d=idx+1;s=t-d+1
        z[s:t+1]=state
        if s==0:break
        lw=alpha[s-1]+logpi[:,state]
        state=_pick(lw,uniform[t,2]);t=s-1
    return z


@njit(cache=True)
def _age_ffbs(log_emit,pi,beta,hazard,uniform):
    """Blocked FFBS on expanded (state,age) representation.

    Last age bucket means age >= D; its exit hazard stays at the NB hazard
    evaluated at D (constant-tail approximation). Thus ONE occupied state
    can persist indefinitely without a forced alternation at D.
    """
    T,K=log_emit.shape;D=hazard.shape[1]
    trace=np.zeros((T,K,D))
    trace[0,:,0]=beta*np.exp(log_emit[0]-np.max(log_emit[0]))
    mass=np.sum(trace[0]);trace[0]/=max(mass,1e-300)
    loglike=np.log(max(mass,1e-300))+np.max(log_emit[0])
    for t in range(1,T):
        previous=trace[t-1]
        dep=np.zeros(K);pred=np.zeros((K,D))
        for i in range(K):
            for age in range(D):
                v=previous[i,age]
                dep[i]+=v*hazard[i,age]
                nextage=min(age+1,D-1)
                pred[i,nextage]+=v*(1-hazard[i,age])
        for i in range(K):
            for j in range(K):
                if i!=j:pred[j,0]+=dep[i]*pi[i,j]
        m=np.max(log_emit[t])
        for i in range(K):
            pred[i]*=np.exp(log_emit[t,i]-m)
        norm=np.sum(pred);trace[t]=pred/max(norm,1e-300)
        loglike+=np.log(max(norm,1e-300))+m
    z=np.zeros(T,np.int64)
    state_idx=_pick(np.log(np.maximum(trace[-1].ravel(),1e-300)),uniform[-1,0])
    state=state_idx//D;age=state_idx%D;z[T-1]=state
    for t in range(T-2,-1,-1):
        if age>0 and age<D-1:
            age-=1
        elif age==D-1:
            w0=trace[t,state,D-2]*(1-hazard[state,D-2])
            w1=trace[t,state,D-1]*(1-hazard[state,D-1])
            age=D-2 if uniform[t,0]*(w0+w1)<w0 else D-1
        else:
            w=np.empty(K*D)
            for i in range(K):
                for j in range(D):
                    w[i*D+j]=trace[t,i,j]*hazard[i,j]*pi[i,state]
            parent=_pick(np.log(np.maximum(w,1e-300)),uniform[t,0])
            state=parent//D;age=parent%D
        z[t]=state
    return z,loglike


def _duration_hazards(p:np.ndarray,D:int,r:int):
    """NB episode hazard; terminal bin is a *survival* age bucket, not forced exit."""
    ds=np.arange(1,D+1)
    density=nbinom.pmf(ds-1,r,p[:,None])
    survival=nbinom.sf(ds[None,:]-2,r,p[:,None])
    haz=np.clip(density/np.maximum(survival,1e-290),1e-9,1-1e-9)
    return haz

def _transition_counts(z,K):
    idx=np.r_[0,np.where(z[1:]!=z[:-1])[0]+1,len(z)]
    states=z[idx[:-1]];lengths=np.diff(idx)
    counts=np.zeros((K,K),dtype=int)
    for a,b in zip(states[:-1],states[1:]):counts[a,b]+=1
    return counts,states,lengths


def _log_beta_conditional(beta,counts,alpha,gamma):
    K=len(beta)
    if np.min(beta)<=0:return -np.inf
    ans=np.sum((gamma/K-1)*np.log(beta))
    for i in range(K):
        b=beta/(1-beta[i]);b[i]=0.
        for j in range(K):
            if i!=j:
                q=max(alpha*b[j],1e-12)
                ans+=gammaln(q+counts[i,j])-gammaln(q)
    return ans


def _dir_logpdf(x,a):
    return gammaln(np.sum(a))-np.sum(gammaln(a))+np.sum((a-1)*np.log(np.maximum(x,1e-300)))


def _update_beta(beta,counts,initial,sp,rng):
    """Metropolis on additive-log-ratio coordinates, with simplex Jacobian.

    This avoids effectively zero acceptance from Dirichlet proposals when
    posterior beta is sparse. Gaussian proposal is symmetric in logit space.
    """
    eta=np.log(np.maximum(beta[:-1],1e-300))-np.log(max(beta[-1],1e-300))
    cand_eta=eta+rng.normal(0,0.24,size=len(eta))
    raw=np.r_[cand_eta,0.]
    raw-=logsumexp(raw)
    cand=np.exp(raw)
    lp=_log_beta_conditional(cand,counts,sp.alpha,sp.gamma)-_log_beta_conditional(beta,counts,sp.alpha,sp.gamma)
    lp+=np.dot(initial,np.log(cand)-np.log(beta))
    lp+=np.sum(np.log(cand))-np.sum(np.log(beta))
    return (cand,True) if np.log(rng.random())<lp else (beta,False)


def _sample_params(x,z,latent,sp,rng,segments,pprev=None):
    T,d=x.shape;K=sp.kcap
    means=np.zeros((K,d));cov=np.zeros((K,d,d));p=np.empty(K)
    dur=[[] for _ in range(K)];right_censored=[[] for _ in range(K)]
    for lo,hi in zip(segments[:-1],segments[1:]):
        _,states,lengths=_transition_counts(z[lo:hi],K)
        for state,length in zip(states[:-1],lengths[:-1]):dur[state].append(int(length))
        right_censored[states[-1]].append(int(lengths[-1]))
    for k in range(K):
        mask=(z==k);xk=x[mask];wk=latent[mask]
        mass=wk.sum();kappa=.5+mass;nu=d+3+len(xk)
        xm=np.sum(xk*wk[:,None],axis=0)/mass if len(xk) else np.zeros(d)
        postmean=mass*xm/kappa
        diff=xk-xm
        scatter=(diff*wk[:,None]).T@diff if len(xk) else np.zeros((d,d))
        scale=2*np.eye(d)+scatter+(.5*mass/kappa)*np.outer(xm,xm)
        sigma=invwishart.rvs(df=nu,scale=scale,random_state=rng)
        sigma=np.atleast_2d(sigma)
        mu=rng.multivariate_normal(postmean,sigma/kappa)
        means[k]=mu;cov[k]=sigma
        # Censoring at the observed window boundary is approximated; flag in protocol.
        ds=np.asarray(dur[k],int)
        a=sp.duration_a+sp.duration_shape*len(ds)
        b=sp.duration_b+np.sum(ds-1)
        proposal=rng.beta(a,b)
        censor=np.asarray(right_censored[k],int)
        if len(censor) and pprev is not None:
            # Right-censored final episodes contribute survival P(D >= observed).
            # The independent Beta proposal cancels prior and *completed*
            # episode likelihood; this MH ratio corrects the censoring factor.
            old=pprev[k]
            def log_surv(v):
                return np.sum(nbinom.logsf(censor-2,sp.duration_shape,v))
            lp=log_surv(proposal)-log_surv(old)
            p[k]=proposal if np.log(rng.random())<lp else old
        else:p[k]=proposal
    return means,cov,p


def _student_log_emissions(x,mu,cov,df):
    T,d=x.shape;K=len(mu)
    ll=np.zeros((T,K))
    for k in range(K):
        sign,ld=np.linalg.slogdet(cov[k]);
        if sign<=0:raise ValueError('invalid covariance')
        diff=x-mu[k];maha=np.einsum('ij,jk,ik->i',diff,np.linalg.inv(cov[k]),diff)
        ll[:,k]=(gammaln((df+d)/2)-gammaln(df/2)-.5*(d*np.log(df*np.pi)+ld)
                  -(df+d)/2*np.log1p(maha/df))
    return ll


def _sample_latent(x,z,mu,cov,df,rng):
    d=x.shape[1];out=np.empty(len(x))
    for k in range(len(mu)):
        ix=np.flatnonzero(z==k)
        if not len(ix):continue
        diff=x[ix]-mu[k]
        m=np.einsum('ij,jk,ik->i',diff,np.linalg.inv(cov[k]),diff)
        out[ix]=rng.gamma((df+d)/2,2/(df+m))
    return out


def _initial_labels(T,K,rng):
    z=np.empty(T,dtype=np.int64);i=0;last=-1
    n_initial=min(4,K)
    while i<T:
        choices=[j for j in range(n_initial) if j!=last]
        s=int(rng.choice(choices))
        d=int(rng.integers(3,19));z[i:min(T,i+d)]=s;i+=d;last=s
    return z


def fit_posterior(x:np.ndarray,sp:Spec=Spec(),segments=None):
    """Return retained posterior draws of occupied K and state paths.
    One contiguous, non-overlapping block series per fit (one asset); all
    hyperparameters fixed before real-data inspection.
    """
    sp.validate();x=np.asarray(x,float)
    if x.ndim!=2 or len(x)<24 or not np.isfinite(x).all():raise ValueError('need >=24 finite observations')
    rng=np.random.default_rng(sp.seed);T,d=x.shape;K=sp.kcap
    segments=np.asarray([0,T] if segments is None else segments,dtype=int)
    if segments[0]!=0 or segments[-1]!=T or np.any(np.diff(segments)<24):
        raise ValueError('segments must start at zero and each contain at least 24 complete blocks')
    beta=rng.dirichlet(np.full(K,sp.gamma/K))
    beta=np.maximum(beta,1e-300);beta/=beta.sum()
    z=np.concatenate([_initial_labels(hi-lo,K,rng) for lo,hi in zip(segments[:-1],segments[1:])])
    latent=np.ones(T)
    mu,cov,p=_sample_params(x,z,latent,sp,rng,segments)
    draws=[];mh=0
    for step in range(sp.n_iter):
        counts=np.zeros((K,K),int);initial=np.zeros(K,int)
        for lo,hi in zip(segments[:-1],segments[1:]):
            cc,_,_=_transition_counts(z[lo:hi],K)
            counts+=cc;initial[z[lo]]+=1
        beta,accepted=_update_beta(beta,counts,initial,sp,rng);mh+=accepted
        pi=np.zeros((K,K))
        for k in range(K):
            ix=np.arange(K)!=k
            pars=sp.alpha*beta[ix]/max(1-beta[k],1e-12)+counts[k,ix]
            pi[k,ix]=rng.dirichlet(pars)
        ll=_student_log_emissions(x,mu,cov,sp.df)
        hazard=_duration_hazards(p,sp.dmax,sp.duration_shape)
        loglik=0.0
        for lo,hi in zip(segments[:-1],segments[1:]):
            uniform=rng.random((hi-lo,3))
            z[lo:hi],l=_age_ffbs(ll[lo:hi],pi,beta,hazard,uniform)
            loglik+=float(l)
        latent=_sample_latent(x,z,mu,cov,sp.df,rng)
        mu,cov,p=_sample_params(x,z,latent,sp,rng,segments,pprev=p)
        if step>=sp.burn and (step-sp.burn)%sp.thin==0:
            all_states=[];all_lengths=[]
            for lo,hi in zip(segments[:-1],segments[1:]):
                _,st,le=_transition_counts(z[lo:hi],K)
                all_states.extend(st);all_lengths.extend(le)
            states=np.asarray(all_states,int);lengths=np.asarray(all_lengths,int)
            occupancy=np.bincount(z,minlength=K)
            draws.append({'K_occupied':int(np.sum(occupancy>0)),'occupancy':occupancy.copy(),
                          'labels':z.astype(np.int16).copy(),'beta':beta.copy(),
                          'pi':pi.copy(),'mu':mu.copy(),'cov':cov.copy(),'duration_p':p.copy(),
                          'episode_lengths':lengths.copy(),'episode_states':states.copy(),
                          'duration_at_cap_rate':float(np.mean(lengths>=sp.dmax)),
                          'likelihood':loglik})
    if not draws:raise RuntimeError('no posterior draws')
    return {'version':VERSION,'spec':asdict(sp),'draws':draws,'mh_acceptance':mh/sp.n_iter}


@njit(cache=True)
def _predictive_filter(ll,pi,beta,haz):
    T,K=ll.shape;D=haz.shape[1]
    state_age=np.zeros((K,D));state_age[:,0]=beta
    out=np.empty(T)
    for t in range(T):
        if t:
            departures=np.zeros(K)
            new_age=np.zeros((K,D))
            for k in range(K):
                for a in range(D):
                    mass=state_age[k,a]
                    departures[k]+=mass*haz[k,a]
                    new_age[k,min(a+1,D-1)]+=mass*(1-haz[k,a])
            for j in range(K):
                for k in range(K):
                    if k!=j:new_age[j,0]+=departures[k]*pi[k,j]
            state_age=new_age
        predictive=0.
        for k in range(K):
            term=np.exp(ll[t,k]);
            for a in range(D):predictive+=state_age[k,a]*term
        if predictive<=1e-280:predictive=1e-280
        out[t]=np.log(predictive)
        for k in range(K):
            for a in range(D):state_age[k,a]*=np.exp(ll[t,k])/predictive
    return out


def predict_logscore(x,pstr,limit_draws=16,segments=None):
    """Causal joint-observation log scores averaged over posterior *predictive densities*.
    Each draw filters test observations sequentially; no test refitting/smoothing.
    """
    x=np.asarray(x,float);sp=Spec(**pstr['spec'])
    segments=np.asarray([0,len(x)] if segments is None else segments,dtype=int)
    draws=pstr['draws'];stride=max(1,len(draws)//limit_draws)
    samples=[]
    for draw in draws[::stride][:limit_draws]:
        ll=_student_log_emissions(x,draw['mu'],draw['cov'],sp.df)
        hazard=_duration_hazards(draw['duration_p'],sp.dmax,sp.duration_shape)
        scores=np.concatenate([_predictive_filter(ll[lo:hi],draw['pi'],draw['beta'],hazard)
            for lo,hi in zip(segments[:-1],segments[1:])])
        samples.append(scores)
    return logsumexp(np.vstack(samples),axis=0)-np.log(len(samples))


def fit_null_t(x,df=6.0):
    """State-free multivariate Student-t with a single robust location/covariance.
    Fixed-df EM for t scales, no HMM or latent switches.
    """
    x=np.asarray(x,float);mu=np.median(x,axis=0);cov=np.cov(x.T)+.05*np.eye(x.shape[1])
    for _ in range(40):
        d=x-mu
        m=np.einsum('ij,jk,ik->i',d,np.linalg.inv(cov),d)
        w=(df+x.shape[1])/(df+m)
        mu=(w[:,None]*x).sum(axis=0)/w.sum()
        d=x-mu;cov=(w[:,None]*d).T@d/len(x)+.02*np.eye(x.shape[1])
    return {'mu':mu,'cov':cov,'df':df}


def null_logscore(x,fit):
    return _student_log_emissions(np.asarray(x,float),fit['mu'][None,:],
                                   fit['cov'][None,:,:],fit['df'])[:,0]


def posterior_summary(pstr):
    draws=pstr['draws'];ks=np.asarray([d['K_occupied'] for d in draws]);cap=pstr['spec']['kcap']
    histogram={str(int(k)):int(np.sum(ks==k)) for k in sorted(set(ks))}
    return {'n_draws':len(draws),'K_occupied_median':float(np.median(ks)),
            'K_occupied_05_95':[float(np.quantile(ks,q)) for q in (.05,.95)],
            'K_histogram':histogram,'at_Kcap_probability':float(np.mean(ks>=cap)),
            'at_duration_cap_mean':float(np.mean([x['duration_at_cap_rate'] for x in draws])),
            'beta_mh_acceptance':pstr['mh_acceptance'],
            'warning':'RERUN_HIGHER_KCAP' if np.mean(ks>=cap-1)>.1 else 'NONE'}


def simulate_price_blocks(n:int,variance:np.ndarray|float=1.,seed=7,df=6):
    """Regime-free heavy-tailed null retaining supplied ex-ante variance series.
    This is a CONDITIONAL negative control, not a jointly refitted HAR return DGP.
    """
    rng=np.random.default_rng(seed)
    v=np.broadcast_to(np.asarray(variance,float),(n,)).copy()
    r=rng.standard_t(df,size=(n,6))*np.sqrt((df-2)/df)*np.sqrt(v[:,None]/6)
    return block_features(r,v)

@njit(cache=True)
def _conditional_return_filter(joint_loglik,ret_loglik,pi,beta,haz):
    """Score x[t,0] BEFORE observing full block t; then update on completed block.
    Per-step return-score is a valid causal log density in normalized-return space.
    """
    T,K=joint_loglik.shape;D=haz.shape[1]
    state_age=np.zeros((K,D));state_age[:,0]=beta
    out=np.empty(T)
    for t in range(T):
        if t:
            departures=np.zeros(K)
            new_age=np.zeros((K,D))
            for k in range(K):
                for a in range(D):
                    mass=state_age[k,a]; departures[k]+=mass*haz[k,a]
                    new_age[k,min(a+1,D-1)]+=mass*(1-haz[k,a])
            for j in range(K):
                for k in range(K):
                    if k!=j:new_age[j,0]+=departures[k]*pi[k,j]
            state_age=new_age
        den=0.;joint=0.
        for k in range(K):
            weight=np.sum(state_age[k])
            den+=weight*np.exp(ret_loglik[t,k])
            joint+=weight*np.exp(joint_loglik[t,k])
        out[t]=np.log(max(den,1e-280))
        joint=max(joint,1e-280)
        for k in range(K):
            weight=np.exp(joint_loglik[t,k])/joint
            for a in range(D):state_age[k,a]*=weight
    return out


def _marginal_return_log_emissions(x,means,covs,df):
    from scipy.stats import t
    vals=np.asarray(x,float)[:,0]
    return np.column_stack([t.logpdf(vals,df,loc=means[k,0],
                  scale=np.sqrt(covs[k,0,0])) for k in range(len(means))])


def predict_return_logscore(x,pstr,limit_draws=16,segments=None):
    """Causal density of normalized NET return, conditioned on prior blocks only.

    Underlying multivariate emissions are used for FILTER UPDATES AFTER the
    observed block; features of block t never enter the prediction for t.
    Convert to raw-return log density by subtracting
    log(sqrt(ex_ante_variance_t) * TRAIN_normalizer.scale[0]).
    """
    x=np.asarray(x,float);sp=Spec(**pstr['spec'])
    segments=np.asarray([0,len(x)] if segments is None else segments,dtype=int)
    draws=pstr['draws'];stride=max(1,len(draws)//limit_draws)
    samples=[]
    for draw in draws[::stride][:limit_draws]:
        full=_student_log_emissions(x,draw['mu'],draw['cov'],sp.df)
        marginal=_marginal_return_log_emissions(x,draw['mu'],draw['cov'],sp.df)
        hazard=_duration_hazards(draw['duration_p'],sp.dmax,sp.duration_shape)
        scores=np.concatenate([_conditional_return_filter(full[lo:hi],marginal[lo:hi],
                            draw['pi'],draw['beta'],hazard)
                            for lo,hi in zip(segments[:-1],segments[1:])])
        samples.append(scores)
    return logsumexp(np.vstack(samples),axis=0)-np.log(len(samples))


def null_return_logscore(x,fit):
    return _marginal_return_log_emissions(np.asarray(x,float),fit['mu'][None,:],
                                          fit['cov'][None,:,:],fit['df'])[:,0]


def build_historical_30m_blocks(price_frame,*,particles=256,cal_limit=512):
    """Re-fit PRE-TEST VD18 from the shared raw price frame, then score future blocks.

    Imported lazily so self-tests work without GitHub source. Intended for
    VD14 make_targets + VD18 add_har_lags, using original 60/20/20 split.
    Baseline fitted on TRAIN + VALID and frozen before historical TEST.
    Only origins on the 00/30-minute grid are integrated; no per-origin arrays saved.
    """
    import pandas as pd
    from data_engineering.modeling import markov_lift_distributions as v14
    from data_engineering.modeling import student_t_return_markov as v15
    from data_engineering.modeling import markov_return_horizon_calibration as v16
    from data_engineering.modeling import markov_conditional_variance as v17
    from data_engineering.modeling import markov_har_variance as v18
    h=30
    tr,va,te=v18.complete(price_frame,h)
    te=te.loc[te.state_time.dt.minute.isin((0,30))].copy().reset_index(drop=True)
    if len(te)<100:raise ValueError('not enough complete 30m forecast origins')
    trainstate=price_frame.loc[(price_frame.split=='train')&price_frame.ready].copy()
    transitions=v14.fit_markov(trainstate)
    tr5,va5,_=v14._get_split(price_frame,5)
    kernels,_=v15.fit_return_kernel(tr5,va5,transitions)
    base,_=v16.prepare_calibrations(price_frame,kernels,transitions,
                     horizons=(h,),particles=particles,max_per_asset=cal_limit)
    base30=base[str(h)]['families']['AM_T_BOTH']
    har_fits={key:v18.fit_har(tr,h,v18.FEATURE_SETS[key]) for key in v18.FEATURE_SETS}
    overlays=v18.fit_har_overlays(price_frame,h,kernels,transitions,base30,har_fits,
                                 particles=particles,limit=cal_limit)
    means=np.zeros(len(te));ll=np.zeros(len(te))
    y=te.R_30.to_numpy(float)
    for asset,g in te.groupby('underlying',sort=True):
        where=np.flatnonzero((te.underlying==asset).to_numpy())
        states=g[['slow','gap']].to_numpy(float)
        har_pred=v18.predict_har(g,har_fits[overlays['selected_har']])
        for sl,variances in v16.variance_particles(states,transitions[asset],
                    kernels['AM_T'][asset],h//5,particles,v17._seed(asset,h),1024):
            weighted=v17.component_variances(variances,base30)
            z=v18._predict_var_ratio(har_pred[sl],variances,base30)
            correction=v17.delta_for('AM_T_VAR_ANCHOR',np.zeros((len(z),0)),z,
                                      overlays['blend_fit'])
            final=weighted*np.exp(correction[:,None])
            means[where[sl]]=final.mean(axis=1)
            ll[where[sl]]=v17.nll_from_components(final,y[where[sl]],np.zeros(len(z)))
    retseries=price_frame.set_index(['underlying','state_time'])['return_5m']
    returns6=np.column_stack([
        retseries.reindex(pd.MultiIndex.from_arrays([
            te.underlying.to_numpy(),te.state_time+pd.Timedelta(minutes=5*j)])).to_numpy(float)
        for j in range(1,7)])
    valid=np.isfinite(returns6).all(axis=1)&np.isfinite(means)&(means>0)
    if np.any(valid & (np.abs(returns6.sum(axis=1)-y)>1e-9)):
        raise AssertionError('source six-bar net return does not match target')
    te=te.loc[valid].copy().reset_index(drop=True)
    out=te[['underlying','state_time','R_30']].copy()
    out['pred_var']=means[valid];out['baseline_nll']=ll[valid]
    out['feature_1']=block_features(returns6[valid],means[valid])[:,0]
    out['feature_2']=block_features(returns6[valid],means[valid])[:,1]
    out['feature_3']=block_features(returns6[valid],means[valid])[:,2]
    return out,{'har_family':overlays['selected_har'],
                'blend_weight':overlays['blend_fit']['anchor_weight'],
                'base_scale':float(np.exp(base30['log_scale'])),
                'base_shape':float(base30['shape']),
                'historical_test_origin_count':len(out)}


def contiguous_segments(times,min_length=1):
    """Closed end indices; dropped short segments are returned explicitly.
    Timestamp gaps NEVER imply a continuous semi-Markov episode.
    """
    import pandas as pd
    t=pd.to_datetime(times,utc=True)
    ix=np.where((t[1:].to_numpy()-t[:-1].to_numpy())!=np.timedelta64(30,'m'))[0]+1
    edges=np.r_[0,ix,len(t)]
    kept=[(int(lo),int(hi)) for lo,hi in zip(edges[:-1],edges[1:]) if hi-lo>=min_length]
    return kept
