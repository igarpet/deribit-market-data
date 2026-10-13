"""VD14: seven-horizon Markovian volatility lift and conditional price distributions.

Independent, research-only comparison on native 5m Binance spot closes. No cross-branch
model imports, options pricing, trading claims, or database writes. For every origin,
features use <= t; targets use (t,t+h]; train/valid/test are disjoint UTC-day blocks.

Model definitions:
 M1 Gaussian, mean 0, variance from two-EMA log-realized-variance forecast.
 M2 Student-t, standardized to unit variance, same M1 scale, df selected on VALID.
 M3 centered two-Gaussian return mixture, constant rare-component probability.
 M4 same mixture with probability logistic in volatility phase.
 AM augmented Markov: causal fast/slow state proxies follow fitted bivariate VAR(1)
    (here diagonal autoregression with correlated innovations), while return diffusion
    and rare-shock likelihood are propagated by the resulting conditional log-variance
    moments and an estimated one-bar conditional mixture. This is an *approximate*
    integrated-variance return mixture, not a calibrated continuous-time jump diffusion.

The Gaussian mixture's 'rare component' is a statistical tail component; a latent
mixture label cannot identify an actual price discontinuity at 5-minute frequency.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logsumexp, ndtr
from scipy.stats import t as student_t

VERSION = 'VD14-0.1.0'
BAR_MIN = 5
HORIZONS = (5, 10, 15, 30, 60, 120, 240)
PRIMARY = 60
FAST_MIN = 30
SLOW_MIN = 240
TRAIN_FRAC = .60
VALID_FRAC = .20
MIN_HISTORY = 288
MIN_COVERAGE = .80
DF_GRID = (4., 6., 10., 20., 40.)
SHOCK_GRID = (3., 5., 8.)
BOOT = 1000
MODELS = ('M1_gaussian', 'M2_student_t', 'M3_constant_mix',
          'M4_conditional_mix', 'AM_markov_lift')
SAMPLE_END = pd.Timestamp('2026-09-19T00:00:00Z')
TAIL_Q = .95
MIN_TEST_DAYS = 30

@dataclass(frozen=True)
class Spec:
    version: str = VERSION
    horizons: tuple[int,...] = HORIZONS
    primary_horizon_min: int = PRIMARY
    bar_min: int = BAR_MIN
    fast_halflife_min: int = FAST_MIN
    slow_halflife_min: int = SLOW_MIN
    train_fraction: float = TRAIN_FRAC
    validation_fraction: float = VALID_FRAC
    min_history_bars: int = MIN_HISTORY
    min_history_coverage: float = MIN_COVERAGE
    student_df_grid: tuple[float,...] = DF_GRID
    shock_scale_grid: tuple[float,...] = SHOCK_GRID
    tail_quantile: float = TAIL_Q
    historical_window_exclusive_end: str = str(SAMPLE_END)
    historical_test_is_exploratory: bool = True
    note: str = 'Fixed historical period was previously inspected in VD12/VD13; not fresh confirmation.'


def registration():
    return asdict(Spec())


def load_returns(directory: Path, assets: tuple[str,...], *, end=SAMPLE_END,
                 min_days=365):
    frames, excluded = [], {}
    for asset in assets:
        path = Path(directory)/f'{asset}.csv.gz'
        if not path.exists():
            excluded[asset] = 'file missing'
            continue
        z = pd.read_csv(path, usecols=['state_time', 'close'])
        z.state_time = pd.to_datetime(z.state_time, utc=True)
        z = z.loc[z.state_time < end].sort_values('state_time')
        if z.state_time.duplicated().any():
            raise ValueError(f'Duplicate timestamps: {asset}')
        if ((z.state_time.dt.minute%5!=0)|(z.state_time.dt.second!=0)).any():
            raise ValueError(f'Off-grid timestamps: {asset}')
        if z.state_time.dt.floor('D').nunique() < min_days:
            excluded[asset] = 'insufficient history'
            continue
        c = pd.to_numeric(z.close,errors='coerce')
        valid = z.state_time.diff().eq(pd.Timedelta(minutes=5)) & c.gt(0) & c.shift().gt(0)
        ret = np.log(c/c.shift()).where(valid)
        frames.append(pd.DataFrame({'state_time':z.state_time.to_numpy(),
                       'underlying':asset,'return_5m':ret.to_numpy(float)}))
    if not frames:
        raise ValueError('No usable historical assets')
    return pd.concat(frames,ignore_index=True),excluded


def panelize(raw:pd.DataFrame, *, min_days=15):
    need={'state_time','underlying','return_5m'}
    if not need.issubset(raw):
        raise ValueError(f'Missing {need-set(raw)}')
    parts=[]
    for asset,g in raw.groupby('underlying',sort=True):
        z=g[['state_time','return_5m']].copy()
        z['state_time']=pd.to_datetime(z.state_time,utc=True)
        if z.state_time.duplicated().any():
            raise ValueError('Duplicate asset-time keys')
        z=z.sort_values('state_time').set_index('state_time')
        idx=pd.date_range(z.index.min(),z.index.max(),freq='5min',tz='UTC')
        if ((z.index.minute%5)!=0).any() or (z.index.second!=0).any():
            raise ValueError('Off-grid timestamp')
        z=z.reindex(idx)
        z['underlying']=asset
        z.index.name='state_time'
        parts.append(z.reset_index())
    p=pd.concat(parts,ignore_index=True)
    p['date']=p.state_time.dt.floor('D')
    dates=np.sort(p.date.unique())
    if len(dates)<min_days:
        raise ValueError('Too few dates')
    tr=int(len(dates)*TRAIN_FRAC)
    va=int(len(dates)*(TRAIN_FRAC+VALID_FRAC))
    if tr<2 or va<=tr or va>=len(dates):
        raise ValueError('Invalid day split')
    p['split']=np.where(p.date<dates[tr],'train',np.where(p.date<dates[va],'valid','test'))
    p['hour']=p.state_time.dt.hour
    p['slot']=p.state_time.dt.hour*12+p.state_time.dt.minute//5
    return p


def _ema_observed(x:np.ndarray,half_life_bars:float):
    """Causal observed-weight EMA; zero contributions are never imputed for missing data."""
    a=1-2**(-1/half_life_bars)
    finite=np.isfinite(x)
    num=pd.Series(np.where(finite,x,0.)).ewm(alpha=a,adjust=True).mean().to_numpy()
    mass=pd.Series(finite.astype(float)).ewm(alpha=a,adjust=True).mean().to_numpy()
    return np.divide(num,mass,out=np.full(len(x),np.nan),where=mass>1e-9)


def make_features(p:pd.DataFrame):
    f=p[['underlying','state_time','date','split','hour','slot','return_5m']].copy()
    for col in ['now','fast','slow','gap','slope15','past_coverage']:
        f[col]=np.nan
    f['ready']=False
    for asset,g in p.groupby('underlying',sort=False):
        ix=g.index.to_numpy()
        r=g.return_5m.to_numpy(float)
        train=g.split.to_numpy()=='train'
        finite=np.isfinite(r)
        sq=r[train & finite]**2
        sq=sq[sq>0]
        if len(sq)<200:
            raise ValueError(f'{asset}: <200 training variance estimates')
        eps=float(np.median(sq)*.001)
        raw_y=np.log(r*r+eps)
        hour=g.hour.to_numpy(int)
        centers=np.empty(24)
        fallback=np.nanmedian(raw_y[train])
        for hh in range(24):
            vals=raw_y[train & (hour==hh) & np.isfinite(raw_y)]
            centers[hh]=np.median(vals) if len(vals)>=20 else fallback
        y=raw_y-centers[hour]
        fast=_ema_observed(y,FAST_MIN/BAR_MIN)
        slow=_ema_observed(y,SLOW_MIN/BAR_MIN)
        t=np.arange(len(g))
        counts=np.r_[0,np.cumsum(finite.astype(int))]
        coverage=(counts[t]-counts[np.maximum(t-MIN_HISTORY,0)])/MIN_HISTORY
        recent=(counts[t]-counts[np.maximum(t-3,0)])>0
        ready=(t>=MIN_HISTORY)&(coverage>=MIN_COVERAGE)&recent&finite
        f.loc[ix,'now']=y
        f.loc[ix,'fast']=fast
        f.loc[ix,'slow']=slow
        f.loc[ix,'gap']=fast-slow
        f.loc[ix,'slope15']=fast-np.r_[np.full(3,np.nan),fast[:-3]]
        f.loc[ix,'past_coverage']=np.where(t>=MIN_HISTORY,coverage,np.nan)
        f.loc[ix,'ready']=ready
    f['abs_gap']=f.gap.abs()
    f['gap_slope']=f.gap*f.slope15
    return f


def make_targets(p:pd.DataFrame, features:pd.DataFrame, horizons=HORIZONS):
    """Construct forward net return and path variance using only (t,t+h], no gaps/split crossings."""
    f=features.copy()
    for horizon in horizons:
        if horizon%BAR_MIN:
            raise ValueError('Horizon is not multiple of 5m')
        n=horizon//BAR_MIN
        rr=np.full(len(p),np.nan)
        vv=np.full(len(p),np.nan)
        for asset,g in p.groupby('underlying',sort=False):
            idx=g.index.to_numpy(); r=g.return_5m.to_numpy(float)
            finite=np.isfinite(r); k=len(r); t=np.arange(k)
            cs=np.r_[0,np.cumsum(np.where(finite,r,0.))]
            cv=np.r_[0,np.cumsum(np.where(finite,r*r,0.))]
            cc=np.r_[0,np.cumsum(finite.astype(int))]
            end=np.minimum(t+n,k-1)
            same=g.split.to_numpy()[end]==g.split.to_numpy()[t]
            valid=(t+n<k)&(cc[end+1]-cc[t+1]==n)&same&finite
            rr[idx]=np.where(valid,cs[end+1]-cs[t+1],np.nan)
            vv[idx]=np.where(valid,cv[end+1]-cv[t+1],np.nan)
        f[f'R_{horizon}']=rr
        f[f'RV_{horizon}']=vv
    return f


def gaussian_nll(r,v):
    v=np.maximum(np.asarray(v,float),1e-14)
    return .5*(np.log(2*np.pi*v)+np.asarray(r,float)**2/v)


def gaussian_logpdf(r,v):
    return -gaussian_nll(r,v)


def normal_abs_prob(q,v):
    return np.maximum(2*ndtr(-q/np.sqrt(np.maximum(v,1e-14))),1e-12)


def t_logpdf(r,v,df):
    return student_t.logpdf(r/np.sqrt(v*(df-2)/df),df=df)-.5*np.log(v*(df-2)/df)


def t_abs_prob(q,v,df):
    return 2*student_t.sf(q/np.sqrt(v*(df-2)/df),df=df)


def mix_logpdf(r,v,p,shock_ratio):
    """Variance-matched centered two-Gaussian mixture (same TOTAL variance v)."""
    p=np.clip(np.asarray(p),1e-6,.50)
    d=1+p*(shock_ratio**2-1)
    narrow=v/d
    return np.logaddexp(np.log1p(-p)+gaussian_logpdf(r,narrow),
                        np.log(p)+gaussian_logpdf(r,narrow*shock_ratio**2))


def mix_abs_prob(q,v,p,shock_ratio):
    p=np.clip(np.asarray(p),1e-6,.50)
    d=1+p*(shock_ratio**2-1)
    narrow=v/d
    return (1-p)*normal_abs_prob(q,narrow)+p*normal_abs_prob(q,narrow*shock_ratio**2)


def fit_conditional_mix(r,v,phase,ratio,initial_p, *, features=('gap','slow'),
                        cap=150000,penalty=.02):
    """Direct penalized TRAIN likelihood; deterministic thinning controls compute cost.

    Uses total-variance-normalized mixture likelihood. Coefficients are NOT fitted
    to hindsight jump labels, which cannot be identified at native 5m resolution.
    """
    if len(r)>cap:
        sel=np.linspace(0,len(r)-1,cap,dtype=int)
    else:
        sel=np.arange(len(r))
    rr=np.asarray(r)[sel];vv=np.asarray(v)[sel]
    z=phase.iloc[sel][list(features)].to_numpy(float)
    center=z.mean(axis=0);std=np.maximum(z.std(axis=0),1e-8)
    x=np.column_stack([np.ones(len(z)),(z-center)/std])
    start=np.zeros(x.shape[1]);start[0]=np.log(initial_p/(1-initial_p))
    def lossgrad(beta):
        a=x@beta
        p=np.clip(expit(a),1e-6,.499999)
        q2=ratio**2
        d=1+p*(q2-1)
        narrow=np.maximum(vv/d,1e-14)
        ln=gaussian_logpdf(rr,narrow)
        lw=gaussian_logpdf(rr,narrow*q2)
        low=np.log1p(-p)+ln;wide=np.log(p)+lw
        ll=np.logaddexp(low,wide)
        resp=np.exp(wide-ll)
        # Derivative of each normal log-density w.r.t. p through D.
        f=.5*(q2-1)/d
        dln=f-.5*(rr*rr/vv)*(q2-1)
        dlw=f-.5*(rr*rr/(vv*q2))*(q2-1)
        dldp=(1-resp)*(dln-1/(1-p))+resp*(dlw+1/p)
        grad=-(x.T@(dldp*p*(1-p)))/len(rr)
        grad[1:]+=penalty*beta[1:]
        loss=-ll.mean()+0.5*penalty*np.sum(beta[1:]**2)
        return loss,grad
    solution=minimize(lossgrad,start,jac=True,method='L-BFGS-B',
                      options={'maxiter':75,'ftol':1e-10})
    return {'beta':solution.x.tolist(),'center':center.tolist(),
            'std':std.tolist(),'features':list(features),
            'train_loss':float(solution.fun),'converged':bool(solution.success)}


def conditional_mix_prob(fit,phase):
    x=phase[fit['features']].to_numpy(float)
    z=(x-np.asarray(fit['center']))/np.asarray(fit['std'])
    return np.clip(expit(np.column_stack([np.ones(len(z)),z])@fit['beta']),1e-5,.49)

def _design(tr:pd.DataFrame,other:pd.DataFrame,columns:list[str]):
    """Shared, train-only standardized design with asset controls and UTC harmonics."""
    names=sorted(tr.underlying.unique())
    data=tr[columns].to_numpy(float)
    center=data.mean(axis=0)
    std=data.std(axis=0)
    std=np.where(std>1e-10,std,1.)
    def mk(frame):
        a=[np.ones(len(frame))]
        for name in names[1:]:
            a.append((frame.underlying.to_numpy()==name).astype(float))
        ang=2*np.pi*frame.slot.to_numpy(float)/288
        a.extend((np.sin(ang),np.cos(ang)))
        z=(frame[columns].to_numpy(float)-center)/std
        a.extend(z[:,j] for j in range(len(columns)))
        return np.column_stack(a)
    return mk(tr),mk(other)


def _ridge(x,y,alpha=.02):
    gram=x.T@x
    penalty=np.eye(x.shape[1])*alpha*len(y)
    penalty[0,0]=0
    return np.linalg.solve(gram+penalty+np.eye(x.shape[1])*1e-10,x.T@y)


def _rv_scale(train,test,horizon):
    """M1..M4 use same two-EMA scale, train-only calibrated to net return variance."""
    cols=['now','fast','slow']
    xtr,xte=_design(train,test,cols)
    ytr=np.log(np.maximum(train[f'RV_{horizon}'].to_numpy(float),1e-16))
    b=_ridge(xtr,ytr)
    lrtr=xtr@b; lrte=xte@b
    rt=train[f'R_{horizon}'].to_numpy(float)
    v=np.zeros(len(test))
    calibration={}
    for asset in sorted(test.underlying.unique()):
        jt=np.where(train.underlying.to_numpy()==asset)[0]
        je=np.where(test.underlying.to_numpy()==asset)[0]
        sq=rt[jt]**2
        cap=np.quantile(sq,.995)
        reference=max(np.mean(np.minimum(sq,cap)),1e-12)
        ratio=float(np.clip(reference/np.mean(np.exp(np.clip(lrtr[jt],-40,5))),.01,100.))
        v[je]=np.maximum(np.exp(np.clip(lrte[je],-40,5))*ratio,1e-12)
        calibration[asset]=dict(variance_ratio=ratio,reference_var=reference)
    return v,calibration


def fit_mixture(train_r,train_v,valid_r,valid_v,train_phase,valid_phase,test_phase):
    """TRAIN-only mixture fitting; VALID chooses shock-width, TEST never tunes."""
    best=None
    if len(train_r)>200000:
        sel=np.linspace(0,len(train_r)-1,200000,dtype=int)
    else:
        sel=np.arange(len(train_r))
    for ratio in SHOCK_GRID:
        def objective(p):return -mix_logpdf(train_r[sel],train_v[sel],p,ratio).mean()
        scalar=minimize_scalar(objective,bounds=(.0002,.35),method='bounded',
                               options={'xatol':1e-6})
        prob=float(scalar.x)
        score=float(-mix_logpdf(valid_r,valid_v,prob,ratio).mean())
        if best is None or score<best[0]:best=(score,ratio,prob)
    _,ratio,p0=best
    cond_fit=fit_conditional_mix(train_r,train_v,train_phase,ratio,p0,
                                 features=('slow','gap','slope15'))
    p_cond=conditional_mix_prob(cond_fit,test_phase)
    return {'shock_ratio':float(ratio),'constant_p':float(p0),
            'conditional_p':p_cond,'phase_fit':cond_fit,
            'validation_nll':float(best[0])}


def fit_markov(train:pd.DataFrame):
    """Estimate two diagonal AR(1) state transitions and correlated innovations.

    Observed state proxies L=slow, F=fast-slow. These are EMA-filtered *observables*;
    their AR transitions are an empirical forecast approximation, not hidden true states.
    Adjacent train bars only, no gap interpolation, no cross-asset transitions.
    """
    fits={}
    for asset,g in train.groupby('underlying'):
        g=g.sort_values('state_time')
        l=g.slow.to_numpy(float)
        ff=g.gap.to_numpy(float)
        t=g.state_time.to_numpy()
        ok=np.isfinite(l[1:])&np.isfinite(l[:-1])&np.isfinite(ff[1:])&np.isfinite(ff[:-1])
        ok &= (t[1:]-t[:-1])==np.timedelta64(5,'m')
        if ok.sum()<1000:
            raise ValueError(f'{asset}: insufficient adjacent training factors')
        y=np.column_stack([l[1:][ok],ff[1:][ok]])
        x=np.column_stack([l[:-1][ok],ff[:-1][ok]])
        mean=x.mean(axis=0)
        xc=x-mean;yc=y-mean
        # Diagonal AR(1) allows correlated state shocks but is tractable to propagate.
        phi=np.sum(xc*yc,axis=0)/np.maximum(np.sum(xc*xc,axis=0),1e-12)
        phi=np.clip(phi,.01,.9999)
        innovation=yc-xc*phi
        cov=np.cov(innovation,rowvar=False)
        eig=np.linalg.eigvalsh(cov)
        if eig.min()<1e-8:
            cov+=np.eye(2)*(1e-8-eig.min())
        fits[asset]={'mean':mean.tolist(),'phi':phi.tolist(),'innovation_cov':cov.tolist(),
                     'transition_rows':int(ok.sum())}
    return fits


def _markov_rv_path(state:np.ndarray, fit:dict, steps:int):
    """Conditional first moment of integrated RV under Gaussian AR(1) log states.

    E[exp(L_j+F_j)] is analytical and includes forecast state innovation variance.
    A horizon-dependent ratio is calibrated on TRAIN to net return variance.
    """
    mean=np.asarray(fit['mean'])
    phi=np.asarray(fit['phi'])
    Q=np.asarray(fit['innovation_cov'])
    j=np.arange(1,steps+1)[:,None]
    powj=phi[None,:]**j
    m=mean[None,:]+(state[:,None,:]-mean[None,None,:])*powj[None,:,:]
    # accumulated covariance of state shocks at each future step
    accum=np.zeros(steps)
    for i in range(2):
        for k in range(2):
            denom=1-phi[i]*phi[k]
            accum+=Q[i,k]*(1-(phi[i]*phi[k])**np.arange(1,steps+1))/denom
    return np.sum(np.exp(np.clip(m.sum(axis=2)+.5*accum[None,:],-20,15)),axis=1)


def markov_variances(tr:pd.DataFrame,va:pd.DataFrame,te:pd.DataFrame,fit:dict,horizon:int):
    """Train-only net-RV calibration; no fitting or scale selection on TEST/VALID."""
    steps=horizon//BAR_MIN
    vt=np.zeros(len(tr));vv=np.zeros(len(va));ve=np.zeros(len(te))
    cal={}
    logrve=np.zeros(len(te))
    for asset,f in fit.items():
        def state(g):
            z=g[g.underlying==asset]
            return np.column_stack([z.slow.to_numpy(float),z.gap.to_numpy(float)])
        mt=_markov_rv_path(state(tr),f,steps)
        mv=_markov_rv_path(state(va),f,steps)
        me=_markov_rv_path(state(te),f,steps)
        tgt=tr[tr.underlying==asset][f'R_{horizon}'].to_numpy(float)
        sq=tgt*tgt;ref=np.mean(np.minimum(sq,np.quantile(sq,.995)))
        ratio=float(np.clip(ref/np.mean(mt),1e-10,1e-2))
        vt[tr.underlying==asset]=np.maximum(mt*ratio,1e-12)
        vv[va.underlying==asset]=np.maximum(mv*ratio,1e-12)
        ve[te.underlying==asset]=np.maximum(me*ratio,1e-12)
        rvtr=tr[tr.underlying==asset][f'RV_{horizon}'].to_numpy(float)
        rv_offset=float(np.mean(np.log(np.maximum(rvtr,1e-16))-np.log(np.maximum(mt,1e-16))))
        logrve[te.underlying==asset]=np.log(np.maximum(me,1e-16))+rv_offset
        cal[asset]={'net_variance_scale':ratio,'log_rv_offset':rv_offset}
    return vt,vv,ve,cal,logrve


def _get_split(frame,horizon):
    cols=['now','fast','slow','gap','slope15','abs_gap','gap_slope',f'R_{horizon}',f'RV_{horizon}']
    valid=frame.ready.to_numpy(bool)&np.isfinite(frame[cols].to_numpy(float)).all(axis=1)
    return tuple(frame.loc[valid&(frame.split==split)].copy() for split in ('train','valid','test'))


def _tail_score(r,q,prob):
    p=np.clip(prob,1e-9,1-1e-9)
    y=(np.abs(r)>=q).astype(float)
    return -(y*np.log(p)+(1-y)*np.log1p(-p)),(y-p)**2,y


def _markov_hazard_prob(te:pd.DataFrame, transitions:dict, onebar_hazard:dict, steps:int):
    """Propagate a 5m conditional tail-mixture hazard through Markov state means.

    p_j depends only on conditional E[L_{t+j}|Z_t], E[F_{t+j}|Z_t]. The
    integrated at-least-one-event probability is 1-prod(1-p_j). This is an
    approximation; correlated jumps and multiple broad marks are not represented.
    """
    out=np.zeros(len(te))
    for asset,g in te.groupby('underlying',sort=False):
        idx=te.underlying.to_numpy()==asset
        trans=transitions[asset];haz=onebar_hazard[asset]
        states=g[['slow','gap']].to_numpy(float)
        center=np.asarray(trans['mean']);phi=np.asarray(trans['phi'])
        # p computed for future post-origin steps. Horizon 5m has one event trial.
        log_survival=np.zeros(len(g))
        for j in range(1,steps+1):
            state=center+(states-center)*(phi**j)
            ff=pd.DataFrame({'slow':state[:,0],'gap':state[:,1]})
            pj=conditional_mix_prob(haz,ff)
            log_survival+=np.log1p(-pj)
        out[idx]=np.clip(-np.expm1(log_survival),1e-6,.49)
    return out


def fit_onebar_hazard(frame:pd.DataFrame,transitions:dict):
    """Fit AM's five-minute tail mechanism ONCE, then reuse at every horizon."""
    tr,va,te=_get_split(frame,5)
    mtr,mva,_,_,_=markov_variances(tr,va,te,transitions,5)
    # M3 selects broad width with validation, using Markov-calibrated 5m scale.
    rt=tr.R_5.to_numpy(float);rv=va.R_5.to_numpy(float)
    best=None
    idx=np.linspace(0,len(rt)-1,min(150000,len(rt)),dtype=int)
    for ratio in SHOCK_GRID:
        sol=minimize_scalar(lambda p:-mix_logpdf(rt[idx],mtr[idx],p,ratio).mean(),
             bounds=(.0002,.35),method='bounded')
        vscore=-mix_logpdf(rv,mva,float(sol.x),ratio).mean()
        if best is None or vscore<best[0]:best=(vscore,ratio,float(sol.x))
    _,ratio,p0=best
    by_asset={}
    for asset,g in tr.groupby('underlying'):
        ix=tr.underlying.to_numpy()==asset
        by_asset[asset]=fit_conditional_mix(rt[ix],mtr[ix],tr.loc[ix],ratio,p0,
                                          features=('slow','gap'),cap=90000)
    return {'ratio_5m':float(ratio),'p0_5m':float(p0),
            'models':by_asset}


def _fit_single_horizon(frame:pd.DataFrame,horizon:int,markov:dict,am_hazard:dict):
    tr,va,te=_get_split(frame,horizon)
    if len(tr)<2000 or len(va)<200 or len(te)<200:
        raise ValueError(f'Insufficient support at {horizon}m')
    vtr,cal=_rv_scale(tr,tr,horizon)
    vva,_=_rv_scale(tr,va,horizon)
    vte,_=_rv_scale(tr,te,horizon)
    rt=tr[f'R_{horizon}'].to_numpy(float)
    rv=va[f'R_{horizon}'].to_numpy(float)
    re=te[f'R_{horizon}'].to_numpy(float)
    df=min(DF_GRID,key=lambda d:-t_logpdf(rv,vva,d).mean())
    mix=fit_mixture(rt,vtr,rv,vva,tr,va,te)
    broad=mix['shock_ratio']
    _,_,mvte,mcal,am_logrv=markov_variances(tr,va,te,markov,horizon)
    # AM is a separate Markov-consistent 5m fitted mixture, NOT a 60m
    # logistic model disguised as an intensity. The shock amplitude is
    # diluted over the h-bar integrated diffusion scale.
    am_p=_markov_hazard_prob(te,markov,am_hazard['models'],horizon//BAR_MIN)
    am_ratio=np.sqrt(1+(am_hazard['ratio_5m']**2-1)/(horizon//BAR_MIN))
    forecasts={
      'M1_gaussian':(gaussian_logpdf(re,vte),'gauss',vte,None,broad),
      'M2_student_t':(t_logpdf(re,vte,df),'student',vte,df,broad),
      'M3_constant_mix':(mix_logpdf(re,vte,mix['constant_p'],broad),'mix',vte,mix['constant_p'],broad),
      'M4_conditional_mix':(mix_logpdf(re,vte,mix['conditional_p'],broad),'mix',vte,mix['conditional_p'],broad),
      'AM_markov_lift':(mix_logpdf(re,mvte,am_p,am_ratio),'mix',mvte,am_p,am_ratio)
    }
    threshold={asset:float(np.quantile(np.abs(tr.loc[tr.underlying==asset,f'R_{horizon}']),TAIL_Q))
               for asset in tr.underlying.unique()}
    q=te.underlying.map(threshold).to_numpy(float)
    # Compare future log realized variance on the same complete origin support.
    # M1-M4 share the calibrated E forecast. AM uses conditional AR state propagation.
    e_logrv=np.zeros(len(te))
    for asset,fit in cal.items():
        idx=(te.underlying==asset).to_numpy()
        e_logrv[idx]=np.log(np.maximum(vte[idx],1e-16))-np.log(fit['variance_ratio'])
    realized_logrv=np.log(np.maximum(te[f'RV_{horizon}'].to_numpy(float),1e-16))
    scores=[];preds=[]
    for name,(logp,kind,v,extra,ratio) in forecasts.items():
        if kind=='gauss':tailprob=normal_abs_prob(q,v)
        elif kind=='student':tailprob=t_abs_prob(q,v,extra)
        else:tailprob=mix_abs_prob(q,v,extra,ratio)
        tail_ll,tail_brier,ytail=_tail_score(re,q,tailprob)
        # Exact central-95 interval computed from the *same distribution*.
        # Instead of a Gaussian proxy, use its predictive P(|R|>z) at a
        # distribution-specific numerical 97.5% upper quantile.
        # We export a distribution-free PIT-style two-sided probability of the
        # observed return magnitude: a value below .05 is 95% non-coverage.
        mag=np.abs(re)
        if kind=='gauss':p_at_real=normal_abs_prob(mag,v)
        elif kind=='student':p_at_real=t_abs_prob(mag,v,extra)
        else:p_at_real=mix_abs_prob(mag,v,extra,ratio)
        miss=(p_at_real<.05).astype(float)
        loss=-logp
        rv_loss=(realized_logrv-(am_logrv if name=='AM_markov_lift' else e_logrv))**2
        preds.append(pd.DataFrame({'origin_id':te.index.to_numpy(),
           'date':te.date.to_numpy(),'horizon_min':horizon,
           'model':name,'nll':loss,'rv_log_mse':rv_loss,'tail_logloss':tail_ll,'tail_brier':tail_brier}))
        scores.append({'horizon_min':horizon,'model':name,'n_origins':len(te),
            'n_days':int(te.date.nunique()),'nll':float(loss.mean()),
            'rv_log_mse':float(rv_loss.mean()),
            'tail_logloss':float(tail_ll.mean()),'tail_brier':float(tail_brier.mean()),
            'tail_observed':float(ytail.mean()),'tail_predicted':float(tailprob.mean()),
            'interval_miss95':float(miss.mean())})
    metadata={'horizon_min':horizon,'student_df':float(df),'selected_shock_ratio':broad,
      'M3_constant_probability':mix['constant_p'],
      'M4_phase_fit':mix['phase_fit'],
      'markov_variance_calibration':mcal,
      'AM_shock_sd_ratio_for_horizon':float(am_ratio),
      'q95_train_by_asset':threshold,
      'train_origins':len(tr),'validation_origins':len(va),'test_origins':len(te)}
    pred=pd.concat(preds,ignore_index=True)
    pred['model']=pd.Categorical(pred.model,categories=MODELS)
    return pd.DataFrame(scores),pred,metadata


PAIRS = (
    ('M2_student_t','M1_gaussian'),
    ('M3_constant_mix','M1_gaussian'),
    ('M3_constant_mix','M2_student_t'),
    ('M4_conditional_mix','M3_constant_mix'),
    ('AM_markov_lift','M4_conditional_mix'),
    ('AM_markov_lift','M2_student_t'),
    ('AM_markov_lift','M1_gaussian'),
)
METRICS = ('nll','rv_log_mse','tail_logloss','tail_brier')


def run_all(frame:pd.DataFrame,*,horizons=HORIZONS,B=BOOT,store_predictions=False):
    """Stream horizon outputs to stay memory-bounded on ~1.3m bars.

    By default returns only compact scores/paired bootstrap/parameter records;
    9m+ origin-level predictions are not retained. Optional store_predictions is
    for tiny synthetic tests only, never for full user historical datasets.
    """
    import gc
    tr=frame.loc[(frame.split=='train')&frame.ready].copy()
    markov=fit_markov(tr)
    hazard=fit_onebar_hazard(frame,markov)
    del tr
    scores=[];boots=[];preds=[];meta={}
    for h in horizons:
        s,p,m=_fit_single_horizon(frame,h,markov,hazard)
        scores.append(s);meta[str(h)]=m
        for i,(a,b) in enumerate(PAIRS):
            for k,metric in enumerate(METRICS):
                boots.append(paired_day_bootstrap(p,a,b,h,metric,B=B,seed=1401+100*h+7*i+k))
        if store_predictions:
            preds.append(p)
        del p
        gc.collect()
    models=pd.concat(scores,ignore_index=True)
    bs=pd.DataFrame(boots)
    metadata={'markov_state':markov,'AM_fitted_5m_hazard':hazard,'fits':meta}
    if store_predictions:
        return models,bs,metadata,pd.concat(preds,ignore_index=True)
    return models,bs,metadata


def paired_day_bootstrap(pred:pd.DataFrame,a:str,b:str,horizon:int,metric='nll',B=BOOT,seed=1401):
    x=pred.loc[(pred.horizon_min==horizon)&(pred.model==a)].set_index('origin_id')
    y=pred.loc[(pred.horizon_min==horizon)&(pred.model==b)].set_index('origin_id')
    ix=x.index.intersection(y.index)
    if not len(ix) or len(ix)!=len(x) or len(ix)!=len(y):
        raise ValueError('Models lack identical complete-case origins')
    x=x.loc[ix];y=y.loc[ix]
    if not np.array_equal(x.date.to_numpy(),y.date.to_numpy()):
        raise ValueError('Model UTC days differ')
    d=x[metric].to_numpy(float)-y[metric].to_numpy(float)
    days,index=np.unique(x.date.to_numpy(),return_inverse=True)
    sums=np.bincount(index,weights=d,minlength=len(days))
    count=np.bincount(index,minlength=len(days))
    rng=np.random.default_rng(seed)
    draw=rng.integers(len(days),size=(B,len(days)))
    samples=sums[draw].sum(axis=1)/count[draw].sum(axis=1)
    return {'horizon_min':horizon,'comparison':f'{a} - {b}','metric':metric,
      'delta':float(d.mean()),'q025':float(np.quantile(samples,.025)),
      'q975':float(np.quantile(samples,.975)),'test_days':len(days),'origins':len(d)}


def decision(score:pd.DataFrame,bs:pd.DataFrame):
    primary=bs.loc[(bs.horizon_min==PRIMARY)&(bs.metric=='nll')]
    def beats(a,b):
        s=primary.loc[primary.comparison==f'{a} - {b}']
        return bool(len(s)==1 and s.delta.iloc[0]<0 and s.q975.iloc[0]<0)
    flags={
      't_better_than_gaussian':beats('M2_student_t','M1_gaussian'),
      'constant_mix_better_than_gaussian':beats('M3_constant_mix','M1_gaussian'),
      'constant_mix_better_than_t':beats('M3_constant_mix','M2_student_t'),
      'conditional_mix_better_than_constant':beats('M4_conditional_mix','M3_constant_mix'),
      'markov_better_than_conditional':beats('AM_markov_lift','M4_conditional_mix'),
      'markov_better_than_t':beats('AM_markov_lift','M2_student_t'),
    }
    ndays=int(score[score.horizon_min==PRIMARY].n_days.min())
    return {'status':'INSUFFICIENT_DATA' if ndays<MIN_TEST_DAYS else 'HISTORICAL_EXPLORATORY',
      'n_test_days':ndays,'primary_horizon_min':PRIMARY,'flags':flags,
      'strong_markov_candidate':flags['markov_better_than_conditional'] and flags['markov_better_than_t'],
      'not_prospective':True,
      'interpretation':'Historical 2024-2026 data was previously examined. All positive flags require fresh prospective confirmation.'}
