"""CS01: continuous 5-minute signed-return streaks and honest next-sign forecasts.

Input: observed Binance log close-to-close returns. No fixed-length windows.
Zero returns end a directional run. Gaps/missing returns censor (not complete)
run episodes. Completed-run distribution and in-progress opportunities differ.
Train models only on observations with an immediately following nonzero return
in the SAME chronological split. Use no out-of-sample labels to fit forecasts.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

VERSION = 'CS01-0.1.0'
STEP = pd.Timedelta(minutes=5)

@dataclass(frozen=True)
class Specification:
    version: str = VERSION
    assets: tuple = ('BTC', 'ETH', 'SOL', 'XRP')
    cutoff_utc: str = '2026-10-14T00:00:00Z'  # full UTC days only; fixed before evaluation
    train_fraction: float = 0.60
    valid_fraction: float = 0.20
    length_prior: float = 40.0  # shrink exact-k hazard to first-order Markov
    logistic_l2: float = .015
    bootstrap_reps: int = 1000
    permutation_reps: int = 100
    seed: int = 2601
    status: str = 'HISTORICAL_EXPLORATORY'


def registration():
    return asdict(Specification())


def _checked(g: pd.DataFrame):
    need = {'underlying', 'state_time', 'return_5m'}
    if not need.issubset(g.columns):
        raise ValueError(f'Missing {need-set(g.columns)}')
    g = g[['underlying','state_time','return_5m']].copy()
    g['state_time'] = pd.to_datetime(g.state_time, utc=True)
    g = g.sort_values(['underlying', 'state_time']).reset_index(drop=True)
    if g.duplicated(['underlying','state_time']).any():
        raise ValueError('duplicate asset/timestamp')
    if ((g.state_time.dt.minute % 5 != 0) | g.state_time.dt.second.ne(0)).any():
        raise ValueError('timestamps must be aligned on the 5m grid')
    return g


def _labels_and_age(t, returns):
    r=np.asarray(returns,float)
    n=len(r)
    s=np.zeros(n,dtype=np.int8)
    good=np.isfinite(r)
    s[good]=np.sign(r[good]).astype(np.int8)
    dt=(pd.to_datetime(t,utc=True).to_numpy()[1:]-pd.to_datetime(t,utc=True).to_numpy()[:-1])==np.timedelta64(5,'m')
    contiguous=np.r_[False,dt]
    idx=np.arange(n,dtype=np.int64)
    beginnings=(s==0)|(s!=np.r_[0,s[:-1]])|(~contiguous)
    laststart=np.maximum.accumulate(np.where(beginnings,idx,0))
    age=np.where(s!=0,idx-laststart+1,0).astype(np.int64)
    return s,age,good,contiguous


def build_events_and_runs(raw: pd.DataFrame):
    """One row per nonzero current bar; episode lengths have no arbitrary cap.

    `next_valid` means the next observed bar is exactly +5m, with a finite
    return. `next_nonzero` additionally excludes zero-return target bars.
    Fully observed sign-to-opposite or sign-to-zero endings are completed;
    file/gap/end-of-series boundaries are right censored.
    """
    raw=_checked(raw)
    event_parts=[];run_parts=[];quality=[]
    for asset,g in raw.groupby('underlying',sort=True):
        g=g.reset_index(drop=True)
        t=g.state_time
        r=g.return_5m.to_numpy(float)
        s,k,good,adj=_labels_and_age(t,r)
        n=len(g)
        next_adj=np.r_[adj[1:],False]
        next_s=np.r_[s[1:],0]
        next_good=np.r_[good[1:],False]
        next_valid=next_adj & next_good & good
        good_current=s!=0
        outcomes=np.where(~next_valid,'censored',np.where(next_s==0,'zero',np.where(next_s==s,'continue','reverse')))
        select=good_current
        events=pd.DataFrame({
            'underlying':asset,'state_time':t[select].to_numpy(),
            'sign':s[select], 'length':k[select],
            'signed_streak':(s*k)[select], 'next_sign':np.where(next_valid,next_s,0)[select],
            'next_valid':next_valid[select],
            'next_nonzero':(next_valid & (next_s!=0))[select],
            'continuation':(next_valid & (next_s==s))[select],
            'reversal':(next_valid & (next_s==-s))[select],
            'next_zero':(next_valid & (next_s==0))[select],
            'outcome':outcomes[select],
        })
        events['date']=pd.to_datetime(events.state_time,utc=True).dt.floor('D')
        event_parts.append(events)
        ended=select & (outcomes!='continue')
        ix=np.flatnonzero(ended)
        run_parts.append(pd.DataFrame({
            'underlying':asset,'end_time':t.iloc[ix].to_numpy(),
            'sign':s[ix],'length':k[ix], 'signed_streak':(s*k)[ix],
            'end_reason':outcomes[ix],
            'completed':np.isin(outcomes[ix],['reverse','zero']),
        }))
        quality.append({'asset':asset,'rows':n,'valid_returns':int(good.sum()),'up':int((s==1).sum()),
                        'down':int((s==-1).sum()),'zero':int((good & (s==0)).sum()),
                        'missing':int((~good).sum()),'discontinuities':int(np.count_nonzero(~adj[1:])),
                        'first_date':str(t.min()),'last_date':str(t.max())})
    return pd.concat(event_parts,ignore_index=True),pd.concat(run_parts,ignore_index=True),pd.DataFrame(quality)


def add_chronological_splits(events:pd.DataFrame,train_fraction=.6,valid_fraction=.2):
    """Split by complete UTC dates; exclude forecasting targets crossing split."""
    q=events.copy().sort_values(['underlying','state_time']).reset_index(drop=True)
    dates=np.sort(q.date.unique())
    if len(dates)<60:raise ValueError('need at least 60 observed UTC days')
    i=int(len(dates)*train_fraction);j=int(len(dates)*(train_fraction+valid_fraction))
    if not 0<i<j<len(dates):raise ValueError('invalid fractions')
    q['split']=np.where(q.date<dates[i],'train',np.where(q.date<dates[j],'valid','test'))
    # An observed NEXT return may be from a later split: don't train across it.
    next_date=(q.state_time+STEP).dt.floor('D')
    next_split=np.where(next_date<dates[i],'train',np.where(next_date<dates[j],'valid','test'))
    q['scorable']=q.next_nonzero & (q.split==next_split)
    # Bars after gap are present but next_nonzero already has gap guard.
    return q, {'train_before':str(dates[i]),'valid_before':str(dates[j]),'n_dates':int(len(dates))}


def _wilson(success,n):
    if n<=0:return np.nan,np.nan
    z=norm.ppf(.975);p=success/n;den=1+z*z/n
    c=(p+z*z/(2*n))/den
    rad=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return float(max(0,c-rad)),float(min(1,c+rad))


def distribution_tables(events,runs,*,base_rates=None):
    """All observed lengths preserved; report immediate-next exposure separately.

    Counts of completed runs are not the counts of reached length k. Geometric
    expected counts condition on observed run starts and constant i.i.d. sign
    probability, with boundary censoring omitted in the approximation.
    """
    e=events.copy();e=e.loc[e.sign.isin([-1,1])]
    rr=runs.copy()
    comp=rr.loc[rr.completed].groupby(['underlying','sign','length']).size().rename('n_completed')
    cens=rr.loc[~rr.completed].groupby(['underlying','sign','length']).size().rename('n_right_censored')
    out=pd.concat([comp,cens],axis=1).fillna(0).astype(int).reset_index()
    starts=rr.groupby(['underlying','sign']).size().to_dict()
    probabilities={}
    for asset,g in e.groupby('underlying'):
        b=g.loc[g.next_valid]
        total=len(b)
        for sign in [-1,1]:
            key=(asset,sign)
            probabilities[key]=(b.next_sign.eq(sign).sum()/total) if total else np.nan
    if base_rates:probabilities.update(base_rates)
    # Also compare a first-order sign Markov reference: under constant sign-
    # specific continuation hazard its completed-run lengths are geometric.
    markov_prob={}
    for (asset,sign),g in e.groupby(['underlying','sign']):
        eligible=g.loc[g.next_valid]
        markov_prob[(asset,sign)]=float(eligible.continuation.mean()) if len(eligible) else np.nan
    out['run_starts']=out.apply(lambda row:starts.get((row.underlying,row.sign),0),axis=1)
    out['iid_same_probability']=out.apply(lambda row:probabilities.get((row.underlying,row.sign),np.nan),axis=1)
    out['markov_same_probability']=out.apply(lambda row:markov_prob.get((row.underlying,row.sign),np.nan),axis=1)
    out['expected_completed_iid_approx']=out.run_starts*(1-out.iid_same_probability)*np.power(out.iid_same_probability,out.length-1)
    out['expected_completed_markov_approx']=out.run_starts*(1-out.markov_same_probability)*np.power(out.markov_same_probability,out.length-1)
    out['signed_streak']=(out.sign*out.length).astype(int)
    e2=e.copy()
    e2['next_known_nonzero']=e2.next_nonzero.astype(int)
    e2['n_next_observed']=e2.next_valid.astype(int)
    e2['n_continue']=e2.continuation.astype(int)
    e2['n_reverse']=e2.reversal.astype(int)
    e2['n_zero']=e2.next_zero.astype(int)
    table=e2.groupby(['underlying','sign','length'],as_index=False).agg(
        n_reached=('length','size'),n_next_observed=('n_next_observed','sum'),
        n_next_nonzero=('next_known_nonzero','sum'),n_continue=('n_continue','sum'),
        n_reverse=('n_reverse','sum'),n_zero=('n_zero','sum'))
    table['n_censored']=table.n_reached-table.n_next_observed
    table['p_continue_given_next_nonzero']=table.n_continue/table.n_next_nonzero.replace(0,np.nan)
    table['p_continue_including_zero']=table.n_continue/table.n_next_observed.replace(0,np.nan)
    intervals=[_wilson(int(x.n_continue),int(x.n_next_nonzero)) for x in table.itertuples()]
    table['wilson95_low']=[p[0] for p in intervals];table['wilson95_high']=[p[1] for p in intervals]
    table['signed_streak']=table.sign*table.length
    return out.sort_values(['underlying','sign','length']).reset_index(drop=True),table.sort_values(['underlying','sign','length']).reset_index(drop=True)


def fit_direction_models(train:pd.DataFrame,*,pseudo=40.,ridge=.015):
    """Models predict P(next nonzero sign == current sign).

    No cap on observed lengths. Exact-k model shrinks rare lengths back to
    first-order Markov. Logistic duration model smooths log(length) for power.
    """
    tr=train.loc[train.scorable].copy()
    if len(tr)<100:raise ValueError('too few training transitions')
    sign=tr.sign.to_numpy(int);age=tr.length.to_numpy(int)
    y=tr.continuation.to_numpy(float)
    p_up=(float(np.sum(tr.next_sign==1))+.5)/(len(tr)+1.)
    markov={}
    for direction in (-1,1):
        m=sign==direction;n=int(m.sum());success=int(y[m].sum())
        markov[direction]=(success+.5)/(n+1.) if n else .5
    exact={}
    for (s,k),g in tr.groupby(['sign','length']):
        n=len(g);count=int(g.continuation.sum())
        exact[(int(s),int(k))]=(count+pseudo*markov[int(s)])/(n+pseudo)
    logk=np.log(age)
    plus=(sign==1).astype(float)
    X=np.column_stack([np.ones(len(tr)),plus,logk,plus*logk])
    # Start from sign-specific Markov hazards; penalize slope coefficients.
    logit=lambda p:np.log(p/(1-p))
    b0=np.array([logit(markov[-1]), logit(markov[1])-logit(markov[-1]),0.,0.])
    def objective(beta):
        logits=X@beta
        nll=np.logaddexp(0,logits).sum()-y@logits
        reg=ridge*len(y)*(beta[2]**2+beta[3]**2)
        grad=X.T@(expit(logits)-y)
        grad[2:]+=2*ridge*len(y)*beta[2:]
        return (nll+reg)/len(y),grad/len(y)
    opt=minimize(objective,b0,jac=True,method='L-BFGS-B',options={'maxiter':150})
    if not np.isfinite(opt.fun):raise RuntimeError('logistic failed')
    return {'p_up':p_up,'markov':markov,'exact':exact,'pseudo':pseudo,
            'logistic_coefficients':opt.x.tolist(),'training_rows':len(tr),'converged':bool(opt.success)}


def predict_direction(frame:pd.DataFrame,fit:dict):
    s=frame.sign.to_numpy(int);length=frame.length.to_numpy(int)
    marc=np.array([fit['markov'][int(x)] for x in s])
    exact=np.array([fit['exact'].get((int(a),int(b)),fit['markov'][int(a)]) for a,b in zip(s,length)])
    plus=(s==1).astype(float);logk=np.log(length)
    X=np.column_stack([np.ones(len(s)),plus,logk,plus*logk])
    duration=expit(X@np.asarray(fit['logistic_coefficients']))
    # Convert continuation probability into next-up probability.
    def to_up(p):return np.where(s==1,p,1-p)
    return pd.DataFrame({'random_50':np.full(len(frame),.5),
        'iid_empirical':np.full(len(frame),fit['p_up']),
        'markov_sign':to_up(marc),'streak_exact_shrunk':to_up(exact),
        'streak_log_duration':to_up(duration)},index=frame.index)


def forecast_scores(eval_events:pd.DataFrame,preds:pd.DataFrame):
    ev=eval_events.loc[eval_events.scorable].copy()
    if len(ev)!=len(preds):raise ValueError('prediction/target length mismatch')
    targets=(ev.next_sign.to_numpy(int)==1).astype(float)
    t=pd.to_datetime(ev.state_time,utc=True)
    rows=[];perday=[]
    for method in preds.columns:
        p=np.clip(preds[method].to_numpy(float),1e-10,1-1e-10)
        ll=-(targets*np.log(p)+(1-targets)*np.log1p(-p))
        b=(p-targets)**2
        rows.append({'model':method,'n':len(p),'logloss':float(ll.mean()),'brier':float(b.mean()),
                     'forecast_up_average':float(p.mean()),'realized_up_rate':float(targets.mean())})
        perday.append(pd.DataFrame({'date':t.dt.floor('D').to_numpy(), 'model':method,'logloss':ll,'brier':b}))
    return pd.DataFrame(rows),pd.concat(perday,ignore_index=True)


def bootstrap_comparison(per_origin:pd.DataFrame, *, model='streak_log_duration',reference='markov_sign',
                         reps=1000,seed=2601):
    """Paired resampling of entire UTC days (equal day weight)."""
    day=per_origin.groupby(['date','model'])[['logloss','brier']].mean().unstack('model')
    a=np.column_stack([day[(m,model)]-day[(m,reference)] for m in ('logloss','brier')])
    rng=np.random.default_rng(seed)
    out=[]
    for j,metric in enumerate(('logloss','brier')):
        samp=a[rng.integers(0,len(a),size=(reps,len(a))),j].mean(axis=1)
        out.append({'model':model,'reference':reference,'metric':metric,
                    'mean_day_difference':float(a[:,j].mean()),
                    'ci025':float(np.quantile(samp,.025)),'ci975':float(np.quantile(samp,.975)),
                    'n_days':len(a)})
    return pd.DataFrame(out)


def _runs_for_null(s,adj):
    n=len(s)
    idx=np.arange(n)
    starts=(s==0)|(s!=np.r_[0,s[:-1]])|(~adj)
    origin=np.maximum.accumulate(np.where(starts,idx,0))
    age=np.where(s!=0,idx-origin+1,0)
    end=np.r_[~adj[1:],True] | (s!=np.r_[s[1:],0])
    ages=age[end & (s!=0)];signs=s[end & (s!=0)]
    return signs,ages


def iid_permutation_null(raw:pd.DataFrame,*,reps=100,seed=2601,tail_min=8):
    """Keep zero/missing/gaps fixed; shuffle only up/down labels per asset.

    Descriptive null does not preserve serially varying drift or clock effects.
    Returns per-asset observation, Monte Carlo bands and empirical rank.
    """
    raw=_checked(raw);rng=np.random.default_rng(seed);rows=[]
    for asset,g in raw.groupby('underlying'):
        s,_,_,adj=_labels_and_age(g.state_time,g.return_5m.to_numpy(float))
        where=np.flatnonzero(s!=0)
        if len(where)<100:continue
        def stats(z):
            ss,ll=_runs_for_null(z,adj)
            return [int(ll.max(initial=0)),int(np.sum(ll>=tail_min)),
                    int(np.sum((ss==1)&(ll>=tail_min))),int(np.sum((ss==-1)&(ll>=tail_min)))]
        obs=np.array(stats(s))
        sim=np.empty((reps,4),float)
        for i in range(reps):
            z=s.copy();z[where]=rng.permutation(s[where]);sim[i]=stats(z)
        for j,metric in enumerate(('max_run','completed_or_censored_runs_ge_tail',
                                  'up_runs_ge_tail','down_runs_ge_tail')):
            rows.append({'asset':asset,'metric':metric,'tail_min':tail_min,'observed':int(obs[j]),
                         'null_05':float(np.quantile(sim[:,j],.05)),
                         'null_median':float(np.median(sim[:,j])),
                         'null_95':float(np.quantile(sim[:,j],.95)),
                         'null_fraction_gte_observed':float(np.mean(sim[:,j]>=obs[j])),
                         'permutations':reps})
    return pd.DataFrame(rows)


def analyze_asset(train,valid,test,*,pseudo=40.,ridge=.015,reps=1000,seed=2601):
    """Holdout model scoring; validation is descriptive and never used for fitting."""
    fit=fit_direction_models(train,pseudo=pseudo,ridge=ridge)
    scored=[];summary=[];boot=[]
    for period,g in [('valid',valid),('test',test)]:
        ev=g.loc[g.scorable].copy()
        preds=predict_direction(ev,fit)
        scores,day=forecast_scores(ev,preds)
        scores['period']=period;summary.append(scores)
        # Exactly aligned targets/predictions; day model differences are paired.
        for model in ('streak_exact_shrunk','streak_log_duration'):
            for reference in ('markov_sign','iid_empirical'):
                b=bootstrap_comparison(day,model=model,reference=reference,reps=reps,seed=seed)
                b['period']=period;boot.append(b)
        scored.append({'period':period,'n_eval':len(ev)})
    return fit,pd.concat(summary,ignore_index=True),pd.concat(boot,ignore_index=True),pd.DataFrame(scored)
