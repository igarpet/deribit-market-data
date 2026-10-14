"""CS03: as-of-five-minute-close Binance USD-M taker trade flow vs spot streak transitions.

Spot 5m close-to-close streaks are unchanged; USD-M 1m *executed* trade
volumes (not L2 resting orders) are collapsed to the same 5m bar END.
Next-bar states are DOWN/ZERO/UP; no jumps over flat bars and no lookahead.
Research-only, historical exploratory, fixed controls and model family.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.preprocessing import StandardScaler

STEP = pd.Timedelta(minutes=5)
MINUTE = pd.Timedelta(minutes=1)
CLASSES = np.array([-1, 0, 1], dtype=int)
VERSION = 'CS03-0.1.1'
MODELS = ('iid', 'sign', 'sign_streak', 'sign_streak_vol',
          'sign_streak_flow', 'sign_streak_vol_flow')

@dataclass(frozen=True)
class Specification:
    version: str = VERSION
    assets: tuple[str, ...] = ('BTC', 'ETH', 'SOL', 'XRP')
    train_fraction: float = .60
    valid_fraction: float = .20
    fast_bars: int = 12
    slow_bars: int = 288
    flow_bars: int = 6
    logistic_C: float = .25
    bootstrap_reps: int = 1000
    shuffle_reps: int = 100
    seed: int = 2604
    historical_status: str = 'HISTORICAL_EXPLORATORY'


def registration():
    return asdict(Specification())


def one_minute_futures_flow(frame: pd.DataFrame) -> pd.DataFrame:
    """Map EXACTLY five 1m candles ending in [t-5m,t) onto a five-minute END t.

    No interpolation or filling missing minutes. Quotes and taker-buy quote
    volumes are in the same USDT units. Sell quote = total - taker-buy.
    """
    cols = {'open_time', 'quote_volume', 'taker_buy_quote_volume', 'trades'}
    if not cols.issubset(frame):
        raise ValueError(f'Missing futures kline columns: {cols-set(frame)}')
    q = frame[list(cols)].copy()
    q['open_time'] = pd.to_datetime(q.open_time, utc=True)
    if q.open_time.isna().any() or q.open_time.duplicated().any():
        raise ValueError('NaT or duplicated futures minute time')
    if (q.open_time.dt.second.ne(0) | q.open_time.dt.microsecond.ne(0)).any():
        raise ValueError('off-minute futures timestamp')
    for c in ('quote_volume', 'taker_buy_quote_volume', 'trades'):
        q[c] = pd.to_numeric(q[c], errors='coerce')
    if (~np.isfinite(q[['quote_volume','taker_buy_quote_volume','trades']].to_numpy(float))).any():
        raise ValueError('nonfinite futures flow')
    if ((q[['quote_volume','taker_buy_quote_volume','trades']] < 0).any().any() or
        (q.taker_buy_quote_volume > q.quote_volume+1e-6).any()):
        raise ValueError('invalid aggressive buy/total volume')
    q['state_time'] = q.open_time.dt.floor('5min') + STEP
    # A complete bucket must contain exactly minutes 0,1,2,3,4, not simply 5 rows.
    z = q.groupby('state_time', sort=True).agg(
        n_minutes=('open_time','size'),
        first_minute=('open_time','min'), last_minute=('open_time','max'),
        quote_volume=('quote_volume','sum'),
        taker_buy_quote_volume=('taker_buy_quote_volume','sum'),
        trade_count=('trades','sum'))
    good = ((z.n_minutes == 5) & (z.first_minute == z.index-STEP) &
            (z.last_minute == z.index-MINUTE))
    z['flow_valid'] = good & (z.quote_volume > 0)
    z['signed_quote'] = 2*z.taker_buy_quote_volume-z.quote_volume
    z['flow_5m'] = np.where(z.flow_valid, z.signed_quote/z.quote_volume.replace(0,np.nan), np.nan)
    z.loc[~good, ['quote_volume','taker_buy_quote_volume','signed_quote','trade_count']] = np.nan
    z['flow_valid'] = z.flow_valid.astype(bool)
    return z[['flow_valid','n_minutes','quote_volume','taker_buy_quote_volume',
              'signed_quote','trade_count','flow_5m']].reset_index()


def read_futures_parquet(path: Path, start=None, end=None) -> pd.DataFrame:
    """Only read columns needed for aggressor-side volume; on-disk data are untouched."""
    if not path.exists():
        raise FileNotFoundError(f'Missing downloaded Binance USD-M file: {path}')
    names=['open_time','quote_volume','taker_buy_quote_volume','trades']
    q=pd.read_parquet(path, columns=names)
    if start is not None or end is not None:
        ts=pd.to_datetime(q.open_time, utc=True)
        if start is not None: q=q.loc[ts >= pd.Timestamp(start)].copy();ts=ts.loc[q.index]
        if end is not None: q=q.loc[ts < pd.Timestamp(end)].copy()
    if q.empty: raise ValueError(f'No Binance futures minutes in interval: {path}')
    return one_minute_futures_flow(q)


def _contiguous_roll(values: np.ndarray, times: pd.Series, bars: int, op='mean') -> np.ndarray:
    t=pd.to_datetime(times,utc=True).reset_index(drop=True)
    gap=np.r_[True, ((t.diff().iloc[1:] != STEP).to_numpy())]
    segment=pd.Series(np.cumsum(gap))
    val=pd.Series(np.asarray(values, float))
    if op=='mean':
        return val.groupby(segment).transform(lambda x:x.rolling(bars,min_periods=bars).mean()).to_numpy(float)
    if op=='sum':
        return val.groupby(segment).transform(lambda x:x.rolling(bars,min_periods=bars).sum()).to_numpy(float)
    raise ValueError(op)


def build_origin_panel(spot: pd.DataFrame, flow5m: pd.DataFrame, *,
                       fast_bars=12, slow_bars=288, flow_bars=6) -> pd.DataFrame:
    """Every price bar is retained; flat *current* bars get sign=0, age=0.

    All predictors end at t, target is price return at t+5m. Missing futures
    minute buckets cannot be forwarded into subsequent bars.
    """
    req={'state_time','return_5m'}
    if not req.issubset(spot):raise ValueError(f'Missing spot fields: {req-set(spot)}')
    x=spot[['state_time','return_5m']].copy()
    x['state_time']=pd.to_datetime(x.state_time,utc=True)
    if x.state_time.duplicated().any():raise ValueError('duplicate spot state_time')
    x=x.sort_values('state_time').reset_index(drop=True)
    if ((x.state_time.dt.minute%5!=0)|x.state_time.dt.second.ne(0)).any():
        raise ValueError('spot times off 5m grid')
    if flow5m.state_time.duplicated().any():raise ValueError('duplicate 5m flow time')
    x=x.merge(flow5m,on='state_time',how='left',validate='one_to_one').sort_values('state_time').reset_index(drop=True)
    r=pd.to_numeric(x.return_5m,errors='coerce').to_numpy(float)
    valid=np.isfinite(r)
    s=np.zeros(len(x),dtype=np.int8);s[valid]=np.sign(r[valid]).astype(np.int8)
    time=x.state_time
    contiguous=np.r_[False,(time.diff().iloc[1:]==STEP).to_numpy()]
    idx=np.arange(len(x))
    begins=(s==0)|(s!=np.r_[0,s[:-1]])|(~contiguous)
    laststart=np.maximum.accumulate(np.where(begins,idx,0))
    age=np.where(s!=0,idx-laststart+1,0)
    x['sign']=s;x['length']=age;x['signed_streak']=s*age
    x['next_sign']=np.r_[s[1:],0]
    next_finite=np.r_[valid[1:],False]
    next_adj=np.r_[contiguous[1:],False]
    x['next_valid']=valid & next_finite & next_adj
    x['date']=x.state_time.dt.floor('D')
    x['outcome']=np.where(~x.next_valid,'censored',
                          np.where(s==0,np.where(x.next_sign==0,'flat_stays',
                                   np.where(x.next_sign==1,'start_up_after_flat','start_down_after_flat')),
                          np.where(x.next_sign==s,'continue',
                                   np.where(x.next_sign==0,'stop_flat','reverse_new_streak'))))
    sq=r*r
    x['rv_fast']=_contiguous_roll(sq,time,fast_bars)
    x['rv_slow']=_contiguous_roll(sq,time,slow_bars)
    x['log_rv24h']=np.log(np.maximum(x.rv_slow.to_numpy(float),1e-30))
    x['log_fastslow']=np.log(np.maximum(x.rv_fast.to_numpy(float),1e-30) /
                           np.maximum(x.rv_slow.to_numpy(float),1e-30))
    flow_ok=x.flow_valid.fillna(False).to_numpy(bool)
    signed=np.where(flow_ok,x.signed_quote.to_numpy(float),np.nan)
    quot=np.where(flow_ok,x.quote_volume.to_numpy(float),np.nan)
    a=_contiguous_roll(signed,time,flow_bars,'sum')
    b=_contiguous_roll(quot,time,flow_bars,'sum')
    x['flow_30m']=a/b
    x['log_quote_volume']=np.log1p(np.maximum(x.quote_volume.to_numpy(float),0))
    x['log_trades']=np.log1p(np.maximum(x.trade_count.to_numpy(float),0))
    feat=['log_rv24h','log_fastslow','flow_5m','flow_30m','log_quote_volume','log_trades']
    x['features_ready']=valid&flow_ok&np.isfinite(x[feat].to_numpy(float)).all(axis=1)
    # Make missing coverage explicit; no ffill or interpolation.
    return x


def full_days(panel:pd.DataFrame) -> pd.DatetimeIndex:
    """Require a complete 288-bar UTC day for BOTH spot returns and futures flow."""
    q=panel.groupby('date').agg(n=('state_time','size'),start=('state_time','min'),
                                end=('state_time','max'),ready=('features_ready','sum'))
    good=(q.n.eq(288)&q.ready.eq(288)&
          q.start.eq(q.index.to_series())&
          q.end.eq(q.index.to_series()+pd.Timedelta(hours=23,minutes=55)))
    return pd.DatetimeIndex(q.index[good])


def split_panel(panel:pd.DataFrame,dates:pd.DatetimeIndex,train_fraction=.6,valid_fraction=.2):
    dates=pd.DatetimeIndex(sorted(dates))
    if len(dates)<60:raise ValueError(f'Insufficient complete intersecting spot/futures days: {len(dates)}')
    cut1=dates[int(len(dates)*train_fraction)]
    cut2=dates[int(len(dates)*(train_fraction+valid_fraction))]
    z=panel.loc[panel.date.isin(dates)&panel.features_ready].copy()
    z['period']=np.where(z.date<cut1,'train',np.where(z.date<cut2,'valid','test'))
    nxt=(z.state_time+STEP).dt.floor('D')
    nextpart=np.where(nxt<cut1,'train',np.where(nxt<cut2,'valid','test'))
    z['scorable']=z.next_valid & nxt.isin(dates) & (nextpart==z.period)
    return z,{'train_before':str(cut1),'test_from':str(cut2),
              'complete_intersection_days':len(dates),
              'train_days':int(sum(dates<cut1)),
              'valid_days':int(sum((dates>=cut1)&(dates<cut2))),
              'test_days':int(sum(dates>=cut2))}


def raw_design(frame:pd.DataFrame) -> pd.DataFrame:
    """Fixed model terms; age enters continuously and isn't capped."""
    s=frame.sign.to_numpy(int)
    age=np.log1p(frame.length.to_numpy(float))
    f5=frame.flow_5m.to_numpy(float)
    f30=frame.flow_30m.to_numpy(float)
    return pd.DataFrame({
       's_down':(s==-1).astype(float),'s_up':(s==1).astype(float),
       'age_signed':s*age,'age_abs':age,
       'log_rv24h':frame.log_rv24h.to_numpy(float),
       'log_fastslow':frame.log_fastslow.to_numpy(float),
       'flow5':f5,'flow30':f30,
       'flow5_by_sign':f5*s,'flow30_by_sign':f30*s,
       'log_quote_volume':frame.log_quote_volume.to_numpy(float),
       'log_trades':frame.log_trades.to_numpy(float),
    },index=frame.index)

BASE_FEATURES={
    'sign':('s_down','s_up'),
    'sign_streak':('s_down','s_up','age_signed','age_abs'),
    'sign_streak_vol':('s_down','s_up','age_signed','age_abs','log_rv24h','log_fastslow'),
    'sign_streak_flow':('s_down','s_up','age_signed','age_abs','flow5','flow30',
                        'flow5_by_sign','flow30_by_sign','log_quote_volume','log_trades'),
    'sign_streak_vol_flow':('s_down','s_up','age_signed','age_abs','log_rv24h','log_fastslow',
                            'flow5','flow30','flow5_by_sign','flow30_by_sign',
                            'log_quote_volume','log_trades')
}


def fit_models(train:pd.DataFrame,*,C=.25) -> dict:
    tr=train.loc[train.scorable].copy()
    if len(tr)<200:raise ValueError('Too few training transitions')
    y=tr.next_sign.to_numpy(int)
    if set(np.unique(y))!=set(CLASSES):
        raise ValueError('Need DOWN/ZERO/UP observations to fit three-class model')
    designs=raw_design(tr)
    prior=(np.array([(y==c).sum() for c in CLASSES])+0.5)/(len(y)+1.5)
    fits={'iid':prior,'sample_size':len(tr),'models':{}}
    for name,terms in BASE_FEATURES.items():
        scaler=StandardScaler()
        X=scaler.fit_transform(designs[list(terms)].to_numpy(float))
        # With rare zero-return targets and correlated flow features, L-BFGS can
        # exhaust 300 steps before meeting the strict 1e-7 gradient tolerance.
        # Newton-Cholesky targets exactly the same multinomial L2 objective and C,
        # but is better suited to this small dense design (~12 columns).
        # Multinomial support requires scikit-learn >= 1.6.
        model=LogisticRegression(C=C,max_iter=120,tol=1e-7,solver='newton-cholesky')
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always',ConvergenceWarning)
            model.fit(X,y)
        convergence=[str(w.message) for w in seen
                     if issubclass(w.category,ConvergenceWarning)]
        if convergence or int(model.n_iter_[0])>=model.max_iter:
            raise RuntimeError(
                f'{name} logistic did not converge with {model.solver}: '
                f'iterations={int(model.n_iter_[0])}/{model.max_iter}; '
                f'warnings={convergence}')
        fits['models'][name]=(terms,scaler,model)
    return fits


def predict_models(frame:pd.DataFrame,fit:dict) -> dict[str,np.ndarray]:
    d=raw_design(frame)
    out={'iid':np.broadcast_to(fit['iid'],(len(frame),3)).copy()}
    for name,(terms,scaler,model) in fit['models'].items():
        p=model.predict_proba(scaler.transform(d[list(terms)].to_numpy(float)))
        if not np.array_equal(model.classes_,CLASSES):raise ValueError('Class mismatch')
        out[name]=p
    return out


def score_models(events:pd.DataFrame,preds:dict[str,np.ndarray]):
    """Joint three-class and next-nonzero UP/DOWN diagnostics on same origins."""
    truth=events.next_sign.to_numpy(int)
    if not np.isin(truth,CLASSES).all():raise ValueError('Target classes invalid')
    target=np.searchsorted(CLASSES,truth)
    count=np.arange(len(truth));onehot=np.eye(3)[target]
    dates=events.date.to_numpy()
    keep=truth!=0
    rows=[];byday=[]
    for name,p in preds.items():
        if p.shape!=(len(events),3) or not np.allclose(p.sum(axis=1),1):
            raise ValueError('Bad probabilities')
        p=np.clip(p,1e-12,1-1e-12)
        nll=-np.log(p[count,target])
        brier=((p-onehot)**2).sum(axis=1)
        up=p[:,2]/(p[:,0]+p[:,2])
        binary=-np.log(np.where(truth[keep]==1,up[keep],1-up[keep]))
        rows.append({'model':name,'n':len(p),'threeclass_logloss':float(nll.mean()),
                     'threeclass_brier':float(brier.mean()),'n_nonzero_target':int(keep.sum()),
                     'binary_logloss':float(binary.mean()),
                     'binary_brier':float(np.mean((up[keep]-(truth[keep]==1))**2)),
                     'forecast_zero':float(p[:,1].mean()),'observed_zero':float((truth==0).mean())})
        byday.append(pd.DataFrame({'date':dates,'model':name,'threeclass_logloss':nll,
                                   'threeclass_brier':brier,'binary_logloss':np.where(keep,np.nan,-1.)}))
        byday[-1]['binary_logloss']=np.nan
        byday[-1].loc[keep,'binary_logloss']= -np.log(np.where(truth[keep]==1,up[keep],1-up[keep]))
    return pd.DataFrame(rows),pd.concat(byday,ignore_index=True)


def paired_day_bootstrap(day_losses:pd.DataFrame,*,B=1000,seed=2604) -> pd.DataFrame:
    """Complete-day clustered bootstrap, equal-day mean, paired model losses."""
    pairs=[('sign_streak','sign'),('sign_streak_vol','sign_streak'),
           ('sign_streak_flow','sign_streak'),
           ('sign_streak_vol_flow','sign_streak_vol'),
           ('sign_streak_vol_flow','sign_streak_flow'),
           ('sign_streak_vol_flow','sign')]
    out=[];rng=np.random.default_rng(seed)
    for metric in ('threeclass_logloss','threeclass_brier','binary_logloss'):
        tab=day_losses.groupby(['date','model'])[metric].mean().unstack('model')
        for a,b in pairs:
            vals=(tab[a]-tab[b]).dropna().to_numpy(float)
            if len(vals)<20:raise ValueError('insufficient full days for bootstrap')
            sel=rng.integers(0,len(vals),size=(B,len(vals)))
            means=vals[sel].mean(axis=1)
            out.append({'model':a,'reference':b,'metric':metric,
                        'mean_day_delta':float(vals.mean()),
                        'ci025':float(np.quantile(means,.025)),
                        'ci975':float(np.quantile(means,.975)),
                        'days':int(len(vals))})
    return pd.DataFrame(out)


def fit_flow_cuts(train:pd.DataFrame):
    vals=train.loc[train.scorable,'flow_5m'].to_numpy(float)
    q=np.quantile(vals,[1/3,2/3])
    if q[0]>=q[1]:raise ValueError('flow tertiles collapse')
    return q.tolist()


def observed_transitions(panel:pd.DataFrame,cuts:list[float],*,min_n=1) -> pd.DataFrame:
    """Descriptive start, continuation, reversal and flat-stop probabilities.

    Age groups are *display only*; the regression uses exact log(age).
    """
    d=panel.loc[panel.scorable].copy()
    d['flow_tier']=np.searchsorted(cuts,d.flow_5m.to_numpy(float),side='right')
    d['age_group']=pd.cut(d.length,bins=[-1,0,1,2,3,5,8,np.inf],
                           labels=['0 flat','1','2','3','4-5','6-8','9+'])
    keys=['period','sign','age_group','flow_tier']
    count=d.groupby(keys,observed=True).agg(n=('outcome','size'),
       p_continue=('outcome',lambda x:(x=='continue').mean()),
       p_reverse=('outcome',lambda x:(x=='reverse_new_streak').mean()),
       p_stop_flat=('outcome',lambda x:(x=='stop_flat').mean()),
       p_start_up_after_flat=('outcome',lambda x:(x=='start_up_after_flat').mean()),
       p_start_down_after_flat=('outcome',lambda x:(x=='start_down_after_flat').mean()),
       p_flat_stays=('outcome',lambda x:(x=='flat_stays').mean()),
       p_next_up=('next_sign',lambda x:(x==1).mean()),
       p_next_down=('next_sign',lambda x:(x==-1).mean()),
       p_next_zero=('next_sign',lambda x:(x==0).mean())).reset_index()
    return count.loc[count.n>=min_n]


def shuffle_test_flow(test:pd.DataFrame,fit:dict,*,B=100,seed=2604):
    """Diagnostic only: keep sign/streak/vol fixed, permute entire flow feature
    vector across rows within UTC hour across days; trained model frozen.
    This does not preserve all market structure and is NOT a causal p-value.
    """
    te=test.loc[test.scorable].copy().reset_index(drop=True)
    obs=predict_models(te,fit)
    target=np.searchsorted(CLASSES,te.next_sign.to_numpy(int))
    base=-np.log(np.maximum(obs['sign_streak_vol'][np.arange(len(te)),target],1e-12)).mean()
    actual=-np.log(np.maximum(obs['sign_streak_vol_flow'][np.arange(len(te)),target],1e-12)).mean()-base
    rng=np.random.default_rng(seed)
    fields=['flow_5m','flow_30m','log_quote_volume','log_trades']
    groups=te.groupby(te.state_time.dt.hour).indices
    vals=te[fields].to_numpy(float)
    sim=[]
    for _ in range(B):
        q=te.copy();arr=vals.copy()
        for ids in groups.values():arr[ids]=vals[rng.permutation(ids)]
        q[fields]=arr
        p=predict_models(q,fit)['sign_streak_vol_flow']
        sim.append(float(-np.log(np.maximum(p[np.arange(len(te)),target],1e-12)).mean()-base))
    return {'observed_delta':float(actual),'null_p05':float(np.quantile(sim,.05)),
            'null_median':float(np.median(sim)), 'null_p95':float(np.quantile(sim,.95)),
            'fraction_null_le_observed':float(np.mean(np.array(sim)<=actual)), 'reps':B}
