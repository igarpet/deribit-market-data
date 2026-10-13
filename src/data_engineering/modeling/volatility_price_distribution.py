"""VD13: two-timescale volatility, price martingale and conditional reversion.

Research-only. Independent raw-data experiment, never imports another research
branch or prior research outputs. No option pricing or execution assumptions.

Unit: Binance spot close-to-close native 5m log returns, regular UTC grid.
Forecast made *after* observing return r_t, targets bars t+1..t+k.
No gap interpolation, no test-conditioned features, no mixed-split targets.
All projections and per-asset scale calibrations use training data only.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

VERSION = 'VD13-0.1.0'
SAMPLE_END = pd.Timestamp('2026-09-19T00:00:00Z')  # already-viewed history; exploratory
HORIZONS = (15, 60, 240)
PRIMARY = 60
BAR_MIN = 5
FAST_MIN = 30
SLOW_MIN = 240
MIN_HISTORY = 288
MIN_COVERAGE = 0.8
RIDGE_VAR = 0.02
RIDGE_MEAN = 0.10
TAIL_QUANTILE = 0.95
DEFAULT_BOOT = 1000
MIN_TEST_DAYS = 30
MIN_DISTINCT_TAIL_TIMES = 40


@dataclass(frozen=True)
class Parameters:
    end_exclusive: str = str(SAMPLE_END)
    horizons: tuple[int, ...] = HORIZONS
    primary_horizon: int = PRIMARY
    fast_half_life_min: int = FAST_MIN
    slow_half_life_min: int = SLOW_MIN
    minimum_previous_bars: int = MIN_HISTORY
    previous_coverage_min: float = MIN_COVERAGE
    ridge_var: float = RIDGE_VAR
    ridge_mean: float = RIDGE_MEAN
    tail_quantile: float = TAIL_QUANTILE
    split_train: float = .60
    split_valid: float = .20
    min_test_days: int = MIN_TEST_DAYS
    min_unique_tail_times: int = MIN_DISTINCT_TAIL_TIMES
    historical_sample_is_exploratory: bool = True


def load_returns(directory: Path, assets: list[str], *, end=SAMPLE_END,
                 min_days: int = 365) -> tuple[pd.DataFrame, dict]:
    """Read shared 5m price-history files, not any research-branch outputs.

    Expected <ASSET>.csv.gz with state_time and close columns. Returns on
    nonconsecutive bars are missing, never bridged over an outage.
    """
    frames, excluded = [], {}
    for asset in assets:
        path = Path(directory) / f'{asset}.csv.gz'
        if not path.exists():
            excluded[asset] = 'missing file'
            continue
        z = pd.read_csv(path, usecols=['state_time', 'close'])
        z['state_time'] = pd.to_datetime(z.state_time, utc=True)
        z = z[(z.state_time < end)].sort_values('state_time')
        if z.state_time.duplicated().any():
            raise ValueError(f'Duplicate five-minute timestamps in {path}')
        if ((z.state_time.dt.minute % 5 != 0) |
            (z.state_time.dt.second != 0)).any():
            raise ValueError(f'Off-grid timestamps in {path}')
        ndays = z.state_time.dt.floor('D').nunique()
        if ndays < min_days:
            excluded[asset] = f'{ndays} days, require {min_days}'
            continue
        close = pd.to_numeric(z.close, errors='coerce')
        consecutive = z.state_time.diff().eq(pd.Timedelta(minutes=5))
        ret = np.log(close / close.shift(1)).where(consecutive &
                 close.gt(0) & close.shift(1).gt(0))
        frames.append(pd.DataFrame({'state_time':z.state_time.to_numpy(),
                     'underlying':asset, 'return_5m':ret.to_numpy(float)}))
    if not frames:
        raise ValueError(f'No assets with >= {min_days} days of history in {directory}')
    return pd.concat(frames, ignore_index=True), excluded


def prepare_panel(raw: pd.DataFrame, min_days: int=12) -> pd.DataFrame:
    need = {'underlying', 'state_time', 'return_5m'}
    if need - set(raw):
        raise ValueError(f'Missing columns: {need - set(raw)}')
    f = raw[['underlying','state_time','return_5m']].copy()
    f['underlying'] = f.underlying.astype(str)
    f['state_time'] = pd.to_datetime(f.state_time, utc=True)
    f['return_5m'] = pd.to_numeric(f.return_5m, errors='coerce')
    if f[['underlying','state_time']].duplicated().any():
        raise ValueError('Duplicate asset-timestamp')
    parts=[]
    for asset,g in f.groupby('underlying',sort=True):
        g=g.sort_values('state_time').set_index('state_time')
        if ((g.index.minute%5)!=0).any() or (g.index.second!=0).any():
            raise ValueError('Off-grid timestamp')
        ind=pd.date_range(g.index.min(),g.index.max(),freq='5min',tz='UTC')
        g=g.reindex(ind)
        g['underlying']=asset
        g.index.name='state_time'
        parts.append(g.reset_index())
    p=pd.concat(parts,ignore_index=True)
    p['date']=p.state_time.dt.floor('D')
    dates=np.sort(p.date.unique())
    if len(dates)<min_days:
        raise ValueError('Insufficient distinct UTC days')
    trainend=max(1,int(np.floor(.6*len(dates))))
    validend=max(trainend+1,int(np.floor(.8*len(dates))))
    if validend>=len(dates):
        raise ValueError('No held-out days')
    p['split']=np.where(p.date<dates[trainend],'train',
               np.where(p.date<dates[validend],'valid','test'))
    p['slot']=(p.state_time.dt.hour*12 + p.state_time.dt.minute//5).astype(int)
    p['hour']=p.state_time.dt.hour.astype(int)
    return p


def _ewma_observed(z: np.ndarray, alpha: float) -> np.ndarray:
    """A missing bar has zero observed weight, not a zero return."""
    mask=np.isfinite(z)
    numer=pd.Series(np.where(mask,z,0.)).ewm(alpha=alpha,adjust=False).mean().to_numpy()
    mass=pd.Series(mask.astype(float)).ewm(alpha=alpha,adjust=False).mean().to_numpy()
    return np.divide(numer,mass,out=np.full(len(z),np.nan),where=mass>1e-10)


def make_features(p: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Time-t features: all inputs are available after the current bar closes."""
    f=p[['underlying','state_time','date','split','slot','hour','return_5m']].copy()
    f['now']=np.nan
    f['fast']=np.nan
    f['slow']=np.nan
    f['slope15']=np.nan
    f['r30']=np.nan
    f['past_coverage']=np.nan
    f['history_ok']=False
    clock=[]
    for asset,g in p.groupby('underlying',sort=False):
        ix=g.index.to_numpy()
        r=g.return_5m.to_numpy(float)
        train=(g.split.to_numpy()=='train')
        good=np.isfinite(r)
        pos=r[train & good]**2
        pos=pos[pos>0]
        if len(pos)<100:
            raise ValueError(f'Not enough valid training bars for {asset}')
        eps=np.median(pos)*.001
        y=np.log(r*r+eps)
        tr=y[train & np.isfinite(y)]
        fallback=float(np.median(tr))
        hour=g.hour.to_numpy(int)
        centers=np.empty(24,dtype=float)
        for h in range(24):
            v=y[train & np.isfinite(y) & (hour==h)]
            centers[h]=float(np.median(v)) if len(v)>=20 else fallback
            clock.append({'underlying':asset,'hour':h,'train_center':centers[h],
                          'train_count':int(len(v))})
        y=y-centers[hour]
        af=1-np.exp(-np.log(2)*BAR_MIN/FAST_MIN)
        aslow=1-np.exp(-np.log(2)*BAR_MIN/SLOW_MIN)
        fast=_ewma_observed(y,af)
        slow=_ewma_observed(y,aslow)
        t=np.arange(len(g))
        cs=np.r_[0,np.cumsum(good.astype(np.int64))]
        past=(cs[t]-cs[np.maximum(0,t-MIN_HISTORY)])/MIN_HISTORY
        recent=(cs[t]-cs[np.maximum(0,t-3)])>0
        ready=(t>=MIN_HISTORY)&(past>=MIN_COVERAGE)&recent&good
        # r30 is sum of the six available returns through origin t.
        s=pd.Series(r)
        r30=s.rolling(6,min_periods=6).sum().to_numpy(float)
        f.loc[ix,'now']=y
        f.loc[ix,'fast']=fast
        f.loc[ix,'slow']=slow
        f.loc[ix,'slope15']=fast-np.r_[np.full(3,np.nan),fast[:-3]]
        f.loc[ix,'r30']=r30
        f.loc[ix,'past_coverage']=np.where(t>=MIN_HISTORY,past,np.nan)
        f.loc[ix,'history_ok']=ready
    f['gap']=f.fast-f.slow
    f['abs_gap']=np.abs(f.gap)
    f['gap_slope']=f.gap*f.slope15
    f['r5']=f.return_5m
    f['r30_gap']=f.r30*f.gap
    f['r30_slope']=f.r30*f.slope15
    f['r5_gap']=f.r5*f.gap
    return f,pd.DataFrame(clock)


def add_forward_targets(p: pd.DataFrame,f: pd.DataFrame,horizons=HORIZONS)->tuple[pd.DataFrame,dict]:
    """Exactly future bars (t,t+h], all finite and in the same split."""
    out=f.copy()
    threshold={}
    for horizon in horizons:
        if horizon%BAR_MIN:
            raise ValueError('Horizon must divide 5m')
        nbar=horizon//BAR_MIN
        net=np.full(len(p),np.nan)
        rv=np.full(len(p),np.nan)
        for asset,g in p.groupby('underlying',sort=False):
            ix=g.index.to_numpy()
            r=g.return_5m.to_numpy(float)
            finite=np.isfinite(r)
            sr=np.r_[0,np.cumsum(np.where(finite,r,0.))]
            sv=np.r_[0,np.cumsum(np.where(finite,r*r,0.))]
            sg=np.r_[0,np.cumsum(finite.astype(np.int64))]
            n=len(r)
            t=np.arange(n)
            ends=t+nbar
            inbounds=ends<n
            end=np.minimum(ends,n-1)
            cnt=sg[end+1]-sg[t+1]
            splits=g.split.to_numpy()
            valid=inbounds & finite & (cnt==nbar) & (splits[end]==splits[t])
            currentnet=sr[end+1]-sr[t+1]
            currentrv=sv[end+1]-sv[t+1]
            net[ix]=np.where(valid,currentnet,np.nan)
            rv[ix]=np.where(valid,currentrv,np.nan)
        out[f'R_{horizon}']=net
        out[f'RV_{horizon}']=rv
        q={}
        for asset,g in out.groupby('underlying',sort=False):
            sample=np.abs(g.loc[(g.split=='train') & np.isfinite(g[f'R_{horizon}']),f'R_{horizon}'].to_numpy(float))
            if len(sample)<200:
                raise ValueError(f'{asset}: insufficient train return targets')
            q[asset]=float(np.quantile(sample,TAIL_QUANTILE))
        thresh=out.underlying.map(q).to_numpy(float)
        out[f'tail_{horizon}']=np.where(np.isfinite(net),(np.abs(net)>=thresh).astype(float),np.nan)
        threshold[str(horizon)]=q
    return out,threshold


BASE=['now','fast','slow']
PHASE=BASE+['slope15','abs_gap','gap_slope']
MEAN=['r5','r30']
INTERACTION=MEAN+['r30_gap','r30_slope','r5_gap']


def _matrix(train:pd.DataFrame,other:pd.DataFrame,names:list[str]):
    """Train-only standardized numeric features + train-known asset and clock effects."""
    assets=sorted(train.underlying.unique())
    if names:
        xtr=train[names].to_numpy(float)
        center=xtr.mean(axis=0)
        scale=xtr.std(axis=0)
        scale=np.where(scale>1e-10,scale,1.)
    else:
        center=np.array([]);scale=np.array([])
    def build(g):
        n=len(g)
        cols=[np.ones(n)]
        for asset in assets[1:]:
            cols.append((g.underlying.to_numpy()==asset).astype(float))
        phase=2*np.pi*g.slot.to_numpy(float)/288
        cols.extend([np.sin(phase),np.cos(phase)])
        if names:
            x=(g[names].to_numpy(float)-center)/scale
            cols.extend(x[:,j] for j in range(len(names)))
        return np.column_stack(cols)
    return build(train),build(other)


def _ridge_beta(X:np.ndarray,y:np.ndarray,alpha:float):
    gram=X.T@X
    pen=np.eye(X.shape[1])*alpha*len(y)
    pen[0,0]=0
    return np.linalg.solve(gram+pen+np.eye(len(gram))*1e-10,X.T@y)


def _fitted_values(train,eval_,columns,target,alpha):
    a,b=_matrix(train,eval_,columns)
    y=train[target].to_numpy(float)
    beta=_ridge_beta(a,y,alpha)
    return a@beta,b@beta


def _nll(y:np.ndarray,mu:np.ndarray,v:np.ndarray):
    return .5*(np.log(2*np.pi*v)+(y-mu)**2/v)


def _prob_abs_gaussian(threshold,mu,sd):
    return ndtr((-threshold-mu)/sd)+(1-ndtr((threshold-mu)/sd))


def fit_horizon(frame:pd.DataFrame,horizon:int)->tuple[pd.DataFrame,pd.DataFrame]:
    """K: fixed-scale RW; E: two-EMA volatility RW; P: phase volatility RW;
       M: price-mean reversion + P scale; PM: conditional mean-phase + P scale.
    All models compared on exactly the same complete, frozen support.
    """
    names=set(PHASE+INTERACTION)
    req=list(names)+[f'R_{horizon}',f'RV_{horizon}',f'tail_{horizon}']
    ready=frame.history_ok.to_numpy(bool)&np.isfinite(frame[req].to_numpy(float)).all(axis=1)
    tr=frame[(frame.split=='train')&ready].copy()
    te=frame[(frame.split=='test')&ready].copy()
    if len(tr)<1000 or len(te)<100:
        raise ValueError('Insufficient complete train or test origins')
    ytr=tr[f'R_{horizon}'].to_numpy(float)
    yte=te[f'R_{horizon}'].to_numpy(float)
    rvtr=tr[f'RV_{horizon}'].to_numpy(float)
    logrvtr=np.log(np.maximum(rvtr,1e-16))
    tr['log_rv']=logrvtr
    te['log_rv']=np.log(np.maximum(te[f'RV_{horizon}'].to_numpy(float),1e-16))
    _, lr_e=_fitted_values(tr,te,BASE,'log_rv',RIDGE_VAR)
    _, lr_p=_fitted_values(tr,te,PHASE,'log_rv',RIDGE_VAR)
    _, mu_m=_fitted_values(tr,te,MEAN,f'R_{horizon}',RIDGE_MEAN)
    _, mu_pm=_fitted_values(tr,te,INTERACTION,f'R_{horizon}',RIDGE_MEAN)
    # Model-specific train-only net-variance correction for stochastic scale.
    e_train,_=_fitted_values(tr,te,BASE,'log_rv',RIDGE_VAR)
    p_train,_=_fitted_values(tr,te,PHASE,'log_rv',RIDGE_VAR)
    v_k=np.full(len(te),np.nan)
    lr_k=np.full(len(te),np.nan)
    v_e=np.full(len(te),np.nan)
    v_p=np.full(len(te),np.nan)
    for asset,g in te.groupby('underlying',sort=False):
        ix=np.where(te.underlying.to_numpy()==asset)[0]
        jx=np.where(tr.underlying.to_numpy()==asset)[0]
        net_tr=ytr[jx]
        if len(jx)<100:
            raise ValueError(f'Insufficient training asset: {asset}')
        # At most 2% of extreme squared returns are trimmed for robust scale.
        raw_square=net_tr**2
        cap=np.quantile(raw_square,.995)
        v0=float(np.maximum(np.mean(np.minimum(raw_square,cap)),1e-12))
        v_k[ix]=v0
        lr_k[ix]=float(np.mean(logrvtr[jx]))
        for fitted_train, fitted_test, dest in [(e_train,lr_e,v_e),(p_train,lr_p,v_p)]:
            pred0=np.exp(np.clip(fitted_train[jx],-40,5))
            ratio=np.clip(v0 / np.maximum(np.mean(pred0),1e-14),.01,100.)
            dest[ix]=np.maximum(np.exp(np.clip(fitted_test[ix],-40,5))*ratio,1e-12)
    estimates={'K_random_walk':(np.zeros(len(te)),v_k,lr_k),
               'E_two_ema':(np.zeros(len(te)),v_e,lr_e),
               'P_vol_phase':(np.zeros(len(te)),v_p,lr_p),
               'M_price_reversion':(mu_m,v_p,lr_p),
               'PM_conditional':(mu_pm,v_p,lr_p)}
    q={}
    for asset,g in tr.groupby('underlying',sort=False):
        q[asset]=float(np.quantile(np.abs(g[f'R_{horizon}'].to_numpy(float)),TAIL_QUANTILE))
    thresholds=te.underlying.map(q).to_numpy(float)
    tail=te[f'tail_{horizon}'].to_numpy(float)
    score_rows=[];pred_rows=[]
    for model,(mu,v,logrv_forecast) in estimates.items():
        v=np.maximum(v,1e-12)
        sd=np.sqrt(v)
        prob=np.clip(_prob_abs_gaussian(thresholds,mu,sd),1e-8,1-1e-8)
        negll=_nll(yte,mu,v)
        rv_err=(te['log_rv'].to_numpy(float)-logrv_forecast)**2
        tail_loss=-(tail*np.log(prob)+(1-tail)*np.log1p(-prob))
        tail_brier=(tail-prob)**2
        up_prob=np.clip(ndtr(mu/sd),1e-8,1-1e-8)
        sign=(yte>0).astype(float)
        sign_loss=-(sign*np.log(up_prob)+(1-sign)*np.log1p(-up_prob))
        score_rows.append({'horizon_min':horizon,'model':model,
                  'n_origins':len(te),'n_days':te.date.nunique(),
                  'nll':float(negll.mean()),
                  'rv_log_mse':float(rv_err.mean()),
                  'tail_logloss':float(tail_loss.mean()),
                  'tail_brier':float(tail_brier.mean()),
                  'sign_logloss':float(sign_loss.mean()),
                  'tail_count':int(tail.sum()),
                  'tail_forecast_avg':float(prob.mean()),
                  'tail_observed_avg':float(tail.mean())})
        pred_rows.append(pd.DataFrame({'date':te.date.to_numpy(),
                     'state_time':te.state_time.to_numpy(),
                     'underlying':te.underlying.to_numpy(),
                     'horizon_min':horizon,'model':model,
                     'nll_loss':negll,'rv_loss':rv_err,'tail_loss':tail_loss,
                     'tail_brier':tail_brier,'sign_loss':sign_loss},index=te.index))
    return pd.DataFrame(score_rows),pd.concat(pred_rows)


def paired_day_bootstrap(predictions:pd.DataFrame,comparison:tuple[str,str],
                         metric:str, horizon:int, B:int=DEFAULT_BOOT, seed:int=1301)->dict:
    a,b=comparison
    v=predictions[(predictions.horizon_min==horizon)&(predictions.model==a)]
    w=predictions[(predictions.horizon_min==horizon)&(predictions.model==b)]
    common=v.index.intersection(w.index)
    if len(common)<100:
        raise ValueError('Insufficient matched forecasts')
    v=v.loc[common];w=w.loc[common]
    if not np.array_equal(v.state_time.to_numpy(),w.state_time.to_numpy()):
        raise ValueError('Misaligned forecast origins')
    loss_map={'nll':'nll_loss','rv_log_mse':'rv_loss','tail_logloss':'tail_loss','tail_brier':'tail_brier','sign_logloss':'sign_loss'}
    key=loss_map[metric]
    delta=v[key].to_numpy()-w[key].to_numpy()
    days=v.date.to_numpy()
    ud,idx=np.unique(days,return_inverse=True)
    sums=np.bincount(idx,weights=delta,minlength=len(ud))
    counts=np.bincount(idx,minlength=len(ud))
    rng=np.random.default_rng(seed)
    draw=rng.integers(len(ud),size=(B,len(ud)))
    samples=sums[draw].sum(axis=1)/counts[draw].sum(axis=1)
    return {'comparison':f'{a} - {b}', 'metric':metric,'horizon_min':horizon,
        'n_origins':len(common),'test_days':len(ud),
        'delta':float(delta.mean()), 'q025':float(np.quantile(samples,.025)),
        'q975':float(np.quantile(samples,.975))}


def decide(scores:pd.DataFrame,paired:pd.DataFrame,frame:pd.DataFrame)->dict:
    p=paired[paired.horizon_min==PRIMARY].set_index(['comparison','metric'])
    def better(a,b,metric):
        key=(f'{a} - {b}',metric)
        if key not in p.index:return False
        z=p.loc[key]
        return bool(z.delta<0 and z.q975<0)
    enough_days=int(scores.loc[scores.horizon_min==PRIMARY,'n_days'].min())>=MIN_TEST_DAYS
    # Tail event timestamp counted once across synchronous assets and overlapping windows.
    tail_rows=frame.loc[(frame.split=='test') & frame.history_ok &
              (frame[f'tail_{PRIMARY}']==1)]
    distinct=int(tail_rows.state_time.nunique())
    enough=bool(enough_days and distinct>=MIN_DISTINCT_TAIL_TIMES)
    flags={'E_distribution_vs_K':better('E_two_ema','K_random_walk','nll'),
           'P_phase_vs_E_rv':better('P_vol_phase','E_two_ema','rv_log_mse'),
           'P_phase_vs_E_nll':better('P_vol_phase','E_two_ema','nll'),
           'P_phase_vs_E_tail':better('P_vol_phase','E_two_ema','tail_logloss'),
           'M_price_vs_P_nll':better('M_price_reversion','P_vol_phase','nll'),
           'PM_interaction_vs_M_nll':better('PM_conditional','M_price_reversion','nll')}
    return {'status':('INSUFFICIENT_DATA' if not enough else 'HISTORICAL_EXPLORATORY'),
        'historical_confirmation_permitted':False,
        'primary_horizon_min':PRIMARY,
        'n_test_days':int(scores.loc[scores.horizon_min==PRIMARY,'n_days'].min()),
        'n_unique_positive_tail_origin_times':distinct,
        'predeclared_flags':flags,
        'conditional_phase_distribution_support':bool(flags['P_phase_vs_E_nll'] and flags['P_phase_vs_E_tail']),
        'conditional_price_mean_support':bool(flags['M_price_vs_P_nll']),
        'reason':'Past data have been inspected in prior branches; fresh post-2026-10-13 days are required for confirmation.'}


def registration()->dict:
    return asdict(Parameters()) | {'version':VERSION,'source':'shared history/price_5m raw close files',
              'not_an_options_backtest':True, 'no_tradability_claim':True}
