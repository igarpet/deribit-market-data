"""CS04-A: causal family-subset discovery on existing spot 5m / USD-M 1m data.

Exploratory only. A feature at bar end t never uses data after t; forward
log-return and |return| targets are constructed separately on (t,t+h].
No previous CS03 TEST is used to select combinations. Magnitude targets and
forecast scores stay independent of three-class sign scores.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from itertools import combinations
import hashlib
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from data_engineering.modeling import candle_streak_trade_flow as tf

VERSION = 'CS04A-0.1.0'
HORIZONS = (5, 15, 30)
DAY_BARS = 288
FAMILY_COLUMNS = {
    'momentum': ('mom_r5', 'mom_r15_mean', 'mom_r60_mean', 'mom_accel'),
    'candle_geometry': ('candle_body', 'candle_range', 'candle_upper_wick', 'candle_lower_wick'),
    'minute_path': ('last_1m_r', 'minute_rv', 'minute_accel'),
    'signed_flow': ('flow5', 'flow30', 'flow5_by_sign', 'flow30_by_sign'),
    'flow_timing': ('flow_early', 'flow_late', 'flow_accel'),
    'trade_activity': ('log_quote_volume', 'log_trades', 'log_avg_trade'),
    'flow_price_impact': ('flow_price_divergence', 'flow_return_alignment'),
    'spot_futures_basis': ('basis_bps', 'basis_change_bps', 'futures_spot_return_gap'),
    'peer_market': ('peer_ret5', 'peer_flow5'),
    'utc_calendar': ('clock_sin', 'clock_cos', 'weekday_sin', 'weekday_cos'),
}
BASE = ('s_down','s_up','age_signed','age_abs','log_rv24h','log_fastslow')
FAMILY_NAMES = tuple(FAMILY_COLUMNS)

@dataclass(frozen=True)
class Specification:
    version: str = VERSION
    assets: tuple[str,...] = ('BTC','ETH','SOL','XRP')
    cutoff_utc: str = '2026-10-14T00:00:00Z'
    discovery_fraction: float = 0.8
    fold_positions: tuple[float,...] = (.50,.64,.78)
    validation_days_fraction: float = .08
    embargo_days: int = 1
    family_max_order: int = 3
    C: float = .25
    ridge_alpha: float = 10.
    logistic_tol: float = 1e-6
    logistic_max_iter: int = 120
    seed: int = 2606
    status: str = 'HISTORICAL_DISCOVERY_NOT_CONFIRMATION'


def registration(): return asdict(Specification())


def combinations_frozen(max_order=3):
    if not 1<=max_order<=len(FAMILY_NAMES):raise ValueError('invalid family order')
    return [tuple(c) for k in range(1,max_order+1) for c in combinations(FAMILY_NAMES,k)]


def combo_id(families):
    return 'base' if not families else '+'.join(families)


def validate_1m(frame:pd.DataFrame) -> pd.DataFrame:
    required={'open_time','open','high','low','close','quote_volume',
              'taker_buy_quote_volume','trades'}
    if not required.issubset(frame):raise ValueError(f'missing minute columns: {required-set(frame)}')
    z=frame[list(sorted(required))].copy()
    z['open_time']=pd.to_datetime(z.open_time,utc=True)
    if z.open_time.isna().any() or z.open_time.duplicated().any():raise ValueError('bad minute timestamps')
    z=z.sort_values('open_time').reset_index(drop=True)
    if (z.open_time.dt.second.ne(0)|z.open_time.dt.microsecond.ne(0)).any():
        raise ValueError('not a minute-open grid')
    for c in required-{'open_time'}:z[c]=pd.to_numeric(z[c],errors='coerce')
    if not np.isfinite(z[list(required-{'open_time'})].to_numpy(float)).all():
        raise ValueError('nonfinite minute')
    if (z[['open','high','low','close']]<=0).any().any():raise ValueError('invalid price')
    if ((z.high<z[['open','close','low']].max(axis=1))|
        (z.low>z[['open','close','high']].min(axis=1))).any():raise ValueError('OHLC inconsistent')
    if ((z[['quote_volume','taker_buy_quote_volume','trades']]<0).any().any()|
        (z.taker_buy_quote_volume>z.quote_volume+1e-6).any()):raise ValueError('invalid trades')
    return z


def minute_to_five(frame:pd.DataFrame):
    """Full 5 x 1-minute buckets only; OHLC and flow are observed by end t."""
    z=validate_1m(frame)
    core=tf.one_minute_futures_flow(z)
    z['state_time']=z.open_time.dt.floor('5min')+tf.STEP
    z['slot']=z.open_time.dt.minute%5
    z['buyfrac']=np.where(z.quote_volume.gt(0),
                          (2*z.taker_buy_quote_volume-z.quote_volume)/z.quote_volume,0.)
    z['one_minute_r']=np.log(z.close/z.open)
    z['four_quote']=np.where(z.slot.lt(4),z.quote_volume,0.)
    z['four_signed']=np.where(z.slot.lt(4),2*z.taker_buy_quote_volume-z.quote_volume,0.)
    z['last_quote']=np.where(z.slot.eq(4),z.quote_volume,0.)
    z['last_signed']=np.where(z.slot.eq(4),2*z.taker_buy_quote_volume-z.quote_volume,0.)
    z['rv_term']=z.one_minute_r**2
    q=z.groupby('state_time',sort=True).agg(
        f_open=('open','first'),f_high=('high','max'),f_low=('low','min'),
        f_close=('close','last'),last_1m_r=('one_minute_r','last'),
        minute_rv=('rv_term','sum'),mean_minute_r=('one_minute_r','mean'),
        four_quote=('four_quote','sum'),four_signed=('four_signed','sum'),
        last_quote=('last_quote','sum'),last_signed=('last_signed','sum'))
    q=q.reset_index()
    out=core.merge(q,on='state_time',how='left',validate='one_to_one')
    den=(out.f_high-out.f_low).replace(0,np.nan)
    out['futures_r5']=np.log(out.f_close/out.f_open)
    out['candle_body']=out.futures_r5
    out['candle_range']=(out.f_high-out.f_low)/out.f_open
    out['candle_upper_wick']=(out.f_high-out[['f_close','f_open']].max(axis=1))/den
    out['candle_lower_wick']=(out[['f_close','f_open']].min(axis=1)-out.f_low)/den
    out[['candle_upper_wick','candle_lower_wick']]=out[['candle_upper_wick','candle_lower_wick']].fillna(0)
    out['minute_accel']=out.last_1m_r-out.mean_minute_r
    out['flow_early']=out.four_signed/out.four_quote.replace(0,np.nan)
    out['flow_late']=out.last_signed/out.last_quote.replace(0,np.nan)
    out['flow_accel']=out.flow_late-out.flow_early
    out['log_avg_trade']=np.log1p(out.quote_volume/out.trade_count.replace(0,np.nan))
    # Incomplete 5-minute bucket MUST NOT produce derived features.
    derived=['f_open','f_high','f_low','f_close','last_1m_r','minute_rv',
             'mean_minute_r','futures_r5','candle_body','candle_range',
             'candle_upper_wick','candle_lower_wick','minute_accel',
             'flow_early','flow_late','flow_accel','log_avg_trade']
    out.loc[~out.flow_valid,derived]=np.nan
    return out


def _rolling(x:np.ndarray,t:pd.Series,k:int):
    return tf._contiguous_roll(x,t,k)


def attach_features(spot:pd.DataFrame,flow5:pd.DataFrame,spot_close:pd.DataFrame,
                    peer5:pd.DataFrame | None=None, *,fast_bars=12,slow_bars=288):
    """Every family computed on/through t, with strictly timestamp-matched peer."""
    x=tf.build_origin_panel(spot,flow5,fast_bars=fast_bars,slow_bars=slow_bars,flow_bars=6)
    c=spot_close[['state_time','close']].rename(columns={'close':'spot_close'}).copy()
    c.state_time=pd.to_datetime(c.state_time,utc=True)
    if c.state_time.duplicated().any():raise ValueError('duplicate spot close')
    x=x.merge(c,on='state_time',how='left',validate='one_to_one')
    t=x.state_time
    r=x.return_5m.to_numpy(float)
    x['mom_r5']=r
    x['mom_r15_mean']=_rolling(r,t,3)
    x['mom_r60_mean']=_rolling(r,t,12)
    x['mom_accel']=r-x.mom_r15_mean
    x['flow5']=x.flow_5m
    x['flow30']=x.flow_30m
    x['flow5_by_sign']=x.flow5*x.sign
    x['flow30_by_sign']=x.flow30*x.sign
    x['flow_price_divergence']=x.flow5-np.tanh(x.futures_r5.to_numpy(float)/
                                                   np.maximum(np.sqrt(x.rv_slow.to_numpy(float)),1e-10))
    x['flow_return_alignment']=x.flow5*np.sign(x.futures_r5)
    x['basis_bps']=1e4*(x.f_close/x.spot_close-1)
    priorbasis=x.basis_bps.shift()
    contiguous=t.diff().eq(tf.STEP)
    x['basis_change_bps']=(x.basis_bps-priorbasis).where(contiguous)
    x['futures_spot_return_gap']=x.futures_r5-x.return_5m
    if peer5 is None:
        x['peer_ret5']=np.nan;x['peer_flow5']=np.nan
    else:
        peer=peer5[['state_time','futures_r5','flow_5m','flow_valid']].rename(columns={
            'futures_r5':'peer_ret5','flow_5m':'peer_flow5','flow_valid':'peer_valid'})
        x=x.merge(peer,on='state_time',how='left',validate='one_to_one')
        x.loc[~x.peer_valid.astype('boolean').fillna(False),['peer_ret5','peer_flow5']]=np.nan
    hour=t.dt.hour.to_numpy(float)+t.dt.minute.to_numpy(float)/60.
    dow=t.dt.dayofweek.to_numpy(float)
    x['clock_sin']=np.sin(2*np.pi*hour/24)
    x['clock_cos']=np.cos(2*np.pi*hour/24)
    x['weekday_sin']=np.sin(2*np.pi*dow/7)
    x['weekday_cos']=np.cos(2*np.pi*dow/7)
    x['s_down']=(x.sign==-1).astype(float)
    x['s_up']=(x.sign==1).astype(float)
    age=np.log1p(x.length.astype(float))
    x['age_signed']=x.sign*age
    x['age_abs']=age
    all_features=list(BASE)+[col for family in FAMILY_NAMES for col in FAMILY_COLUMNS[family]]
    if len(set(all_features))!=len(all_features):raise AssertionError('duplicate family columns')
    x['feature_ready']=x.features_ready & np.isfinite(x[all_features].to_numpy(float)).all(axis=1)
    return x


def attach_targets(panel:pd.DataFrame,horizons=HORIZONS):
    """Future NET log returns, not intrabar sums of absolute returns.

    Same-day condition avoids boundary and fold-crossing targets, and a missing
    intermediate spot candle invalidates the entire future horizon.
    """
    x=panel.copy();t=pd.to_datetime(x.state_time,utc=True)
    r=x.return_5m.to_numpy(float);T=t.to_numpy();n=len(x)
    for horizon in horizons:
        if horizon%5:raise ValueError('non-five-minute horizon')
        k=horizon//5
        acc=np.zeros(n,dtype=float);acc_sq=np.zeros(n,dtype=float)
        acc_abs=np.zeros(n,dtype=float);ok=np.ones(n,dtype=bool)
        for j in range(1,k+1):
            future=np.full(n,np.nan);future[:n-j]=r[j:]
            diff=np.zeros(n,dtype=bool)
            diff[:n-j]=(T[j:]-T[:n-j])==np.timedelta64(j*5,'m')
            ok &= np.isfinite(future)&diff
            acc+=np.nan_to_num(future,nan=0.)
            acc_sq+=np.nan_to_num(future**2,nan=0.)
            acc_abs+=np.nan_to_num(np.abs(future),nan=0.)
        same_day=((t+pd.Timedelta(minutes=horizon)).dt.floor('D')==t.dt.floor('D')).to_numpy()
        acc=np.where(ok&same_day,acc,np.nan)
        name=f'ret_{horizon}m'
        x[name]=acc
        x[f'abs_{horizon}m']=np.abs(acc)
        x[f'sq_{horizon}m']=acc**2
        x[f'upside_{horizon}m']=np.maximum(acc,0)
        x[f'downside_{horizon}m']=np.maximum(-acc,0)
        x[f'path_rv_{horizon}m']=np.where(ok&same_day,acc_sq,np.nan)
        x[f'path_abs_sum_{horizon}m']=np.where(ok&same_day,acc_abs,np.nan)
    x['target_5m_sign']=np.where(np.isfinite(x.ret_5m),np.sign(x.ret_5m),np.nan)
    x['all_targets_ready']=np.isfinite(x[[f'ret_{h}m' for h in horizons]].to_numpy(float)).all(axis=1)
    return x


def eligible_complete_days(panel:pd.DataFrame):
    """Complete input days; the final six target origins/day are intentionally censored."""
    x=panel.copy();x['date']=pd.to_datetime(x.state_time,utc=True).dt.floor('D')
    q=x.groupby('date').agg(n=('state_time','size'),start=('state_time','min'),
                            finish=('state_time','max'),ready=('feature_ready','sum'))
    date=q.index.to_series()
    keep=q.n.eq(DAY_BARS)&q.ready.eq(DAY_BARS)&q.start.eq(date)&q.finish.eq(date+pd.Timedelta(hours=23,minutes=55))
    return pd.DatetimeIndex(q.index[keep])


def folds_from_days(days:pd.DatetimeIndex,cfg:Specification):
    days=pd.DatetimeIndex(sorted(days))
    disc=days[:int(len(days)*cfg.discovery_fraction)]
    if len(disc)<48:raise ValueError('Too few complete historical discovery days')
    width=max(5,int(len(disc)*cfg.validation_days_fraction))
    result=[]
    for index,f in enumerate(cfg.fold_positions,1):
        s=int(len(disc)*f)
        e=min(s+width,len(disc))
        stop=s-cfg.embargo_days
        if stop<15 or e-s<5:raise ValueError('Fold too short')
        result.append({'fold':index,'train_days':disc[:stop],
                       'validation_days':disc[s:e],
                       'train_end_exclusive':str(disc[stop]),
                       'validation_start':str(disc[s]),
                       'validation_end_inclusive':str(disc[e-1]),
                       'embargo_days':cfg.embargo_days})
    return result


def _model_scores(y,p):
    actual=np.searchsorted([-1,0,1],y.astype(int))
    if p.shape!=(len(y),3) or not np.isfinite(p).all() or not np.allclose(p.sum(axis=1),1):
        raise ValueError('invalid class forecasts')
    pp=np.clip(p,1e-12,1.)
    nll=-np.log(pp[np.arange(len(y)),actual])
    brier=((p-np.eye(3)[actual])**2).sum(axis=1)
    mask=y!=0
    up=np.clip(p[:,2]/(p[:,0]+p[:,2]),1e-12,1-1e-12)
    binary=-np.log(np.where(y[mask]>0,up[mask],1-up[mask]))
    return {'threeclass_logloss':float(nll.mean()),'threeclass_brier':float(brier.mean()),
            'binary_logloss':float(binary.mean()) if mask.any() else np.nan,
            'observed_zero':float((y==0).mean()),'forecast_zero':float(p[:,1].mean()),
            'n_nonzero':int(mask.sum())}


def fit_score(X_train,X_val,y_train,y_val,mag_train,mag_val,cfg:Specification):
    """TRAIN-standardized multinomial L2 logit and separate mean-|return| Ridge.

    Magnitude scores are in basis points, one Ridge per horizon. No target-side
    parameters estimated from validation. No reweighting of rare zero classes.
    """
    if set(np.unique(y_train))!={-1,0,1}:raise ValueError('TRAIN missing sign class')
    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():raise ValueError('nonfinite feature')
    scaler=StandardScaler()
    a=scaler.fit_transform(X_train);b=scaler.transform(X_val)
    model=LogisticRegression(C=cfg.C,max_iter=cfg.logistic_max_iter,tol=cfg.logistic_tol,solver='newton-cholesky')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        model.fit(a,y_train)
    if any(issubclass(w.category,ConvergenceWarning) for w in caught) or model.n_iter_[0]>=cfg.logistic_max_iter:
        raise RuntimeError('CS04-A logistic solver did not converge; no silent tuning')
    prob=model.predict_proba(b)
    if not np.array_equal(model.classes_,np.array([-1,0,1])):raise ValueError('class order mismatch')
    scores=_model_scores(y_val,prob)
    for h in HORIZONS:
        target=f'abs_{h}m'
        fit=np.asarray(mag_train[target],float)*1e4
        truth=np.asarray(mag_val[target],float)*1e4
        ridge=Ridge(alpha=cfg.ridge_alpha).fit(a,fit)
        pred=np.maximum(ridge.predict(b),0)
        scores[f'abs_{h}m_mae_bps']=float(np.mean(np.abs(pred-truth)))
        scores[f'abs_{h}m_mse_bps2']=float(np.mean((pred-truth)**2))
        scores[f'abs_{h}m_forecast_mean_bps']=float(pred.mean())
        scores[f'abs_{h}m_observed_mean_bps']=float(truth.mean())
        scores[f'abs_{h}m_rmse_bps']=float(np.sqrt(scores[f'abs_{h}m_mse_bps2']))
    scores['n_train']=len(y_train);scores['n_validation']=len(y_val)
    return scores


def scan_asset(panel:pd.DataFrame,cfg:Specification,*,max_combos=None,checkpoint:Path | None=None):
    """Walk-forward combination scan with optional resumable per-asset checkpoint.

    Checkpoint row is committed only after all scores for the fit are available.
    Compatible with restart if (and only if) the saved version, asset and frozen
    combination set match; no partial held-out predictions saved to repository.
    """
    if panel.asset.nunique()!=1:raise ValueError('one asset per scan')
    asset=str(panel.asset.iloc[0]);dates=eligible_complete_days(panel)
    folds=folds_from_days(dates,cfg)
    combos=[()] + combinations_frozen(cfg.family_max_order)
    if max_combos is not None:combos=combos[:max_combos+1]
    cols=list(BASE)+[c for f in FAMILY_NAMES for c in FAMILY_COLUMNS[f]]
    if not np.isfinite(panel.loc[panel.feature_ready,cols].to_numpy(float)).all():
        raise ValueError('ready predictor corrupted')
    rows=[]
    if checkpoint is not None and checkpoint.exists():
        old=pd.read_csv(checkpoint)
        required={'version','asset','fold','combination','n_train','n_validation'}
        if not required.issubset(old):raise ValueError('incompatible checkpoint schema')
        if not (old.version.eq(VERSION)&old.asset.eq(asset)).all():raise ValueError('stale checkpoint; refuse reuse')
        known={combo_id(c) for c in combos}
        if not set(old.combination).issubset(known):raise ValueError('checkpoint combination mismatch')
        if old[['fold','combination']].duplicated().any():raise ValueError('duplicate checkpoint')
        rows=old.to_dict('records')
    existing={(int(z['fold']),z['combination']) for z in rows}
    for fold in folds:
        tr=panel.loc[panel.date.isin(fold['train_days'])&panel.feature_ready&panel.all_targets_ready]
        va=panel.loc[panel.date.isin(fold['validation_days'])&panel.feature_ready&panel.all_targets_ready]
        if len(tr)<500 or len(va)<100:raise ValueError('insufficient folds')
        ytr=tr.target_5m_sign.to_numpy(int);yva=va.target_5m_sign.to_numpy(int)
        for combo in combos:
            name=combo_id(combo)
            if (fold['fold'],name) in existing:continue
            terms=list(BASE)+[c for f in combo for c in FAMILY_COLUMNS[f]]
            score=fit_score(tr[terms].to_numpy(float),va[terms].to_numpy(float),
                            ytr,yva,tr,va,cfg)
            rows.append({'version':VERSION,'asset':asset,'fold':fold['fold'],
                         'combination':name,'family_count':len(combo),
                         'n_features':len(terms),**score})
            if checkpoint is not None:
                checkpoint.parent.mkdir(parents=True,exist_ok=True)
                tmp=checkpoint.with_suffix('.tmp')
                pd.DataFrame(rows).to_csv(tmp,index=False)
                tmp.replace(checkpoint)
    return pd.DataFrame(rows),folds,dates


def rank_combinations(scores:pd.DataFrame):
    """Each candidate compared only with its SAME-ASSET, SAME-FOLD baseline.

    Cross asset aggregation equal-weights asset-fold comparisons: exploratory
    ranking, not a confidence bound nor selection-adjusted p value.
    """
    ref=scores.loc[scores.combination.eq('base')].set_index(['asset','fold'])
    metric=['threeclass_logloss','binary_logloss']+[f'abs_{h}m_mse_bps2' for h in HORIZONS]
    z=scores.loc[~scores.combination.eq('base')].copy()
    if z.empty:return z
    for m in metric:
        base=ref[m].reindex(pd.MultiIndex.from_frame(z[['asset','fold']])).to_numpy(float)
        if not np.isfinite(base).all():raise ValueError('missing same-fold benchmark')
        z['delta_'+m]=z[m].to_numpy(float)-base
    agg={'family_count':'first','n_features':'first','n_train':'min','n_validation':'min'}
    for m in metric:agg['delta_'+m]='mean'
    result=z.groupby('combination',as_index=False).agg(agg)
    result['n_asset_folds']=z.groupby('combination').size().to_numpy()
    result['n_assets']=z.groupby('combination').asset.nunique().to_numpy()
    result['direction_better_folds']=(z.delta_threeclass_logloss<0).groupby(z.combination).sum().reindex(result.combination).to_numpy()
    result['magnitude_5m_better_folds']=(z.delta_abs_5m_mse_bps2<0).groupby(z.combination).sum().reindex(result.combination).to_numpy()
    return result.sort_values(['delta_threeclass_logloss','combination']).reset_index(drop=True)


def coverage_manifest(panel:pd.DataFrame,full_dates,folds,cfg):
    return {'version':VERSION,'asset':str(panel.asset.iloc[0]),
            'complete_days':len(full_dates),'first_complete':str(full_dates.min()),
            'last_complete':str(full_dates.max()),'folds':[
                {'fold':x['fold'],'n_train_days':len(x['train_days']),
                 'n_validation_days':len(x['validation_days']),
                 'train_end_exclusive':x['train_end_exclusive'],
                 'validation_start':x['validation_start'],
                 'validation_end_inclusive':x['validation_end_inclusive'],
                 'embargo_days':x['embargo_days']} for x in folds],
            'features':{k:list(v) for k,v in FAMILY_COLUMNS.items()},
            'target_horizons_minutes':list(HORIZONS),
            'target_units':'log return; 1e4-scaled absolute log returns for magnitude scores',
            'historical_test_reused_for_selection':False,
            'research_status':cfg.status}
