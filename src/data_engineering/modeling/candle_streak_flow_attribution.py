"""CS03b: attribute out-of-sample spot streak-transition information in Binance USD-M flow.

Frozen CS03 spot/futures alignment and model objective. Four nested L2-regularized
multinomial logits isolate signed aggressive trade imbalance from activity
(quote notional and number of trades). This module never uses order-book OBI.
Historical test dates were examined in CS03: any positive result is exploratory.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Mapping
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from data_engineering.modeling import candle_streak_trade_flow as tf

VERSION = 'CS03b-0.1.0'
CLASSES = (-1, 0, 1)
MODEL_NAMES = ('price_vol', 'price_vol_signed', 'price_vol_activity', 'price_vol_both')
BASIC = tf.BASE_FEATURES['sign_streak_vol']
SIGNED = ('flow5', 'flow30', 'flow5_by_sign', 'flow30_by_sign')
ACTIVITY = ('log_quote_volume', 'log_trades')
FEATURES = {
    'price_vol': BASIC,
    'price_vol_signed': BASIC + SIGNED,
    'price_vol_activity': BASIC + ACTIVITY,
    'price_vol_both': BASIC + SIGNED + ACTIVITY,
}
assert FEATURES['price_vol_both'] == tf.BASE_FEATURES['sign_streak_vol_flow']
COMPARISONS = (
    ('price_vol_signed', 'price_vol'),
    ('price_vol_activity', 'price_vol'),
    ('price_vol_both', 'price_vol'),
    ('price_vol_both', 'price_vol_signed'),
    ('price_vol_both', 'price_vol_activity'),
    ('price_vol_signed', 'price_vol_activity'),
)
METRICS = ('threeclass_logloss', 'threeclass_brier',
           'binary_logloss', 'binary_brier',
           'flat_vs_move_logloss', 'flat_vs_move_brier',
           'reversal_logloss', 'reversal_brier',
           'start_up_logloss', 'start_up_brier')

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
    max_iter: int = 120
    tol: float = 1e-7
    bootstrap_reps: int = 1000
    shuffle_reps: int = 100
    min_days_ci: int = 20
    min_events_ci: int = 200
    seed: int = 2605
    status: str = 'HISTORICAL_EXPLORATORY'


def registration() -> dict:
    return asdict(Specification())


def fit_ablation(train: pd.DataFrame, *, C:float=.25, max_iter:int=120,
                 tol:float=1e-7) -> dict:
    """Same conditional model, sample, standardization, C and solver as CS03 v0.1.1."""
    x = train.loc[train.scorable].copy()
    if len(x) < 200: raise ValueError('insufficient train rows')
    y = x.next_sign.to_numpy(int)
    if set(np.unique(y)) != set(CLASSES):
        raise ValueError('all DOWN/ZERO/UP outcomes are required in TRAIN')
    design = tf.raw_design(x)
    if not np.isfinite(design.to_numpy(float)).all():
        raise ValueError('nonfinite predictors')
    fitted = {}
    for name in MODEL_NAMES:
        terms = FEATURES[name]
        scaler = StandardScaler()
        z = scaler.fit_transform(design[list(terms)].to_numpy(float))
        model = LogisticRegression(C=C, max_iter=max_iter, tol=tol, solver='newton-cholesky')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(z, y)
        failures = [str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
        if failures or int(model.n_iter_[0]) >= max_iter:
            raise RuntimeError(f'{name} did not converge after {model.n_iter_[0]} iterations: {failures}')
        if not np.array_equal(model.classes_, np.array(CLASSES)):
            raise ValueError('class mismatch')
        fitted[name] = (terms, scaler, model)
    return {'models': fitted, 'n_train': int(len(x)), 'C': C, 'solver': 'newton-cholesky'}


def predict_ablation(frame:pd.DataFrame, fit:dict) -> dict[str,np.ndarray]:
    design = tf.raw_design(frame)
    out = {}
    for name,(terms,scaler,model) in fit['models'].items():
        p = model.predict_proba(scaler.transform(design[list(terms)].to_numpy(float)))
        if (not np.isfinite(p).all() or p.shape != (len(frame),3)
                or not np.allclose(p.sum(axis=1),1.0)):
            raise ValueError('bad output probabilities')
        out[name] = p
    return out


def _binary_loss(y:np.ndarray,p:np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    if len(y) != len(p) or not np.isin(y,[0,1]).all(): raise ValueError('bad binary outcomes')
    q = np.clip(np.asarray(p,float),1e-12,1.-1e-12)
    return -(y*np.log(q)+(1-y)*np.log1p(-q)),(q-y)**2


def origin_losses(events:pd.DataFrame,pred:Mapping[str,np.ndarray]) -> pd.DataFrame:
    """One row per scored origin, per model, per prespecified scoring task.

    Three-class losses always include zero targets. Binary and reversal/start
    diagnostics condition on next nonzero (unknown at decision time). Flat
    termination uses all current-running origins and P(next=ZERO).
    """
    if not bool(events.scorable.all()):raise ValueError('events must be prefiltered to scorable origins')
    y = events.next_sign.to_numpy(int)
    if not np.isin(y,CLASSES).all():raise ValueError('invalid labels')
    n = len(events); dates = pd.to_datetime(events.state_time,utc=True).dt.floor('D').to_numpy()
    current = events.sign.to_numpy(int)
    alive = current != 0
    nonzero = y != 0
    idx = np.searchsorted(CLASSES,y)
    rows = np.arange(n)
    pieces = []
    for name,p in pred.items():
        if p.shape != (n,3) or not np.allclose(p.sum(axis=1),1):raise ValueError('invalid prediction')
        pp = np.clip(p,1e-12,1-1e-12)
        probs = p[:,2]/(p[:,0]+p[:,2])
        onehot = np.eye(3)[idx]
        multiclass = -np.log(pp[rows,idx])
        brier = ((p-onehot)**2).sum(axis=1)
        stop_ll,stop_brier = _binary_loss((y==0).astype(int),p[:,1])
        for metric,mask,loss in (
            ('threeclass_logloss',np.ones(n,bool),multiclass),
            ('threeclass_brier',np.ones(n,bool),brier),
            ('binary_logloss',nonzero,_binary_loss((y==1).astype(int),probs)[0]),
            ('binary_brier',nonzero,_binary_loss((y==1).astype(int),probs)[1]),
            ('flat_vs_move_logloss',alive,stop_ll),
            ('flat_vs_move_brier',alive,stop_brier),
            ('reversal_logloss',alive&nonzero,
             _binary_loss((y!=current).astype(int),np.where(current==1,1-probs,probs))[0]),
            ('reversal_brier',alive&nonzero,
             _binary_loss((y!=current).astype(int),np.where(current==1,1-probs,probs))[1]),
            ('start_up_logloss',~alive&nonzero,_binary_loss((y==1).astype(int),probs)[0]),
            ('start_up_brier',~alive&nonzero,_binary_loss((y==1).astype(int),probs)[1]),
        ):
            if mask.any():
                pieces.append(pd.DataFrame({'date':dates[mask], 'model':name,
                                             'metric':metric,'loss':loss[mask]}))
    return pd.concat(pieces,ignore_index=True)


def scores_from_losses(losses:pd.DataFrame) -> pd.DataFrame:
    return losses.groupby(['model','metric'],sort=False).agg(n=('loss','size'),score=('loss','mean'),
        n_days=('date','nunique')).reset_index()


def day_bootstrap(losses:pd.DataFrame,*,B:int=1000,seed:int=2605,
                  min_days:int=20,min_events:int=200) -> pd.DataFrame:
    """Paired entire-UTC-day resampling; missing/rare strata reported, not dropped silently."""
    rng=np.random.default_rng(seed);out=[]
    for metric in METRICS:
        m=losses.loc[losses.metric==metric]
        if m.empty:continue
        q=m.groupby(['date','model'],sort=False).agg(loss=('loss','mean'),n=('loss','size')).unstack('model')
        for model,reference in COMPARISONS:
            if ('loss',model) not in q or ('loss',reference) not in q:continue
            a=q[('loss',model)];b=q[('loss',reference)]
            valid=a.notna()&b.notna()
            delta=(a[valid]-b[valid]).to_numpy(float)
            n_a=q[('n',model)][valid].to_numpy(float)
            n_b=q[('n',reference)][valid].to_numpy(float)
            if not np.array_equal(n_a,n_b):raise ValueError('nonpaired event grid')
            days=len(delta);n=int(n_a.sum())
            if days >= min_days and n >= min_events:
                ix=rng.integers(0,days,size=(B,days))
                draws=delta[ix].mean(axis=1)
                low,high=np.quantile(draws,[.025,.975])
                status='OK'
            else:
                low,high=np.nan,np.nan
                status='SPARSE'
            out.append({'model':model,'reference':reference,'metric':metric,
                'delta_equal_day':float(delta.mean()) if days else np.nan,
                'delta_event_weighted':float(np.average(delta,weights=n_a)) if days else np.nan,
                'ci025':float(low),'ci975':float(high),'n_days':days,'n_origins':n,
                'min_origins_day':int(n_a.min()) if days else 0,
                'max_origins_day':int(n_a.max()) if days else 0,'status':status})
    return pd.DataFrame(out)


def transition_calibration(events:pd.DataFrame, preds:Mapping[str,np.ndarray],
                           cuts:list[float]) -> pd.DataFrame:
    """Diagnostic only. Original TRAIN terciles; TEST outcome cells are descriptive."""
    if not events.scorable.all():raise ValueError('requires scorable events')
    frame=events.copy().reset_index(drop=True)
    frame['flow_tier']=np.searchsorted(cuts,frame.flow_5m.to_numpy(float),'right')
    frame['origin']=np.where(frame.sign==0,'flat_origin','running_streak')
    frame['age_group']=pd.cut(frame.length,bins=[-1,0,1,2,3,5,8,np.inf],
                            labels=['0 flat','1','2','3','4-5','6-8','9+'])
    frames=[]
    for name,prob in preds.items():
        z=frame.copy();z['model']=name
        z['p_continue']=np.where(z.sign==1,prob[:,2],np.where(z.sign==-1,prob[:,0],np.nan))
        z['p_reversal']=np.where(z.sign==1,prob[:,0],np.where(z.sign==-1,prob[:,2],np.nan))
        z['p_flat']=prob[:,1]
        z['p_start_up']=np.where(z.sign==0,prob[:,2],np.nan)
        z['obs_continue']=(z.outcome=='continue').astype(int)
        z['obs_reversal']=(z.outcome=='reverse_new_streak').astype(int)
        z['obs_flat']=(z.next_sign==0).astype(int)
        z['obs_start_up']=(z.outcome=='start_up_after_flat').astype(int)
        q=z.groupby(['model','origin','sign','age_group','flow_tier'],observed=True).agg(
            n=('next_sign','size'),p_continue=('p_continue','mean'),
            p_reversal=('p_reversal','mean'),p_flat=('p_flat','mean'),
            p_start_up=('p_start_up','mean'),obs_continue=('obs_continue','mean'),
            obs_reversal=('obs_reversal','mean'),obs_flat=('obs_flat','mean'),
            obs_start_up=('obs_start_up','mean')).reset_index()
        frames.append(q)
    return pd.concat(frames,ignore_index=True)


def feature_shuffle_null(test:pd.DataFrame,fit:dict,*,reps:int=100,seed:int=2605) -> pd.DataFrame:
    """Hour-stratified label alignment check, not a significance/p-value test."""
    test=test.loc[test.scorable].copy().reset_index(drop=True)
    y=np.searchsorted(CLASSES,test.next_sign.to_numpy(int));idx=np.arange(len(test))
    original=predict_ablation(test,fit)
    baseline=-np.log(np.maximum(original['price_vol'][idx,y],1e-12)).mean()
    rng=np.random.default_rng(seed)
    groups=test.groupby(test.state_time.dt.hour).indices
    rows=[]
    for kind,fields,model in (
        ('signed',('flow_5m','flow_30m'),'price_vol_signed'),
        ('activity',('log_quote_volume','log_trades'),'price_vol_activity')):
        obs=-np.log(np.maximum(original[model][idx,y],1e-12)).mean()-baseline
        data=test[list(fields)].to_numpy(float)
        null=[]
        for _ in range(reps):
            z=test.copy(); shuffled=data.copy()
            for ids in groups.values():shuffled[ids]=data[rng.permutation(ids)]
            z[list(fields)]=shuffled
            terms,scaler,classifier=fit['models'][model]
            d=tf.raw_design(z)
            p=classifier.predict_proba(scaler.transform(d[list(terms)].to_numpy(float)))
            null.append(float(-np.log(np.maximum(p[idx,y],1e-12)).mean()-baseline))
        rows.append({'component':kind,'model':model,'observed_delta':float(obs),
                     'null05':float(np.quantile(null,.05)),
                     'null_median':float(np.median(null)),
                     'null95':float(np.quantile(null,.95)),
                     'fraction_null_le_observed':float(np.mean(np.asarray(null)<=obs)),
                     'n_shuffle':reps})
    return pd.DataFrame(rows)


def decision(boot:pd.DataFrame, assets:tuple=('BTC','ETH'),*,synthetic=False) -> dict:
    """Frozen exploratory gates; no gate silently relaxes due to missing assets."""
    if synthetic:return {'verdict':'SYNTHETIC_NO_MARKET_INFERENCE','status':'SYNTHETIC',
                          'primary':[],'AM_T_HAR_Blend_modified':False}
    tests=[]
    for asset in assets:
        q=boot[(boot.asset==asset)&(boot.period=='test')]
        for component,model,ref in (
            ('signed_vs_price','price_vol_signed','price_vol'),
            ('activity_vs_price','price_vol_activity','price_vol'),
            ('signed_given_activity','price_vol_both','price_vol_activity'),
            ('activity_given_signed','price_vol_both','price_vol_signed'),
        ):
            for metric in ('threeclass_logloss','binary_logloss'):
                r=q[(q.metric==metric)&(q.model==model)&(q.reference==ref)]
                ok=len(r)==1 and r.iloc[0].status=='OK'
                tests.append({'asset':asset,'component':component,'metric':metric,
                    'delta':float(r.iloc[0].delta_equal_day) if ok else None,
                    'upper95':float(r.iloc[0].ci975) if ok else None,
                    'pass':bool(ok and r.iloc[0].ci975<0)})
    def gate(part,metric):return all(t['pass'] for t in tests if t['component']==part and t['metric']==metric) and sum(t['component']==part and t['metric']==metric for t in tests)==len(assets)
    three_signed=gate('signed_vs_price','threeclass_logloss') and gate('signed_given_activity','threeclass_logloss')
    binary_signed=gate('signed_vs_price','binary_logloss') and gate('signed_given_activity','binary_logloss')
    three_activity=gate('activity_vs_price','threeclass_logloss') and gate('activity_given_signed','threeclass_logloss')
    verdict=('HISTORICAL_SIGNED_FLOW_CANDIDATE' if three_signed and binary_signed else
             'HISTORICAL_THREECLASS_SIGNED_CANDIDATE' if three_signed else
             'HISTORICAL_ACTIVITY_CANDIDATE' if three_activity else
             'NO_CONFIRMED_INCREMENTAL_ATTRIBUTION')
    if any(t['delta'] is None for t in tests):verdict='INSUFFICIENT_PRIMARY_EVALUATION'
    return {'verdict':verdict,'status':'HISTORICAL_EXPLORATORY',
            'signed_threeclass_gate':three_signed,'signed_binary_gate':binary_signed,
            'activity_threeclass_gate':three_activity,'primary':tests,
            'prospective_confirmation':False,'AM_T_HAR_Blend_modified':False,
            'note':'Previously inspected CS03 TEST period; no independent confirmation. '
                   'Binary scores condition on immediate next return being nonzero.'}
