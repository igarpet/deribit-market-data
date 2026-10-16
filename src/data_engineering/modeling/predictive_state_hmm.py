"""PSH01: predictive-evidence-gated volatility HMM; perps and options are separate endpoints.

Research-only: no trading/order placement and no modifications to frozen models.
Cross-branch reuse: VD12 causal 30m/240m memory and RD02 Student-t HMM,
authorized by project owner 2026-10-17. Historical results are exploratory.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from data_engineering.modeling import volatility_lifted_memory as vd
from data_engineering.modeling import volatility_lifted_memory_long as history
from data_engineering.modeling import latent_regime_tournament as rd

VERSION = 'PSH01-0.1.0'
EMISSIONS = ('ema_30m', 'ema_240m')
BASE = ('now', 'ema_30m', 'ema_240m', 'clock_sin', 'clock_cos')
COST_ROUNDTRIP_BP = 12.0  # user-configurable research scenario, not actual fill quote

@dataclass(frozen=True)
class Spec:
    version: str = VERSION
    assets: tuple[str, ...] = ('BTC', 'ETH', 'SOL', 'XRP')
    min_history_days: int = 365
    hmm_states: tuple[int, ...] = (2, 3, 4)
    anchor_minutes: int = 30
    primary_shock_horizon_min: int = 60
    perp_horizon_min: int = 30
    perp_roundtrip_cost_bp: float = COST_ROUNDTRIP_BP
    logistic_C: float = 0.1
    ridge_alpha: float = 100.0
    bootstrap_days: int = 500
    min_test_days: int = 30
    seed: int = 2617
    status: str = 'HISTORICAL_EXPLORATORY'


def registration(cfg: Spec = Spec()) -> dict:
    return asdict(cfg) | {'emissions': list(EMISSIONS), 'baseline_features': list(BASE),
                          'allowed_predictor_evidence': {'ema_30m': 'VD12', 'ema_240m': 'VD12'},
                          'option_PnL': 'NOT_TESTED_WITHOUT_POINT_IN_TIME_QUOTES',
                          'perp_source': 'Binance USD-M historical proxy; not Deribit execution'}


def load_panel(price_directory: Path, cfg: Spec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every feature uses observations through origin t, except EMA which uses t-1 and earlier."""
    end = pd.Timestamp('2026-10-17', tz='UTC')  # frozen historical discovery ceiling
    raw, _ = history.load_returns(price_directory, cfg.assets, end=end,
                                  min_days=cfg.min_history_days)
    panel = vd.dense_panel(raw, min_days=cfg.min_history_days)
    panel, clock = vd.training_clock(panel)
    ema = vd.ema_prior(panel, np.asarray([30., 240.]))
    panel['now'] = panel.y
    panel['ema_30m'] = ema[:, 0]
    panel['ema_240m'] = ema[:, 1]
    mem = vd.memory_diagnostics(panel)
    panel['memory_ready'] = mem.memory_ready.to_numpy(bool)
    theta = 2*np.pi*(panel.state_time.dt.hour*60 + panel.state_time.dt.minute)/1440.
    panel['clock_sin'] = np.sin(theta)
    panel['clock_cos'] = np.cos(theta)
    labels, thresholds = vd.q99_labels(panel, (cfg.primary_shock_horizon_min,))
    panel['shock_60m'] = labels[f'y_{cfg.primary_shock_horizon_min}m'].to_numpy(float)
    panel['eligible_now'] = labels.eligible_now.to_numpy(bool)
    anchor = (panel.state_time.dt.hour*60 + panel.state_time.dt.minute) % cfg.anchor_minutes == 0
    panel = panel.loc[anchor].reset_index(drop=True)
    panel['state_ready'] = panel.memory_ready & np.isfinite(panel[list(BASE)].to_numpy(float)).all(axis=1)
    return panel, thresholds


def perp_30m_returns(path: Path) -> pd.DataFrame:
    """End-timed 30 one-minute bar closes; drop incomplete buckets, never bridge gaps."""
    minutes = pd.read_parquet(path, columns=['open_time', 'close'])
    minutes['open_time'] = pd.to_datetime(minutes.open_time, utc=True)
    if minutes.open_time.isna().any() or minutes.open_time.duplicated().any():
        raise ValueError('Invalid/duplicate futures minute')
    if minutes.open_time.dt.second.ne(0).any():
        raise ValueError('Futures timestamps must be exact minute opens')
    minutes = minutes.sort_values('open_time')
    minutes['close'] = pd.to_numeric(minutes.close, errors='coerce')
    minutes['state_time'] = minutes.open_time.dt.floor('30min') + pd.Timedelta(minutes=30)
    groups = minutes.groupby('state_time', sort=True).agg(
        n=('open_time', 'size'), first=('open_time', 'min'), last=('open_time', 'max'),
        close=('close', 'last'), all_positive=('close', lambda x: bool(np.isfinite(x).all() and (x > 0).all())))
    good = ((groups.n == 30) & (groups['first'] == groups.index-pd.Timedelta(minutes=30)) &
            (groups['last'] == groups.index-pd.Timedelta(minutes=1)) & groups.all_positive)
    groups.loc[~good, 'close'] = np.nan
    later = groups.close.shift(-1)
    exact_next = groups.index.to_series().shift(-1).eq(groups.index.to_series()+pd.Timedelta(minutes=30))
    groups['perp_fwd_30m'] = np.log(later/groups.close).where(exact_next & good)
    return groups[['perp_fwd_30m']].reset_index()


def _partition(frame: pd.DataFrame):
    needed = frame.state_ready & np.isfinite(frame[list(EMISSIONS)].to_numpy(float)).all(axis=1)
    return {s: frame.loc[needed & (frame.split == s)].copy().reset_index(drop=True)
            for s in ('train', 'valid', 'test')}


def _predictor_matrix(part: pd.DataFrame, scale: StandardScaler):
    return scale.transform(part[list(BASE)].to_numpy(float))


def _hmm_fit(parts, k, cfg):
    scale = StandardScaler().fit(parts['train'][list(EMISSIONS)].to_numpy(float))
    x = {s: np.clip(scale.transform(p[list(EMISSIONS)].to_numpy(float)), -6, 6)
         for s, p in parts.items()}
    model_cfg = replace(rd.Spec(), k_states=k, anchor_minutes=cfg.anchor_minutes,
                        em_iterations=8, seed=cfg.seed)
    m = rd.fit_hmm(x['train'], parts['train'].state_time, model_cfg)
    posterior = {}
    for s, p in parts.items():
        l = rd.emission_loglik(x[s], m['loc'], m['var'], model_cfg.emission_df)
        posterior[s] = {
            'hmm': rd.filter_hmm(l, m['A'], m['pi'], p.state_time, cfg.anchor_minutes),
            'static': rd.static_states(l, m['pi'])}
    n, d = x['train'].shape
    npar = (2*k*d) + k*(k-1) + (k-1)
    bic = -2*m['loglike'][-1] + npar*np.log(n)
    occupancy = posterior['train']['hmm'].mean(axis=0)
    return m, posterior, float(bic), occupancy


def _binary_models(parts, posterior, cfg):
    """Train-only conditional outcome calibration; exactly matched TEST origins."""
    scaler = StandardScaler().fit(parts['train'][list(BASE)].to_numpy(float))
    tr, te = parts['train'], parts['test']
    fit = tr.eligible_now & tr.shock_60m.notna()
    score = te.eligible_now & te.shock_60m.notna()
    if fit.sum() < 500 or score.sum() < 100:
        raise ValueError('Not enough q99 outcome origins')
    y = tr.loc[fit, 'shock_60m'].to_numpy(int)
    if len(set(y)) < 2: raise ValueError('No positive shock events in train')
    xtr = _predictor_matrix(tr, scaler)[fit.to_numpy()]
    xte = _predictor_matrix(te, scaler)[score.to_numpy()]
    out = {}
    for name in ('continuous', 'static', 'hmm'):
        if name == 'continuous':
            a, b = xtr, xte
        else:
            a = np.column_stack([xtr, posterior['train'][name][fit.to_numpy(), :-1]])
            b = np.column_stack([xte, posterior['test'][name][score.to_numpy(), :-1]])
        mdl = LogisticRegression(C=cfg.logistic_C, max_iter=250, solver='lbfgs')
        mdl.fit(a, y)
        out[name] = mdl.predict_proba(b)[:, 1]
    tree = HistGradientBoostingClassifier(max_leaf_nodes=5, max_iter=75,
                                           l2_regularization=5., random_state=cfg.seed)
    tree.fit(xtr, y)
    out['nonlinear'] = tree.predict_proba(xte)[:, 1]
    actual = te.loc[score, 'shock_60m'].to_numpy(int)
    days = te.loc[score, 'state_time'].dt.floor('D').to_numpy()
    return actual, days, out


def bootstrap_delta(actual, p1, p0, days, cfg):
    e = 1e-7
    def losses(p):
        p = np.clip(p, e, 1-e)
        return -(actual*np.log(p) + (1-actual)*np.log1p(-p))
    diff = losses(p1)-losses(p0)
    u, inv = np.unique(days, return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    counts = np.bincount(inv)
    rng = np.random.default_rng(cfg.seed)
    ix = rng.integers(0, len(u), size=(cfg.bootstrap_days, len(u)))
    bs = sums[ix].sum(axis=1)/counts[ix].sum(axis=1)
    return {'delta_logloss': float(np.mean(diff)), 'ci025': float(np.quantile(bs,.025)),
            'ci975': float(np.quantile(bs,.975)), 'test_days': len(u)}


def _perp_models(parts, posterior, cfg):
    """Perp-price directional simulation; no option outcomes and no spot proxy execution."""
    scaler = StandardScaler().fit(parts['train'][list(BASE)].to_numpy(float))
    tr, te = parts['train'], parts['test']
    oktr = np.isfinite(tr.perp_fwd_30m.to_numpy(float))
    okte = np.isfinite(te.perp_fwd_30m.to_numpy(float))
    if oktr.sum() < 500 or okte.sum() < 100:
        return {'status': 'INSUFFICIENT_PERP_ROWS', 'train': int(oktr.sum()), 'test': int(okte.sum())}
    xtr = _predictor_matrix(tr, scaler)[oktr]; xte = _predictor_matrix(te, scaler)[okte]
    ytr = 1e4*tr.perp_fwd_30m.to_numpy(float)[oktr]
    yte = 1e4*te.perp_fwd_30m.to_numpy(float)[okte]
    out = {'status': 'HISTORICAL_PERP_COST_SCENARIO', 'n_test_origins': int(okte.sum())}
    for name in ('continuous','static','hmm'):
        a, b = (xtr,xte) if name=='continuous' else (
            np.column_stack([xtr,posterior['train'][name][oktr,:-1]]),
            np.column_stack([xte,posterior['test'][name][okte,:-1]]))
        reg = Ridge(alpha=cfg.ridge_alpha).fit(a,ytr)
        forecast = reg.predict(b)
        side = np.where(np.abs(forecast) > cfg.perp_roundtrip_cost_bp,
                        np.sign(forecast), 0.)
        pnl = side*yte - (side != 0)*cfg.perp_roundtrip_cost_bp
        trades = side != 0
        out[name] = {'n_trades': int(trades.sum()), 'mean_bp_per_trade_net':
                     float(pnl[trades].mean()) if trades.any() else None,
                     'total_bp_per_notional_slot_net': float(np.sum(pnl)),
                     'mean_bp_per_origin_net': float(pnl.mean()),
                     'gross_bp_per_trade': float((side*yte)[trades].mean()) if trades.any() else None,
                     'hit_rate': float((side[trades]*yte[trades]>0).mean()) if trades.any() else None}
    return out


def _durations(p, posterior, m, cfg):
    z = posterior['test']['hmm'].argmax(axis=1)
    out = []
    for k in range(m['K']):
        exposures, exits = np.zeros(24), np.zeros(24)
        for seg in rd.segments(p.state_time, cfg.anchor_minutes):
            seq = z[seg]
            starts = np.r_[0,np.flatnonzero(seq[1:]!=seq[:-1])+1]
            ends = np.r_[starts[1:],len(seq)]
            for start,end in zip(starts,ends):
                if seq[start] != k: continue
                duration = end-start
                for age in range(1, duration+1): exposures[min(age,24)-1] += 1
                if end < len(seq): exits[min(duration,24)-1] += 1
        for age in range(1,25):
            if exposures[age-1] > 0:
                out.append({'state':k,'age_blocks':age,'n_at_risk':int(exposures[age-1]),
                            'n_exits':int(exits[age-1]),
                            'observed_exit_probability':exits[age-1]/exposures[age-1],
                            'hmm_geometric_exit_probability':1-m['A'][k,k]})
    return out


def run(price_directory: Path, perp_directory: Path, output_directory: Path,
        cfg: Spec = Spec()):
    outdir = Path(output_directory); outdir.mkdir(parents=True,exist_ok=True)
    panel, thresholds = load_panel(Path(price_directory),cfg)
    scores=[]; economics=[]; profiles=[]; transitions=[]; durations=[]; covers=[]; decisions=[]
    for asset in cfg.assets:
        a = panel.loc[panel.underlying==asset].copy().reset_index(drop=True)
        f = Path(perp_directory)/f'{asset}USDT_um_1m.parquet'
        if f.exists():
            perps=perp_30m_returns(f)
            a=a.merge(perps,on='state_time',how='left',validate='one_to_one')
        else:
            a['perp_fwd_30m']=np.nan
        # A target ending outside the split is ineligible for model response fitting.
        next_t=a.state_time+pd.Timedelta(minutes=30)
        same_day_split=(next_t < pd.Timestamp('2026-10-17',tz='UTC'))
        # A 30m target may cross midnight; reject chronological split boundaries explicitly.
        split_ends = a.groupby('split').state_time.max().sort_values()
        for sp, last in split_ends.items():
            a.loc[(a.split==sp)&(next_t>last),'perp_fwd_30m']=np.nan
        a.loc[~same_day_split,'perp_fwd_30m']=np.nan
        parts=_partition(a)
        if min(len(z) for z in parts.values()) < 300:
            covers.append({'asset':asset,'status':'INSUFFICIENT_STATES','n_train':len(parts['train'])})
            continue
        best=None
        for k in cfg.hmm_states:
            m,posterior,bic,occ=_hmm_fit(parts,k,cfg)
            covers.append({'asset':asset,'K':k,'bic':bic,'min_train_occupancy':float(occ.min()),
                           'n_train':len(parts['train']),'n_valid':len(parts['valid']),
                           'n_test':len(parts['test'])})
            if occ.min()>=.01 and (best is None or bic<best[0]):best=(bic,k,m,posterior)
        if best is None:
            decisions.append({'asset':asset,'verdict':'NO_OCCUPANCY_VALID_HMM'})
            continue
        _,k,m,p=best
        actual,days,prob=_binary_models(parts,p,cfg)
        for name, pred in prob.items():
            eps=1e-7; pred=np.clip(pred,eps,1-eps)
            loss=-(actual*np.log(pred)+(1-actual)*np.log1p(-pred))
            scores.append({'asset':asset,'model':name,'K':k,'n_test':len(actual),
                           'n_shock_origins':int(actual.sum()),'logloss':float(loss.mean()),
                           'brier':float(np.mean((actual-pred)**2))})
        stat=bootstrap_delta(actual,prob['hmm'],prob['static'],days,cfg)
        cont=bootstrap_delta(actual,prob['hmm'],prob['continuous'],days,cfg)
        nonlin=bootstrap_delta(actual,prob['hmm'],prob['nonlinear'],days,cfg)
        pnl=_perp_models(parts,p,cfg)
        economics.append({'asset':asset,'K':k,'perp_evaluation':pnl,
                          'options_evaluation':'NOT_TESTED_MISSING_MATCHED_DERIBIT_QUOTES'})
        verdict=('EXPLORATORY_HMM_INCREMENT' if max(stat['ci975'],cont['ci975'],nonlin['ci975'])<0
                 and min(stat['test_days'],cont['test_days'],nonlin['test_days'])>=cfg.min_test_days
                 else 'NO_STABLE_HMM_INCREMENT')
        decisions.append({'asset':asset,'K':k,'verdict':verdict,'vs_static':stat,
                          'vs_continuous':cont,'vs_nonlinear':nonlin,
                          'prospective_confirmation':False,'execution_approved':False})
        for s in range(k):
            profiles.append({'asset':asset,'K':k,'state':s,'occupancy_test':float(p['test']['hmm'][:,s].mean()),
                             'ema30_location_z':float(m['loc'][s,0]),
                             'ema240_location_z':float(m['loc'][s,1]),
                             'self_transition':float(m['A'][s,s])})
            for j in range(k): transitions.append({'asset':asset,'K':k,'from_state':s,'to_state':j,
                                                   'probability':float(m['A'][s,j])})
        for row in _durations(parts['test'],p,m,cfg):
            durations.append({'asset':asset,'K':k,**row})
    pd.DataFrame(scores).to_csv(outdir/'01_forecast_scores.csv',index=False)
    pd.DataFrame(profiles).to_csv(outdir/'01_state_profiles.csv',index=False)
    pd.DataFrame(transitions).to_csv(outdir/'01_transition_matrix.csv',index=False)
    pd.DataFrame(durations).to_csv(outdir/'01_duration_hazards.csv',index=False)
    pd.DataFrame(covers).to_csv(outdir/'01_coverage.csv',index=False)
    with (outdir/'01_perp_economics.json').open('w') as fd: json.dump(economics,fd,indent=2)
    with (outdir/'01_decision.json').open('w') as fd: json.dump(decisions,fd,indent=2)
    with (outdir/'01_manifest.json').open('w') as fd:
        json.dump({'spec':registration(cfg),'frozen_history_end_exclusive':'2026-10-17T00:00:00Z',
                   'assets':cfg.assets,'actual_executed':True,'synthetic':False,
                   'caveat':'Entire historical period repeatedly researched; exploratory only'},fd,indent=2)
    return decisions