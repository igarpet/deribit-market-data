"""LA09: observable level -> first touch -> second touch -> post-signal lifecycle.

Frozen Binance A02b signals and A03b fixed 1:1 outcomes. A03c maker touch is
secondary eligibility-only metadata, never treated as a confirmed fill.
Observation times are *completed* Binance USD-M 1-minute candles, not tick touches.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = 'LA-LIFECYCLE-09-0.1.0'
MIN_MS = 60_000
OUTCOMES = ('TARGET','STOP','TIMEOUT')
MILESTONES = ('LEVEL_SOURCE','LEVEL_AVAILABLE','FIRST_TOUCH','SECOND_TOUCH')


@dataclass(frozen=True)
class Config:
    pivot_right_5m: int = 2
    prior_day_delay_min: int = 1440
    max_post_min: int = 240
    alignment_before_min: int = 15
    alignment_after_min: int = 30
    history_min: int = 60


def to_ms(s:pd.Series)->pd.Series:
    t=pd.to_datetime(s,utc=True,errors='raise')
    if t.isna().any():raise ValueError('Null UTC timestamp')
    return t.dt.as_unit('ms').astype('int64')


def load_cohort(root:Path,cfg:Config=Config())->pd.DataFrame:
    base=root/'notebooks'/'Liquidity_Absorption'
    f=json.loads((base/'02'/'02_feature_contract.json').read_text(encoding='utf-8'))
    if f.get('version')!='LA-A02-0.1.0' or f['config']['pivot_right_5m']!=cfg.pivot_right_5m:
        raise ValueError('Frozen A02 pivot availability contract changed')
    a=pd.read_csv(base/'02b'/'02b_touch_features.csv',low_memory=False)
    b=pd.read_csv(base/'03b'/'03b_1to1_events.csv',low_memory=False)
    c=pd.read_csv(base/'03c'/'03c_events.csv',low_memory=False)
    frozen={'a02_candidate_row','symbol','side','signal_time_utc','level_type','level_source_time','first_touch_close','second_touch_close','level_price','zone_halfwidth'}
    if not frozen.issubset(a):raise ValueError('A02b missing '+str(sorted(frozen-set(a))))
    if len(a)!=452 or a.a02_candidate_row.duplicated().any() or set(a.a02_candidate_row)!=set(range(452)):
        raise ValueError('Frozen 452 A02b IDs mismatch')
    if len(b)!=452 or b.a02_candidate_row.duplicated().any() or not b['mode'].eq('FIXED').all():
        raise ValueError('Expected frozen one-per-ID A03b fixed mode')
    keep=b[['a02_candidate_row','symbol','side','signal_time_utc','level_price','reason','target','initial_stop','entry_time_ms','exit_time_ms']].copy()
    z=a.merge(keep,on='a02_candidate_row',validate='one_to_one',suffixes=('','_a03b'))
    for col in ('symbol','side'):
        if not z[col].eq(z[col+'_a03b']).all():raise ValueError('A02b/A03b '+col+' differs')
    if not np.allclose(z.level_price,z.level_price_a03b,rtol=0,atol=1e-7):raise ValueError('A02b/A03b level mismatch')
    if not np.array_equal(to_ms(z.signal_time_utc),to_ms(z.signal_time_utc_a03b)):
        raise ValueError('A02b/A03b signal times differ')
    if not z.reason.isin(OUTCOMES+('NO_STRUCTURAL_TARGET','CENSORED_TAPE_END')).all():
        raise ValueError('Unexpected A03b reason')
    maker=c[c['mode'].eq('MAKER_TOUCH_3RD')].copy()
    if len(maker)!=452 or maker.a02_candidate_row.duplicated().any():
        raise ValueError('Expected one A03c maker-touch eligibility row per candidate')
    maker=maker[['a02_candidate_row','source_side','signal_time_utc','reason','observed_zone_touch','entry_time_ms']].rename(columns={
        'reason':'a03c_maker_touch_reason','observed_zone_touch':'a03c_observed_zone_touch',
        'entry_time_ms':'a03c_maker_hypothetical_entry_ms'})
    z=z.merge(maker,on='a02_candidate_row',validate='one_to_one',suffixes=('','_a03c'))
    if not z.side.eq(z.source_side).all() or not np.array_equal(to_ms(z.signal_time_utc),to_ms(z.signal_time_utc_a03c)):
        raise ValueError('A02b/A03c maker source identity drift')
    for field in ('level_source_time','first_touch_close','second_touch_close','signal_time_utc'):
        z[field+'_ms']=to_ms(z[field])
    z['available_ms']=np.where(z.level_type.eq('pivot_5m'),
         z.level_source_time_ms+cfg.pivot_right_5m*5*MIN_MS,
         z.level_source_time_ms+cfg.prior_day_delay_min*MIN_MS).astype('int64')
    if not z.level_type.isin(('pivot_5m','prior_day')).all():raise ValueError('Unknown level availability family')
    if not ((z.available_ms<=z.first_touch_close_ms)&(z.first_touch_close_ms<z.second_touch_close_ms)&
            (z.second_touch_close_ms==z.signal_time_utc_ms)).all():
        raise ValueError('Temporal causality violation: availability < first < signal')
    if z['level_price'].isna().any() or (z.zone_halfwidth<=0).any():raise ValueError('Invalid level geometry')
    # If first touch closes exactly when a pivot is first confirmed, its intraminute
    # prints preceded confirmation; it cannot be advertised as an actionable touch.
    z['first_touch_confirms_at_availability']=z.first_touch_close_ms.eq(z.available_ms)
    z['first_touch_strictly_after_availability']=z.first_touch_close_ms.gt(z.available_ms)
    z['minutes_available_to_first']=(z.first_touch_close_ms-z.available_ms)/MIN_MS
    z['minutes_first_to_second']=(z.second_touch_close_ms-z.first_touch_close_ms)/MIN_MS
    z['minutes_source_to_available']=(z.available_ms-z.level_source_time_ms)/MIN_MS
    return z.sort_values('a02_candidate_row').reset_index(drop=True)


def load_minutes(file:Path)->pd.DataFrame:
    from src.data_engineering.modeling.liquidity_absorption_stage_a import parse_kline_open_time
    cols=['open_time','open','high','low','close','quote_volume','taker_buy_quote_volume']
    bars=pd.read_parquet(file,columns=cols)
    t=parse_kline_open_time(bars['open_time'])
    if t.isna().any() or t.duplicated().any() or not t.is_monotonic_increasing:
        raise ValueError('Invalid minute timestamps: '+str(file))
    z=bars.drop(columns=['open_time']).copy()
    z['close_ms']=t.dt.as_unit('ms').astype('int64')+MIN_MS
    for c in cols[1:]:z[c]=pd.to_numeric(z[c],errors='raise')
    q=z['quote_volume'];tb=z['taker_buy_quote_volume']
    if (not np.isfinite(z[cols[1:]].to_numpy(dtype=float)).all() or
        (z[['open','high','low','close']]<=0).any().any() or
        (z.high<z[['open','close']].max(axis=1)).any() or
        (z.low>z[['open','close']].min(axis=1)).any() or
        (z.low>z.high).any() or (q<0).any() or (tb<0).any() or
        (tb>q*(1+1e-7)+1e-8).any()):raise ValueError('Invalid OHLC or quote fields')
    if z.close_ms.duplicated().any():raise ValueError('Duplicate close_ms')
    z=z.set_index('close_ms',drop=False).sort_index()
    diff=z.close_ms.diff();seg=diff.ne(MIN_MS).cumsum()
    z['logret_bp']=(np.log(z.close/z.close.shift())*1e4).where(diff.eq(MIN_MS))
    z['quote_prior60']=z.quote_volume.groupby(seg).transform(
        lambda s:s.shift(1).rolling(60,min_periods=60).mean())
    z['quote_vs_prior60']=z.quote_volume/z.quote_prior60.replace(0,np.nan)
    return z


def slice_minutes(bars:pd.DataFrame,lo:int,hi:int)->pd.DataFrame:
    """Required minute close stamps (lo,hi], no forward fill; support zero-length interval."""
    if hi<=lo:return bars.iloc[0:0].copy()
    stamps=np.arange(lo+MIN_MS,hi+1,MIN_MS,dtype='int64')
    return bars.reindex(stamps)


def phase_segment(e:dict,bars:pd.DataFrame,name:str,lo:int,hi:int,known:bool)->dict:
    z=slice_minutes(bars,lo,hi)
    d=1 if e['side']=='long' else -1
    required=len(z);ok=required>0 and z['close'].notna().all()
    result={'a02_candidate_row':int(e['a02_candidate_row']),'symbol':e['symbol'],
        'side':e['side'],'eventual_a03b_outcome':e['reason'],'level_type':e['level_type'],
        'stage':name,'stage_from_ms':int(lo),'stage_to_ms':int(hi),
        'expected_minute_bars':required,'observed_minute_bars':int(z['close'].notna().sum()),
        'complete':bool(ok),'level_known_during_stage':bool(known),
        'a03c_maker_touch_reason':e['a03c_maker_touch_reason']}
    vals=['signed_price_change_bp','aligned_aggressive_quote_fraction','mean_quote_volume',
          'median_quote_vs_prior60','mean_abs_1m_return_bp','zone_intersect_fraction',
          'favorable_close_fraction','favorable_extreme_bp','adverse_extreme_bp']
    for c in vals:result[c]=np.nan
    if not ok:return result
    entry=float(z.open.iloc[0]);last=float(z.close.iloc[-1]);level=float(e['level_price']);width=float(e['zone_halfwidth'])
    vol=float(z.quote_volume.sum());buy=float(z.taker_buy_quote_volume.sum())
    result['signed_price_change_bp']=d*(last-entry)/entry*1e4
    result['aligned_aggressive_quote_fraction']=d*(2*buy-vol)/vol if vol>0 else np.nan
    result['mean_quote_volume']=float(z.quote_volume.mean())
    result['median_quote_vs_prior60']=float(z.quote_vs_prior60.median())
    result['mean_abs_1m_return_bp']=float(z.logret_bp.abs().mean())
    if known:
        result['zone_intersect_fraction']=float(((z.high>=level-width)&(z.low<=level+width)).mean())
        result['favorable_close_fraction']=float((d*(z.close-level)>0).mean())
        result['favorable_extreme_bp']=d*((z.high.max() if d==1 else z.low.min())-entry)/entry*1e4
        result['adverse_extreme_bp']=d*((z.low.min() if d==1 else z.high.max())-entry)/entry*1e4
    return result


def event_stages(e:dict,bars:pd.DataFrame,cfg:Config=Config())->list[dict]:
    a=int(e['available_ms']);t=int(e['first_touch_close_ms']);s=int(e['second_touch_close_ms'])
    source=int(e['level_source_time_ms'])
    return [
        phase_segment(e,bars,'PRE_SOURCE_60',source-60*MIN_MS,source,False),
        phase_segment(e,bars,'SOURCE_TO_AVAILABLE',source,a,False),
        phase_segment(e,bars,'AVAILABLE_TO_FIRST',a,t,True),
        phase_segment(e,bars,'FIRST_TO_SECOND',t,s,True),
        phase_segment(e,bars,'AFTER_SIGNAL_1_5',s,s+5*MIN_MS,True),
        phase_segment(e,bars,'AFTER_SIGNAL_6_15',s+5*MIN_MS,s+15*MIN_MS,True),
        phase_segment(e,bars,'AFTER_SIGNAL_16_60',s+15*MIN_MS,s+60*MIN_MS,True),
        phase_segment(e,bars,'AFTER_SIGNAL_61_240',s+60*MIN_MS,s+cfg.max_post_min*MIN_MS,True),
    ]


def event_alignment(e:dict,bars:pd.DataFrame,cfg:Config=Config())->pd.DataFrame:
    """Vectorized complete-minute observations around each of four stage landmarks.

    Level/zone features are undefined before the actual availability timestamp.
    Retrospective outcome labels do not enter the price/flow feature definitions.
    """
    anchors={'LEVEL_SOURCE':int(e['level_source_time_ms']),
             'LEVEL_AVAILABLE':int(e['available_ms']),
             'FIRST_TOUCH':int(e['first_touch_close_ms']),
             'SECOND_TOUCH':int(e['second_touch_close_ms'])}
    sign=1 if e['side']=='long' else -1
    level=float(e['level_price']);width=float(e['zone_halfwidth'])
    ext=e.get('exit_time_ms',np.nan)
    ext=float(ext) if pd.notna(ext) else np.nan
    parts=[]
    for milestone,ref in anchors.items():
        rel=np.arange(-cfg.alignment_before_min,cfg.alignment_after_min+1,dtype='int64')
        ms=ref+rel*MIN_MS
        window=bars.reindex(ms)
        close=window.close.to_numpy(dtype='float64')
        high=window.high.to_numpy(dtype='float64')
        low=window.low.to_numpy(dtype='float64')
        quote=window.quote_volume.to_numpy(dtype='float64')
        buy=window.taker_buy_quote_volume.to_numpy(dtype='float64')
        valid=np.isfinite(close)
        known=ms>=int(e['available_ms'])
        base=float(bars['close'].get(ref,np.nan))
        anchor_ret=sign*(close-base)/base*1e4 if np.isfinite(base) else np.full(len(ms),np.nan)
        delta=np.divide(sign*(2*buy-quote),quote,out=np.full(len(ms),np.nan),where=quote>0)
        level_dist=np.where(known & valid,sign*(close-level)/close*1e4,np.nan)
        overlap=np.where(known & valid,((high>=level-width)&(low<=level+width)).astype(float),np.nan)
        parts.append(pd.DataFrame({
            'a02_candidate_row':int(e['a02_candidate_row']),'symbol':e['symbol'],
            'side':e['side'],'outcome':e['reason'],'level_type':e['level_type'],
            'milestone':milestone,'relative_min':rel,'minute_close_ms':ms,
            'observed':valid,'level_known':known,
            'signed_price_from_anchor_bp':anchor_ret,
            'signed_close_from_level_bp':level_dist,
            'flow_alignment':delta,'zone_intersect':overlap,
            'post_exit_market_only':np.isfinite(ext)&(ms>=ext)}))
    return pd.concat(parts,ignore_index=True)


def summary_frames(stages:pd.DataFrame,aligned:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    keep=stages[stages.eventual_a03b_outcome.isin(OUTCOMES)&stages.complete]
    metrics=['signed_price_change_bp','aligned_aggressive_quote_fraction','mean_quote_volume',
             'median_quote_vs_prior60','mean_abs_1m_return_bp','zone_intersect_fraction',
             'favorable_close_fraction']
    groups=['symbol','level_type','eventual_a03b_outcome','stage']
    out=keep.groupby(groups,observed=True).agg(n_events=('a02_candidate_row','nunique'),
        median_duration_min=('expected_minute_bars','median'),
        **{'median_'+k:(k,'median') for k in metrics}).reset_index()
    allgroups=keep.assign(level_type='ALL_LEVELS').groupby(groups,observed=True).agg(
        n_events=('a02_candidate_row','nunique'),median_duration_min=('expected_minute_bars','median'),
        **{'median_'+k:(k,'median') for k in metrics}).reset_index()
    out=pd.concat([out,allgroups],ignore_index=True)
    a=aligned[(aligned.outcome.isin(OUTCOMES))&aligned.observed]
    grp=['symbol','level_type','outcome','milestone','relative_min']
    px=a.groupby(grp,observed=True).agg(n_events=('a02_candidate_row','nunique'),
        n_post_exit=('post_exit_market_only','sum'),
        median_signed_price_bp=('signed_price_from_anchor_bp','median'),
        median_aligned_flow=('flow_alignment','median'),
        median_signed_level_distance_bp=('signed_close_from_level_bp','median'),
        mean_zone_overlap=('zone_intersect','mean')).reset_index()
    total=a.assign(level_type='ALL_LEVELS').groupby(grp,observed=True).agg(
        n_events=('a02_candidate_row','nunique'),n_post_exit=('post_exit_market_only','sum'),
        median_signed_price_bp=('signed_price_from_anchor_bp','median'),
        median_aligned_flow=('flow_alignment','median'),
        median_signed_level_distance_bp=('signed_close_from_level_bp','median'),
        mean_zone_overlap=('zone_intersect','mean')).reset_index()
    return pd.concat([out],ignore_index=True),pd.concat([px,total],ignore_index=True)


def plots(stages:pd.DataFrame,aligned_summary:pd.DataFrame,out:Path)->None:
    import matplotlib.pyplot as plt
    colors={'TARGET':'tab:green','STOP':'tab:red','TIMEOUT':'tab:orange'}
    fig,axs=plt.subplots(2,4,figsize=(20,9),sharex=True)
    for i,sym in enumerate(('BTCUSDT','ETHUSDT')):
        for j,m in enumerate(MILESTONES):
            ax=axs[i,j]
            for outcome in OUTCOMES:
                z=aligned_summary[(aligned_summary.symbol==sym)&(aligned_summary.level_type=='ALL_LEVELS')&
                                  (aligned_summary.milestone==m)&(aligned_summary.outcome==outcome)].sort_values('relative_min')
                if z.empty:continue
                ax.plot(z.relative_min,z.median_signed_price_bp,color=colors[outcome],label=f'{outcome} (n={int(z.n_events.max())})')
            ax.axvline(0,color='black',ls='--',lw=1)
            ax.axhline(0,color='0.6',lw=.7)
            ax.set(title=f'{sym}: {m}',xlabel='Completed minutes from landmark',ylabel='Signed price from landmark close (bp)')
            if i==0 and j==0:ax.legend(fontsize=8)
    fig.suptitle('LA09: observable-level lifecycle; after exit = market context, not P&L')
    fig.tight_layout();fig.savefig(out/'09_milestone_price_paths.png',dpi=145);plt.close(fig)
    phases=['PRE_SOURCE_60','SOURCE_TO_AVAILABLE','AVAILABLE_TO_FIRST','FIRST_TO_SECOND','AFTER_SIGNAL_1_5','AFTER_SIGNAL_6_15','AFTER_SIGNAL_16_60','AFTER_SIGNAL_61_240']
    fig,axs=plt.subplots(1,2,figsize=(15,5),sharey=True)
    for ax,sym in zip(axs,('BTCUSDT','ETHUSDT')):
        for outcome in OUTCOMES:
            z=stages[(stages.symbol==sym)&stages.eventual_a03b_outcome.eq(outcome)&stages.complete]
            vals=z.groupby('stage').signed_price_change_bp.median().reindex(phases)
            ax.plot(np.arange(len(phases)),vals.values,marker='o',color=colors[outcome],label=outcome)
        ax.set_xticks(np.arange(len(phases)),[x.replace('AFTER_SIGNAL_','+') for x in phases],rotation=35,ha='right')
        ax.set(title=sym,ylabel='Median signed phase price change (bp)');ax.legend()
    fig.tight_layout();fig.savefig(out/'09_lifecycle_phase_changes.png',dpi=145);plt.close(fig)


def run(repo:Path,binance_root:Path,out:Path,cfg:Config=Config())->dict:
    cohort=load_cohort(Path(repo),cfg)
    bars={sym:load_minutes(Path(binance_root)/f'{sym}_um_1m.parquet') for sym in ('BTCUSDT','ETHUSDT')}
    s=[];align=[]
    for e in cohort.to_dict('records'):
        tape=bars[e['symbol']]
        s.extend(event_stages(e,tape,cfg))
        align.append(event_alignment(e,tape,cfg))
    stages=pd.DataFrame(s)
    aligned=pd.concat(align,ignore_index=True)
    summ,paths=summary_frames(stages,aligned)
    out.mkdir(parents=True,exist_ok=True)
    cohort[['a02_candidate_row','symbol','side','reason','level_type','level_source_time','first_touch_close',
            'signal_time_utc','minutes_source_to_available','minutes_available_to_first',
            'minutes_first_to_second','first_touch_confirms_at_availability',
            'first_touch_strictly_after_availability','a03c_maker_touch_reason','a03c_observed_zone_touch']].to_csv(out/'09_cohort_timing.csv',index=False)
    stages.to_csv(out/'09_lifecycle_stages.csv',index=False)
    aligned.to_csv(out/'09_landmark_minute_panel.csv.gz',index=False,compression='gzip')
    summ.to_csv(out/'09_lifecycle_summary.csv',index=False)
    paths.to_csv(out/'09_landmark_summary.csv',index=False)
    plots(stages,paths,out)
    count={c:int((cohort.reason==c).sum()) for c in OUTCOMES+('NO_STRUCTURAL_TARGET','CENSORED_TAPE_END')}
    decision={'version':VERSION,'status':'RETROSPECTIVE_FORMATION_ANATOMY_NOT_PREDICTIVE',
        'original_A02_signals':len(cohort),'outcomes_original_A03b':count,
        'pivot_confirm_delay_min':cfg.pivot_right_5m*5,'prior_day_available_at':'next UTC midnight',
        'level_available_before_or_at_first_touch':int((cohort.available_ms<=cohort.first_touch_close_ms).sum()),
        'median_available_to_first_min':float(cohort.minutes_available_to_first.median()),
        'median_first_to_second_min':float(cohort.minutes_first_to_second.median()),
        'first_touch_same_close_as_availability_caution':int(cohort.first_touch_confirms_at_availability.sum()),
        'stage_rows':len(stages),'stage_missing_or_empty':int((~stages.complete).sum()),
        'minute_landmark_rows':len(aligned),'landmark_missing_bars':int((~aligned.observed).sum()),
        'a03c_maker_note':'eligibility outcomes only; not actual passive fills',
        'market_after_exit':'descriptive market path; not held-position pnl',
        'input_prices':'Binance USD-M futures 1m bar data; touch instant within minute unknown',
        'status_scope':'already inspected Jul-Sep 2026; no holdout confirmation',
        'config':asdict(cfg)}
    (out/'09_decision.json').write_text(json.dumps(decision,indent=2)+'\n',encoding='utf-8')
    return decision


def main(argv=None):
    from src.data_engineering.modeling.liquidity_absorption_stage_a import repo_root,binance_root
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path);ap.add_argument('--binance-root',type=Path);ap.add_argument('--out',type=Path)
    a=ap.parse_args(argv)
    repo=a.repo or repo_root();root=a.binance_root or binance_root(repo)
    print(json.dumps(run(repo,root,a.out or repo/'notebooks'/'Liquidity_Absorption'/'09'),indent=2))

if __name__=='__main__':main()
