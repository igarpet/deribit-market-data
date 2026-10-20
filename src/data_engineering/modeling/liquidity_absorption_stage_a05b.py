"""LA-A05b: Deribit trade-timestamp microstructure around A02 first and second touches.

Fixed UTC anchors from A05a; BTC/ETH Deribit PERPETUAL prints are a
CROSS-VENUE observational analogue, not the Binance futures fills/level.
No tick-order-book reconstruction. Absent 1ms prints remain absent.
Read-only UTC daily Parquet scan, materializes only ±60s around each anchor.
"""
from __future__ import annotations
import argparse
import gzip
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = 'LA-A05b-0.1.0'
WINDOW_MS = 60_000
BINS_MS = (1,10,100,1_000)
PHASES = ((-60,-10),(-10,-1),(-1,0),(0,1),(1,10),(10,60))
OUTCOMES = ('TARGET','STOP','TIMEOUT')


@dataclass(frozen=True)
class Config:
    window_ms: int = WINDOW_MS
    bins_ms: tuple[int,...] = BINS_MS
    batch_size: int = 100_000
    max_anchor_stale_ms: int = 30_000
    earliest_day: str = '2026-07-01'
    latest_day: str = '2026-09-30'


def ms(v: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(v):
        x=pd.to_numeric(v,errors='coerce')
        if x.notna().any() and not (1e12 < x.dropna().abs().median() < 1e14):
            raise ValueError('Expected Unix millisecond timestamps')
        return x.astype('Int64')
    t=pd.to_datetime(v,utc=True,errors='coerce')
    return t.dt.as_unit('ms').astype('int64').where(t.notna()).astype('Int64')


def load_anchors(repo:Path) -> pd.DataFrame:
    """Frozen historical A05a metadata only; do not recompute or rescore A02."""
    repo=Path(repo)
    p=repo/'notebooks'/'Liquidity_Absorption'/'05a'/'05a_cohort_metadata.csv'
    z=pd.read_csv(p,low_memory=False)
    required={'a02_candidate_row','symbol','side','signal_time_utc','eventual_outcome',
              'first_touch_close','second_touch_close','level_source_time','target_to_sigma60'}
    if not required.issubset(z.columns) or len(z)!=452 or z.a02_candidate_row.duplicated().any():
        raise ValueError('Frozen A05a 452-ID cohort is absent or changed')
    if not z.symbol.isin(('BTCUSDT','ETHUSDT')).all() or not z.side.isin(('long','short')).all():
        raise ValueError('Unexpected event instrument/direction')
    if not z.eventual_outcome.isin(OUTCOMES+('NO_STRUCTURAL_TARGET','CENSORED_TAPE_END')).all():
        raise ValueError('Unexpected original outcome')
    if not np.array_equal(ms(z.signal_time_utc).to_numpy(),ms(z.second_touch_close).to_numpy()):
        raise ValueError('Second-touch UTC close must equal signal close')
    if (ms(z.first_touch_close)>ms(z.signal_time_utc)).any():
        raise ValueError('First touch is after signal')
    out=[]
    for _,e in z.iterrows():
        for anchor,col in (('FIRST_TOUCH','first_touch_close'),('SIGNAL','signal_time_utc')):
            out.append({'a02_candidate_row':int(e.a02_candidate_row),'anchor':anchor,
                        'symbol':str(e.symbol),'deribit_instrument':str(e.symbol).replace('USDT','-PERPETUAL'),
                        'side':str(e.side),'eventual_outcome':str(e.eventual_outcome),
                        'target_to_sigma60':float(e.target_to_sigma60) if pd.notna(e.target_to_sigma60) else np.nan,
                        'anchor_ms':int(ms(pd.Series([e[col]])).iloc[0]),
                        'level_source_ms':int(ms(pd.Series([e.level_source_time])).iloc[0])})
    return pd.DataFrame(out)


def needed_days(anchors:pd.DataFrame,window_ms=WINDOW_MS)->dict[tuple[str,str],list[dict]]:
    """Return requested anchor windows keyed (currency, UTC day). Boundary-safe."""
    todo=defaultdict(list)
    for e in anchors.to_dict('records'):
        anchor=int(e['anchor_ms'])
        first=pd.Timestamp(anchor-window_ms,unit='ms',tz='UTC').date()
        last=pd.Timestamp(anchor+window_ms,unit='ms',tz='UTC').date()
        d=first
        while d<=last:
            todo[(e['deribit_instrument'].split('-')[0],d.isoformat())].append(e)
            d+=timedelta(days=1)
    return dict(todo)


def filter_trade_batch(b: pd.DataFrame,perp:str, events:list[dict],window_ms=WINDOW_MS) -> dict[tuple[int,str],pd.DataFrame]:
    """Filter one archive batch to exact trade observations near UTC anchors."""
    req={'timestamp','instrument_name','trade_id','trade_seq','price','amount','direction'}
    if not req.issubset(b.columns):raise ValueError('Missing required Deribit trade fields: '+str(sorted(req-set(b))))
    z=b.loc[b.instrument_name.astype(str).eq(perp)].copy()
    if z.empty:return {}
    z['timestamp_ms']=ms(z['timestamp'])
    z['price']=pd.to_numeric(z.price,errors='coerce')
    z['amount']=pd.to_numeric(z.amount,errors='coerce')
    z['trade_seq']=pd.to_numeric(z.trade_seq,errors='coerce')
    if (z.timestamp_ms.isna().any() or (~np.isfinite(z[['price','amount','trade_seq']].to_numpy(float))).any()
        or (z.price<=0).any() or (z.amount<=0).any()
        or not z.direction.astype(str).isin(('buy','sell')).all()):
        raise ValueError('Invalid Deribit prints in event window source')
    out={}
    for e in events:
        anchor=int(e['anchor_ms']);sel=z.timestamp_ms.between(anchor-window_ms,anchor+window_ms)
        if sel.any():out[(int(e['a02_candidate_row']),e['anchor'])]=z.loc[sel, ['timestamp_ms','trade_id','trade_seq','price','amount','direction']].copy()
    return out


def parquet_batches(path: Path,batch_size:int):
    import pyarrow.parquet as pq
    pf=pq.ParquetFile(path)
    cols=['timestamp','instrument_name','trade_id','trade_seq','price','amount','direction']
    if not set(cols).issubset(pf.schema.names):
        raise ValueError(f'Archive file missing necessary columns: {path}')
    for b in pf.iter_batches(batch_size=batch_size,columns=cols):yield b.to_pandas()


def iter_capture_windows(history:Path,anchors:pd.DataFrame,cfg:Config=Config(),batch_source=None,day_status=None):
    """Read ONLY requested UTC days once each. Missing days are explicit.

    `batch_source(path)` injects fake batches for unit tests. Never loads a whole
    currency-day archive into RAM, but buffers short event neighborhoods.
    """
    if cfg.window_ms!=WINDOW_MS or cfg.bins_ms!=BINS_MS:raise ValueError('Frozen A05b window changed')
    todo=needed_days(anchors,cfg.window_ms)
    cache=defaultdict(list)
    pending=defaultdict(int)
    keys=set()
    for (cur,d),ev in todo.items():
        for e in ev:
            key=(int(e['a02_candidate_row']),e['anchor'])
            pending[key]+=1;keys.add(key)
    seen_ready=set()
    if day_status is None:day_status=[]
    for (cur,d),ev in sorted(todo.items(),key=lambda v:(v[0][1],v[0][0])):
        p=Path(history)/'trades'/cur/'future'/d[:4]/f'{d}.parquet'
        source=batch_source(p) if batch_source is not None else (parquet_batches(p,cfg.batch_size) if p.is_file() else None)
        if source is None:
            day_status.append({'currency':cur,'day':d,'status':'MISSING_DAY','rows_near_anchors':0})
        else:
            try:
                scanned=0;near=0
                for batch in source:
                    scanned+=len(batch)
                    res=filter_trade_batch(batch,f'{cur}-PERPETUAL',ev,cfg.window_ms)
                    for key, part in res.items():
                        cache[key].append(part)
                        near+=len(part)
                day_status.append({'currency':cur,'day':d,'status':'SCANNED','rows_scanned':scanned,'rows_near_anchors':near})
            except Exception as exc:
                day_status.append({'currency':cur,'day':d,'status':'ERROR','error':repr(exc)})
                raise
        for e in ev:
            key=(int(e['a02_candidate_row']),e['anchor'])
            pending[key]-=1
            if pending[key]==0:
                parts=cache.pop(key,[])
                ticks=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=['timestamp_ms','trade_id','trade_seq','price','amount','direction'])
                seen_ready.add(key)
                yield key,ticks
    if seen_ready!=keys:raise AssertionError('Event windows lost')


def capture_windows(history:Path,anchors:pd.DataFrame,cfg:Config=Config(),batch_source=None):
    """Convenience collector for synthetic tests/small subsets; not used in full run."""
    logs=[]
    records=list(iter_capture_windows(history,anchors,cfg,batch_source,logs))
    return records,pd.DataFrame(logs)


def summarize_one(e:dict,trades:pd.DataFrame,cfg:Config=Config()) -> tuple[pd.DataFrame,pd.DataFrame,list[dict]]:
    """Sparse 1/10/100/1000-ms executed-print OHLC and event phase stats.

    Within-ms ordering uses trade_seq. No zero-volume bars are invented. A lack
    of trades exactly at t=0 is ordinary; the most recent *pre/t0* print gives
    the reference if not older than max_anchor_stale_ms.
    """
    anchor=int(e['anchor_ms']);base={k:e[k] for k in ['a02_candidate_row','anchor','symbol','side','eventual_outcome','target_to_sigma60']}
    if trades.empty:
        return pd.DataFrame(),pd.DataFrame(),[base|{'status':'NO_TRADES_IN_WINDOW','n_prints':0}]
    z=trades.copy()
    z['timestamp_ms']=pd.to_numeric(z.timestamp_ms,errors='raise').astype('int64')
    z['trade_seq']=pd.to_numeric(z.trade_seq,errors='raise')
    z=z.sort_values(['timestamp_ms','trade_seq','trade_id'],kind='stable').drop_duplicates(['trade_id'],keep='first')
    if (z.timestamp_ms.lt(anchor-cfg.window_ms)|z.timestamp_ms.gt(anchor+cfg.window_ms)).any():
        raise ValueError('Event capture contains out-of-window trades')
    pre=z[z.timestamp_ms<=anchor]
    reference=float(pre.price.iloc[-1]) if not pre.empty else np.nan
    staleness=anchor-int(pre.timestamp_ms.iloc[-1]) if not pre.empty else np.nan
    reference_status='PRESENT' if np.isfinite(reference) and staleness<=cfg.max_anchor_stale_ms else 'NO_RECENT_PRE_ANCHOR_PRINT'
    sign=1 if e['side']=='long' else -1
    z['relative_ms']=z.timestamp_ms-anchor
    z['signed_amount']=np.where(z.direction.eq('buy'),z.amount,-z.amount)
    z['buy_trades']=z.direction.eq('buy').astype(int)
    z['sell_trades']=z.direction.eq('sell').astype(int)
    ticks=z[['timestamp_ms','relative_ms','trade_seq','price','amount','signed_amount','buy_trades','sell_trades']]
    # Keep separately labelled actual prints with n>=1, no continuous 1ms ticks.
    bucket_parts=[]
    for b in cfg.bins_ms:
        x=ticks.assign(bucket_relative_ms=np.floor_divide(ticks.relative_ms,b)*b)
        t=x.groupby('bucket_relative_ms',sort=True).agg(trade_count=('price','size'),
             buy_trades=('buy_trades','sum'),sell_trades=('sell_trades','sum'),
             open_price=('price','first'),close_price=('price','last'),
             high_price=('price','max'),low_price=('price','min'),
             total_amount_reported=('amount','sum'),signed_amount_reported=('signed_amount','sum')).reset_index()
        t['bin_ms']=b
        t['price_signed_bp_vs_anchor']=sign*(t.close_price-reference)/reference*10000 if reference_status=='PRESENT' else np.nan
        t['signed_count_imbalance']=(t.buy_trades-t.sell_trades)/t.trade_count
        t['signed_amount_imbalance']=(t.signed_amount_reported/t.total_amount_reported)
        for k,v in base.items():t[k]=v
        bucket_parts.append(t)
    stats=[]
    for a,c in PHASES:
        lo,hi=a*1000,c*1000
        phase=z[(z.relative_ms>=lo)&(z.relative_ms<hi)]
        n=len(phase)
        signed_count=(int(phase.buy_trades.sum())-int(phase.sell_trades.sum()))/n if n else np.nan
        signed_amount=float(phase.signed_amount.sum()/phase.amount.sum()) if n and phase.amount.sum()>0 else np.nan
        change=sign*(float(phase.price.iloc[-1])-float(phase.price.iloc[0]))/float(phase.price.iloc[0])*10000 if n>=2 else np.nan
        stats.append(base|{'phase_s_start':a,'phase_s_end':c,'n_prints':n,
            'trades_per_sec':n/(c-a),'aligned_count_imbalance':sign*signed_count if n else np.nan,
            'aligned_amount_imbalance':sign*signed_amount if n else np.nan,
            'signed_price_change_bp':change,'anchor_price_reference':reference if reference_status=='PRESENT' else np.nan,
            'anchor_ref_staleness_ms':staleness,'status':reference_status})
    overall=base|{'status':reference_status,'n_prints':len(z),'distinct_trade_milliseconds':z.timestamp_ms.nunique(),
        'pre_anchor_prints':len(pre),'anchor_price_reference':reference if reference_status=='PRESENT' else np.nan,
        'anchor_ref_staleness_ms':staleness,'max_trades_one_ms':int(z.groupby('timestamp_ms').size().max()),
        'first_trade_relative_ms':int(z.relative_ms.min()),'last_trade_relative_ms':int(z.relative_ms.max())}
    return pd.concat(bucket_parts,ignore_index=True),pd.DataFrame(stats),[overall]


def build_results(repo:Path,history:Path,out:Path,cfg:Config=Config(),batch_source=None) -> dict:
    anchors=load_anchors(repo)
    lookup={(int(r.a02_candidate_row),r.anchor):r._asdict() for r in anchors.itertuples(index=False)}
    out.mkdir(parents=True,exist_ok=True)
    archive=out/'05b_sparse_trade_buckets.csv.gz'
    day_status=[];phases=[];summary=[];count=0
    # Calendar completeness is required before trusting a pair of +/-60s
    # neighborhoods; a missing adjacent midnight file may hide a price move.
    calendar_days=needed_days(anchors,cfg.window_ms)
    missing_anchor_keys=set()
    if batch_source is None:
        for (cur,d),ev in calendar_days.items():
            path=Path(history)/'trades'/cur/'future'/d[:4]/f'{d}.parquet'
            if not path.is_file():
                missing_anchor_keys.update((int(e['a02_candidate_row']),e['anchor']) for e in ev)
    example_keys={(sym,reason,'SIGNAL'):min((int(e['a02_candidate_row']) for e in anchors.to_dict('records')
                 if e['symbol']==sym and e['eventual_outcome']==reason and e['anchor']=='SIGNAL'), default=None)
                 for sym in ('BTCUSDT','ETHUSDT') for reason in OUTCOMES}
    example_frames=[]
    # Process completed short windows one at a time; never retain multi-year
    # raw trade archives or the complete 1ms bucket output in RAM.
    with gzip.open(archive,'wt',encoding='utf-8',newline='') as stream:
        for key,chunk in iter_capture_windows(history,anchors,cfg,batch_source,day_status):
            b,p,r=summarize_one(lookup[key],chunk,cfg)
            complete=key not in missing_anchor_keys
            if len(b):
                b['complete_calendar_days']=complete
                b.to_csv(stream,index=False,header=(count==0))
                count+=len(b)
                e=lookup[key]
                if e['anchor']=='SIGNAL' and example_keys.get((e['symbol'],e['eventual_outcome'],'SIGNAL'))==key[0]:
                    z=b[(b.bin_ms==1)&b.bucket_relative_ms.between(-2000,2000)]
                    if len(z):example_frames.append(z.copy())
            if len(p):
                p['complete_calendar_days']=complete
                phases.append(p)
            for row in r:
                if not complete:row['status']='INCOMPLETE_CALENDAR_DAYS'
            summary+=r
        if count==0:
            # Empty study is distinguishable from a failed unreadable gzip.
            stream.write('a02_candidate_row,anchor,symbol,status\n')
    phase=pd.concat(phases,ignore_index=True) if phases else pd.DataFrame()
    qc=pd.DataFrame(summary)
    days=pd.DataFrame(day_status)
    group=phase[(phase.eventual_outcome.isin(OUTCOMES)) & (phase.complete_calendar_days)] if len(phase) else phase
    cols=['signed_price_change_bp','aligned_count_imbalance','aligned_amount_imbalance','trades_per_sec']
    grouped=group.groupby(['anchor','symbol','eventual_outcome','phase_s_start','phase_s_end']).agg(
        n_events=('a02_candidate_row','nunique'),**{'median_'+k:(k,'median') for k in cols}).reset_index() if len(group) else pd.DataFrame()
    phase.to_csv(out/'05b_event_phases.csv',index=False)
    grouped.to_csv(out/'05b_phase_summary.csv',index=False)
    qc.to_csv(out/'05b_anchor_coverage.csv',index=False)
    days.to_csv(out/'05b_day_scan_status.csv',index=False)
    report={'version':VERSION,'status':'HISTORICAL_EXPLORATORY_CROSS_VENUE_TRADE_ONLY',
       'signal_events':int(anchors.a02_candidate_row.nunique()),'anchors':len(anchors),'anchor_rows':len(qc),
       'captured_trade_buckets':count,'days_scanned':int((days.status=='SCANNED').sum()),
       'missing_days':int((days.status=='MISSING_DAY').sum()),
       'anchor_reference_present':int((qc.status=='PRESENT').sum()),
       'original_outcome_is_descriptive_only':True,
       'clock':'UTC timestamp milliseconds, first-touch and confirmed signal CANDLE CLOSE; actual touch instant unknown',
       'price':'Deribit PERPETUAL executed-trade price, NOT the Binance UM level or fill price',
       'size':'Deribit API reported amount (compare within same instrument only; do not assume USD notional)',
       'limitations':'No L2 bid/ask, queue, cancelling liquidity, nanosecond ordering, tick continuity or same-venue fill validation',
       'config':asdict(cfg)}
    (out/'05b_decision.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    _plot(grouped,out)
    _plot_examples(pd.concat(example_frames,ignore_index=True) if example_frames else pd.DataFrame(),out)
    return report


def _plot(group:pd.DataFrame,out:Path):
    if group.empty:return
    import matplotlib.pyplot as plt
    for anchor in ('FIRST_TOUCH','SIGNAL'):
        fig,axes=plt.subplots(1,2,figsize=(13,4))
        for ax,sym in zip(axes,('BTCUSDT','ETHUSDT')):
            for label in OUTCOMES:
                g=group[(group.anchor==anchor)&(group.symbol==sym)&(group.eventual_outcome==label)].sort_values('phase_s_start')
                if not g.empty:
                    x=(g.phase_s_start+g.phase_s_end)/2
                    ax.plot(x,g.median_signed_price_change_bp,marker='o',label=f'{label} (n={int(g.n_events.max())})')
            ax.axvline(0,color='black',ls='--',lw=1)
            ax.set(xlabel='Seconds from UTC anchor',ylabel='Median event-phase direction-aligned return (bp)',title=f'{sym} — {anchor}')
            if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
        fig.suptitle('A05b Deribit perpetual trade prints (not Binance execution paths)')
        fig.tight_layout();fig.savefig(out/f'05b_{anchor.lower()}_millisecond_phase_reactions.png',dpi=145);plt.close(fig)


def _plot_examples(examples:pd.DataFrame,out:Path):
    """Sparse executed 1ms prints: there is no invented price between executions."""
    if examples.empty:return
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(16,8),sharex=True)
    for i,sym in enumerate(('BTCUSDT','ETHUSDT')):
        for j,reason in enumerate(OUTCOMES):
            ax=axes[i,j]
            z=examples[(examples.symbol==sym)&(examples.eventual_outcome==reason)]
            if len(z):
                z=z.sort_values('bucket_relative_ms')
                ax.scatter(z.bucket_relative_ms,z.price_signed_bp_vs_anchor,s=np.minimum(55,z.trade_count*7),alpha=.75)
                ax.set_title(f'{sym} {reason} — earliest event ID {int(z.a02_candidate_row.iloc[0])}')
                ax.text(.02,.05,'Dots = ms with executions ONLY; size ∝ trade count',transform=ax.transAxes,fontsize=7)
            else:ax.set_title(f'{sym} {reason} — no visible ms prints')
            ax.axvline(0,c='black',lw=.8,ls='--')
            ax.set(xlabel='ms from Binance signal candle CLOSE',ylabel='Deribit price markout (bp)')
    fig.suptitle('A05b sparse millisecond prints, Deribit perpetual — 6 earliest-ID examples (NOT representative)')
    fig.tight_layout();fig.savefig(out/'05b_signal_sparse_1ms_examples.png',dpi=145);plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=None)
    p.add_argument('--history',type=Path,default=None)
    p.add_argument('--out',type=Path,default=None)
    a=p.parse_args(argv)
    repo=a.repo or Path(__file__).resolve().parents[3]
    if a.history is None:
        import os
        d=os.environ.get('DERIBIT_DATA_ROOT')
        if not d:
            env=repo/'.env'
            if env.is_file():
                for line in env.read_text().splitlines():
                    if line.strip().startswith('DERIBIT_DATA_ROOT='):
                        d=line.split('=',1)[1].strip().strip('"').strip("'")
        if not d:raise ValueError('Set DERIBIT_DATA_ROOT or pass --history')
        history=Path(d)/'history'
    else:
        history=Path(a.history)
        if (history/'history'/'trades').exists():history=history/'history'
    out=a.out or repo/'notebooks'/'Liquidity_Absorption'/'05b'
    print(json.dumps(build_results(repo,history,out),indent=2))


if __name__=='__main__':main()
