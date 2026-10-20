"""Read-only streaming QA for Deribit history/trades/{BTC,ETH}/future/YYYY/YYYY-MM-DD.parquet.

Never downloads or repairs data. `kind=future` files mix perpetual and dated futures;
all sequence/price analyses must be per INSTRUMENT, not per currency.

Unknown completeness is not proof of completeness: original downloader pages by
millisecond in blocks of 1000, and it cannot guarantee catching overflow or
truncation. A local saved Parquet archive alone cannot certify remote coverage.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, timedelta
import json
from pathlib import Path
import os

import numpy as np
import pandas as pd

VERSION = 'DERIBIT-HIST-VALIDATOR-0.1.0'
REQUIRED = ('trade_id', 'trade_seq', 'instrument_name', 'timestamp', 'direction', 'price', 'amount')
BATCH_SIZE = 125_000
PAGE_LIMIT = 1_000


def epoch_ms(v: pd.Series) -> pd.Series:
    """Convert timezone-aware/naive UTC timestamps or numeric epochs to Unix ms.

    Historical downloader writes timezone-aware pandas Timestamp[ms source] to Parquet.
    There must be no implicit assumption about pandas integer datetime unit.
    """
    if pd.api.types.is_numeric_dtype(v):
        x = pd.to_numeric(v, errors='coerce')
        a = x.dropna()
        if len(a) and not (1e12 < a.abs().median() < 1e14):
            raise ValueError('Numeric timestamps are not plausible Unix milliseconds')
        return x.astype('Int64')
    d = pd.to_datetime(v, utc=True, errors='coerce')
    return d.dt.as_unit('ms').astype('int64').where(d.notna()).astype('Int64')


def audit_frames(chunks, currency: str, kind: str, day: date, exact_ids: bool = False) -> dict:
    """Audit iterable of record batches. `exact_ids` keeps IDs for *one daily file* only.

    Sequence span gaps are indicators, not unquestionable data loss: historical
    instrument trade_seq behavior should be spot-checked with the exchange.
    """
    first = int(pd.Timestamp(day, tz='UTC').timestamp() * 1000)
    end = first + 86_400_000
    totals = defaultdict(lambda: {'rows': 0, 'min_seq': None, 'max_seq': None})
    bad = defaultdict(int)
    count = 0
    seen = set() if exact_ids else None
    seen_batch_boundary = set()
    prev_time = None
    prev_id_key = None
    ms_tail = None
    ms_tail_count = 0
    ms_overflow = 0
    worst_ms = 0
    unique_ms = 0
    min_time, max_time = None, None
    for batch in chunks:
        if not set(REQUIRED).issubset(batch.columns):
            bad['missing_required_columns'] += 1
            bad['missing_columns_list'] = ','.join(sorted(set(REQUIRED) - set(batch.columns)))
            break
        if batch.empty:
            continue
        count += len(batch)
        times = epoch_ms(batch['timestamp'])
        t = times.to_numpy(dtype='float64', na_value=np.nan)
        valid_t = np.isfinite(t)
        bad['null_timestamp'] += int((~valid_t).sum())
        bad['out_of_day_timestamp'] += int((valid_t & ((t < first) | (t >= end))).sum())
        if valid_t.any():
            vals = t[valid_t].astype('int64')
            min_time = int(vals.min()) if min_time is None else min(min_time, int(vals.min()))
            max_time = int(vals.max()) if max_time is None else max(max_time, int(vals.max()))
            if prev_time is not None and vals[0] < prev_time:
                bad['out_of_order_chunk_boundary'] += 1
            bad['out_of_order_in_batch'] += int(np.sum(np.diff(vals) < 0))
            prev_time = int(vals[-1])
            # Full-day same-millisecond cluster counts including batch boundaries.
            ms, freq = np.unique(vals, return_counts=True)
            if ms_tail is not None and len(ms):
                if ms[0] == ms_tail:
                    freq[0] += ms_tail_count
                    unique_ms -= 1
                else:
                    # Finalize prior batch's last millisecond, which otherwise
                    # disappears when the next batch begins with a NEW ms.
                    ms_overflow += int(ms_tail_count >= PAGE_LIMIT)
                    worst_ms = max(worst_ms, ms_tail_count)
            if len(ms):
                if ms_tail is not None and ms[0] == ms_tail:
                    # Tail bucket was not finalized previously.
                    pass
                if len(ms) > 1:
                    closed = freq[:-1]
                    ms_overflow += int(np.sum(closed >= PAGE_LIMIT))
                    worst_ms = max(worst_ms, int(closed.max()))
                ms_tail, ms_tail_count = int(ms[-1]), int(freq[-1])
                unique_ms += len(ms)
        ids = batch['trade_id'].astype('string')
        bad['null_trade_id'] += int(ids.isna().sum() + ids.fillna('').eq('').sum() - ids.isna().sum())
        if exact_ids:
            for val in ids.dropna():
                key = str(val)
                if key in seen:
                    bad['duplicate_trade_id_exact'] += 1
                else:
                    seen.add(key)
        else:
            # Adjacent duplicate IDs are common pagination artifacts. Only a
            # lower bound; non-adjacent duplicates need --exact-ids.
            keys = ids.fillna('').to_numpy(str)
            if len(keys):
                bad['duplicate_adjacent_id_lower_bound'] += int(np.sum((keys[1:] == keys[:-1]) & (keys[1:] != '')))
                if prev_id_key and keys[0] == prev_id_key:
                    bad['duplicate_adjacent_id_lower_bound'] += 1
                prev_id_key = keys[-1]
        price = pd.to_numeric(batch['price'], errors='coerce').to_numpy(float)
        amount = pd.to_numeric(batch['amount'], errors='coerce').to_numpy(float)
        bad['bad_price'] += int(np.sum(~np.isfinite(price) | (price <= 0)))
        bad['bad_amount'] += int(np.sum(~np.isfinite(amount) | (amount <= 0)))
        direct = batch['direction'].astype('string').str.lower()
        bad['bad_direction'] += int((~direct.isin(('buy','sell'))).sum())
        inst = batch['instrument_name'].astype('string')
        bad['empty_instrument_name'] += int(inst.isna().sum() + inst.fillna('').eq('').sum() - inst.isna().sum())
        bad['wrong_currency_instrument'] += int((~inst.fillna('').str.startswith(currency + '-')).sum())
        if kind == 'future':
            bad['unexpected_option_in_future_file'] += int(inst.fillna('').str.match(r'.*-[CP]$').sum())
        seq = pd.to_numeric(batch['trade_seq'], errors='coerce')
        bad['missing_or_nonintegral_seq'] += int((seq.isna() | ((seq % 1) != 0)).sum())
        # Sequence span/count per instrument: catches missing middle parts if
        # sequence increments by one and batch contains all expected fills.
        tmp = pd.DataFrame({'instrument': inst, 'seq': seq}).dropna()
        if not tmp.empty:
            x = tmp.groupby('instrument')['seq'].agg(['count', 'min', 'max'])
            for name, r in x.iterrows():
                a = totals[str(name)]
                a['rows'] += int(r['count'])
                a['min_seq'] = int(r['min']) if a['min_seq'] is None else min(a['min_seq'], int(r['min']))
                a['max_seq'] = int(r['max']) if a['max_seq'] is None else max(a['max_seq'], int(r['max']))
    if ms_tail is not None:
        ms_overflow += int(ms_tail_count >= PAGE_LIMIT)
        worst_ms = max(worst_ms, ms_tail_count)
    seq_gap = sum(max(0, x['max_seq'] - x['min_seq'] + 1 - x['rows']) for x in totals.values())
    # Group-level IDs can hide missing seq offset by duplicates; flag only as possible.
    must_fail = ['missing_required_columns','out_of_day_timestamp','bad_price','bad_amount',
                 'bad_direction','empty_instrument_name','wrong_currency_instrument',
                 'null_timestamp','null_trade_id','missing_or_nonintegral_seq',
                 'out_of_order_chunk_boundary','out_of_order_in_batch',
                 'duplicate_trade_id_exact','duplicate_adjacent_id_lower_bound',
                 'unexpected_option_in_future_file']
    is_bad = any(bool(bad.get(k,0)) for k in must_fail)
    # A daily currency file may contain several futures and the perp; missing
    # perp prints is a coverage warning, not a justification to manufacture data.
    perp = f'{currency}-PERPETUAL'
    suspicious = bool(ms_overflow or seq_gap > 0 or (kind == 'future' and perp not in totals) or count == 0)
    status = 'FAIL' if is_bad else 'WARN' if suspicious else 'PASS_WITH_LIMITATIONS'
    return {'currency': currency,'kind':kind,'day':day.isoformat(),'status':status,
        'rows':count,'instrument_count':len(totals),'perpetual_rows':totals.get(perp,{}).get('rows',0),
        'min_trade_ms':min_time,'max_trade_ms':max_time,'distinct_milliseconds':unique_ms,
        'max_trades_same_ms_currency':worst_ms,'millisecond_page_limit_buckets':ms_overflow,
        'possible_missing_trade_seq_span':int(seq_gap),'exact_ids_checked':exact_ids,
        'checks':dict(bad),'instruments':dict(totals),
        'certainty':'Local files cannot certify complete remote trade history; timestamp paging may omit crowded-ms tails.'}


def audit_path(path: Path, currency: str, kind: str, d: date, batch_size=BATCH_SIZE, exact_ids=False) -> dict:
    if not path.exists():
        # The downloader atomically renames DATE.tmp -> DATE.parquet when the day
        # finishes; never treat a .tmp as complete, and don't re-download it.
        status = 'IN_PROGRESS_TMP' if path.with_suffix('.tmp').exists() else 'MISSING_FILE'
        return {'currency':currency,'kind':kind,'day':d.isoformat(),'status':status,'rows':0,'path':str(path)}
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        columns = list(set(REQUIRED) & set(pf.schema.names))
        batches = (b.to_pandas() for b in pf.iter_batches(batch_size=batch_size, columns=columns))
        result = audit_frames(batches, currency,kind,d,exact_ids=exact_ids)
        result['path'] = str(path)
        result['parquet_rows'] = pf.metadata.num_rows
        result['file_bytes'] = path.stat().st_size
        if result['rows'] != pf.metadata.num_rows:
            result['status'] = 'FAIL'
            result['checks']['parquet_vs_observed_row_mismatch'] = 1
        return result
    except Exception as exc:
        return {'currency':currency,'kind':kind,'day':d.isoformat(),'status':'UNREADABLE',
                'path':str(path),'error':repr(exc)}


def archive_root(raw: Path | None) -> Path:
    if raw is None:
        val = os.getenv('DERIBIT_DATA_ROOT')
        if not val:
            raise ValueError('Pass --history ROOT or set DERIBIT_DATA_ROOT')
        raw = Path(val)
    raw = Path(raw)
    # support DERIBIT_DATA_ROOT, its history dir, or explicit trades dir.
    if (raw/'history'/'trades').exists():return raw/'history'
    if (raw/'trades').exists():return raw
    if raw.name == 'trades': return raw.parent
    return raw / 'history'


def inventory(history: Path, currencies=('BTC','ETH'),kind='future',start='2024-01-01',end=None,
              batch_size=BATCH_SIZE,exact_ids=False,days=None) -> tuple[pd.DataFrame,dict]:
    if days is None:
        end = end or str(date.today()-timedelta(days=1))
        first,last=date.fromisoformat(start),date.fromisoformat(end)
        if last<first:raise ValueError('End date precedes start')
        dates=[first+timedelta(days=i) for i in range((last-first).days+1)]
    else:
        dates=sorted({date.fromisoformat(str(x)) for x in days})
    records=[]; detail=[]
    for cur in currencies:
        for d in dates:
            p=history/'trades'/cur/kind/f'{d:%Y}'/f'{d:%Y-%m-%d}.parquet'
            row=audit_path(p,cur,kind,d,batch_size,exact_ids)
            detail.append(row)
            records.append({k:v for k,v in row.items() if k not in ('checks','instruments','certainty')}
                           | {'invalid_fields':json.dumps(row.get('checks',{}),sort_keys=True)})
    result=pd.DataFrame(records)
    counts=result.status.value_counts().to_dict()
    # Compare per-instrument adjacent-day trade_seq bounds. Only warn: some
    # exchange feed adjustments can violate ideal consecutive sequences.
    chain=[]
    by_instrument=defaultdict(list)
    for item in detail:
        if item['status'] in ('FAIL','UNREADABLE','MISSING_FILE','IN_PROGRESS_TMP'):
            continue
        for inst, stat in item.get('instruments',{}).items():
            if stat['min_seq'] is not None:
                by_instrument[(item['currency'],inst)].append((date.fromisoformat(item['day']),stat))
    for (cur,inst), history_rows in by_instrument.items():
        for (prev_date,prev),(cur_date,cur_stat) in zip(history_rows,history_rows[1:]):
            if (cur_date-prev_date).days != 1:
                continue
            gap=int(cur_stat['min_seq'])-int(prev['max_seq'])-1
            if gap != 0:
                chain.append({'currency':cur,'instrument':inst,'prev_day':str(prev_date),
                              'current_day':str(cur_date),'missing_seq_if_continuous':max(0,gap),
                              'overlap_seq_if_continuous':max(0,-gap)})
    meta={'version':VERSION,'kind':kind,'currencies':list(currencies), 'date_start':str(min(dates)) if dates else None,
          'date_end':str(max(dates)) if dates else None,'calendar_days_per_currency':len(dates),
          'files_scanned':len(records),'status_counts':counts,'cross_day_sequence_anomalies':chain,'per_file':detail,
          'archive_complete_certified':False,'caveat':'PASS_WITH_LIMITATIONS is local integrity only, not exchange-completeness proof.'}
    return result,meta


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history',type=Path,help='DERIBIT_DATA_ROOT, history directory, or trades directory')
    p.add_argument('--currencies',nargs='+',choices=('BTC','ETH'),default=['BTC','ETH'])
    p.add_argument('--kind',choices=('future','option'),default='future')
    p.add_argument('--start',default='2024-01-01')
    p.add_argument('--end',default=None)
    p.add_argument('--batch-size',type=int,default=BATCH_SIZE)
    p.add_argument('--exact-ids',action='store_true',help='Memory-intensive exact duplicate trade ID detection per day')
    p.add_argument('--out',type=Path,default=Path('notebooks/Liquidity_Absorption/05b/validator'))
    a=p.parse_args(argv)
    history=archive_root(a.history)
    days,meta=inventory(history,tuple(a.currencies),a.kind,a.start,a.end,a.batch_size,a.exact_ids)
    a.out.mkdir(parents=True,exist_ok=True)
    days.to_csv(a.out/'deribit_daily_qc.csv',index=False)
    (a.out/'deribit_daily_qc.json').write_text(json.dumps(meta,indent=2,default=str)+'\n',encoding='utf8')
    print(json.dumps({'version':VERSION,'rows':len(days),'statuses':meta['status_counts'],
                      'out':str(a.out),'certified_complete':False},indent=2))
    return 1 if any(k in meta['status_counts'] for k in ('FAIL','UNREADABLE','MISSING_FILE')) else 0


if __name__=='__main__':
    raise SystemExit(main())
