"""Unit tests run without local market archives or pyarrow."""
import importlib.util
import sys
from pathlib import Path
from datetime import date
import unittest
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dhv',ROOT/'tools'/'validate_deribit_history_trades.py')
mod=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=mod
spec.loader.exec_module(mod)


def prints(timestamps, seq=None, names=None, ids=None):
    n=len(timestamps)
    return pd.DataFrame({'trade_id':ids or [f'BTC-PERPETUAL-{i}' for i in range(n)],
        'trade_seq':seq or list(range(10,10+n)),
        'instrument_name':names or ['BTC-PERPETUAL']*n,
        'timestamp':pd.to_datetime(timestamps,unit='ms',utc=True).as_unit('ms'),
        'direction':['buy','sell']*(n//2)+(['buy'] if n%2 else []),
        'price':[50000+i for i in range(n)], 'amount':[10]*n})

class TestValidator(unittest.TestCase):
    def test_datetime_precision_utc_ms_us_ns(self):
        v=pd.Series(pd.to_datetime([1782864000000,1782864000001],unit='ms',utc=True))
        for unit in ('ms','us','ns'):
            self.assertEqual(mod.epoch_ms(v.dt.as_unit(unit)).tolist(),[1782864000000,1782864000001])

    def test_basic_ok(self):
        a=prints([1782864000001,1782864000001,1782864000005])
        z=mod.audit_frames([a], 'BTC','future',date(2026,7,1))
        self.assertEqual(z['status'],'PASS_WITH_LIMITATIONS')
        self.assertEqual(z['distinct_milliseconds'],2)
        self.assertEqual(z['max_trades_same_ms_currency'],2)
        self.assertEqual(z['possible_missing_trade_seq_span'],0)
        self.assertEqual(z['perpetual_rows'],3)

    def test_ms_bucket_across_chunks_not_double_counted(self):
        a=prints([1782864000001]*4)
        z=mod.audit_frames([a.iloc[:2],a.iloc[2:]],'BTC','future',date(2026,7,1))
        self.assertEqual(z['distinct_milliseconds'],1)
        self.assertEqual(z['max_trades_same_ms_currency'],4)

    def test_same_millisecond_page_limit_warn(self):
        x=prints([1782864000001]*1001)
        z=mod.audit_frames([x.iloc[:640],x.iloc[640:]],'BTC','future',date(2026,7,1))
        self.assertEqual(z['status'],'WARN')
        self.assertEqual(z['millisecond_page_limit_buckets'],1)
        self.assertEqual(z['max_trades_same_ms_currency'],1001)

    def test_ms_tail_finalized_on_new_timestamp_across_batches(self):
        x=prints([1782864000001]*1000+[1782864000003])
        z=mod.audit_frames([x.iloc[:1000],x.iloc[1000:]],'BTC','future',date(2026,7,1))
        self.assertEqual(z['millisecond_page_limit_buckets'],1)
        self.assertEqual(z['max_trades_same_ms_currency'],1000)

    def test_seq_missing_trades_warning(self):
        x=prints([1782864000001,1782864001001],seq=[100,103])
        z=mod.audit_frames([x],'BTC','future',date(2026,7,1))
        self.assertEqual(z['possible_missing_trade_seq_span'],2)
        self.assertEqual(z['status'],'WARN')

    def test_mixed_future_instruments_no_global_seq_overlap_errors(self):
        x=prints([1782864000001,1782864000002,1782864000003,1782864000004],
          seq=[10,1,11,2],names=['BTC-PERPETUAL','BTC-25DEC26','BTC-PERPETUAL','BTC-25DEC26'])
        z=mod.audit_frames([x],'BTC','future',date(2026,7,1))
        self.assertEqual(z['instrument_count'],2)
        self.assertEqual(z['possible_missing_trade_seq_span'],0)

    def test_exact_duplicate_ids_caught_even_nonadjacent(self):
        x=prints([1782864000001,1782864000002,1782864000003],ids=['a','b','a'])
        z=mod.audit_frames([x],'BTC','future',date(2026,7,1),exact_ids=True)
        self.assertEqual(z['checks']['duplicate_trade_id_exact'],1)
        self.assertEqual(z['status'],'FAIL')

    def test_bad_timestamp_out_of_day(self):
        x=prints([1782863999000])
        z=mod.audit_frames([x],'BTC','future',date(2026,7,1))
        self.assertEqual(z['checks']['out_of_day_timestamp'],1)
        self.assertEqual(z['status'],'FAIL')

    def test_invalid_fields(self):
        x=prints([1782864000001]); x.loc[0,'price']=0; x.loc[0,'direction']='unknown'
        z=mod.audit_frames([x],'BTC','future',date(2026,7,1))
        self.assertEqual(z['status'],'FAIL')
        self.assertEqual(z['checks']['bad_price'],1)
        self.assertEqual(z['checks']['bad_direction'],1)

    def test_missing_future_day_never_imputed(self):
        p=ROOT/'never-historical-file-should-exist.parquet'
        z=mod.audit_path(p,'BTC','future',date(2026,7,1))
        self.assertEqual(z['status'],'MISSING_FILE')

    def test_empty_daily_future_file_warning(self):
        x=prints([])
        z=mod.audit_frames([x],'BTC','future',date(2026,7,1))
        self.assertEqual(z['status'],'WARN')
        self.assertEqual(z['perpetual_rows'],0)

if __name__=='__main__':unittest.main()
