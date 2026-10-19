"""Tests for LA-A02 causal candidate detection. Run via python -m unittest."""
import unittest
from unittest.mock import patch
import tempfile
import sys
import types
from pathlib import Path
import numpy as np
import pandas as pd

from src.data_engineering.modeling.liquidity_absorption_stage_a02 import (
    Config, get_minutes, get_reference_levels, detect_price_only_events,
    trade_profile_window_features,
)


def sample_bars():
    tt = pd.date_range('2026-07-01', periods=1440+25, freq='min', tz='UTC')
    n = len(tt)
    data = pd.DataFrame({'open_time': tt, 'open': np.full(n, 100.5), 'high': np.full(n, 100.8),
                         'low': np.full(n, 100.0), 'close': np.full(n, 100.5),
                         'volume': np.full(n, 10.), 'quote_volume': np.full(n, 1000.),
                         'taker_buy_quote_volume': np.full(n, 400.)})
    for i in range(1440, n):
        data.loc[i, ['open','high','low','close']] = [100.5, 100.7, 100.3, 100.5]
    # Prior day low is 100.0. Two one-minute touches with a completed excursion.
    data.loc[1441, ['open','high','low','close']] = [100.06, 100.15, 99.99, 100.08]
    data.loc[1442, ['open','high','low','close']] = [100.12, 100.23, 100.1, 100.2]
    data.loc[1443, ['open','high','low','close']] = [100.14, 100.22, 100.12, 100.18]
    data.loc[1445, ['open','high','low','close']] = [100.06, 100.16, 99.99, 100.08]
    return data


class TestA02(unittest.TestCase):
    def cfg(self):
        return Config(start='2026-07-02', end='2026-07-02', warmup_days=1)

    def test_two_touch_requires_prior_intervening_departure(self):
        c = self.cfg()
        x = get_minutes(sample_bars(), c)
        e = detect_price_only_events(x, c, 'BTCUSDT')
        sup = e[(e.side == 'long') & (e.level_type == 'prior_day')]
        self.assertEqual(len(sup), 1)
        self.assertEqual(sup.iloc[0]['signal_time_utc'], pd.Timestamp('2026-07-02 00:06Z'))
        self.assertEqual(sup.iloc[0]['level_price'], 100.)
        self.assertGreater(sup.iloc[0]['minutes_between_touches'], 3)

    def test_no_early_second_touch_without_excursion(self):
        c = self.cfg()
        raw = sample_bars()
        # Replace intermediate departure candles with close just above zone; keep both touches.
        for i in (1442,1443,1444):
            raw.loc[i, ['open','high','low','close']] = [100.05, 100.08, 100.04, 100.05]
        e = detect_price_only_events(get_minutes(raw,c),c,'BTCUSDT')
        sup=e[(e.side=='long') & (e.level_type=='prior_day')]
        self.assertEqual(len(sup), 0)

    def test_no_future_information_in_features(self):
        c = self.cfg()
        before = sample_bars()
        after = before.copy()
        after.loc[1448:, ['open','high','low','close']] = [200, 250, 180, 210]
        a=detect_price_only_events(get_minutes(before,c),c,'BTCUSDT')
        b=detect_price_only_events(get_minutes(after,c),c,'BTCUSDT')
        a=a[(a.side=='long') & (a.level_type=='prior_day')].iloc[0]
        b=b[(b.side=='long') & (b.level_type=='prior_day')].iloc[0]
        for key in ('signal_time_utc','bar_close','delta_5m','delta_15m','level_price'):
            self.assertEqual(a[key], b[key])

    def test_unconfirmed_pivot_not_available(self):
        c=self.cfg()
        raw=sample_bars()
        x=get_minutes(raw,c)
        z=get_reference_levels(x,c)
        # Pivot time must be at least 10 minutes old at first availability.
        y=z.loc[z.pivot_support_price.notna()]
        if not y.empty:
            self.assertTrue(((y.close_time-y.pivot_support_source_time)>=pd.Timedelta(minutes=10)).all())

    def test_epoch_ms_us_and_datetime_yield_identical_events(self):
        c=self.cfg()
        original=sample_bars()
        a=detect_price_only_events(get_minutes(original,c),c,'BTCUSDT')
        for unit in ('ms','us'):
            n=original.copy()
            t=original.open_time.astype('int64')
            n['open_time']=(t//(1_000_000 if unit=='ms' else 1_000)).astype('int64')
            b=detect_price_only_events(get_minutes(n,c),c,'BTCUSDT')
            pd.testing.assert_frame_equal(a,b,check_dtype=False)

    def test_profile_empty_candidates_does_not_need_trades(self):
        empty=pd.DataFrame(columns=['signal_time_utc'])
        got=trade_profile_window_features(empty,None,'BTCUSDT',self.cfg())
        self.assertEqual(len(got),0)


    def test_profile_uses_only_prior_ten_minutes_with_correct_side(self):
        c=self.cfg()
        signal=pd.Timestamp('2026-07-02 00:11:00Z')
        events=pd.DataFrame([{'signal_time_utc':signal,'level_price':100.,'zone_halfwidth':0.02}])
        ms=[int(pd.Timestamp(f'2026-07-02 00:{m:02d}:00Z').timestamp()*1000) for m in range(1,12)]
        trade=pd.DataFrame({'time_ms':ms,'price':[100.]*11,
                            'qty':[0.5]*5+[1.]*5+[100.],
                            'is_buyer_maker':[False]*5+[True]*6})
        class Batch:
            def to_pandas(self): return trade.copy()
        class ParquetFile:
            def __init__(self, path): pass
            def iter_batches(self, **kwargs): yield Batch()
        parent=types.ModuleType('pyarrow')
        parent.__path__=[]
        fake=types.ModuleType('pyarrow.parquet')
        fake.ParquetFile=ParquetFile
        parent.parquet=fake
        # Deliberately do not mock actual Parquet bytes; isolate time/side/bin logic.
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'aggTrades'/'BTCUSDT'/'BTCUSDT-aggTrades-2026-07-02.parquet'
            f.parent.mkdir(parents=True)
            f.touch()
            with patch.dict(sys.modules, {'pyarrow':parent,'pyarrow.parquet':fake}):
                observed=trade_profile_window_features(events,Path(tmp),'BTCUSDT',c).iloc[0]
        self.assertEqual(observed.profile_status,'PRESENT')
        self.assertAlmostEqual(observed.zone_quote_prev5m,250.)
        self.assertAlmostEqual(observed.zone_quote_last5m,500.)
        self.assertAlmostEqual(observed.zone_buy_quote_last5m,0.)
        self.assertAlmostEqual(observed.zone_sell_quote_last5m,500.)
        self.assertAlmostEqual(observed.zone_growth_quote_ratio,2.)
        self.assertAlmostEqual(observed.zone_delta_last5m,-500.)

    def test_price_only_features_do_not_include_future_returns_or_pnl(self):
        c=self.cfg()
        e=detect_price_only_events(get_minutes(sample_bars(),c),c,'BTCUSDT')
        self.assertTrue({'delta_5m','delta_15m','signal_time_utc','level_price'}.issubset(e.columns))
        self.assertFalse({'future_return','pnl','target_hit','stop_hit'} & set(e.columns))


if __name__=='__main__':
    unittest.main()
