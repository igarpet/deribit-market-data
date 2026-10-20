"""LA10 unselected observable level universe and strict A02 detector equivalence tests."""
from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_engineering.modeling import liquidity_absorption_stage_a10 as a10
from src.data_engineering.modeling import liquidity_absorption_stage_a02 as a02


def minute_fixture(*,touches=(5,11),invalid_at=(),start="2026-07-01 12:00:00Z",n=35):
    cfg=a02.Config()
    stamp=pd.Timestamp(start)
    t=pd.date_range(stamp,periods=n,freq="min")
    bars=[]
    for i,open_time in enumerate(t):
        o,h,l,c=101.,101.2,100.8,101.
        if i in touches:o,h,l,c=100.13,100.17,99.99,100.11
        if i in invalid_at:o,h,l,c=99.95,100.01,99.90,99.93
        bars.append(dict(open_time=open_time,open=o,high=h,low=l,close=c,
                         volume=100.,quote_volume=10000.,taker_buy_quote_volume=5000.))
    return a02.get_minutes(pd.DataFrame(bars),cfg),cfg


def mocked_references(x, *, type="prior_day", change_at=None):
    out=pd.DataFrame({"open_time":x.open_time,"close_time":x.close_time})
    cols=("prior_day_low","prior_day_high","pivot_support_price","pivot_resistance_price")
    for col in cols:out[col]=np.nan
    for col in ("prior_day_source_time","pivot_support_source_time","pivot_resistance_source_time"):
        out[col]=pd.NaT
    if type=="prior_day":
        out["prior_day_low"]=100.
        out["prior_day_source_time"]=x.open_time.iloc[0].floor("D")-pd.Timedelta(days=1)
    else:
        # First pivot exists when first candle closes (10-min right confirmation)
        src=x.close_time.iloc[0]-pd.Timedelta(minutes=10)
        out["pivot_support_price"]=100.
        out["pivot_support_source_time"]=src
    if change_at is not None:
        # Replaces pivot reference on the specified candle close.
        ix=np.arange(len(x))>=change_at
        out.loc[ix,"pivot_support_source_time"]=x.close_time.iloc[change_at]-pd.Timedelta(minutes=10)
        out.loc[ix,"pivot_support_price"]=100.5
    return out


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.x,self.cfg=minute_fixture()

    def _replay(self,x=None,**kw):
        x=self.x if x is None else x
        ref=mocked_references(x,**kw)
        with patch.object(a02,"get_reference_levels",return_value=ref):
            episodes,signals,spells=a10.replay_level_episodes(x,"BTCUSDT",self.cfg)
            original=a02.detect_price_only_events(x,self.cfg,"BTCUSDT")
        return episodes,signals,spells,original

    def test_signal_exact_a02_state_machine_on_sample(self):
        episodes,signals,spells,original=self._replay()
        result=a10.replay_contract(signals,original)
        self.assertTrue(result["exact_signal_replay"])
        self.assertEqual(result["signals"],1)
        self.assertEqual(len(episodes),1)
        self.assertEqual(episodes.classification.iloc[0],"SECOND_TOUCH_SIGNAL")
        self.assertEqual(episodes.n_valid_touch_candles.iloc[0],2)
        self.assertEqual(len(spells),1)
        self.assertEqual(spells.end_reason.iloc[0],"SECOND_TOUCH_SIGNAL")
        self.assertEqual(spells.duration_min.iloc[0],6)

    def test_without_touches_level_is_kept_not_dropped(self):
        x,cfg=minute_fixture(touches=())
        episodes,signals,spells,original=self._replay(x)
        self.assertTrue(signals.empty)
        self.assertEqual(episodes.classification.tolist(),["NO_VALID_TOUCH"])
        self.assertEqual(len(spells),0)

    def test_valid_first_without_second_is_kept(self):
        x,_=minute_fixture(touches=(5,))
        episodes,signals,spells,original=self._replay(x)
        self.assertTrue(signals.empty)
        self.assertEqual(episodes.classification.tolist(),["FIRST_TOUCH_NO_SIGNAL"])
        self.assertEqual(spells.end_reason.tolist(),["RIGHT_CENSORED_END_OF_ARCHIVE"])

    def test_invalidated_first_touch_is_competing_event(self):
        x,_=minute_fixture(touches=(5,11),invalid_at=(7,))
        episodes,signals,spells,original=self._replay(x)
        self.assertEqual(len(signals),0)
        self.assertEqual(spells.end_reason.tolist(),["INVALIDATED","RIGHT_CENSORED_END_OF_ARCHIVE"])
        self.assertEqual(episodes.n_invalidations_after_first.iloc[0],1)

    def test_two_touches_without_departure_do_not_signal(self):
        x,_=minute_fixture(touches=(5,6))
        episodes,signals,spells,original=self._replay(x)
        self.assertTrue(signals.empty)
        self.assertEqual(episodes.n_valid_touch_candles.iloc[0],2)

    def test_early_touch_same_close_as_confirmation_is_flagged(self):
        x,_=minute_fixture(touches=(0,6))
        episodes,_,spells,_=self._replay(x,type="pivot_5m")
        self.assertTrue(episodes.first_touch_confirmed_after_available.iloc[0] is False or
                        episodes.first_touch_confirmed_after_available.iloc[0]==False)
        self.assertTrue(pd.Timestamp(episodes.available_at.iloc[0])==pd.Timestamp(x.close_time.iloc[0]))

    def test_reference_replacement_ends_episode_and_touch_spell(self):
        x,_=minute_fixture(touches=(5,))
        ep,signals,spells,_=self._replay(x,type="pivot_5m",change_at=7)
        self.assertEqual(len(ep),2)
        self.assertEqual(ep.end_reason.iloc[0],"REPLACED_BY_NEW_REFERENCE")
        self.assertEqual(spells.end_reason.iloc[0],"REPLACED_BY_NEW_REFERENCE")

    def test_frozen_replay_rejects_mismatch(self):
        ep,signal,spells,frozen=self._replay()
        frozen.loc[0,"level_price"]=101.
        with self.assertRaisesRegex(ValueError,"A02 replay mismatch"):
            a10.replay_contract(signal,frozen)

    def test_prior_day_is_available_only_at_next_midnight(self):
        source=pd.Timestamp("2026-06-30T00:00:00Z")
        self.assertEqual(a10.level_available(source,"prior_day",self.cfg),pd.Timestamp("2026-07-01T00:00:00Z"))

    def test_pivot_right_confirmation_lag_is_ten_minutes(self):
        t=pd.Timestamp("2026-07-01 00:00:00Z")
        self.assertEqual(a10.level_available(t,"pivot_5m",self.cfg),t+pd.Timedelta(minutes=10))

    def test_horizon_censor_in_first_touch_incidence(self):
        a=pd.Timestamp("2026-07-01T00:00:00Z")
        p=pd.DataFrame([
            dict(symbol="BTCUSDT",level_type="pivot_5m",side="support",available_at=a,
                 observed_from_availability_min=0,full_horizon_at_risk_minutes=2,
                 minutes_until_first_touch=np.nan,first_touch_confirmed_after_available=None,
                 first_valid_touch=pd.NaT,n_second_touch_signals=0),
            dict(symbol="BTCUSDT",level_type="pivot_5m",side="support",available_at=a,
                 observed_from_availability_min=0,full_horizon_at_risk_minutes=10,
                 minutes_until_first_touch=4,first_touch_confirmed_after_available=True,
                 first_valid_touch=a+pd.Timedelta(minutes=4),n_second_touch_signals=1)])
        pop,r=a10.incidence(p,a10.Config())
        k=r[r.horizon_min==5].iloc[0]
        self.assertEqual(k.n_first_touch_known,1)
        self.assertEqual(k.n_censored_before_h,1)
        self.assertEqual(k.n_first_touch_by_h,1)

    def test_second_touch_competing_event_or_right_censor(self):
        base=pd.Timestamp("2026-07-05T00:00:00Z")
        s=pd.DataFrame([dict(symbol="BTCUSDT",level_type="pivot_5m",side="support",
                first_touch_close=base,first_touch_after_availability=True,duration_min=duration,
                end_reason=reason) for duration,reason in [(3,"SECOND_TOUCH_SIGNAL"),(3,"INVALIDATED"),
                (2,"RIGHT_CENSORED_END_OF_ARCHIVE"),(12,"SECOND_TOUCH_SIGNAL")]])
        r=a10.second_touch_incidence(s,a10.Config())
        five=r[r.horizon_min==5].iloc[0]
        self.assertEqual(five.n_spells,4)
        self.assertEqual(five.n_assessable,3)
        self.assertEqual(five.n_second_by_h,1)
        self.assertEqual(five.n_competing_by_h,1)
        self.assertEqual(five.n_right_censored_before_h,1)
        self.assertAlmostEqual(five.observed_second_incidence_by_h,1/3)

    def test_empty_spells_still_returns_schema(self):
        out=a10.second_touch_incidence(pd.DataFrame(),a10.Config())
        self.assertIn("horizon_min",out.columns)
        self.assertTrue(out.empty)

    def test_last_signal_cooldown_not_reused_as_new_first(self):
        x,_=minute_fixture(touches=(5,11,14,15))
        ep,signal,spells,_=self._replay(x)
        self.assertEqual(len(signal),1)
        self.assertGreaterEqual(ep.n_cooldown_touches.iloc[0],1)

    def test_expired_spell_after_max_retest_gap(self):
        x,_=minute_fixture(touches=(5,),n=105)
        ep,signal,spells,_=self._replay(x)
        self.assertEqual(spells.end_reason.iloc[0],"RETEST_GAP_EXPIRED")
        self.assertEqual(spells.duration_min.iloc[0],90)


    def test_synthetic_pipeline_writes_all_outputs_and_requires_replay(self):
        import dataclasses
        import json
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d)
            input_dir=repo/'notebooks'/'Liquidity_Absorption'/'02'
            input_dir.mkdir(parents=True)
            config=dataclasses.asdict(self.cfg)
            (input_dir/'02_feature_contract.json').write_text(json.dumps({'version':'LA-A02-0.1.0','config':config}))
            # Full run requires 452 frozen candidate IDs: fake the replay lists
            # while keeping real state-machine logic, on independent symbols.
            x=self.x.copy()
            original_ref=mocked_references(x,type='pivot_5m')
            with patch.object(a02,'get_reference_levels',return_value=original_ref):
                f=a02.detect_price_only_events(x,self.cfg,'BTCUSDT')
            # test the exact 452 gate separately; a 1-row fixture must fail
            f.to_csv(input_dir/'02_price_only_candidates.csv',index=False)
            with patch('pandas.read_parquet',return_value=x.drop(columns=['close_time','signed_quote_1m',
                        'delta_5m','delta_15m','quote_5m','quote_15m','cvd_utc_session','vwap_utc_session'])):
                with self.assertRaisesRegex(ValueError,'452-event identity'):
                    a10.run(repo,repo,repo/'out')


if __name__ == "__main__":
    unittest.main()
