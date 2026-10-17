"""Checks for VWAP Events 02 descriptive statistics (vwap_touch_stats.py)."""
import numpy as np
import pandas as pd

from data_engineering.modeling import vwap_events as ve
from data_engineering.modeling import vwap_touch_stats as vt


def _klines(close, hi_pad=0.0, lo_pad=0.0, start="2026-01-01 08:00"):
    t = pd.date_range(start, periods=len(close), freq="1min", tz="UTC")
    c = np.asarray(close, float)
    return pd.DataFrame({"open_time": t, "high": c + hi_pad, "low": c - lo_pad, "close": c,
                         "volume": 1.0, "quote_volume": c})


def test_hand_built_path():
    # equal volume -> VWAP = running mean of closes since 08:00
    close = [100, 100, 104, 104, 96, 96, 96, 103, 103, 103]
    df = _klines(close)
    fr = ve.prepare(df, ve.Params())
    st = vt.bar_states(fr, 1440, 480)
    v = np.cumsum(close) / np.arange(1, len(close) + 1)
    side = np.sign(np.array(close) - v)
    assert list(st["side"].to_numpy()) == list(side.astype(int))
    ev, c = vt.events(st)
    # sides: 0,0,+,+,-,-,-,+,+,+  -> crosses at bars 4 (down) and 7 (up)
    assert list(ev.direction) == ["down", "up"]
    assert c["crosses"] == 2
    # high=low=close: touch only when the level equals the close exactly (bar 1: level 100, close 100)
    assert c["touch_bars"] == 1
    # after the down-cross at bar 4 the next up side is bar 7 -> 3 minutes
    assert ev.iloc[0]["min_to_cross_back"] == 3
    # after the up-cross at bar 7 no cross back before data ends -> censored
    assert np.isnan(ev.iloc[1]["min_to_cross_back"])


def test_touch_uses_previous_vwap_and_episodes_merge():
    close = [100, 100, 100, 100, 110, 110]
    df = _klines(close, hi_pad=0.5, lo_pad=0.5)
    fr = ve.prepare(df, ve.Params())
    st = vt.bar_states(fr, 1440, 480)
    # bars 1-3 sit on VWAP=100 -> touch run (one episode); bar 4 low 109.5 > level 100 -> no touch
    assert st["touch"].tolist() == [False, True, True, True, False, False]
    _, c = vt.events(st)
    assert c["touch_bars"] == 3 and c["touch_episodes"] == 1


def test_window_reset_censors():
    # 4H anchor: a cross late in a window cannot be "answered" by bars of the next window
    n = 480
    t = pd.date_range("2026-01-01 00:00", periods=n, freq="1min", tz="UTC")
    close = np.full(n, 100.0)
    close[200:238] = 99.0
    close[238:240] = 101.0          # cross up at bar 238, window ends at bar 240
    close[240:] = 98.0              # new window: would be a cross back, but must not count
    df = pd.DataFrame({"open_time": t, "high": close, "low": close, "close": close, "volume": 1.0,
                       "quote_volume": close})
    fr = ve.prepare(df, ve.Params())
    ev, _ = vt.events(vt.bar_states(fr, 240, 0))
    r = ev[ev.t == t[238]].iloc[0]
    assert np.isnan(r.min_to_cross_back) and r.min_to_window_end == 2


def test_km_median_handles_censoring():
    time = np.array([1, 2, np.nan, np.nan, 3.0])
    cens = np.array([9, 9, 10, 10, 9.0])
    assert vt.km_median(time, cens) == 3.0
    assert np.isnan(vt.km_median(np.array([np.nan, np.nan]), np.array([5.0, 5.0])))


def test_runs_on_synthetic_and_real_equals_placebo_shape():
    df = ve.synthetic_klines(n_days=20, seed=1)
    cnt, summ, ev = vt.run_asset(df, "SYN")
    assert set(cnt.kind) == {"real", "placebo"} and (cnt.crosses > 0).all()
    rvp = vt.real_vs_placebo(summ)
    assert len(rvp) == 2 * 3
    # no VWAP effect in synthetic data: real and placebo touch frequencies within 25%
    r = cnt.groupby(["anchor", "kind"]).touch_episodes_per_window.mean().unstack()
    assert ((r.real / r.placebo - 1).abs() < 0.25).all()
