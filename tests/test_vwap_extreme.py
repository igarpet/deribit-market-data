"""Checks for VWAP Events 04 (vwap_extreme.py)."""
import numpy as np
import pandas as pd
import pytest

from data_engineering.modeling import vwap_events as ve
from data_engineering.modeling import vwap_extreme as vx
from data_engineering.modeling import vwap_fade as vf
from data_engineering.modeling import vwap_touch_stats as vt

XP = vx.XParams(n_boot=300)


def test_events_respect_threshold_warmup_and_cooldown():
    n = 600
    t = pd.date_range("2026-01-01 08:00", periods=n, freq="1min", tz="UTC")
    c = np.full(n, 100.0)
    c[100:400] = 101.2                     # ~1.2% above a VWAP that starts at 100
    df = pd.DataFrame({"open_time": t, "high": c, "low": c, "close": c, "volume": 1.0, "quote_volume": c})
    fr = ve.prepare(df, ve.Params()); st = vt.bar_states(fr, 1440, 480)
    ev = vx.extreme_events(fr, st, "pct", 0.01, XP)
    assert len(ev) > 0 and (ev.side == 1).all() and (ev.dist_bp >= 100).all()
    assert ev.i.iloc[0] == 100 and (np.diff(ev.i) >= XP.cooldown).all()
    early = vx.extreme_events(fr, st, "pct", 0.0, XP)
    assert early.i.min() >= XP.warmup_min


def test_day_ids_are_unit_safe_and_split_is_correct():
    df = vf.synthetic_path_klines(n_days=6, seed=1, start="2026-03-29")
    R = vx.run_asset(df, "S", vx.XParams(n_boot=100, k_sigma=(3.0,), pct=(), targets=(20,), stops=(20,), holds=(60,)))
    assert R.test.any() and (~R.test).any()
    days = pd.Timestamp("1970-01-01", tz="UTC") + pd.to_timedelta(np.unique(R.day), unit="D")
    assert days.min() >= pd.Timestamp("2026-03-29", tz="UTC") and days.max() <= pd.Timestamp("2026-04-04", tz="UTC")


@pytest.fixture(scope="module")
def null_run():
    xp = vx.XParams(n_boot=300, k_sigma=(3.0, 4.0), pct=(0.0035,), targets=(20, "vwap"), stops=(20, 40), holds=(60,))
    R = pd.concat([vx.run_asset(vf.synthetic_path_klines(n_days=140, seed=s, start="2025-12-01"), f"S{s}", xp)
                   for s in (1, 2)], ignore_index=True)
    return R, xp


def test_null_has_no_edge(null_run):
    R, xp = null_run
    dev = vx.dev_table(R, xp)
    cell = vx.select(dev, xp)
    res = vx.evaluate(R, cell, xp)
    assert res["verdict"] == "NO_EDGE"
    assert abs(res["test_gross"]) < 5
    ft = vx.full_table(R, xp)
    assert (ft[ft.kind == "real"].gross.abs() < 6).all()
