"""Checks for VWAP Events 03 (vwap_fade.py)."""
import numpy as np
import pandas as pd

from data_engineering.modeling import vwap_events as ve
from data_engineering.modeling import vwap_fade as vf
from data_engineering.modeling import vwap_touch_stats as vt

FP = vf.FadeParams(n_boot=300)


def _fr(close, pad=0.0, start="2026-01-01 08:00"):
    t = pd.date_range(start, periods=len(close), freq="1min", tz="UTC")
    c = np.asarray(close, float)
    df = pd.DataFrame({"open_time": t, "high": c * (1 + pad), "low": c * (1 - pad), "close": c,
                       "volume": 1.0, "quote_volume": c})
    return ve.prepare(df, ve.Params())


def test_simulate_target_stop_and_time_exit():
    n = 200
    close = np.full(n, 100.0)
    fr = _fr(close)
    st = vt.bar_states(fr, 1440, 480)
    ev = pd.DataFrame({"i": [10], "side": [1]})
    # long from bar 11 close; price rises 0.3% at bar 15 -> 10 bp target hit
    c2 = close.copy(); c2[15:] = 100.3
    fr2 = _fr(c2); st2 = vt.bar_states(fr2, 1440, 480)
    assert vf.simulate(fr2, st2, ev, np.array([1]), 10, 20, FP)[0] == 10
    # falls 0.3% -> 20 bp stop
    c3 = close.copy(); c3[15:] = 99.7
    fr3 = _fr(c3); st3 = vt.bar_states(fr3, 1440, 480)
    assert vf.simulate(fr3, st3, ev, np.array([1]), 40, 20, FP)[0] == -20
    # flat -> time exit at 0
    assert vf.simulate(fr, st, ev, np.array([1]), 40, 20, FP)[0] == 0


def test_stop_wins_ties_in_one_bar():
    n = 200
    close = np.full(n, 100.0)
    t = pd.date_range("2026-01-01 08:00", periods=n, freq="1min", tz="UTC")
    hi, lo = close.copy(), close.copy()
    hi[15], lo[15] = 101, 99
    df = pd.DataFrame({"open_time": t, "high": hi, "low": lo, "close": close, "volume": 1.0, "quote_volume": close})
    fr = ve.prepare(df, ve.Params()); st = vt.bar_states(fr, 1440, 480)
    assert vf.simulate(fr, st, pd.DataFrame({"i": [10], "side": [1]}), np.array([1]), 10, 10, FP)[0] == -10


def test_vwap_exit_pays_distance_to_level():
    # price jumps to 101, then falls back to the VWAP area; short (fade) from 101 exits at the VWAP touch level
    close = np.r_[np.full(100, 100.0), np.full(40, 101.0), np.full(160, 100.3)]
    fr = _fr(close, pad=0.0005)
    st = vt.bar_states(fr, 1440, 480)
    ev = pd.DataFrame({"i": [120], "side": [1]})
    p = vf.simulate(fr, st, ev, np.array([-1]), "vwap", 200, FP)[0]
    lev = st["lev"].iloc[140]
    assert abs(p - (-(np.log(lev) - np.log(101.0)) * 1e4)) < 1e-6 and p > 50
    # the same path never returning: time exit, not a phantom target
    c2 = np.r_[np.full(100, 100.0), np.full(200, 101.0)]
    fr2 = _fr(c2, pad=0.0005); st2 = vt.bar_states(fr2, 1440, 480)
    assert vf.simulate(fr2, st2, ev, np.array([-1]), "vwap", 200, FP)[0] == 0


def test_random_walk_has_zero_gross_and_real_equals_placebo():
    rows = []
    for seed in (1, 2):
        df = vf.synthetic_path_klines(n_days=40, seed=seed)
        rows.append(vf.run_asset(df, f"S{seed}", vf.FadeParams(n_boot=200, k_grid=(1.0,), targets=(10, "vwap"),
                                                                 stops=(10, 20))))
    R = pd.concat(rows, ignore_index=True)
    res = vf.primary(R, vf.FadeParams(n_boot=300, k_grid=(1.0,)))
    assert res["G1_vwap_specific"] == "FAIL" and res["G2_tradeable"] == "FAIL"
    assert abs(res["real_gross_bp"]) < 2.0
    g = vf.grid_table(R, vf.FadeParams(n_boot=200))
    sym = g[(g.kind == "real") & (g.target == "10") & (g.stop == 10) & (g["mode"] == "fade")].iloc[0]
    assert 0.4 < sym.win_rate < 0.6          # symmetric target/stop on a driftless walk


def test_no_false_vwap_specific_pass_across_seeds():
    """Seasonality peaks near the real 08:00 anchor; the hour-matched G1 must not pass on a pure random walk."""
    fp = vf.FadeParams(n_boot=300, k_grid=(1.0,), targets=("vwap",), stops=(20,))
    passes = 0
    for seed in range(3, 9):
        R = pd.concat([vf.run_asset(vf.synthetic_path_klines(n_days=45, seed=10 * seed + j), f"A{j}", fp)
                       for j in (0, 1)], ignore_index=True)
        res = vf.primary(R, fp)
        passes += res["G1_vwap_specific"] == "PASS"
        assert abs(res["real_gross_bp"]) < 2.5 and res["G2_tradeable"] == "FAIL"
    assert passes <= 1
