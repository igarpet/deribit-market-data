"""Synthetic validation for Signed Jumps 04 (signed_jumps_seconds.py)."""
import numpy as np
import pandas as pd
import pytest

from data_engineering.modeling import signed_jumps as sj
from data_engineering.modeling import signed_jumps_seconds as ss

P = sj.Params(n_boot=200, half_lives=(1,))
SP = ss.SecParams(n_boot=300)


def _pipe(**kw):
    secs, one = ss.simulate_seconds(n_days=14, seed=kw.pop("seed", 3), **kw)
    assets, pan, _ = sj.prepare(one, P)
    ev = ss.onset_seconds(ss.jump_minutes(assets, pan, SP.cluster_gap_min), secs)
    act = {a: ss.activity(secs[a], SP.baseline_s) for a in secs}
    plc = ss.placebo_onsets(ev, secs, SP)
    return secs, ev, act, plc


@pytest.fixture(scope="module")
def null():
    return _pipe()


@pytest.fixture(scope="module")
def planted():
    return _pipe(precursor=0.7, drift_bp=20, straddle_share=0.8)


def test_activity_baseline_is_strictly_causal():
    secs, _ = ss.simulate_seconds(n_days=1, n_assets=1, seed=1)
    df = secs["S0"]
    a0 = ss.activity(df, 600)
    df2 = df.copy()
    df2.iloc[50000:, df2.columns.get_loc("trades")] *= 10
    a1 = ss.activity(df2, 600)
    assert np.allclose(np.nan_to_num(a0[:50000], nan=-9), np.nan_to_num(a1[:50000], nan=-9))


def test_onset_located_within_jump_seconds(null):
    _, ev, _, _ = null
    e = ev[ev.onset.notna()]
    assert len(e) > 100
    assert (e.peak_share > 0.2).mean() > 0.9          # the planted 3-second jumps dominate their minute
    assert e.onset_u.between(1, 60).all()


def test_p1_null_fails_and_precursor_passes(null, planted):
    assert ss.p1_test(*_args(null), SP)["status"] == "FAIL"
    r = ss.p1_test(*_args(planted), SP)
    assert r["status"] == "PASS" and r["excess"] > 0.3


def _args(f):
    secs, ev, act, plc = f
    return ev, plc, secs, act


def test_p2_straddle_is_artefact_and_distinct_pairs_are_not(planted):
    _, ev, _, _ = planted
    _, s = ss.split_pairs(ev, SP)
    assert s["verdict"] == "MOSTLY_ARTEFACT"
    secs, one = ss.simulate_seconds(n_days=14, seed=5, pair_share=0.8, pair_gap_s=90)
    assets, pan, _ = sj.prepare(one, P)
    ev2 = ss.onset_seconds(ss.jump_minutes(assets, pan, SP.cluster_gap_min), secs)
    _, s2 = ss.split_pairs(ev2, SP)
    assert s2["pairs_same_sign"] >= 10 and s2["verdict"] == "MOSTLY_DISTINCT_EVENTS"


def test_p3_detects_drift_only_when_present(null, planted):
    for f, want in ((null, "FAIL"), (planted, "PASS")):
        secs, ev, _, _ = f
        _, r = ss.p3_table(ss.post_onset(ev, secs, SP), SP)
        assert r["status"] == want
