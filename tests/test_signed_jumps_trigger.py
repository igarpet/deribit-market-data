"""Synthetic validation for Signed Jumps 04b (signed_jumps_trigger.py)."""
import numpy as np
import pandas as pd
import pytest

from data_engineering.modeling import signed_jumps as sj
from data_engineering.modeling import signed_jumps_seconds as ss
from data_engineering.modeling import signed_jumps_trigger as st

P = sj.Params(n_boot=200, half_lives=(1,))
TP = st.TrigParams(n_boot=300)


def _run(**kw):
    secs, one = ss.simulate_seconds(n_days=14, seed=kw.pop("seed", 4), **kw)
    assets, _, _ = sj.prepare(one, P)
    out = []
    for a in assets:
        tr = st.triggers(secs[a.name], st.sigma_minute(a), TP.c_trig, TP)
        out.append(st.forward(tr, secs[a.name], TP))
    ev = pd.concat(out, ignore_index=True)
    return secs, assets, ev, st.grid_table(ev, TP, TP.c_trig)


def test_trigger_is_causal():
    secs, one = ss.simulate_seconds(n_days=10, n_assets=1, seed=2)
    assets, _, _ = sj.prepare(one, P)
    a = assets[0]
    sec = secs[a.name]
    sig = st.sigma_minute(a)
    t0 = st.triggers(sec, sig, 4.0, TP)
    cut = len(sec) - 86400
    sec2 = sec.copy()
    sec2.iloc[cut:, sec2.columns.get_loc("logp")] += np.random.default_rng(0).normal(0, 0.01, len(sec2) - cut).cumsum()
    t1 = st.triggers(sec2, sig, 4.0, TP)
    assert t0[t0.pos < cut].pos.tolist() == t1[t1.pos < cut].pos.tolist()


def test_sigma_minute_matches_engine_scale():
    secs, one = ss.simulate_seconds(n_days=10, n_assets=1, seed=3, sigma_bp=0.8)
    assets, _, _ = sj.prepare(one, P)
    s = st.sigma_minute(assets[0]).dropna()
    # 1s sd 0.8 bp -> 1-minute sd ~ 0.8 * sqrt(60) = 6.2 bp
    assert 4e-4 < s.median() < 9e-4


def test_triggers_fire_on_planted_jumps():
    secs, assets, ev, _ = _run()
    assert 100 < len(ev) < 400                         # 6 jumps/day x 14 days x 2 assets ~ 168 (+ few noise)
    assert (ev.z.abs() >= 4).all() and (ev.R60_bp.abs() >= 5).all()


def test_no_continuation_without_drift():
    _, _, ev, tab = _run()
    r = st.decide(tab, TP)
    assert r["verdict"] == "NO_CONTINUATION" or abs(r["gross_bp"]) < 2


def test_detects_planted_post_jump_drift():
    _, _, ev, tab = _run(drift_bp=30, drift_s=120)
    r = st.decide(tab, TP)
    assert r["verdict"] == "CONTINUATION" and r["gross_bp"] > 8
    g = tab.set_index(["delta_s", "h_s"]).gross_bp
    assert g[(30, 60)] < g[(2, 60)]                    # later entry captures less of a decaying drift
