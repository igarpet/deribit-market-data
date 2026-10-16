"""Synthetic validation for Signed Jumps 02a (signed_jumps_events.py)."""
import numpy as np
import pandas as pd
import pytest

from data_engineering.modeling import signed_jumps as sj
from data_engineering.modeling import signed_jumps_events as se

P = sj.Params(n_boot=200, half_lives=(1, 3), min_test_jumps_per_sign=20)
EP = se.EventParams(n_boot=300, min_test_events=50)


def _setup(eta=((0.0, 0.0), (0.0, 0.0)), seed=3, n_days=50):
    S = sj.simulate(n_days=n_days, n_assets=2, eta=eta, half_life=1, seed=seed, df=8, base_rate=1e-3)
    assets, pan, _ = sj.prepare(S, P)
    fits, _ = sj.fit_ladder(assets, pan, P, models=("e",))
    rows = []
    for (m, s), f in fits.items():
        r = {"model": m, "sign": s, "half_life_min": f.half_life, "ll_dev": f.ll_dev, "n_dev": f.n_dev,
             "converged": f.converged}
        r.update(dict(zip(sj.design_names(pan, m, P), f.b)))
        r.update({f"eta{k}": v for k, v in enumerate(f.eta)})
        rows.append(r)
    return S, assets, pan, fits, pd.DataFrame(rows)


@pytest.fixture(scope="module")
def excited():
    return _setup(eta=((0.25, 0.0), (0.0, 0.3)))


@pytest.fixture(scope="module")
def null():
    return _setup(seed=11)


def test_fits_table_roundtrip(excited):
    S, assets, pan, fits, tab = excited
    f2 = se.fits_from_table(tab, pan, P)
    lam1 = se.intensities(fits, assets, pan)
    lam2 = se.intensities(f2, assets, pan)
    for s in sj.SIGNS:
        assert np.allclose(lam1[s], lam2[s])


def test_pi_c_reflects_continuation_excitation(excited):
    S, assets, pan, fits, tab = excited
    lam = se.intensities(se.fits_from_table(tab, pan, P), assets, pan)
    ev = se.find_events(pan, lam, EP.cluster_gap)
    assert ev.pi_c.notna().mean() > 0.95
    assert ev.pi_c.median() > 0.6              # diagonal-only eta -> continuation favoured right after a jump
    assert set(ev.sign.unique()) <= {-1, 1}


def test_entry_price_is_at_jump_close_and_paths_are_signed(excited):
    S, assets, pan, fits, tab = excited
    lam = se.intensities(se.fits_from_table(tab, pan, P), assets, pan)
    ev = se.find_events(pan, lam, EP.cluster_gap).head(200)
    ev2 = se.attach_paths(ev, S, EP)          # perp == spot: the jump bar itself is NOT in any horizon return
    a = assets[0]
    s = S[a.name]
    r0 = ev2[ev2.asset == a.name].iloc[0]
    i = s.index.get_loc(r0.time)
    expect = r0.sign * (s.iloc[i + 5] - s.iloc[i]) * 1e4
    assert abs(r0.g_imm_5 - expect) < 1e-9
    expect_del = r0.sign * (s.iloc[i + 6] - s.iloc[i + 1]) * 1e4
    assert abs(r0.g_del_5 - expect_del) < 1e-9


def test_no_edge_without_drift_or_excitation(null):
    S, assets, pan, fits, tab = null
    lam = se.intensities(se.fits_from_table(tab, pan, P), assets, pan)
    ev = se.attach_paths(se.find_events(pan, lam, EP.cluster_gap), se.plant_drift(S, assets, 0.0, 10), EP)
    m = ev[ev.starter & ~ev.dev]
    for k in (5, 15, 60):
        b = se.boot_mean(m[f"g_imm_{k}"].to_numpy(), m.day.to_numpy(), 500, 1)
        assert b["lo"] < 0 < b["hi"] or abs(b["mean"]) < 6, k
    cuts = se.quintile_cuts(ev)
    res = se.evaluate(ev, se.select_rule(se.surface(ev, EP, cuts, "dev"), EP), cuts, EP)
    assert res["verdict"] != "NET_EDGE"


def test_recovers_planted_post_jump_drift(null):
    S, assets, pan, fits, tab = null
    lam = se.intensities(se.fits_from_table(tab, pan, P), assets, pan)
    ev0 = se.find_events(pan, lam, EP.cluster_gap)
    base = se.attach_paths(ev0, se.plant_drift(S, assets, 0.0, 10), EP)
    drift = se.attach_paths(ev0, se.plant_drift(S, assets, 3.0, 10), EP)
    m = ev0.starter.to_numpy()
    d10 = (drift.g_imm_10 - base.g_imm_10)[m].mean()
    assert 25 < d10 < 35                       # 3 bp x 10 min (overlapping clusters add a little)
    d60 = (drift.g_imm_60 - base.g_imm_60)[m].mean()
    assert abs(d60 - d10) < 5                  # drift stops after 10 minutes
    cuts = se.quintile_cuts(drift)
    res = se.evaluate(drift, se.select_rule(se.surface(drift, EP, cuts, "dev"), EP), cuts, EP)
    assert res["gross_lo"] > 0 and res["rule"]["direction"] == "continuation"
