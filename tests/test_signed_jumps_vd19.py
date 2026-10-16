"""Synthetic validation for Signed Jumps 05 (signed_jumps_vd19.py)."""
import numpy as np
import pandas as pd
import pytest

from data_engineering.modeling import model_vs_implied as mv
from data_engineering.modeling import signed_jumps as sj
from data_engineering.modeling import signed_jumps_vd19 as sv

P = sj.Params(n_boot=200, half_lives=(1,))
VP = sv.VdParams()


def test_entry_times_follow_vd19_sessions():
    r = pd.DataFrame({"session": ["2026-09-21", "2026-09-21", "2026-09-21"], "h": [240, 120, 1200]})
    t = sv.entry_times(r)
    assert list(t) == [pd.Timestamp("2026-09-21 04:00", tz="UTC"), pd.Timestamp("2026-09-21 06:00", tz="UTC"),
                       pd.Timestamp("2026-09-20 12:00", tz="UTC")]


@pytest.fixture(scope="module")
def eng():
    S = sj.simulate(n_days=30, n_assets=2, eta=((0.2, 0.05), (0.05, 0.3)), half_life=1, seed=3, df=8, base_rate=1e-3,
                    start="2026-09-01")
    S = {n: s for n, s in zip(("BTC", "ETH"), S.values())}
    assets, pan, _ = sj.prepare(S, P)
    fits, _ = sj.fit_ladder(assets, pan, P, models=("b", "e"))
    return assets, pan, fits


def test_origin_is_bar_after_entry(eng):
    assets, pan, _ = eng
    rows = pd.DataFrame({"session": ["2026-09-25", "2026-09-26"], "underlying": ["BTC", "ETH"], "h": [240, 120]})
    o = sv.origin_rows(pan, rows, VP)
    t = pd.DatetimeIndex(pan.time[o])
    t = t.tz_localize("UTC") if t.tz is None else t
    assert list(t) == [pd.Timestamp("2026-09-25 04:01", tz="UTC"), pd.Timestamp("2026-09-26 06:01", tz="UTC")]
    assert [pan.names[i] for i in pan.asset_idx[o]] == ["BTC", "ETH"]


def test_baseline_hazard_matches_exact_poisson_forecast(eng):
    """Model (b) has no excitation: Lambda over 60 min must equal -log P(no jump) of the exact SJ 01 forecast."""
    assets, pan, fits = eng
    rows_o = np.flatnonzero(~pan.dev)[::500][:40]
    lam, lam_b = sv.hazard_to_expiry(fits, assets, pan, P, rows_o, np.full(len(rows_o), 60), model="b")
    pb, _ = sj.horizon_forecast(fits, "b", assets, pan, P, rows_o)
    exact = -np.log1p(-pb[60]["up"]) - np.log1p(-pb[60]["down"])
    assert np.allclose(lam, exact, rtol=1e-6) and np.allclose(lam, lam_b)


def test_excitation_adds_hazard_after_jumps(eng):
    assets, pan, fits = eng
    J = (pan.J["up"] + pan.J["down"]) > 0
    after = np.flatnonzero(np.r_[False, J[:-1]] & (pan.asset_idx == np.r_[-1, pan.asset_idx[:-1]]))[:30]
    lam, lam_b = sv.hazard_to_expiry(fits, assets, pan, P, after, np.full(len(after), 30), model="e")
    assert (lam > lam_b).all()


@pytest.mark.parametrize("gamma,want", [(0.8, "JUMP_HAZARD_ADDS_INFORMATION"), (0.0, None)])
def test_decision_on_synthetic_rows(gamma, want):
    d = sv.synthetic_rows(n_sessions=30, gamma=gamma, seed=1)
    info = sv.information(d, mv, VP)
    dec = sv.decide(info, VP)
    if want:
        assert dec["verdict"] == want
    else:
        assert dec["verdict"] != "JUMP_HAZARD_ADDS_INFORMATION"
        assert dec["gamma_ci"][0] < 0 < dec["gamma_ci"][1]
