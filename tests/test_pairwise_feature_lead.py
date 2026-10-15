import numpy as np
import pandas as pd
import pytest

from data_engineering.modeling import pairwise_feature_lead as pl


def test_forward_returns_sum_and_gaps():
    t = pd.Series(pd.date_range("2026-10-01", periods=6, freq="5min", tz="UTC"))
    r = pd.Series([0.001, 0.002, 0.003, np.nan, 0.005, 0.006])
    y = pl.forward_returns(t, r, (5, 15))
    assert y.y_5.iloc[0] == pytest.approx(20.0)
    assert y.y_15.iloc[0] == pytest.approx(np.nan, nan_ok=True)      # includes the NaN bar
    assert y.y_5.iloc[3] == pytest.approx(50.0)
    assert np.isnan(y.y_5.iloc[5])


def test_forward_returns_missing_row_is_gap():
    t = pd.Series(pd.to_datetime(["2026-10-01 00:00", "2026-10-01 00:05", "2026-10-01 00:15"], utc=True))
    y = pl.forward_returns(t, pd.Series([0.001, 0.002, 0.003]), (5, 15))
    assert y.y_5.iloc[0] == pytest.approx(20.0)
    assert np.isnan(y.y_5.iloc[1])                                     # 00:10 bar missing


def test_moment_correlation_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((500, 3)); y = x[:, 0] * .3 + rng.standard_normal(500)
    x[::7, 1] = np.nan
    day = np.repeat(np.arange(5), 100)
    m = pl.day_moments(x, y, day, 5)
    r = pl.corr_from_sums({k: v.sum(0, keepdims=True) for k, v in m.items()})[0]
    ok = np.isfinite(x[:, 1])
    assert r[0] == pytest.approx(np.corrcoef(x[:, 0], y)[0, 1])
    assert r[1] == pytest.approx(np.corrcoef(x[ok, 1], y[ok])[0, 1])
    b = pl.bootstrap_corr(m, 200, 1)
    assert b.shape == (200, 3) and np.isfinite(b).all()


def test_residualize_removes_control():
    rng = np.random.default_rng(1)
    c = rng.standard_normal(1000); v = 2 * c + rng.standard_normal(1000)
    r = pl.residualize(v, c, np.ones(1000, bool))
    assert abs(np.corrcoef(r, c)[0, 1]) < 1e-10


def _screen(**kw):
    f, t, r = pl.synthetic_states(n_days=30, **kw)
    spec = pl.Specification(bootstrap_reps=400, min_obs=300)
    tab, meta = pl.screen_asset(f, t, r, spec)
    return tab.set_index(["feature", "horizon_min"])


def test_planted_lead_confirmed_noise_not():
    tab = _screen(seed=2, beta_lead=0.15)
    assert tab.loc[("lead", 5), "confirmed"]
    assert tab.loc[("lead", 5), "conf_long_short_bp"] > 0
    assert not tab.loc[("noise1", 5), "confirmed"] and not tab.loc[("noise2", 5), "confirmed"]


def test_bounce_artifact_removed_by_partial():
    tab = _screen(seed=3, beta_lead=0.0, beta_same=0.9, bounce=-0.15)
    assert abs(tab.loc[("same", 5), "conf_raw_ic"]) > 0.05             # raw IC picks up the bounce
    assert not tab.loc[("same", 5), "confirmed"]                         # partial IC does not


def test_decision_labels():
    rows = [{"asset": a, "feature": "obi_20", "horizon_min": 5, "confirmed": True, "covers_cost": False,
             "disc_partial_ic": .05, "conf_partial_ic": .04, "conf_long_short_bp": 1.0} for a in ("BTC", "ETH")]
    d = pl.decision(pd.DataFrame(rows))
    assert d["verdict"] == "DIRECTIONAL_CANDIDATES_BELOW_COST"
    assert d["worth_looking_at"][0]["feature"] == "obi_20"
