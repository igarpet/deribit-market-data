import numpy as np
import pandas as pd
import pytest

from data_engineering.modeling import markov_har_variance as v18
from data_engineering.modeling import markov_lift_distributions as v14
from data_engineering.modeling import model_vs_implied as mv


def _boot(skill, seed, ycol="y_rv"):
    d, sm, rv = mv.synthetic_comparison(n_sessions=25, seed=seed, model_skill=skill)
    c = mv.comparison_frame(d, sm, rv)
    return c, mv.session_bootstrap(c, ["x"], ycol, mv.Params(n_boot=600))


def test_informative_model_detected():
    c, b = _boot(0.8, 1)
    assert b["x_lo"] > 0


def test_uninformative_model_not_detected():
    _, b = _boot(0.0, 2)
    assert not (b["x_lo"] > 0)


def test_settlement_move_outcome_is_much_noisier():
    _, b_rv = _boot(0.8, 1)
    _, b_mv = _boot(0.8, 1, "y")
    assert (b_mv["x_hi"] - b_mv["x_lo"]) > 2 * (b_rv["x_hi"] - b_rv["x_lo"])


def test_realised_variance_window():
    t = pd.date_range("2026-10-01 00:05", periods=10, freq="5min", tz="UTC")
    panel = pd.DataFrame({"state_time": t, "underlying": "BTC", "return_5m": np.arange(1, 11) / 100})
    o = pd.DataFrame({"underlying": ["BTC", "BTC"], "state_time": [t[1], t[8]], "h": [10, 10]})
    rv = mv.realised_variance(panel, o)
    assert rv.iloc[0] == pytest.approx(0.03 ** 2 + 0.04 ** 2)       # bars ending t[2], t[3]
    assert np.isnan(rv.iloc[1])                                      # runs past the data


def test_comparison_frame_arithmetic():
    d = pd.DataFrame({"session": ["s"], "underlying": ["BTC"], "T": [0.01], "F": [100.0], "S_T": [100.0 * np.exp(0.02)],
                      "iv_straddle_mid": [0.1]})
    c = mv.comparison_frame(d, pd.Series([0.02]))
    assert c.sigma_imp.iloc[0] == pytest.approx(0.01)
    assert c.x.iloc[0] == pytest.approx(np.log(2.0))
    assert c.y.iloc[0] == pytest.approx(np.log(0.02 / (0.01 * np.sqrt(2 / np.pi))))


def test_entries_match_horizon_and_hour():
    t = pd.to_datetime(["2026-10-01 04:00", "2026-10-01 04:00", "2026-10-01 05:00", "2026-10-01 12:00"], utc=True)
    s = pd.DataFrame({"state_time": t, "minutes_to_expiry": [240, 240, 180, 1200], "session": "2026-10-01",
                      "underlying": ["BTC", "BTC", "BTC", "ETH"]})
    e = mv.entries(s)
    assert sorted(zip(e.underlying, e.h)) == [("BTC", 240), ("ETH", 1200)]


def test_har_pipeline_predicts_without_lookahead():
    rng = np.random.default_rng(3)
    n = 120 * 288
    t = pd.date_range("2025-01-01 00:05", periods=n, freq="5min", tz="UTC")
    lv = np.zeros(n)
    for i in range(1, n):
        lv[i] = .995 * lv[i - 1] + .1 * rng.standard_normal()
    r = 0.001 * np.exp(lv) * rng.standard_normal(n)
    raw = pd.DataFrame({"state_time": t, "underlying": "BTC", "return_5m": r})
    frame = mv.build_frame(raw, (120, 240), v14, v18)
    fits = mv.fit_har_models(frame, (120, 240), v18)
    panel = mv.forecast_panel(raw, v14, v18)
    o = pd.DataFrame({"underlying": ["BTC", "BTC"], "state_time": [t[-500], t[-200]], "h": [240, 120]})
    sig = mv.model_sigma(panel, o, fits, v18)
    assert np.isfinite(sig).all() and (sig > 0).all()
    # changing returns AFTER the origin must not change the forecast
    raw2 = raw.copy(); raw2.loc[raw2.state_time > t[-500], "return_5m"] *= 5
    sig2 = mv.model_sigma(mv.forecast_panel(raw2, v14, v18), o.iloc[:1], fits, v18)
    assert sig2.iloc[0] == pytest.approx(sig.iloc[0])


def test_decision():
    assert mv.decision({"x": .5, "x_lo": .1, "x_hi": .9}, {"lo": -0.2})["verdict"] == "MODEL_ADDS_INFORMATION"
    assert mv.decision({"x": .5, "x_lo": .1, "x_hi": .9}, {"lo": 0.05})["verdict"] == "MODEL_EDGE_CANDIDATE"
    assert mv.decision({"x": 0, "x_lo": -.1, "x_hi": .1}, {})["verdict"] == "NO_EVIDENCE"
