import numpy as np
import pandas as pd
import pytest

from data_engineering.modeling import candle_streak_flow_attribution as ab
from data_engineering.modeling import candle_streak_perp_target as pt
from data_engineering.modeling import candle_streak_trade_flow as tf


def _kl(closes, start="2026-01-01 00:00"):
    t = pd.date_range(start, periods=len(closes), freq="1min", tz="UTC")
    return pd.DataFrame({"open_time": t, "close": closes})


def test_bucket_end_label_and_return():
    c = np.r_[np.full(5, 100.0), np.full(4, 101.0), 102.0]
    r = pt.perp_returns_5m(_kl(c))
    assert list(r.state_time) == list(pd.to_datetime(["2026-01-01 00:05", "2026-01-01 00:10"], utc=True))
    assert np.isnan(r.return_5m.iloc[0])
    assert r.return_5m.iloc[1] == pytest.approx(np.log(102 / 100))


def test_incomplete_bucket_breaks_returns():
    k = _kl(np.linspace(100, 101, 20))
    k = k.drop(index=7)                                   # bucket 00:05-00:09 incomplete
    r = pt.perp_returns_5m(k).set_index("state_time").return_5m
    assert np.isnan(r[pd.Timestamp("2026-01-01 00:10", tz="UTC")])
    assert np.isnan(r[pd.Timestamp("2026-01-01 00:15", tz="UTC")])   # previous bucket bad
    assert np.isfinite(r[pd.Timestamp("2026-01-01 00:20", tz="UTC")])


def test_pnl_arithmetic():
    t = pd.date_range("2026-01-01 00:05", periods=4, freq="5min", tz="UTC")
    ev = pd.DataFrame({"date": t.date, "state_time": t, "next_return": [0.01, -0.02, 0.03, 0.01]})
    e = np.array([0.5, 0.5, -0.5, 0.0])
    d = pt.daily_pnl(ev, e, 0.4, 0.001)
    # positions +1,+1,-1,0: gross .01-.02-.03 = -.04; turnover 1+0+2+1 = 4
    assert d.gross.iloc[0] == pytest.approx(-0.04)
    assert d.turnover.iloc[0] == pytest.approx(4)
    assert d.net.iloc[0] == pytest.approx(-0.044)
    assert d.active_bars.iloc[0] == 3


def test_gap_forces_flat():
    t = pd.to_datetime(["2026-01-01 00:05", "2026-01-01 00:20"], utc=True)
    ev = pd.DataFrame({"date": t.date, "state_time": t, "next_return": [0.0, 0.0]})
    d = pt.daily_pnl(ev, np.array([1.0, 1.0]), 0.5, 1.0)
    assert d.turnover.iloc[0] == 4                     # open, close before gap, open, close at end


def _run(beta, seed, lag=0.0, target="perp"):
    kl, sp = pt.synthetic_pair(n_days=90, seed=seed, perp_beta=beta, spot_lag_beta=lag)
    p, f = pt.perp_returns_5m(kl), tf.one_minute_futures_flow(kl)
    x = pt.build_panel(p if target == "perp" else sp, f)
    x, _ = tf.split_panel(x, tf.full_days(x), .6, .2)
    fit = ab.fit_ablation(x[x.period.eq("train")])
    ev = x[x.period.eq("test") & x.scorable]
    loss = ab.origin_losses(ev, ab.predict_ablation(ev, fit))
    b = ab.day_bootstrap(loss, B=200, seed=1, min_days=10, min_events=200)
    r = b[(b.metric == "binary_logloss") & (b.model == "price_vol_signed") & (b.reference == "price_vol")].iloc[0]
    return r


def test_planted_perp_effect_detected():
    r = _run(0.35, 3)
    assert r.ci975 < 0


def test_null_not_detected():
    r = _run(0.0, 4)
    assert not (r.ci975 < 0 and r.delta_equal_day < -0.002)


def test_pure_lead_lag_shows_in_spot_not_perp():
    assert _run(0.0, 5, lag=0.6, target="spot").ci975 < 0
    r = _run(0.0, 5, lag=0.6, target="perp")
    assert not (r.ci975 < 0 and r.delta_equal_day < -0.002)


def test_decision_requires_both_gates():
    boot = pd.DataFrame([{"asset": a, "period": "test", "target": "perp", "metric": m, "model": "price_vol_signed",
                          "reference": "price_vol", "status": "OK", "delta_equal_day": -.001, "ci975": -.0001}
                         for a in ("BTC", "ETH") for m in ("threeclass_logloss", "binary_logloss")])
    econ = pd.DataFrame([{"asset": a, "period": "test", "target": "perp", "model": "price_vol_signed", "cost": "taker",
                          "rule": "top_decile", "net_bp_per_day": -5.0, "net_bp_per_day_lo": -9.0} for a in ("BTC", "ETH")])
    assert pt.decision(boot, econ)["verdict"] == "PERP_STATISTICAL_ONLY"
    econ["net_bp_per_day_lo"] = 1.0
    assert pt.decision(boot, econ)["verdict"] == "PERP_TRADEABLE_CANDIDATE"
