"""Checks for VD20 (daily_vol_forecast.py)."""
import numpy as np
import pandas as pd
import pytest

from data_engineering.modeling import daily_vol_forecast as dv
from data_engineering.modeling import model_vs_implied as mv

P = dv.Params(n_boot=300)


def _run(raw, split_frac=0.6):
    D = dv.daily_origins(raw, 8, 1440, P)
    cut = D.date.quantile(split_frac)
    tr, ev = D[D.date < cut], D[D.date >= cut + pd.Timedelta(days=1)]
    fits = dv.fit_daily(tr)
    ev = pd.concat([ev, dv.predict_daily(ev, fits)], axis=1)
    tr = pd.concat([tr, dv.predict_daily(tr, fits)], axis=1)
    return D, tr, ev


def test_target_and_features_are_causal():
    raw = dv.synthetic_returns(n_days=60, seed=1)
    D = dv.daily_origins(raw, 8, 1440, P)
    a = raw[raw.underlying == "A"].set_index("state_time").return_5m
    row = D[(D.underlying == "A") & np.isfinite(D.rv_next)].iloc[10]
    t = row.state_time
    fut = a[(a.index > t) & (a.index <= t + pd.Timedelta(minutes=1440))]
    pst = a[(a.index > t - pd.Timedelta(minutes=1440)) & (a.index <= t)]
    assert len(fut) == 288 and abs(row.rv_next - (fut ** 2).sum()) < 1e-15 and abs(row.R_next - fut.sum()) < 1e-12
    assert abs(row.rv1 - (pst ** 2).sum()) < 1e-15
    # changing every future return leaves features unchanged
    raw2 = raw.copy()
    m = (raw2.underlying == "A") & (raw2.state_time > t)
    raw2.loc[m, "return_5m"] *= 5
    D2 = dv.daily_origins(raw2, 8, 1440, P)
    r2 = D2[(D2.underlying == "A") & (D2.state_time == t)].iloc[0]
    for c in ("rv1", "rv5", "rv22", "rv30", "rv240", "r1"):
        assert r2[c] == pytest.approx(row[c], rel=1e-12)


def test_persistent_vol_is_forecastable():
    raw = dv.synthetic_returns(n_days=700, seed=2)
    D, tr, ev = _run(raw)
    sc, E = dv.score_table(ev, ["NAIVE", "HAR_D", "HAR_DX"], P)
    pr = dv.pair_diffs(E, [("HAR_D", "NAIVE")], P)
    assert pr.iloc[0].hi < 0                         # HAR_D beats naive persistence
    cl = dv.classification(E, ["HAR_D"], P)
    assert cl.iloc[0].auc_lo > 0.6
    assert 0.8 < sc.set_index("model").loc["HAR_D", "bias_mean_ratio"] < 1.25


def test_constant_vol_has_no_classification_skill():
    raw = dv.synthetic_returns(n_days=700, seed=3, persistent=False)
    D, tr, ev = _run(raw)
    sc, E = dv.score_table(ev, ["HAR_D"], P)
    cl = dv.classification(E, ["HAR_D"], P)
    assert cl.iloc[0].auc_lo < 0.5 < cl.iloc[0].auc_hi and abs(cl.iloc[0].auc - 0.5) < 0.04


def test_gaussian_tails_are_calibrated():
    raw = dv.synthetic_returns(n_days=700, seed=4)
    D, tr, ev = _run(raw)
    sc, E = dv.score_table(ev, ["HAR_D"], P)
    T = dv.tail_table(tr, E, ["HAR_D"], P).set_index("k")
    assert abs(T.loc[1.0, "observed"] - T.loc[1.0, "gaussian"]) < 0.05
    assert abs(T.loc[2.0, "observed"] - T.loc[2.0, "gaussian"]) < 0.025


def test_implied_dominates_when_implied_is_the_truth():
    rng = np.random.default_rng(7)
    rows, F = [], []
    for s in pd.date_range("2026-01-02", periods=40, freq="D", tz="UTC"):
        for a in ("BTC", "ETH", "SOL"):
            v = 1e-4 * np.exp(rng.normal(0, .5))                    # true variance, known to implied
            rv = v * np.exp(rng.normal(0, .3))
            rows.append({"session": s.strftime("%Y-%m-%d"), "underlying": a, "entry_hour": 12,
                         "sigma_imp": np.sqrt(v), "rv_path": rv, "R_net": rng.normal(-.2, .5)})
            F.append({"date": s - pd.Timedelta(days=1), "underlying": a, "F_M": v * np.exp(rng.normal(0, .6))})
    M, info, econ, ia = dv.implied_check(pd.DataFrame(rows), pd.DataFrame(F), ["M"], P, mv)
    r = info.iloc[0]
    assert r["enc_log_var_imp_lo"] > 0.5 and r["enc_log_F_M_lo"] < 0 < r["enc_log_F_M_hi"]
    assert dv.decide(pd.DataFrame([{"model": "HAR_D", "vs": "NAIVE", "hi": -1, "lo": -2, "better": "model"}]),
                     info)["verdict"] == "FORECASTS_VOL_BUT_NOT_BEYOND_IMPLIED"
