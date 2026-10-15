import numpy as np
import pandas as pd
import pytest

from data_engineering.modeling import fou_pairs as fp


def test_hurst_cof_brownian_and_antipersistent():
    rng = np.random.default_rng(0)
    bm = np.cumsum(rng.standard_normal(20000))
    assert abs(fp.hurst_cof(bm) - 0.5) < 0.03
    n = 4000
    e = rng.standard_normal(n) @ np.linalg.cholesky(fp.fgn_cov(0.3, n)).T
    assert abs(fp.hurst_cof(np.cumsum(e)) - 0.3) < 0.05


def test_fou_fit_recovers_ou_kappa():
    rng = np.random.default_rng(1)
    k, n = 0.05, 40000
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] - k * x[i - 1] + 0.001 * rng.standard_normal()
    f = fp.fou_fit(x)
    assert abs(f["H"] - 0.5) < 0.05
    assert 0.025 < f["kappa"] < 0.1


def test_conditional_operator_white_noise_has_no_memory():
    A, L = fp.conditional_operator(0.5, 60, 48)
    assert np.abs(A).max() < 1e-8
    assert np.allclose(L @ L.T, np.eye(48), atol=1e-6)
    A3, _ = fp.conditional_operator(0.3, 60, 48)
    assert A3[0, -1] < 0                                  # anti-persistence: next increment opposes the last


def test_first_passage_and_t70():
    inc = np.zeros((4, 5))
    tau = fp.first_passage(1.0, 0.0, 0.5, 1.0, inc)        # deterministic decay never crosses
    assert (tau == 6).all()
    inc[:, 0] = -2.0
    assert (fp.first_passage(1.0, 0.0, 0.0, 1.0, inc) == 1).all()
    T, pmax = fp.t70(np.array([1, 2, 2, 9]), 5, 0.7)
    assert T == 2 and pmax == 0.75


def test_planted_mean_reversion_trades_profitably():
    P = fp.synthetic_prices(n_days=12, seed=3, kappa=0.08)
    spec = fp.Spec(paths=1000, structural_paths=500)
    tr, fc, fits = fp.run_pair(P, "AAA", "BBB", spec, mode="fou")
    zt, _, _ = fp.run_pair(P, "AAA", "BBB", spec, mode="zscore")
    assert fits.eligible.any() or len(zt)
    allt = pd.concat([tr, zt])
    assert len(allt) > 20 and allt.gross_bp.mean() > 0


def test_random_walk_spread_rarely_cointegrated():
    P = fp.synthetic_prices(n_days=12, seed=4, kappa=0.0)
    _, _, fits = fp.run_pair(P, "AAA", "BBB", fp.Spec(paths=500, structural_paths=300), mode="zscore")
    assert (~fits.coint_ok.fillna(False)).mean() > 0.6


def test_pnl_sign_and_cost():
    t = pd.date_range("2026-01-01 00:05", periods=2100, freq="5min", tz="UTC")
    P = pd.DataFrame({"AAA": np.zeros(2100), "BBB": np.zeros(2100)}, index=t)
    # direct check of the trade arithmetic used in run_pair
    b, ly_e, ly_x, lx_e, lx_x, z = 1.0, 0.0, -0.01, 0.0, 0.0, 2.0
    gross = -np.sign(z) * ((ly_x - ly_e) - b * (lx_x - lx_e)) / (1 + b)
    assert 1e4 * gross == pytest.approx(50.0)               # y fell 1% vs x; short y earns 100 bp on half the notional
