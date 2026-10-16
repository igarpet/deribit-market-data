"""Synthetic validation for Signed Jumps 01 (signed_jumps.py). Must pass before any real-data run."""
import numpy as np
import pandas as pd
import pytest

from data_engineering.modeling import signed_jumps as sj

P = sj.Params(n_boot=300, half_lives=(3, 10), min_test_jumps_per_sign=50, mc_paths=200)


def _run(models=sj.MODELS, **sim):
    S = sj.simulate(**sim)
    assets, pan, _ = sj.prepare(S, P)
    fits, E = sj.fit_ladder(assets, pan, P, models=models)
    cmp, nj = sj.compare(fits, pan, E, P)
    return assets, pan, fits, E, cmp.set_index("test"), nj


def test_excitation_kernel_matches_explicit_sum():
    rng = np.random.default_rng(0)
    J = (rng.random(400) < 0.05).astype(float)
    h = 7.0
    q = np.exp(-np.log(2) / h)
    E = sj.excitation(J, h)
    for t in (0, 1, 50, 399):
        expl = sum((1 - q) * q ** (k - 1) * J[t - k] for k in range(1, t + 1))
        assert abs(E[t] - expl) < 1e-12


def test_gradient_matches_finite_difference():
    rng = np.random.default_rng(1)
    n = 3000
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    Ex = np.abs(rng.standard_normal((n, 2))) * 0.1
    J = (rng.random(n) < 0.02).astype(float)
    th = np.array([-4.0, 0.3, 0.2, 0.1])
    f0, g = sj._negll(th, X, Ex, J, 2)
    for k in range(4):
        e = np.zeros(4)
        e[k] = 1e-6
        num = (sj._negll(th + e, X, Ex, J, 2)[0] - sj._negll(th - e, X, Ex, J, 2)[0]) / 2e-6
        assert abs(num - g[k]) < 1e-6 * max(1, abs(num))


def test_covariates_and_jumps_are_causal():
    S = sj.simulate(n_days=12, n_assets=1, seed=3)
    k = next(iter(S))
    s0 = S[k]
    dev_end = s0.index[0].normalize() + pd.Timedelta(days=8)
    cut = len(s0) - 2000                       # inside TEST, so DEV-fitted coefficients are unchanged
    s1 = s0.copy()
    s1.iloc[cut + 1:] += np.random.default_rng(9).standard_normal(len(s1) - cut - 1).cumsum() * 1e-3
    out = []
    for s in (s0, s1):
        a = sj.make_asset(k, s, dev_end)
        sj.build_engine(a, P)
        out.append(a.cols)
    for key in ("Z", "sig2c", "fast", "slow", "TSK", "TV10", "TV60", "Jup", "Jdn"):
        x0, x1 = out[0][key][: cut + 1], out[1][key][: cut + 1]
        assert np.allclose(np.nan_to_num(x0, nan=-9), np.nan_to_num(x1, nan=-9)), key


def test_recovers_planted_signed_excitation():
    _, _, fits, _, cmp, _ = _run(models=("a", "b", "c", "d"), n_days=60, n_assets=2,
                                 eta=((0.35, 0.0), (0.0, 0.45)), half_life=10, seed=1, df=30, base_rate=1e-3)
    M = sj.eta_matrix(fits).set_index("child")
    assert M.loc["up", "from_up"] > 0.15 and M.loc["down", "from_down"] > 0.15
    assert M.loc["up", "from_down"] < 0.08 and M.loc["down", "from_up"] < 0.08
    for t in ("G1_c_vs_b", "G2a_d_vs_c"):
        assert cmp.loc[t, "delta_per_row"] > 0 and cmp.loc[t, "q025"] > 0, t


@pytest.mark.parametrize("seed", [100, 101])
def test_no_false_excitation_under_fast_volatility_bursts(seed):
    """VD 08 trap: common stochastic volatility with fast bursts, Student-t(4), NO excitation."""
    _, _, fits, _, cmp, _ = _run(models=("a", "b", "c", "d"), n_days=50, n_assets=2, seed=seed, df=4,
                                 vol_sd=0.15, vol_persist=0.99, common=0.8)
    assert cmp.loc["G1_c_vs_b", "status"] != "PASS"
    assert cmp.loc["G2a_d_vs_c", "status"] != "PASS"
    assert sj.branching(fits, "c") < 0.08


def test_detects_planted_pre_jump_skew():
    _, _, fits, _, cmp, _ = _run(n_days=50, n_assets=2, seed=7, df=30, base_rate=1e-3, skew_link=1.0)
    assert fits[("e", "up")].b[-1] > 0.4 and fits[("e", "down")].b[-1] < -0.4
    assert cmp.loc["G2b_e_vs_d", "q0125"] > 0


def test_no_skew_effect_without_planted_skew():
    _, _, _, _, cmp, _ = _run(n_days=50, n_assets=2, seed=8, df=30, base_rate=1e-3, skew_link=0.0)
    assert cmp.loc["G2b_e_vs_d", "status"] != "PASS"


def test_horizon_forecast_beats_poisson_under_excitation():
    assets, pan, fits, _, _, _ = _run(models=("a", "b", "c", "d"), n_days=60, n_assets=2,
                                      eta=((0.35, 0.0), (0.0, 0.45)), half_life=10, seed=1, df=30, base_rate=1e-3)
    g3, cal = sj.g3_table(fits, "d", assets, pan, P)
    prim = g3.set_index("horizon_min").loc[P.primary_horizon]
    assert prim["delta_per_row"] > 0 and prim["q025"] > 0
    assert set(g3.horizon_min) == set(P.horizons)


def test_conditional_mc_equals_exact_when_no_excitation():
    """With eta = 0 the conditional MC estimator must reproduce the exact Poisson probabilities of (b)."""
    from dataclasses import replace
    S = sj.simulate(n_days=20, n_assets=1, seed=4, df=30, base_rate=1e-3)
    assets, pan, _ = sj.prepare(S, P)
    fits, _ = sj.fit_ladder(assets, pan, P, models=("a", "b"))
    for s in sj.SIGNS:
        fb = fits[("b", s)]
        fits[("d", s)] = replace(fb, model="d", eta=np.zeros(2), half_life=10)
    pan_d = pan
    rows = np.flatnonzero(~pan.dev)[::30]
    pb, _ = sj.horizon_forecast(fits, "b", assets, pan, P, rows)
    # model "d" uses design(pan, "d") == design(pan, "b") (TSK only enters "e")
    pdd, _ = sj.horizon_forecast(fits, "d", assets, pan_d, P, rows)
    for h in P.horizons:
        for s in sj.SIGNS:
            assert np.allclose(pb[h][s], pdd[h][s], rtol=1e-6, atol=1e-12), (h, s)


def test_conditional_mc_matches_brute_force_simulation():
    """With excitation, the conditional estimator matches a brute-force hit-count simulation."""
    from dataclasses import replace
    S = sj.simulate(n_days=20, n_assets=1, seed=5, df=30, base_rate=1e-3)
    assets, pan, _ = sj.prepare(S, P)
    fits, _ = sj.fit_ladder(assets, pan, P, models=("a", "b"))
    fits[("d", "up")] = replace(fits[("b", "up")], model="d", eta=np.array([0.0, 0.6]), half_life=3)
    fits[("d", "down")] = replace(fits[("b", "down")], model="d", eta=np.array([0.5, 0.0]), half_life=3)
    rows = np.flatnonzero(~pan.dev)[::200]
    # plant a recent down jump at every origin so cross-excitation matters
    for i, a in enumerate(assets):
        idx = pan._idx[i]
        g = idx[rows]
        a.cols["Jdn"][g - 1] = 1.0
    Pm = replace(P, mc_paths=4000)
    pc, _ = sj.horizon_forecast(fits, "d", assets, pan, Pm, rows, seed=1)
    # brute force
    q = 2 ** (-1 / 3)
    E = sj.panel_excitation(assets, pan, 3)
    rng = np.random.default_rng(2)
    Bu = sj.horizon_forecast(fits, "b", assets, pan, P, rows)[0]  # not used; ensures b path works
    X = sj.design(pan, "d")
    base = {s: np.exp(X[rows] @ fits[("d", s)].b) for s in sj.SIGNS}   # clock frozen: compare at h=5 only
    M = 20000
    Eu = np.repeat(E["up"][rows][:, None], M, 1)
    Ed = np.repeat(E["down"][rows][:, None], M, 1)
    hit_u = np.zeros_like(Eu, bool)
    for j in range(5):
        lu = base["up"][:, None] + 0.6 * Ed
        ld = base["down"][:, None] + 0.5 * Eu
        ju = rng.random(Eu.shape) < -np.expm1(-lu)
        jd = rng.random(Eu.shape) < -np.expm1(-ld)
        hit_u |= ju
        Eu = q * Eu + (1 - q) * ju
        Ed = q * Ed + (1 - q) * jd
    bf = hit_u.mean(1)
    assert np.abs(pc[5]["up"] - bf).max() < 0.02                  # per origin: brute-force MC s.d. ~0.0034
    assert abs(pc[5]["up"].mean() - bf.mean()) < 0.002              # pooled: no bias
    assert pc[5]["up"].mean() > Bu[5]["up"].mean() * 1.5     # excitation from the planted down jump shows
