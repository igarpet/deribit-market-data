"""Synthetic validation for Signed Jumps 03 (signed_jumps_flow.py)."""
import numpy as np
import pandas as pd
import pytest

from data_engineering.modeling import signed_jumps as sj
from data_engineering.modeling import signed_jumps_flow as sf

P = sj.Params(n_boot=300, min_test_jumps_per_sign=20, mc_paths=100)
FP = sf.FlowParams(n_boot=300)


def _run(seed, n_days=50, **kw):
    S, F = sj.simulate(n_days=n_days, n_assets=2, eta=((0.2, 0.05), (0.05, 0.3)), half_life=1, seed=seed, df=8,
                       base_rate=8e-4, flow=True, **kw)
    assets, pan, _ = sj.prepare(S, P)
    flows = {a.name: F[a.name].reindex(a.time) for a in assets}
    pf = sf.attach_flow(assets, pan, flows, FP)
    fits, E = sf.fit_models(assets, pf, P, FP)
    cmp, _ = sf.compare(fits, pf, E, P)
    vp, info = sf.vhat_panel(assets, pf, P)
    f3, _ = sf.compare_f3(fits, E, vp, assets, P, pf)
    return dict(assets=assets, pan=pan, pf=pf, flows=flows, fits=fits, E=E,
                st=dict(zip(cmp.test, cmp.status)), f3=f3, info=info)


def test_flow_covariates_are_causal():
    S, F = sj.simulate(n_days=4, n_assets=1, seed=1, flow=True, flow_vol=0.5)
    fl = F["S0"]
    k = 3000
    fl2 = fl.copy()
    fl2.iloc[k + 1:] *= 3.0
    a, _ = sf.flow_covariates(fl, FP)
    b, _ = sf.flow_covariates(fl2, FP)
    assert np.allclose(np.nan_to_num(a[: k + 2], nan=-9), np.nan_to_num(b[: k + 2], nan=-9))
    assert not np.allclose(np.nan_to_num(a[k + 2:], nan=-9), np.nan_to_num(b[k + 2:], nan=-9))


def test_subset_panel_keeps_excitation_alignment():
    S, F = sj.simulate(n_days=10, n_assets=2, seed=2, flow=True, eta=((0.2, 0), (0, 0.2)), half_life=1)
    assets, pan, _ = sj.prepare(S, P)
    pf = sf.attach_flow(assets, pan, {a.name: F[a.name].reindex(a.time) for a in assets}, FP)
    assert len(pf.r) <= len(pan.r) and pf.extra.shape == (len(pf.r), 5)
    E_full = sj.panel_excitation(assets, pan, 1.0)
    E_sub = sj.panel_excitation(assets, pf, 1.0)
    t_full = pd.Series(E_full["up"], index=list(zip(pan.asset_idx, pd.DatetimeIndex(pan.time))))
    t_sub = pd.Series(E_sub["up"], index=list(zip(pf.asset_idx, pd.DatetimeIndex(pf.time))))
    assert np.allclose(t_full.loc[t_sub.index].to_numpy(), t_sub.to_numpy())
    assert sj.design_names(pf, "f", P)[-5:] == list(pf.extra_names)


def test_uninformative_flow_does_not_pass():
    r = _run(9, heat_link=1.0, flow_heat=0.0, flow_vol=0.0)
    assert r["st"]["F1_f_vs_e"] != "PASS" and r["f3"]["status"] != "PASS"


@pytest.mark.parametrize("seed", [9, 10])
def test_volatility_tracking_flow_fails_F3(seed):
    """Flow that only tracks volatility may pass F1 (better volatility nowcast) but must fail F3."""
    r = _run(seed, heat_link=1.0, flow_heat=0.0, flow_vol=0.5)
    assert r["f3"]["status"] != "PASS"
    assert r["info"]["fwd_var_R2_with_flow_dev"] > r["info"]["fwd_var_R2_base_dev"] + 0.02


def test_planted_heat_passes_F1_and_F3():
    r = _run(12, n_days=90, heat_link=1.2, flow_heat=1.2, flow_vol=0.5)
    assert r["st"]["F1_f_vs_e"] == "PASS" and r["f3"]["status"] == "PASS"
    prof = sf.event_profile(r["assets"], r["pf"], r["flows"], FP)
    pre = prof[(prof.series == "trades") & (prof.k.between(-10, -1))].excess.mean()
    assert pre > 0.2      # trade activity is elevated in the minutes before jumps
