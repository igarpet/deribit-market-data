import numpy as np
import pandas as pd
import pytest

from data_engineering.modeling import basis_fou as bf
from data_engineering.modeling import fou_pairs as fp


def test_screen_basis_stationary_vs_random_walk():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(2016)) * 1e-3
    s = fp.Spec()
    ou = np.zeros(2016)
    for i in range(1, 2016):
        ou[i] = 0.9 * ou[i - 1] + 1e-4 * rng.standard_normal()
    assert bf.screen_basis(x + ou, x, s)["coint_ok"]
    assert not bf.screen_basis(x + np.cumsum(rng.standard_normal(2016)) * 1e-4, x, s)["coint_ok"]
    assert bf.screen_basis(x + ou, x, s)["b"] == 1.0


def test_quarterly_expiries_last_friday_0800():
    e = bf.quarterly_expiries("2025-01-01", "2025-12-31")
    assert pd.Timestamp("2025-03-28 08:00", tz="UTC") in e
    assert pd.Timestamp("2025-12-26 08:00", tz="UTC") in e
    assert all(x.weekday() == 4 for x in e)


def test_front_series_rolls_with_gap():
    t = pd.date_range("2025-03-01", "2025-04-30", freq="1D", tz="UTC")
    e1, e2 = pd.Timestamp("2025-03-28 08:00", tz="UTC"), pd.Timestamp("2025-06-27 08:00", tz="UTC")
    c = {e1: pd.Series(1.0, index=t[t <= e1]), e2: pd.Series(2.0, index=t)}
    f = bf.front_series(c, roll_days=14)
    assert f[pd.Timestamp("2025-03-10", tz="UTC")] == 1.0
    switch_day = f.index[(f == 2.0).argmax()]
    assert np.isnan(f[switch_day - pd.Timedelta(days=1)]) or f[switch_day - pd.Timedelta(days=1)] == 1.0
    assert f.isna().sum() == 1


def test_funding_series_parsing():
    df = pd.DataFrame({"calc_time": [1735689600000, 1735718400000], "funding_interval_hours": [8, 8],
                       "last_funding_rate": [0.0001, -0.0002]})
    s = bf.funding_series(df)
    assert s.index[0] == pd.Timestamp("2025-01-01 00:00", tz="UTC") and s.iloc[1] == -0.0002


def test_funding_and_costs_in_trades():
    P, f = bf.synthetic_basis(n_days=10, seed=2, kappa=0.08, sd_bp=25)
    spec = fp.Spec(paths=500, structural_paths=300)
    tr, _, _ = fp.run_pair(P, "PERP_AAA", "SPOT_AAA", spec, mode="zscore", screen_fn=bf.screen_basis,
                           funding_y=f, cost_y_bp=6.0, cost_x_bp=11.0)
    assert len(tr) > 10
    assert tr.cost_bp.iloc[0] == pytest.approx(2 * (6 + 11) / 2)
    held = tr[(tr.exit_bar - tr.entry_bar) > pd.Timedelta(hours=8)]
    # short perp receives positive funding, long perp pays it
    for _, r in tr.iterrows():
        n = ((f.index > r.entry_bar) & (f.index <= r.exit_bar)).sum()
        exp = (1 if r.side == "short_y" else -1) * n * 1e-4 / 2 * 1e4
        assert r.funding_bp == pytest.approx(exp)
    assert tr.gross_bp.mean() > 0


def _dl():
    import importlib.util
    from pathlib import Path
    sp = importlib.util.spec_from_file_location("dlb", Path(__file__).resolve().parents[1] / "tools" / "download_binance_basis_data.py")
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


def test_downloader_parses_spot_microseconds_and_header():
    dl = _dl()
    row = b"1735689600000000,1,2,0.5,1.5,10,1735689659999999,15,3,4,6,0\n"
    hdr = b"open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
    for raw in (row, hdr + row.replace(b"1735689600000000", b"1735689600000").replace(b"1735689659999999", b"1735689659999")):
        k = dl.parse_klines(raw)
        assert k.open_time.iloc[0] == pd.Timestamp("2025-01-01", tz="UTC") and k.close.iloc[0] == 1.5


def test_downloader_last_fridays_match_module():
    from datetime import date
    dl = _dl()
    d = dl.last_fridays(date(2025, 1, 1), date(2025, 12, 31))
    assert date(2025, 3, 28) in d and date(2025, 12, 26) in d
