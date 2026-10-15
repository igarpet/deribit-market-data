import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_engineering.modeling import candle_streak_passive_fill as pf

M = pf.MIN_MS


def _sim(t, p, side, k=0.0, hold=5):
    return pf.simulate(np.array([0]), np.array([side]), np.array(t, "int64"), np.array(p, float),
                       k_bp=k, fill_min=5, hold_min=hold).iloc[0]


def test_trade_at_limit_does_not_fill():
    r = _sim([-1000, 1000, 2000, 5 * M], [100, 100, 100, 101], +1)
    assert not r.filled
    assert r.gross_taker == pytest.approx(np.log(101 / 100))


def test_trade_through_fills_and_exit_is_first_trade_at_hold():
    r = _sim([-1000, 1000, 2000, 5 * M - 1, 5 * M, 5 * M + 9], [100, 99.9, 99.0, 98, 100.5, 120], +1)
    assert r.filled and r.limit == 100 and r.fill_ms == 1000
    assert r.exit_px == 100.5
    assert r.gross_passive == pytest.approx(np.log(100.5 / 100))


def test_offset_and_sell_side():
    r = _sim([-1, 1000, 2000, 5 * M], [100, 100.05, 100.2, 99.0], -1, k=10.0)   # sell limit 100.1
    assert r.limit == pytest.approx(100.1)
    assert r.filled and r.fill_ms == 2000
    assert r.gross_passive == pytest.approx(-(np.log(99.0) - np.log(100.1)))


def test_order_expires_after_fill_window():
    r = _sim([-1, 6 * M, 15 * M], [100, 90, 95], +1, hold=15)
    assert not r.filled and r.exit_px == 95


def test_price_running_away_never_fills_buy():
    t = np.arange(-1000, 6 * M, 1000)
    r = _sim(t, 100 + 0.01 * np.arange(len(t)), +1)
    assert not r.filled


def test_random_walk_passive_has_no_free_lunch():
    t, p = pf.synthetic_trades(n_min=3000, seed=1)
    o = t[0] + np.arange(10, 2990, 5) * M
    side = np.where(np.random.default_rng(2).random(len(o)) < .5, 1.0, -1.0)
    r = pf.simulate(o, side, t, p, k_bp=0.0, fill_min=5, hold_min=5)
    g = r.loc[r.filled, "gross_passive"]
    assert len(g) > 200
    assert g.mean() < 2 * g.std() / np.sqrt(len(g))      # no significant positive gross on a martingale


def test_day_table_and_costs():
    o = pd.DataFrame({"date": ["d1", "d1", "d2"], "side": [1, -1, 1], "filled": [True, False, True],
                      "gross_passive": [0.001, np.nan, -0.0005], "gross_taker": [0.0, 0.0, 0.0]})
    d = pf.day_table(o, cost_total_bp=8.0)
    assert d.trades.tolist() == [1, 1]
    assert d.net_bp.tolist() == pytest.approx([10 - 8, -5 - 8])
    assert pf.cost_bp("maker_in_taker_out", pf.Specification()) == 8.0


def test_downloader_parses_with_and_without_header():
    spec = importlib.util.spec_from_file_location("dl", Path(__file__).resolve().parents[1] / "tools" / "download_binance_um_aggtrades.py")
    dl = importlib.util.module_from_spec(spec); spec.loader.exec_module(dl)
    body = b"1,100.5,0.2,10,11,1751328000123,true\n2,100.6,0.1,12,12,1751328000456,false\n"
    hdr = b"agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
    for raw in (body, hdr + body):
        x = dl.parse_csv(raw)
        assert x.time_ms.tolist() == [1751328000123, 1751328000456]
        assert x.price.tolist() == [100.5, 100.6] and x.is_buyer_maker.tolist() == [True, False]


def test_decision_requires_same_offset_both_assets():
    rows = []
    for a in ("BTC", "ETH"):
        for k in (0.0, 1.0, 2.0, 5.0):
            lo = 1.0 if (a, k) in {("BTC", 1.0), ("ETH", 2.0)} else -1.0
            rows.append({"arm": "signal", "hold_min": 5, "cost": "maker_in_taker_out", "kind": "passive",
                         "asset": a, "k_bp": k, "net_bp_per_day": 0.0, "net_bp_per_day_lo_bonf": lo})
    s = pd.DataFrame(rows)
    assert pf.decision(s)["verdict"] == "NO_PASSIVE_EDGE"
    s.loc[(s.asset == "ETH") & (s.k_bp == 1.0), "net_bp_per_day_lo_bonf"] = 0.5
    assert pf.decision(s)["passing_offsets_bp"] == [1.0]


def test_to_ms_resolution_independent():
    ts = pd.Timestamp("2026-07-01 00:05", tz="UTC")
    for unit in ("ns", "us", "ms"):
        s = pd.Series([ts]).astype(f"datetime64[{unit}, UTC]")
        assert pf.to_ms(s)[0] == 1782864300000
