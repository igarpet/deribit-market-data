"""Synthetic tests for Breakout Trend 01 (breakout_trend).

Run from repo root: python -m unittest discover -s tests -p "test_breakout_trend.py" -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_engineering.modeling import breakout_trend as bt

T0 = pd.Timestamp("2026-01-05 00:05", tz="UTC")


def frame(close, S=None, U=110.0, L=90.0, A=2.0, T14=None):
    n = len(close)
    close = np.asarray(close, float)
    return pd.DataFrame({
        "state_time": pd.date_range(T0, periods=n, freq="5min"),
        "close": close,
        "S": close if S is None else np.asarray(S, float),
        "U": np.full(n, U), "L": np.full(n, L), "A": np.full(n, A),
        "T14": np.full(n, 100.0) if T14 is None else np.asarray(T14, float),
        "valid": np.ones(n, bool)})


class RuleTests(unittest.TestCase):
    def test_selftest(self):
        s = bt.selftest()
        for k in ("no_lookahead", "causal_trades", "equity_matches_trades", "flip_consistent"):
            self.assertTrue(s[k], k)
        self.assertAlmostEqual(s["calib_k"], 1.2, places=2)

    def test_long_entry_then_stop_and_reverse(self):
        # enter long at 111 (S>U=110), stop at 111 - 1.5*2 = 108 -> reverse short at 108
        fr = frame([100, 111, 110, 108, 107])
        t = bt.run_strategy(fr, bt.Params())
        self.assertEqual(t.loc[0, "dir"], 1)
        self.assertEqual(t.loc[0, "entry_px"], 111.0)
        self.assertEqual(t.loc[0, "exit_reason"], "stop_reverse")
        self.assertEqual(t.loc[0, "exit_px"], 108.0)
        self.assertEqual(t.loc[1, "dir"], -1)
        self.assertEqual(t.loc[1, "entry_px"], 108.0)
        self.assertEqual(t.loc[1, "i_entry"], t.loc[0, "i_exit"])
        self.assertEqual(t.loc[1, "entry_kind"], "reverse")

    def test_trail_needs_arming(self):
        # T14 = 112 above price from the start: no exit until +1A (113) arms the trail
        T14 = [112, 112, 112, 112, 114, 114]
        fr = frame([100, 111, 111.5, 113, 113.5, 113.9], T14=T14)
        t = bt.run_strategy(fr, bt.Params())
        self.assertEqual(t.loc[0, "exit_reason"], "trail")
        self.assertEqual(t.loc[0, "i_exit"], 4)          # armed at bar 3 (113 >= 111+2), exits at 113.5 < 114

    def test_reentry_cross_vs_state(self):
        close = [100, 111, 113, 111.5, 111.5, 111.5, 109, 111]
        T14 = [100, 100, 100, 112, 112, 112, 112, 112]
        fr = frame(close, T14=T14)
        t_cross = bt.run_strategy(fr, bt.Params(reentry="cross"))
        t_state = bt.run_strategy(fr, bt.Params(reentry="state"))
        self.assertEqual(t_cross.loc[0, "exit_reason"], "trail")
        # cross: needs S <= U (bar 6) before the next long at bar 7
        self.assertEqual(t_cross.loc[1, "i_entry"], 7)
        # state: re-enters at the first flat bar with S > U
        self.assertEqual(t_state.loc[1, "i_entry"], 4)

    def test_invalid_bars_are_skipped(self):
        fr = frame([100, 111, 100, 100])
        fr.loc[1, "valid"] = False
        self.assertEqual(len(bt.run_strategy(fr, bt.Params())), 0)

    def test_costs(self):
        fr = frame([100, 111, 113, 111.5, 111.5], T14=[100, 100, 100, 112, 112])
        t = bt.perp_returns(bt.run_strategy(fr, bt.Params()))
        c = bt.PERP_TAKER_FEE + bt.PERP_SLIPPAGE
        self.assertAlmostEqual(t.loc[0, "perp_net"], 111.5 / 111 - 1 - 2 * c)
        eq = bt.perp_equity(fr, t)
        self.assertAlmostEqual(eq[1], -c)                  # entry bar marks the entry cost
        self.assertAlmostEqual(eq[-1], t["perp_net"].sum())


class DataTests(unittest.TestCase):
    def test_hourly_bars_use_completed_hours_only(self):
        t = pd.date_range("2026-01-01 00:05", periods=12 * 3 + 6, freq="5min", tz="UTC")
        df = pd.DataFrame({"state_time": t, "open": 1.0, "high": np.arange(len(t)) + 2.0,
                           "low": np.arange(len(t)) + 0.5, "close": np.arange(len(t)) + 1.0})
        h = bt.hourly_bars(bt.clean_5m(df), bt.Params(anchor_h=1, range_h=1, fast_h=1, trail_h=1))
        self.assertEqual(list(h.index), list(pd.date_range("2026-01-01 01:00", periods=3, freq="h", tz="UTC")))
        self.assertEqual(h.loc["2026-01-01 01:00", "close"], 12.0)       # bar ending 01:00 closes the first hour
        self.assertEqual(h.loc["2026-01-01 01:00", "high"], 13.0)

    def test_decision_frame_maps_last_completed_hour(self):
        t = pd.date_range("2026-01-01 00:05", periods=12 * 3, freq="5min", tz="UTC")
        df = pd.DataFrame({"state_time": t, "open": 1.0, "high": 2.0, "low": 0.5, "close": np.arange(len(t)) + 1.0})
        fr = bt.decision_frame(df, bt.Params(anchor_h=1, range_h=1, fast_h=1, trail_h=1))
        x = fr.set_index("state_time")
        self.assertEqual(x.loc["2026-01-01 01:00", "hour_end"], pd.Timestamp("2026-01-01 01:00", tz="UTC"))
        self.assertEqual(x.loc["2026-01-01 01:55", "hour_end"], pd.Timestamp("2026-01-01 01:00", tz="UTC"))
        self.assertTrue(pd.isna(x.loc["2026-01-01 00:55", "hour_end"]))

    def test_next_expiry(self):
        f = lambda s: bt.next_expiry(pd.Timestamp(s, tz="UTC"))
        self.assertEqual(f("2026-01-01 07:00"), pd.Timestamp("2026-01-02 08:00", tz="UTC"))
        self.assertEqual(f("2026-01-01 08:00"), pd.Timestamp("2026-01-02 08:00", tz="UTC"))
        self.assertEqual(f("2026-01-01 08:05"), pd.Timestamp("2026-01-03 08:00", tz="UTC"))


class OptionTests(unittest.TestCase):
    def test_roll_at_expiry(self):
        df = bt.synthetic_5m(10, seed=1)
        fr = bt.decision_frame(df)
        times = fr["state_time"]
        i0 = int(times.searchsorted(pd.Timestamp("2025-01-06 07:00", tz="UTC")))
        i1 = int(times.searchsorted(pd.Timestamp("2025-01-08 09:00", tz="UTC")))
        tr = pd.DataFrame({"i_entry": [i0], "i_exit": [i1], "dir": [1], "entry_time": [times[i0]],
                           "exit_time": [times[i1]]})
        pcs = bt.option_pieces(tr, fr)
        self.assertEqual(list(pcs["end_kind"]), ["expiry", "expiry", "close"])
        self.assertEqual(pcs.loc[0, "expiry"], pd.Timestamp("2025-01-07 08:00", tz="UTC"))
        self.assertEqual(times[pcs.loc[1, "i_start"]], pd.Timestamp("2025-01-07 08:00", tz="UTC"))
        self.assertEqual(pcs.loc[1, "expiry"], pd.Timestamp("2025-01-08 08:00", tz="UTC"))
        self.assertEqual(pcs.loc[1, "end_kind"], "expiry")
        self.assertEqual(len(pcs), 3)

    def test_model_matches_real_on_synthetic_book(self):
        df = bt.synthetic_5m(90, seed=7)
        fr = bt.decision_frame(df)
        tr = bt.run_strategy(fr)
        vol = bt.ewma_vol(fr)
        start = str(fr["state_time"].iloc[-288 * 20].date())
        qs = bt.synthetic_quotes(fr, k=1.1, rel_half=0.04, start=start)
        m, _ = bt.calibrate_option_model(qs, fr, vol)
        self.assertAlmostEqual(m.k, 1.1, places=3)
        pcs, eq = bt.model_option_leg(fr, tr, m, vol)
        self.assertAlmostEqual(eq[-1], pcs["opt_ret"].sum(), places=9)
        real = bt.real_option_leg(fr, tr, qs)
        c = real.merge(pcs[["trade_id", "piece", "opt_ret"]], on=["trade_id", "piece"]).dropna(subset=["opt_ret_real"])
        c = c[c["exit_flag"] != "no_bid_intrinsic"]
        self.assertGreater(len(c), 5)
        self.assertGreater(np.corrcoef(c["opt_ret"], c["opt_ret_real"])[0, 1], 0.9)

    def test_long_option_loses_premium_when_price_flat(self):
        n = 288 * 8
        t = pd.date_range("2025-01-01 00:05", periods=n, freq="5min", tz="UTC")
        rng = np.random.default_rng(0)
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
        df = pd.DataFrame({"state_time": t, "open": c, "high": c * 1.001, "low": c * 0.999, "close": c})
        fr = bt.decision_frame(df)
        i0 = 288 * 5
        tr = pd.DataFrame({"i_entry": [i0], "i_exit": [i0 + 12], "dir": [1], "entry_time": [t[i0]],
                           "exit_time": [t[i0 + 12]]})
        fr.loc[i0:i0 + 12, "close"] = fr.loc[i0, "close"]     # price unchanged for an hour
        pcs, _ = bt.model_option_leg(fr, tr, bt.OptionModel(k=1.0, h=(0.0, 0.0, 0.0)))
        self.assertLess(pcs.loc[0, "opt_ret"], 0.0)            # theta + fees

    def test_load_option_quotes_inverse_conversion(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = root / "daily" / "option_grid" / "expiry=2026-10-08"
            p.mkdir(parents=True)
            exp_ms = int(pd.Timestamp("2026-10-08 08:00", tz="UTC").value // 10 ** 6)
            rows = pd.DataFrame({
                "scheduled_snapshot_time": [pd.Timestamp("2026-10-07 12:00", tz="UTC")] * 3,
                "instrument_name": ["BTC-8OCT26-100000-C", "BTC_USDC-8OCT26-100000-C", "BTC-8OCT26-150000-C"],
                "underlying": ["BTC"] * 3, "product_group": ["btc_inv", "btc_usdc", "btc_inv"],
                "instrument_type": ["reversed", "linear", "reversed"],
                "expiration_timestamp": [exp_ms] * 3, "strike": [100000.0, 100000.0, 150000.0],
                "option_type": ["call"] * 3, "underlying_price": [100000.0] * 3,
                "best_bid_price": [0.01, 900.0, 0.0], "best_ask_price": [0.011, 1100.0, 0.0001],
                "bid_iv": [40.0, 40.0, 0.0], "ask_iv": [44.0, 44.0, 90.0], "delta": [0.5, 0.5, 0.0]})
            rows.to_parquet(p / "option_grid.parquet")
            q, info = bt.load_option_quotes(root, "BTC")
            self.assertEqual(info["family"], "btc_inv")
            self.assertEqual(len(q), 1)                               # far OTM strike filtered out
            self.assertAlmostEqual(q.loc[0, "bid_usd"], 1000.0)
            self.assertAlmostEqual(q.loc[0, "ask_usd"], 1100.0)
            self.assertEqual(q.loc[0, "expiry_time"], pd.Timestamp("2026-10-08 08:00", tz="UTC"))


class NullTests(unittest.TestCase):
    def test_null_rate_and_share(self):
        df = bt.synthetic_5m(40, seed=2)
        fr = bt.decision_frame(df)
        t = bt.run_strategy(fr)
        rate = bt.flat_entry_rate(fr, t)
        self.assertGreater(rate, 0.0)
        w = {"all": (None, None)}
        null = bt.null_distribution(fr, bt.Params(), rate, w, n_runs=5, seed=1)
        self.assertEqual(len(null), 5)
        self.assertTrue(np.isfinite(bt.null_share(null, "all", 0.0)))
        self.assertEqual(bt.verdict(1.0, 0.01), "EDGE_CANDIDATE")
        self.assertEqual(bt.verdict(1.0, 0.2), "NO_EDGE")
        self.assertEqual(bt.verdict(-1.0, 0.01), "NO_EDGE")


class TestUmSource(unittest.TestCase):
    def test_ohlc5_from_um_1m(self):
        t = pd.date_range("2026-01-05 00:00", periods=15, freq="1min", tz="UTC")
        c = np.arange(15, dtype=float) + 100
        df1 = pd.DataFrame({"open_time": t, "open": c - 0.5, "high": c + 1, "low": c - 1, "close": c, "volume": 1.0})
        df1 = df1.drop(index=[12])                       # third 5m bar incomplete -> dropped
        o = bt.ohlc5_from_um_1m(df1, "BTCUSDT")
        self.assertEqual(len(o), 2)
        self.assertEqual(o["state_time"].iloc[0], pd.Timestamp("2026-01-05 00:05", tz="UTC"))   # bar END
        self.assertEqual(o["open"].iloc[0], 99.5)
        self.assertEqual(o["close"].iloc[0], 104.0)
        self.assertEqual(o["high"].iloc[1], 110.0)
        self.assertEqual(o["low"].iloc[1], 104.0)
        self.assertEqual(o["volume"].iloc[1], 5.0)


if __name__ == "__main__":
    unittest.main()
